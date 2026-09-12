"""tshark/pyshark-based QUIC/HTTP-3 extraction (paper §6).

Decodes a pcap/pcapng capture into `mlcfq.model.Session` objects. Requires
`tshark` on PATH (Wireshark's QUIC dissector, which decodes QUIC natively in
recent versions per the paper) and the optional `pyshark` dependency.

If a `SSLKEYLOGFILE` is supplied (captured from a controlled generating
client), HTTP/3 frame-level detail (SETTINGS, QPACK) becomes visible;
without it, `x_A` extraction is best-effort and may be sparse, matching the
honest caveat in paper §6.

This module intentionally isolates the only external-process dependency in
the codebase so the rest of MLCF-Q can be tested without tshark installed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from typing import Any, Iterator

from mlcfq.features.behavioral import build_behavioral_features
from mlcfq.features.handshake import build_handshake_features
from mlcfq.features.http3 import build_http3_features
from mlcfq.features.quic_transport import build_transport_features
from mlcfq.model import Session


class PcapExtractionError(RuntimeError):
    """Raised when tshark/pyshark is unavailable or decoding fails."""


def _normalize_key(value: str) -> str:
    """Normalize a correlation-key string for use as a dict key.

    pyshark's object model formats some fields (like connection IDs) with
    colons between byte pairs (e.g. `c4:9a:93:b6:1f:d1:d7:94`), while
    tshark's direct `-T fields` output for the same field returns it
    without colons (e.g. `c49a93b61fd1d794`) -- the identical value, two
    different string representations. Routing every correlation key
    through this function means a lookup like `key in some_dict` compares
    like with like regardless of which tool produced which string.
    """
    return value.replace(":", "").lower()


def extract_sessions_from_pcap(
    pcap_path: str,
    keylog_path: str | None = None,
    display_filter: str = "quic",
) -> list[Session]:
    """Decode a pcap into a list of Sessions, one per UDP flow.

    Groups packets by UDP flow (`udp.stream`, Wireshark's stable index per
    source/destination IP:port 4-tuple) rather than QUIC connection ID.
    QUIC deliberately rotates connection IDs during a connection's
    lifetime for anti-tracking purposes (RFC 9000 §5.1.1) -- a server
    routinely issues a fresh ID for packets sent after the handshake,
    specifically so an observer *can't* link them to the ClientHello by ID
    alone. Grouping by connection ID therefore fragments one real
    connection into several session buckets, and specifically breaks
    HTTP/3 SETTINGS correlation, since SETTINGS travels in a post-handshake
    packet that legitimately carries a different, rotated connection ID
    than the one the ClientHello used -- confirmed empirically against a
    real capture, where the handshake and SETTINGS data were each
    correctly extracted but keyed under IDs differing only in their first
    byte (the rotated portion), causing every correlation lookup to miss.
    UDP 4-tuple stays constant for the life of an ordinary connection
    (baring genuine network-path migration, which is a separate, rarer
    case this basic grouping doesn't handle), so it's a substantially more
    reliable session-grouping key for this purpose.
    """
    try:
        import pyshark  # noqa: PLC0415  (optional dependency, imported lazily)
    except ImportError as exc:  # pragma: no cover - exercised only without pyshark
        raise PcapExtractionError(
            "pyshark is required for pcap extraction; install with "
            "`pip install mlcfq[pcap]` and ensure tshark is on PATH."
        ) from exc

    override_prefs = {}
    if keylog_path:
        override_prefs["tls.keylog_file"] = keylog_path

    try:
        capture = pyshark.FileCapture(
            pcap_path,
            display_filter=display_filter,
            override_prefs=override_prefs,
            use_json=True,
            include_raw=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local tshark install
        raise PcapExtractionError(f"failed to open capture: {exc}") from exc

    raw_by_conn: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "transport": {},
            "handshake": {},
            "http3": {},
            "request_timestamps_ms": [],
        }
    )

    try:
        for packet in capture:
            conn_id = _connection_key(packet)
            if conn_id is None:
                continue
            bucket = raw_by_conn[conn_id]
            _merge_transport_fields(packet, bucket["transport"])
            ts = _packet_timestamp_ms(packet)
            if ts is not None:
                bucket["request_timestamps_ms"].append(ts)
    finally:
        capture.close()

    # TLS handshake fields (ALPN, cipher suites, extensions, supported
    # groups) and HTTP/3 SETTINGS are nested inside QUIC CRYPTO/STREAM
    # frames respectively. pyshark's per-packet JSON object model does not
    # reliably surface fields nested under a different protocol than the
    # packet's top-level layer -- confirmed empirically against a real
    # capture, where `packet.tls` / `packet.quic._all_fields` surfaced
    # nothing despite the data being present and correctly dissected (as
    # shown by Wireshark's GUI and by `tshark -T fields -e
    # tls.handshake.extensions_alpn_str`, which extracts it correctly).
    # We use that same proven-reliable direct field-extraction method here
    # instead of relying on pyshark's object model for these two layers.
    handshake_by_stream, http3_settings_by_stream = _extract_nested_fields_via_tshark(pcap_path, keylog_path)
    # QUIC transport parameters (idle timeout, max data, stream limits,
    # etc.) are also nested inside the ClientHello, same as ALPN/handshake
    # fields above -- same fix, separate function since these use a
    # different tshark field namespace (`tls.quic.parameter.*`).
    transport_by_stream = _extract_transport_params_via_tshark(pcap_path)
    for conn_id, bucket in raw_by_conn.items():
        if conn_id in handshake_by_stream:
            bucket["handshake"].update(handshake_by_stream[conn_id])
        if conn_id in http3_settings_by_stream:
            bucket["http3"]["settings"] = http3_settings_by_stream[conn_id]
            bucket["http3"]["h3_negotiated"] = True
        if conn_id in transport_by_stream:
            bucket["transport"].update(transport_by_stream[conn_id])

    sessions = []
    for conn_id, bucket in raw_by_conn.items():
        sessions.append(
            Session(
                session_id=conn_id,
                transport=build_transport_features(bucket["transport"]),
                handshake=build_handshake_features(bucket["handshake"]),
                http3=build_http3_features(bucket["http3"]),
                behavioral=build_behavioral_features(
                    {"request_timestamps_ms": bucket["request_timestamps_ms"]}
                ),
            )
        )
    return sessions


def _connection_key(packet: Any) -> str | None:
    """Session-grouping key for one packet: UDP flow (4-tuple) index,
    falling back to QUIC connection ID only if `udp.stream` genuinely
    isn't available (unexpected for QUIC traffic, but defensive). See
    `extract_sessions_from_pcap`'s docstring for why UDP flow is preferred.
    """
    udp_layer = getattr(packet, "udp", None)
    if udp_layer is not None:
        stream = getattr(udp_layer, "stream", None)
        if stream is not None and str(stream) != "":
            return f"udpstream-{stream}"

    quic_layer = getattr(packet, "quic", None)
    if quic_layer is None:
        return None
    for attr in ("dcid", "scid", "connection_number"):
        value = getattr(quic_layer, attr, None)
        if value:
            return _normalize_key(str(value))
    return None


def _merge_transport_fields(packet: Any, target: dict[str, Any]) -> None:
    """Merge the one transport-layer field that actually works via pyshark's
    object model: `quic.version`, a native top-level QUIC field (not nested
    inside another protocol's frames, unlike the transport *parameters*
    below).

    The QUIC transport parameters themselves (idle timeout, max data,
    stream limits, etc.) are sent as a TLS extension inside the ClientHello
    per RFC 9001 Section 8.2 -- nested inside QUIC's CRYPTO frame, the same
    structural pattern as ALPN and cipher suites. This function used to
    also attempt those via `quic_layer.tp.*` attributes, which never
    worked for the same reason ALPN didn't: pyshark's object model doesn't
    surface fields nested under a different protocol than the packet's
    top-level layer. See `_extract_transport_params_via_tshark` for the
    fix, using the same proven direct-field-extraction approach as
    handshake and HTTP/3 fields.
    """
    quic_layer = getattr(packet, "quic", None)
    if quic_layer is None:
        return
    value = _get_field(quic_layer, "version")
    if value is not None:
        target["quic.version"] = value


def _extract_nested_fields_via_tshark(
    pcap_path: str, keylog_path: str | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[int, int]]]:
    """Extract TLS handshake fields and HTTP/3 SETTINGS via tshark's direct
    field extractor (`-T fields`), keyed by UDP flow (matching
    `_connection_key`'s primary grouping key -- see
    `extract_sessions_from_pcap`'s docstring for why UDP flow, not QUIC
    connection ID, is used for correlation).

    `keylog_path`, if given, is passed to tshark via `-o
    tls.keylog_file:...` so it can decrypt QUIC's 1-RTT (application data)
    packets. This matters specifically for HTTP/3 SETTINGS: unlike the
    ClientHello (carried in Initial packets, which use well-known public
    keys per RFC 9001 and are always decryptable), HTTP/3 request/response
    and SETTINGS frames travel over 1-RTT packets encrypted with real
    per-session keys -- without the keylog, tshark cannot decrypt them at
    all, and the http3 display filter matches zero packets (not an error,
    just nothing to decode). Handshake extraction works without a keylog;
    HTTP/3 extraction effectively requires one.

    Returns `(handshake_by_stream, http3_settings_by_stream)`, keyed by the
    same `"udpstream-<n>"` strings `_connection_key` produces.
    """
    handshake_by_stream: dict[str, dict[str, list[str]]] = {}
    http3_settings_by_stream: dict[str, dict[int, int]] = {}

    if shutil.which("tshark") is None:
        return handshake_by_stream, http3_settings_by_stream

    keylog_opts = ["-o", f"tls.keylog_file:{keylog_path}"] if keylog_path else []

    handshake_cmd = [
        "tshark", "-r", pcap_path, *keylog_opts,
        # Presence-based filter, matching what was empirically verified to
        # work against a real capture (`tshark -Y
        # "tls.handshake.extensions_alpn_str"` correctly returned ALPN
        # values). An earlier version of this filtered on
        # "tls.handshake.type==1" (ClientHello type-code match) instead,
        # which is a reasonable-looking but *unverified* filter expression
        # -- if it doesn't evaluate the way expected for QUIC-embedded TLS
        # in a given tshark version, it silently matches zero packets
        # rather than erroring, which is indistinguishable from "no ALPN
        # in this capture" without directly testing the filter itself.
        # Filtering on presence of the exact field being extracted avoids
        # that whole class of mistake.
        "-Y", "tls.handshake.extensions_alpn_str",
        "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
        "-e", "udp.stream",
        "-e", "tls.handshake.ciphersuite",
        "-e", "tls.handshake.extension.type",
        "-e", "tls.handshake.extensions_supported_group",
        "-e", "tls.handshake.extensions_alpn_str",
    ]
    result = subprocess.run(handshake_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Surface this loudly rather than silently leaving
        # handshake_by_stream empty -- a non-zero exit here (bad -E option
        # syntax for this tshark version, malformed filter, etc.) is
        # indistinguishable from "no ALPN present" if we don't report it,
        # which is exactly the kind of silent failure that previously made
        # a real bug look like a successful-but-empty result.
        import sys

        print(
            f"[mlcfq.extraction] warning: tshark handshake field extraction failed "
            f"(exit code {result.returncode}); handshake fields will be empty for this "
            f"capture.\ntshark said:\n{result.stderr}",
            file=sys.stderr,
        )
    else:
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 5 or not parts[0]:
                continue
            stream, ciphers, exts, groups, alpn = parts[:5]
            key = f"udpstream-{stream}"
            # Union (not overwrite) across multiple matching lines for the
            # same stream. A large ClientHello -- especially one carrying a
            # PQ hybrid key share, which can run over a kilobyte -- can be
            # fragmented across multiple QUIC Initial packets, each of
            # which may independently match this presence filter and
            # report only part of the full field list. Overwriting on each
            # line silently kept only the last fragment seen, which is
            # exactly the pattern found against real captures: only a
            # small minority of sessions showed the full 3-cipher-suite
            # ClientHello confirmed in Wireshark's GUI, with most showing
            # a single cipher or nothing at all.
            existing = handshake_by_stream.setdefault(
                key, {"cipher_suites": [], "extensions": [], "supported_groups": [], "alpn": []}
            )
            for field_name, raw in (
                ("cipher_suites", ciphers),
                ("extensions", exts),
                ("supported_groups", groups),
                ("alpn", alpn),
            ):
                for v in raw.split(","):
                    if v and v not in existing[field_name]:
                        existing[field_name].append(v)

    http3_cmd = [
        "tshark", "-r", pcap_path, *keylog_opts,
        "-Y", "http3.settings.id",
        "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
        "-e", "udp.stream",
        "-e", "udp.dstport",
        "-e", "http3.settings.id",
        "-e", "http3.settings.value",
    ]
    result3 = subprocess.run(http3_cmd, capture_output=True, text=True)
    if result3.returncode != 0:
        import sys

        print(
            f"[mlcfq.extraction] warning: tshark HTTP/3 field extraction failed "
            f"(exit code {result3.returncode}); http3 settings will be empty for this "
            f"capture.\ntshark said:\n{result3.stderr}",
            file=sys.stderr,
        )
    else:
        for line in result3.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 4 or not parts[0]:
                continue
            stream, dst_port, ids_raw, values_raw = parts[:4]
            # HTTP/3 SETTINGS frames flow in both directions (client->server
            # and server->client each send their own) -- without this
            # filter, whichever direction's frame tshark happened to list
            # would silently win, and empirically that was the *server's*
            # settings, not the client's. x^(A) is meant to be a client-stack
            # fingerprint (paper Section 5.1), so only the frame the client
            # sent *to* the server (standard HTTPS/QUIC port 443 as
            # destination) is kept here. Confirmed necessary against real
            # captures: Chrome and aioquic showed *identical* settings when
            # visiting the same site and *different* settings across sites,
            # before this fix -- a clean signature of capturing the server's
            # settings rather than the client's.
            if dst_port != "443":
                continue
            settings: dict[int, int] = {}
            for ident, val in zip(ids_raw.split(","), values_raw.split(",")):
                try:
                    settings[int(ident, 0)] = int(val)
                except (TypeError, ValueError):
                    continue
            if settings:
                # Merge, not overwrite -- same fragmentation-safety reasoning
                # as the handshake fix above: if the client's own SETTINGS
                # frame is split across multiple packets for any reason, an
                # overwrite would silently drop everything but the last one
                # seen for this stream.
                key = f"udpstream-{stream}"
                http3_settings_by_stream.setdefault(key, {}).update(settings)

    return handshake_by_stream, http3_settings_by_stream


def _extract_transport_params_via_tshark(
    pcap_path: str,
) -> dict[str, dict[str, int]]:
    """Extract QUIC transport parameters via tshark's direct field
    extractor, keyed by UDP flow (matching `_connection_key`'s grouping).

    These are sent as a TLS extension inside the ClientHello (RFC 9001
    Section 8.2) -- nested inside QUIC's CRYPTO frame, the same structural
    pattern as ALPN and cipher suites (see `_extract_nested_fields_via_tshark`),
    which is why they need the same direct-extraction treatment rather than
    pyshark's object model. Field names confirmed against this tshark
    version's own field registry (`tshark -G fields | Select-String
    "tls.quic.parameter"`) rather than assumed -- an earlier version of this
    code guessed `quic.tp.*`, which doesn't exist in the registry at all.

    Like the ClientHello's other contents, these don't require a keylog to
    decrypt (Initial packets use well-known public keys per RFC 9001).

    Note: `disable_active_migration` is deliberately not extracted here.
    It's a zero-length "present means true" parameter per RFC 9000, and
    doesn't appear as its own named field in this tshark version's
    registry (unlike the length-carrying parameters below) -- rather than
    guess at how it's represented, it's left as a known gap. Sessions will
    simply have `disable_active_migration=None` (no evidence either way)
    rather than a wrong value.

    Returns a dict keyed by `"udpstream-<n>"`, with values already using
    the normalized attribute names `build_transport_features` expects
    directly (e.g. `"idle_timeout_ms"`, not a raw tshark field name) --
    simpler than round-tripping through `_TSHARK_FIELD_MAP` for data that
    didn't come from that map's assumed field names in the first place.
    """
    transport_by_stream: dict[str, dict[str, int]] = {}

    if shutil.which("tshark") is None:
        return transport_by_stream

    fields = [
        "max_idle_timeout",
        "initial_max_data",
        "initial_max_stream_data_bidi_local",
        "initial_max_stream_data_bidi_remote",
        "initial_max_stream_data_uni",
        "initial_max_streams_bidi",
        "initial_max_streams_uni",
        "active_connection_id_limit",
        "max_udp_payload_size",
    ]
    normalized_names = {
        "max_idle_timeout": "idle_timeout_ms",
        "initial_max_data": "initial_max_data",
        "initial_max_stream_data_bidi_local": "initial_max_stream_data_bidi_local",
        "initial_max_stream_data_bidi_remote": "initial_max_stream_data_bidi_remote",
        "initial_max_stream_data_uni": "initial_max_stream_data_uni",
        "initial_max_streams_bidi": "initial_max_streams_bidi",
        "initial_max_streams_uni": "initial_max_streams_uni",
        "active_connection_id_limit": "active_connection_id_limit",
        "max_udp_payload_size": "max_udp_payload_size",
    }

    cmd = [
        "tshark", "-r", pcap_path,
        "-Y", "tls.quic.parameter.initial_max_data",  # presence filter, same proven pattern
        "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
        "-e", "udp.stream",
    ]
    for f in fields:
        cmd += ["-e", f"tls.quic.parameter.{f}"]

    cmd = [
        "tshark", "-r", pcap_path,
        "-Y", "tls.quic.parameter.initial_max_data",  # presence filter, same proven pattern
        "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=a", "-E", "aggregator=,",
        "-e", "udp.stream",
        "-e", "udp.dstport",
    ]
    for f in fields:
        cmd += ["-e", f"tls.quic.parameter.{f}"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import sys

        print(
            f"[mlcfq.extraction] warning: tshark transport-parameter extraction failed "
            f"(exit code {result.returncode}); transport fields will fall back to "
            f"quic.version only for this capture.\ntshark said:\n{result.stderr}",
            file=sys.stderr,
        )
        return transport_by_stream

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 + len(fields) or not parts[0]:
            continue
        stream, dst_port = parts[0], parts[1]
        # Same fix as HTTP/3 SETTINGS: QUIC transport parameters are sent
        # by both client (in the ClientHello) and server (in
        # EncryptedExtensions) as the same TLS extension type, so without
        # a direction filter either side's values could be captured.
        # Client's own parameters go out to the server on port 443.
        # Real captures didn't show the same site-correlated symptom
        # HTTP/3 settings did (values were identical across both test
        # sites), but that could plausibly be coincidence -- both
        # cloudflare-quic.com and quic.nginx.org's *server* transport
        # parameters happening to share values -- rather than proof this
        # was already direction-safe, so this filter is applied here too
        # rather than leaving it as an unverified assumption.
        if dst_port != "443":
            continue
        values = parts[2 : 2 + len(fields)]
        entry: dict[str, int] = {}
        for field_name, raw_value in zip(fields, values):
            if not raw_value:
                continue
            try:
                entry[normalized_names[field_name]] = int(raw_value.split(",")[0])
            except (TypeError, ValueError):
                continue
        if entry:
            # Merge, not overwrite -- same fragmentation-safety reasoning as
            # the handshake and HTTP/3 fixes above.
            key = f"udpstream-{stream}"
            transport_by_stream.setdefault(key, {}).update(entry)

    return transport_by_stream


def _packet_timestamp_ms(packet: Any) -> float | None:
    sniff_timestamp = getattr(packet, "sniff_timestamp", None)
    if sniff_timestamp is None:
        return None
    try:
        return float(sniff_timestamp) * 1000.0
    except ValueError:
        return None


def _get_field(layer: Any, field_name: str, all_values: bool = False) -> Any:
    try:
        if all_values and hasattr(layer, "get_field_value"):
            value = layer.get_field_value(field_name, raw=False)
            if value is None:
                return None
            return value if isinstance(value, list) else [value]
        return getattr(layer, field_name.replace(".", "_"), None)
    except AttributeError:
        return None


def iter_sessions_streaming(
    pcap_path: str, keylog_path: str | None = None
) -> Iterator[Session]:
    """Streaming variant for large captures — yields Sessions as connections close.

    Reference implementation note: the batch `extract_sessions_from_pcap`
    above is sufficient for the offline evaluation protocol in paper §7; a
    production deployment watching a live interface should replace this with
    proper connection-close/idle-timeout eviction rather than end-of-capture
    flushing. Left as an extension point.
    """
    raise NotImplementedError(
        "Streaming extraction is an extension point for live-interface "
        "deployments; use extract_sessions_from_pcap for offline captures."
    )
