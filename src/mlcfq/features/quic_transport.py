"""x_Q: QUIC transport parameter feature layer (paper §5.1, §3.2).

This is the layer JA4-Q omits by design. Values here come straight off the
QUIC transport parameter extension carried in the Initial/Handshake crypto
frames, plus a couple of derived Initial-packet artifacts.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from mlcfq.model import QuicTransportFeatures

# Field names as they typically appear in tshark's QUIC dissector
# (quic.tp.*) — used by mlcfq.extraction, kept here so the mapping lives
# next to the feature it produces.
_TSHARK_FIELD_MAP = {
    "quic.version": "quic_version",
    "quic.tp.max_idle_timeout": "idle_timeout_ms",
    "quic.tp.initial_max_data": "initial_max_data",
    "quic.tp.initial_max_stream_data_bidi_local": "initial_max_stream_data_bidi_local",
    "quic.tp.initial_max_stream_data_bidi_remote": "initial_max_stream_data_bidi_remote",
    "quic.tp.initial_max_stream_data_uni": "initial_max_stream_data_uni",
    "quic.tp.initial_max_streams_bidi": "initial_max_streams_bidi",
    "quic.tp.initial_max_streams_uni": "initial_max_streams_uni",
    "quic.tp.active_connection_id_limit": "active_connection_id_limit",
    "quic.tp.max_udp_payload_size": "max_udp_payload_size",
    "quic.tp.disable_active_migration": "disable_active_migration",
}


def build_transport_features(raw: Mapping[str, Any]) -> QuicTransportFeatures:
    """Build x_Q from a raw field dict (tshark JSON keys or already-normalized keys).

    `raw` may use either the tshark dissector field names (`quic.tp.*`) or
    the already-normalized attribute names on `QuicTransportFeatures` —
    this lets the same builder consume both live-capture output and
    hand-authored test/example fixtures.
    """
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        target = _TSHARK_FIELD_MAP.get(key, key)
        normalized[target] = value

    # tshark reports booleans as "0"/"1" strings for some TP flags
    if "disable_active_migration" in normalized:
        normalized["disable_active_migration"] = _to_bool(
            normalized["disable_active_migration"]
        )

    known = {f.name for f in dataclasses.fields(QuicTransportFeatures)}
    filtered = {k: v for k, v in normalized.items() if k in known}
    return QuicTransportFeatures(**filtered)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def transport_signature(features: QuicTransportFeatures) -> tuple:
    """A coarse, hashable signature over transport params for cohort grouping.

    Used by mlcfq.cohort for secondary-feature dispersion measurement
    (paper §5.3) — deliberately coarse-grained (bucketed) rather than exact
    values, since benign clients of the same stack/version still show minor
    jitter in some fields.
    """
    return (
        features.quic_version,
        _bucket(features.idle_timeout_ms, 5_000),
        _bucket(features.initial_max_data, 1_000_000),
        _bucket(features.initial_max_streams_bidi, 10),
        features.active_connection_id_limit,
        features.disable_active_migration,
    )


def _bucket(value: Any, width: int) -> Any:
    if value is None or width <= 0:
        return None
    try:
        return round(float(value) / width)
    except (TypeError, ValueError):
        return None
