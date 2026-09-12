"""Core data model for MLCF-Q sessions and per-layer feature vectors.

A `Session` corresponds to one QUIC/HTTP-3 connection observed by a passive
sensor. It bundles the four feature layers from paper Section 5.1:

    x = { x_Q (transport), x_H (handshake), x_A (HTTP/3), x_B (behavioral) }

Implemented with stdlib `dataclasses` (no external schema-validation
dependency) so the core pipeline installs with zero third-party packages;
`click`/`numpy`/`pyshark`/`scikit-learn` remain optional, feature-gated
dependencies for the CLI, learned scorer, and live pcap extraction
respectively.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any


def _known_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Filter a raw dict down to the keys `cls` actually declares as fields.

    Mirrors the "extra fields ignored" behavior of a lenient schema: raw
    extraction output may carry implementation-specific keys the feature
    builders don't need, and callers shouldn't have to strip those by hand.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass
class QuicTransportFeatures:
    """x_Q: QUIC transport parameters and Initial-packet artifacts.

    These are exactly the fields JA4-Q omits (paper §3.2) and are the
    primary differentiator of MLCF-Q from single-layer QUIC fingerprints.
    """

    quic_version: str | None = None
    idle_timeout_ms: int | None = None
    initial_max_data: int | None = None
    initial_max_stream_data_bidi_local: int | None = None
    initial_max_stream_data_bidi_remote: int | None = None
    initial_max_stream_data_uni: int | None = None
    initial_max_streams_bidi: int | None = None
    initial_max_streams_uni: int | None = None
    active_connection_id_limit: int | None = None
    max_udp_payload_size: int | None = None
    disable_active_migration: bool | None = None
    migration_observed: bool | None = None
    initial_packet_size: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuicTransportFeatures":
        return cls(**_known_kwargs(cls, data))


@dataclass
class HandshakeFeatures:
    """x_H: embedded TLS ClientHello structure, incl. PQ key-share presence."""

    cipher_suites: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    supported_groups: list[str] = field(default_factory=list)
    alpn: list[str] = field(default_factory=list)
    pq_key_share_offered: bool | None = None
    pq_group_name: str | None = None
    key_share_size_bytes: int | None = None
    ja4_q: str | None = None  # primary handshake key used for cohorting (§5.3)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandshakeFeatures":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Http3Features:
    """x_A: HTTP/3 SETTINGS, QPACK dynamics, stream/pseudo-header behavior."""

    settings: dict[str, int] = field(default_factory=dict)
    qpack_max_table_capacity: int | None = None
    qpack_blocked_streams: int | None = None
    stream_concurrency: int | None = None
    pseudo_header_order: list[str] = field(default_factory=list)
    h3_negotiated: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Http3Features":
        return cls(**_known_kwargs(cls, data))


@dataclass
class BehavioralFeatures:
    """x_B: session-level timing behavior."""

    inter_request_cadence_ms: list[float] = field(default_factory=list)
    connection_reuse: bool | None = None
    zero_rtt_used: bool | None = None
    burstiness: float | None = None  # e.g. coefficient of variation of cadence

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BehavioralFeatures":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Session:
    """One observed QUIC/HTTP-3 session with all four feature layers."""

    session_id: str
    src_context: str | None = None  # e.g. anonymized/hashed vantage identifier
    claimed_class: str | None = None  # optional label, e.g. for eval datasets
    transport: QuicTransportFeatures = field(default_factory=QuicTransportFeatures)
    handshake: HandshakeFeatures = field(default_factory=HandshakeFeatures)
    http3: Http3Features = field(default_factory=Http3Features)
    behavioral: BehavioralFeatures = field(default_factory=BehavioralFeatures)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        kwargs = _known_kwargs(cls, data)
        if "transport" in kwargs:
            kwargs["transport"] = QuicTransportFeatures.from_dict(kwargs["transport"])
        if "handshake" in kwargs:
            kwargs["handshake"] = HandshakeFeatures.from_dict(kwargs["handshake"])
        if "http3" in kwargs:
            kwargs["http3"] = Http3Features.from_dict(kwargs["http3"])
        if "behavioral" in kwargs:
            kwargs["behavioral"] = BehavioralFeatures.from_dict(kwargs["behavioral"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ClientProfile:
    """A known client-stack profile (paper §5.1/5.2), e.g. 'Chrome-124/Windows'.

    `expected` holds representative/summary values per layer used by the
    rule-based scorer's predicates (Instantiation R, §5.2). This is a
    lightweight profile format for the reference implementation; a learned
    scorer would instead fit p_k(x_L) distributions from labeled data.
    """

    name: str
    expected: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientProfile":
        return cls(**_known_kwargs(cls, data))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
