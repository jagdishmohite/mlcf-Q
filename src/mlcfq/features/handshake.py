"""x_H: embedded ClientHello / handshake feature layer (paper §5.1, §3.3).

Includes post-quantum key-share presence as an explicit, first-class
feature (paper §1, §9) — the field that distinguishes pre-PQ-pinned
impersonation profiles from a current genuine browser stack.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from mlcfq.model import HandshakeFeatures

# Named/hybrid PQ groups as advertised in the TLS supported_groups extension.
# Extend this set as new IANA codepoints are registered; kept as data rather
# than hardcoded logic so it's a one-line update.
PQ_GROUP_NAMES = {
    "X25519Kyber768Draft00",
    "X25519MLKEM768",
    "SecP256r1MLKEM768",
    "GREASE_ML-KEM",  # placeholder for GREASE-permuted PQ codepoints
}


def build_handshake_features(raw: Mapping[str, Any]) -> HandshakeFeatures:
    """Build x_H from a raw ClientHello field dict.

    Expects (optionally) `cipher_suites`, `extensions`, `supported_groups`,
    `alpn` as lists of strings, plus `key_share_size_bytes`. PQ presence is
    derived automatically from `supported_groups` if `pq_key_share_offered`
    isn't already provided.
    """
    data = dict(raw)

    groups = list(data.get("supported_groups", []) or [])
    pq_offered = data.get("pq_key_share_offered")
    pq_group = data.get("pq_group_name")
    if pq_offered is None:
        matched = [g for g in groups if g in PQ_GROUP_NAMES]
        pq_offered = bool(matched)
        pq_group = matched[0] if matched else None

    known = {f.name for f in dataclasses.fields(HandshakeFeatures)}
    filtered = {k: v for k, v in data.items() if k in known}

    filtered["pq_key_share_offered"] = pq_offered
    filtered["pq_group_name"] = pq_group

    return HandshakeFeatures(**filtered)


def handshake_key(features: HandshakeFeatures) -> str:
    """The 'primary handshake key' used for cohort grouping (paper §5.3).

    Falls back to a synthesized JA4-Q-like string from cipher/extension/ALPN
    ordering if `ja4_q` wasn't supplied directly.
    """
    if features.ja4_q:
        return features.ja4_q
    parts = [
        ",".join(features.cipher_suites),
        ",".join(features.extensions),
        ",".join(features.alpn),
    ]
    return "synth:" + "|".join(parts)
