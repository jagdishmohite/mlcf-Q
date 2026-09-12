"""Instantiation R — rule-based coherence scorer (paper §5.2).

Δ_k sums indicator penalties over expert predicates encoding known QUIC
couplings (e.g. "ALPN advertises h3 ⇒ QUIC transport actually negotiated
HTTP/3", or "handshake claims a recent Chrome build ⇒ transport parameters
fall within the observed Chrome-QUIC family"). This instantiation is
auditable and needs no adversarial training data, at the cost of only
covering anticipated couplings — exactly the tradeoff described in the
paper.

`ClientProfile.expected` is read with the following optional keys:

    expected = {
        "transport": {
            "quic_version": "1" | ["1", "2"],
            "idle_timeout_ms": [min, max],
            "initial_max_data": [min, max],
            "initial_max_streams_bidi": [min, max],
            "active_connection_id_limit": [min, max],
            "disable_active_migration": true | false,
        },
        "handshake": {
            "alpn_contains": "h3",
            "cipher_suites_any_of": [...],
            "supported_groups_any_of": [...],
            "pq_key_share_expected": true | false | null,  # null = no constraint
        },
        "http3": {
            "qpack_max_table_capacity": [min, max],
            "requires_settings": ["QPACK_MAX_TABLE_CAPACITY", ...],
        },
        "behavioral": {
            "min_burstiness": 0.15,   # near-zero cadence variance is suspicious
            "zero_rtt_typical": true | false | null,
        },
    }

Every constraint is optional; omitted constraints simply aren't scored.
"""

from __future__ import annotations

from typing import Any, Callable

from mlcfq.model import ClientProfile, Session

# ---- per-layer likelihood: fraction of declared constraints satisfied -----


def layer_log_likelihood(layer_name: str, session: Session, profile: ClientProfile) -> float:
    """log p_k(x_L), approximated as log of (satisfied / declared) constraints.

    With no declared constraints for a layer, returns 0.0 (no evidence
    either way, rather than penalizing). Each violated constraint costs a
    fixed log-penalty; this is a rule-based stand-in for a real density,
    intentionally simple so it stays auditable.
    """
    constraints = profile.expected.get(layer_name, {}) or {}
    if not constraints:
        return 0.0

    checker = _LAYER_CHECKERS.get(layer_name)
    if checker is None:
        return 0.0

    satisfied, total = checker(session, constraints)
    if total == 0:
        return 0.0
    violated = total - satisfied
    # Each violated constraint costs a fixed penalty in log-space; matches
    # accrue no bonus beyond 0 (we're scoring "did this layer contradict the
    # profile", not rewarding coincidental extra similarity).
    return -1.0 * violated


def _check_transport(session: Session, constraints: dict[str, Any]) -> tuple[int, int]:
    t = session.transport
    checks = []
    if "quic_version" in constraints:
        expected = constraints["quic_version"]
        allowed = expected if isinstance(expected, list) else [expected]
        checks.append(t.quic_version in allowed if t.quic_version is not None else True)
    for field_name in (
        "idle_timeout_ms",
        "initial_max_data",
        "initial_max_streams_bidi",
        "active_connection_id_limit",
    ):
        if field_name in constraints:
            checks.append(_in_range(getattr(t, field_name), constraints[field_name]))
    if "disable_active_migration" in constraints:
        expected = constraints["disable_active_migration"]
        checks.append(
            t.disable_active_migration == expected
            if t.disable_active_migration is not None
            else True
        )
    return sum(checks), len(checks)


def _check_handshake(session: Session, constraints: dict[str, Any]) -> tuple[int, int]:
    h = session.handshake
    checks = []
    if "alpn_contains" in constraints:
        checks.append(constraints["alpn_contains"] in h.alpn if h.alpn else True)
    if "cipher_suites_any_of" in constraints:
        allowed = set(constraints["cipher_suites_any_of"])
        checks.append(bool(allowed & set(h.cipher_suites)) if h.cipher_suites else True)
    if "supported_groups_any_of" in constraints:
        allowed = set(constraints["supported_groups_any_of"])
        checks.append(bool(allowed & set(h.supported_groups)) if h.supported_groups else True)
    if "pq_key_share_expected" in constraints and constraints["pq_key_share_expected"] is not None:
        checks.append(h.pq_key_share_offered == constraints["pq_key_share_expected"])
    return sum(checks), len(checks)


def _check_http3(session: Session, constraints: dict[str, Any]) -> tuple[int, int]:
    a = session.http3
    if a.h3_negotiated is None:
        # No HTTP/3 telemetry captured for this session at all (extraction
        # gap, not evidence the client doesn't use HTTP/3) -- treat as "no
        # evidence" the same way a missing numeric observation already is,
        # rather than scoring every requires_settings entry as violated.
        # Confirmed necessary against real captures: a small fraction of
        # genuine sessions have no HTTP/3 data purely because a capture
        # ended before SETTINGS was sent, and without this check they were
        # scoring -3.0 against their own correct profile, occasionally
        # enough to cause real misattribution.
        return 0, 0
    checks = []
    if "qpack_max_table_capacity" in constraints:
        checks.append(
            _in_range(a.qpack_max_table_capacity, constraints["qpack_max_table_capacity"])
        )
    if "requires_settings" in constraints:
        for name in constraints["requires_settings"]:
            checks.append(name in a.settings)
    return sum(checks), len(checks)


def _check_behavioral(session: Session, constraints: dict[str, Any]) -> tuple[int, int]:
    b = session.behavioral
    checks = []
    if "min_burstiness" in constraints and b.burstiness is not None:
        checks.append(b.burstiness >= constraints["min_burstiness"])
    if "zero_rtt_typical" in constraints and constraints["zero_rtt_typical"] is not None:
        checks.append(
            b.zero_rtt_used == constraints["zero_rtt_typical"]
            if b.zero_rtt_used is not None
            else True
        )
    return sum(checks), len(checks)


_LAYER_CHECKERS: dict[str, Callable[[Session, dict[str, Any]], tuple[int, int]]] = {
    "transport": _check_transport,
    "handshake": _check_handshake,
    "http3": _check_http3,
    "behavioral": _check_behavioral,
}


def _in_range(value: Any, bounds: list) -> bool:
    if value is None:
        return True  # missing observation isn't treated as a violation
    lo, hi = bounds
    try:
        return lo <= value <= hi
    except TypeError:
        return True


# ---- cross-layer mismatch predicates (Δ_k) --------------------------------

# Each predicate returns a penalty in [0, PREDICATE_PENALTY] and documents
# the coupling it encodes, per paper §5.2's worked examples.
PREDICATE_PENALTY = 1.0


def _predicate_alpn_h3_implies_negotiated(session: Session, profile: ClientProfile) -> float:
    """'ALPN advertises h3 ⇒ QUIC transport actually negotiated' (paper §5.2)."""
    h, a = session.handshake, session.http3
    if h.alpn and "h3" in h.alpn:
        if a.h3_negotiated is False:
            return PREDICATE_PENALTY
    return 0.0


def _predicate_h3_implies_qpack_settings(session: Session, profile: ClientProfile) -> float:
    """If HTTP/3 is negotiated, QPACK table-capacity setting should be present."""
    a = session.http3
    if a.h3_negotiated and a.qpack_max_table_capacity is None and a.settings:
        return PREDICATE_PENALTY
    return 0.0


def _predicate_pq_consistency_with_profile(session: Session, profile: ClientProfile) -> float:
    """'Handshake claims [recent stack] ⇒ transport parameters within expected family.'

    Approximated here via the profile's own declared pq_key_share_expected:
    if the profile expects PQ but the transport-layer version/parameters
    look like an older pre-PQ-era client family, that's a coupling
    violation worth flagging (a pre-PQ-pinned impersonation profile is
    exactly the scenario paper §1/§9 calls out).
    """
    h = session.handshake
    handshake_constraints = profile.expected.get("handshake", {}) or {}
    expected_pq = handshake_constraints.get("pq_key_share_expected")
    if expected_pq is None or h.pq_key_share_offered is None:
        return 0.0
    if expected_pq and not h.pq_key_share_offered:
        return PREDICATE_PENALTY
    return 0.0


def _predicate_migration_flag_consistency(session: Session, profile: ClientProfile) -> float:
    """disable_active_migration=True but migration was actually observed."""
    t = session.transport
    if t.disable_active_migration and t.migration_observed:
        return PREDICATE_PENALTY
    return 0.0


def _predicate_zero_rtt_vs_reuse(session: Session, profile: ClientProfile) -> float:
    """0-RTT used without any indication of connection reuse is anomalous."""
    b = session.behavioral
    if b.zero_rtt_used and b.connection_reuse is False:
        return PREDICATE_PENALTY
    return 0.0


# Maps a dependency-graph edge (layer_i, layer_j) to the predicates that
# apply to it. A pair may have zero, one, or several predicates; unmatched
# edges simply contribute 0 penalty (§5.2: "covers only anticipated
# couplings").
_EDGE_PREDICATES: dict[tuple[str, str], list[Callable[[Session, ClientProfile], float]]] = {
    ("handshake", "http3"): [_predicate_alpn_h3_implies_negotiated],
    ("transport", "http3"): [_predicate_h3_implies_qpack_settings],
    ("handshake", "transport"): [_predicate_pq_consistency_with_profile],
    ("transport", "behavioral"): [_predicate_migration_flag_consistency],
    ("http3", "behavioral"): [_predicate_zero_rtt_vs_reuse],
}


def mismatch_penalty(layer_i: str, layer_j: str, session: Session, profile: ClientProfile) -> float:
    predicates = _EDGE_PREDICATES.get((layer_i, layer_j)) or _EDGE_PREDICATES.get(
        (layer_j, layer_i), []
    )
    if not predicates:
        return 0.0
    return sum(p(session, profile) for p in predicates)


class RuleBasedScorer:
    """Instantiation R implementation of the `CoherenceScorer` protocol."""

    def layer_log_likelihood(self, layer_name: str, session: Session, profile: ClientProfile) -> float:
        return layer_log_likelihood(layer_name, session, profile)

    def mismatch_penalty(
        self, layer_i: str, layer_j: str, session: Session, profile: ClientProfile
    ) -> float:
        return mismatch_penalty(layer_i, layer_j, session, profile)
