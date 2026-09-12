"""Baseline attribution methods (paper §7: "Baselines: JA4-Q-only and
handshake-only attribution").

Both baselines are single-layer by construction -- they exist specifically
to be the thing MLCF-Q's cross-layer coherence is compared against (paper
§1's central claim: coherence catches what single-layer fingerprints miss).
"""

from __future__ import annotations

from mlcfq.evaluation.dataset import LabeledSession
from mlcfq.features.handshake import handshake_key
from mlcfq.model import ClientProfile
from mlcfq.scoring.coherence import CoherenceResult, rank_profiles, score_session
from mlcfq.scoring.rule_based import RuleBasedScorer


def ja4q_only_baseline(
    labeled_session: LabeledSession, profiles: list[ClientProfile]
) -> list[str]:
    """Rank profiles purely by exact JA4-Q (handshake-key) string match.

    This is the simplest possible baseline: it's exactly what an observer
    using JA4-Q alone would report, with no notion of transport, HTTP/3, or
    behavioral coherence at all. Returns profile names ranked best-first;
    an exact match is rank 0, everything else is an arbitrary tied rank
    (JA4-Q alone gives no way to break ties among non-matching profiles).

    Expects each profile's reference JA4-Q string under
    `profile.expected["handshake"]["ja4q_reference"]` -- add this field to
    your profile JSON (alongside the existing `alpn_contains`,
    `pq_key_share_expected`, etc.) for classes you want this baseline to
    recognize.
    """
    observed_key = handshake_key(labeled_session.session.handshake)
    matches = [
        p.name
        for p in profiles
        if p.expected.get("handshake", {}).get("ja4q_reference") == observed_key
    ]
    non_matches = [p.name for p in profiles if p.name not in matches]
    return matches + non_matches


def handshake_only_baseline(
    labeled_session: LabeledSession, profiles: list[ClientProfile]
) -> list[CoherenceResult]:
    """Rank profiles using MLCF-Q's own rule-based scorer, but with only the
    handshake layer weighted and cross-layer mismatch penalties disabled
    (lambda=0).

    This isolates exactly the information a handshake-only fingerprint
    (like JA4-Q, if it scored gradually instead of exact-matching) would
    have -- structured cipher/extension/ALPN/PQ comparison, but with no
    visibility into transport, HTTP/3, or behavioral layers, and no
    cross-layer coherence term. Comparing this against the full model's
    results isolates the value added by cross-layer coherence specifically
    (as opposed to just the extra features in x^Q, x^A, x^B on their own --
    see `ablations.py` for that finer-grained comparison).
    """
    weights = {"transport": 0.0, "handshake": 1.0, "http3": 0.0, "behavioral": 0.0}
    scorer = RuleBasedScorer()
    return rank_profiles(labeled_session.session, profiles, scorer, weights=weights, lam=0.0)


def top_k_predictions(ranked_names: list[str], k: int) -> list[str]:
    """Convenience: first k names from a ranked list (baselines return names;
    full-model results return `CoherenceResult` -- use
    `[r.profile_name for r in ranked_results]` to get names from those).
    """
    return ranked_names[:k]
