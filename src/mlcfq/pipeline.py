"""End-to-end pipeline glue: load sessions/profiles, score, run cohort test.

Mirrors the reference-implementation blueprint in paper §6:
capture → per-layer features → coherence score.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlcfq.cohort.dispersion import CohortDispersionResult, compute_cohort_dispersion
from mlcfq.model import ClientProfile, Session
from mlcfq.scoring.coherence import CoherenceResult, rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer


def load_sessions(path: str | Path) -> list[Session]:
    """Load sessions from a JSON file: either a single session object or a
    list of session objects (as produced by `mlcfq extract` or hand-authored
    fixtures like those in examples/).
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    return [Session.from_dict(item) for item in data]


def load_profiles(path: str | Path) -> list[ClientProfile]:
    """Load candidate client-stack profiles from a JSON file (list of objects)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    return [ClientProfile.from_dict(item) for item in data]


def score_sessions(
    sessions: list[Session],
    profiles: list[ClientProfile],
    lam: float = 1.0,
) -> dict[str, list[CoherenceResult]]:
    """Score every session against every profile using Instantiation R.

    Returns a dict of session_id -> ranked list of CoherenceResult
    (best/Top-1 match first).
    """
    scorer = RuleBasedScorer()
    return {
        session.session_id: rank_profiles(session, profiles, scorer, lam=lam)
        for session in sessions
    }


def run_cohort_test(
    sessions: list[Session],
    baseline_by_key: dict[str, float] | None = None,
    flag_threshold: float = 1.5,
    min_cohort_size: int = 3,
) -> list[CohortDispersionResult]:
    """Run the §5.3 cohort-dispersion test over a batch of sessions."""
    return compute_cohort_dispersion(
        sessions,
        baseline_by_key=baseline_by_key,
        flag_threshold=flag_threshold,
        min_cohort_size=min_cohort_size,
    )


def result_to_dict(result: CoherenceResult) -> dict:
    """JSON-friendly serialization of a CoherenceResult, for CLI output."""
    return {
        "profile": result.profile_name,
        "score": round(result.score, 4),
        "layers": [
            {"layer": t.layer, "log_likelihood": round(t.log_likelihood, 4), "weight": t.weight}
            for t in result.layer_terms
        ],
        "mismatches": [
            {"layers": [m.layer_i, m.layer_j], "penalty": round(m.penalty, 4)}
            for m in result.mismatch_terms
            if m.penalty > 0
        ],
    }
