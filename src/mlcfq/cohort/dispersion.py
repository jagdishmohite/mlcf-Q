"""Cohort dispersion test implementation (paper §5.3).

Groups sessions by primary handshake key (JA4-Q-like), then measures
dispersion of secondary descriptors (QUIC transport-parameter signature)
within each group as normalized categorical entropy, optionally normalized
against a per-fingerprint benign baseline.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from mlcfq.features.handshake import handshake_key
from mlcfq.features.quic_transport import transport_signature
from mlcfq.model import Session


@dataclass
class CohortDispersionResult:
    """Dispersion result for one cohort (all sessions sharing a handshake key)."""

    handshake_key: str
    session_count: int
    unique_signature_count: int
    entropy: float
    max_entropy: float
    normalized_dispersion: float  # entropy / max_entropy, in [0, 1]
    baseline: float | None = None
    dispersion_ratio: float | None = None  # normalized_dispersion / baseline
    flagged: bool = False
    session_ids: list[str] = field(default_factory=list)


def compute_cohort_dispersion(
    sessions: list[Session],
    baseline_by_key: dict[str, float] | None = None,
    flag_threshold: float = 1.5,
    min_cohort_size: int = 3,
) -> list[CohortDispersionResult]:
    """Compute per-cohort transport-parameter dispersion (paper §5.3).

    - `baseline_by_key`: optional per-handshake-key benign baseline for
      `normalized_dispersion` (e.g. measured offline on trusted traffic for
      that fingerprint). When absent, only `normalized_dispersion` and
      `entropy` are reported and `flagged` is left False — there's nothing
      to compare against yet.
    - `flag_threshold`: a cohort is flagged when
      `normalized_dispersion / baseline >= flag_threshold`.
    - `min_cohort_size`: cohorts smaller than this are skipped (dispersion
      over 1-2 sessions is not a meaningful population signal).

    Returns one `CohortDispersionResult` per handshake-key cohort with at
    least `min_cohort_size` sessions.
    """
    cohorts: dict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        key = handshake_key(session.handshake)
        cohorts[key].append(session)

    results = []
    for key, cohort_sessions in cohorts.items():
        if len(cohort_sessions) < min_cohort_size:
            continue

        signatures = [transport_signature(s.transport) for s in cohort_sessions]
        counts = Counter(signatures)
        n = len(signatures)
        entropy = -sum(
            (c / n) * math.log2(c / n) for c in counts.values() if c > 0
        )
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
        normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0

        baseline = (baseline_by_key or {}).get(key)
        ratio = (normalized / baseline) if baseline and baseline > 0 else None
        flagged = bool(ratio is not None and ratio >= flag_threshold)

        results.append(
            CohortDispersionResult(
                handshake_key=key,
                session_count=n,
                unique_signature_count=len(counts),
                entropy=entropy,
                max_entropy=max_entropy,
                normalized_dispersion=normalized,
                baseline=baseline,
                dispersion_ratio=ratio,
                flagged=flagged,
                session_ids=[s.session_id for s in cohort_sessions],
            )
        )

    return sorted(results, key=lambda r: r.normalized_dispersion, reverse=True)
