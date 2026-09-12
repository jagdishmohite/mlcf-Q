"""Coherence scoring — paper Section 5.2.

    S(k | x) = Σ_L wL·log p_k(x_L)  −  λ Σ_(i,j)∈E Δ_k(x_i, x_j)

`coherence.py` implements the scoring objective and dependency graph E.
`rule_based.py` implements Instantiation R (Δ_k as expert predicates).
`learned.py` implements Instantiation L (Δ_k as a learned conditional,
requires scikit-learn + labeled data; extension point in this reference
release).
"""

from mlcfq.scoring.coherence import CoherenceResult, DependencyGraph, score_session
from mlcfq.scoring.rule_based import RuleBasedScorer

__all__ = ["score_session", "CoherenceResult", "DependencyGraph", "RuleBasedScorer"]
