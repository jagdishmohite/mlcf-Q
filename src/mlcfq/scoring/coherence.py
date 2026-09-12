"""Coherence scoring objective and dependency graph (paper §5.2, eq. 1).

    S(k | x) = Σ_L wL·log p_k(x_L)  −  λ Σ_(i,j)∈E Δ_k(x_i, x_j)

This module defines the `CoherenceScorer` protocol that both Instantiation R
(rule-based, `rule_based.py`) and Instantiation L (learned, `learned.py`)
implement, plus the dependency graph E and the top-level `score_session`
that combines per-layer terms and cross-layer mismatch penalties into S(k|x)
for every candidate client profile, returning a ranked result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mlcfq.model import ClientProfile, Session

LAYER_NAMES = ("transport", "handshake", "http3", "behavioral")

# Default dependency graph E seeded from protocol-stack knowledge (§5.2):
# every layer pair is checked for mismatch by default. A deployment that has
# measured mutual information between layers on benign data can prune this
# to the edges that actually carry signal.
DEFAULT_DEPENDENCY_GRAPH: tuple[tuple[str, str], ...] = (
    ("handshake", "transport"),
    ("handshake", "http3"),
    ("transport", "http3"),
    ("http3", "behavioral"),
    ("transport", "behavioral"),
)


class DependencyGraph:
    """The graph E over layers used for cross-layer mismatch penalties."""

    def __init__(self, edges: tuple[tuple[str, str], ...] = DEFAULT_DEPENDENCY_GRAPH):
        for a, b in edges:
            if a not in LAYER_NAMES or b not in LAYER_NAMES:
                raise ValueError(f"unknown layer in edge ({a}, {b})")
        self.edges = edges

    def __iter__(self):
        return iter(self.edges)


class CoherenceScorer(Protocol):
    """Interface shared by Instantiation R and Instantiation L (§5.2)."""

    def layer_log_likelihood(self, layer_name: str, session: Session, profile: ClientProfile) -> float:
        """log p_k(x_L) — how well one layer alone matches profile k."""
        ...

    def mismatch_penalty(
        self, layer_i: str, layer_j: str, session: Session, profile: ClientProfile
    ) -> float:
        """Δ_k(x_i, x_j) — cross-layer inconsistency penalty, >= 0."""
        ...


@dataclass
class LayerBreakdown:
    layer: str
    log_likelihood: float
    weight: float


@dataclass
class MismatchBreakdown:
    layer_i: str
    layer_j: str
    penalty: float


@dataclass
class CoherenceResult:
    """Score for one (session, profile) pair, with a full audit trail."""

    profile_name: str
    score: float
    layer_terms: list[LayerBreakdown] = field(default_factory=list)
    mismatch_terms: list[MismatchBreakdown] = field(default_factory=list)

    @property
    def total_mismatch(self) -> float:
        return sum(m.penalty for m in self.mismatch_terms)

    @property
    def worst_mismatch(self) -> MismatchBreakdown | None:
        if not self.mismatch_terms:
            return None
        return max(self.mismatch_terms, key=lambda m: m.penalty)


def score_session(
    session: Session,
    profile: ClientProfile,
    scorer: CoherenceScorer,
    weights: dict[str, float] | None = None,
    lam: float = 1.0,
    graph: DependencyGraph | None = None,
) -> CoherenceResult:
    """Compute S(k | x) for one session against one candidate profile k.

    `weights` maps layer name -> wL (defaults to 1.0 for every layer).
    `lam` is the λ cross-layer penalty coefficient from eq. 1.
    """
    weights = weights or {name: 1.0 for name in LAYER_NAMES}
    graph = graph or DependencyGraph()

    layer_terms = [
        LayerBreakdown(
            layer=layer,
            log_likelihood=scorer.layer_log_likelihood(layer, session, profile),
            weight=weights.get(layer, 1.0),
        )
        for layer in LAYER_NAMES
    ]
    layer_score = sum(t.weight * t.log_likelihood for t in layer_terms)

    mismatch_terms = [
        MismatchBreakdown(
            layer_i=i,
            layer_j=j,
            penalty=scorer.mismatch_penalty(i, j, session, profile),
        )
        for (i, j) in graph
    ]
    mismatch_score = sum(m.penalty for m in mismatch_terms)

    total = layer_score - lam * mismatch_score
    return CoherenceResult(
        profile_name=profile.name,
        score=total,
        layer_terms=layer_terms,
        mismatch_terms=mismatch_terms,
    )


def rank_profiles(
    session: Session,
    profiles: list[ClientProfile],
    scorer: CoherenceScorer,
    weights: dict[str, float] | None = None,
    lam: float = 1.0,
    graph: DependencyGraph | None = None,
) -> list[CoherenceResult]:
    """Score a session against every candidate profile, best (Top-1) first."""
    results = [
        score_session(session, profile, scorer, weights=weights, lam=lam, graph=graph)
        for profile in profiles
    ]
    return sorted(results, key=lambda r: r.score, reverse=True)
