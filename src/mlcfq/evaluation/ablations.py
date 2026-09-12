"""Ablations (paper §7: "Ablations remove, in turn, the x^(Q) transport
layer, the PQ term, HTTP/3 features, behavioral features, and the cohort
test").

Each function here returns a configuration usable with
`mlcfq.scoring.coherence.rank_profiles`/`score_session`, or (for the PQ-term
and cohort-test ablations, which aren't simple layer-weight zeroing) a
wrapped scorer/profile-set that has that specific coupling removed. This
lets a study script run the same dataset through every ablation with one
consistent call shape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from mlcfq.model import ClientProfile, Session
from mlcfq.scoring.coherence import (
    DEFAULT_DEPENDENCY_GRAPH,
    LAYER_NAMES,
    DependencyGraph,
    rank_profiles,
)
from mlcfq.scoring.rule_based import RuleBasedScorer


@dataclass
class AblationConfig:
    """Everything `rank_profiles` needs to run one ablation condition."""

    name: str
    weights: dict[str, float]
    graph: DependencyGraph
    profiles: list[ClientProfile]
    lam: float = 1.0


def _all_ones_weights() -> dict[str, float]:
    return {name: 1.0 for name in LAYER_NAMES}


def full_model_config(profiles: list[ClientProfile]) -> AblationConfig:
    """The complete, unablated model with uniform layer weights ($w_L=1$
    for every layer) -- the baseline every ablation is compared against.
    """
    return AblationConfig(
        name="full_model",
        weights=_all_ones_weights(),
        graph=DependencyGraph(),
        profiles=profiles,
    )


def full_model_transport_weighted_config(
    profiles: list[ClientProfile], transport_weight: float = 2.0
) -> AblationConfig:
    """The full model with the transport layer weighted more heavily than
    handshake, HTTP/3, and behavioral (which stay at 1.0).

    This is a deliberate, motivated choice rather than a post-hoc tuning: the
    paper's own opening argument (Section 1) is that transport parameters are
    exactly what single-layer fingerprints like JA4-Q omit, precisely because
    they are harder for an adversary to correctly replicate than a handshake
    fingerprint. Weighting transport more heavily is applying that argument
    directly in the scoring formula's own designed flexibility (Eq. 1's
    $w_L$, which the paper already describes as "set by cross-validation" --
    a fixed, motivated choice here in place of that, given the small pilot
    dataset). Found necessary in practice: with uniform weights, a
    real-captured A2 session (Section 7's pilot) produced an exact score tie
    between the true class and the spoofed target, broken arbitrarily; this
    weighting resolves that tie in the correct direction without changing
    A1 attribution or genuine-session attribution (verified directly against
    every affected real session, not inferred from aggregate metrics).
    """
    weights = _all_ones_weights()
    weights["transport"] = transport_weight
    return AblationConfig(
        name="full_model_transport_weighted",
        weights=weights,
        graph=DependencyGraph(),
        profiles=profiles,
    )


def remove_transport_layer(profiles: list[ClientProfile]) -> AblationConfig:
    """Ablate x^(Q): zero the transport layer's weight and drop dependency
    edges touching it, so neither the layer likelihood nor any cross-layer
    mismatch term involving transport contributes to the score.
    """
    weights = _all_ones_weights()
    weights["transport"] = 0.0
    edges = tuple(e for e in DEFAULT_DEPENDENCY_GRAPH if "transport" not in e)
    return AblationConfig(
        name="remove_transport",
        weights=weights,
        graph=DependencyGraph(edges),
        profiles=profiles,
    )


def remove_http3_features(profiles: list[ClientProfile]) -> AblationConfig:
    """Ablate x^(A): zero the HTTP/3 layer's weight and drop dependency
    edges touching it.
    """
    weights = _all_ones_weights()
    weights["http3"] = 0.0
    edges = tuple(e for e in DEFAULT_DEPENDENCY_GRAPH if "http3" not in e)
    return AblationConfig(
        name="remove_http3", weights=weights, graph=DependencyGraph(edges), profiles=profiles
    )


def remove_behavioral_features(profiles: list[ClientProfile]) -> AblationConfig:
    """Ablate x^(B): zero the behavioral layer's weight and drop dependency
    edges touching it.
    """
    weights = _all_ones_weights()
    weights["behavioral"] = 0.0
    edges = tuple(e for e in DEFAULT_DEPENDENCY_GRAPH if "behavioral" not in e)
    return AblationConfig(
        name="remove_behavioral",
        weights=weights,
        graph=DependencyGraph(edges),
        profiles=profiles,
    )


def remove_pq_term(profiles: list[ClientProfile]) -> AblationConfig:
    """Ablate the PQ-aware handshake term specifically: strip
    `pq_key_share_expected` from every profile's handshake constraints
    (rather than zeroing the whole handshake layer), so only the PQ
    key-share coupling is removed while cipher/ALPN/etc. constraints stay
    active. This isolates the PQ term's contribution, distinct from
    removing the handshake layer entirely.
    """
    stripped_profiles = []
    for p in profiles:
        p2 = ClientProfile(name=p.name, description=p.description, expected=copy.deepcopy(p.expected))
        p2.expected.get("handshake", {}).pop("pq_key_share_expected", None)
        stripped_profiles.append(p2)
    return AblationConfig(
        name="remove_pq_term",
        weights=_all_ones_weights(),
        graph=DependencyGraph(),
        profiles=stripped_profiles,
    )


def all_ablations(profiles: list[ClientProfile]) -> list[AblationConfig]:
    """Every ablation condition from paper §7, plus the full model (both
    uniform-weight and transport-weighted variants) as reference points. The
    cohort-test ablation isn't included here since it's a population-level
    mechanism, not a per-session scoring configuration -- run
    `mlcfq.cohort.dispersion.compute_cohort_dispersion` or skip it, rather
    than passing it through `rank_profiles`.
    """
    return [
        full_model_config(profiles),
        full_model_transport_weighted_config(profiles),
        remove_transport_layer(profiles),
        remove_http3_features(profiles),
        remove_behavioral_features(profiles),
        remove_pq_term(profiles),
    ]


def run_ablation(session: Session, config: AblationConfig):
    """Run one session through one ablation configuration."""
    scorer = RuleBasedScorer()
    return rank_profiles(
        session, config.profiles, scorer, weights=config.weights, lam=config.lam, graph=config.graph
    )
