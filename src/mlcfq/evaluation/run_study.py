"""End-to-end study runner: labeled dataset -> RQ1/RQ2/RQ3 results (paper §7).

This is the script that turns a dataset you've actually captured (via
`capture_harness.py` + `adversarial.py`, real traffic + real recombination)
into the results tables the paper's evaluation protocol calls for. It does
not itself generate or collect any data -- see `docs/EVALUATION_GUIDE.md`
for the full workflow this plugs into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcfq.evaluation.ablations import all_ablations, full_model_transport_weighted_config, run_ablation
from mlcfq.evaluation.baselines import handshake_only_baseline, ja4q_only_baseline
from mlcfq.evaluation.dataset import LabeledSession
from mlcfq.evaluation.metrics import (
    macro_f1,
    per_tier_detection,
    top_k_accuracy,
)
from mlcfq.model import ClientProfile
from mlcfq.scoring.coherence import rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer


@dataclass
class RQ1Result:
    """Attribution accuracy: full model and baselines, Top-1/Top-k/macro-F1."""

    method: str
    top1: float
    top3: float
    macro_f1_top1: float


@dataclass
class RQ2Result:
    """Mimicry-resistance: per-tier ROC/PR-AUC and detection rate at a fixed
    false-positive budget, for the full model vs. each ablation.
    """

    method: str
    per_tier: dict[str, dict[str, float]]


@dataclass
class StudyResults:
    rq1: list[RQ1Result] = field(default_factory=list)
    rq2: list[RQ2Result] = field(default_factory=list)
    dataset_summary: dict[str, Any] = field(default_factory=dict)


def _full_model_ranked_names(session, profiles: list[ClientProfile]) -> list[str]:
    scorer = RuleBasedScorer()
    ranked = rank_profiles(session, profiles, scorer)
    return [r.profile_name for r in ranked]


def run_rq1(
    sessions: list[LabeledSession], profiles: list[ClientProfile]
) -> list[RQ1Result]:
    """RQ1 (attribution): does MLCF-Q improve Top-1/Top-k attribution over
    JA4-Q-only and handshake-only baselines?

    Reports the transport-weighted full model (see
    `ablations.full_model_transport_weighted_config`) as an additional,
    separately-labeled method alongside the uniform-weight full model,
    rather than replacing it -- so the effect of that weighting choice
    stays visible and comparable rather than silently baked in.
    """
    true_labels = [s.ground_truth_class for s in sessions]

    full_model_ranked = [_full_model_ranked_names(s.session, profiles) for s in sessions]

    weighted_config = full_model_transport_weighted_config(profiles)
    weighted_ranked = [
        [r.profile_name for r in run_ablation(s.session, weighted_config)] for s in sessions
    ]

    ja4q_ranked = [ja4q_only_baseline(s, profiles) for s in sessions]
    handshake_ranked = [
        [r.profile_name for r in handshake_only_baseline(s, profiles)] for s in sessions
    ]

    results = []
    for method, ranked in (
        ("full_model", full_model_ranked),
        ("full_model_transport_weighted", weighted_ranked),
        ("ja4q_only_baseline", ja4q_ranked),
        ("handshake_only_baseline", handshake_ranked),
    ):
        top1_preds = [r[0] if r else None for r in ranked]
        results.append(
            RQ1Result(
                method=method,
                top1=top_k_accuracy(true_labels, ranked, k=1),
                top3=top_k_accuracy(true_labels, ranked, k=3),
                macro_f1_top1=macro_f1(true_labels, top1_preds),
            )
        )
    return results


def run_rq2(
    sessions: list[LabeledSession], profiles: list[ClientProfile]
) -> list[RQ2Result]:
    """RQ2 (mimicry resistance): how well does MLCF-Q detect spoofing across
    tiers A1-A3, for the full model and each ablation (paper §7: "pooling
    would let easy A1 detections mask A3 failure", hence per-tier
    reporting).

    For each adversarial tier present in the dataset, this compares that
    tier's sessions against the *same pool of genuine sessions* (not
    against each other, and not pooled across tiers) -- a tier can only be
    meaningfully flagged as anomalous relative to what genuine traffic
    looks like. The decision statistic is each session's own best (Top-1)
    coherence score against its own claimed/true profile; lower score
    (more incoherent) should indicate higher adversarial likelihood.
    """
    configs = all_ablations(profiles)
    genuine_sessions = [s for s in sessions if s.tier == "genuine"]
    adversarial_tiers = sorted({s.tier for s in sessions if s.tier != "genuine"})

    results = []
    for config in configs:
        def score_of(s: LabeledSession) -> float:
            ranked = run_ablation(s.session, config)
            best_score = ranked[0].score if ranked else float("-inf")
            return -best_score  # negate: higher = more anomalous

        genuine_scores = [score_of(s) for s in genuine_sessions]
        adversarial_scores_by_tier = {
            tier: [score_of(s) for s in sessions if s.tier == tier] for tier in adversarial_tiers
        }
        results.append(
            RQ2Result(
                method=config.name,
                per_tier=per_tier_detection(genuine_scores, adversarial_scores_by_tier),
            )
        )
    return results


def run_study(sessions: list[LabeledSession], profiles: list[ClientProfile]) -> StudyResults:
    """Run the full RQ1 + RQ2 study over a labeled dataset.

    RQ3 (robustness across network conditions) isn't a separate function --
    call `run_rq1`/`run_rq2` again after filtering `sessions` down to one
    `network_condition` slice at a time (e.g. `[s for s in sessions if
    s.network_condition.label == "high-loss-mobile-like"]`) and compare
    the resulting accuracy/AUC across slices.
    """
    from mlcfq.evaluation.dataset import summarize

    return StudyResults(
        rq1=run_rq1(sessions, profiles),
        rq2=run_rq2(sessions, profiles),
        dataset_summary=summarize(sessions),
    )


def format_results_markdown(results: StudyResults) -> str:
    """Render study results as a Markdown report -- these are the tables
    reported in the paper's §7.2 pilot-study section, generated from
    exactly this function's output.
    """
    lines = ["## Dataset summary", "", "```", str(results.dataset_summary), "```", ""]

    lines += ["## RQ1: Attribution accuracy", "", "| Method | Top-1 | Top-3 | Macro-F1 |", "|---|---|---|---|"]
    for r in results.rq1:
        lines.append(f"| {r.method} | {r.top1:.3f} | {r.top3:.3f} | {r.macro_f1_top1:.3f} |")
    lines.append("")

    lines += ["## RQ2: Mimicry resistance (per-tier ROC-AUC / PR-AUC)", ""]
    for r in results.rq2:
        lines.append(f"### {r.method}")
        lines.append("")
        lines.append("| Tier | n | ROC-AUC | PR-AUC |")
        lines.append("|---|---|---|---|")
        for tier, metrics in sorted(r.per_tier.items()):
            lines.append(
                f"| {tier} | {metrics['n']} | {metrics['roc_auc']:.3f} | {metrics['pr_auc']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)
