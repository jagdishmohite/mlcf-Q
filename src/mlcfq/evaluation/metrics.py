"""Metrics for the evaluation study (paper §7).

Implemented in pure Python/stdlib rather than depending on scikit-learn, so
running the study's metrics doesn't require a heavier dependency than the
rest of the core pipeline. If you already have scikit-learn installed (e.g.
for `mlcfq.scoring.learned`), its `roc_auc_score`/`average_precision_score`
will agree with `roc_auc` / `pr_auc` here up to floating-point tie-handling
differences; these are provided so the metrics work standalone.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def top_k_accuracy(true_labels: list[str], ranked_predictions: list[list[str]], k: int) -> float:
    """Fraction of sessions where the true label appears in the top k
    ranked predictions (paper §7 RQ1 metric).

    `ranked_predictions[i]` is the ranked list of predicted class names
    (best first) for the i-th session; `true_labels[i]` is its ground
    truth class.
    """
    if len(true_labels) != len(ranked_predictions):
        raise ValueError("true_labels and ranked_predictions must be the same length")
    if not true_labels:
        return float("nan")
    hits = sum(
        1 for true, ranked in zip(true_labels, ranked_predictions) if true in ranked[:k]
    )
    return hits / len(true_labels)


def macro_f1(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Macro-averaged F1 (paper §7 RQ1 metric): compute F1 per class using
    one-vs-rest, then average across classes unweighted (so rare classes
    count as much as common ones).
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must be the same length")
    classes = sorted(set(true_labels) | set(predicted_labels))
    if not classes:
        return float("nan")

    f1_scores = []
    for c in classes:
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def roc_auc(labels: list[bool], scores: list[float]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney U) formula -- no sklearn
    required. `labels[i]` is True for the positive class (e.g. "this
    session is adversarial"), `scores[i]` is the decision statistic (higher
    = more likely positive).
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined without both classes present

    # Rank scores (average rank for ties), ascending.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for m in range(i, j + 1):
            ranks[order[m]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, label in zip(ranks, labels) if label)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def pr_auc(labels: list[bool], scores: list[float]) -> float:
    """PR-AUC (average precision) -- no sklearn required.

    Computed as the standard step-function average precision: sort by
    score descending, and sum precision at each recall-increasing point,
    weighted by the recall increase.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def per_tier_detection(
    genuine_scores: list[float], adversarial_scores_by_tier: dict[str, list[float]]
) -> dict[str, dict[str, float]]:
    """ROC-AUC and PR-AUC computed separately per adversarial tier (paper
    §7 RQ2: "ROC/PR-AUC reported separately for A1/A2/A3... pooling would
    let easy A1 detections mask A3 failure").

    `genuine_scores` is the decision statistic (higher = more anomalous)
    for every genuine-tier session; `adversarial_scores_by_tier` maps each
    adversarial tier name ("A1", "A2", "A3", ...) to that tier's scores.
    Each tier is compared against the *same* pool of genuine scores --
    this shape makes that the only thing you can do with this function, on
    purpose, since comparing a tier's sessions only to each other (rather
    than to genuine traffic) silently produces a meaningless, undefined
    comparison (no negatives to separate from).
    """
    results = {}
    for tier, tier_scores in adversarial_scores_by_tier.items():
        labels = [False] * len(genuine_scores) + [True] * len(tier_scores)
        combined = genuine_scores + tier_scores
        results[tier] = {
            "roc_auc": roc_auc(labels, combined),
            "pr_auc": pr_auc(labels, combined),
            "n": len(combined),
        }
    return results


def detection_rate_at_fp_budget(
    labels: list[bool], scores: list[float], fp_budget: float
) -> float:
    """True positive rate at a fixed false-positive rate budget (paper §7
    RQ2: "per-tier detection rate at fixed false-positive budgets").

    Sweeps thresholds and returns the best TPR achievable while keeping FPR
    at or below `fp_budget` (e.g. 0.01 for a 1% false-positive budget).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = sorted(set(scores), reverse=True)
    best_tpr = 0.0
    for t in thresholds:
        tp = sum(1 for lbl, s in zip(labels, scores) if lbl and s >= t)
        fp = sum(1 for lbl, s in zip(labels, scores) if not lbl and s >= t)
        fpr = fp / n_neg
        if fpr <= fp_budget:
            tpr = tp / n_pos
            best_tpr = max(best_tpr, tpr)
    return best_tpr


def calibration_curve(
    scores: list[float], outcomes: list[bool], n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability data for S(k|x) (paper §7: "we also recommend a
    calibration curve for S(k | x), since operational triage depends on
    score interpretability").

    Bins sessions by score into `n_bins` equal-width bins and reports the
    observed positive-outcome frequency per bin against the bin's mean
    score, so a plot of one against the other shows whether the score is a
    well-calibrated probability-like quantity.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    width = (hi - lo) / n_bins if hi > lo else 1.0

    bins: dict[int, list[int]] = defaultdict(list)
    for score, outcome in zip(scores, outcomes):
        bin_idx = min(int((score - lo) / width), n_bins - 1)
        bins[bin_idx].append(1 if outcome else 0)

    curve = []
    for bin_idx in sorted(bins):
        bin_scores = bins[bin_idx]
        bin_lo = lo + bin_idx * width
        bin_hi = bin_lo + width
        curve.append(
            {
                "bin_range": (bin_lo, bin_hi),
                "mean_observed_rate": sum(bin_scores) / len(bin_scores),
                "n": len(bin_scores),
            }
        )
    return curve
