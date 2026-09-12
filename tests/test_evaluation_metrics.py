from mlcfq.evaluation.metrics import (
    calibration_curve,
    detection_rate_at_fp_budget,
    macro_f1,
    per_tier_detection,
    pr_auc,
    roc_auc,
    top_k_accuracy,
)


def test_top_k_accuracy_basic():
    true_labels = ["A", "B", "C"]
    ranked = [["A", "B"], ["A", "B"], ["A", "B", "C"]]
    assert top_k_accuracy(true_labels, ranked, k=1) == 1 / 3  # only "A" case hits at k=1
    assert top_k_accuracy(true_labels, ranked, k=2) == 2 / 3  # A and B hit within top 2
    assert top_k_accuracy(true_labels, ranked, k=3) == 1.0  # all hit within top 3


def test_top_k_accuracy_empty_is_nan():
    result = top_k_accuracy([], [], k=1)
    assert result != result  # NaN != NaN


def test_macro_f1_perfect_predictions():
    true = ["A", "B", "A", "B"]
    pred = ["A", "B", "A", "B"]
    assert macro_f1(true, pred) == 1.0


def test_macro_f1_all_wrong():
    true = ["A", "A"]
    pred = ["B", "B"]
    assert macro_f1(true, pred) == 0.0


def test_macro_f1_rare_class_counts_equally():
    # class "rare" appears once; macro-F1 must not let it get drowned out
    # by the majority class the way micro-F1/accuracy would.
    true = ["common"] * 9 + ["rare"]
    pred = ["common"] * 9 + ["common"]  # rare class always missed
    score = macro_f1(true, pred)
    assert 0.4 < score < 0.6  # common class F1=1.0, rare class F1=0.0, macro avg=0.5


def test_roc_auc_perfect_separation():
    labels = [False, False, True, True]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == 1.0


def test_roc_auc_random_separation():
    labels = [False, True, False, True]
    scores = [0.5, 0.5, 0.5, 0.5]  # all tied -> chance-level AUC
    assert roc_auc(labels, scores) == 0.5


def test_roc_auc_inverse_separation():
    labels = [True, True, False, False]
    scores = [0.1, 0.2, 0.8, 0.9]  # positives score lowest -> worse than chance
    assert roc_auc(labels, scores) == 0.0


def test_roc_auc_undefined_without_both_classes():
    result = roc_auc([True, True], [0.1, 0.2])
    assert result != result  # NaN


def test_pr_auc_perfect_separation():
    labels = [False, False, True, True]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(labels, scores) == 1.0


def test_pr_auc_worst_case_ordering():
    # all negatives ranked above all positives
    labels = [True, True, False, False]
    scores = [0.1, 0.2, 0.8, 0.9]
    score = pr_auc(labels, scores)
    assert score < 0.6  # much worse than perfect (1.0), reflects poor ranking


def test_detection_rate_at_fp_budget_perfect_case():
    labels = [False, False, True, True]
    scores = [0.1, 0.2, 0.8, 0.9]
    # perfect separation: should catch all positives at essentially 0 FP budget
    assert detection_rate_at_fp_budget(labels, scores, fp_budget=0.0) == 1.0


def test_per_tier_detection_separates_tiers():
    genuine_scores = [0.1, 0.15]
    adversarial_scores_by_tier = {
        "A1": [0.9, 0.95],  # easily separated from genuine
        "A3": [0.1, 0.12],  # not separated at all -- worst case for A3
    }
    result = per_tier_detection(genuine_scores, adversarial_scores_by_tier)
    assert set(result.keys()) == {"A1", "A3"}
    assert result["A1"]["n"] == 4  # 2 genuine + 2 A1
    assert result["A3"]["n"] == 4  # 2 genuine + 2 A3
    assert result["A1"]["roc_auc"] == 1.0  # perfectly separated
    assert result["A3"]["roc_auc"] < result["A1"]["roc_auc"]  # much worse separation than A1


def test_calibration_curve_basic_shape():
    scores = [0.1, 0.2, 0.8, 0.9]
    outcomes = [False, False, True, True]
    curve = calibration_curve(scores, outcomes, n_bins=2)
    assert len(curve) <= 2
    total_n = sum(bin_["n"] for bin_ in curve)
    assert total_n == 4


def test_calibration_curve_empty_input():
    assert calibration_curve([], []) == []
