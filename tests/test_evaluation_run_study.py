from mlcfq.evaluation.adversarial import make_a1_session
from mlcfq.evaluation.dataset import LabeledSession
from mlcfq.evaluation.run_study import format_results_markdown, run_rq1, run_rq2, run_study
from mlcfq.model import ClientProfile, Session

CHROME_PROFILE = ClientProfile(
    name="Chrome-QUIC",
    expected={
        "transport": {"idle_timeout_ms": [20000, 30000]},
        "handshake": {"ja4q_reference": "chrome-ja4q", "pq_key_share_expected": True},
        "behavioral": {"min_burstiness": 0.15},
    },
)
LIBRARY_PROFILE = ClientProfile(
    name="quic-go-client",
    expected={
        "transport": {"idle_timeout_ms": [30000, 30000]},
        "handshake": {"ja4q_reference": "quicgo-ja4q", "pq_key_share_expected": False},
        "behavioral": {"min_burstiness": 0.0005},
    },
)
PROFILES = [CHROME_PROFILE, LIBRARY_PROFILE]


def _genuine_chrome(session_id: str) -> LabeledSession:
    session = Session.from_dict(
        {
            "session_id": session_id,
            "transport": {"idle_timeout_ms": 25000},
            "handshake": {"ja4_q": "chrome-ja4q", "pq_key_share_offered": True},
            "behavioral": {"burstiness": 0.3},
        }
    )
    return LabeledSession(session=session, ground_truth_class="Chrome-QUIC", tier="genuine")


def _genuine_library(session_id: str) -> LabeledSession:
    session = Session.from_dict(
        {
            "session_id": session_id,
            "transport": {"idle_timeout_ms": 30000},
            "handshake": {"ja4_q": "quicgo-ja4q", "pq_key_share_offered": False},
            "behavioral": {"burstiness": 0.0005},
        }
    )
    return LabeledSession(session=session, ground_truth_class="quic-go-client", tier="genuine")


def _small_dataset() -> list[LabeledSession]:
    chrome_sessions = [_genuine_chrome(f"chrome-{i}") for i in range(3)]
    library_sessions = [_genuine_library(f"lib-{i}") for i in range(3)]
    a1_sessions = [
        make_a1_session(chrome_sessions[0], library_sessions[i], f"a1-{i}")
        for i in range(len(library_sessions))
    ]
    return chrome_sessions + library_sessions + a1_sessions


def test_run_rq1_full_model_beats_ja4q_baseline_on_spoofed_sessions():
    dataset = _small_dataset()
    results = run_rq1(dataset, PROFILES)
    by_method = {r.method: r for r in results}

    # The full model should score at least as well as the JA4Q-only
    # baseline overall, since JA4Q-only is fooled by every A1 session by
    # construction (it only looks at the handshake).
    assert by_method["full_model"].top1 >= by_method["ja4q_only_baseline"].top1


def test_run_rq2_produces_per_tier_results_for_every_ablation():
    dataset = _small_dataset()
    results = run_rq2(dataset, PROFILES)
    method_names = {r.method for r in results}
    assert "full_model" in method_names
    assert "remove_transport" in method_names

    full_model_result = next(r for r in results if r.method == "full_model")
    assert "A1" in full_model_result.per_tier
    assert full_model_result.per_tier["A1"]["n"] > 0


def test_run_study_end_to_end_and_markdown_report():
    dataset = _small_dataset()
    results = run_study(dataset, PROFILES)
    assert results.dataset_summary["total"] == len(dataset)
    assert len(results.rq1) == 4  # full model + weighted variant + 2 baselines
    assert len(results.rq2) == 6  # full model + weighted variant + 4 ablations

    report = format_results_markdown(results)
    assert "RQ1: Attribution accuracy" in report
    assert "RQ2: Mimicry resistance" in report
    assert "full_model" in report


def test_transport_weighting_resolves_a2_exact_tie():
    """Regression test for the exact bug found against real captured data
    (paper §7.3): an A2 session (handshake+http3 spliced from the target,
    transport native to the true source) can land on an exact score tie
    under uniform layer weights, broken arbitrarily. This reproduces that
    tie deterministically with minimal profiles, then confirms
    `full_model_transport_weighted_config` resolves it in the correct
    direction (favoring the true class) without needing real capture data.
    """
    from mlcfq.evaluation.adversarial import make_a2_session
    from mlcfq.evaluation.ablations import full_model_config, full_model_transport_weighted_config, run_ablation

    profile_a = ClientProfile(
        name="A-target", expected={"transport": {"idle_timeout_ms": [20000, 25000]}}
    )
    profile_b = ClientProfile(
        name="B-true-source",
        expected={
            "transport": {"idle_timeout_ms": [30000, 30000]},
            "http3": {"requires_settings": ["NEEDED_SETTING"]},
        },
    )
    profiles = [profile_a, profile_b]

    native_a = LabeledSession(
        session=Session.from_dict(
            {"session_id": "native-a", "transport": {"idle_timeout_ms": 22000}, "http3": {"h3_negotiated": True, "settings": {}}}
        ),
        ground_truth_class="A-target",
        tier="genuine",
    )
    native_b = LabeledSession(
        session=Session.from_dict(
            {
                "session_id": "native-b",
                "transport": {"idle_timeout_ms": 30000},
                "http3": {"h3_negotiated": True, "settings": {"NEEDED_SETTING": 1}},
            }
        ),
        ground_truth_class="B-true-source",
        tier="genuine",
    )

    a2 = make_a2_session(native_a, native_b, "a2-tie-repro")
    assert a2.ground_truth_class == "B-true-source"

    uniform_ranked = run_ablation(a2.session, full_model_config(profiles))
    assert uniform_ranked[0].score == uniform_ranked[1].score  # confirms the tie reproduces

    weighted_ranked = run_ablation(a2.session, full_model_transport_weighted_config(profiles))
    assert weighted_ranked[0].profile_name == "B-true-source"  # tie resolved, correct class wins
