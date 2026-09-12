from mlcfq.evaluation.ablations import (
    all_ablations,
    full_model_config,
    remove_behavioral_features,
    remove_http3_features,
    remove_pq_term,
    remove_transport_layer,
    run_ablation,
)
from mlcfq.model import ClientProfile, Session

PROFILE = ClientProfile(
    name="Chrome-QUIC",
    expected={
        "transport": {"idle_timeout_ms": [20000, 30000]},
        "handshake": {"pq_key_share_expected": True},
        "http3": {"requires_settings": ["QPACK_MAX_TABLE_CAPACITY"]},
        "behavioral": {"min_burstiness": 0.15},
    },
)


def _violating_session() -> Session:
    """A session that violates every single constraint in PROFILE."""
    return Session.from_dict(
        {
            "session_id": "s1",
            "transport": {"idle_timeout_ms": 60000},  # outside [20000,30000]
            "handshake": {"pq_key_share_offered": False},  # violates expected True
            # h3_negotiated=True makes this unambiguously "HTTP/3 was
            # confirmed negotiated but the required setting is genuinely
            # absent" (a real violation), as opposed to h3_negotiated=None
            # ("no telemetry captured at all" -- correctly not penalized,
            # see rule_based._check_http3).
            "http3": {"h3_negotiated": True, "settings": {}},  # missing required setting
            "behavioral": {"burstiness": 0.001},  # below min_burstiness
        }
    )


def test_full_model_penalizes_every_violated_layer():
    result = run_ablation(_violating_session(), full_model_config([PROFILE]))[0]
    layers_with_penalty = {t.layer for t in result.layer_terms if t.log_likelihood < 0}
    assert layers_with_penalty == {"transport", "handshake", "http3", "behavioral"}


def test_remove_transport_layer_zeroes_that_layer_only():
    result = run_ablation(_violating_session(), remove_transport_layer([PROFILE]))[0]
    transport_term = next(t for t in result.layer_terms if t.layer == "transport")
    assert transport_term.weight == 0.0
    # other layers still active
    handshake_term = next(t for t in result.layer_terms if t.layer == "handshake")
    assert handshake_term.weight == 1.0


def test_remove_http3_features_zeroes_that_layer_only():
    result = run_ablation(_violating_session(), remove_http3_features([PROFILE]))[0]
    http3_term = next(t for t in result.layer_terms if t.layer == "http3")
    assert http3_term.weight == 0.0


def test_remove_behavioral_features_zeroes_that_layer_only():
    result = run_ablation(_violating_session(), remove_behavioral_features([PROFILE]))[0]
    behavioral_term = next(t for t in result.layer_terms if t.layer == "behavioral")
    assert behavioral_term.weight == 0.0


def test_remove_pq_term_strips_constraint_without_touching_layer_weight():
    result = run_ablation(_violating_session(), remove_pq_term([PROFILE]))[0]
    handshake_term = next(t for t in result.layer_terms if t.layer == "handshake")
    # handshake layer still weighted normally...
    assert handshake_term.weight == 1.0
    # ...but the PQ violation should no longer be penalized, since the
    # constraint itself was removed from the profile.
    assert handshake_term.log_likelihood == 0.0


def test_remove_pq_term_does_not_mutate_original_profiles():
    original_expected_keys = set(PROFILE.expected["handshake"].keys())
    remove_pq_term([PROFILE])
    assert set(PROFILE.expected["handshake"].keys()) == original_expected_keys


def test_all_ablations_includes_full_model_and_four_removals():
    configs = all_ablations([PROFILE])
    names = {c.name for c in configs}
    assert names == {
        "full_model",
        "remove_transport",
        "remove_http3",
        "remove_behavioral",
        "remove_pq_term",
    }
