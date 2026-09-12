from mlcfq.model import ClientProfile, Session
from mlcfq.scoring.coherence import rank_profiles, score_session
from mlcfq.scoring.rule_based import RuleBasedScorer

CHROME_PROFILE = ClientProfile(
    name="Chrome-QUIC",
    expected={
        "transport": {
            "idle_timeout_ms": [15000, 25000],
            "initial_max_streams_bidi": [90, 110],
        },
        "handshake": {"alpn_contains": "h3", "pq_key_share_expected": True},
        "behavioral": {"min_burstiness": 0.15},
    },
)

LIBRARY_PROFILE = ClientProfile(
    name="quic-go-client",
    expected={
        "transport": {
            "idle_timeout_ms": [30000, 30000],
            "initial_max_streams_bidi": [100, 100],
        },
        "handshake": {"alpn_contains": "h3", "pq_key_share_expected": False},
        "behavioral": {"min_burstiness": 0.0005},
    },
)


def _genuine_chrome_session() -> Session:
    return Session.from_dict(
        {
            "session_id": "s1",
            "transport": {"idle_timeout_ms": 25000, "initial_max_streams_bidi": 100},
            "handshake": {"alpn": ["h3"], "pq_key_share_offered": True},
            "behavioral": {"burstiness": 0.3},
        }
    )


def _a1_spoofer_session() -> Session:
    """Chrome-looking handshake, quic-go-looking transport/timing (paper §4, A1)."""
    return Session.from_dict(
        {
            "session_id": "s2",
            "transport": {"idle_timeout_ms": 30000, "initial_max_streams_bidi": 100},
            "handshake": {"alpn": ["h3"], "pq_key_share_offered": True},
            "behavioral": {"burstiness": 0.001},
        }
    )


def test_genuine_chrome_session_scores_best_against_chrome_profile():
    session = _genuine_chrome_session()
    ranked = rank_profiles(session, [CHROME_PROFILE, LIBRARY_PROFILE], RuleBasedScorer())
    assert ranked[0].profile_name == "Chrome-QUIC"
    assert ranked[0].score > ranked[1].score


def test_a1_spoofer_does_not_score_best_against_the_spoofed_profile():
    """The core claim of the paper: a Chrome-handshake-only spoof should not
    win Top-1 against the genuine Chrome profile once transport/behavioral
    layers are taken into account, even though its handshake alone matches.
    """
    session = _a1_spoofer_session()
    chrome_result = score_session(session, CHROME_PROFILE, RuleBasedScorer())
    library_result = score_session(session, LIBRARY_PROFILE, RuleBasedScorer())

    # Handshake layer alone matches Chrome perfectly...
    handshake_term = next(t for t in chrome_result.layer_terms if t.layer == "handshake")
    assert handshake_term.log_likelihood == 0.0

    # ...but overall coherence (incl. transport + behavioral) favors the
    # profile whose *non-handshake* layers actually match the observed
    # traffic, exposing the single-layer spoof.
    assert library_result.score > chrome_result.score


def test_layer_log_likelihood_is_zero_with_no_declared_constraints():
    empty_profile = ClientProfile(name="no-constraints")
    session = _genuine_chrome_session()
    result = score_session(session, empty_profile, RuleBasedScorer())
    assert all(t.log_likelihood == 0.0 for t in result.layer_terms)
    assert result.score == 0.0


def test_missing_observation_is_not_treated_as_a_violation():
    profile = ClientProfile(
        name="p", expected={"transport": {"idle_timeout_ms": [1, 2]}}
    )
    session = Session.from_dict({"session_id": "s3"})  # no transport data at all
    result = score_session(session, profile, RuleBasedScorer())
    transport_term = next(t for t in result.layer_terms if t.layer == "transport")
    assert transport_term.log_likelihood == 0.0


def test_missing_http3_telemetry_is_not_treated_as_a_violation():
    """Regression test: a session with no HTTP/3 telemetry captured at all
    (h3_negotiated=None) must not be penalized against requires_settings
    constraints, the same way a missing numeric observation isn't
    penalized. Found via a real capture where genuine sessions missing
    HTTP/3 data (an extraction gap, not evidence of a different client)
    were scoring -3.0 against their own correct profile, occasionally
    causing real misattribution.
    """
    profile = ClientProfile(
        name="p",
        expected={"http3": {"requires_settings": ["QPACK_MAX_TABLE_CAPACITY"], "qpack_max_table_capacity": [4000, 5000]}},
    )
    session = Session.from_dict({"session_id": "s4", "http3": {}})  # no http3 telemetry: h3_negotiated defaults to None
    result = score_session(session, profile, RuleBasedScorer())
    http3_term = next(t for t in result.layer_terms if t.layer == "http3")
    assert http3_term.log_likelihood == 0.0


def test_http3_confirmed_negotiated_but_setting_missing_is_a_real_violation():
    """Contrast case for the above: once h3_negotiated=True is known (HTTP/3
    telemetry genuinely captured), a specific missing required setting IS
    still a real violation, not silently ignored.
    """
    profile = ClientProfile(
        name="p", expected={"http3": {"requires_settings": ["QPACK_MAX_TABLE_CAPACITY"]}}
    )
    session = Session.from_dict(
        {"session_id": "s5", "http3": {"h3_negotiated": True, "settings": {}}}
    )
    result = score_session(session, profile, RuleBasedScorer())
    http3_term = next(t for t in result.layer_terms if t.layer == "http3")
    assert http3_term.log_likelihood < 0.0


def test_alpn_h3_without_negotiation_triggers_mismatch_predicate():
    profile = ClientProfile(name="p", expected={})
    session = Session.from_dict(
        {
            "session_id": "s4",
            "handshake": {"alpn": ["h3"]},
            "http3": {"h3_negotiated": False},
        }
    )
    result = score_session(session, profile, RuleBasedScorer())
    mismatch = next(
        m for m in result.mismatch_terms if {m.layer_i, m.layer_j} == {"handshake", "http3"}
    )
    assert mismatch.penalty > 0


def test_migration_flag_contradiction_triggers_mismatch_predicate():
    profile = ClientProfile(name="p", expected={})
    session = Session.from_dict(
        {
            "session_id": "s5",
            "transport": {"disable_active_migration": True, "migration_observed": True},
        }
    )
    result = score_session(session, profile, RuleBasedScorer())
    mismatch = next(
        m for m in result.mismatch_terms if {m.layer_i, m.layer_j} == {"transport", "behavioral"}
    )
    assert mismatch.penalty > 0


def test_supported_groups_any_of_satisfied_when_group_present():
    profile = ClientProfile(
        name="p", expected={"handshake": {"supported_groups_any_of": ["0x001e"]}}
    )
    session = Session.from_dict(
        {"session_id": "s6", "handshake": {"supported_groups": ["0x0017", "0x001e"]}}
    )
    result = score_session(session, profile, RuleBasedScorer())
    handshake_term = next(t for t in result.layer_terms if t.layer == "handshake")
    assert handshake_term.log_likelihood == 0.0


def test_supported_groups_any_of_violated_when_group_absent():
    profile = ClientProfile(
        name="p", expected={"handshake": {"supported_groups_any_of": ["0x001e"]}}
    )
    session = Session.from_dict(
        {"session_id": "s7", "handshake": {"supported_groups": ["0x0017", "0x0018", "0x11ec"]}}
    )
    result = score_session(session, profile, RuleBasedScorer())
    handshake_term = next(t for t in result.layer_terms if t.layer == "handshake")
    assert handshake_term.log_likelihood < 0.0


def test_supported_groups_any_of_missing_observation_not_a_violation():
    profile = ClientProfile(
        name="p", expected={"handshake": {"supported_groups_any_of": ["0x001e"]}}
    )
    session = Session.from_dict({"session_id": "s8", "handshake": {}})  # no supported_groups captured
    result = score_session(session, profile, RuleBasedScorer())
    handshake_term = next(t for t in result.layer_terms if t.layer == "handshake")
    assert handshake_term.log_likelihood == 0.0
