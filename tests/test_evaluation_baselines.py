from mlcfq.evaluation.baselines import handshake_only_baseline, ja4q_only_baseline
from mlcfq.evaluation.dataset import LabeledSession
from mlcfq.model import ClientProfile, Session

CHROME_PROFILE = ClientProfile(
    name="Chrome-QUIC",
    expected={
        "handshake": {"ja4q_reference": "chrome-ja4q", "pq_key_share_expected": True},
    },
)
LIBRARY_PROFILE = ClientProfile(
    name="quic-go-client",
    expected={
        "handshake": {"ja4q_reference": "quicgo-ja4q", "pq_key_share_expected": False},
    },
)


def _labeled_session(ja4_q: str, pq_offered: bool) -> LabeledSession:
    session = Session.from_dict(
        {"session_id": "s1", "handshake": {"ja4_q": ja4_q, "pq_key_share_offered": pq_offered}}
    )
    return LabeledSession(session=session, ground_truth_class="whatever", tier="genuine")


def test_ja4q_only_baseline_exact_match_ranks_first():
    session = _labeled_session("chrome-ja4q", True)
    ranked = ja4q_only_baseline(session, [LIBRARY_PROFILE, CHROME_PROFILE])
    assert ranked[0] == "Chrome-QUIC"


def test_ja4q_only_baseline_no_match_returns_all_names_unordered_but_complete():
    session = _labeled_session("unknown-ja4q", True)
    ranked = ja4q_only_baseline(session, [LIBRARY_PROFILE, CHROME_PROFILE])
    assert set(ranked) == {"Chrome-QUIC", "quic-go-client"}


def test_ja4q_only_baseline_fooled_by_spoofed_handshake():
    """The point of this baseline existing: a spoofed JA4-Q fools it
    completely, since it has no other signal to fall back on.
    """
    # session claims Chrome's JA4-Q even though (hypothetically) it's really
    # a library client -- JA4-Q-only baseline has no way to know that.
    spoofed_session = _labeled_session("chrome-ja4q", True)
    ranked = ja4q_only_baseline(spoofed_session, [LIBRARY_PROFILE, CHROME_PROFILE])
    assert ranked[0] == "Chrome-QUIC"  # fooled, as expected


def test_handshake_only_baseline_ignores_other_layers():
    session = Session.from_dict(
        {
            "session_id": "s1",
            "handshake": {"pq_key_share_offered": True},
            # deliberately library-like transport/behavioral that would
            # betray a spoof at the full-model level -- handshake-only
            # baseline must not use this information.
            "transport": {"idle_timeout_ms": 30000},
            "behavioral": {"burstiness": 0.001},
        }
    )
    labeled = LabeledSession(session=session, ground_truth_class="x", tier="genuine")
    ranked = handshake_only_baseline(labeled, [CHROME_PROFILE, LIBRARY_PROFILE])
    # Chrome profile's handshake constraint (pq_key_share_expected=True) is
    # satisfied; library profile's (False) is violated -- Chrome should win
    # despite the library-like transport/behavioral layers, because this
    # baseline can't see those layers at all.
    assert ranked[0].profile_name == "Chrome-QUIC"
