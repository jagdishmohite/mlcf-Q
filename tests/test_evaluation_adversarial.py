import pytest

from mlcfq.evaluation.adversarial import build_a1_a2_partitions, make_a1_session, make_a2_session
from mlcfq.evaluation.dataset import LabeledSession
from mlcfq.model import Session


def _chrome_like_session(session_id: str) -> LabeledSession:
    session = Session.from_dict(
        {
            "session_id": session_id,
            "transport": {"idle_timeout_ms": 25000, "initial_max_streams_bidi": 100},
            "handshake": {"alpn": ["h3"], "pq_key_share_offered": True, "ja4_q": "chrome-ja4q"},
            "http3": {"h3_negotiated": True, "settings": {"QPACK_MAX_TABLE_CAPACITY": 4096}},
            "behavioral": {"burstiness": 0.3},
        }
    )
    return LabeledSession(session=session, ground_truth_class="Chrome-QUIC", tier="genuine")


def _library_like_session(session_id: str) -> LabeledSession:
    session = Session.from_dict(
        {
            "session_id": session_id,
            "transport": {"idle_timeout_ms": 30000, "initial_max_streams_bidi": 100},
            "handshake": {"alpn": ["h3"], "pq_key_share_offered": False, "ja4_q": "quicgo-ja4q"},
            "http3": {"h3_negotiated": True, "settings": {"QPACK_MAX_TABLE_CAPACITY": 4096}},
            "behavioral": {"burstiness": 0.001},
        }
    )
    return LabeledSession(session=session, ground_truth_class="quic-go-library-client", tier="genuine")


def test_make_a1_session_uses_target_handshake_and_native_everything_else():
    target = _chrome_like_session("chrome-1")
    native = _library_like_session("lib-1")

    a1 = make_a1_session(target, native, "a1-test")

    # Ground truth is the *actual* stack, not the spoofed target.
    assert a1.ground_truth_class == "quic-go-library-client"
    assert a1.tier == "A1"
    assert a1.spoofed_as == "Chrome-QUIC"

    # Handshake matches the target (spoofed) profile.
    assert a1.session.handshake.pq_key_share_offered is True
    assert a1.session.handshake.ja4_q == "chrome-ja4q"

    # Transport/HTTP3/behavioral remain native to the library client.
    assert a1.session.transport.idle_timeout_ms == 30000
    assert a1.session.behavioral.burstiness == 0.001


def test_make_a2_session_aligns_handshake_and_http3_only():
    target = _chrome_like_session("chrome-1")
    native = _library_like_session("lib-1")

    a2 = make_a2_session(target, native, "a2-test")

    assert a2.ground_truth_class == "quic-go-library-client"
    assert a2.tier == "A2"

    # Handshake AND http3 come from the target.
    assert a2.session.handshake.pq_key_share_offered is True

    # Transport and behavioral remain native.
    assert a2.session.transport.idle_timeout_ms == 30000
    assert a2.session.behavioral.burstiness == 0.001


def test_a1_a2_construction_rejects_non_genuine_sources():
    target = _chrome_like_session("chrome-1")
    target.tier = "A1"  # already adversarial -- shouldn't be usable as a splice source
    native = _library_like_session("lib-1")

    with pytest.raises(ValueError):
        make_a1_session(target, native, "bad")
    with pytest.raises(ValueError):
        make_a2_session(target, native, "bad")


def test_build_a1_a2_partitions_produces_labeled_and_counted_output():
    genuine = [
        _chrome_like_session("chrome-1"),
        _chrome_like_session("chrome-2"),
        _library_like_session("lib-1"),
        _library_like_session("lib-2"),
    ]
    a1_sessions, a2_sessions = build_a1_a2_partitions(
        genuine,
        target_classes=["Chrome-QUIC"],
        source_classes=["quic-go-library-client"],
        seed=42,
    )
    # 2 library sessions x 1 target class = 2 of each tier
    assert len(a1_sessions) == 2
    assert len(a2_sessions) == 2
    assert all(s.tier == "A1" for s in a1_sessions)
    assert all(s.tier == "A2" for s in a2_sessions)
    assert all(s.ground_truth_class == "quic-go-library-client" for s in a1_sessions + a2_sessions)


def test_build_a1_a2_partitions_raises_on_missing_class():
    genuine = [_chrome_like_session("chrome-1")]
    with pytest.raises(ValueError):
        build_a1_a2_partitions(
            genuine, target_classes=["Chrome-QUIC"], source_classes=["nonexistent-class"]
        )
