import json

from mlcfq.model import ClientProfile, Session
from mlcfq.pipeline import load_profiles, load_sessions, score_sessions


def test_session_from_dict_to_dict_round_trip():
    raw = {
        "session_id": "s1",
        "transport": {"idle_timeout_ms": 20000},
        "handshake": {"alpn": ["h3"], "pq_key_share_offered": True},
        "http3": {"settings": {"QPACK_MAX_TABLE_CAPACITY": 4096}},
        "behavioral": {"burstiness": 0.2},
    }
    session = Session.from_dict(raw)
    assert session.session_id == "s1"
    assert session.transport.idle_timeout_ms == 20000
    assert session.handshake.alpn == ["h3"]

    back = session.to_dict()
    assert back["session_id"] == "s1"
    assert back["transport"]["idle_timeout_ms"] == 20000
    # round-trips through JSON cleanly too
    json.dumps(back)


def test_session_from_dict_ignores_unknown_keys():
    raw = {"session_id": "s1", "transport": {"idle_timeout_ms": 1000, "bogus_field": "x"}}
    session = Session.from_dict(raw)
    assert session.transport.idle_timeout_ms == 1000
    assert not hasattr(session.transport, "bogus_field")


def test_client_profile_round_trip():
    profile = ClientProfile.from_dict(
        {"name": "p", "expected": {"transport": {"idle_timeout_ms": [1, 2]}}}
    )
    assert profile.name == "p"
    assert profile.to_dict()["expected"]["transport"]["idle_timeout_ms"] == [1, 2]


def test_load_sessions_from_example_fixture():
    sessions = load_sessions("examples/sample_session.json")
    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-001-suspected-spoofer"


def test_load_profiles_from_example_fixture():
    profiles = load_profiles("examples/profiles.json")
    names = {p.name for p in profiles}
    assert "Chrome-124-Windows-QUIC" in names
    assert "quic-go-library-client" in names


def test_score_sessions_end_to_end_flags_the_spoofer():
    sessions = load_sessions("examples/sample_session.json")
    profiles = load_profiles("examples/profiles.json")
    results = score_sessions(sessions, profiles)
    ranked = results["sess-001-suspected-spoofer"]
    # Top-1 should not be the browser profile the handshake alone mimics.
    assert ranked[0].profile_name != "Chrome-124-Windows-QUIC"
