import json

from mlcfq.evaluation.dataset import (
    LabeledSession,
    NetworkCondition,
    load_labeled_sessions,
    save_labeled_sessions,
    summarize,
)
from mlcfq.model import Session


def _make_session(session_id: str, idle_timeout_ms: int = 20000) -> Session:
    return Session.from_dict({"session_id": session_id, "transport": {"idle_timeout_ms": idle_timeout_ms}})


def test_labeled_session_round_trip(tmp_path):
    labeled = LabeledSession(
        session=_make_session("s1"),
        ground_truth_class="Chrome-QUIC",
        tier="genuine",
        network_condition=NetworkCondition(rtt_ms=50, label="medium-rtt"),
    )
    back = LabeledSession.from_dict(labeled.to_dict())
    assert back.ground_truth_class == "Chrome-QUIC"
    assert back.tier == "genuine"
    assert back.network_condition.rtt_ms == 50
    assert back.session.transport.idle_timeout_ms == 20000
    # round-trips through JSON cleanly
    json.dumps(labeled.to_dict())


def test_labeled_session_defaults_to_genuine_tier():
    labeled = LabeledSession(session=_make_session("s1"), ground_truth_class="X")
    assert labeled.tier == "genuine"
    assert labeled.spoofed_as is None


def test_save_and_load_labeled_sessions(tmp_path):
    sessions = [
        LabeledSession(session=_make_session("s1"), ground_truth_class="A", tier="genuine"),
        LabeledSession(session=_make_session("s2"), ground_truth_class="B", tier="A1", spoofed_as="A"),
    ]
    path = tmp_path / "dataset.json"
    save_labeled_sessions(sessions, path)
    loaded = load_labeled_sessions(path)
    assert len(loaded) == 2
    assert loaded[0].ground_truth_class == "A"
    assert loaded[1].tier == "A1"
    assert loaded[1].spoofed_as == "A"


def test_summarize_counts_and_flags_missing_tiers():
    sessions = [
        LabeledSession(session=_make_session("s1"), ground_truth_class="A", tier="genuine"),
        LabeledSession(session=_make_session("s2"), ground_truth_class="A", tier="genuine"),
        LabeledSession(session=_make_session("s3"), ground_truth_class="B", tier="A1"),
    ]
    summary = summarize(sessions)
    assert summary["total"] == 3
    assert summary["by_tier"]["genuine"] == 2
    assert summary["by_tier"]["A1"] == 1
    assert summary["by_class"]["A"] == 2
    assert "A2" in summary["missing_tiers"]
    assert "A3" in summary["missing_tiers"]
    assert "genuine" not in summary["missing_tiers"]
