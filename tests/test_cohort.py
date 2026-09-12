from mlcfq.cohort.dispersion import compute_cohort_dispersion
from mlcfq.model import Session


def _session(session_id, ja4q, idle_timeout_ms, max_data, streams_bidi=100):
    return Session.from_dict(
        {
            "session_id": session_id,
            "handshake": {"ja4_q": ja4q, "alpn": ["h3"]},
            "transport": {
                "quic_version": "1",
                "idle_timeout_ms": idle_timeout_ms,
                "initial_max_data": max_data,
                "initial_max_streams_bidi": streams_bidi,
            },
        }
    )


def test_tight_cohort_has_zero_dispersion():
    sessions = [
        _session("a1", "shared-key", 20000, 10000000),
        _session("a2", "shared-key", 20050, 10050000),
        _session("a3", "shared-key", 19950, 9950000),
    ]
    results = compute_cohort_dispersion(sessions, min_cohort_size=3)
    assert len(results) == 1
    assert results[0].normalized_dispersion == 0.0
    assert results[0].unique_signature_count == 1


def test_mixed_cohort_has_high_dispersion():
    sessions = [
        _session("b1", "shared-key", 20000, 10000000, streams_bidi=100),
        _session("b2", "shared-key", 60000, 1048576, streams_bidi=16),
        _session("b3", "shared-key", 10000, 524288, streams_bidi=32),
        _session("b4", "shared-key", 30000, 2000000, streams_bidi=8),
    ]
    results = compute_cohort_dispersion(sessions, min_cohort_size=3)
    assert len(results) == 1
    assert results[0].normalized_dispersion > 0.8


def test_cohorts_below_min_size_are_skipped():
    sessions = [
        _session("c1", "rare-key", 20000, 10000000),
        _session("c2", "rare-key", 20000, 10000000),
    ]
    results = compute_cohort_dispersion(sessions, min_cohort_size=3)
    assert results == []


def test_multiple_handshake_keys_produce_separate_cohorts():
    sessions = [
        _session("d1", "key-a", 20000, 10000000),
        _session("d2", "key-a", 20000, 10000000),
        _session("d3", "key-a", 20000, 10000000),
        _session("e1", "key-b", 30000, 5000000),
        _session("e2", "key-b", 30000, 5000000),
        _session("e3", "key-b", 30000, 5000000),
    ]
    results = compute_cohort_dispersion(sessions, min_cohort_size=3)
    keys = {r.handshake_key for r in results}
    assert keys == {"key-a", "key-b"}


def test_flagging_uses_baseline_ratio():
    sessions = [
        _session("f1", "shared-key", 20000, 10000000, streams_bidi=100),
        _session("f2", "shared-key", 60000, 1048576, streams_bidi=16),
        _session("f3", "shared-key", 10000, 524288, streams_bidi=32),
    ]
    results = compute_cohort_dispersion(
        sessions, baseline_by_key={"shared-key": 0.1}, flag_threshold=1.5, min_cohort_size=3
    )
    assert results[0].flagged is True
    assert results[0].dispersion_ratio is not None


def test_no_baseline_means_not_flagged():
    sessions = [
        _session("g1", "shared-key", 20000, 10000000, streams_bidi=100),
        _session("g2", "shared-key", 60000, 1048576, streams_bidi=16),
        _session("g3", "shared-key", 10000, 524288, streams_bidi=32),
    ]
    results = compute_cohort_dispersion(sessions, min_cohort_size=3)
    assert results[0].flagged is False
    assert results[0].dispersion_ratio is None
