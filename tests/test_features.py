from mlcfq.features.behavioral import build_behavioral_features, coefficient_of_variation
from mlcfq.features.handshake import build_handshake_features, handshake_key
from mlcfq.features.http3 import build_http3_features
from mlcfq.features.quic_transport import build_transport_features, transport_signature


def test_transport_features_maps_tshark_field_names():
    raw = {
        "quic.version": "1",
        "quic.tp.max_idle_timeout": 30000,
        "quic.tp.initial_max_data": 1048576,
        "quic.tp.disable_active_migration": "1",
    }
    features = build_transport_features(raw)
    assert features.quic_version == "1"
    assert features.idle_timeout_ms == 30000
    assert features.initial_max_data == 1048576
    assert features.disable_active_migration is True


def test_transport_features_accepts_normalized_keys():
    features = build_transport_features({"idle_timeout_ms": 20000, "quic_version": "1"})
    assert features.idle_timeout_ms == 20000


def test_transport_signature_buckets_similar_values_together():
    a = build_transport_features({"quic_version": "1", "idle_timeout_ms": 20000, "initial_max_data": 10000000})
    b = build_transport_features({"quic_version": "1", "idle_timeout_ms": 20100, "initial_max_data": 10200000})
    assert transport_signature(a) == transport_signature(b)


def test_transport_signature_distinguishes_different_families():
    browser = build_transport_features(
        {"quic_version": "1", "idle_timeout_ms": 20000, "initial_max_data": 10000000, "initial_max_streams_bidi": 100}
    )
    library = build_transport_features(
        {"quic_version": "1", "idle_timeout_ms": 60000, "initial_max_data": 1048576, "initial_max_streams_bidi": 16}
    )
    assert transport_signature(browser) != transport_signature(library)


def test_handshake_features_detects_pq_key_share():
    features = build_handshake_features(
        {"supported_groups": ["X25519", "X25519Kyber768Draft00"], "alpn": ["h3"]}
    )
    assert features.pq_key_share_offered is True
    assert features.pq_group_name == "X25519Kyber768Draft00"


def test_handshake_features_no_pq_group_present():
    features = build_handshake_features({"supported_groups": ["X25519", "secp256r1"]})
    assert features.pq_key_share_offered is False
    assert features.pq_group_name is None


def test_handshake_features_respects_explicit_pq_flag():
    features = build_handshake_features({"supported_groups": [], "pq_key_share_offered": True})
    assert features.pq_key_share_offered is True


def test_handshake_key_prefers_explicit_ja4q():
    features = build_handshake_features({"ja4_q": "q13d0310h3_abc"})
    assert handshake_key(features) == "q13d0310h3_abc"


def test_handshake_key_synthesizes_when_absent():
    features = build_handshake_features(
        {"cipher_suites": ["A", "B"], "extensions": ["x"], "alpn": ["h3"]}
    )
    key = handshake_key(features)
    assert key.startswith("synth:")
    assert "A,B" in key


def test_http3_features_normalizes_numeric_settings_ids():
    features = build_http3_features({"settings": {0x01: 4096, 0x07: 16}})
    assert features.settings["QPACK_MAX_TABLE_CAPACITY"] == 4096
    assert features.qpack_max_table_capacity == 4096
    assert features.qpack_blocked_streams == 16


def test_behavioral_features_derives_cadence_from_timestamps():
    features = build_behavioral_features({"request_timestamps_ms": [1000, 1200, 1450]})
    assert features.inter_request_cadence_ms == [200, 250]
    assert features.burstiness is not None


def test_coefficient_of_variation_handles_degenerate_input():
    assert coefficient_of_variation([]) is None
    assert coefficient_of_variation([5.0]) is None
    assert coefficient_of_variation([0.0, 0.0]) is None
    assert coefficient_of_variation([10.0, 10.0]) == 0.0
