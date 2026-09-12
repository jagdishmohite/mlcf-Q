# diagnose_handshake.py
#
# Checks the actual pq_key_share_offered (and other handshake fields) for
# every genuine session, plus which specific source session each A1/A2
# splice actually used -- to find out why handshake_only_baseline is
# scoring the "spoofed Chrome" handshake as matching aioquic's profile
# instead of Chrome's.

from mlcfq.evaluation.dataset import load_labeled_sessions

sessions = load_labeled_sessions("examples/real_dataset.json")

print("=== Genuine sessions: actual handshake fields ===")
for s in sessions:
    if s.tier == "genuine":
        h = s.session.handshake
        print(
            f"  {s.session.session_id} ({s.ground_truth_class}): "
            f"pq_key_share_offered={h.pq_key_share_offered}, "
            f"alpn={h.alpn}, "
            f"ja4_q={h.ja4_q}"
        )

print("\n=== A1/A2 sessions: which source session was actually used, and resulting handshake ===")
for s in sessions:
    if s.tier in ("A1", "A2"):
        h = s.session.handshake
        print(
            f"  {s.session.session_id} (spoofed_as={s.spoofed_as}): "
            f"source_meta={s.source_meta}, "
            f"resulting pq_key_share_offered={h.pq_key_share_offered}"
        )
