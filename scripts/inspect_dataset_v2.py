# inspect_dataset_v2.py
#
# Computes actual observed distributions (not just single examples) across
# the full scaled-up dataset, per class: min/max/most-common for transport
# fields, and the True/False/None split for pq_key_share_offered.

from collections import Counter

from mlcfq.evaluation.dataset import load_labeled_sessions

sessions = load_labeled_sessions("examples/real_dataset.json")
genuine = [s for s in sessions if s.tier == "genuine"]

by_class: dict[str, list] = {}
for s in genuine:
    by_class.setdefault(s.ground_truth_class, []).append(s.session)

for class_name, class_sessions in by_class.items():
    print(f"\n=== {class_name} (n={len(class_sessions)}) ===")

    pq_counts = Counter(s.handshake.pq_key_share_offered for s in class_sessions)
    print(f"  pq_key_share_offered distribution: {dict(pq_counts)}")

    for field_name in (
        "idle_timeout_ms",
        "initial_max_data",
        "initial_max_streams_bidi",
        "active_connection_id_limit",
    ):
        values = [getattr(s.transport, field_name) for s in class_sessions]
        non_none = [v for v in values if v is not None]
        n_missing = len(values) - len(non_none)
        if non_none:
            print(
                f"  {field_name}: min={min(non_none)}, max={max(non_none)}, "
                f"most_common={Counter(non_none).most_common(3)}, missing={n_missing}"
            )
        else:
            print(f"  {field_name}: no data captured (all None)")

    burstiness_values = [s.behavioral.burstiness for s in class_sessions if s.behavioral.burstiness is not None]
    print(f"  burstiness: {len(burstiness_values)} of {len(class_sessions)} sessions have data")
