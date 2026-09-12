# inspect_dataset.py
#
# Prints the actual observed transport and behavioral values per genuine
# class in your real dataset, so profile constraints can be grounded in
# what was really captured rather than guessed.

from mlcfq.evaluation.dataset import load_labeled_sessions

sessions = load_labeled_sessions("examples/real_dataset.json")
genuine = [s for s in sessions if s.tier == "genuine"]

by_class: dict[str, list] = {}
for s in genuine:
    by_class.setdefault(s.ground_truth_class, []).append(s.session)

for class_name, class_sessions in by_class.items():
    print(f"\n=== {class_name} (n={len(class_sessions)}) ===")
    for s in class_sessions:
        t = s.transport
        b = s.behavioral
        print(
            f"  {s.session_id}: "
            f"idle_timeout_ms={t.idle_timeout_ms}, "
            f"initial_max_data={t.initial_max_data}, "
            f"initial_max_streams_bidi={t.initial_max_streams_bidi}, "
            f"active_connection_id_limit={t.active_connection_id_limit}, "
            f"disable_active_migration={t.disable_active_migration}, "
            f"quic_version={t.quic_version}, "
            f"burstiness={b.burstiness}, "
            f"cadence_n={len(b.inter_request_cadence_ms)}"
        )
        