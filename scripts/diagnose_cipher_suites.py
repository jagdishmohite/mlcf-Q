# diagnose_cipher_suites.py
#
# Checks real observed cipher_suites and supported_groups per class -- an
# untapped handshake signal that might close the A1 detection gap, since
# ALPN alone doesn't currently discriminate between Chrome and aioquic.

from collections import Counter

from mlcfq.evaluation.dataset import load_labeled_sessions

sessions = load_labeled_sessions("examples/real_dataset.json")
genuine = [s for s in sessions if s.tier == "genuine"]

by_class: dict[str, list] = {}
for s in genuine:
    by_class.setdefault(s.ground_truth_class, []).append(s.session)

for class_name, class_sessions in by_class.items():
    print(f"\n=== {class_name} (n={len(class_sessions)}) ===")

    cipher_sets = Counter(tuple(sorted(s.handshake.cipher_suites)) for s in class_sessions)
    print(f"  cipher_suites combinations seen: {cipher_sets.most_common(5)}")

    group_sets = Counter(tuple(sorted(s.handshake.supported_groups)) for s in class_sessions)
    print(f"  supported_groups combinations seen: {group_sets.most_common(5)}")
