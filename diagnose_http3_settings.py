# diagnose_http3_settings.py
#
# Inspects real observed HTTP/3 named settings per class, so we can build
# grounded requires_settings/value-range constraints. Deliberately
# excludes GREASE/unrecognized settings (keys starting with "0x") --
# those are randomized per session by design (RFC 9114), so including
# them as a constraint would be actively wrong, not just unhelpful.

from collections import Counter

from mlcfq.evaluation.dataset import load_labeled_sessions

NAMED_KEYS = {
    "QPACK_MAX_TABLE_CAPACITY",
    "MAX_FIELD_SECTION_SIZE",
    "QPACK_BLOCKED_STREAMS",
    "ENABLE_CONNECT_PROTOCOL",
    "H3_DATAGRAM",
}

sessions = load_labeled_sessions("examples/real_dataset.json")
genuine = [s for s in sessions if s.tier == "genuine"]

by_class: dict[str, list] = {}
for s in genuine:
    by_class.setdefault(s.ground_truth_class, []).append(s.session)

for class_name, class_sessions in by_class.items():
    print(f"\n=== {class_name} (n={len(class_sessions)}) ===")

    # How many sessions have each named key present at all?
    presence_counts = Counter()
    value_lists: dict[str, list[int]] = {k: [] for k in NAMED_KEYS}
    h3_negotiated_none = 0

    for s in class_sessions:
        if s.http3.h3_negotiated is None:
            h3_negotiated_none += 1
        for key in NAMED_KEYS:
            if key in s.http3.settings:
                presence_counts[key] += 1
                value_lists[key].append(s.http3.settings[key])

    print(f"  sessions with h3_negotiated=None (missing http3 data entirely): {h3_negotiated_none} of {len(class_sessions)}")
    for key in NAMED_KEYS:
        present = presence_counts[key]
        values = value_lists[key]
        if values:
            print(f"  {key}: present in {present}/{len(class_sessions)} sessions, "
                  f"values={Counter(values).most_common(3)}, min={min(values)}, max={max(values)}")
        else:
            print(f"  {key}: never present")
