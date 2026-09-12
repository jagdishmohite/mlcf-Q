# diagnose_http3_by_site.py
#
# Checks whether HTTP/3 named settings correlate with which SITE was
# visited (source_meta.captured_url) rather than which CLIENT made the
# request. If they do, that's strong evidence we're extracting
# server-side settings, not client-side -- which would mean this data
# isn't usable as a client-stack fingerprint the way the paper intends.

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

print("=== HTTP/3 named settings, grouped by (class, site) ===")
by_class_and_site: dict[tuple[str, str], list] = {}
for s in genuine:
    url = s.source_meta.get("captured_url", "unknown")
    by_class_and_site.setdefault((s.ground_truth_class, url), []).append(s.session)

for (class_name, url), class_sessions in sorted(by_class_and_site.items()):
    present_keys = set()
    for s in class_sessions:
        present_keys |= (set(s.http3.settings.keys()) & NAMED_KEYS)
    print(f"  {class_name} @ {url} (n={len(class_sessions)}): named settings seen = {sorted(present_keys)}")
