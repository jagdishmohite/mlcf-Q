# diagnose_a2.py
#
# Compares A1 vs A2 sessions directly: their http3 layer data, and their
# actual coherence scores, to find out why A2 shows below-chance detection
# while A1 doesn't -- since both splice transport identically, whatever's
# different has to come from HTTP/3 (the layer A2 additionally splices).

from collections import Counter

from mlcfq.evaluation.dataset import load_labeled_sessions
from mlcfq.pipeline import load_profiles
from mlcfq.scoring.coherence import rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer

sessions = load_labeled_sessions("examples/real_dataset.json")
profiles = load_profiles("examples/real_profiles.json")
scorer = RuleBasedScorer()

genuine = [s for s in sessions if s.tier == "genuine"]
a1 = [s for s in sessions if s.tier == "A1"]
a2 = [s for s in sessions if s.tier == "A2"]

print("=== HTTP/3 data comparison ===")
for label, group in [("genuine (Chrome)", [s for s in genuine if s.ground_truth_class == "Chrome-QUIC"]),
                       ("genuine (aioquic)", [s for s in genuine if s.ground_truth_class == "aioquic-library-client"]),
                       ("A1", a1), ("A2", a2)]:
    h3_negotiated_counts = Counter(s.session.http3.h3_negotiated for s in group)
    settings_populated = sum(1 for s in group if s.session.http3.settings)
    print(f"  {label} (n={len(group)}): h3_negotiated={dict(h3_negotiated_counts)}, "
          f"{settings_populated} of {len(group)} have non-empty settings")

print("\n=== Score comparison: genuine vs A1 vs A2 (best/Top-1 score each) ===")
for label, group in [("genuine", genuine), ("A1", a1), ("A2", a2)]:
    scores = []
    for s in group:
        ranked = rank_profiles(s.session, profiles, scorer)
        scores.append(ranked[0].score)
    avg_score = sum(scores) / len(scores) if scores else float("nan")
    print(f"  {label}: n={len(scores)}, mean best-score={avg_score:.3f}, "
          f"min={min(scores):.2f}, max={max(scores):.2f}")

print("\n=== Sample A2 session full breakdown (first 3) ===")
for s in a2[:3]:
    ranked = rank_profiles(s.session, profiles, scorer)
    print(f"\n  {s.session.session_id} (ground truth={s.ground_truth_class}):")
    for r in ranked:
        print(f"    {r.profile_name}: score={r.score:.2f}")
        for t in r.layer_terms:
            print(f"      {t.layer}: {t.log_likelihood:.2f}")
        for m in r.mismatch_terms:
            if m.penalty > 0:
                print(f"      MISMATCH {m.layer_i}<->{m.layer_j}: {m.penalty:.2f}")
