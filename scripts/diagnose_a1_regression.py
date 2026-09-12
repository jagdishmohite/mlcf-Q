# diagnose_a1_regression.py
#
# Checks whether genuine sessions missing HTTP/3 data (extraction gaps,
# not fakeness) are scoring as artificially anomalous now that real HTTP/3
# constraints exist -- which would explain A1's ROC-AUC drop without any
# change to A1 sessions themselves.

from mlcfq.evaluation.dataset import load_labeled_sessions
from mlcfq.pipeline import load_profiles
from mlcfq.scoring.coherence import rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer

sessions = load_labeled_sessions("examples/real_dataset.json")
profiles = load_profiles("examples/real_profiles.json")
scorer = RuleBasedScorer()

genuine = [s for s in sessions if s.tier == "genuine"]

print("=== Genuine session scores, split by whether HTTP/3 data was captured ===")
with_h3 = [s for s in genuine if s.session.http3.h3_negotiated is not None]
without_h3 = [s for s in genuine if s.session.http3.h3_negotiated is None]

for label, group in [("HAS http3 data", with_h3), ("MISSING http3 data", without_h3)]:
    scores = [rank_profiles(s.session, profiles, scorer)[0].score for s in group]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {label}: n={len(scores)}, mean best-score={avg:.3f}, scores={[round(s, 2) for s in scores]}")
    else:
        print(f"  {label}: n=0")

print("\n=== Full breakdown of genuine sessions missing http3 data ===")
for s in without_h3:
    ranked = rank_profiles(s.session, profiles, scorer)
    print(f"\n  {s.session.session_id} (ground truth={s.ground_truth_class}):")
    for r in ranked:
        print(f"    {r.profile_name}: score={r.score:.2f}  layers={[(t.layer, round(t.log_likelihood,2)) for t in r.layer_terms]}")
