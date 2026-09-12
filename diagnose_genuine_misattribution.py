# diagnose_genuine_misattribution.py
#
# Checks every genuine session's actual full_model prediction (not just
# A1/A2 sessions this time) to find out which sessions are being
# misattributed and why, following the Top-1 accuracy drop from 0.973 to
# 0.771 after adding the supported_groups_any_of constraint.

from mlcfq.evaluation.dataset import load_labeled_sessions
from mlcfq.pipeline import load_profiles
from mlcfq.scoring.coherence import rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer

sessions = load_labeled_sessions("examples/real_dataset.json")
profiles = load_profiles("examples/real_profiles.json")
scorer = RuleBasedScorer()

genuine = [s for s in sessions if s.tier == "genuine"]

wrong = []
for s in genuine:
    ranked = rank_profiles(s.session, profiles, scorer)
    predicted = ranked[0].profile_name
    if predicted != s.ground_truth_class:
        wrong.append((s, ranked))

print(f"=== {len(wrong)} of {len(genuine)} genuine sessions misattributed ===\n")

for s, ranked in wrong:
    print(f"{s.session.session_id} (ground truth={s.ground_truth_class}) -> predicted {ranked[0].profile_name}")
    for r in ranked:
        print(f"    {r.profile_name}: score={r.score:.2f}  layers={[(t.layer, round(t.log_likelihood,2)) for t in r.layer_terms]}")
    h = s.session.handshake
    print(f"    actual handshake: supported_groups={h.supported_groups}, pq_key_share_offered={h.pq_key_share_offered}, alpn={h.alpn}")
    print()
