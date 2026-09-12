# diagnose_rq1.py
#
# Shows, for every A1/A2 session in your real dataset, exactly what
# full_model and handshake_only_baseline each predicted -- and why -- so
# we can see directly why the aggregate Top-1 scores tied instead of
# guessing from summary statistics alone.

from mlcfq.evaluation.baselines import handshake_only_baseline
from mlcfq.evaluation.dataset import load_labeled_sessions
from mlcfq.model import ClientProfile
from mlcfq.pipeline import load_profiles
from mlcfq.scoring.coherence import rank_profiles
from mlcfq.scoring.rule_based import RuleBasedScorer

sessions = load_labeled_sessions("examples/real_dataset.json")
profiles = load_profiles("examples/real_profiles.json")
scorer = RuleBasedScorer()

adversarial = [s for s in sessions if s.tier in ("A1", "A2")]

for s in adversarial:
    full_ranked = rank_profiles(s.session, profiles, scorer)
    hs_ranked = handshake_only_baseline(s, profiles)

    full_top1 = full_ranked[0].profile_name
    hs_top1 = hs_ranked[0].profile_name

    print(f"\n=== {s.session.session_id} (tier={s.tier}, ground truth={s.ground_truth_class}, spoofed_as={s.spoofed_as}) ===")
    print(f"  full_model predicts:       {full_top1}  {'CORRECT' if full_top1 == s.ground_truth_class else 'WRONG'}")
    print(f"    scores: {[(r.profile_name, round(r.score, 2)) for r in full_ranked]}")
    print(f"  handshake_only predicts:   {hs_top1}  {'CORRECT' if hs_top1 == s.ground_truth_class else 'WRONG'}")
    print(f"    scores: {[(r.profile_name, round(r.score, 2)) for r in hs_ranked]}")
    