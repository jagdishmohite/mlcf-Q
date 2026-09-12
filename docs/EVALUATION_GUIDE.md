# Running the MLCF-Q Evaluation Study (paper §7)

This guide walks through turning the paper's evaluation *protocol*
(Section 7) into actual measured results, using the `mlcfq.evaluation`
package. It's written for the specific goal of resubmitting to a technical
track that requires evaluated results, not just a framework.

**Read this first:** the honest scope split is that this package gives you
every piece of *tooling* the study needs — dataset schema, adversarial
partition construction, baselines, ablations, metrics, network-condition
injection — fully implemented and tested. What it cannot do is generate
real data for you: capturing actual QUIC/HTTP-3 traffic from real browsers
and library clients requires a real machine with real network access, real
installed browsers, and (for some steps) root/sudo. That part is on you;
everything downstream of "I have some labeled sessions" is done.

## 1. Install the extras this needs

```bash
pip install -e ".[dev]"          # includes playwright, aioquic for capture
playwright install chromium firefox
```

You'll also need `tshark` on PATH (same requirement as `mlcfq extract`),
and, if you want quic-go or ngtcp2 in your dataset, their example clients
built separately (they're not Python packages — consult each project's own
build instructions).

## 2. Collect genuine-tier traffic

Use `mlcfq.evaluation.capture_harness` to drive real clients and capture
their traffic:

```python
from mlcfq.evaluation.capture_harness import capture_browser_session
from mlcfq.extraction import extract_sessions_from_pcap
from mlcfq.evaluation.dataset import LabeledSession, save_labeled_sessions

pcap, keylog = capture_browser_session(
    browser="chromium",
    url="https://example-http3-site.com",
    out_pcap="captures/chrome-1.pcapng",
    keylog_path="captures/chrome-1.keylog",
)
sessions = extract_sessions_from_pcap(pcap, keylog_path=keylog)
labeled = [
    LabeledSession(session=s, ground_truth_class="Chrome-124-Windows-QUIC", tier="genuine")
    for s in sessions
]
```

Repeat across:
- **Browsers**: Chromium and Firefox at minimum (paper §7: "major browsers
  across OSes") — run this on each OS you can access.
- **Library stacks**: `capture_aioquic_session` (pure Python, no separate
  build) plus `capture_external_client_session` pointed at built quic-go /
  ngtcp2 example client binaries.
- **Repeat counts**: capture each client class multiple times (aim for
  enough repeats that `mlcfq.evaluation.dataset.summarize` shows a
  reasonable count per class — a handful of sessions per class is a bare
  minimum, more is better for the ROC/PR-AUC estimates in RQ2 to be
  meaningful).

Every real capture is ground-truth labeled by construction, since you
control which client produced it — matching paper §7's "because traffic is
self-generated, every session is ground-truth labeled."

## 3. Vary network conditions (for RQ3)

Wrap each capture with `mlcfq.evaluation.network_conditions`:

```python
from mlcfq.evaluation.network_conditions import apply_condition, clear_condition, STANDARD_CONDITIONS

for condition in STANDARD_CONDITIONS:
    apply_condition("eth0", condition)
    # ... run a capture_harness capture here, set
    #     labeled.network_condition = NetworkCondition(..., label=condition.label)
    clear_condition("eth0")
```

Requires root/sudo and Linux. For connection-migration events specifically,
`capture_harness.trigger_connection_migration` is a best-effort
approximation (bouncing the interface) — a real mobile Wi-Fi/cellular
handover on an actual device is more authentic if you have that rig
available.

## 4. Build the A1/A2 adversarial partitions

This is the one part that doesn't need any additional real-world
infrastructure beyond the genuine sessions you already captured —
`mlcfq.evaluation.adversarial` constructs these mechanically, exactly per
paper §4's definitions:

```python
from mlcfq.evaluation.adversarial import build_a1_a2_partitions

a1_sessions, a2_sessions = build_a1_a2_partitions(
    genuine_sessions=all_your_genuine_labeled_sessions,
    target_classes=["Chrome-124-Windows-QUIC", "Firefox-127-QUIC"],  # who's being spoofed
    source_classes=["quic-go-library-client", "aioquic-library-client"],  # who's doing the spoofing
)
```

## 5. Build the A3 partition

A3 ("a real browser engine driven by automation") is *not* a recombination
— it's `capture_harness.capture_browser_session` again, just labeled
differently:

```python
pcap, keylog = capture_browser_session("chromium", url, "captures/a3-1.pcapng", "captures/a3-1.keylog")
sessions = extract_sessions_from_pcap(pcap, keylog)
a3_labeled = [LabeledSession(session=s, ground_truth_class="Chrome-124-Windows-QUIC", tier="A3") for s in sessions]
```

If you also have access to a QUIC-capable impersonation toolchain (paper
§3.3 mentions these exist commercially, e.g. for scraping), captures from
that count as A3 too, labeled with the toolchain's *actual* underlying
identity as `ground_truth_class`.

## 6. Sanity-check the dataset before running the study

```bash
python3 -c "
from mlcfq.evaluation.dataset import load_labeled_sessions, summarize
sessions = load_labeled_sessions('my_dataset.json')
print(summarize(sessions))
"
```

Confirms `missing_tiers` is empty and gives you the per-class/per-tier
counts paper §7's dataset section asks you to be able to report.

## 7. Run the study

```bash
mlcfq study --dataset my_dataset.json --profiles my_profiles.json --out report.md
```

This runs:
- **RQ1** (attribution): full model vs. `ja4q_only_baseline` vs.
  `handshake_only_baseline`, reporting Top-1/Top-3/macro-F1.
- **RQ2** (mimicry resistance): full model and all four ablations
  (transport, HTTP/3, behavioral, PQ term removed in turn), reporting
  ROC-AUC/PR-AUC separately per adversarial tier — deliberately not pooled
  across tiers, since paper §7 specifically warns pooling "would let easy
  A1 detections mask A3 failure."

**RQ3** (robustness across network conditions) isn't a separate CLI flag —
filter your dataset down to one `network_condition.label` at a time and
run `mlcfq.evaluation.run_study.run_rq1`/`run_rq2` again per slice, then
compare accuracy/AUC across slices yourself:

```python
from mlcfq.evaluation.run_study import run_rq1, run_rq2

for label in {s.network_condition.label for s in dataset}:
    slice_sessions = [s for s in dataset if s.network_condition.label == label]
    print(label, run_rq1(slice_sessions, profiles))
```

## 8. Update the paper

Once you have real results:

- Replace §6.1's "Worked example on synthetic fixtures" framing — that
  section was explicitly labeled as a mechanics demonstration, not a
  result, and should either be removed or clearly separated from the new
  real results section.
- §7 (Evaluation Protocol) becomes an actual **Evaluation** section
  reporting what you found, not just what you plan to do — update the
  section title and rewrite it from future tense ("we specify...") to past
  tense ("we found...").
- The abstract and contributions list (§2) currently say "we report the
  protocol rather than claim measured efficacy" — this sentence needs to
  come out once you have measured efficacy to report.
- Add the `mlcfq study` output tables (or your own formatted version of
  them) as the new Tables 1/2, replacing the synthetic-fixture tables.

## What's genuinely untested here

Everything in `dataset.py`, `adversarial.py`, `baselines.py`,
`ablations.py`, `metrics.py`, and `run_study.py` has passing unit tests
against synthetic data (`tests/test_evaluation_*.py`) verifying the logic
is correct. `capture_harness.py` and `network_conditions.py` orchestrate
real browsers/tc/tshark and cannot be exercised in an automated test suite
— test these manually on your own machine (e.g. capture one short session,
open the resulting pcap in Wireshark, and confirm it looks like what you
expect) before running a full capture campaign.

## Example: the synthetic demo dataset

`examples/eval_dataset_demo.json` + `examples/profiles_with_ja4q.json` are
a tiny (16-session) synthetic dataset exercising the full pipeline —
genuine Chrome/quic-go sessions plus A1/A2 constructions — so you can see
`mlcfq study`'s output shape before you have real data:

```bash
mlcfq study --dataset examples/eval_dataset_demo.json --profiles examples/profiles_with_ja4q.json
```

**This demo is for checking the pipeline runs, not a research result** —
16 synthetic sessions tell you nothing about real-world detection rates.
Treat its numbers exactly the same way the paper's §6.1 worked example
was treated: illustrative of mechanics, not a claim about efficacy.
