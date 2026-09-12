"""Evaluation harness for the MLCF-Q study protocol (paper §7).

This package implements the *mechanics* of the evaluation protocol —
labeled-dataset schema, adversarial-partition construction, baselines,
ablations, metrics, and network-condition variation — so that running the
actual study is "collect real traffic, then run these scripts" rather than
"design and implement an evaluation pipeline from scratch."

What this package can and cannot do for you:

- `metrics.py`, `ablations.py`, `baselines.py`, `adversarial.py` are fully
  deterministic and tested against synthetic data in this repo's test
  suite. They work correctly regardless of where your labeled sessions
  came from.
- `capture_harness.py` and `network_conditions.py` orchestrate *real*
  browser automation, packet capture, and `tc netem` network-condition
  injection. These require a real environment (installed browsers,
  `tshark`, root/sudo for `tc`) and cannot be exercised or verified inside
  a sandboxed CI environment — you run these yourself, on your own
  machine, per `docs/EVALUATION_GUIDE.md`.

See `docs/EVALUATION_GUIDE.md` for the end-to-end walkthrough mapping each
module back to the paper's RQ1/RQ2/RQ3 and dataset/baselines/ablations/
metrics description.
"""
