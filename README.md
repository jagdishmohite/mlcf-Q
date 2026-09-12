# MLCF-Q — Multi-Layer Coherence Forensics for QUIC

Reference implementation of **MLCF-Q**, a passive attribution framework that
scores QUIC/HTTP-3 sessions by how coherently their transport, handshake,
application, and behavioral layers align with known client-stack profiles.
Residual cross-layer inconsistency is used as forensic evidence of mimicry.

This repo implements the blueprint described in the accompanying paper
*"Beyond JA4-Q: Multi-Layer Coherence Forensics for QUIC and HTTP/3 Traffic
Attribution"* (Section 6, Reference Implementation, and Section 7, Evaluation
Protocol). It is a **framework/tooling release**, not a claim of measured
detection performance — see [Status](#status) below.

## Why

JA4-Q summarizes the QUIC handshake but by design omits QUIC transport
parameters and other connection artifacts, and — like any single-layer
fingerprint — can be replayed by impersonation toolchains. MLCF-Q instead
scores **coherence across layers**:

- `x_Q` — QUIC transport parameters (idle timeout, initial_max_data,
  stream limits, active_connection_id_limit, migration flags, ...)
- `x_H` — embedded TLS ClientHello (cipher/extension/group structure, ALPN,
  post-quantum key-share presence)
- `x_A` — HTTP/3 SETTINGS, QPACK dynamic-table behavior, stream
  concurrency, pseudo-header ordering
- `x_B` — behavioral/session timing (inter-request cadence, connection
  reuse, 0-RTT usage, burstiness)

A genuine browser stack produces all four layers from one integrated
codebase, so they co-vary in stable, implementation-specific ways. MLCF-Q
scores sessions with

```
S(k | x) = Σ_L wL·log p_k(x_L)  −  λ Σ_(i,j)∈E Δ_k(x_i, x_j)
```

and flags large negative coherence terms as likely mimicry (Section 5.2),
plus a population-level cohort-dispersion test over transport parameters
within a shared handshake fingerprint (Section 5.3).

## Project layout

```
mlcf-q/
├── src/mlcfq/
│   ├── features/        # per-layer feature extraction (x_Q, x_H, x_A, x_B)
│   ├── extraction/       # tshark/pyshark-based pcap → structured session records
│   ├── scoring/           # coherence scoring: rule-based (R) and learned (L)
│   ├── cohort/             # cohort-level transport-parameter dispersion test (§5.3)
│   ├── evaluation/          # evaluation study harness (§7) — see docs/EVALUATION_GUIDE.md
│   ├── pipeline.py         # capture → features → score, end to end
│   └── cli.py              # `mlcfq` command-line entry point
├── tests/                  # unit tests (synthetic sessions, no pcap/tshark needed)
├── examples/                # sample profile/config + synthetic demo session sets
├── docs/ARCHITECTURE.md     # design notes, threat-model mapping, extension points
├── docs/EVALUATION_GUIDE.md # step-by-step: turning §7's protocol into real results
└── .github/workflows/ci.yml
```

## Install

Requires Python 3.10+. The core pipeline (feature extraction from
pre-parsed records, scoring, cohort test, CLI) depends only on `click`.
Live pcap decoding needs the optional `pyshark` extra and
[`tshark`](https://tshark.dev/) on `PATH`; the learned scorer
(Instantiation L) needs the optional `numpy`/`scikit-learn` extras; running
a real evaluation study (capturing traffic) needs the optional `playwright`
and `aioquic` extras.

```bash
git clone <this-repo>
cd mlcf-q
pip install -e ".[dev]"       # everything, for development
# or, minimal:
pip install -e .              # core pipeline only
pip install -e ".[pcap]"      # + live pcap extraction
pip install -e ".[learned]"   # + Instantiation L
pip install -e ".[capture]"   # + real browser/library traffic capture for evaluation
```

## Quick start (no pcap required)

Score a synthetic session against bundled example client-stack profiles:

```bash
mlcfq score --session examples/sample_session.json --profiles examples/profiles.json
```

Run the cohort-dispersion test over a batch of sessions sharing a handshake
fingerprint:

```bash
mlcfq cohort --sessions examples/sample_cohort.json
```

With a per-fingerprint benign baseline, cohorts are also flagged when their
dispersion exceeds it by `--threshold` (default 1.5x):

```bash
mlcfq cohort --sessions examples/sample_cohort.json --baseline examples/baseline.json
```

## Quick start (from a live/offline pcap)

```bash
mlcfq extract --pcap capture.pcapng --keylog secrets.log --out sessions.json
mlcfq score --session sessions.json --profiles examples/profiles.json
```

`--keylog` should point at a `SSLKEYLOGFILE` captured from a controlled
generating client (Section 6); without it, HTTP/3 frame-level features
(`x_A`) are extracted best-effort from what's visible and may be sparse.

## Quick start (evaluation study, §7)

Run the RQ1/RQ2 study over a labeled dataset (see
[`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) for how to build one
from real captured traffic):

```bash
mlcfq study --dataset examples/eval_dataset_demo.json --profiles examples/profiles_with_ja4q.json
```

The bundled `eval_dataset_demo.json` is a 16-session **synthetic** dataset
for checking the pipeline runs end to end — it is not a research result;
see the guide for turning this into a real measurement study.

## Status

This is a **framework and protocol implementation**, matching the paper's
scope: it implements Instantiation R (rule-based coherence, auditable, no
training data required) fully, and Instantiation L (learned, Section 5.2) as
a scikit-learn-based extension point that requires labeled QUIC traffic you
supply. No detection-rate numbers are claimed here — Section 7 of the paper
specifies the evaluation protocol (datasets, baselines, ablations, metrics)
for a future measurement study; `tests/` and `examples/` exist to validate
the *mechanics* of the pipeline, not its real-world efficacy. The
`mlcfq.evaluation` package (see `docs/EVALUATION_GUIDE.md`) implements the
tooling side of that future study — dataset schema, adversarial partition
construction, baselines, ablations, and metrics are fully implemented and
tested; only the actual traffic capture is left to be run on a real
machine.

## Threat-model scope (paper §4)

- **A1 (single-layer spoofing)** — targeted directly; this is the primary
  case the rule-based scorer is built to catch.
- **A2 (partial multi-layer spoofing)** — detection degrades gracefully as
  more layers are aligned by the adversary; still flagged while any modeled
  coupling is violated.
- **A3 (full-stack impersonation)** — not defeated at the single-session
  level by design. The cohort-dispersion test (`mlcfq/cohort/`) is the
  mechanism most likely to retain power here, since it operates over
  populations rather than single sessions.

## Ethics

This tool is for passive software-stack attribution and spoof/bot detection
on traffic you are authorized to monitor (enterprise sensor, ISP vantage
point, CDN edge) — not for identifying individuals. Apply data minimization,
short retention, and access control to any captured artifacts (paper §9).

## License

MIT — see [LICENSE](LICENSE).
