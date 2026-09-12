# MLCF-Q Architecture Notes

This document maps the codebase onto the paper's sections and records the
implementation decisions/extension points that the paper leaves open.

## Data flow

```
   pcap/pcapng                     hand-authored JSON
  (+ optional keylog)              (tests/, examples/)
        │                                  │
        ▼                                  │
  mlcfq.extraction                         │
  (tshark/pyshark, §6)                     │
        │                                  │
        └──────────────┬───────────────────┘
                        ▼
              mlcfq.model.Session
        (x_Q, x_H, x_A, x_B — §5.1)
                        │
          ┌─────────────┼─────────────────┐
          ▼                               ▼
   mlcfq.scoring                  mlcfq.cohort
   (per-session coherence,        (population-level transport-
    §5.2 — Instantiation R/L)      parameter dispersion, §5.3)
          │                               │
          ▼                               ▼
   ranked ClientProfile              flagged handshake-key
   matches + mismatch terms          cohorts
```

## Paper section → code

| Paper section | Module |
|---|---|
| §5.1 feature spaces (x_Q, x_H, x_A, x_B) | `mlcfq/features/*.py`, `mlcfq/model.py` |
| §5.2 coherence score S(k\|x), dependency graph E | `mlcfq/scoring/coherence.py` |
| §5.2 Instantiation R (rule-based Δ_k) | `mlcfq/scoring/rule_based.py` |
| §5.2 Instantiation L (learned Δ_k) | `mlcfq/scoring/learned.py` |
| §5.3 cohort dispersion test | `mlcfq/cohort/dispersion.py` |
| §6 reference implementation (tshark/Zeek, keylog) | `mlcfq/extraction/pcap_extractor.py` |
| §7 evaluation protocol | not implemented here — this repo is the *tool*, §7 describes the *study* that would be run with it (datasets, baselines, ablations, metrics). `tests/` validates pipeline mechanics, not detection efficacy. |
| §4 threat tiers A1/A2/A3 | not a separate module; A1/A2 are the cases the scorer is built to catch (see `tests/test_rule_based.py`'s spoofer test), A3 is the case the cohort test targets (see `tests/test_cohort.py`) |

## Design decisions and why

**No required schema-validation dependency.** `mlcfq.model` uses stdlib
`dataclasses` rather than a validation library. The core pipeline (feature
building, scoring, cohort test, CLI) therefore installs with a single
lightweight dependency (`click`). `pyshark` (pcap decoding) and
`numpy`/`scikit-learn` (learned scorer) are optional extras, imported
lazily so importing `mlcfq` never requires them.

**Rule-based scorer ships working; learned scorer ships as a scaffold.**
Instantiation R needs no training data and is fully implemented and tested.
Instantiation L needs the labeled corpus described in §7, which this repo
doesn't include (that's the measurement study, not the tool). `learned.py`
implements the `fit`/`score` interface against `GaussianMixture` densities
per (class, layer) and per dependency-graph edge so a group with labeled
captures can plug data in directly; swap in a different density estimator
or a learned classifier for Δ_k without touching `coherence.py`.

**Dependency graph E is a flat, editable tuple of layer pairs.** The paper
notes E should be "seeded from protocol-stack knowledge and refined by
measured mutual information between layers on benign data" (§5.2). This
repo ships the seed graph (`DEFAULT_DEPENDENCY_GRAPH` in `coherence.py`);
refining it from measured MI is part of the §7 study, not the tool.

**Rule-based predicates are intentionally small and named.** Each
cross-layer predicate in `rule_based.py` documents which specific coupling
from the paper it encodes (e.g. "ALPN advertises h3 ⇒ QUIC transport
actually negotiated"). Extending coverage means adding a new named
predicate function and registering it against a dependency-graph edge in
`_EDGE_PREDICATES` — no changes needed elsewhere.

**Transport-parameter cohort signature is bucketed, not exact.** §5.3
measures dispersion of "secondary descriptors" within a cohort; exact
values would treat benign micro-jitter (e.g. `initial_max_data` off by a
few KB between two genuine Chrome sessions) as distinct signatures and
inflate dispersion. `transport_signature()` in `quic_transport.py` buckets
numeric fields before hashing into a signature tuple; bucket widths are a
tuning parameter for a real deployment, not something the paper pins down.

**Cohort baseline is optional and per-key.** §5.3 says dispersion should be
"normalized against a per-fingerprint benign baseline." This repo lets you
supply `baseline_by_key: dict[str, float]` (one float per handshake key,
representing that fingerprint's typical benign dispersion); without a
baseline, `compute_cohort_dispersion` still reports raw/normalized entropy
but leaves every cohort unflagged, since there's nothing to compare against.

## Extension points (left open on purpose)

- **`mlcfq/extraction/pcap_extractor.py::iter_sessions_streaming`** — batch
  extraction only; a live-interface deployment needs proper connection-close
  / idle-timeout eviction instead of end-of-capture flushing.
- **`mlcfq/scoring/learned.py`** — scaffold only; needs a labeled dataset
  per §7 to actually fit.
- **Dependency graph refinement** — `DEFAULT_DEPENDENCY_GRAPH` is the seed
  graph; §7's benign-traffic mutual-information analysis to prune/extend it
  is part of the measurement study, not shipped here.
- **Environment-aware modeling / proxy detection** (§9) — CDN/proxy QUIC
  termination can rewrite transport parameters and would need explicit
  detection to avoid over-penalizing benign rewriting; not implemented.
