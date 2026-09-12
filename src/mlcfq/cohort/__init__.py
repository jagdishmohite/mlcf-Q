"""Cohort-level incoherence detection for QUIC (paper §5.3).

Sessions sharing a primary handshake key (a JA4-Q-like descriptor) are
grouped, and dispersion of secondary features — chiefly QUIC transport
parameters — is measured within each group. High dispersion under a shared
handshake fingerprint is anomalous: a genuine primary fingerprint should map
to a small set of real stacks, so a population keyed to one target JA4-Q
that shows wide transport-parameter spread indicates mimicry, aggregation,
or mixed tooling.

This is the mechanism the paper identifies as most likely to retain power
against full-stack impersonation (A3, §4/§5.3): a single impersonator may be
internally coherent, but a population of them keyed to the same replayed
fingerprint need not be.
"""

from mlcfq.cohort.dispersion import CohortDispersionResult, compute_cohort_dispersion

__all__ = ["compute_cohort_dispersion", "CohortDispersionResult"]
