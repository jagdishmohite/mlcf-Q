"""x_B: session-level behavioral/timing feature layer (paper §5.1, §4).

This is one of the two residual mechanisms the paper identifies as retaining
power against full-stack impersonation (A3, §4): automation toolchains that
faithfully reproduce transport/handshake/HTTP-3 layers often still leave
timing artifacts (perfectly regular cadence, no 0-RTT usage, no connection
reuse) that diverge from organic browser behavior.
"""

from __future__ import annotations

import dataclasses
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from mlcfq.model import BehavioralFeatures


def build_behavioral_features(raw: Mapping[str, Any]) -> BehavioralFeatures:
    """Build x_B from raw per-request timestamps / flags.

    If `inter_request_cadence_ms` is supplied directly, it's used as-is; if
    only `request_timestamps_ms` (absolute) is given, cadence is derived by
    differencing, and `burstiness` (coefficient of variation) is computed
    automatically when not already provided.
    """
    data = dict(raw)

    cadence = data.get("inter_request_cadence_ms")
    if cadence is None:
        timestamps: Sequence[float] = data.get("request_timestamps_ms", []) or []
        ordered = sorted(timestamps)
        cadence = [b - a for a, b in zip(ordered, ordered[1:])]
    data["inter_request_cadence_ms"] = cadence

    if "burstiness" not in data or data["burstiness"] is None:
        data["burstiness"] = coefficient_of_variation(cadence)

    known = {f.name for f in dataclasses.fields(BehavioralFeatures)}
    filtered = {k: v for k, v in data.items() if k in known}
    return BehavioralFeatures(**filtered)


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    """Std/mean of a sample; None if undefined (fewer than 2 points or mean 0)."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = mean(values)
    if m == 0:
        return None
    return pstdev(values) / m
