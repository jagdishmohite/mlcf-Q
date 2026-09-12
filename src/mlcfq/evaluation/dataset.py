"""Labeled session records for the evaluation study (paper §7).

A `LabeledSession` wraps `mlcfq.model.Session` with the ground-truth
metadata the study protocol needs: which client-stack class actually
produced it, which threat tier it belongs to (genuine traffic, or an
adversarial A1/A2/A3 construction per paper §4), and which network
condition slice it was collected under (for RQ3).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mlcfq.model import Session

Tier = Literal["genuine", "A1", "A2", "A3"]


@dataclass
class NetworkCondition:
    """One network-condition slice for RQ3 (paper §7: "RTT, loss, NAT,
    migration events").
    """

    rtt_ms: float | None = None
    loss_pct: float | None = None
    nat: bool | None = None
    migration_observed: bool | None = None
    label: str | None = None  # human-readable slice name, e.g. "high-loss-mobile"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkCondition":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LabeledSession:
    """A session plus the ground truth a study needs to score itself.

    `ground_truth_class` is the real client-stack that produced the
    traffic (e.g. "Chrome-124-Windows-QUIC", "quic-go-library-client") --
    this is what Top-1/Top-k accuracy and macro-F1 (RQ1) are measured
    against, regardless of tier.

    `tier` distinguishes organic ("genuine") traffic from the three
    adversarial constructions in paper §4. `spoofed_as` records which
    profile an adversarial session's handshake was made to resemble, for
    tiers where that's meaningful (A1/A2); it's the profile a JA4-Q-only
    baseline would mistake the session for.
    """

    session: Session
    ground_truth_class: str
    tier: Tier = "genuine"
    spoofed_as: str | None = None
    network_condition: NetworkCondition = field(default_factory=NetworkCondition)
    source_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledSession":
        return cls(
            session=Session.from_dict(data["session"]),
            ground_truth_class=data["ground_truth_class"],
            tier=data.get("tier", "genuine"),
            spoofed_as=data.get("spoofed_as"),
            network_condition=NetworkCondition.from_dict(data.get("network_condition", {})),
            source_meta=data.get("source_meta", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "ground_truth_class": self.ground_truth_class,
            "tier": self.tier,
            "spoofed_as": self.spoofed_as,
            "network_condition": dataclasses.asdict(self.network_condition),
            "source_meta": self.source_meta,
        }


def load_labeled_sessions(path: str | Path) -> list[LabeledSession]:
    """Load a JSON array of labeled sessions (list of `LabeledSession.to_dict()`)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    return [LabeledSession.from_dict(item) for item in data]


def save_labeled_sessions(sessions: list[LabeledSession], path: str | Path) -> None:
    Path(path).write_text(json.dumps([s.to_dict() for s in sessions], indent=2))


def summarize(sessions: list[LabeledSession]) -> dict[str, Any]:
    """Quick dataset-composition sanity check: counts per class and tier.

    Useful before running the study to confirm the dataset actually
    populates all three adversarial tiers and a reasonable class spread,
    per paper §7's dataset requirement.
    """
    by_tier: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_class_and_tier: dict[str, dict[str, int]] = {}
    for s in sessions:
        by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
        by_class[s.ground_truth_class] = by_class.get(s.ground_truth_class, 0) + 1
        by_class_and_tier.setdefault(s.ground_truth_class, {}).setdefault(s.tier, 0)
        by_class_and_tier[s.ground_truth_class][s.tier] += 1
    return {
        "total": len(sessions),
        "by_tier": by_tier,
        "by_class": by_class,
        "by_class_and_tier": by_class_and_tier,
        "missing_tiers": [t for t in ("genuine", "A1", "A2", "A3") if by_tier.get(t, 0) == 0],
    }
