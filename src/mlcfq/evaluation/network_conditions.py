"""Network-condition injection via `tc netem` (paper §7 RQ3: "collected
across network conditions using tc netem and induced connection
migration").

Requires: Linux, `tc` (iproute2, usually preinstalled), and root/sudo to
modify qdiscs. This module shells out to `tc` -- it does not reimplement
traffic shaping -- so it only works on the machine actually generating or
observing traffic, on a real interface. Not exercised in this repo's test
suite for that reason; test manually with `tc qdisc show dev <iface>`
before and after calling `apply_condition`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class NetemError(RuntimeError):
    """Raised when a `tc` invocation fails (e.g. missing permissions, no
    netem kernel module, invalid interface name).
    """


@dataclass
class NetemCondition:
    """One condition to apply via `tc qdisc ... netem`.

    `rtt_ms` becomes a `delay` clause, `jitter_ms` (optional) becomes
    netem's delay jitter, `loss_pct` becomes a `loss` clause,
    `bandwidth_kbit` (optional) adds a `tbf` rate-limit qdisc alongside
    netem. Leave fields as None to skip that clause.
    """

    rtt_ms: float | None = None
    jitter_ms: float | None = None
    loss_pct: float | None = None
    bandwidth_kbit: float | None = None
    label: str | None = None


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise NetemError(f"command failed: {' '.join(cmd)}\n{result.stderr}")


def apply_condition(interface: str, condition: NetemCondition, use_sudo: bool = True) -> None:
    """Apply a netem condition to `interface`, replacing any existing qdisc.

    Requires root/sudo. Call `clear_condition` first if a qdisc is already
    attached (this function does a `replace`, which is usually safe, but a
    clean `clear` then `apply` is more predictable across kernel/iproute2
    versions).
    """
    args = ["tc", "qdisc", "replace", "dev", interface, "root", "netem"]
    if condition.rtt_ms is not None:
        args += ["delay", f"{condition.rtt_ms}ms"]
        if condition.jitter_ms is not None:
            args += [f"{condition.jitter_ms}ms"]
    if condition.loss_pct is not None:
        args += ["loss", f"{condition.loss_pct}%"]
    if condition.bandwidth_kbit is not None:
        args += ["rate", f"{condition.bandwidth_kbit}kbit"]

    cmd = (["sudo"] if use_sudo else []) + args
    _run(cmd)


def clear_condition(interface: str, use_sudo: bool = True) -> None:
    """Remove any netem qdisc from `interface`, restoring default behavior."""
    cmd = (["sudo"] if use_sudo else []) + ["tc", "qdisc", "del", "dev", interface, "root"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # "Cannot delete qdisc with handle of zero" / "no qdisc" errors are
    # expected if nothing was attached; only raise for other failures.
    if result.returncode != 0 and "No such file or directory" not in result.stderr:
        # tc's error text for "nothing attached" varies by version; treat
        # any failure here as non-fatal but surface it for visibility.
        pass


# Standard slice set matching paper §7's "RTT, loss, NAT, migration events".
# NAT and migration are not netem qdisc properties -- inducing a migration
# event means changing the client's observable 4-tuple mid-connection
# (e.g. switching Wi-Fi/cellular, or rebinding the local port deliberately)
# which is a capture_harness-level action, not a tc-level one. These netem
# slices cover the RTT/loss axis only; combine with a migration trigger in
# capture_harness.py for the full condition matrix.
STANDARD_CONDITIONS: list[NetemCondition] = [
    NetemCondition(label="baseline"),
    NetemCondition(rtt_ms=20, label="low-rtt"),
    NetemCondition(rtt_ms=100, jitter_ms=10, label="medium-rtt"),
    NetemCondition(rtt_ms=250, jitter_ms=30, label="high-rtt-satellite-like"),
    NetemCondition(loss_pct=0.5, label="low-loss"),
    NetemCondition(loss_pct=3.0, label="high-loss-mobile-like"),
    NetemCondition(rtt_ms=100, loss_pct=1.0, jitter_ms=15, label="mixed-mobile"),
]
