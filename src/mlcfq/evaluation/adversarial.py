"""Construct A1/A2 adversarial sessions from real captured layer data (paper §4, §7).

Paper §7's dataset requirement is precise about what each tier means
mechanically:

    A1 via JA4-Q replay leaving native transport parameters
    A2 via aligned handshake+HTTP/3 with native transport/timing
    A3 via QUIC-capable impersonation profiles and a real browser engine
       driven by automation

A1 and A2 are, by this definition, *recombinations* of real layer data: a
handshake captured from one genuine client class, spliced with transport
(and for A2, HTTP/3) captured from a different genuine client class. This
module does exactly that splicing, deterministically, from a pool of real
`LabeledSession` objects you've already captured with `genuine` tier.

A3 is different by design -- "a real browser engine driven by automation"
is not a recombination of captured layers, it's a real browser actually
running. That's a capture-harness task (see `capture_harness.py`), not a
data-transformation task, so it isn't implemented in this module.
"""

from __future__ import annotations

import copy
import random

from mlcfq.evaluation.dataset import LabeledSession, NetworkCondition
from mlcfq.model import Session


def _clone_session(session: Session, session_id: str) -> Session:
    cloned = Session.from_dict(session.to_dict())
    cloned.session_id = session_id
    return cloned


def make_a1_session(
    target_handshake_from: LabeledSession,
    native_stack_from: LabeledSession,
    session_id: str,
) -> LabeledSession:
    """A1: replay `target_handshake_from`'s handshake on `native_stack_from`'s
    native transport, HTTP/3, and behavioral layers.

    Ground truth is `native_stack_from`'s class -- that's the stack that
    actually produced the session; `spoofed_as` records the class whose
    JA4-Q it's impersonating, which is what a JA4-Q-only baseline would be
    fooled into reporting.
    """
    if target_handshake_from.tier != "genuine" or native_stack_from.tier != "genuine":
        raise ValueError("A1 construction requires two genuine-tier source sessions")

    session = _clone_session(native_stack_from.session, session_id)
    session.handshake = copy.deepcopy(target_handshake_from.session.handshake)

    return LabeledSession(
        session=session,
        ground_truth_class=native_stack_from.ground_truth_class,
        tier="A1",
        spoofed_as=target_handshake_from.ground_truth_class,
        network_condition=native_stack_from.network_condition,
        source_meta={
            "handshake_source": target_handshake_from.session.session_id,
            "transport_http3_behavioral_source": native_stack_from.session.session_id,
        },
    )


def make_a2_session(
    target_handshake_http3_from: LabeledSession,
    native_transport_from: LabeledSession,
    session_id: str,
) -> LabeledSession:
    """A2: align handshake AND HTTP/3 with a target profile, leaving
    transport and behavioral layers native.

    Ground truth is still the native stack (`native_transport_from`) --
    this is the harder partial-spoofing case (paper §4's A2), where more
    layers agree with the target than in A1, but transport/timing still
    diverge.
    """
    if target_handshake_http3_from.tier != "genuine" or native_transport_from.tier != "genuine":
        raise ValueError("A2 construction requires two genuine-tier source sessions")

    session = _clone_session(native_transport_from.session, session_id)
    session.handshake = copy.deepcopy(target_handshake_http3_from.session.handshake)
    session.http3 = copy.deepcopy(target_handshake_http3_from.session.http3)

    return LabeledSession(
        session=session,
        ground_truth_class=native_transport_from.ground_truth_class,
        tier="A2",
        spoofed_as=target_handshake_http3_from.ground_truth_class,
        network_condition=native_transport_from.network_condition,
        source_meta={
            "handshake_http3_source": target_handshake_http3_from.session.session_id,
            "transport_behavioral_source": native_transport_from.session.session_id,
        },
    )


def build_a1_a2_partitions(
    genuine_sessions: list[LabeledSession],
    target_classes: list[str],
    source_classes: list[str],
    seed: int = 0,
) -> tuple[list[LabeledSession], list[LabeledSession]]:
    """Bulk-construct A1 and A2 partitions by pairing every genuine session
    of a `source_classes` stack with a randomly chosen genuine handshake
    from each `target_classes` stack.

    Typical usage: `target_classes` are the browser profiles being spoofed
    (e.g. Chrome, Firefox), `source_classes` are the library stacks doing
    the spoofing (e.g. quic-go, aioquic) -- matching paper §7's framing of
    A1/A2 as library clients impersonating browser JA4-Q fingerprints.
    Returns `(a1_sessions, a2_sessions)`.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[LabeledSession]] = {}
    for s in genuine_sessions:
        if s.tier != "genuine":
            continue
        by_class.setdefault(s.ground_truth_class, []).append(s)

    missing = [c for c in target_classes + source_classes if c not in by_class]
    if missing:
        raise ValueError(
            f"no genuine sessions found for class(es) {missing}; "
            f"available classes: {sorted(by_class)}"
        )

    a1_sessions, a2_sessions = [], []
    counter = 0
    for source_class in source_classes:
        for source_session in by_class[source_class]:
            for target_class in target_classes:
                target_session = rng.choice(by_class[target_class])
                counter += 1
                a1_sessions.append(
                    make_a1_session(target_session, source_session, f"a1-{counter:05d}")
                )
                counter += 1
                a2_sessions.append(
                    make_a2_session(target_session, source_session, f"a2-{counter:05d}")
                )
    return a1_sessions, a2_sessions
