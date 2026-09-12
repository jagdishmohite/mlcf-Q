"""Per-layer feature extraction (x_Q, x_H, x_A, x_B) — paper Section 5.1.

Each module in this package knows how to build its `*Features` model from a
raw dict of decoded protocol fields (as produced by `mlcfq.extraction`), and
exposes a small set of derived/normalized attributes used by the coherence
scorer in `mlcfq.scoring`.
"""

from mlcfq.features.behavioral import build_behavioral_features
from mlcfq.features.handshake import build_handshake_features
from mlcfq.features.http3 import build_http3_features
from mlcfq.features.quic_transport import build_transport_features

__all__ = [
    "build_transport_features",
    "build_handshake_features",
    "build_http3_features",
    "build_behavioral_features",
]
