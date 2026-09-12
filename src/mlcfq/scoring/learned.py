"""Instantiation L — learned coherence scorer (paper §5.2).

    Δ_k = − log p_k(x_j | x_i)

estimated from labeled benign QUIC traffic, capturing unenumerated
couplings with graded penalties, at the cost of training data and
auditability (vs. Instantiation R's hand-written predicates).

This reference implementation ships a working *scaffold* — feature
vectorization, a per-class conditional-density estimator per dependency-
graph edge, and a `fit`/`score` interface matching `CoherenceScorer` — but
does not ship trained weights, since that requires the labeled dataset
described in paper §7 (collected under the evaluation protocol, not
included in this repo). Swap in your own labeled sessions via `fit()`.

Requires the optional `scikit-learn` dependency (`pip install mlcfq[learned]`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from mlcfq.model import ClientProfile, Session
from mlcfq.scoring.coherence import LAYER_NAMES, DEFAULT_DEPENDENCY_GRAPH


def _flatten_layer(session: Session, layer_name: str) -> dict[str, Any]:
    """Turn a layer's dataclass into a flat, vectorizable dict."""
    layer = getattr(session, layer_name)
    out: dict[str, Any] = {}
    for key, value in dataclasses.asdict(layer).items():
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, str):
            out[key] = value
        elif isinstance(value, list):
            out[f"{key}__len"] = len(value)
        elif isinstance(value, dict):
            out[f"{key}__len"] = len(value)
    return out


class LearnedScorer:
    """Instantiation L: fits one density model per class per layer, plus
    one conditional model per dependency-graph edge for the mismatch term.

    This is intentionally a thin, swappable scaffold — e.g. a
    `GaussianMixture`/`KernelDensity` per (class, layer) for
    `layer_log_likelihood`, and a classifier-based conditional density
    estimate per edge for `mismatch_penalty`. Production use should
    validate calibration on held-out benign traffic per paper §5.2 and §7.
    """

    def __init__(self, dependency_graph=DEFAULT_DEPENDENCY_GRAPH):
        self.dependency_graph = dependency_graph
        self._layer_models: dict[tuple[str, str], Any] = {}  # (class, layer) -> model
        self._edge_models: dict[tuple[str, str, str], Any] = {}  # (class, i, j) -> model
        self._vectorizers: dict[str, Any] = {}
        self.fitted = False

    def fit(self, labeled_sessions: list[Session]) -> None:
        """Fit per-class, per-layer density models from labeled benign traffic.

        `labeled_sessions` must have `claimed_class` set (paper §7's
        self-generated, ground-truth-labeled corpus). Raises if
        scikit-learn isn't installed.
        """
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Instantiation L requires scikit-learn: pip install mlcfq[learned]"
            ) from exc

        by_class: dict[str, list[Session]] = {}
        for s in labeled_sessions:
            if not s.claimed_class:
                continue
            by_class.setdefault(s.claimed_class, []).append(s)

        for layer in LAYER_NAMES:
            vec = DictVectorizer(sparse=False)
            all_flat = [_flatten_layer(s, layer) for s in labeled_sessions]
            if not all_flat:
                continue
            vec.fit(all_flat)
            self._vectorizers[layer] = vec

            for class_name, sessions in by_class.items():
                flat = [_flatten_layer(s, layer) for s in sessions]
                if len(flat) < 2:
                    continue
                X = vec.transform(flat)
                n_components = min(2, len(flat))
                model = GaussianMixture(n_components=n_components, random_state=0)
                model.fit(X)
                self._layer_models[(class_name, layer)] = model

        # Edge models: for each dependency-graph edge, fit a joint density
        # over the concatenation of both layers per class, used to derive
        # Δ_k = -log p_k(x_j | x_i) via the joint/marginal ratio.
        for (i, j) in self.dependency_graph:
            for class_name, sessions in by_class.items():
                if len(sessions) < 2:
                    continue
                flat_i = [_flatten_layer(s, i) for s in sessions]
                flat_j = [_flatten_layer(s, j) for s in sessions]
                vec_i = self._vectorizers.get(i)
                vec_j = self._vectorizers.get(j)
                if vec_i is None or vec_j is None:
                    continue
                Xi = vec_i.transform(flat_i)
                Xj = vec_j.transform(flat_j)
                joint = np.concatenate([Xi, Xj], axis=1)
                n_components = min(2, len(sessions))
                model = GaussianMixture(n_components=n_components, random_state=0)
                model.fit(joint)
                self._edge_models[(class_name, i, j)] = model

        self.fitted = True

    def layer_log_likelihood(self, layer_name: str, session: Session, profile: ClientProfile) -> float:
        if not self.fitted:
            raise RuntimeError("LearnedScorer.fit() must be called before scoring")
        model = self._layer_models.get((profile.name, layer_name))
        vec = self._vectorizers.get(layer_name)
        if model is None or vec is None:
            return 0.0
        X = vec.transform([_flatten_layer(session, layer_name)])
        return float(model.score_samples(X)[0])

    def mismatch_penalty(
        self, layer_i: str, layer_j: str, session: Session, profile: ClientProfile
    ) -> float:
        if not self.fitted:
            raise RuntimeError("LearnedScorer.fit() must be called before scoring")
        model = self._edge_models.get((profile.name, layer_i, layer_j))
        vec_i, vec_j = self._vectorizers.get(layer_i), self._vectorizers.get(layer_j)
        if model is None or vec_i is None or vec_j is None:
            return 0.0
        Xi = vec_i.transform([_flatten_layer(session, layer_i)])
        Xj = vec_j.transform([_flatten_layer(session, layer_j)])
        joint = np.concatenate([Xi, Xj], axis=1)
        # -log p_k(x_i, x_j) as a stand-in for -log p_k(x_j | x_i); a full
        # conditional would additionally divide out the x_i marginal.
        return float(-model.score_samples(joint)[0])
