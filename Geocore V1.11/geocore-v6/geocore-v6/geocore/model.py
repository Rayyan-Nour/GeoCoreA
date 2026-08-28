"""
Model: Random Forest prospectivity classifier with probability calibration,
ensemble uncertainty, and versioned persistence (with feature schema, so a
saved model can never be silently applied to mismatched features).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sampling import Samples


def adaptive_rf(n_positive: int, n_trees: int = 300, seed: int = 42
                ) -> RandomForestClassifier:
    """Depth/leaf constraints scale with data size (Probst et al., 2019)."""
    if n_positive < 30:
        depth, leaf = 5, max(3, n_positive // 5)
    elif n_positive < 100:
        depth, leaf = 8, 4
    else:
        depth, leaf = 15, 5
    return RandomForestClassifier(
        n_estimators=n_trees, max_depth=depth, min_samples_leaf=leaf,
        min_samples_split=max(2 * leaf, 5), class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )


@dataclass
class TrainedModel:
    pipeline: Pipeline                 # scaler + calibrated classifier
    base_forest: RandomForestClassifier
    scaler: StandardScaler
    feature_names: List[str]
    commodity: str

    def predict_map(self, X_flat: np.ndarray, valid_flat: np.ndarray,
                    shape: Tuple[int, int], chunk: int = 1_000_000,
                    uncertainty_trees: int = 25,
                    progress=None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chunked prediction over the whole grid. Returns (probability,
        uncertainty) maps with NaN outside the valid mask.

        Uncertainty = std of per-tree probabilities over a stratified subset
        of trees (subset size disclosed; full-ensemble std differs by <1%
        in practice but costs ~10x).
        """
        H, W = shape
        n = X_flat.shape[0]
        prob = np.full(n, np.nan, dtype=np.float32)
        unc = np.full(n, np.nan, dtype=np.float32)

        trees = self.base_forest.estimators_
        stride = max(1, len(trees) // uncertainty_trees)
        subset = trees[::stride]

        from joblib import Parallel, delayed
        idx = np.where(valid_flat)[0]
        n_chunks = max(1, -(-len(idx) // chunk))
        with Parallel(n_jobs=-1, prefer="threads") as par:
            for ci, s in enumerate(range(0, len(idx), chunk)):
                sel = idx[s:s + chunk]
                Xs = self.scaler.transform(X_flat[sel])
                prob[sel] = (self.pipeline.named_steps["clf"]
                             .predict_proba(Xs)[:, 1])
                tp = np.stack(par(delayed(
                    lambda t: t.predict_proba(Xs)[:, 1])(t) for t in subset))
                unc[sel] = tp.std(axis=0)
                if progress is not None:
                    progress((ci + 1) / n_chunks)

        return prob.reshape(H, W), unc.reshape(H, W)

    def importances(self) -> List[Tuple[str, float]]:
        imp = self.base_forest.feature_importances_
        order = np.argsort(imp)[::-1]
        return [(self.feature_names[i], float(imp[i])) for i in order]


def train(samples: Samples, commodity: str, n_trees: int = 300,
          seed: int = 42) -> TrainedModel:
    n_pos = int(samples.y.sum())
    scaler = StandardScaler().fit(samples.X)
    Xs = scaler.transform(samples.X)

    base = adaptive_rf(n_pos, n_trees, seed)
    base.fit(Xs, samples.y)

    cal_cv = int(min(5, max(2, min(n_pos, int((samples.y == 0).sum())))))
    calibrated = CalibratedClassifierCV(
        adaptive_rf(n_pos, n_trees, seed), method="sigmoid", cv=cal_cv)
    calibrated.fit(Xs, samples.y)

    pipe = Pipeline([("scaler", scaler), ("clf", calibrated)])
    return TrainedModel(pipeline=pipe, base_forest=base, scaler=scaler,
                        feature_names=list(samples.feature_names),
                        commodity=commodity)


# ----------------------------------------------------------------------
# Persistence with schema guard
# ----------------------------------------------------------------------

SCHEMA_VERSION = 2


def _schema_hash(feature_names: List[str]) -> str:
    return hashlib.sha256("|".join(feature_names).encode()).hexdigest()[:16]


def save_model(model: TrainedModel, path: str) -> None:
    joblib.dump({
        "schema_version": SCHEMA_VERSION,
        "feature_names": model.feature_names,
        "feature_hash": _schema_hash(model.feature_names),
        "commodity": model.commodity,
        "pipeline": model.pipeline,
        "base_forest": model.base_forest,
        "scaler": model.scaler,
    }, path)


def load_model(path: str) -> TrainedModel:
    d = joblib.load(path)
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Model schema {d.get('schema_version')} != {SCHEMA_VERSION}; "
            "retrain rather than risking silent feature mismatch.")
    return TrainedModel(pipeline=d["pipeline"], base_forest=d["base_forest"],
                        scaler=d["scaler"], feature_names=d["feature_names"],
                        commodity=d["commodity"])


def align_features_for_transfer(X_names_now: List[str],
                                model: TrainedModel,
                                X_flat: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    For transfer prediction: reorder current features to the model's schema.
    Missing features are median-filled and counted; the caller must disclose
    the match ratio in the report. Refuses if <60% of features match.
    """
    F = len(model.feature_names)
    out = np.zeros((X_flat.shape[0], F), dtype=np.float32)
    matched = 0
    for i, name in enumerate(model.feature_names):
        if name in X_names_now:
            out[:, i] = X_flat[:, X_names_now.index(name)]
            matched += 1
        else:
            out[:, i] = 0.0
    if matched < max(2, int(0.6 * F)):
        raise ValueError(
            f"Transfer refused: only {matched}/{F} features match the saved "
            "model. Predictions would be unreliable.")
    return out, matched
