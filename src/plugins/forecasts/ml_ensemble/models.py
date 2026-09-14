"""Forecasting models.

Six base learners with genuinely different inductive biases, blended by soft
voting:

* **LightGBM** — gradient-boosted trees. Usually the strongest single model on
  tabular data of this kind, and fast enough to retrain in a walk-forward loop.
* **HistGradientBoosting** — sklearn's boosting implementation. Similar family,
  different binning; correlated with LightGBM but not identical.
* **ExtraTrees** — randomised bagging. Decorrelated from the boosted models
  because it fits deep, high-variance trees independently rather than
  sequentially. This is where most of the ensemble's diversity comes from.
* **RandomForest** — bootstrap aggregation with conventional split selection.
  It is deliberately down-weighted by default because it is related to
  ExtraTrees, but is less sensitive to the latter's random split thresholds.
* **Shrinkage LDA** — a regularised generative covariance model. This gives the
  blend a smooth, non-tree decision surface that remains usable when features
  are numerous and correlated.
* **Logistic regression** — a linear baseline. It is included partly because it
  regularises the blend, and mostly because if the linear model is competitive
  with the boosted ones, the features are probably not carrying much signal and
  that is worth knowing.

Non-tree models cannot consume NaN, so every pipeline imputes with a
train-fitted median. Imputing before splitting would leak test-set statistics
into training, so the imputer lives inside the pipeline and is refit per fold.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def lightgbm_available() -> bool:
    """Whether LightGBM can be imported and its native library loaded.

    On macOS the wheel links against OpenMP, so importing LightGBM fails with a
    dlopen error when ``libomp`` is missing. That is an install problem, not a
    code problem — the right response is to drop to the remaining learners rather
    than take the whole platform down. ``brew install libomp`` fixes it.
    """
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
    except Exception:
        return False
    return True


def build_lightgbm(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    from lightgbm import LGBMClassifier

    options = {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 120,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    options.update(parameters)

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                LGBMClassifier(**options),
            ),
        ]
    )


def build_hist_gradient(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    # max_iter is deliberately modest. HistGradientBoosting converges well before
    # 300 iterations on this data, and the extra time buys nothing — early
    # stopping usually fires around 80-120.
    options = {
        "max_iter": 150,
        "learning_rate": 0.06,
        "max_depth": 6,
        "min_samples_leaf": 120,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 12,
        "random_state": random_state,
    }
    options.update(parameters)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(**options),
            ),
        ]
    )


def build_extra_trees(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    options = {
        "n_estimators": 250,
        "max_depth": 12,
        "min_samples_leaf": 60,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    options.update(parameters)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(**options),
            ),
        ]
    )


def build_random_forest(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    options = {
        "n_estimators": 220,
        "max_depth": 11,
        "min_samples_leaf": 70,
        "max_features": 0.35,
        "class_weight": "balanced_subsample",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    options.update(parameters)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**options)),
        ]
    )


def build_shrinkage_lda(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    options = {"solver": "lsqr", "shrinkage": "auto"}
    options.update(parameters)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(**options)),
        ]
    )


def build_logistic(
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **parameters,
) -> Pipeline:
    options = {
        "C": 0.05,
        "max_iter": 1500,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    options.update(parameters)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(**options),
            ),
        ]
    )


BUILDERS = {
    "lightgbm": build_lightgbm,
    "hist_gbm": build_hist_gradient,
    "extra_trees": build_extra_trees,
    "random_forest": build_random_forest,
    "shrinkage_lda": build_shrinkage_lda,
    "logistic": build_logistic,
}

DEFAULT_WEIGHTS = {
    "lightgbm": 1.25,
    "hist_gbm": 1.0,
    "extra_trees": 1.0,
    "random_forest": 0.75,
    "shrinkage_lda": 0.7,
    "logistic": 0.65,
}


def build_models(
    names: Sequence[str] | None = None,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    parameters: Mapping[str, Mapping] | None = None,
) -> dict[str, Pipeline]:
    """Instantiate the requested base learners.

    LightGBM is silently skipped when its native library will not load, so a
    missing ``libomp`` degrades the ensemble rather than breaking training.

    ``n_jobs`` is threaded through to the estimators. When folds are trained in
    parallel it should be set to 1: leaving both levels at ``-1`` oversubscribes
    the CPU and runs measurably slower than either level alone.
    """
    if names is None:
        names = tuple(BUILDERS)
    unknown = sorted(set(names) - set(BUILDERS))
    if unknown:
        raise ValueError(
            f"unknown batch models: {', '.join(unknown)}; available: {', '.join(BUILDERS)}"
        )
    configured = parameters or {}
    unknown_parameters = sorted(set(configured) - set(BUILDERS))
    if unknown_parameters:
        raise ValueError(f"parameters configured for unknown models: {', '.join(unknown_parameters)}")

    out: dict[str, Pipeline] = {}
    for name in names:
        if name == "lightgbm" and not lightgbm_available():
            continue
        options = dict(configured.get(name, {}))
        out[name] = BUILDERS[name](random_state=random_state, n_jobs=n_jobs, **options)
    if not out:
        raise ValueError("none of the selected batch models is available")
    return out


class DirectionEnsemble:
    """Soft-voting blend of directional classifiers.

    Probabilities are averaged rather than votes counted, which preserves the
    confidence information the strategy layer needs. A 0.55 and a 0.95 both count
    as "UP" under hard voting; only the probability carries how much the model
    actually believes it.
    """

    def __init__(
        self,
        models: dict[str, Pipeline],
        weights: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.models = models
        if not models:
            raise ValueError("an ensemble requires at least one model")
        unknown_weights = sorted(set(weights or {}) - set(models))
        if unknown_weights:
            raise ValueError(f"weights configured for absent models: {', '.join(unknown_weights)}")
        self.weights = (
            dict(weights)
            if weights is not None
            else {name: DEFAULT_WEIGHTS.get(name, 1.0) for name in models}
        )
        for name in models:
            weight = float(self.weights.get(name, 1.0))
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"weight for {name} must be a finite non-negative number")
            self.weights[name] = weight
        self.feature_names = feature_names
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> DirectionEnsemble:
        self.feature_names = list(X.columns)
        self.fitted = True
        for model in self.models.values():
            try:
                if sample_weight is not None:
                    model.fit(X, y, clf__sample_weight=sample_weight)
                else:
                    model.fit(X, y)
            except (TypeError, ValueError):
                # Some estimators reject sample_weight; fall back to unweighted.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of an UP move, averaged across base learners."""
        if not self.fitted:
            raise RuntimeError("ensemble is not fitted")

        total = 0.0
        weight_sum = 0.0
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            if weight == 0:
                continue
            probabilities = _up_probability(model, X)
            total = total + probabilities * weight
            weight_sum += weight

        if weight_sum == 0:
            return np.full(len(X), 0.5)
        return total / weight_sum

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def for_single_row_inference(self) -> DirectionEnsemble:
        """Stop the estimators spawning a worker pool to predict one row.

        ``ExtraTrees(n_jobs=-1)`` runs its trees through joblib's process backend,
        so every ``predict_proba`` on a single row starts a loky pool — around
        200ms of process startup, per call, per instrument, to parallelise work
        that takes microseconds. Training keeps its parallelism; this only
        touches inference, where the rows to share out number exactly one.
        """
        for model in self.models.values():
            for _, step in getattr(model, "steps", []) or []:
                if hasattr(step, "n_jobs"):
                    step.n_jobs = 1
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
        return self

    def member_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-model probabilities — lets the UI show where disagreement sits."""
        return pd.DataFrame(
            {name: _up_probability(model, X) for name, model in self.models.items()},
            index=X.index,
        )

    def feature_importance(self, top: int = 25) -> pd.Series:
        """Mean absolute importance across models, normalised to sum to 1."""
        contributions: list[pd.Series] = []
        names = self.feature_names or []

        for model in self.models.values():
            estimator = model.named_steps.get("clf") if isinstance(model, Pipeline) else model
            if estimator is None:
                continue

            if hasattr(estimator, "feature_importances_"):
                values = np.asarray(estimator.feature_importances_, dtype="float64")
            elif hasattr(estimator, "coef_"):
                values = np.abs(np.ravel(estimator.coef_)).astype("float64")
            else:
                continue

            if len(values) != len(names):
                continue
            total = values.sum()
            if total > 0:
                contributions.append(pd.Series(values / total, index=names))

        if not contributions:
            return pd.Series(dtype="float64")

        combined = pd.concat(contributions, axis=1).mean(axis=1).sort_values(ascending=False)
        return combined.head(top)

    def disagreement(self, X: pd.DataFrame) -> pd.Series:
        """Standard deviation of member probabilities — high means the models split."""
        return self.member_probabilities(X).std(axis=1, ddof=0)


def _up_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Extract P(class = 1) regardless of how the estimator orders classes."""
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if isinstance(model, Pipeline):
        classes = list(model.named_steps["clf"].classes_)

    if classes == [0]:
        return np.zeros(len(X), dtype="float64")
    if classes == [1]:
        return np.ones(len(X), dtype="float64")
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    raise ValueError(f"model exposes unsupported classes {classes!r}; expected binary labels 0 and 1")
