"""Evaluation metrics for directional forecasts.

Accuracy is reported but is never the headline. On a dataset with a 53% UP base
rate, predicting UP unconditionally scores 53% — so accuracy alone cannot
distinguish a model from a constant. Every report here includes the base rate and
the lift over it, plus precision at a confidence threshold, because that is the
regime the strategy actually trades in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    """Metrics for one model on one out-of-sample set."""

    n: int
    base_rate: float
    accuracy: float
    auc: float
    brier: float
    logloss: float
    mcc: float
    precision_up: float
    recall_up: float
    f1_up: float
    precision_down: float
    recall_down: float
    f1_down: float
    accuracy_lift: float
    mean_confidence: float = 0.0
    high_conf_accuracy: float = 0.0
    high_conf_coverage: float = 0.0
    threshold: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "lift": round(self.accuracy_lift, 4),
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "logloss": round(self.logloss, 4),
            "mcc": round(self.mcc, 4),
            "prec_up": round(self.precision_up, 4),
            "prec_down": round(self.precision_down, 4),
            "hi_conf_acc": round(self.high_conf_accuracy, 4),
            "hi_conf_cov": round(self.high_conf_coverage, 4),
        }

    def lines(self) -> list[str]:
        return [
            f"samples              {self.n:,}",
            f"base rate (UP)       {self.base_rate:.3f}",
            f"accuracy             {self.accuracy:.4f}   (lift {self.accuracy_lift:+.4f} over base)",
            f"ROC AUC              {self.auc:.4f}",
            f"Brier / log loss     {self.brier:.4f} / {self.logloss:.4f}",
            f"MCC                  {self.mcc:+.4f}",
            f"precision UP / DOWN  {self.precision_up:.4f} / {self.precision_down:.4f}",
            f"recall    UP / DOWN  {self.recall_up:.4f} / {self.recall_down:.4f}",
            (
                f"high-confidence      {self.high_conf_accuracy:.4f} accuracy on "
                f"{self.high_conf_coverage:.1%} of bars (|p-0.5| >= {self.threshold:.2f})"
            ),
        ]


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.06,
) -> ClassificationReport:
    """Score predicted UP-probabilities against realised labels."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype="float64")
    predictions = (probabilities >= 0.5).astype(int)

    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    accuracy = float((predictions == y_true).mean()) if len(y_true) else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=[1, 0], zero_division=0
    )

    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = float("nan")

    try:
        logloss = float(log_loss(y_true, probabilities, labels=[0, 1]))
    except ValueError:
        logloss = float("nan")

    confidence = np.abs(probabilities - 0.5)
    confident = confidence >= (threshold / 2.0)
    high_conf_accuracy = float((predictions[confident] == y_true[confident]).mean()) if confident.any() else float("nan")
    high_conf_coverage = float(confident.mean())

    return ClassificationReport(
        n=len(y_true),
        base_rate=base_rate,
        accuracy=accuracy,
        auc=auc,
        brier=float(brier_score_loss(y_true, probabilities)),
        logloss=logloss,
        mcc=float(matthews_corrcoef(y_true, predictions)),
        precision_up=float(precision[0]),
        recall_up=float(recall[0]),
        f1_up=float(f1[0]),
        precision_down=float(precision[1]),
        recall_down=float(recall[1]),
        f1_down=float(f1[1]),
        accuracy_lift=accuracy - max(base_rate, 1.0 - base_rate),
        mean_confidence=float(confidence.mean()),
        high_conf_accuracy=high_conf_accuracy,
        high_conf_coverage=high_conf_coverage,
        threshold=threshold,
    )


def edge_by_confidence(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 5,
) -> pd.DataFrame:
    """Accuracy bucketed by model confidence.

    This is the single most useful diagnostic for a trading model. If accuracy
    does not rise with confidence, the probabilities carry no information about
    when to trade — even if overall accuracy looks respectable.
    """
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(int),
            "p": np.asarray(probabilities, dtype="float64"),
        }
    )
    frame["prediction"] = (frame["p"] >= 0.5).astype(int)
    frame["confidence"] = (frame["p"] - 0.5).abs() * 2.0
    frame["correct"] = (frame["prediction"] == frame["y"]).astype(int)
    frame["bucket"] = pd.cut(
        frame["confidence"], bins=bins, labels=[f"Q{i + 1}" for i in range(bins)], include_lowest=True
    )

    grouped = frame.groupby("bucket", observed=True).agg(
        samples=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_probability=("p", "mean"),
        realised_up_rate=("y", "mean"),
    )
    grouped["calibration_error"] = (grouped["mean_probability"] - grouped["realised_up_rate"]).abs()
    return grouped.round(4)


def expected_move_curve(
    forward_returns: pd.Series,
    probabilities: np.ndarray,
    bins: int = 8,
    min_per_bin: int = 100,
) -> dict:
    """Map model conviction to the realised size of the move it preceded.

    A heuristic like ``edge x ATR x sqrt(horizon)`` has no basis in the data and
    will happily promise moves that never materialise. This measures it instead:
    group the out-of-sample predictions by conviction and record how far price
    actually travelled. That curve is what makes a cost-hurdle check honest —
    without it, deciding whether a forecast can pay for its own round trip is
    guesswork.

    Returns knots suitable for ``np.interp``, plus the fallback the caller should
    use when conviction falls outside the fitted range.
    """
    forward_returns = pd.Series(forward_returns).astype("float64")
    probabilities = np.asarray(probabilities, dtype="float64").ravel()
    edge = np.abs(probabilities - 0.5) * 2.0

    frame = pd.DataFrame(
        {"edge": edge, "abs_move_bps": forward_returns.abs().to_numpy() * 10_000}
    ).dropna()
    if frame.empty or len(frame) < bins * min_per_bin:
        return {"edges": [], "moves": [], "mean_move_bps": float(frame["abs_move_bps"].mean()) if len(frame) else 0.0}

    frame["bucket"] = pd.qcut(frame["edge"], bins, duplicates="drop")
    grouped = frame.groupby("bucket", observed=True).agg(
        edge=("edge", "mean"),
        move=("abs_move_bps", "mean"),
        samples=("edge", "size"),
    )
    grouped = grouped[grouped["samples"] >= min_per_bin].sort_values("edge")

    if grouped.empty:
        return {
            "edges": [],
            "moves": [],
            "mean_move_bps": float(frame["abs_move_bps"].mean()),
        }

    # Enforce monotonicity: a more confident model must not map to a smaller
    # expected move merely because a bucket happened to land that way.
    moves = np.maximum.accumulate(grouped["move"].to_numpy(dtype="float64"))

    return {
        "edges": grouped["edge"].to_numpy(dtype="float64").tolist(),
        "moves": moves.tolist(),
        "mean_move_bps": float(frame["abs_move_bps"].mean()),
    }


def apply_expected_move_curve(
    edge: float,
    curve: dict,
) -> float:
    """Interpolate an expected move in bps from a fitted conviction curve."""
    edges = curve.get("edges") or []
    moves = curve.get("moves") or []
    if not edges or not moves:
        return float(curve.get("mean_move_bps", 0.0))
    if edge <= edges[0]:
        return float(moves[0])
    if edge >= edges[-1]:
        return float(moves[-1])
    return float(np.interp(edge, edges, moves))


def realised_return_by_signal(
    forward_returns: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.06,
) -> pd.DataFrame:
    """Average realised forward return conditioned on the model's view.

    Ties the statistics back to money: a model can be 55% accurate and still have
    a negative expectancy once the losing moves are bigger than the winning ones.
    """
    frame = pd.DataFrame(
        {"forward": forward_returns.to_numpy(dtype="float64"), "p": probabilities},
        index=forward_returns.index,
    )
    edge = frame["p"] - 0.5

    frame["view"] = np.where(
        edge >= threshold / 2,
        "long",
        np.where(edge <= -threshold / 2, "short", "flat"),
    )

    grouped = frame.groupby("view").agg(
        bars=("forward", "size"),
        mean_return_bps=("forward", lambda values: float(np.mean(values) * 10_000)),
        median_return_bps=("forward", lambda values: float(np.median(values) * 10_000)),
        hit_rate=("forward", lambda values: float((values > 0).mean())),
        volatility_bps=("forward", lambda values: float(np.std(values) * 10_000)),
    )
    return grouped.round(3)
