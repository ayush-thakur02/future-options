"""Inference: turn a feature row into a calibrated forecast."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..models import Direction, Prediction
from ..trading_calendar import IST
from .metrics import apply_expected_move_curve
from .trainer import load_artifact


class Predictor:
    """Loads trained artifacts and produces calibrated forecasts.

    Kept deliberately thin: it holds no state between calls beyond the loaded
    models, so the same instance serves both the backtest replay and the live
    dashboard without behavioural drift between the two.
    """

    def __init__(self, settings: Settings, horizons: tuple[int, ...] | None = None) -> None:
        self.settings = settings
        self.horizons = horizons or settings.horizons
        self.artifacts: dict[int, dict] = {}
        self.load_errors: dict[int, str] = {}

    def load(self) -> Predictor:
        for horizon in self.horizons:
            try:
                self.artifacts[horizon] = load_artifact(self.settings, horizon)
            except (FileNotFoundError, Exception) as exc:  # noqa: BLE001
                self.load_errors[horizon] = str(exc)
        return self

    @property
    def is_ready(self) -> bool:
        return bool(self.artifacts)

    @property
    def loaded_horizons(self) -> list[int]:
        return sorted(self.artifacts)

    def predict(self, features: pd.DataFrame, timestamp: datetime | None = None) -> list[Prediction]:
        """Forecast every loaded horizon from the final row of ``features``."""
        if features.empty or not self.artifacts:
            return []

        timestamp = timestamp or _index_time(features.index)
        predictions: list[Prediction] = []
        for artifact in self.artifacts.values():
            prediction = self._predict_horizon(features, artifact, timestamp)
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    def _predict_horizon(
        self,
        features: pd.DataFrame,
        artifact: dict,
        timestamp: datetime,
    ) -> Prediction | None:
        horizon = int(artifact["horizon"])
        names = artifact["feature_names"]
        present = [name for name in names if name in features.columns]
        if len(present) < len(names) * 0.5:
            return None

        row = features[present].iloc[[-1]]
        model = artifact["model"]

        try:
            raw = float(model.predict_proba(row)[0])
        except Exception:
            return None

        calibrator = artifact.get("calibrator")
        probability = (
            float(calibrator.transform(np.array([raw]))[0])
            if calibrator is not None and getattr(calibrator, "is_fitted", False)
            else raw
        )
        probability = float(np.clip(probability, 1e-6, 1 - 1e-6))

        direction = Direction.UP if probability > 0.5 else Direction.DOWN
        confidence = abs(probability - 0.5) * 2.0

        # Expected move comes from the fitted conviction curve measured on
        # out-of-sample data, not from a volatility heuristic. Deciding whether a
        # forecast can pay for its own round trip is only meaningful if the
        # expected move is something the model actually demonstrated.
        curve = artifact.get("move_curve") or {}
        expected_move = apply_expected_move_curve(confidence, curve)
        # The curve stores magnitude, so restore direction from the forecast.
        signed_move = expected_move if probability > 0.5 else -expected_move
        hurdle = float(artifact.get("hurdle_bps", 0.0))

        contributions: dict = {}
        try:
            members = model.member_probabilities(row)
            contributions = {
                name: round(float(value), 4) for name, value in members.iloc[0].items()
            }
        except Exception:
            pass

        return Prediction(
            ts=timestamp,
            horizon_min=horizon,
            p_up=probability,
            direction=direction,
            confidence=confidence,
            model="ensemble",
            expected_move_bps=signed_move,
            hurdle_bps=hurdle,
            contributions=contributions,
        )

    def predict_frame(self, features: pd.DataFrame, horizon: int) -> pd.Series:
        """Vectorised calibrated probabilities for a whole frame.

        Used by the backtester to replay historical decisions using the same
        inference path as live.
        """
        artifact = self.artifacts.get(horizon)
        if artifact is None or features.empty:
            return pd.Series(dtype="float64")

        names = artifact["feature_names"]
        present = [name for name in names if name in features.columns]
        if not present:
            return pd.Series(dtype="float64")

        raw = artifact["model"].predict_proba(features[present])
        calibrator = artifact.get("calibrator")
        if calibrator is not None and getattr(calibrator, "is_fitted", False):
            raw = calibrator.transform(raw)
        return pd.Series(np.clip(raw, 1e-6, 1 - 1e-6), index=features.index)

    def describe(self) -> str:
        if not self.artifacts:
            return "no models loaded"
        parts = []
        for horizon, artifact in sorted(self.artifacts.items()):
            metrics = artifact.get("metrics", {})
            parts.append(
                f"{horizon}m acc={metrics.get('accuracy', float('nan')):.4f} "
                f"auc={metrics.get('auc', float('nan')):.4f}"
            )
        return " | ".join(parts)


def _index_time(index: pd.Index) -> datetime:
    if len(index) == 0:
        return datetime.now(IST)
    stamp = index[-1]
    if hasattr(stamp, "to_pydatetime"):
        # Bar timestamps are whole seconds; nanosecond precision is meaningless
        # here and only produces a conversion warning.
        return stamp.to_pydatetime(warn=False)
    return datetime.now(IST)


def artifacts_present(settings: Settings) -> bool:
    return any(settings.model_dir.glob("direction_*m.joblib"))


def artifact_paths(settings: Settings) -> list[Path]:
    return sorted(settings.model_dir.glob("direction_*m.joblib"))
