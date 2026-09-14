"""Public records returned by the online research plugin."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class ResearchAction(StrEnum):
    """A research classification, never an instruction sent to a broker."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class AlgorithmPrediction:
    prediction_id: str
    algorithm: str
    p_up: float
    action: ResearchAction
    confidence: float
    trust_score: float


@dataclass(frozen=True, slots=True)
class ResearchSignal:
    """One research-only ensemble view and every member behind it."""

    issued_at: datetime
    target_at: datetime
    instrument: str
    horizon_min: int
    p_up: float
    action: ResearchAction
    confidence: float
    trust_score: float
    predictions: tuple[AlgorithmPrediction, ...]

    @property
    def label(self) -> str:
        return f"RESEARCH SIGNAL: {self.action.value}"


@dataclass(slots=True)
class PredictionRecord:
    """An immutable-at-issue forecast, filled with outcome fields at maturity."""

    prediction_id: str
    algorithm: str
    instrument: str
    issued_at: datetime
    target_at: datetime
    horizon_min: int
    anchor_price: float
    p_up: float
    action: ResearchAction
    confidence: float
    trust_at_issue: float
    cost_bps: float
    features: dict[str, float]
    matured_at: datetime | None = None
    actual_price: float | None = None
    actual_return_bps: float | None = None
    label_up: int | None = None
    hit: bool | None = None
    brier: float | None = None
    gross_pnl_bps: float | None = None
    net_pnl_bps: float | None = None

    @property
    def is_scored(self) -> bool:
        return self.matured_at is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "algorithm": self.algorithm,
            "instrument": self.instrument,
            "issued_at": self.issued_at.isoformat(),
            "target_at": self.target_at.isoformat(),
            "horizon_min": self.horizon_min,
            "anchor_price": self.anchor_price,
            "p_up": self.p_up,
            "action": self.action.value,
            "confidence": self.confidence,
            "trust_at_issue": self.trust_at_issue,
            "cost_bps": self.cost_bps,
            "features": dict(self.features),
            "matured_at": self.matured_at.isoformat() if self.matured_at else None,
            "actual_price": self.actual_price,
            "actual_return_bps": self.actual_return_bps,
            "label_up": self.label_up,
            "hit": self.hit,
            "brier": self.brier,
            "gross_pnl_bps": self.gross_pnl_bps,
            "net_pnl_bps": self.net_pnl_bps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionRecord:
        matured = payload.get("matured_at")
        return cls(
            prediction_id=str(payload["prediction_id"]),
            algorithm=str(payload["algorithm"]),
            instrument=str(payload.get("instrument", "")),
            issued_at=datetime.fromisoformat(str(payload["issued_at"])),
            target_at=datetime.fromisoformat(str(payload["target_at"])),
            horizon_min=int(payload["horizon_min"]),
            anchor_price=float(payload["anchor_price"]),
            p_up=float(payload["p_up"]),
            action=ResearchAction(str(payload["action"])),
            confidence=float(payload.get("confidence", 0.0)),
            trust_at_issue=float(payload.get("trust_at_issue", 0.0)),
            cost_bps=float(payload.get("cost_bps", 0.0)),
            features={
                str(name): float(value)
                for name, value in dict(payload.get("features", {})).items()
            },
            matured_at=datetime.fromisoformat(str(matured)) if matured else None,
            actual_price=_optional_float(payload.get("actual_price")),
            actual_return_bps=_optional_float(payload.get("actual_return_bps")),
            label_up=_optional_int(payload.get("label_up")),
            hit=_optional_bool(payload.get("hit")),
            brier=_optional_float(payload.get("brier")),
            gross_pnl_bps=_optional_float(payload.get("gross_pnl_bps")),
            net_pnl_bps=_optional_float(payload.get("net_pnl_bps")),
        )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    sample_count: int
    base_rate: float
    accuracy: float
    accuracy_lift: float
    brier_score: float
    calibration_error: float
    trade_count: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl_bps: float
    cost_paid_bps: float
    profit_bps: float
    loss_bps: float
    net_pnl_bps: float
    mean_net_pnl_bps: float
    max_drawdown_bps: float


@dataclass(frozen=True, slots=True)
class AlgorithmScorecard:
    algorithm: str
    all_time: MetricSummary
    rolling: MetricSummary
    trust_score: float

    def as_row(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "samples": self.all_time.sample_count,
            "accuracy": round(self.all_time.accuracy, 4),
            "rolling_accuracy": round(self.rolling.accuracy, 4),
            "brier": round(self.all_time.brier_score, 4),
            "calibration_error": round(self.all_time.calibration_error, 4),
            "trades": self.all_time.trade_count,
            "net_pnl_bps": round(self.all_time.net_pnl_bps, 2),
            "trust_score": round(self.trust_score, 4),
        }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


__all__ = [
    "AlgorithmPrediction",
    "AlgorithmScorecard",
    "MetricSummary",
    "PredictionRecord",
    "ResearchAction",
    "ResearchSignal",
]
