"""Convert domain snapshots into finite, browser-safe JSON payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from core.types import BoardSnapshot, LegSnapshot, MarketSnapshot


def serialize_snapshot(
    snapshot: BoardSnapshot | MarketSnapshot,
    *,
    status: str = "",
    max_candles: int = 160,
) -> dict[str, Any]:
    """Return a complete JSON-safe view without leaking pandas/numpy objects."""
    if isinstance(snapshot, BoardSnapshot):
        return {
            "ready": True,
            "kind": "board",
            "status": str(status),
            "ts": _time(snapshot.ts),
            "symbol": snapshot.symbol,
            "spot": _number(snapshot.spot),
            "headline": snapshot.headline,
            "note": snapshot.note,
            "chain": _json(snapshot.chain),
            "simulation": _json(snapshot.simulation),
            "legs": [
                _leg(item, max_candles=max_candles)
                for item in snapshot.legs
            ],
        }
    return {
        "ready": True,
        "kind": "market",
        "status": str(status),
        "ts": _time(snapshot.ts),
        "symbol": snapshot.symbol,
        "market": _market(snapshot, max_candles=max_candles),
    }


def _leg(leg: LegSnapshot, *, max_candles: int) -> dict[str, Any]:
    verdict = _json(leg.verdict)
    if leg.verdict is not None:
        verdict["edge_bps"] = _number(leg.verdict.edge_bps)
        verdict["is_trade"] = leg.verdict.is_trade
    return {
        "label": leg.label,
        "kind": leg.kind,
        "strike": _number(leg.strike),
        "greeks": _json(leg.greeks),
        "verdict": verdict,
        "market": _market(leg.snapshot, max_candles=max_candles),
    }


def _market(snapshot: MarketSnapshot, *, max_candles: int) -> dict[str, Any]:
    return {
        "ts": _time(snapshot.ts),
        "symbol": snapshot.symbol,
        "last_price": _number(snapshot.last_price),
        "prev_close": _number(snapshot.prev_close),
        "change": _number(snapshot.change),
        "change_pct": _number(snapshot.change_pct),
        "regime": snapshot.regime,
        "conviction": _number(snapshot.conviction),
        "source": snapshot.source,
        "instrument_key": snapshot.instrument_key,
        "last_tick_ts": _time(snapshot.last_tick_ts),
        "projection_ts": _time(snapshot.projection_ts),
        "projection_error_bps": _number(snapshot.projection_error_bps),
        "candles": _candles(snapshot.candles, max_candles=max_candles),
        "projections": [_json(item) for item in snapshot.projections],
        "signals": [_json(item) for item in snapshot.signals],
        "predictions": [_json(item) for item in snapshot.predictions],
        "indicators": _json(snapshot.indicators),
        "research": _json(snapshot.research),
    }


def _candles(frame: pd.DataFrame | None, *, max_candles: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    selected = frame.tail(max(int(max_candles), 1))
    rows: list[dict[str, Any]] = []
    for timestamp, values in selected.iterrows():
        row = {"ts": _time(timestamp)}
        for name in ("open", "high", "low", "close", "volume", "oi"):
            if name in selected.columns:
                row[name] = _number(values[name])
        rows.append(row)
    return rows


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return _time(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _number(value)
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(item) for item in value]
    return str(value)


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() and isinstance(value, (int, np.integer)) else number


def _time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


__all__ = ["serialize_snapshot"]
