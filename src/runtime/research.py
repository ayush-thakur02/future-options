"""Causal online learning and an append/update audit trail for research runs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor


class OnlineLearner:
    """Incremental return regressions, one model per completed-bar horizon.

    Inputs and targets use fixed ATR scaling, so no full-sample scaler leaks
    future distribution information. A sample becomes eligible only at its
    target's close. Gaps and overnight labels are excluded; checkpoints remember
    the last target learned per horizon to prevent fitting it again on restart.
    """

    VERSION = 1
    MIN_SAMPLES = 80

    def __init__(self, bar_minutes: int, horizons=(1, 2, 3, 4)):
        self.bar_minutes = bar_minutes
        self.horizons = tuple(horizons)
        self.models = {h: SGDRegressor(loss="huber", epsilon=0.25, alpha=0.01,
                                      learning_rate="constant", eta0=0.005,
                                      shuffle=False, random_state=42 + h)
                       for h in self.horizons}
        self.samples = {h: 0 for h in self.horizons}
        self.through = {h: None for h in self.horizons}
        self.live_updates = 0
        self.path: Path | None = None
        self.version = self.VERSION

    @staticmethod
    def matrix(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        atr = features["atr_norm"].clip(lower=1e-5)
        columns = [features.get(k, pd.Series(0.0, index=features.index)) / atr
                   for k in ("ret_1", "ret_3", "ret_5", "ema_dist_9", "ema_dist_21", "macd_hist")]
        columns += [(features["rsi_14"] - 50) / 50, features["adx_14"] / 50,
                    features["bb_pct_b"] * 2 - 1, features["efficiency_ratio_10"]]
        x = np.column_stack(columns)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(atr.to_numpy())
        return np.clip(np.nan_to_num(x), -5, 5), valid

    def update(self, bars: pd.DataFrame, features: pd.DataFrame, live: bool = True) -> int:
        if features.empty or "atr_norm" not in features:
            return 0
        x, valid = self.matrix(features)
        close = bars["close"].reindex(features.index)
        atr = features["atr_norm"].clip(lower=1e-5)
        times = features.index
        learned = 0
        for h in self.horizons:
            targets = pd.Series(times, index=times).shift(-h)
            contiguous = (targets - pd.Series(times, index=times)) == pd.Timedelta(minutes=h * self.bar_minutes)
            same_day = targets.dt.date == times.date
            y = (close.shift(-h) / close - 1) / atr
            mask = valid & contiguous.to_numpy() & same_day.to_numpy() & np.isfinite(y.to_numpy())
            if self.through[h] is not None:
                mask &= (targets > self.through[h]).to_numpy()
            count = int(mask.sum())
            if count:
                self.models[h].partial_fit(x[mask], np.clip(y.to_numpy()[mask], -5, 5))
                self.samples[h] += count
                self.through[h] = targets[mask].iloc[-1]
                learned += count
        if live:
            self.live_updates += learned
        return learned

    def moves(self, features: pd.DataFrame) -> dict[int, float]:
        if features.empty:
            return {}
        x, valid = self.matrix(features.tail(1))
        if not valid[-1]:
            return {}
        atr = max(float(features["atr_norm"].iloc[-1]), 1e-5)
        return {h: float(np.clip(model.predict(x)[0], -3, 3)) * atr
                for h, model in self.models.items() if self.samples[h] >= self.MIN_SAMPLES}

    def attach(self, directory: Path, identity: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (hashlib.sha256(identity.encode()).hexdigest()[:24] + ".joblib")
        if path.exists():
            previous = joblib.load(path)
            if (isinstance(previous, OnlineLearner) and previous.version == self.VERSION
                    and previous.bar_minutes == self.bar_minutes and previous.horizons == self.horizons):
                self.models, self.samples, self.through = previous.models, previous.samples, previous.through
        self.path = path
        self.live_updates = 0

    def save(self) -> None:
        if self.path is None:
            return
        fd, name = tempfile.mkstemp(prefix="online-", dir=self.path.parent)
        os.close(fd)
        try:
            joblib.dump(self, name)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def describe(self) -> dict:
        return {"samples": dict(self.samples), "live_updates": self.live_updates,
                "ready": self.samples[1] >= self.MIN_SAMPLES,
                "model": "online SGD + rules", "max_ml_weight": 0.25}


class ResearchJournal:
    """Durable first-issued forecasts, misses, missing bars, and context.

    Each engine owns one connection; callers serialize access with its state lock.
    WAL and a busy timeout let all three instrument workers share the database.
    """

    def __init__(self, path: Path, instrument: str, source: str, timeframe: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=20, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS forecasts (
            run TEXT, instrument TEXT, source TEXT, timeframe INTEGER,
            target TEXT, horizon INTEGER, issued_at TEXT, anchor REAL,
            predicted_close REAL, predicted_high REAL, predicted_low REAL,
            actual_close REAL, hit INTEGER, error_bps REAL, status TEXT,
            context TEXT, PRIMARY KEY(run,instrument,target,horizon))""")
        self.run = uuid.uuid4().hex
        self.instrument, self.source, self.timeframe = instrument, source, timeframe

    def issued(self, candle, anchor: float, issued_at, context: dict) -> None:
        self.db.execute("""INSERT OR IGNORE INTO forecasts VALUES
            (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,'pending',?)""",
            (self.run, self.instrument, self.source, self.timeframe, str(candle.ts), candle.horizon,
             str(issued_at), anchor, candle.close, candle.high, candle.low, json.dumps(context, default=str)))
        self.db.commit()

    def scored(self, score) -> None:
        self.db.execute("""UPDATE forecasts SET actual_close=?,hit=?,error_bps=?,status=?
            WHERE run=? AND instrument=? AND target=? AND horizon=?""",
            (score.actual_close, int(score.hit), score.error_bps, "hit" if score.hit else "miss",
             self.run, self.instrument, str(score.target), score.horizon))
        self.db.commit()

    def missing(self, target, horizon) -> None:
        self.db.execute("""UPDATE forecasts SET status='missing_bar'
            WHERE run=? AND instrument=? AND target=? AND horizon=?""",
            (self.run, self.instrument, str(target), horizon))
        self.db.commit()

    def close(self) -> None:
        self.db.execute("UPDATE forecasts SET status='unresolved' WHERE run=? AND status='pending'", (self.run,))
        self.db.commit()
        self.db.close()


def recent_scores(tracker) -> list[dict]:
    return [asdict(score) for score in tracker.recent(6)]
