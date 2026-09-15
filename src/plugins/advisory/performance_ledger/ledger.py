"""SQLite ledger for frozen strategy signals and their matured outcomes."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.scoring import conservative_strategy_trust, max_drawdown

from .types import StrategyOutcome, StrategyScorecard


class StrategyPerformanceLedger:
    """Persist first-issued active signals, then score exact target bars.

    A process that misses a target bar marks the signal expired instead of using
    a later price. That makes restart behaviour conservative and prevents a gap
    from becoming a fabricated win or loss.
    """

    def __init__(self, path: Path, min_trust_samples: int = 50) -> None:
        self.path = Path(path)
        self.min_trust_samples = max(int(min_trust_samples), 1)
        self._lock = threading.RLock()

    def issue(
        self,
        signals: Iterable,
        *,
        timestamp: datetime,
        anchor_price: float,
        horizon_min: int,
        instrument: str,
        cost_bps: float,
    ) -> int:
        issued = _utc(timestamp)
        target = issued + timedelta(minutes=int(horizon_min))
        anchor = float(anchor_price)
        if anchor <= 0 or not math.isfinite(anchor):
            return 0

        rows = []
        for signal in signals:
            direction = str(getattr(getattr(signal, "direction", ""), "value", "")).upper()
            if direction not in {"UP", "DOWN"}:
                continue
            strategy = str(getattr(signal, "strategy", ""))
            if not strategy:
                continue
            identity = f"{instrument}|{strategy}|{issued.isoformat()}|{target.isoformat()}"
            rows.append(
                (
                    hashlib.blake2s(identity.encode(), digest_size=16).hexdigest(),
                    strategy,
                    str(instrument),
                    issued.isoformat(),
                    target.isoformat(),
                    int(horizon_min),
                    anchor,
                    direction,
                    float(getattr(signal, "strength", 0.0)),
                    max(float(cost_bps), 0.0),
                )
            )
        if not rows:
            return 0

        with self._lock, self._connect() as database:
            database.executemany(
                """INSERT OR IGNORE INTO strategy_signals
                (id,strategy,instrument,issued_at,target_at,horizon,anchor,direction,strength,cost_bps)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            database.commit()
        return len(rows)

    def observe(
        self, *, timestamp: datetime, price: float, instrument: str
    ) -> list[StrategyOutcome]:
        observed = _utc(timestamp)
        actual = float(price)
        if actual <= 0 or not math.isfinite(actual):
            return []
        outcomes: list[StrategyOutcome] = []
        with self._lock, self._connect() as database:
            rows = database.execute(
                """SELECT id,strategy,target_at,anchor,direction,cost_bps
                FROM strategy_signals WHERE instrument=? AND status='pending' AND target_at<=?
                ORDER BY target_at,strategy""",
                (str(instrument), observed.isoformat()),
            ).fetchall()
            for row in rows:
                target = datetime.fromisoformat(row[2])
                if target != observed:
                    database.execute(
                        "UPDATE strategy_signals SET status='expired',resolved_at=? WHERE id=?",
                        (observed.isoformat(), row[0]),
                    )
                    continue
                realised = (actual / float(row[3]) - 1.0) * 10_000.0
                position = 1.0 if row[4] == "UP" else -1.0
                gross = position * realised
                net = gross - float(row[5])
                hit = gross > 0.0
                database.execute(
                    """UPDATE strategy_signals SET status='scored',resolved_at=?,actual_price=?,
                    hit=?,gross_pnl_bps=?,net_pnl_bps=? WHERE id=?""",
                    (observed.isoformat(), actual, int(hit), gross, net, row[0]),
                )
                outcomes.append(
                    StrategyOutcome(row[1], str(instrument), target, row[4], hit, gross, net)
                )
            database.commit()
        return outcomes

    def scorecards(self, instrument: str | None = None) -> list[StrategyScorecard]:
        where = "WHERE status='scored'"
        params: tuple = ()
        if instrument is not None:
            where += " AND instrument=?"
            params = (str(instrument),)
        with self._lock, self._connect() as database:
            rows = database.execute(
                f"""SELECT strategy,instrument,hit,gross_pnl_bps,net_pnl_bps,cost_bps
                FROM strategy_signals {where} ORDER BY resolved_at,id""",  # noqa: S608
                params,
            ).fetchall()

        grouped: dict[tuple[str, str], list[tuple]] = {}
        for row in rows:
            grouped.setdefault((row[0], row[1]), []).append(row)
        return [self._score(key, values) for key, values in sorted(grouped.items())]

    def counts(self, instrument: str | None = None) -> dict[str, int]:
        where = ""
        params: tuple = ()
        if instrument is not None:
            where = " WHERE instrument=?"
            params = (str(instrument),)
        with self._lock, self._connect() as database:
            rows = database.execute(
                f"SELECT status,COUNT(*) FROM strategy_signals{where} GROUP BY status",  # noqa: S608
                params,
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def _score(
        self, key: tuple[str, str], rows: list[tuple]
    ) -> StrategyScorecard:
        pnl = [float(row[4]) for row in rows]
        gross = [float(row[3]) for row in rows]
        samples = len(rows)
        hits = sum(bool(row[2]) for row in rows)
        wins = sum(value > 0 for value in pnl)
        losses = sum(value < 0 for value in pnl)
        accuracy = hits / samples if samples else 0.0
        profit = sum(value for value in pnl if value > 0)
        loss = -sum(value for value in pnl if value < 0)
        mean = sum(pnl) / samples if samples else 0.0
        drawdown = max_drawdown(pnl)
        # Shared with the warm-up replay, which has to reach the same number from
        # a running total rather than from the stored rows.
        trust = conservative_strategy_trust(
            hits=hits,
            samples=samples,
            net_pnl_sum=sum(pnl),
            min_trust_samples=self.min_trust_samples,
        )
        return StrategyScorecard(
            strategy=key[0],
            instrument=key[1],
            samples=samples,
            hits=hits,
            accuracy=accuracy,
            wins=wins,
            losses=losses,
            gross_pnl_bps=sum(gross),
            cost_paid_bps=sum(float(row[5]) for row in rows),
            profit_bps=profit,
            loss_bps=loss,
            net_pnl_bps=sum(pnl),
            mean_net_pnl_bps=mean,
            max_drawdown_bps=drawdown,
            trust_score=trust,
        )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self.path, timeout=20.0)
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=FULL")
        database.executescript(
            """CREATE TABLE IF NOT EXISTS strategy_signals (
                id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                instrument TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                target_at TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                anchor REAL NOT NULL,
                direction TEXT NOT NULL,
                strength REAL NOT NULL,
                cost_bps REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT,
                actual_price REAL,
                hit INTEGER,
                gross_pnl_bps REAL,
                net_pnl_bps REAL
            );
            CREATE INDEX IF NOT EXISTS strategy_due
                ON strategy_signals(instrument,status,target_at);
            CREATE INDEX IF NOT EXISTS strategy_name
                ON strategy_signals(strategy,instrument,status);
            """
        )
        return database


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


__all__ = ["StrategyPerformanceLedger"]
