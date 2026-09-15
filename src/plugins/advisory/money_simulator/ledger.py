"""The simulator's memory: closed trades, wallet balances, and the equity path.

A live run is a series of sessions, not one process. Without this, every restart
would hand each leg a fresh 25,000 and the reserve would look untouched no matter
how much had been spent — which would make the simulator's headline number a
property of how often the laptop was rebooted.

Wallet state is therefore written on every close rather than at shutdown, so a
kill -9 costs at most the position that was open, and that position is recorded
as abandoned rather than quietly rewritten.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .wallet import Reserve, Wallet


@dataclass(slots=True)
class ClosedTrade:
    """One completed round trip, in the units the reader cares about."""

    leg: str
    kind: str
    instrument: str
    side: str
    units: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    gross: float
    costs: float
    net: float
    return_bps: float
    exit_reason: str = ""
    reason: str = ""
    view: float = 0.0
    edge_bps: float = 0.0

    @property
    def won(self) -> bool:
        return self.net > 0

    def as_row(self) -> dict:
        return {
            "leg": self.leg,
            "kind": self.kind,
            "side": self.side,
            "units": self.units,
            "entry_ts": self.entry_ts.isoformat(),
            "entry_price": round(self.entry_price, 2),
            "exit_ts": self.exit_ts.isoformat(),
            "exit_price": round(self.exit_price, 2),
            "gross": round(self.gross, 2),
            "costs": round(self.costs, 2),
            "net": round(self.net, 2),
            "return_bps": round(self.return_bps, 2),
            "exit_reason": self.exit_reason,
            "reason": self.reason,
            "view": round(self.view, 4),
            "edge_bps": round(self.edge_bps, 3),
        }


@dataclass
class RestoredState:
    """Whatever the last run left behind."""

    wallets: dict[str, Wallet] = field(default_factory=dict)
    reserve: Reserve | None = None
    trades: int = 0


class SimulationLedger:
    """SQLite-backed history for the money simulator."""

    def __init__(self, path: Path | str, mode: str = "simulation") -> None:
        self.path = Path(path)
        self.mode = str(mode)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ write

    def record(self, trade: ClosedTrade) -> None:
        with self._lock, self._connect() as database:
            database.execute(
                """INSERT INTO money_trades
                (mode,leg,kind,instrument,side,units,entry_ts,entry_price,exit_ts,exit_price,
                 gross,costs,net,return_bps,exit_reason,reason,view,edge_bps)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    self.mode, trade.leg, trade.kind, trade.instrument, trade.side, int(trade.units),
                    trade.entry_ts.isoformat(), float(trade.entry_price),
                    trade.exit_ts.isoformat(), float(trade.exit_price),
                    float(trade.gross), float(trade.costs), float(trade.net), float(trade.return_bps),
                    trade.exit_reason, trade.reason, float(trade.view), float(trade.edge_bps),
                ),
            )
            database.commit()

    def save_wallets(self, wallets: dict[str, Wallet], reserve: Reserve) -> None:
        """Persist balances. Called after every close, not only at shutdown."""
        with self._lock, self._connect() as database:
            database.executemany(
                """INSERT INTO money_wallets
                (mode,leg,opening,cash,reserve_drawn,realized,costs_paid,trades,wins,losses,
                 best_trade,worst_trade,peak_equity,top_ups,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(mode,leg) DO UPDATE SET
                    cash=excluded.cash, reserve_drawn=excluded.reserve_drawn,
                    realized=excluded.realized, costs_paid=excluded.costs_paid,
                    trades=excluded.trades, wins=excluded.wins, losses=excluded.losses,
                    best_trade=excluded.best_trade, worst_trade=excluded.worst_trade,
                    peak_equity=excluded.peak_equity, top_ups=excluded.top_ups,
                    updated_at=excluded.updated_at""",
                [
                    (
                        self.mode, wallet.label, float(wallet.opening), float(wallet.cash),
                        float(wallet.reserve_drawn), float(wallet.realized), float(wallet.costs_paid),
                        int(wallet.trades), int(wallet.wins), int(wallet.losses),
                        float(wallet.best_trade), float(wallet.worst_trade),
                        float(wallet.peak_equity), int(wallet.top_ups), datetime.now(UTC).isoformat(),
                    )
                    for wallet in wallets.values()
                ],
            )
            database.execute(
                """INSERT INTO money_reserve (mode,opening,remaining,calls,lent,updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(mode) DO UPDATE SET
                    remaining=excluded.remaining, calls=excluded.calls,
                    lent=excluded.lent, updated_at=excluded.updated_at""",
                (
                    self.mode, float(reserve.opening), float(reserve.remaining), int(reserve.calls),
                    json.dumps({leg: round(value, 2) for leg, value in reserve.lent.items()}),
                    datetime.now(UTC).isoformat(),
                ),
            )
            database.commit()

    def record_equity(self, leg: str, equity: float, cash: float, at: datetime) -> None:
        with self._lock, self._connect() as database:
            database.execute(
                "INSERT INTO money_equity (mode,leg,ts,equity,cash) VALUES (?,?,?,?,?)",
                (self.mode, leg, at.isoformat(), float(equity), float(cash)),
            )
            database.commit()

    # ------------------------------------------------------------------- read

    def restore(self, opening: float) -> RestoredState:
        """Reload the last run's balances, if there was one."""
        state = RestoredState()
        if not self.path.exists():
            # Reading a ledger that is not there must not create one. Opening the
            # connection would build the schema, and simply constructing the
            # simulator — which every board does, including in tests — would leave
            # an empty book behind in the data directory.
            return state
        try:
            with self._lock, self._connect() as database:
                rows = database.execute(
                    "SELECT leg,opening,cash,reserve_drawn,realized,costs_paid,trades,wins,losses,"
                    "best_trade,worst_trade,peak_equity,top_ups FROM money_wallets WHERE mode=?",
                    (self.mode,),
                ).fetchall()
                reserve_row = database.execute(
                    "SELECT opening,remaining,calls,lent FROM money_reserve WHERE mode=?", (self.mode,)
                ).fetchone()
                count = database.execute(
                    "SELECT COUNT(*) FROM money_trades WHERE mode=?", (self.mode,)
                ).fetchone()
        except sqlite3.Error:
            return state

        for row in rows:
            state.wallets[str(row[0])] = Wallet(
                label=str(row[0]),
                opening=float(row[1]),
                cash=float(row[2]),
                reserve_drawn=float(row[3]),
                realized=float(row[4]),
                costs_paid=float(row[5]),
                trades=int(row[6]),
                wins=int(row[7]),
                losses=int(row[8]),
                best_trade=float(row[9]),
                worst_trade=float(row[10]),
                peak_equity=float(row[11]),
                top_ups=int(row[12]),
            )
        if reserve_row is not None:
            try:
                lent = {str(k): float(v) for k, v in json.loads(reserve_row[3] or "{}").items()}
            except (TypeError, ValueError):
                lent = {}
            state.reserve = Reserve(
                opening=float(reserve_row[0]),
                remaining=float(reserve_row[1]),
                calls=int(reserve_row[2]),
                lent=lent,
            )
        state.trades = int(count[0]) if count else 0
        return state

    def recent_trades(self, limit: int = 40) -> list[dict]:
        try:
            with self._lock, self._connect() as database:
                rows = database.execute(
                    "SELECT leg,kind,side,units,entry_ts,entry_price,exit_ts,exit_price,gross,costs,"
                    "net,return_bps,exit_reason,reason,view,edge_bps FROM money_trades "
                    "WHERE mode=? ORDER BY id DESC LIMIT ?",
                    (self.mode, max(int(limit), 1)),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "leg": str(row[0]),
                "kind": str(row[1]),
                "side": str(row[2]),
                "units": int(row[3]),
                "entry_ts": str(row[4]),
                "entry_price": float(row[5]),
                "exit_ts": str(row[6]),
                "exit_price": float(row[7]),
                "gross": float(row[8]),
                "costs": float(row[9]),
                "net": float(row[10]),
                "return_bps": float(row[11]),
                "exit_reason": str(row[12]),
                "reason": str(row[13]),
                "view": float(row[14]),
                "edge_bps": float(row[15]),
            }
            for row in rows
        ]

    def totals(self) -> dict:
        """All-time totals across every session this ledger has seen."""
        try:
            with self._lock, self._connect() as database:
                row = database.execute(
                    "SELECT COUNT(*),COALESCE(SUM(net),0),COALESCE(SUM(costs),0),"
                    "COALESCE(SUM(CASE WHEN net>0 THEN 1 ELSE 0 END),0) "
                    "FROM money_trades WHERE mode=?",
                    (self.mode,),
                ).fetchone()
        except sqlite3.Error:
            return {"trades": 0, "net": 0.0, "costs": 0.0, "wins": 0}
        return {
            "trades": int(row[0] or 0),
            "net": round(float(row[1] or 0.0), 2),
            "costs": round(float(row[2] or 0.0), 2),
            "wins": int(row[3] or 0),
        }

    # --------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget everything. Only ever called by an explicit reset."""
        with self._lock, self._connect() as database:
            for table in ("money_trades", "money_wallets", "money_reserve", "money_equity"):
                database.execute(f"DELETE FROM {table} WHERE mode=?", (self.mode,))  # noqa: S608
            database.commit()

    def close(self) -> None:
        """Nothing is held open: every write commits and returns its connection."""

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self.path, timeout=20.0)
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=FULL")
        database.executescript(
            """CREATE TABLE IF NOT EXISTS money_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                leg TEXT NOT NULL,
                kind TEXT NOT NULL,
                instrument TEXT NOT NULL,
                side TEXT NOT NULL,
                units INTEGER NOT NULL,
                entry_ts TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_ts TEXT NOT NULL,
                exit_price REAL NOT NULL,
                gross REAL NOT NULL,
                costs REAL NOT NULL,
                net REAL NOT NULL,
                return_bps REAL NOT NULL,
                exit_reason TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                view REAL NOT NULL DEFAULT 0,
                edge_bps REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS money_trades_mode ON money_trades(mode,id);

            CREATE TABLE IF NOT EXISTS money_wallets (
                mode TEXT NOT NULL,
                leg TEXT NOT NULL,
                opening REAL NOT NULL,
                cash REAL NOT NULL,
                reserve_drawn REAL NOT NULL DEFAULT 0,
                realized REAL NOT NULL DEFAULT 0,
                costs_paid REAL NOT NULL DEFAULT 0,
                trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                best_trade REAL NOT NULL DEFAULT 0,
                worst_trade REAL NOT NULL DEFAULT 0,
                peak_equity REAL NOT NULL DEFAULT 0,
                top_ups INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (mode, leg)
            );

            CREATE TABLE IF NOT EXISTS money_reserve (
                mode TEXT PRIMARY KEY,
                opening REAL NOT NULL,
                remaining REAL NOT NULL,
                calls INTEGER NOT NULL DEFAULT 0,
                lent TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS money_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                leg TEXT NOT NULL,
                ts TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS money_equity_mode ON money_equity(mode,ts);
            """
        )
        return database


__all__ = ["ClosedTrade", "RestoredState", "SimulationLedger"]
