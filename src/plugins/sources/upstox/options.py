"""Exchange option contracts, candles and cached chain context for a live board."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST, SESSION_CLOSE, SESSION_OPEN
from plugins.sources import DataUnavailable
from plugins.sources.option_chain.chain import (
    LegQuote,
    OptionChain,
    annualised_vol,
    trading_minutes_to,
)


@dataclass
class LiveLeg:
    kind: str
    strike: float
    instrument_key: str
    symbol: str
    lot_size: int
    bars: pd.DataFrame
    greeks: dict


class LiveOptionSource:
    """Pin a real ATM pair for the session; never manufacture option prices.

    The pinned contract identity keeps its learning and score history comparable.
    Restart/reconnect the dashboard to select a new ATM pair or expiry.
    """

    def __init__(self, broker, executor: Executor, days: int = 15) -> None:
        self.broker = broker
        self.executor = executor
        self.days = days
        self.settings = broker.settings
        self.contracts: dict[str, dict] = {}
        self._chain: OptionChain | None = None
        self._expiry: datetime | None = None
        self.strike_step = 50.0
        self.realised_vol = 0.0

    def expiry(self):
        return self._expiry

    def select(self, index_bars: pd.DataFrame) -> dict[str, LiveLeg]:
        rest = self.broker.rest()
        now = datetime.now(IST)
        rows = rest.option_contracts(self.settings.instrument_key)
        valid = [c for c in rows if c.get("instrument_type") in {"CE", "PE"}
                 and datetime.combine(pd.Timestamp(c["expiry"]).date(), SESSION_CLOSE, IST) > now]
        if not valid:
            raise DataUnavailable("Upstox returned no unexpired call/put contracts for this index.")
        expiry = min(pd.Timestamp(c["expiry"]).date() for c in valid)
        self._expiry = datetime.combine(expiry, SESSION_CLOSE, IST)
        valid = [c for c in valid if pd.Timestamp(c["expiry"]).date() == expiry]
        quotes = rest.ltp(self.settings.instrument_key)
        spot = float(next(iter(quotes.values()))["last_price"])
        paired = set(float(c["strike_price"]) for c in valid if c["instrument_type"] == "CE") & set(
            float(c["strike_price"]) for c in valid if c["instrument_type"] == "PE")
        if not paired:
            raise DataUnavailable("No matching call/put strike for the nearest expiry.")
        strike = min(paired, key=lambda k: abs(k - spot))
        strikes = sorted(paired)
        self.strike_step = min((b - a for a, b in zip(strikes, strikes[1:], strict=False)), default=50.0)
        self.contracts = {c["instrument_type"]: c for c in valid if float(c["strike_price"]) == strike}
        self.realised_vol = annualised_vol(index_bars["close"])
        self.refresh_chain()
        futures = {kind: self.executor.submit(self._load, kind, contract)
                   for kind, contract in self.contracts.items()}
        return {kind: future.result() for kind, future in futures.items()}

    def _load(self, kind: str, contract: dict) -> LiveLeg:
        rest = self.broker.rest()
        today = datetime.now(IST).date()
        key = contract["instrument_key"]
        historical = rest.fetch_history_range(key, "minutes", self.settings.bar_minutes,
                                             today - timedelta(days=self.days), today)
        intraday = rest.fetch_intraday(key, "minutes", self.settings.bar_minutes)
        bars = normalize_candles(pd.concat([historical, intraday]))
        # Only completed bars may be used for fitting and outcome labels.
        cutoff = pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=self.settings.bar_minutes)
        bars = bars[(bars.index <= cutoff) & (bars.index.time >= SESSION_OPEN) & (bars.index.time < SESSION_CLOSE)]
        quote = self._chain.at(float(contract["strike_price"]), kind)
        greeks = quote.as_row() if quote is not None else {}
        greeks.update({"lot_size": int(contract.get("lot_size", 1)), "expiry": str(self._expiry.date()),
                       "source": "Upstox", "realised_vol": self.realised_vol,
                       "strike_step": self.strike_step})
        return LiveLeg(kind, float(contract["strike_price"]), key, contract.get("trading_symbol", key),
                       int(contract.get("lot_size", 1)), bars, greeks)

    def refresh_chain(self) -> None:
        rows = self.broker.rest().option_chain(self.settings.instrument_key, str(self._expiry.date()))
        if not rows:
            raise DataUnavailable("Upstox returned an empty option chain.")
        now = datetime.now(IST)
        minutes = trading_minutes_to(now, self._expiry)
        spot = float(rows[0].get("underlying_spot_price", 0))
        quotes = []
        for row in rows:
            for kind, side in (("CE", "call_options"), ("PE", "put_options")):
                data = row.get(side) or {}
                market = data.get("market_data") or {}
                greek = data.get("option_greeks") or {}
                quotes.append(LegQuote(
                    kind=kind, strike=float(row["strike_price"]), spot=spot,
                    premium=float(market.get("ltp") or 0), delta=float(greek.get("delta") or 0),
                    gamma=float(greek.get("gamma") or 0), vega=float(greek.get("vega") or 0),
                    # Upstox theta is per calendar day; show the unit explicitly.
                    theta_per_minute=float(greek.get("theta") or 0) / 1440.0,
                    iv=float(greek.get("iv") or 0) / 100.0,
                    oi=float(market.get("oi") or 0), volume=float(market.get("volume") or 0),
                    minutes_to_expiry=minutes,
                ))
        self._chain = OptionChain(ts=now, spot=spot, expiry=self._expiry,
                                  minutes_to_expiry=minutes, strike_step=int(self.strike_step),
                                  atm_strike=min((q.strike for q in quotes), key=lambda k: abs(k - spot)),
                                  quotes={(q.strike, q.kind): q for q in quotes})

    def chain(self, *args, **kwargs) -> OptionChain:
        return self._chain
