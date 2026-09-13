"""Live inference engine.

Wires the pieces together: market data in, calibrated forecasts and strategy
signals out.

One subtlety worth stating. Features are recomputed over a rolling window of
recent bars rather than the entire history, because rebuilding 120 days of
indicators on every bar close would be wasteful. The window is sized so the
longest recursive indicator (EMA-200) has long since converged — after 2,000 bars
the residual influence of the seed value is around 1e-8 — which keeps live
features numerically identical to the ones the model was trained on. A shorter
window would introduce a train/serve skew that silently degrades every
prediction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import datetime

import pandas as pd

from ..core.calendar import IST, TradingCalendar
from ..core.settings import Settings
from ..core.types import MarketSnapshot, Prediction, Signal, Tick
from ..data.aggregator import CandleAggregator
from ..data.store import normalize_candles
from ..features import build_features
from ..features.context import classify_regime
from ..strategies import CompositeStrategy, StrategyContext, build_strategy, default_ensemble

FEATURE_WINDOW = 2000
# Bars handed to the strategy layer. Strategies look back at most ~100 bars, so
# evaluating all 24 of them across the full feature window is wasted work.
SIGNAL_WINDOW = 600


class LiveEngine:
    """Maintains rolling market state and produces forecasts on each bar close."""

    def __init__(
        self,
        settings: Settings,
        predictor=None,
        strategy: CompositeStrategy | None = None,
        bar_minutes: int | None = None,
        calendar: TradingCalendar | None = None,
        feature_window: int = FEATURE_WINDOW,
        signal_window: int = SIGNAL_WINDOW,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.strategy = strategy or default_ensemble()
        self.bar_minutes = bar_minutes or settings.bar_minutes
        self.calendar = calendar or TradingCalendar()
        self.feature_window = feature_window
        self.signal_window = signal_window

        self.aggregator = CandleAggregator(bar_minutes=self.bar_minutes)
        self.history: pd.DataFrame = pd.DataFrame()
        self.features: pd.DataFrame = pd.DataFrame()
        self.signals: list[Signal] = []
        self.predictions: list[Prediction] = []
        self.last_price: float = 0.0
        self.prev_close: float = 0.0
        self.regime: str = "unknown"
        self.tick_count: int = 0
        self.bar_count: int = 0
        self.status: str = "initialising"
        self._listeners: list[Callable[[MarketSnapshot], None]] = []

    # ------------------------------------------------------------- bootstrap

    def bootstrap(self, history: pd.DataFrame | None = None, days: int = 20) -> LiveEngine:
        """Seed the engine with recent bars so indicators are warm immediately."""
        if history is not None:
            self.history = normalize_candles(history)
        if self.history.empty:
            self.status = "no history"
            return self

        self.history = self.history.tail(self.feature_window).copy()
        last_day = self.history.index[-1].normalize()
        prior = self.history[self.history.index.normalize() < last_day]
        self.prev_close = float(prior["close"].iloc[-1]) if len(prior) else float(self.history["close"].iloc[0])
        self.last_price = float(self.history["close"].iloc[-1])
        self.bar_count = len(self.history)

        self._recompute()
        self.status = "ready"
        return self

    # ----------------------------------------------------------------- ticks

    def on_tick(self, tick: Tick) -> bool:
        """Feed one tick. Returns True when a bar closed and state was refreshed."""
        self.tick_count += 1
        self.last_price = tick.ltp or tick.mid
        if tick.close_prev:
            self.prev_close = tick.close_prev

        closed = self.aggregator.on_tick(tick)
        if closed is None:
            return False

        self._append_closed_bar(closed)
        self._recompute()
        self._notify()
        return True

    def _append_closed_bar(self, bar: dict) -> None:
        row = pd.DataFrame([bar]).set_index("ts")
        row.index = pd.DatetimeIndex(row.index).tz_convert(IST)
        self.history = pd.concat([self.history, row])
        self.history = self.history[~self.history.index.duplicated(keep="last")]
        self.history = self.history.sort_index().tail(self.feature_window)
        self.bar_count = len(self.history)

    # -------------------------------------------------------------- features

    def _recompute(self) -> None:
        """Rebuild features, strategy signals, and forecasts from current state."""
        if len(self.history) < 60:
            self.status = "warming up"
            return

        try:
            self.features = build_features(
                self.history,
                bar_minutes=self.bar_minutes,
                expiry_weekday=self.settings.expiry_weekday,
            )
        except Exception as exc:  # noqa: BLE001
            self.status = f"feature error: {exc}"
            return

        context = StrategyContext(bars=self.history, features=self.features)
        self.signals = self._collect_signals(context)
        self.predictions = self._collect_predictions()
        self.regime = self._current_regime()
        self.status = "live"

    def _collect_signals(self, context: StrategyContext) -> list[Signal]:
        signals: list[Signal] = []

        # Strategies only report their latest reading, so they are evaluated on a
        # bounded tail rather than the whole feature window. The indicators they
        # consume are already computed over full history, and the longest
        # lookback any strategy applies internally is about 100 bars. Without
        # this, evaluating all 24 strategies across 2000 bars dominated the live
        # loop and made the offline replay unwatchable.
        if len(context.features) > self.signal_window:
            context = StrategyContext(
                bars=context.bars.tail(self.signal_window),
                features=context.features.tail(self.signal_window),
                extras=context.extras,
            )

        try:
            ensemble_signal = self.strategy._latest_signal(context, threshold=0.05)
            if ensemble_signal is not None:
                ensemble_signal.strategy = "ensemble"
                signals.append(ensemble_signal)
        except Exception:
            pass

        for name in ("ema_trend", "supertrend", "macd_momentum", "rsi_reversion", "vwap_reversion", "squeeze_release"):
            try:
                signal = build_strategy(name)._latest_signal(context, threshold=0.15)
                if signal is not None:
                    signals.append(signal)
            except Exception:
                continue
        return signals

    def _collect_predictions(self) -> list[Prediction]:
        if self.predictor is None or not getattr(self.predictor, "is_ready", False):
            return []
        try:
            return self.predictor.predict(self.features)
        except Exception:
            return []

    def _current_regime(self) -> str:
        if self.features.empty:
            return "unknown"
        try:
            labels = classify_regime(self.features, window=100)
            value = labels.dropna()
            return str(value.iloc[-1]) if len(value) else "unknown"
        except Exception:
            return "unknown"

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> MarketSnapshot:
        bars = self.aggregator.snapshot_frame(include_current=True)
        if bars.empty:
            bars = self.history.tail(120)

        indicators = {}
        if not self.features.empty:
            last = self.features.iloc[-1]
            for key in ("rsi_14", "adx_14", "atr_norm", "macd_hist", "stoch_k", "bb_pct_b", "vwap_dist", "efficiency_ratio_10"):
                if key in self.features.columns:
                    value = last[key]
                    indicators[key] = float(value) if pd.notna(value) else float("nan")

        return MarketSnapshot(
            ts=datetime.now(IST),
            symbol=self.settings.symbol,
            last_price=self.last_price,
            prev_close=self.prev_close,
            candles=bars,
            signals=self.signals,
            predictions=self.predictions,
            regime=self.regime,
            indicators=indicators,
        )

    # -------------------------------------------------------------- listeners

    def subscribe(self, listener: Callable[[MarketSnapshot], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        if not self._listeners:
            return
        snapshot = self.snapshot()
        for listener in self._listeners:
            try:
                listener(snapshot)
            except Exception:
                continue

    # ----------------------------------------------------------------- feeds

    async def run_live(self, access_token: str | None = None, instrument_keys: list[str] | None = None) -> None:
        """Stream from the Upstox WebSocket until interrupted."""
        from ..data.upstox_feed import UpstoxFeed

        token = access_token or self.settings.credentials.access_token
        if not token:
            raise RuntimeError("live feed requires an Upstox access token; run `niftypulse login`")

        feed = UpstoxFeed(
            access_token=token,
            instrument_keys=instrument_keys or [self.settings.instrument_key],
            on_tick=self.on_tick,
            on_status=self._on_feed_status,
            mode="full",
        )
        self.status = "connecting"
        await feed.run()

    def _on_feed_status(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "connected":
            self.status = "connected"
        elif kind == "disconnected":
            self.status = "reconnecting"
        elif kind == "market_info":
            segments = payload.get("segments", {})
            self.status = segments.get("NSE_INDEX", self.status)
        elif kind == "connection_error":
            self.status = f"feed error: {payload.get('error', '')[:40]}"

    def replay(self, ticks: pd.DataFrame, bar_minutes: int | None = None) -> Iterator[MarketSnapshot]:
        """Drive the engine from a tick frame, yielding a snapshot per bar close."""
        for row in ticks.itertuples():
            tick = Tick(
                ts=row.Index,
                ltp=float(row.ltp),
                ltq=int(getattr(row, "ltq", 0)),
                close_prev=float(getattr(row, "close_prev", 0.0)),
                bid_p=float(getattr(row, "bid_p", 0.0)),
                bid_q=int(getattr(row, "bid_q", 0)),
                ask_p=float(getattr(row, "ask_p", 0.0)),
                ask_q=int(getattr(row, "ask_q", 0)),
                volume_traded=int(getattr(row, "volume_traded", 0)),
                avg_traded_price=float(getattr(row, "avg_traded_price", 0.0)),
                open_interest=float(getattr(row, "open_interest", 0.0)),
                total_buy_qty=float(getattr(row, "total_buy_qty", 0.0)),
                total_sell_qty=float(getattr(row, "total_sell_qty", 0.0)),
            )
            if self.on_tick(tick):
                yield self.snapshot()

    async def run_replay(self, ticks: pd.DataFrame, delay: float = 0.02) -> None:
        """Async replay for the dashboard demo."""
        for snapshot in self.replay(ticks):
            for listener in self._listeners:
                try:
                    listener(snapshot)
                except Exception:
                    continue
            await asyncio.sleep(delay)
