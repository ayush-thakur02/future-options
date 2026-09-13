"""The engine: market data in, state out, at two cadences.

A tick updates the forming candle and the tick momentum. A **bar close** is the
expensive moment: features are rebuilt, every strategy is scored, the model
predicts, and the projections that targeted this bar are scored against what
actually happened. In between, the projected path is refreshed on its own clock —
see :mod:`runtime.nowcast` — so the blue candles move every second instead of
once a minute.

One subtlety worth stating. Features are recomputed over a rolling window of
recent bars rather than the entire history, because rebuilding 120 days of
indicators on every bar close would be wasteful. The window is sized so the
longest recursive indicator (EMA-200) has long since converged — after 2,000 bars
the residual influence of the seed value is around 1e-8 — which keeps live
features numerically identical to the ones the model was trained on. A shorter
window would introduce a train/serve skew that silently degrades every
prediction.

The engine holds no feed and no renderer. Both arrive through the kernel, so live
and simulated runs are the same object driven by different sources.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import datetime

import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST, TradingCalendar
from core.types import ForecastCandle, MarketSnapshot, Prediction, Signal, Tick
from kernel import Kernel
from plugins.strategies import CompositeStrategy, StrategyCatalog, StrategyContext

from .conviction import blend, trend_conviction

FEATURE_WINDOW = 2000
# Bars handed to the strategy layer. Strategies look back at most ~100 bars, so
# evaluating all of them across the full feature window is wasted work.
SIGNAL_WINDOW = 600
# Named in the signal panel. More than a handful is unreadable, and these are the
# ones whose disagreement with the ensemble is worth seeing.
HIGHLIGHTED = ("ema_trend", "supertrend", "macd_momentum", "rsi_reversion", "vwap_reversion", "squeeze_release")
# Bars the projection needs before it will draw anything. Two bars of history
# would produce a path, just not a meaningful one.
MIN_PROJECTION_BARS = 15


class Engine:
    """Rolling market state, forecasts on bar close, projected path every second."""

    def __init__(
        self,
        kernel: Kernel,
        strategy: CompositeStrategy | None = None,
        bar_minutes: int | None = None,
        calendar: TradingCalendar | None = None,
        feature_window: int = FEATURE_WINDOW,
        signal_window: int = SIGNAL_WINDOW,
        symbol: str | None = None,
        forecaster=None,
    ) -> None:
        self.kernel = kernel
        self.settings = kernel.settings
        self.symbol = symbol or kernel.settings.symbol
        self.bar_minutes = bar_minutes or kernel.settings.bar_minutes
        self.calendar = calendar or TradingCalendar()
        self.feature_window = feature_window
        self.signal_window = signal_window

        # Everything below arrives through a capability rather than an import, so
        # any of the three can be swapped by changing which plugin is registered.
        self.pipeline = kernel.capability("features")
        self.broker = kernel.build("source:upstox")
        self.predictor = kernel.capability("forecast") if kernel.registry.capabilities().get("forecast") else None
        self.catalog = StrategyCatalog.from_kernel(kernel)
        self.strategy = strategy or self.catalog.ensemble()
        self.aggregator = kernel.build(
            "aggregator:candle_builder", bar_minutes=self.bar_minutes
        )
        # Its own projection, not the kernel's shared one: several engines can
        # run side by side — one per instrument on the options board — and each
        # needs its own momentum, its own path and its own scorecard.
        self.forecaster = forecaster or (
            kernel.new("forecast:projection")
            if kernel.registry.capabilities().get("projection")
            else None
        )
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
        self.projections: list[ForecastCandle] = []
        # The ensemble's view as of the last bar close: the slow half of the
        # conviction the projection runs on.
        self.slow_conviction: float = 0.0
        self.conviction: float = 0.0
        self.projection_refreshes: int = 0
        self._listeners: list[Callable[[MarketSnapshot], None]] = []

    # ------------------------------------------------------------- bootstrap

    def bootstrap(self, history: pd.DataFrame | None = None, days: int = 20) -> Engine:
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
        """Feed one tick. Returns True when a bar closed and state was refreshed.

        Cheap on purpose: this runs at tick rate, and on an index feed that can be
        thousands of updates a minute. Only the anchor price and the momentum
        estimate move here; everything expensive waits for the bar close.
        """
        self.tick_count += 1
        self.last_price = tick.ltp or tick.mid
        if tick.close_prev:
            self.prev_close = tick.close_prev
        if self.forecaster is not None:
            self.forecaster.on_tick(tick)

        closed = self.aggregator.on_tick(tick)
        if closed is None:
            return False

        self._append_closed_bar(closed)
        self._recompute()
        # Score whatever the projections said about the bar that just closed,
        # then redraw: a bar close changes the volatility, the trend and the
        # ensemble view all at once, so the path must not wait for the next tick
        # of the refresh clock.
        if self.forecaster is not None:
            self.forecaster.observe_bar(closed["ts"], float(closed["close"]))
            self.refresh_projection()
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
            self.features = self.pipeline.build(self.history, bar_minutes=self.bar_minutes)
        except Exception as exc:  # noqa: BLE001
            self.status = f"feature error: {exc}"
            return

        context = StrategyContext(bars=self.history, features=self.features)
        self.signals = self._collect_signals(context)
        self.predictions = self._collect_predictions()
        self.regime = self._current_regime()
        self.slow_conviction = self._ensemble_view(context)
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

        for name in HIGHLIGHTED:
            try:
                signal = self.catalog.get(name)._latest_signal(context, threshold=0.15)
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
            labels = self.pipeline.regime(self.features, window=100)
            value = labels.dropna()
            return str(value.iloc[-1]) if len(value) else "unknown"
        except Exception:
            return "unknown"

    # ------------------------------------------------------------ projection

    def refresh_projection(self) -> bool:
        """Rebuild the projected path from the live price and the current view.

        Called on a fixed cadence rather than per tick: the anchor and the tick
        momentum already move with the tape, and recomputing the whole path
        thousands of times a minute to move it by a hundredth of a point is work
        nobody can see. Returns whether a path was produced.
        """
        if self.forecaster is None:
            return False

        bars = self.market_bars()
        if len(bars) < MIN_PROJECTION_BARS:
            return False

        anchor = float(self.last_price or bars["close"].iloc[-1])
        fast = trend_conviction(bars)
        self.conviction = blend(self.slow_conviction, fast)

        # Stamped from the data's own clock, not the wall clock. Live and replay
        # coincide (ticks arrive at the current time), but a snapshot of a cached
        # session would otherwise draw bars at 15:29 and projections at 20:34 —
        # a chart that contradicts itself about when "now" is.
        self.projections = self.forecaster.refresh(
            bars=bars,
            anchor=anchor,
            conviction=self.conviction,
            now=bars.index[-1].to_pydatetime(),
            bar_minutes=self.bar_minutes,
        )
        self.projection_refreshes += 1
        return bool(self.projections)

    def market_bars(self) -> pd.DataFrame:
        """Closed bars plus the forming one — what the projection is drawn from.

        The forming bar matters: without it a projection taken 20 seconds into a
        new bar would ignore the twenty seconds of price that have already
        happened, and start from the previous close instead of the current price.
        """
        forming = self.aggregator.snapshot_frame(include_current=True)
        if forming.empty:
            return self.history
        if self.history.empty:
            return forming
        overlap = self.history[~self.history.index.isin(forming.index)]
        merged = pd.concat([overlap, forming]).sort_index()
        return merged.tail(self.feature_window)

    def _ensemble_view(self, context: StrategyContext) -> float:
        """The blended strategy score at the latest bar, in [-1, 1]."""
        try:
            scores = self.strategy.score(context).dropna()
        except Exception:  # noqa: BLE001 — an unusable ensemble means no view
            return 0.0
        return float(scores.iloc[-1]) if len(scores) else 0.0

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> MarketSnapshot:
        # History plus the forming bar. The aggregator alone would show a single
        # candle the moment the first tick of a session arrived, because it starts
        # empty and the history lives beside it.
        bars = self.market_bars()
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
            symbol=self.symbol,
            last_price=self.last_price,
            prev_close=self.prev_close,
            candles=bars,
            signals=self.signals,
            predictions=self.predictions,
            regime=self.regime,
            indicators=indicators,
            projections=self.projections,
            conviction=self.conviction,
            projection_ts=self.forecaster.updated_at if self.forecaster is not None else None,
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

    def publish(self) -> None:
        """Hand the current state to every listener. Driven by the refresh clock."""
        self._notify()

    # ----------------------------------------------------------------- feeds

    async def run_live(self, instrument_keys: list[str] | None = None) -> None:
        """Stream from the broker's feed until interrupted."""
        feed = self.broker.feed(
            on_tick=self.on_tick,
            on_status=self._on_feed_status,
            instrument_keys=instrument_keys,
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
