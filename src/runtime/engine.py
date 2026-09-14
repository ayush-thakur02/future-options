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
import hashlib
import itertools
import math
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import wraps
from threading import RLock

import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST, TradingCalendar
from core.types import Direction, ForecastCandle, MarketSnapshot, Prediction, Signal, Tick
from kernel import Kernel
from plugins.strategies import CompositeStrategy, StrategyCatalog, StrategyContext

from .conviction import blend, trend_conviction
from .research import OnlineLearner, ResearchJournal, recent_scores

FEATURE_WINDOW = 2000
# Bars handed to the strategy layer. Strategies look back at most ~100 bars, so
# evaluating all of them across the full feature window is wasted work.
SIGNAL_WINDOW = 600
# Bars the projection needs before it will draw anything. Two bars of history
# would produce a path, just not a meaningful one.
MIN_PROJECTION_BARS = 15


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


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
        instrument_key: str | None = None,
        source: str = "simulation",
        anchor_provider: Callable[[], pd.Series] | None = None,
    ) -> None:
        self.kernel = kernel
        self._lock = RLock()
        self.settings = kernel.settings
        self.instrument_key = instrument_key or self.settings.instrument_key
        self.source = source
        self.anchor_provider = anchor_provider
        self.last_tick_ts = None
        self.last_quote_ts = None
        self.market_status = None
        self.compute_ms = 0.0
        self.rejected_ticks = 0
        self.spread = 0.0
        self.online = OnlineLearner(bar_minutes or self.settings.bar_minutes) if self.settings.online_learning else None
        self.rule_stats = {}
        self._latest_scores = {}
        self.ai_signals = {}
        self.online_lab = None
        self.performance = None
        self.symbol = symbol or kernel.settings.symbol
        self.bar_minutes = bar_minutes or kernel.settings.bar_minutes
        self.calendar = calendar or TradingCalendar()
        self.feature_window = feature_window
        self.signal_window = signal_window

        # Everything below arrives through a capability rather than an import, so
        # any of the three can be swapped by changing which plugin is registered.
        self.pipeline = kernel.capability("features")
        self.broker = kernel.build("source:upstox")
        # Batch artifacts are trained for the index. Never apply them to premiums
        # or to a different timeframe without matching training metadata.
        self.predictor = kernel.capability("forecast") if (self.instrument_key == self.settings.instrument_key
            and kernel.registry.capabilities().get("forecast")) else None
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
        if self.forecaster is not None:
            self.forecaster.tracker.freeze_first = True

    # ------------------------------------------------------------- bootstrap

    @synchronized
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
        if self.online is not None:
            self.online.update(self.history, self.features, live=False)
        self.status = "ready"
        return self

    # ----------------------------------------------------------------- ticks

    @synchronized
    def on_tick(self, tick: Tick) -> bool:
        """Feed one tick. Returns True when a bar closed and state was refreshed.

        Cheap on purpose: this runs at tick rate, and on an index feed that can be
        thousands of updates a minute. Only the anchor price and the momentum
        estimate move here; everything expensive waits for the bar close.
        """
        if not pd.notna(tick.ltp) or tick.ltp <= 0:
            self.rejected_ticks += 1
            return False
        if self.source == "Upstox":
            if self.last_quote_ts is not None and tick.ts < self.last_quote_ts:
                self.rejected_ticks += 1
                return False
            self.last_quote_ts = tick.ts
            self.last_price = tick.ltp
            if tick.close_prev:
                self.prev_close = tick.close_prev
            if (self.market_status not in {None, "NORMAL_OPEN"} or not self.calendar.is_open(tick.ts)
                    or (self.last_tick_ts and tick.ts < self.last_tick_ts)):
                self.rejected_ticks += 1
                return False
            if not self.history.empty and pd.Timestamp(tick.ts) < self.history.index[-1] + pd.Timedelta(minutes=self.bar_minutes):
                self.rejected_ticks += 1
                return False
        self.last_tick_ts = tick.ts
        self.spread = max(tick.spread, 0.0)
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
        self._observe_research(closed)
        self._recompute(issue_research=True)
        if self.online is not None:
            self.online.update(self.history, self.features)
            self.online.save()
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

    def _recompute(self, issue_research: bool = False) -> None:
        """Rebuild features, strategy signals, and forecasts from current state."""
        if len(self.history) < 60:
            self.status = "warming up"
            return

        started = time.perf_counter()
        try:
            self.features = self.pipeline.build(self.history, bar_minutes=self.bar_minutes)
        except Exception as exc:  # noqa: BLE001
            self.status = f"feature error: {exc}"
            return

        context = StrategyContext(
            bars=self.history,
            features=self.features,
            extras=self._strategy_extras(),
        )
        self.signals = self._collect_signals(context)
        if issue_research:
            self._issue_research()
        self.predictions = self._collect_predictions()
        self.regime = self._current_regime()
        rule_view = self._ensemble_view(context)
        ai_view = self._ai_view()
        self.slow_conviction = 0.75 * rule_view + 0.25 * ai_view
        self.status = "live"
        self.compute_ms = (time.perf_counter() - started) * 1000

    def _collect_signals(self, context: StrategyContext) -> list[Signal]:
        signals: list[Signal] = []

        # Strategies only report their latest reading, so they are evaluated on a
        # bounded tail rather than the whole feature window. The indicators they
        # consume are already computed over full history, and the longest
        # lookback any strategy applies internally is about 100 bars. Without
        # this, evaluating all strategies across 2000 bars dominated the live
        # loop and made the offline replay unwatchable.
        if len(context.features) > self.signal_window:
            context = StrategyContext(
                bars=context.bars.tail(self.signal_window),
                features=context.features.tail(self.signal_window),
                extras=context.extras,
            )

        self._latest_scores = {}
        for rule in self.catalog.all():
            name = rule.name
            state = "WAIT"
            value = 0.0
            reason = "conditions not met"
            try:
                volume_ok = any(
                    column in context.bars
                    and context.bars[column].tail(30).fillna(0).abs().sum() > 0
                    for column in ("volume", "tick_count")
                )
                depth_ok = any(k in context.features and context.features[k].tail(30).abs().sum() > 0
                               for k in ("depth_imbalance", "depth_imbalance_ma"))
                if rule.category == "statistical" and not context.extras:
                    state, reason = "N/A", "No aligned anchor instrument"
                elif name in {"vwap_reversion", "chaikin_flow_trend", "money_flow_reversal", "obv_divergence"} and not volume_ok:
                    state, reason = "N/A", "No volume or tick activity"
                elif name == "order_flow" and not depth_ok:
                    state, reason = "N/A", "No order-book depth"
                else:
                    series = rule.score(context).fillna(0.0)
                    value = float(series.iloc[-1])
                    reason = rule.describe(value, context) if abs(value) >= 0.15 else rule.description
                    state = "ACTIVE" if abs(value) >= 0.15 else "WAIT"
            except Exception as exc:
                state, reason = "ERROR", f"{type(exc).__name__}: rule inputs unavailable"
            self._latest_scores[name] = value
            measured = self.rule_stats.get(name, {"scored": 0, "hits": 0, "trust_score": 0.0})
            signals.append(Signal(ts=self.history.index[-1].to_pydatetime(), strategy=name,
                                  direction=Direction.from_value(value, 0.15), strength=abs(value),
                                  reason=reason, meta={
                                      "state": state,
                                      "raw_score": value,
                                      "active_threshold": 0.15,
                                      "category": rule.category,
                                      "description": rule.description,
                                      "trust_min_samples": getattr(
                                          self.performance, "min_trust_samples", 50
                                      ),
                                      **measured,
                                  }))
        return signals

    def _strategy_extras(self) -> dict:
        if self.anchor_provider is None:
            return {}
        try:
            anchor = self.anchor_provider()
        except Exception:  # noqa: BLE001 — a paired rule can abstain independently
            return {}
        return {"anchor_close": anchor} if isinstance(anchor, pd.Series) and not anchor.empty else {}

    def _observe_research(self, bar: dict) -> None:
        stamp = pd.Timestamp(bar["ts"]).to_pydatetime()
        close = float(bar["close"])
        if self.performance is not None:
            self.performance.observe(timestamp=stamp, price=close, instrument=self.instrument_key)
            self._refresh_rule_stats()
        if self.online_lab is not None:
            self.online_lab.observe(timestamp=stamp, price=close, instrument=self.instrument_key)

    def _issue_research(self) -> None:
        if self.features.empty or self.history.empty:
            return
        stamp = self.history.index[-1].to_pydatetime()
        anchor = float(self.history["close"].iloc[-1])
        cost = self._research_cost_bps()
        if self.performance is not None:
            self.performance.issue(
                (signal for signal in self.signals if signal.meta.get("state") == "ACTIVE"),
                timestamp=stamp,
                anchor_price=anchor,
                horizon_min=3 * self.bar_minutes,
                instrument=self.instrument_key,
                cost_bps=cost,
            )
        if self.online_lab is not None:
            row = self._realtime_learning_features()
            self.ai_signals = {
                bars_ahead: self.online_lab.issue(
                    row,
                    timestamp=stamp,
                    anchor_price=anchor,
                    horizon_min=bars_ahead * self.bar_minutes,
                    instrument=self.instrument_key,
                    cost_bps=cost,
                )
                for bars_ahead in range(1, 4)
            }

    def _realtime_learning_features(self) -> dict[str, float]:
        """Technical state plus bounded strategy permutations and past quality.

        Every value is known at issue time. Strategy accuracy/trust comes only
        from previously matured outcomes, so feeding it back into the online
        learner preserves the causal boundary.
        """
        row = {
            str(name): float(value)
            for name, value in self.features.iloc[-1].items()
            if pd.notna(value) and math.isfinite(float(value))
        }
        usable = [
            signal
            for signal in self.signals
            if signal.meta.get("state") not in {"N/A", "ERROR"}
            and math.isfinite(float(signal.score))
        ]
        for signal in usable:
            row[f"strategy__{signal.strategy}"] = float(signal.score)
            stat = self.rule_stats.get(signal.strategy, {})
            row[f"strategy_trust__{signal.strategy}"] = float(stat.get("trust_score", 0.0))
            samples = int(stat.get("scored", 0))
            row[f"strategy_edge__{signal.strategy}"] = (
                (float(stat.get("accuracy", 0.5)) - 0.5) * 2.0 if samples else 0.0
            )

        config = self.settings.plugin_config.get("forecast:online_research", {})
        limit = max(int(config.get("strategy_feature_limit", 8)), 1)
        order = max(1, min(int(config.get("interaction_order", 3)), 3))
        strongest = sorted(usable, key=lambda signal: (-abs(signal.score), signal.strategy))[:limit]
        for size in range(2, min(order, len(strongest)) + 1):
            for combination in itertools.combinations(strongest, size):
                names = "__".join(signal.strategy for signal in combination)
                row[f"strategy_combo{size}__{names}"] = math.prod(
                    float(signal.score) for signal in combination
                )

        scores = [float(signal.score) for signal in usable]
        active = [score for score in scores if abs(score) >= 0.15]
        row["strategy_meta__breadth"] = sum(active) / len(active) if active else 0.0
        row["strategy_meta__agreement"] = (
            abs(sum(1 if score > 0 else -1 for score in active)) / len(active)
            if active
            else 0.0
        )
        row["strategy_meta__active"] = float(len(active))
        return row

    def _refresh_rule_stats(self) -> None:
        if self.performance is None:
            return
        self.rule_stats = {
            card.strategy: {
                "scored": card.samples,
                "hits": card.hits,
                "accuracy": card.accuracy,
                "wins": card.wins,
                "losses": card.losses,
                "net_pnl_bps": card.net_pnl_bps,
                "drawdown_bps": card.max_drawdown_bps,
                "trust_score": card.trust_score,
            }
            for card in self.performance.scorecards(self.instrument_key)
        }

    def _research_cost_bps(self) -> float:
        if self.last_price <= 0:
            return self.settings.cost_hurdle_bps()
        if self.instrument_key == self.settings.instrument_key:
            return self.settings.cost_hurdle_bps()
        fixed_per_unit = self.settings.option_fixed_cost_rupees / max(self.settings.lot_size, 1)
        spread_and_fees = self.spread + fixed_per_unit
        return self.settings.option_variable_cost_bps + spread_and_fees / self.last_price * 10_000

    def _ai_view(self) -> float:
        signal = self.ai_signals.get(1)
        if signal is None:
            return 0.0
        return (float(signal.p_up) - 0.5) * 2.0 * float(signal.trust_score)

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

    @synchronized
    def refresh_projection(self) -> bool:
        """Rebuild the projected path from the live price and the current view.

        Called on a fixed cadence rather than per tick: the anchor and the tick
        momentum already move with the tape, and recomputing the whole path
        thousands of times a minute to move it by a hundredth of a point is work
        nobody can see. Returns whether a path was produced.
        """
        if self.forecaster is None:
            return False

        if self.source == "Upstox":
            if self.last_tick_ts is None or self.market_status not in {None, "NORMAL_OPEN"}:
                self.projections = []
                return False
            stale = (datetime.now(IST) - self.last_tick_ts).total_seconds() > 90
            if stale or not self.calendar.is_open():
                self.projections = []
                return False

        bars = self.market_bars()
        if len(bars) < MIN_PROJECTION_BARS:
            return False

        anchor = float(self.last_price or bars["close"].iloc[-1])
        fast = trend_conviction(bars)
        self.conviction = blend(self.slow_conviction, fast)
        learned_closes = {}
        if self.online is not None and not self.features.empty:
            offset = int((bars.index[-1] - self.history.index[-1]).total_seconds() / (60 * self.bar_minutes))
            moves = self.online.moves(self.features)
            base = float(self.history["close"].iloc[-1])
            learned_closes = {h: base * (1 + moves[h + offset]) for h in range(1, 4) if h + offset in moves}

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
            learned_closes=learned_closes,
            session_only=self.source == "Upstox",
            record=self.source != "Upstox" or self.tick_count > 0,
            issued_at=self.last_tick_ts or bars.index[-1],
            context={"regime": self.regime, "conviction": self.conviction,
                     "strategies": [{"name": s.strategy, "direction": s.direction.value, "reason": s.reason} for s in self.signals],
                     "learner": self.online.describe() if self.online else {}},
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
        total = weight_sum = 0.0
        for rule, prior in self.strategy.components:
            stat = self.rule_stats.get(rule.name, {})
            # Measured trust may raise a prior by at most 25%; unproven rules
            # retain 75% so a fresh install can still produce research signals.
            weight = prior * (0.75 + 0.5 * float(stat.get("trust_score", 0.0)))
            total += self._latest_scores.get(rule.name, 0) * weight
            weight_sum += abs(weight)
        return total / weight_sum if weight_sum else 0.0

    @synchronized
    def aligned_close(self) -> pd.Series:
        """A thread-safe copy for cross-instrument statistical strategies."""
        return self.history["close"].copy() if "close" in self.history else pd.Series(dtype="float64")

    # -------------------------------------------------------------- snapshot

    @synchronized
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
            for key in (
                "ema_dist_9", "ema_dist_21", "ema_dist_50", "ema_dist_200",
                "ema_9_21_spread", "ema_21_50_spread", "ema_50_200_spread",
                "supertrend_dir", "supertrend_dist", "adx_14", "plus_di", "minus_di",
                "di_spread", "aroon_osc", "vortex_spread", "slope_20", "r2_20",
                "roc_5", "roc_10", "roc_20", "rsi_7", "rsi_14", "rsi_21",
                "macd_line", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
                "williams_r", "cci_20", "mfi_14", "cmf_20", "atr_norm", "atr_ratio",
                "bb_pct_b", "bb_bandwidth", "bb_squeeze", "keltner_position",
                "donchian_position", "vwap_dist", "obv_slope_20", "efficiency_ratio_10",
                "hurst_100", "kama_dist_10", "dema_dist_20", "tema_dist_20", "ppo_hist",
                "choppiness_14", "sortino_30", "autocorr_1_50", "variance_ratio_5_60",
                "direction_entropy_50", "amihud_20",
            ):
                if key in self.features.columns:
                    value = last[key]
                    indicators[key] = float(value) if pd.notna(value) else float("nan")
            reference_close = float(self.history["close"].iloc[-1])
            for window in (9, 21, 50, 200):
                distance = indicators.get(f"ema_dist_{window}")
                if distance is not None and pd.notna(distance) and 1.0 + distance != 0:
                    indicators[f"ema_{window}"] = reference_close / (1.0 + distance)
            for name, distance_key in (
                ("supertrend", "supertrend_dist"),
                ("vwap", "vwap_dist"),
            ):
                distance = indicators.get(distance_key)
                if distance is not None and pd.notna(distance) and 1.0 + distance != 0:
                    indicators[name] = reference_close / (1.0 + distance)

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
            instrument_key=self.instrument_key,
            source=self.source,
            last_tick_ts=self.last_tick_ts or self.last_quote_ts,
            research={"horizons": {h: s.as_row() for h, s in self.forecaster.tracker.stats().items()} if self.forecaster else {},
                      "recent": recent_scores(self.forecaster.tracker) if self.forecaster else [],
                      "pending": self.forecaster.tracker.pending if self.forecaster else 0,
                      "missing": self.forecaster.tracker.missed_bars if self.forecaster else 0,
                      "online": self.online.describe() if self.online else {},
                      "ai": self._ai_snapshot(),
                      "strategies": list(self.rule_stats.values()),
                      "compute_ms": self.compute_ms, "ticks": self.tick_count, "rejected_ticks": self.rejected_ticks,
                      "spread": self.spread, "fixed_cost_rupees": self.settings.option_fixed_cost_rupees,
                      "variable_cost_bps": self.settings.option_variable_cost_bps,
                      "workers": self.settings.worker_count},
        )

    @synchronized
    def enable_research(self) -> None:
        mode = "live" if self.source == "Upstox" else "simulation"
        if self.forecaster:
            self.forecaster.tracker.reset()
            self.forecaster.tracker.journal = ResearchJournal(self.settings.data_dir / "research" / f"{mode}.sqlite3",
                self.instrument_key, self.source, self.bar_minutes)
        if self.online:
            try:
                self.online.attach(self.settings.model_dir / "online" / mode,
                                   f"{self.instrument_key}:{self.bar_minutes}")
                self.online.update(self.history, self.features, live=False)
                self.online.save()
            except (OSError, ValueError, EOFError):
                self.status = "online checkpoint unavailable; learning in memory"
        capabilities = self.kernel.registry.capabilities()
        if capabilities.get("online_research") and self.settings.online_learning:
            identity = hashlib.sha256(
                f"{self.instrument_key}:{self.bar_minutes}".encode()
            ).hexdigest()[:24]
            self.online_lab = self.kernel.new(
                self.kernel.provider("online_research"),
                state_path=self.settings.data_dir / "research" / "online_ai" / mode / f"{identity}.sqlite3",
            )
        if capabilities.get("strategy_performance"):
            self.performance = self.kernel.capability("strategy_performance")
            self._refresh_rule_stats()
        # Issue an honest forecast immediately from the warmed state. It remains
        # pending until a future bar closes, but makes the auto-learning system
        # visible without waiting one whole interval for the first issue.
        if self.online_lab is not None and not self.features.empty:
            self._issue_research()

    def _ai_snapshot(self) -> dict:
        if self.online_lab is None:
            return {}
        signals = {
            horizon: {
                "issued_at": signal.issued_at.isoformat(),
                "target_at": signal.target_at.isoformat(),
                "action": signal.action.value,
                "p_up": signal.p_up,
                "confidence": signal.confidence,
                "trust_score": signal.trust_score,
                "algorithms": [
                    {
                        "name": item.algorithm,
                        "action": item.action.value,
                        "p_up": item.p_up,
                        "confidence": item.confidence,
                        "trust_score": item.trust_score,
                    }
                    for item in signal.predictions
                ],
            }
            for horizon, signal in self.ai_signals.items()
        }
        cards = []
        for card in self.online_lab.scorecards():
            row = card.as_row()
            row.update(
                {
                    "wins": card.all_time.wins,
                    "losses": card.all_time.losses,
                    "profit_bps": card.all_time.profit_bps,
                    "loss_bps": card.all_time.loss_bps,
                    "drawdown_bps": card.all_time.max_drawdown_bps,
                    "rolling_net_pnl_bps": card.rolling.net_pnl_bps,
                }
            )
            cards.append(row)
        return {
            "signals": signals,
            "scorecards": cards,
            "today": self.online_lab.daily_accuracy(timezone=IST),
            "pending": self.online_lab.pending_count,
            "expired": self.online_lab.expired_count,
            "policy": {
                "buy_probability": self.online_lab.buy_probability,
                "sell_probability": self.online_lab.sell_probability,
                "min_trust_for_action": self.online_lab.min_trust_for_action,
                "cost_bps": self._research_cost_bps(),
            },
        }

    @synchronized
    def close(self) -> None:
        if self.online:
            self.online.save()
        if self.forecaster and self.forecaster.tracker.journal:
            self.forecaster.tracker.journal.close()
            self.forecaster.tracker.journal = None

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
        if not ticks.empty and not self.history.empty and ticks.index[0] <= self.history.index[-1]:
            # A headless replay may revisit cached bars. Rewind all learning
            # state to the prefix; the replay's future must not remain in memory.
            self.history = self.history[self.history.index < ticks.index[0]].copy()
            self.aggregator = self.kernel.new("aggregator:candle_builder", bar_minutes=self.bar_minutes)
            if self.online is not None:
                self.online = OnlineLearner(self.bar_minutes)
            if self.forecaster is not None:
                self.forecaster.tracker.reset()
                self.forecaster.momentum.reset()
            self.rule_stats.clear()
            self.tick_count = 0
            self.last_tick_ts = None
            self.bootstrap(self.history)
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
