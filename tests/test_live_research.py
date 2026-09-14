from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from core.calendar import IST
from core.settings import Settings, UpstoxCredentials
from core.types import ForecastCandle, Tick
from kernel import Kernel
from plugins.aggregators.candle_builder.aggregator import CandleAggregator
from plugins.features.technical.pipeline import build_features
from plugins.forecasts.projection.tracker import ProjectionTracker
from plugins.sources.simulated.series import generate_candles
from plugins.sources.upstox.auth import AuthenticationError
from plugins.sources.upstox.broker import UpstoxBroker
from plugins.sources.upstox.feed import UpstoxFeed, _decode_tick
from plugins.sources.upstox.proto import MarketDataFeed_pb2 as pb
from plugins.strategies.base import StrategyContext
from plugins.strategies.trend.rules import EmaTrendStrategy, OpeningRangeBreakoutStrategy
from runtime.board import BoardLeg, MarketBoard
from runtime.engine import Engine
from runtime.research import OnlineLearner, ResearchJournal


def test_rejected_env_token_falls_back_only_after_401(monkeypatch, tmp_path):
    import plugins.sources.upstox.broker as module

    calls = []
    class REST:
        def __init__(self, token):
            self.token = token
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def ltp(self, key):
            calls.append(self.token)
            if self.token == "rejected":
                raise AuthenticationError("401")
            return {key: {"last_price": 24000}}
    monkeypatch.setattr(module, "UpstoxREST", REST)
    monkeypatch.setattr(module.TokenStore, "load", lambda self: "working")
    broker = UpstoxBroker(Settings(data_dir=tmp_path, credentials=UpstoxCredentials(access_token="rejected")))
    assert broker.validate_auth() == "working"
    assert calls == ["rejected", "working"]
    assert broker.token == "working"
    assert "rejected token skipped" in broker.auth_status

    def network_error(self, key):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(REST, "ltp", network_error)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        broker.validate_auth()
    assert "working" not in repr(broker)


def test_decoder_preserves_identity_and_exchange_trade_time():
    raw = pb.Feed()
    raw.fullFeed.marketFF.ltpc.ltp = 123.4
    raw.fullFeed.marketFF.ltpc.ltt = 1770010200000
    raw.fullFeed.marketFF.iv = 15
    raw.fullFeed.marketFF.optionGreeks.delta = 0.5
    tick = _decode_tick("NSE_FO|123", raw, datetime.now(IST))
    assert tick.instrument_key == "NSE_FO|123"
    assert tick.ts.timestamp() == 1770010200
    assert tick.greeks["delta"] == 0.5
    assert tick.greeks["iv"] == 15


async def test_websocket_subscription_is_binary(monkeypatch):
    import plugins.sources.upstox.feed as module

    sent = []
    feed = UpstoxFeed("dummy", ["NSE_INDEX|Nifty 50"], on_tick=lambda tick: None)
    class Socket:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def send(self, data):
            sent.append(data)
            feed.stop()
    monkeypatch.setattr(module, "authorize_feed_url", lambda token: "wss://example.test/feed")
    monkeypatch.setattr(module.websockets, "connect", lambda *args, **kwargs: Socket())
    await feed._stream_once()
    assert isinstance(sent[0], bytes)
    assert b'"instrumentKeys"' in sent[0]


def test_live_tick_routes_only_to_its_own_engine():
    board = MarketBoard(Kernel.bootstrap(Settings()), live=True)
    engines = [MagicMock() for _ in range(3)]
    board.legs = [BoardLeg(label, kind, engine, instrument_key=key) for label, kind, key, engine in
                  zip(("INDEX", "CALL", "PUT"), ("index", "CE", "PE"), ("index", "call", "put"), engines, strict=True)]
    tick = Tick(datetime.now(IST), 100, instrument_key="call")
    board.on_tick(tick)
    engines[1].on_tick.assert_called_once_with(tick)
    engines[0].on_tick.assert_not_called()
    engines[2].on_tick.assert_not_called()
    board.executor.shutdown()


def test_volume_is_per_bar_and_old_ticks_cannot_change_a_candle():
    aggregator = CandleAggregator()
    now = datetime(2026, 9, 10, 10, 0, tzinfo=IST)
    aggregator.on_tick(Tick(now, 100, volume_traded=1000))
    aggregator.on_tick(Tick(now + timedelta(seconds=30), 101, volume_traded=1030))
    bar = aggregator.on_tick(Tick(now + timedelta(minutes=1), 102, volume_traded=1050))
    assert bar["volume"] == 30
    assert aggregator.current_bar["volume"] == 20
    aggregator.on_tick(Tick(now, 1, volume_traded=500))
    assert aggregator.current_bar["close"] == 102


def test_first_issued_forecast_is_immutable_and_persisted(tmp_path):
    stamp = pd.Timestamp("2026-09-10 10:02", tz=IST)
    candle = ForecastCandle(stamp, 1, 100, 103, 99, 102)
    tracker = ProjectionTracker(freeze_first=True)
    journal = ResearchJournal(tmp_path / "research.sqlite3", "NSE_FO|123", "Upstox", 1)
    tracker.journal = journal
    tracker.record([candle], 100, issued_at=stamp - pd.Timedelta(minutes=1), context={"regime": "trending"})
    tracker.record([replace(candle, close=98)], 100)
    score = tracker.observe(stamp, 98)[0]
    assert not score.hit
    assert score.projected_close == 102
    stored = journal.db.execute("SELECT predicted_close,actual_close,status,context FROM forecasts").fetchone()
    assert stored[:3] == (102, 98, "miss")
    assert "trending" in stored[3]
    assert tracker.observe(stamp, 98) == []
    journal.close()


def test_missing_bars_are_not_scored_and_flat_is_not_an_automatic_win(tmp_path):
    stamp = pd.Timestamp("2026-09-10 10:02", tz=IST)
    tracker = ProjectionTracker(freeze_first=True)
    tracker.record([ForecastCandle(stamp, 1, 100, 101, 99, 100)], 100)
    assert not tracker.observe(stamp, 102)[0].hit
    tracker.record([ForecastCandle(stamp + pd.Timedelta(minutes=1), 1, 100, 103, 99, 102)], 100)
    assert tracker.observe(stamp + pd.Timedelta(minutes=2), 104) == []
    assert tracker.missed_bars == 1
    assert tracker.overall().scored == 1


@pytest.fixture
def history():
    return generate_candles(days=2, seed=998)


def test_online_learning_uses_only_matured_labels_and_skips_duplicates(history, tmp_path):
    bars = history.iloc[:200]
    features = build_features(bars)
    learner = OnlineLearner(1)
    learned = learner.update(bars, features, live=False)
    assert learned > 0
    assert learner.live_updates == 0
    assert learner.update(bars, features) == 0
    learner.attach(tmp_path, "instrument:1")
    learner.save()
    restored = OnlineLearner(1)
    restored.attach(tmp_path, "instrument:1")
    assert restored.samples == learner.samples
    assert restored.update(bars, features) == 0
    longer = history.iloc[:201]
    assert restored.update(longer, build_features(longer)) == 4
    assert restored.live_updates == 4
    assert all(stamp <= longer.index[-1] for stamp in restored.through.values())
    # Appending unknown future bars cannot change features available at origin.
    np.testing.assert_allclose(features["rsi_14"], build_features(longer)["rsi_14"].iloc[:200], equal_nan=True)


def test_online_labels_do_not_bridge_missing_minutes(history):
    bars = history.iloc[:200].drop(history.index[180])
    learner = OnlineLearner(1)
    learner.update(bars, build_features(bars))
    contiguous = OnlineLearner(1)
    contiguous.update(history.iloc[:200], build_features(history.iloc[:200]))
    assert learner.samples[1] == contiguous.samples[1] - 2
    assert learner.samples[3] == contiguous.samples[3] - 4


def test_ema_bullish_stack_has_positive_sign():
    features = pd.DataFrame({"ema_dist_9": [0.001], "ema_dist_21": [0.002],
                             "ema_dist_50": [0.003], "adx_14": [40], "di_spread": [25]})
    assert EmaTrendStrategy().score(StrategyContext(pd.DataFrame(), features)).iloc[0] > 0


def test_opening_range_waits_until_price_leaves_the_range():
    times = pd.date_range("2026-09-10 09:15", periods=30, freq="min", tz=IST)
    bars = pd.DataFrame({"open": 100, "high": 110, "low": 90, "close": 103}, index=times)
    features = pd.DataFrame({"minutes_from_open": np.arange(30)}, index=times)
    rule = OpeningRangeBreakoutStrategy()
    result = rule.score(StrategyContext(bars, features))
    assert (result == 0).all()
    bars.loc[times[-1], "close"] = 115
    result = rule.score(StrategyContext(bars, features))
    assert result.iloc[-1] > 0
    assert (result.iloc[:-1] == 0).all()


def test_engine_shows_ten_rules_and_disables_index_volume_claims(history, tmp_path):
    settings = Settings(model_dir=tmp_path, data_dir=tmp_path)
    engine = Engine(Kernel.bootstrap(settings)).bootstrap(history)
    assert len(engine.signals) == 10
    assert next(s for s in engine.signals if s.strategy == "vwap_reversion").meta["state"] == "N/A"
    assert next(s for s in engine.signals if s.strategy == "order_flow").meta["state"] == "N/A"
    assert engine.forecaster.tracker.overall().scored == 0
    engine.close()


def test_live_engine_scores_before_new_forecasts_and_trains_on_closed_bars(history, tmp_path):
    settings = Settings(model_dir=tmp_path / "models", data_dir=tmp_path / "data")
    warm = history.iloc[:200].copy()
    now = pd.Timestamp.now(tz=IST).floor("min")
    warm.index = pd.date_range(end=now - pd.Timedelta(minutes=1), periods=len(warm), freq="min")
    engine = Engine(Kernel.bootstrap(settings), source="Upstox").bootstrap(warm)
    engine.calendar.is_open = lambda moment=None: True
    engine.enable_research()
    baseline = engine.online.samples[1]
    price = float(warm["close"].iloc[-1])
    engine.market_status = "NORMAL_CLOSE"
    engine.on_tick(Tick(now.to_pydatetime(), price))
    assert engine.tick_count == 0
    assert not engine.refresh_projection()
    assert engine.online.samples[1] == baseline
    engine.market_status = "NORMAL_OPEN"
    engine.on_tick(Tick(now.to_pydatetime(), price))
    assert engine.online.samples[1] == baseline  # forming candle has no label
    engine.refresh_projection()
    assert len(engine.projections) == 3
    for minute in range(1, 8):
        engine.on_tick(Tick((now + pd.Timedelta(minutes=minute)).to_pydatetime(), price + minute * 2))
    assert engine.online.samples[1] > baseline
    assert engine.online.live_updates > 0
    assert all(engine.forecaster.tracker.stat(h).scored > 0 for h in (1, 2, 3))
    journal = engine.forecaster.tracker.journal
    assert journal.db.execute("SELECT COUNT(*) FROM forecasts WHERE status IN ('hit','miss')").fetchone()[0] > 0
    engine.close()


def test_parallel_manifest_updates_preserve_each_instrument(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from plugins.sources.history.manifest import StoreManifest

    path = tmp_path / "manifest.json"
    def update(index):
        StoreManifest(path).update("ticks", f"instrument-{index}", rows=index)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(update, range(20)))
    assert len(StoreManifest(path).instruments("ticks")) == 20
