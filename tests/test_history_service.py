"""Tests for the history service and the bar loader.

The store underneath these has thorough tests of its own; what was missing was
coverage of the *policy* on top of it — when to serve the cache, when to fetch,
what happens with no token, and what happens when the cache is current. That gap
let a stale attribute reference survive a refactor and reach a user as an
`AttributeError` on `niftypulse fetch`, which is exactly the kind of bug these
tests exist to prevent.

Every path here is a branch a user can reach from the CLI, so every branch is
exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.bars import normalize_candles
from core.calendar import IST
from core.settings import Settings
from kernel import Kernel
from plugins.sources import DataUnavailable
from plugins.sources.history import HistorySource, PartitionedStore
from plugins.sources.simulated.series import generate_candles
from runtime.bars import BarLoader

# ------------------------------------------------------------------- doubles


class FakeREST:
    """A provider client that records what it was asked for."""

    def __init__(self, frame: pd.DataFrame | None = None, intraday: pd.DataFrame | None = None) -> None:
        self.frame = frame if frame is not None else _empty()
        self.intraday = intraday if intraday is not None else _empty()
        self.calls: list[tuple] = []

    def fetch_minute_history(self, instrument_key: str, days: int = 0) -> pd.DataFrame:
        self.calls.append(("history", instrument_key, days))
        return self.frame

    def fetch_intraday(self, instrument_key: str, unit: str = "minutes", interval: int = 1) -> pd.DataFrame:
        self.calls.append(("intraday", instrument_key))
        return self.intraday


class FakeBroker:
    """A broker capability: a token, a REST client, and nothing else."""

    def __init__(self, token: str | None = "token", rest: FakeREST | None = None) -> None:
        self._token = token
        self._rest = rest or FakeREST()
        self.closed = False

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def is_configured(self) -> bool:
        return self._token is not None

    def rest(self) -> FakeREST:
        if self._token is None:
            raise DataUnavailable("no token")
        return self._rest

    def close(self) -> None:
        self.closed = True


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, model_dir=tmp_path / "artifacts", log_dir=tmp_path / "logs")


@pytest.fixture
def recent_bars() -> pd.DataFrame:
    """A cache ending now, so the "cache is current" branch is reachable."""
    end = pd.Timestamp.now(tz=IST).floor("min")
    index = pd.date_range(end=end, periods=120, freq="1min")
    frame = pd.DataFrame(
        {"open": 24_000.0, "high": 24_001.0, "low": 23_999.0, "close": 24_000.0, "volume": 0.0, "oi": 0.0},
        index=index,
    )
    return normalize_candles(frame)


def seeded(settings: Settings, frame: pd.DataFrame) -> PartitionedStore:
    source = HistorySource(settings)
    source.seed(frame)
    return source.store


# ------------------------------------------------------------------ no token


def test_load_history_with_no_token_and_no_broker_returns_empty(settings) -> None:
    """No cache, no provider: an empty frame, not an exception or invented data."""
    source = HistorySource(settings, broker=None)
    assert source.load_history(days=5).empty
    assert source.access_token is None
    assert source.can_fetch is False


def test_load_history_with_no_token_serves_the_cache(settings, bars) -> None:
    """The path that reached a user as an AttributeError: cache present, no token.

    `niftypulse fetch` without `--offline` and without credentials lands here, so
    it is not an edge case — it is what happens on a fresh clone with a cache.
    """
    seeded(settings, bars)
    source = HistorySource(settings, broker=FakeBroker(token=None))

    loaded = source.load_history(days=5, quiet=False)
    assert len(loaded) == len(bars)
    assert loaded["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])


def test_the_cache_message_names_the_store_directory(settings, bars, capsys) -> None:
    """The message was where the stale attribute lived, so assert it renders."""
    seeded(settings, bars)
    HistorySource(settings, broker=None).load_history(days=5, quiet=False)

    printed = capsys.readouterr().out
    assert "cached bars" in printed
    assert "candles" in printed
    assert "None" not in printed


def test_quiet_suppresses_the_message(settings, bars, capsys) -> None:
    seeded(settings, bars)
    HistorySource(settings, broker=None).load_history(days=5, quiet=True)
    assert capsys.readouterr().out == ""


def test_offline_flag_serves_the_cache_without_a_broker(settings, bars) -> None:
    seeded(settings, bars)
    source = HistorySource(settings, broker=FakeBroker(), offline=True)
    assert len(source.load_history(days=5)) == len(bars)
    assert source.can_fetch is False


def test_load_cached_never_raises_on_an_empty_store(settings) -> None:
    source = HistorySource(settings)
    assert source.load_cached().empty


# --------------------------------------------------------------- with a token


def test_a_current_cache_is_not_refetched(settings, recent_bars) -> None:
    """The promise of the whole cache: a morning run with nothing new costs one call."""
    requested_start = (
        pd.Timestamp.now(tz=IST) - pd.Timedelta(days=5)
    ).normalize()
    first = recent_bars.iloc[[0]].copy()
    first.index = pd.DatetimeIndex([requested_start + pd.Timedelta(hours=9, minutes=15)])
    cached = normalize_candles(pd.concat([first, recent_bars]))
    seeded(settings, cached)
    broker = FakeBroker(rest=FakeREST())
    source = HistorySource(settings, broker=broker)

    loaded = source.load_history(days=5, quiet=True)

    assert len(loaded) == len(cached)
    assert broker._rest.calls == [], "a current cache must not hit the provider"


def test_a_current_tail_with_missing_requested_history_is_backfilled(
    settings, recent_bars
) -> None:
    seeded(settings, recent_bars)
    broker = FakeBroker(rest=FakeREST())

    HistorySource(settings, broker=broker).load_history(days=5, quiet=True)

    assert broker._rest.calls[0][0] == "history"


def test_a_stale_cache_fetches_the_tail_and_stores_it(settings, bars) -> None:
    """Only the missing window is requested, and what arrives is written."""
    # Synthetic fixtures include today's entire session, even before the close.
    # Use completed prior sessions so this exercises staleness at any wall time.
    stale = bars[bars.index.normalize() < pd.Timestamp.now(tz=IST).normalize()]
    store = seeded(settings, stale)
    before = store.row_count()

    fresh = generate_candles(days=2, seed=99)
    broker = FakeBroker(rest=FakeREST(frame=fresh))
    source = HistorySource(settings, broker=broker)

    loaded = source.load_history(days=90, quiet=True)

    assert broker._rest.calls, "a stale cache must fetch"
    assert broker._rest.calls[0][0] == "history"
    assert store.row_count() >= before
    assert len(loaded) == store.row_count()


def test_the_intraday_call_is_made_for_the_current_session(settings, bars) -> None:
    """The historical endpoint excludes today; without this a morning run misses it."""
    stale = bars[bars.index.normalize() < pd.Timestamp.now(tz=IST).normalize()]
    seeded(settings, stale)
    broker = FakeBroker(rest=FakeREST(frame=generate_candles(days=1, seed=7)))
    HistorySource(settings, broker=broker).load_history(days=30, quiet=True)

    kinds = [call[0] for call in broker._rest.calls]
    assert "intraday" in kinds


def test_refresh_false_returns_the_cache_untouched(settings, bars) -> None:
    store = seeded(settings, bars)
    broker = FakeBroker(rest=FakeREST())
    source = HistorySource(settings, broker=broker)

    loaded = source.load_history(days=30, refresh=False, quiet=True)

    assert broker._rest.calls == []
    assert len(loaded) == store.row_count()


def test_a_failing_intraday_call_does_not_lose_the_backfill(settings, bars) -> None:
    """One bad request should not cost the other twenty-five."""

    class FlakyIntraday(FakeREST):
        def fetch_intraday(self, *args, **kwargs):
            raise RuntimeError("intraday unavailable")

    seeded(settings, bars)
    broker = FakeBroker(rest=FlakyIntraday(frame=generate_candles(days=2, seed=12)))
    loaded = HistorySource(settings, broker=broker).load_history(days=60, quiet=True)
    assert not loaded.empty


def test_rest_without_a_broker_explains_itself(settings) -> None:
    source = HistorySource(settings, broker=None)
    with pytest.raises(DataUnavailable, match="broker"):
        source.rest()


def test_close_delegates_to_the_broker(settings) -> None:
    broker = FakeBroker()
    HistorySource(settings, broker=broker).close()
    assert broker.closed is True


def test_seed_replaces_the_cache(settings, bars) -> None:
    store = seeded(settings, bars)
    assert store.row_count() == len(bars)

    replacement = bars.tail(50)
    loaded = HistorySource(settings).seed(replacement)
    assert len(loaded) == 50
    assert store.row_count() == 50


# ---------------------------------------------------------------- bar loader


def kernel_for(settings: Settings) -> Kernel:
    return Kernel.bootstrap(settings)


def test_bar_loader_with_a_cache_and_no_token_returns_the_cache(settings, bars) -> None:
    """The exact CLI path that failed: fetch without --offline and without a token."""
    seeded(settings, bars)
    loader = BarLoader(kernel_for(settings))

    loaded = loader.load(days=5, refresh=True, offline=False, quiet=True)
    assert len(loaded) == len(bars)


def test_bar_loader_generates_and_stores_when_offline_with_nothing_cached(settings) -> None:
    loader = BarLoader(kernel_for(settings))
    loaded = loader.load_offline(days=3, quiet=True)

    assert not loaded.empty
    assert loader.kernel.capability("history").store.row_count() == len(loaded)


def test_bar_loader_prefers_the_cache_over_generating(settings, bars) -> None:
    """Generating over a cache would silently replace real bars with invented ones."""
    seeded(settings, bars)
    loader = BarLoader(kernel_for(settings))

    loaded = loader.load_offline(days=3, quiet=True)
    assert len(loaded) == len(bars)
    assert loaded.index[0] == bars.index[0]


def test_bar_loader_falls_back_when_the_provider_cannot_be_reached(settings, bars) -> None:
    seeded(settings, bars)
    loader = BarLoader(kernel_for(settings))

    class Unavailable(FakeBroker):
        def rest(self):
            raise DataUnavailable("token expired")

    # The kernel builds its own broker; the source is what the loader talks to.
    loader.kernel.capability("history").broker = Unavailable(token="stale")
    loaded = loader.load(days=5, quiet=True)
    assert len(loaded) == len(bars)


def test_bar_loader_reports_the_offline_decision(settings, capsys) -> None:
    BarLoader(kernel_for(settings)).load_offline(days=3, quiet=False)
    assert "generating" in capsys.readouterr().out
