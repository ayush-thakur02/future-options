"""Tests for the partitioned store and the live recorders.

The store's promises are all about *how much work* a read or a write does, which
is the kind of thing that is easy to claim and easy to lose. So these tests check
the filesystem directly: that a write touches one file rather than all of them,
that a range read opens only the partitions it needs, and that a chain snapshot
with eighteen strikes on one timestamp keeps all eighteen.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.bars import normalize_candles
from core.calendar import IST
from core.types import Tick
from plugins.sources.history import (
    CANDLES,
    CHAIN,
    HOUR,
    TICKS,
    ChainRecorder,
    PartitionedStore,
    StoreManifest,
    TickRecorder,
    normalize_frames,
    safe_name,
)
from plugins.sources.option_chain import synthetic_chain
from plugins.sources.simulated.series import generate_candles


@pytest.fixture
def bars() -> pd.DataFrame:
    return generate_candles(days=4, seed=41)


@pytest.fixture
def store(tmp_path, bars) -> PartitionedStore:
    store = PartitionedStore(tmp_path, "NSE_INDEX|Nifty 50", bar_minutes=1)
    store.write(bars)
    return store


def day_frame(day: str, minutes: int = 5, price: float = 24_000.0) -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:15", periods=minutes, freq="1min", tz=IST)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 0.0,
            "oi": 0.0,
        },
        index=index,
    )


# ---------------------------------------------------------------- path layout


def test_safe_name_is_reversible_by_eye() -> None:
    assert safe_name("NSE_INDEX|Nifty 50") == "NSE_INDEX_Nifty_50"
    assert safe_name("NSE_FO|43919") == "NSE_FO_43919"
    assert "/" not in safe_name("A/B|C D")


def test_the_trading_day_is_the_partition_key(store) -> None:
    """A session can never be split across files, or merged with the next morning."""
    days = {path.stem for path in store.partitions()}
    assert len(days) == 4
    assert all(len(day) == 10 for day in days)

    loaded = store.load()
    per_day = loaded.groupby(loaded.index.date).size()
    assert per_day.nunique() == 1, "a session landed in pieces"


def test_partition_path_carries_year_and_month(store, tmp_path) -> None:
    path = store.partitions()[0]
    relative = path.relative_to(tmp_path).parts
    assert relative[:2] == ("candles", "NSE_INDEX_Nifty_50")
    assert relative[2].isdigit() and len(relative[2]) == 4
    assert relative[3].isdigit() and len(relative[3]) == 2


def test_each_dataset_gets_its_own_tree(tmp_path) -> None:
    candles = PartitionedStore(tmp_path, "NIFTY", dataset=CANDLES)
    ticks = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    assert candles.base != ticks.base
    assert candles.base.name == "NIFTY" and ticks.base.name == "NIFTY"


# --------------------------------------------------------------------- writes


def test_write_is_idempotent(tmp_path, bars) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    store.write(bars)
    first = {path: path.stat().st_mtime_ns for path in store.partitions()}
    store.write(bars)
    assert store.row_count() == len(bars)
    assert set(store.partitions()) == set(first), "a re-write created new partitions"


def test_re_fetching_a_day_overwrites_rather_than_duplicates(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    store.write(day_frame("2026-03-10"))

    revised = day_frame("2026-03-10")
    revised["close"] = 99.0
    store.write(revised)

    loaded = store.load()
    assert len(loaded) == 5
    assert loaded["close"].iloc[0] == pytest.approx(99.0)


def test_a_write_touches_only_the_days_it_carries(tmp_path) -> None:
    """The property that makes an append cheap: everything else is untouched."""
    store = PartitionedStore(tmp_path, "NIFTY")
    store.write(day_frame("2026-03-10"))
    store.write(day_frame("2026-03-11"))

    before = {path.stem: path.stat().st_mtime_ns for path in store.partitions()}
    time.sleep(0.01)
    store.write(day_frame("2026-03-11", price=25_000.0))
    after = {path.stem: path.stat().st_mtime_ns for path in store.partitions()}

    assert after["2026-03-10"] == before["2026-03-10"], "an untouched day was rewritten"
    assert after["2026-03-11"] > before["2026-03-11"]


def test_write_of_nothing_is_harmless(tmp_path, bars) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    assert store.write(None) == 0
    assert store.write(bars.iloc[0:0]) == 0
    assert store.load().empty


def test_replace_removes_what_it_does_not_contain(store, tmp_path) -> None:
    store.replace(day_frame("2026-04-01"))
    loaded = store.load()
    assert len(loaded) == 5
    assert len(store.partitions()) == 1


# ---------------------------------------------------------------------- reads


def test_a_range_read_opens_only_the_partitions_it_needs(store) -> None:
    days = sorted({path.stem for path in store.partitions()})
    start = pd.Timestamp(days[0], tz=IST)
    end = pd.Timestamp(days[0], tz=IST) + pd.Timedelta(hours=23)

    assert len(store.partitions(start, end)) == 1

    loaded = store.load(start, end)
    assert len(loaded) == 375
    assert loaded.index[0].date() == pd.Timestamp(days[0]).date()


def test_a_read_spanning_days_returns_them_in_order(store) -> None:
    loaded = store.load()
    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == store.row_count()


def test_coverage_comes_from_the_manifest_not_a_scan(store) -> None:
    """The manifest is what makes startup O(1) on a store of any size."""
    for path in store.partitions():
        path.unlink()

    first, last = store.coverage()
    assert first is not None and last is not None
    assert store.row_count() == 4 * 375

    store.manifest.forget("candles", store.key)
    assert store.coverage() == (None, None) or store.load().empty


def test_empty_store_answers_cleanly(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    assert not store.exists()
    assert store.load().empty
    assert store.coverage() == (None, None)
    assert store.row_count() == 0
    assert "no data" in store.describe()


def test_deduplication_is_last_write_wins(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    store.write(day_frame("2026-03-10", price=100.0))
    store.write(day_frame("2026-03-10", price=200.0))
    loaded = store.load()
    assert len(loaded) == 5
    assert set(loaded["close"]) == {200.0}


# ------------------------------------------------------------------- manifest


def test_manifest_records_coverage_and_counts(store, tmp_path) -> None:
    manifest = StoreManifest(tmp_path / "manifest.json")
    entry = manifest.dataset("candles", store.key)
    assert entry["rows"] == store.row_count()
    assert entry["sessions"] == len(store.partitions())
    assert entry["instrument"] == "NSE_INDEX|Nifty 50"
    assert manifest.coverage("candles", store.key)[0] is not None


def test_a_corrupt_manifest_is_rebuilt_not_fatal(tmp_path, store) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{ this is not json")
    fresh = StoreManifest(path)
    assert fresh.read()["datasets"] == {}
    store.write(day_frame("2026-05-01"))
    assert StoreManifest(path).row_count("candles", store.key) > 0


# ------------------------------------------------------------------ migration


def test_legacy_single_file_is_imported_once(tmp_path, bars) -> None:
    legacy = tmp_path / "candles.parquet"
    normalize_candles(bars).to_parquet(legacy)

    store = PartitionedStore(tmp_path, "NIFTY")
    rows = store.migrate_legacy()

    assert rows == len(bars)
    assert store.row_count() == len(bars)
    assert not legacy.exists(), "the old file should be moved aside, not left to confuse"
    assert (tmp_path / "candles.parquet.migrated").exists()

    assert store.migrate_legacy() == 0, "a second migration must be a no-op"


def test_migration_with_no_legacy_file_does_nothing(tmp_path) -> None:
    assert PartitionedStore(tmp_path, "NIFTY").migrate_legacy() == 0


# --------------------------------------------------------------- hour shards


def ticks_frame(hours: int = 2, per_hour: int = 60) -> pd.DataFrame:
    frames = []
    for hour in range(9, 9 + hours):
        index = pd.date_range(
            f"2026-03-10 {hour:02d}:00", periods=per_hour, freq="1s", tz=IST
        )
        frames.append(pd.DataFrame({"ltp": range(per_hour), "ltq": 1}, index=index))
    return pd.concat(frames)


def test_ticks_shard_by_hour(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    store.write(ticks_frame(hours=2, per_hour=60))

    files = store.partitions()
    assert len(files) == 2
    assert {path.stem for path in files} == {"09", "10"}
    assert files[0].parent.name == "2026-03-10"


def test_a_tick_range_read_uses_the_hour(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    store.write(ticks_frame(hours=2, per_hour=60))

    within = store.load(
        start=pd.Timestamp("2026-03-10 09:00:30", tz=IST),
        end=pd.Timestamp("2026-03-10 09:00:40", tz=IST),
    )
    assert 10 <= len(within) <= 11
    assert len(store.partitions(
        pd.Timestamp("2026-03-10 09:30", tz=IST), pd.Timestamp("2026-03-10 09:45", tz=IST)
    )) == 1


def test_tick_columns_survive_the_store(tmp_path) -> None:
    """Forcing ticks through the candle schema would discard everything useful."""
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    store.write(ticks_frame(hours=1, per_hour=10))
    loaded = store.load()
    assert list(loaded.columns) == ["ltp", "ltq"]


# ------------------------------------------------------------ multi-level index


def test_a_chain_snapshot_keeps_every_strike(tmp_path) -> None:
    """Eighteen strikes share one timestamp; an index of time alone keeps one."""
    store = PartitionedStore(tmp_path, "NIFTY-chain", dataset=CHAIN, shard=HOUR, normalizer=normalize_frames)
    chain = synthetic_chain(24_150.0, datetime(2026, 3, 10, 9, 15, tzinfo=IST), atm_iv=0.12, width=4)

    recorder = ChainRecorder(store)
    recorder.record(chain, now_seconds=0.0)

    loaded = store.load()
    assert len(loaded) == len(chain.quotes)
    assert loaded.index.names == ["ts", "strike", "kind"]
    assert "premium" in loaded.columns and "iv" in loaded.columns


def test_a_second_chain_sample_appends(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY-chain", dataset=CHAIN, shard=HOUR, normalizer=normalize_frames)
    recorder = ChainRecorder(store)
    first = synthetic_chain(24_150.0, datetime(2026, 3, 10, 9, 15, tzinfo=IST), atm_iv=0.12, width=4)
    second = synthetic_chain(24_160.0, datetime(2026, 3, 10, 9, 16, tzinfo=IST), atm_iv=0.12, width=4)

    recorder.record(first, now_seconds=0.0)
    recorder.record(second, now_seconds=100.0)

    loaded = store.load()
    assert len(loaded) == len(first.quotes) + len(second.quotes)
    assert loaded.index.get_level_values("ts").nunique() == 2


# ------------------------------------------------------------------ recorders


def tick(second: int, price: float = 24_000.0, hour: int = 9) -> Tick:
    return Tick(ts=datetime(2026, 3, 10, hour, 15, tzinfo=IST) + timedelta(seconds=second), ltp=price, ltq=5)


def test_tick_recorder_buffers_until_it_is_full(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    recorder = TickRecorder(store=store, buffer_size=3)

    for index in range(2):
        recorder.record(tick(index))
    assert recorder.buffered == 2
    assert recorder.flushes == 0, "writing per tick would spend the session in the filesystem"

    recorder.record(tick(3))
    assert recorder.flushes == 1
    assert recorder.buffered == 0
    assert store.row_count() == 3


def test_tick_recorder_flushes_at_the_hour_boundary(tmp_path) -> None:
    """The hour is the partition key, so a buffered row must land before it turns.

    The flush writes the hour that *ended*; the ticks that opened the new hour are
    still buffered, which is the point — writing them now would mean reopening a
    shard on every tick that crosses a boundary.
    """
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    recorder = TickRecorder(store=store, buffer_size=1_000)

    recorder.record(tick(0, hour=9))
    recorder.record(tick(0, hour=10))

    assert recorder.flushes == 1
    assert {path.stem for path in store.partitions()} == {"09"}
    assert recorder.buffered == 1

    recorder.close()
    assert {path.stem for path in store.partitions()} == {"09", "10"}
    assert store.row_count() == 2


def test_tick_recorder_writes_the_fields_the_tape_carries(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    recorder = TickRecorder(store=store, buffer_size=1)
    recorder.record(
        Tick(
            ts=datetime(2026, 3, 10, 9, 15, tzinfo=IST),
            ltp=24_000.0,
            ltq=10,
            bid_p=23_999.5,
            ask_p=24_000.5,
            bid_q=100,
            ask_q=120,
            volume_traded=5_000,
            open_interest=1_500.0,
            total_buy_qty=900.0,
            total_sell_qty=700.0,
        )
    )
    loaded = store.load()
    for column in ("ltp", "ltq", "bid_p", "ask_p", "bid_q", "ask_q", "oi", "buy_qty", "sell_qty"):
        assert column in loaded.columns, column
    assert loaded["ltp"].iloc[0] == pytest.approx(24_000.0)
    assert loaded["buy_qty"].iloc[0] == pytest.approx(900.0)


def test_tick_recorder_close_flushes(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    recorder = TickRecorder(store=store, buffer_size=100)
    recorder.record(tick(0))
    assert recorder.buffered == 1

    recorder.close()
    assert recorder.buffered == 0
    assert store.row_count() == 1
    assert "written" in recorder.describe()


def test_tick_recorder_survives_writing_nothing(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY", dataset=TICKS, shard=HOUR, normalizer=normalize_frames)
    recorder = TickRecorder(store=store)
    assert recorder.flush() == 0
    recorder.close()


def test_chain_recorder_respects_its_sample_interval(tmp_path) -> None:
    """Polling a chain faster than it changes mostly records the same numbers twice."""
    store = PartitionedStore(tmp_path, "NIFTY-chain", dataset=CHAIN, shard=HOUR, normalizer=normalize_frames)
    recorder = ChainRecorder(store, sample_seconds=60.0)

    assert recorder.due(0.0) is True
    recorder.record(
        synthetic_chain(24_150.0, datetime(2026, 3, 10, 9, 15, tzinfo=IST), atm_iv=0.12, width=2),
        now_seconds=0.0,
    )
    assert recorder.due(30.0) is False
    assert recorder.due(61.0) is True


def test_chain_recorder_reports_what_it_wrote(tmp_path) -> None:
    store = PartitionedStore(tmp_path, "NIFTY-chain", dataset=CHAIN, shard=HOUR, normalizer=normalize_frames)
    recorder = ChainRecorder(store)
    chain = synthetic_chain(24_150.0, datetime(2026, 3, 10, 9, 15, tzinfo=IST), atm_iv=0.12, width=2)
    written = recorder.record(chain, now_seconds=0.0)
    assert written == len(chain.quotes)
    assert "chain samples" in recorder.describe()


# ------------------------------------------------------------------ normalise


def test_normalize_frames_handles_empty() -> None:
    assert normalize_frames(pd.DataFrame()).empty
    assert normalize_frames(None).empty


def test_normalize_frames_sorts_and_dedupes() -> None:
    index = pd.DatetimeIndex(
        ["2026-03-10 09:16", "2026-03-10 09:15", "2026-03-10 09:15"], tz=IST
    )
    frame = pd.DataFrame({"ltp": [2.0, 1.0, 9.0]}, index=index)
    out = normalize_frames(frame)
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert out["ltp"].iloc[0] == pytest.approx(9.0), "the last write must win"
