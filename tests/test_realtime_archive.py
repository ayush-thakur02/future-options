"""Durability and reuse contracts for the real-time market-data archive."""

from __future__ import annotations

import multiprocessing
import os
from datetime import datetime

import pandas as pd
import pytest

from core.calendar import IST
from core.settings import Settings
from core.types import Tick
from plugins.sources.history import (
    HOUR,
    TICKS,
    HistorySource,
    PartitionedStore,
    TickRecorder,
    normalize_frames,
)
from plugins.sources.history.recorder import tick_journal_path


def _tick_store(root, instrument: str = "NSE_INDEX|Nifty 50") -> PartitionedStore:
    return PartitionedStore(
        root,
        instrument,
        dataset=TICKS,
        shard=HOUR,
        normalizer=normalize_frames,
    )


def _abrupt_writer(root: str, stamp: str) -> None:
    """Child target: leave only the journal, as a killed process would."""
    store = _tick_store(root)
    recorder = TickRecorder(store, buffer_size=100, journal_sync_every=1)
    moment = pd.Timestamp(stamp).to_pydatetime()
    recorder.record(Tick(ts=moment, ltp=24_000.0, ltq=2, instrument_key=store.instrument))
    recorder.record(Tick(ts=moment, ltp=24_002.0, ltq=3, instrument_key=store.instrument))
    os._exit(0)


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, model_dir=tmp_path / "models", log_dir=tmp_path / "logs")


def test_same_timestamp_keeps_distinct_updates_and_deduplicates_replays(tmp_path) -> None:
    store = _tick_store(tmp_path)
    recorder = TickRecorder(store, buffer_size=3)
    stamp = datetime(2026, 9, 14, 9, 15, tzinfo=IST)

    first = Tick(ts=stamp, ltp=24_000.0, ltq=2, instrument_key=store.instrument)
    changed = Tick(ts=stamp, ltp=24_001.0, ltq=2, instrument_key=store.instrument)
    recorder.record(first)
    recorder.record(first)  # reconnect replay of the exact same exchange update
    recorder.record(changed)
    recorder.close()

    tape = store.load()
    assert len(tape) == 2
    assert list(tape["ltp"]) == [24_000.0, 24_001.0]
    assert tape["event_id"].nunique() == 2


def test_history_startup_recovers_a_killed_recorder_and_reuses_its_candle(tmp_path) -> None:
    stamp = "2026-09-14T09:15:20+05:30"
    process = multiprocessing.get_context("spawn").Process(
        target=_abrupt_writer,
        args=(str(tmp_path), stamp),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    tick_store = _tick_store(tmp_path)
    assert tick_journal_path(tick_store).exists()
    assert not tick_store.exists(), "the child was killed before its parquet checkpoint"

    source = HistorySource(_settings(tmp_path), broker=None)
    bars = source.load_cached()

    assert len(bars) == 1
    assert bars.iloc[0]["open"] == pytest.approx(24_000.0)
    assert bars.iloc[0]["high"] == pytest.approx(24_002.0)
    assert not tick_journal_path(tick_store).exists()
    tick_entry = tick_store.manifest.dataset(TICKS, tick_store.key)
    assert tick_entry["rows"] == 2
    assert tick_entry["recovered_rows"] == 2


def test_archived_closed_minute_prevents_an_unnecessary_provider_pull(tmp_path) -> None:
    now = pd.Timestamp.now(tz=IST)
    stamp = (now.floor("min") - pd.Timedelta(minutes=1) + pd.Timedelta(seconds=30)).to_pydatetime()
    settings = _settings(tmp_path)
    requested_start = (now - pd.Timedelta(days=5)).normalize()
    seed_index = pd.DatetimeIndex([requested_start + pd.Timedelta(hours=9, minutes=15)])
    seed = pd.DataFrame(
        {
            "open": [24_000.0],
            "high": [24_000.0],
            "low": [24_000.0],
            "close": [24_000.0],
            "volume": [0.0],
            "oi": [0.0],
        },
        index=seed_index,
    )
    HistorySource(settings).seed(seed)
    store = _tick_store(tmp_path)
    recorder = TickRecorder(store, buffer_size=1)
    recorder.record(Tick(ts=stamp, ltp=24_100.0, instrument_key=store.instrument))
    recorder.close()

    class Broker:
        token = "configured"

        def __init__(self) -> None:
            self.calls = 0

        def rest(self):
            self.calls += 1
            raise AssertionError("locally covered history must not hit the provider")

        def close(self) -> None:
            return None

    broker = Broker()
    bars = HistorySource(settings, broker=broker).load_history(days=5, quiet=True)

    assert not bars.empty
    assert bars.index[-1] == pd.Timestamp(stamp).floor("min")
    assert broker.calls == 0


def test_manifest_carries_storage_and_archive_coverage_metadata(tmp_path) -> None:
    store = _tick_store(tmp_path)
    recorder = TickRecorder(store, buffer_size=1)
    recorder.record(
        Tick(
            ts=datetime(2026, 9, 12, 10, 0, tzinfo=IST),
            ltp=100.0,
            instrument_key=store.instrument,
            greeks={"delta": 0.51, "gamma": 0.002, "iv": 14.2},
        )
    )
    recorder.close()

    entry = store.manifest.dataset(TICKS, store.key)
    assert entry["coverage"]["rows"] == 1
    assert entry["storage"]["partition"] == HOUR
    assert entry["storage"]["deduplication"] == "timestamp+event_id"
    assert {"delta", "gamma", "iv", "greeks_json"} <= set(entry["columns"])


def test_failed_atomic_rewrite_keeps_the_previous_parquet_readable(tmp_path, monkeypatch) -> None:
    store = PartitionedStore(tmp_path, "NIFTY")
    stamp = pd.DatetimeIndex(["2026-09-14 09:15"], tz=IST)
    original = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1], "oi": [0]},
        index=stamp,
    )
    store.write(original)

    real_to_parquet = pd.DataFrame.to_parquet

    def interrupted(frame, path, *args, **kwargs):
        real_to_parquet(frame, path, *args, **kwargs)
        raise OSError("simulated interruption before rename")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    revised = original.assign(close=999.0)
    with pytest.raises(OSError, match="simulated interruption"):
        store.write(revised)

    assert pd.read_parquet(store.partitions()[0])["close"].iloc[0] == pytest.approx(100.0)
