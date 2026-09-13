"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plugins.sources.simulated.series import generate_candles

SEED = 1234


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    """A modest synthetic history, small enough to keep tests fast."""
    return generate_candles(days=25, seed=SEED)


@pytest.fixture(scope="session")
def long_bars() -> pd.DataFrame:
    return generate_candles(days=120, seed=SEED + 1)


@pytest.fixture(scope="session")
def trend_bars() -> pd.DataFrame:
    """A deterministic series with a genuine, learnable drift.

    Alternating up and down runs of fixed length. Any model with access to recent
    returns should detect this trivially, which makes it a positive control: if a
    model cannot learn this, the pipeline is broken rather than the data being
    merely hard.
    """
    bars = generate_candles(days=40, seed=SEED + 2)
    n = len(bars)
    run_length = 40
    direction = np.repeat([1.0, -1.0], run_length)
    direction = np.resize(direction, n)

    step = 2.5
    close = 24_000.0 + np.cumsum(direction * step)
    frame = pd.DataFrame(index=bars.index)
    frame["close"] = close
    frame["open"] = np.concatenate([[close[0]], close[:-1]])
    frame["high"] = np.maximum(frame["open"], frame["close"]) + 1.0
    frame["low"] = np.minimum(frame["open"], frame["close"]) - 1.0
    frame["volume"] = 0.0
    frame["oi"] = 0.0
    return frame


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def offline_session():
    """A composed session on generated data, for the runtime tests.

    Session-scoped because building it loads the kernel and warms the engine,
    which is the expensive part and is identical for every test that reads from
    it. Tests that need to mutate it build their own.
    """
    from core.settings import Settings
    from kernel import Kernel
    from runtime.session import Session, SessionConfig

    session = Session(
        kernel=Kernel.bootstrap(Settings()),
        config=SessionConfig(offline=True, days=3, refresh_history=False),
    )
    session.bootstrap()
    return session
