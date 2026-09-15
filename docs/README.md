# NIFTY Pulse — Documentation

A scalping platform for NIFTY 50: live Upstox data in, cost-aware forecasts and
projected candles out, with the calls and the puts beside the index.

---

## Start here

| If you want to… | Read |
|---|---|
| Get it running | [Getting started](getting-started.md) |
| Understand why it behaves as it does | [Scalping economics](scalping-economics.md) |
| Understand how the pieces fit | [Architecture](architecture.md) |
| Add or replace a component | [Plugins](plugins.md) |
| Read the dashboard | [Dashboard](dashboard.md) |
| Know what the blue candles are | [Projection](projection.md) |
| Understand the call/index/put board | [Options](options.md) |
| See what the strategies make in rupees | [Money simulator](money-simulator.md) |
| Know why it is already ready at 09:15 | [Warm-up](warmup.md) |
| Know what is on disk | [Storage](storage.md) |
| Do anything but watch | [Operations](operations.md) |
| Know what every setting does | [Configuration](configuration.md) |
| Interpret a training run | [ML pipeline](ml-pipeline.md) |
| Contribute | [Development](development.md) |
| Fix something broken | [Troubleshooting](troubleshooting.md) |

---

## The short version

Scalping is a cost game before it is a prediction game. A NIFTY round trip costs
roughly **3.9 basis points**. Most 1-minute price moves are smaller than that, so
most scalps are unprofitable no matter how good the forecast is.

This shapes the whole platform:

- Training **refuses** to learn moves that cannot pay for themselves.
- Signals stay **silent** unless the expected move clears the round trip.
- A long option leg is required to clear its **premium, costs and theta** before
  the board will call it a trade — 60–100bp for an at-the-money leg depending on
  the time left, against a median 1-minute move of 1.3bp.
- The projected candles are **scored** against what actually printed.
- The most common correct output is **no trade**.

| Horizon | Median move | Clears the 3.90 bp hurdle |
|---|---|---|
| 1m | 1.30 bp | 5.7% |
| 2m | 1.80 bp | 16.0% |
| 3m | 2.20 bp | 24.3% |
| 5m | 2.87 bp | 36.3% |

Full reasoning in [Scalping economics](scalping-economics.md). Read that before
tuning anything.

---

## Documentation map

### Concepts
- [Scalping economics](scalping-economics.md) — the cost hurdle, and why it dominates everything
- [Architecture](architecture.md) — layer map, data flow, design decisions
- [Plugins](plugins.md) — the plugin contract, kinds, capabilities, how to write one

### Components
- [Data layer](data-layer.md) — Upstox auth, REST, WebSocket, protobuf, aggregation
- [Storage](storage.md) — the partitioned store, the manifest, recording the tape
- [Features](features.md) — all 147 columns, grouped and explained
- [Strategies](strategies.md) — 30 rule-based strategies, the model, and the ensemble
- [ML pipeline](ml-pipeline.md) — labelling, purged validation, models, calibration
- [Projection](projection.md) — the next three candles, and how they are scored
- [Options](options.md) — pricing, the chain, the board, the verdicts
- [Warm-up](warmup.md) — replaying the recent past so the day does not start cold
- [Money simulator](money-simulator.md) — the funded book, the reserve, and real rupees
- [Dashboard](dashboard.md) — reading the browser terminal
- [Backtesting](backtesting.md) — execution model, cost model, reporting

### Reference
- [Operations](operations.md) — login, history, training, backtests, diagnostics
- [Configuration](configuration.md) — every setting

### Working on it
- [Development](development.md) — tests, conventions, extending the platform
- [Troubleshooting](troubleshooting.md) — common failures and what they mean

---

## Status

**Full test suite and lint clean.** Verified end to end on Python 3.14.7 —
live-path logic exercised by tests, the interactive dashboard and the full
pipeline run offline on generated data.

The platform places **no orders** and connects to **no broker's order routing**.
It is research software. See the disclaimer at the end of the main README.
