# NIFTY Pulse — Documentation

A scalping platform for NIFTY 50: live Upstox data in, cost-aware directional
forecasts out.

---

## Start here

| If you want to... | Read |
|---|---|
| Get it running | [Getting started](getting-started.md) |
| Understand why the platform behaves the way it does | [Scalping economics](scalping-economics.md) |
| Understand how the pieces fit | [Architecture](architecture.md) |
| Know what every command does | [CLI reference](cli-reference.md) |
| Know what every setting does | [Configuration](configuration.md) |
| Interpret a training run | [ML pipeline](ml-pipeline.md) |
| Read the dashboard | [Dashboard](dashboard.md) |
| Know what the platform refuses to do, and why | [Scalping economics](scalping-economics.md) |
| Contribute | [Development](development.md) |
| Fix something broken | [Troubleshooting](troubleshooting.md) |

---

## The short version

Scalping is a cost game before it is a prediction game. A NIFTY round trip costs
roughly **3.9 basis points**. Most 1-minute price moves are smaller than that, so
most scalps are unprofitable regardless of how good the forecast is.

This shapes the entire platform:

- Training **refuses** to learn moves that cannot pay for themselves.
- Signals stay **silent** unless the expected move clears the round trip.
- The UI reports the **cost decision**, not just the direction.
- The most common correct output is **no trade**.

The measured ceiling, on the bundled synthetic data:

| Horizon | Median move | Clears 3.90 bp hurdle |
|---|---|---|
| 1m | 1.30 bp | 5.7% |
| 2m | 1.80 bp | 16.0% |
| 3m | 2.20 bp | 24.3% |
| 5m | 2.87 bp | 36.3% |

Full reasoning in [Scalping economics](scalping-economics.md). That document is
worth reading before tuning anything.

---

## Documentation map

### Concepts
- [Scalping economics](scalping-economics.md) — the cost hurdle, and why it dominates everything
- [Architecture](architecture.md) — data flow, module map, design decisions

### Components
- [Data layer](data-layer.md) — Upstox REST, WebSocket, protobuf, aggregation, storage
- [Features](features.md) — all 117 model inputs, grouped and explained
- [Strategies](strategies.md) — the 18 rule-based strategies, the ML strategy, the ensemble
- [ML pipeline](ml-pipeline.md) — labelling, purged validation, models, calibration, metrics
- [Backtesting](backtesting.md) — execution model, cost model, reporting
- [Dashboard](dashboard.md) — reading the terminal UI

### Reference
- [CLI reference](cli-reference.md) — every command and flag
- [Configuration](configuration.md) — every setting

### Working on it
- [Development](development.md) — tests, conventions, extending the platform
- [Troubleshooting](troubleshooting.md) — common failures and what they mean

---

## Status

85 tests, lint clean. Verified end to end on macOS with Python 3.14.7.

The platform places **no orders** and connects to **no broker's order routing**.
It is research software. See the disclaimer at the end of the main README.
