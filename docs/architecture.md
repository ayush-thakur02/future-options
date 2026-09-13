# Architecture

## Data flow

```
                        ┌──────────────────────────────────────┐
   Upstox v3 WebSocket  │  upstox_feed.py                      │
   (protobuf frames) ──▶│  decode → Tick                       │
                        └───────────────┬──────────────────────┘
                                        │
                        ┌───────────────▼──────────────────────┐
                        │  aggregator.py                       │
                        │  tick → 1-minute OHLCV bar           │
                        └───────────────┬──────────────────────┘
                                        │ on bar close
                        ┌───────────────▼──────────────────────┐
                        │  features/pipeline.py                │
                        │  123 columns, 117 model-usable       │
                        └───────┬───────────────────┬──────────┘
                                │                   │
              ┌─────────────────▼──────┐   ┌────────▼─────────────────┐
              │  strategies/           │   │  ml/predictor.py         │
              │  18 rules + ensemble   │   │  calibrated P(up)        │
              │  signed conviction     │   │  + expected move         │
              └─────────────────┬──────┘   └────────┬─────────────────┘
                                │                   │
                        ┌───────▼───────────────────▼──────────┐
                        │  live/engine.py                      │
                        │  MarketSnapshot                      │
                        └───────────────┬──────────────────────┘
                                        │
                        ┌───────────────▼──────────────────────┐
                        │  ui/dashboard.py                     │
                        │  cost decision first                 │
                        └──────────────────────────────────────┘
```

The same `build_features` → `strategies` → `predictor` path serves both the live
dashboard and the backtester. That is deliberate: a strategy that looks good in
research is literally the code that runs live, not a re-implementation of it.

## Module map

```
src/niftypulse/
├── config.py              Settings, credential resolution, cost hurdle derivation
├── models.py              Tick, Signal, Prediction, Trade, Direction, MarketSnapshot
├── trading_calendar.py    NSE session clock, holiday bookkeeping
├── cli.py                 Typer entrypoints
│
├── data/
│   ├── upstox_auth.py     OAuth 2.0 flow, token lifecycle and expiry
│   ├── upstox_rest.py     Historical / intraday / quote endpoints, rate limiting
│   ├── upstox_feed.py     v3 WebSocket client, protobuf decoding
│   ├── aggregator.py      Tick → candle, session-aware bucketing
│   ├── resample.py        OHLCV aggregation across timeframes
│   ├── store.py           Parquet persistence, deduplication
│   ├── synthetic.py       Offline data source
│   └── proto/             MarketDataFeed.proto + generated stubs
│
├── features/
│   ├── indicators.py      40+ indicators, no TA-Lib dependency
│   ├── context.py         Session, calendar, prior-session, regime features
│   └── pipeline.py        Feature matrix assembly
│
├── strategies/
│   ├── base.py            Strategy ABC, vectorised contract, CompositeStrategy
│   ├── trend.py           5 strategies
│   ├── reversion.py       5 strategies
│   ├── momentum.py        4 strategies
│   ├── volatility.py      4 strategies
│   └── registry.py        Registry, MLStrategy, HybridStrategy, ensemble weights
│
├── ml/
│   ├── dataset.py         Dead-banded, hurdle-aware labelling
│   ├── splits.py          Purged, embargoed walk-forward CV
│   ├── models.py          4 base learners + soft-voting ensemble
│   ├── calibration.py     Gated Platt / isotonic calibration
│   ├── metrics.py         Lift-aware evaluation, expected-move curve
│   ├── trainer.py         Training orchestration, artifact persistence
│   └── predictor.py       Inference
│
├── backtest/
│   ├── costs.py           Indian derivatives cost model
│   ├── engine.py          Event-driven execution
│   └── report.py          Performance statistics
│
├── live/engine.py         Orchestrator: feed → features → strategies → forecasts
└── ui/
    ├── charts.py          Candlestick, probability bars, sparklines
    └── dashboard.py       Rich layout
```

## Design decisions

These are the choices that shape the platform. Each one was made to avoid a
specific failure mode.

### Strategies are vectorised, not event-driven

The conventional design evaluates a rule per bar inside a loop. That makes
research slow, and it means the live path and the research path are two different
implementations that can silently diverge.

Here a strategy receives the full feature frame and returns a signed conviction
*series* in `[-1, 1]`. A backtest is then a single pass over a Series, live
inference is `.iloc[-1]`, and the two cannot drift apart. See
[Strategies](strategies.md).

### Features are computed on a rolling window in live, but full history in training

Rebuilding 120 days of indicators on every bar close would be wasteful. Live uses
the last 2,000 bars instead.

The window is sized so the longest recursive indicator (EMA-200) has long since
converged — after 2,000 bars the residual influence of the seed value is around
1e-8. A shorter window would introduce a train/serve skew that degrades every
prediction quietly. This is why `FEATURE_WINDOW` is 2000 and not something
smaller.

The strategy layer gets a shorter tail (600 bars) for a different reason: 24
strategies evaluated over 2,000 bars dominated the live loop, and the longest
lookback any strategy applies internally is about 100 bars. See
[SIGNAL_WINDOW](../src/niftypulse/live/engine.py).

### Costs are a first-class input, not a reporting afterthought

The cost model feeds backwards into labelling, into the signal gate, and into the
UI. Most platforms apply costs at the end and discover the strategy never had an
edge. See [Scalping economics](scalping-economics.md).

### Validation purges overlapping labels

Standard k-fold is invalid here. A label at bar *t* uses prices through *t + h*,
so a training sample can overlap a test sample and the model effectively sees the
answer. Purging and embargoing remove that. See [ML pipeline](ml-pipeline.md).

### Calibration is gated twice, and defaults to Platt

A Brier-score improvement alone is not sufficient justification for calibrating.
When a model has no signal, mapping toward the base rate *always* improves Brier
while flattening the distribution a threshold-based strategy needs. The
calibrator is rejected unless it improves Brier **and** preserves the score
spread. See [ML pipeline](ml-pipeline.md).

### Execution enters on the next bar's open

A signal computed from a bar's close cannot be filled at that close — by the time
you know the signal, that price is gone. Every backtest enters at the next bar's
open, which is the first price a decision made at the close can actually get.
Trades that would cross the session close are skipped. See
[Backtesting](backtesting.md).

### The index has no volume, and the code says so

NIFTY 50 is an index. The exchange publishes no traded volume and no order book
for it. Rather than feed a column of zeros into volume indicators — which would
produce constant, meaningless signals — the aggregator records `tick_count` as an
activity proxy and the feature layer falls back to it. Strategies needing order
book depth return **no opinion** rather than a fabricated one.

This is why `order_flow` and `activity_spike` fire on 0% of bars against the
index. It is correct behaviour, not a dead strategy. Subscribe to the front-month
NIFTY future to activate them.

## Latency

Measured on the bundled synthetic data, per bar close:

| Stage | Cost |
|---|---|
| `build_features` (2,000 bars) | ~50 ms |
| `_collect_signals` (24 strategies, 600-bar tail) | ~150 ms |
| Render | ~3 ms |
| **Total** | **~204 ms** |

Bars close once per minute, so the live loop uses about 0.3% of the interval.
Latency only becomes a constraint if you move to sub-minute bars.

`hurst_exponent` was originally 71% of feature time, running a Python callback
for every bar, each looping over the lags. It is now vectorised across windows
via `sliding_window_view`, verified against the reference implementation to
2.2e-16, and roughly 7x faster.

## Storage

| Format | Why |
|---|---|
| Parquet for candles | Columnar, typed, fast range scans, no server |
| joblib for models | Handles sklearn Pipelines and numpy arrays natively |
| JSON for tokens and reports | Human-readable, easy to inspect when something breaks |

Candle writes are deduplicated on the timestamp index, so re-fetching an
overlapping window is idempotent. This matters because the current session is
routinely re-pulled.

## What is deliberately absent

**No order placement.** The platform stops at forecasting. SEBI's algorithmic
trading framework imposes registration and order-tagging obligations as soon as
you connect to a broker's order API, and that is a decision to make
deliberately, not by accident.

**No TA-Lib.** It requires a C toolchain and is the single most common reason a
Python trading project fails to install. Every indicator is implemented directly
on pandas, which also makes the smoothing conventions explicit rather than hidden
behind a library's defaults.

**No database.** Parquet files are sufficient at this scale and remove an entire
class of operational problems.
