# Architecture

Three layers, and the rule that keeps them apart: **`core` knows nothing,
`kernel` knows plugins, `plugins` know their job, `runtime` decides which ones to
use.**

```
core/       domain vocabulary, settings, calendar, the OHLCV schema
kernel/     the plugin runtime: discovery, capability wiring, the event bus
plugins/    everything that does work, as packs of one or two modules each
runtime/    composition: what a run needs and how it is driven
backtest/   offline research against the same code the live path runs
webapp.py   the launcher: flags in, one dashboard out
```

---

## The plugin tree

```
src/
├── core/               types, settings, calendar, bars — no dependencies above it
├── kernel/             contracts, registry, loader, bus, context, kernel
├── plugins/
│   ├── sources/
│   │   ├── upstox/         OAuth, REST, the v3 WebSocket feed, proto stubs
│   │   ├── simulated/      generated series, clock-paced feed, tick expansion
│   │   ├── history/        partitioned store, manifest, recorders, resampling
│   │   └── option_chain/   pricing, the chain, the money legs
│   ├── aggregators/candle_builder/     tick → session-anchored time bars
│   ├── features/technical/             indicators, session context, assembly
│   ├── strategies/
│   │   ├── trend/ momentum/ reversion/ volatility/     nineteen rules
│   │   ├── channels/ flow/ regime/ statistical_anchor/ eleven rules
│   │   └── ml_forecast/                 the model as a strategy
│   ├── forecasts/
│   │   ├── ml_ensemble/    labelling, splits, models, calibration, inference
│   │   ├── projection/     the next three candles, re-projected every second
│   │   └── online_research/ six causal incremental classifiers and ledgers
│   ├── advisory/
│   │   ├── breakeven_gate/  does the move pay for the position?
│   │   ├── performance_ledger/ strategy accuracy, P&L, drawdown and trust
│   │   └── money_simulator/ funded paper book: wallets, reserve, real rupees
│   └── renderers/
│       └── web/            Flask snapshot API, SVG charts, browser terminal
├── runtime/            session, engine, board, nowcast, conviction, bars, warmup
├── backtest/           costs, engine, report
└── webapp.py           the launcher that serves the dashboard
```

What is actually registered is the kernel's own index:
**22 bundled plugins, 44 capabilities**.

---

## Data flow

```
                 ┌──────────────────────────────────────────┐
 Upstox v3 WS ──▶│ source:upstox     protobuf → Tick        │
 or simulated ──▶│ source:simulated  clock-paced replay     │
                 └───────────────────┬──────────────────────┘
                                     │ Tick
                 ┌───────────────────▼──────────────────────┐
                 │ runtime/session  record the tape ────────┼──▶ data/ticks/…
                 └───────────────────┬──────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │ runtime/board: the index, and the legs it implies       │
        │                                                          │
        │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
        │  │ aggregator   │   │ aggregator   │   │ aggregator   │ │
        │  │ index        │   │ CE premium   │   │ PE premium   │ │
        │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
        │         │ bars            │                  │          │
        │  ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐ │
        │  │ features     │   │ features     │   │ features     │ │
        │  │ strategies   │   │ strategies   │   │ strategies   │ │
        │  │ ml_ensemble  │   │ ml_ensemble  │   │ ml_ensemble  │ │
        │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
        │         │                   │                  │         │
        │  ┌──────▼───────────────────▼──────────────────▼───────┐ │
        │  │ forecast:projection  next 3 candles each, every 1s  │ │
        │  └──────────────────────┬──────────────────────────────┘ │
        └─────────────────────────┼────────────────────────────────┘
                                  │ BoardSnapshot
                 ┌────────────────▼─────────────────┐
                 │ advisory:breakeven_gate          │
                 │ a verdict per leg, with its maths│
                 └────────────────┬─────────────────┘
                                  │
                 ┌────────────────▼─────────────────┐
                 │ renderer:web                     │
                 │ charts, verdicts, greeks, scores │
                 └──────────────────────────────────┘
```

One tick fans out to three engines. The legs have no feed of their own — a leg's
premium **is** a function of the index price, the strike, the time left and the
implied vol — so deriving a leg tick from an index tick is the definition rather
than a substitute for a missing feed.

---

## Two cadences

The engine does different amounts of work at different moments, on purpose.

**On a tick** (thousands a minute on an index feed): update the anchor price and
fold the tick into the momentum estimate. Nothing else. A tick that starts a new
bar also closes the old one, which is the expensive path.

**On a bar close**: rebuild the 147-column feature matrix, score thirty strategies,
observe exact-target AI and rule outcomes, issue new one-/two-/three-bar AI
forecasts and three-bar rule forecasts, blend the ensemble view, score projected
candles, and redraw the path.

**Every second** (`NowcastLoop`): rebuild the projected path from the live price
and the current view, and publish. Ticks drive the anchor; the clock drives the
path. Keeping them apart is what makes the projection tick every second on a quiet
tape instead of only when a print arrives.

Measured: one refresh across a three-leg board is **~10–17 ms** against a 1,000 ms
budget.

---

## Design decisions

Each of these was made to avoid a specific failure mode.

### Plugins link by capability, not by import

`source:history` does not import the Upstox client; it declares
`requires=("broker",)` and asks the kernel for it. The alternative is what the
platform actually started with: the candle cache imported the provider's REST
client directly, so pointing the platform at a different provider meant editing
four modules.

The visible consequence is that the plugin index can answer "what is wired to
what" without running anything, and `kernel.validate()` reports every unsatisfied
requirement at once.

### Features are computed on a rolling window live, full history in training

Rebuilding 120 days of indicators on every bar close is wasteful. Live uses the
last 2,000 bars, sized so the longest recursive indicator (EMA-200) has long since
converged — after 2,000 bars the residual influence of the seed is around 1e-8.
A shorter window would introduce a train/serve skew that silently degrades every
prediction.

The strategy layer gets a 600-bar tail for a different reason: evaluating the full
strategy catalog across 2,000 bars dominated the live loop, and the longest lookback any
strategy applies internally is about 100 bars.

### Costs are a first-class input, not a reporting afterthought

The cost model feeds backwards into labelling, into the signal gate, into the
advisory verdict, and into the UI. Most platforms apply costs at the end and
discover the strategy never had an edge. See
[Scalping economics](scalping-economics.md).

### Strategies are vectorised, not event-driven

A strategy receives the full feature frame and returns a signed conviction
*series*. A backtest is then a single pass over a Series, live inference is
`.iloc[-1]`, and the two cannot drift apart — the strategy that looks good in
research is literally the strategy running live.

### Anything shared by several plugins moves to `core`

The OHLCV schema started in the cache, and the simulated generator imported the
cache's module to normalise its own output while the provider client imported it
too — three packs coupled through a utility function. It lives in `core/bars.py`
now, with the aggregator, both sources and the engine agreeing on it.

### Validation purges overlapping labels

A label at bar *t* uses prices through *t + h*, so a standard k-fold training
sample can overlap a test sample and the model effectively sees the answer.
Purging and embargoing remove that. See [ML pipeline](ml-pipeline.md).

### Calibration is gated twice, and defaults to Platt

A Brier-score improvement alone does not justify calibrating. When a model has no
signal, mapping toward the base rate *always* improves Brier while flattening the
distribution a threshold strategy needs. The calibrator is rejected unless it
improves Brier **and** preserves the score spread.

### Execution enters on the next bar's open

A signal computed from a bar's close cannot be filled at that close. Every
backtest enters at the next bar's open, and trades that would cross the session
close are skipped rather than carried overnight.

### The projected path is scored, not just drawn

A projection is easy to make look convincing in motion. The tracker records every
projection against its target bar and grades it when that bar closes, per horizon,
and the panel prints the score. See [Projection](projection.md).

---

## Latency

Measured on the bundled generated data.

**Startup**, opening the dashboard on a 45,000-bar cache:

| Stage | Cost |
|---|---|
| Load the bars | ~100 ms |
| Warm three instruments (features + strategy catalog each) | hardware-dependent |
| **First frame on screen** | **~0.6 s** |

It was 40 seconds, and the whole of it was one function: pricing premium candles
for the legs called a day-looping session clock once per bar *per leg* across the
entire history, and that call's cost grew with the distance to the expiry. The
clock and the premium path are vectorised now, the board carries only the window
its engines actually read, and the model no longer spawns a worker pool to predict
a single row. Per-refresh cost across the board is **~7–9 ms**.

**Steady state**, per bar close:

| Stage | Cost |
|---|---|
| `build_features` (2,000 bars) | ~50 ms |
| strategies (30 rules, 600-bar tail) | hardware-dependent |
| model inference (single row, parallel inference off) | ~40 ms |
| projection refresh (whole board, 1 Hz) | ~7–9 ms |
| render | ~3 ms |

Bars close once a minute, so the expensive path uses well under 1% of the
interval. Two earlier wins are worth keeping in mind: `hurst_exponent` was 71% of
feature time before it was vectorised across windows via `sliding_window_view`
(roughly 7x faster), and LightGBM/sklearn parallelism had to be turned *off* for
inference — `ExtraTrees(n_jobs=-1)` runs its trees through joblib's process
backend, so every single-row `predict_proba` started a loky pool.

Anything that runs per bar, per instrument or per second is array arithmetic
rather than a Python loop. When a loop is unavoidable it is commented as such.

---

## Storage

| Format | Why |
|---|---|
| Parquet, partitioned by day and hour | A write costs one file, a read costs one range. See [Storage](storage.md) |
| joblib for models | Handles sklearn Pipelines and numpy arrays natively |
| JSON for tokens and manifests | Human-readable, and inspectable when something breaks |

---

## What is deliberately absent

**No order placement.** The platform stops at forecasting. SEBI's algorithmic
trading framework imposes registration and order-tagging obligations as soon as
you connect to a broker's order API, and that is a decision to make deliberately,
not by accident.

**No TA-Lib.** It needs a C toolchain and is the most common reason a Python
trading project fails to install. Every indicator is implemented on pandas, which
also makes the smoothing conventions explicit rather than hidden behind a
library's defaults.

**No database.** Parquet files are sufficient at this scale and remove an entire
class of operational problems.
