# Live research dashboard

```bash
niftypulse doctor --live
niftypulse dashboard --workers -1
niftypulse research
niftypulse research --failures --json
niftypulse research --section ai --limit 100
```

In this checkout the executable is `.venv/bin/niftypulse` if the virtual
environment is not activated. A terminal around 180 columns × 44 rows gives the
charts and rule explanations enough room. Use `1` for results, `2` for position
scenarios, `3` for indicators, `4` for online AI, `j`/`k` to scroll strategies,
`r` to refresh projections, and `q` to exit.

## Authentication and market data

Startup validates the environment token with Upstox. If Upstox rejects it with
HTTP 401, the app tries the unexpired token saved by `niftypulse login`. This
fixes an invalid `.env` token hiding a valid saved login. Neither credential is
printed or overwritten. If both are rejected, run `niftypulse login` and restart.
Network errors and rate limits do not trigger a credential fallback.

Live mode first recovers its local tick journal and materializes recorded ticks
into closed candles. The history plugin validates the store's provenance and asks
Upstox only for a missing prefix or tail. Generated prices live under a separate
simulation key and cannot enter the live cache. Current, incomplete bars are
excluded from training. Live mode does not fall back to generated data. Use
`--offline` explicitly for simulation.

The board discovers the nearest unexpired option contracts from Upstox and
selects a matching ATM call/put pair using the current index quote. Each chart
receives its own instrument's historical candles and WebSocket ticks. The pair
is pinned for the session; restart to select a new strike or expiry. The screen
shows the expiry, source and last trade age. Calculated option premiums are used
only in simulation. Live snapshots from a closed market are not new training
bars. Projections are hidden until current regular-session ticks arrive, and
when a stream becomes stale (90 seconds).

The client uses the [Upstox v3 binary subscription protocol](https://upstox.com/developer/api-documentation/v3/get-market-data-feed/),
the [option contract endpoint](https://upstox.com/developer/api-documentation/get-option-contracts/),
and [option-chain context](https://upstox.com/developer/api-documentation/get-pc-option-chain/).

## Thirty strategies and two learning layers

The live engine discovers every rule from eight independent strategy packs. The
original trend, momentum, reversion and volatility families are joined by:

| Pack | Added mathematics |
|---|---|
| Channels | Keltner continuation, regression-channel breakout, sustained channel pressure |
| Flow | Chaikin-flow trend, money-flow reversal, price/OBV divergence |
| Regime | Hurst-adaptive direction, sign entropy, volatility-state rotation |
| Statistical anchor | Beta-neutral spread reversion and relative-strength rotation against the aligned index |

`WAIT` means conditions are not met; `N/A` means the required input is absent.
An index has no order book and some option/futures features lack volume, so those
rules abstain rather than inventing a value. New packs join the default ensemble
at their declared weight without editing a central registry. On option engines,
paired statistical rules receive the aligned index close as their anchor.

Every active rule issues a frozen three-bar research signal. Its ledger records
direction, strength, issue and target prices, estimated cost, exact-target result,
gross and net return, drawdown, and a conservative trust score. A missed target
expires; a later bar is never substituted. Historical warm-up does not count.

Each instrument also has an independent incremental SGD return model. Fixed ATR
scaling avoids a scaler fitted using future observations. Labels become
available only when their target candles close. Missing minutes and overnight
gaps are excluded. Four internal horizons support projecting the next three
complete candles while another candle is forming. Model estimates contribute
at most 25% of projected closes after 80 eligible samples; rules and tick
momentum supply the rest. `--no-learn` disables this online component.

Online models are saved atomically under `artifacts/online/live/`, keyed by
instrument and timeframe. The last target learned is saved per horizon so a
restart does not train twice on the same labels. Simulation uses a separate
directory. Sample counts include warm-up; **new labels** counts only updates
made during this run. Existing batch models remain a separate research facility
through `train`, `models`, and `backtest`; option engines do not use index models.

Alongside that return learner, the `forecast:online_research` plugin runs three
prequential classifiers: online logistic regression, passive-aggressive
classification, and online Gaussian Naive Bayes. Each gets its own immutable
prediction at one-, two-, and three-bar horizons. The plugin learns only after
the exact target closes and stores its model state and ledger in one SQLite
transaction under `data/research/online_ai/{live|simulation}/`.

The AI scorecard retains all-time and rolling accuracy, majority-baseline lift,
Brier score, calibration error, wins, losses, gross/cost/net P&L, profit, loss,
maximum drawdown, pending and expired predictions, and trust. BUY/SELL/HOLD is a
research classification. Low evidence or weak probability produces HOLD. Trusted
AI conviction contributes at most 25% of the slow ensemble view.

## What success means

Blue candles represent the next complete bars after the current/forming bar.
Their timestamps are **bar open times**. A target is observed only after that
bar closes. The chart refreshes continuously, but the scorecard freezes the
first forecast for each target/horizon. It cannot replace a losing forecast with
a later, more convenient forecast.

A directional hit means the forecast and realised close moved in the same
direction from the price at issuance, using a 0.1 basis-point flat band. A flat
forecast is only a hit if the outcome is also flat. MAE is the absolute close
error in basis points of the issue price. This does not assert that every part
of the projected OHLC candle was correct. High/low ranges are heuristics, not
calibrated confidence intervals. Success rate is not trading profitability.

The per-instrument, per-horizon table shows wins, losses, number scored and MAE.
Missing bars are tracked separately, and forecasts outstanding at shutdown are
marked unresolved. No warm-up accuracy is displayed as live accuracy.

`data/research/live.sqlite3` records the instrument, timeframe, run ID, target,
issue timestamp/price, forecast close/range, realised close, result, error and
the regime/strategy/model context. `research --failures --json` exports misses
and that original context. `research --offline` selects the separate simulation
database. Reports aggregate runs per instrument/timeframe/horizon; repeated
observations across separate runs are not independent experiments.

`data/research/strategies.sqlite3` holds the strategy ledger. The online AI
state and prediction ledgers are separated by mode and instrument below
`data/research/online_ai/`. `niftypulse research` prints several normal Rich
tables instead of opening a full-screen view, so the terminal's native scrollback
contains the full report. `--section forecasts|ai|predictions|strategies` narrows
it, `--limit` controls row count, and `--json` emits every selected field.

The position view compares an option's projected resale premium with entry premium,
quoted spread and configured round-trip charge assumptions. It does not require
recovering the entire option premium during a scalp. Contract lot sizes come
from Upstox. Set `backtest.option_fixed_cost_rupees` and
`backtest.option_variable_cost_bps` from your actual charges before interpreting
the net estimate. These are assumptions, not a live broker tariff or fill model.
All research P&L is a hypothetical directional return in basis points; the app
does not observe or claim broker fills.

## Parallel work and limits

`--workers -1` uses all CPUs available to the process/container. History requests,
instrument processing and projection refreshes use bounded worker pools. The
WebSocket reader and terminal stay on their own event-loop cadence. Updates for
one instrument remain ordered and snapshots are protected from concurrent
mutation. The tick queue is bounded; overload stops the session with a resync
error rather than silently dropping data.

Batch training also uses all available CPUs, with inner model thread counts
bounded to avoid oversubscription. Memory and queues are bounded: idle periods
do not burn CPU merely to report 100% utilisation. Performance improvements do
not establish forecast quality; assess forward sample counts and errors over
multiple actual sessions. The app does not place orders.
