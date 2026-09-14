# Live research dashboard

```bash
uv run niftypulse --workers -1           # the live dashboard
uv run niftypulse --offline --speed 60   # the same board on generated ticks
```

The page carries every view at once — results, position scenarios, indicators,
online AI, prediction graphs and the strategy matrix — so a window around 180
columns wide gives the charts and rule explanations room. The online AI scorecard
updates from the first bar. The durable stores behind it are readable without a
browser; see
[Operations § Read the research stores](operations.md#read-the-research-stores).

## Authentication and market data

Startup validates the environment token with Upstox. If Upstox rejects it with
HTTP 401, the app tries the unexpired token saved by the login flow. This fixes an
invalid `.env` token hiding a valid saved login. Neither credential is printed or
overwritten. If both are rejected, log in again
(`plugins.sources.upstox.auth.interactive_login`) and restart. Network errors and
rate limits do not trigger a credential fallback.

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
through the ensemble trainer, the artifact listing and `backtest` — all in
[Operations](operations.md); option engines do not use index models.

Alongside that return learner, the `forecast:online_research` plugin runs six
prequential classifiers: online logistic regression, passive-aggressive
classification, online Gaussian Naive Bayes, sparse adaptive FTRL-Proximal, and
a bounded recency-weighted KNN for nonlinear local regimes. The sixth is a
strategy-combination learner that evaluates bounded singles, pairs, and triples
of the strongest active strategy directions. Each gets its own immutable
prediction at one-, two-, and three-bar horizons. The plugin learns only after
the exact target closes and stores its model state and ledger in one SQLite
transaction under `data/research/online_ai/{live|simulation}/`.
Every learner's parameters are editable in
`config/plugins/forecast/online_research.yaml`.

At startup, the first live prediction is issued immediately from the warmed
technical and strategy state. At each exact target close, success/failure and
net outcome are recorded, the learners update, and the next prediction receives
the latest per-strategy accuracy and trust. This path never requires batch
training.

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
the regime/strategy/model context. It is ordinary SQLite: the `forecasts` table
holds every issued forecast, and the `status = 'miss'` rows keep the frozen issue
context of the misses. The simulated run writes the same schema to
`simulation.sqlite3`. Reports aggregate runs per instrument/timeframe/horizon;
repeated observations across separate runs are not independent experiments.

`data/research/strategies.sqlite3` holds the strategy ledger, and the online AI
state and prediction ledgers are separated by mode and instrument below
`data/research/online_ai/`. Both have readers —
`StrategyPerformanceLedger` and `OnlineResearchLab` — shown in
[Operations § Read the research stores](operations.md#read-the-research-stores).
The dashboard's scorecards come from those same objects, so a script and the page
cannot disagree.

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
WebSocket reader and the dashboard serve on their own event-loop cadence. Updates for
one instrument remain ordered and snapshots are protected from concurrent
mutation. The tick queue is bounded; overload stops the session with a resync
error rather than silently dropping data.

Batch training also uses all available CPUs, with inner model thread counts
bounded to avoid oversubscription. Memory and queues are bounded: idle periods
do not burn CPU merely to report 100% utilisation. Performance improvements do
not establish forecast quality; assess forward sample counts and errors over
multiple actual sessions. The app does not place orders.
