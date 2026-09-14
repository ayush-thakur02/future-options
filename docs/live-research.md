# Live research dashboard

```bash
niftypulse doctor --live
niftypulse dashboard --workers -1
niftypulse research
niftypulse research --failures --json
```

In this checkout the executable is `.venv/bin/niftypulse` if the virtual
environment is not activated. A terminal around 180 columns × 44 rows gives the
charts and rule explanations enough room. Use `1` for research, `2` for scalp
costs, `3` for indicators, `r` to refresh projections, and `q` to exit.

## Authentication and market data

Startup validates the environment token with Upstox. If Upstox rejects it with
HTTP 401, the app tries the unexpired token saved by `niftypulse login`. This
fixes an invalid `.env` token hiding a valid saved login. Neither credential is
printed or overwritten. If both are rejected, run `niftypulse login` and restart.
Network errors and rate limits do not trigger a credential fallback.

Live mode requests fresh historical and intraday candles directly from Upstox
because older versions could write generated prices into the shared candle
cache. Current, incomplete bars are excluded from training. Live mode does not
fall back to generated data. Use `--offline` explicitly for simulation.

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

## Ten rules and online learning

| Rule | Evidence / condition |
|---|---|
| EMA trend | EMA 9/21/50 ordering with ADX and directional movement |
| SuperTrend | ATR trend direction, persistence and trend quality |
| Donchian breakout | Prior 20-bar high/low breakout and squeeze context |
| Opening range | Price must leave the completed first 15-minute range |
| MACD momentum | Histogram level and recent change |
| RSI reversion | Oversold/overbought extremes, damped during strong trends |
| Bollinger reversion | Price outside the bands, damped by trend strength |
| VWAP reversion | Extension from session VWAP; requires traded volume |
| Squeeze release | Volatility compression followed by directional expansion |
| Order flow | Bid/ask depth imbalance; requires real order-book inputs |

All ten are displayed for all three instruments. `WAIT` means conditions are
not met; `N/A` means the input is unavailable. An index has no traded volume or
order book, so its VWAP and flow rules remain unavailable. The nineteen-rule
catalog is still available through `niftypulse strategies` for separate studies.
Rule weights adjust within bounds using previously observed three-bar outcomes;
historical warm-up does not count toward those outcomes.

Each instrument has an independent incremental SGD return model. Fixed ATR
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

The costs view compares an option's projected resale premium with entry premium,
quoted spread and configured round-trip charge assumptions. It does not require
recovering the entire option premium during a scalp. Contract lot sizes come
from Upstox. Set `backtest.option_fixed_cost_rupees` and
`backtest.option_variable_cost_bps` from your actual charges before interpreting
the net estimate. These are assumptions, not a live broker tariff or fill model.

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
