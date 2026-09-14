# Troubleshooting

---

## Start here

```bash
uv run niftypulse doctor
```

Checks Python, LightGBM, credentials, cached data, trained models, and whether
the market is currently open.

---

## Installation

### `Library not loaded: @rpath/libomp.dylib`

LightGBM links against OpenMP, which macOS does not ship.

```bash
brew install libomp
```

The platform detects the failure and **drops LightGBM from the ensemble** rather
than crashing, so you will still get three working learners. `doctor` reports
`lightgbm: unavailable (brew install libomp)`.

Training is slightly worse without it — LightGBM is usually the strongest single
model — but everything still works.

### `OSError: Readme file does not exist: README.md`

You deleted or moved `README.md`. It is referenced in `pyproject.toml`; restore it
or remove the `readme` line.

### Build backend errors during `uv sync`

The project path contains spaces and an ampersand (`Future & Options`). uv handles
this, but some shell invocations need quoting:

```bash
cd "/Users/ayushthakur/Projects/Future & Options"
```

---

## Data

### `No Upstox access token`

No token found. Either:

- `uv run niftypulse login` — needs credentials in `.env`
- add `--offline` to any command
- set `UPSTOX_ACCESS_TOKEN` for a scheduled job

### Token expired

**Upstox tokens expire at 03:30 IST the next morning.** This is by design, not a
session length. Re-run `uv run niftypulse login` each morning.

### `feed authorization failed (401)`

The token is present but rejected. Usually expired — `doctor` will show `warn`
next to `upstox token` if that is the case.

### `Invalid Instrument key`

The instrument key does not match anything. Check the exact spelling and prefix —
`NSE_INDEX|Nifty 50`, not `NSE_INDEX|NIFTY50` or `NIFTY 50`.

Keys are case-sensitive and the prefix matters. Resolve one with the instrument
search API rather than a master-file download — it is the server-side endpoint,
and it supports ATM-relative option lookup:

```python
from scripts.instrument_search import resolve_instrument_key, find_option

resolve_instrument_key("Nifty 50", exchanges="NSE", segments="INDEX")["instrument_key"]
find_option("Nifty 50", expiry="current_week", option_type="CE", atm_offset=0)
```

Do not hardcode an F&O key: the numeric token changes every expiry. See
`.agents/skills/upstox/references/instruments.md`.

### Fetch is slow

The v3 endpoint caps 1–15 minute candles at about one month per request, so 180
days is about 7 requests. Rate limiting holds to 8/second and 180/minute
internally.

A long first fetch is expected. **Subsequent runs only fetch the tail** — `sync`
requests the missing window and nothing else, so a morning run with a current
cache makes no API calls at all.

### `data/candles.parquet.migrated` is sitting in my data folder

That is your history, imported into the partitioned store and moved aside. It is
safe to delete once `niftypulse data` shows the bars in `candles/`. It is kept by
default because starting from an empty store would look exactly like losing your
history. See [Storage](storage.md).

### `doctor` shows `legacy candle file` as a warning

A pre-partitioning `candles.parquet` is present and has not been imported yet.
The import happens on first use of the store, so any of `sync`, `fetch`, or
`dashboard` will do it.

### The store is bigger than I expected

Ticks and chain samples are the bulk of it, and neither can be re-fetched. If you
need to bound it, delete old `data/ticks/` and `data/chain/` partitions — nothing
else depends on them. `niftypulse data --verbose` lists every file with its size.

### `no candles retrieved`

Every window failed. Usually a bad instrument key or an expired token. Run
`doctor`.

---

## Training

### `skipped 1m — only N of the bars have a forward move above the X bp hurdle`

**Not an error. This is the platform working.**

Most 1-minute NIFTY moves are smaller than the round-trip cost, so there are too
few tradeable moves to learn from. Training a model here would produce something
worse than useless, because it would create the impression of a working
1-minute scalper.

Options, in order of preference:

1. **Train a longer horizon.** `--horizons 3,5`
2. **Lower `hurdle_multiple`** if you want the signal without a cost margin
3. **`enforce_cost_hurdle: false`** to study signal alone — the backtest still
   charges costs, so this isolates the signal rather than flattering it

See [Scalping economics](scalping-economics.md).

### `no valid folds produced; check min_train and n_splits`

Not enough samples for the requested fold count. Lower `min_train_bars` or
`n_splits`, or fetch more history.

The minimum is roughly `min_train + n_splits` samples.

### Training takes a long time

Roughly 3 minutes for 4 horizons on 45,000 bars with folds in parallel. To speed
up iteration:

```yaml
model:
  n_splits: 2
  min_train_bars: 1500
```

Roughly halves the time at the cost of a noisier estimate. Go back to defaults
before drawing conclusions.

**Do not set `train_n_jobs` and the estimator `n_jobs` both to `-1`.** They
oversubscribe the CPU and run measurably slower than either alone.

### `waiting for realtime learner` in the dashboard

Realtime research is enabled by default and issues its first prediction from the
warmed state during startup. If it remains waiting, check that `--learn` is on
and that enough valid history loaded to build features. Batch artifacts are
optional; `uv run niftypulse train` is only needed for the separate offline
walk-forward ensemble.

### Accuracy is 50% and AUC is 0.50

**Expected on synthetic data.** The generated series is close to a random walk.
The value of an offline run is confirming the pipeline is wired correctly.

The test suite contains a positive control that injects a deterministic pattern
and asserts the pipeline learns it, so a null result here is a statement about
the data, not a bug.

**If you see this on real data**, it is also a legitimate result. Intraday index
direction at 1–5 minute horizons is close to unpredictable.

### Accuracy looks great (55%+)

Probably a leak. Check:

1. **Confidence buckets.** Does accuracy rise monotonically with confidence? If
   it is flat, the headline number is not measuring what you think.
2. **Lift.** What is the majority-class base rate? On a 53% UP dataset, predicting
   UP always scores 53%.
3. **Sample count.** A 0.66 bucket with 53 samples is not evidence.
4. **Positive control still passing?** `uv run pytest tests/test_ml.py -k pipeline`

A genuinely good result should survive all four.

---

## Backtesting

### `no walk-forward predictions for 5m`

`--strategy ml` needs saved out-of-sample predictions. Run
`uv run niftypulse train` first.

They are stored at `artifacts/oof_<horizon>m.parquet`. If training skipped that
horizon as untradeable, there will be no file.

### Costs look absurd (millions of basis points)

A historical bug, fixed. Position sizing treated the target notional as a unit
count, so exposure came out at `price × notional`. Guarded by
`test_costs_are_plausible_magnitude`, which asserts the cost drag is under 50 bps.

If you see this, you are running modified code. Check
`BacktestConfig.size_for`.

### Every bar trades

The conviction series is probably in `[0, 1]` rather than `[-1, 1]`. The engine
thresholds `abs(score)`, so `abs(0.48)` — a model with no opinion — clears any
threshold below 0.48.

Probabilities must be mapped before being handed over:

```python
conviction = 2.0 * probability - 1.0
```

Guarded by `test_probability_series_is_not_valid_conviction`.

### Net expectancy gets *worse* as the threshold rises

The diagnostic, not a bug. It means the signal carries no information about its
own confidence. No threshold tuning will rescue it.

See [Backtesting § Threshold sweeps](backtesting.md#threshold-sweeps).

### `position 1 lot(s) = Rs 1,800,000` when I asked for ₹500,000

Positions size in whole lots. At NIFTY 24,000 with a 75-unit lot, one lot is
about ₹1.8 million, and that is the floor. Any smaller target is unreachable.

---

## Dashboard

### The chart looks scrambled, with price labels on separate lines

A historical rendering bug, fixed. Two causes compounded:

1. Right-stripping style runs deleted the leading blank pad that right-aligns the
   chart.
2. Rows were 3 characters wider than the panel interior, so Rich wrapped the price
   axis onto its own line.

If you see this, you are running modified code. The guards are in
`tests/test_ui.py`, verified to fail against the broken version.

### Candles are bunched at the left with the right side empty

The right-alignment pad is being stripped. See above.

### The replay is too slow — or too fast

`--offline` is **paced against the wall clock** on purpose: one minute of wall
clock per one-minute bar at `--speed 1.0`, so the projected candles have seconds
to move in. `--speed 60` makes a minute take a second, which is the practical
setting for watching the board work.

### The chart shows only one candle after the first tick

A bug, fixed: the snapshot took its bars from the aggregator alone, which starts
empty, so the history beside it vanished the moment the first tick arrived. If you
see it, you are running modified code.

### The blue candles do not move

They are refreshed by `runtime/nowcast.py` on its own clock (`--nowcast`, default
1 s), not on ticks alone — so they should move even on a quiet tape. If they are
frozen, check the status line for a feed error, and that `forecast:projection` is
still registered (`niftypulse plugins --kind forecast`).

The one deliberate reset: a **roll** re-strikes a leg and clears its path. A leg
more than two strikes from the money is no longer the trade the board is about.

### There are no CALL and PUT charts

No `option_chain` capability is registered, or `--no-legs` was passed. Check with
`niftypulse plugins --capabilities`. Without it the board shows the index alone
rather than failing.

### The dashboard prints its startup lines and then nothing

A bug, fixed: the session pushed a frame a second into the renderer without ever
opening its live block, and `live_update` draws nothing outside one. The whole
platform was running — feed, projection clock, strategies — behind a screen it
had never taken, which is why it read as a hang rather than as a crash. The render
loop now opens the block, and
`tests/test_runtime.py::test_the_render_loop_draws_inside_the_live_block` fails
against the old version.

### The dashboard exits immediately

Terminal too small. The layout needs roughly 24 rows and 60 columns. Resize, or
set a smaller font.

### `regime: unknown`

Fewer than about 60 bars have been processed, so the regime classifier has no
window. It resolves after warm-up.

### `order_flow` and `activity_spike` show 0% strength

**Correct behaviour for the index.** NIFTY 50 publishes no order book and no
traded volume, so the strategies return no opinion rather than fabricating one.

They activate on the option legs, which *do* carry volume and open interest, and
on the front-month future if you point `instrument_key` at one.

### Every leg says FLAT, and the headline says no trade

**That is the finding, not a failure.** An at-the-money leg needs the underlying
to travel 60–100bp to pay for its own premium, costs and theta, against a median
1-minute move of 1.3bp. The arithmetic is printed next to the
verdict so it can be checked. See [Options](options.md).

### Timeframe mismatch warning

Feature windows are expressed in bars, so a model trained on 1-minute bars does
not transfer to 5-minute ones. `--timeframe 5` changes the chart but the forecasts
still come from the 1-minute models.

To run at another timeframe properly, set `bar_minutes: 5` in configuration,
refetch, and retrain.

---

## Interpreting results

### Everything is red

Expected, for now. See [Scalping economics](scalping-economics.md). At a ~3.9 bp
round trip and ~1.3 bp median 1-minute moves, the arithmetic does not work. The
platform is reporting that honestly rather than hiding it.

### Can I make it profitable?

Realistically, the levers are:

1. **Trade the futures, not the index** — activates order flow, which is where
   short-timescale edge actually lives
2. **Get your real costs** — a basis point of slippage beats any modelling change
3. **Prefer 3m and 5m** over 1m
4. **Accept fewer trades** — the cost gate exists for this reason

What will not work: tuning the entry threshold, adding indicators, or training
longer. If expectancy does not improve with the threshold sweep, the signal has no
information about its own confidence, and none of those changes create it.

### Inflating results by accident

The easiest way to fool yourself here is to change the cost assumption to
something optimistic. Every result is conditional on it.

Before trusting anything:

```bash
uv run niftypulse backtest --strategy ml --horizon 5 --slippage 2.0 --notional 500000
```

If it survives pessimistic costs, it is worth a closer look.

---

## Plugins

### A plugin is missing from `niftypulse plugins`

Either its `plugin.py` failed to import — the report at the bottom of the command's
output names the module and the error — or it is nested without an `__init__.py`
at every level, in which case the loader cannot walk into it.

### `capability 'x' is provided by both A and B`

Registration rejects two providers of one capability, deliberately: a capability
that resolved to different plugins on different runs would make a composition
impossible to reason about. Either rename one, or give it no capability and let
the runtime pick it by handle.

### `source:history requires 'broker', which no registered plugin provides`

A composition is missing a pack its consumers declared. `kernel.validate()` lists
every unsatisfied requirement at once, so fix them together rather than one run at
a time. `niftypulse plugins` prints them at the bottom.

### `strategy:ml_forecast skipped: no trained model for horizon 1m`

Normal offline. The pack cannot be built without artifacts, and the catalog records
the reason and carries on with the rule-based engine. Run `niftypulse train`.

---

## Still stuck

1. `uv run niftypulse doctor`
2. `uv run pytest` — if tests fail, something is genuinely broken
3. `uv run pytest tests/test_features.py -k causal` — rules out lookahead
4. `uv run pytest tests/test_ml.py -k pipeline` — rules out a broken ML pipeline

If tests pass and the platform behaves unexpectedly, the behaviour is probably
intentional and documented. Check
[Scalping economics](scalping-economics.md) and
[Architecture](architecture.md#design-decisions).
