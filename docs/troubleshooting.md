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

Keys are case-sensitive and the prefix matters. Look yours up in the instrument
master:

```bash
curl -s https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz \
  | gunzip | python -c "
import json,sys
for row in json.load(sys.stdin):
    if row.get('segment')=='NSE_INDEX':
        print(row['instrument_key'], '|', row['name'])
"
```

### Fetch is slow

The v3 endpoint caps 1–15 minute candles at one month per request, so 180 days is
about 7 requests. Rate limiting holds to 8/second and 180/minute internally.

A long first fetch is expected. Subsequent runs only fetch the tail.

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

### `no trained models` in the dashboard

Run `uv run niftypulse train` first. The forecasts panel says so explicitly.

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

### Replay is slow

Expected. The engine recomputes features on every bar close, at roughly 200 ms per
bar, so replay is **compute-bound** regardless of `--speed`.

Live operation is unaffected — bars close once a minute, so 200 ms is 0.3% of the
interval.

`--speed 0` removes the artificial delay but the replay still takes about 30
seconds per 150 bars.

### The dashboard exits immediately

Terminal too small. The layout needs roughly 24 rows and 60 columns. Resize, or
set a smaller font.

### `regime: unknown`

Fewer than about 60 bars have been processed, so the regime classifier has no
window. It resolves after warm-up.

### `order_flow` and `activity_spike` show 0% strength

**Correct behaviour for an index.** NIFTY 50 publishes no order book and no traded
volume, so the strategies return no opinion rather than fabricating one.

Point `instrument_key` at the front-month NIFTY future to activate them.

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

## Still stuck

1. `uv run niftypulse doctor`
2. `uv run pytest` — if tests fail, something is genuinely broken
3. `uv run pytest tests/test_features.py -k causal` — rules out lookahead
4. `uv run pytest tests/test_ml.py -k pipeline` — rules out a broken ML pipeline

If tests pass and the platform behaves unexpectedly, the behaviour is probably
intentional and documented. Check
[Scalping economics](scalping-economics.md) and
[Architecture](architecture.md#design-decisions).
