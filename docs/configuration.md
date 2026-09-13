# Configuration

All settings live in `config/default.yaml`. Every one has a working default, so
you only need to edit what you want to change.

Environment variables override credential fields; see below.

---

## `data`

### `instrument_key`
**Default:** `NSE_INDEX|Nifty 50`

The instrument to trade. Index keys use the `NSE_INDEX|` prefix.

> **Prefer the front-month future for scalping.** An index publishes no traded
> volume — there is no consolidated tape for NIFTY 50 — and no order book. Seven
> activity features fall back to tick count and both order-flow strategies return
> no opinion. Pointing this at an `NSE_FO|...` key activates them, and order flow
> is where short-timescale edge actually lives.

Find keys in the instrument master:

```
https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
```

Filter on `segment` (`NSE_FO`, `NSE_INDEX`) and `instrument_type` (`FUT`, `INDEX`).

### `symbol`
**Default:** `NIFTY 50`

Display name only.

### `bar_minutes`
**Default:** `1`

Base bar size.

> **Changing this invalidates trained models.** Feature windows are expressed in
> bars, so a model trained on 1-minute bars cannot be applied to 5-minute ones.
> Refetch and retrain.

### `dir`
**Default:** `data`

Where candles and the token are stored.

---

## `model`

### The cost hurdle

These three settings decide whether the platform is honest. Read
[Scalping economics](scalping-economics.md) before changing them.

#### `enforce_cost_hurdle`
**Default:** `true`

Require labelled training moves to clear the round-trip cost of trading them.

With this on, the model is only taught to classify moves that could actually be
traded, rather than spending its capacity on sub-cost noise it can never profit
from. A model trained on 0.3 bp moves can be consistently right and still lose
money on every correct call.

Set `false` only to study the raw signal in isolation. The backtest still charges
costs, so you will see the honest picture either way — but the model will be
worse, not better.

#### `hurdle_multiple`
**Default:** `1.5`

Multiplier on the cost when setting the labelling threshold.

At `1.0` a labelled move merely breaks even, which is not worth taking. At `1.5`
with a 3.90 bp hurdle, labels require a 5.86 bp move.

| Value | Effect |
|---|---|
| `1.0` | Break-even. More samples, all marginal |
| `1.5` | Default. Requires a real margin |
| `2.0`+ | Fewer, larger moves. May leave too few samples to train on |

#### `reference_notional`
**Default:** `2000000`

Position size used to compute the cost hurdle, in rupees.

Flat per-order brokerage makes the hurdle **higher for smaller positions**. At
₹2M the round trip is ~3.90 bps; at ₹500k it rises because the fixed ₹20
brokerage is spread over less notional.

Set this to what you actually intend to trade.

#### `lot_size`
**Default:** `75`

Contract lot size.

> **Verify this against the current NIFTY specification.** NSE revises it
> periodically. It sets your minimum position: at NIFTY 24,000 with a 75-unit
> lot, the smallest possible trade is about **₹1.8 million of notional**. A
> smaller target is unreachable and the backtest floors at one lot rather than
> pretending otherwise.

### Labelling

#### `horizons`
**Default:** `[1, 2, 3, 5]`

Forecast horizons in bars.

Scalping horizons only. A 15-minute forecast is position trading, and averaging
it into the same view would blur the only timescale this platform is built for.

Horizons whose moves cannot cover costs are **skipped with an explanation**, not
trained.

#### `deadband_atr`
**Default:** `0.12`

Fraction of ATR used as the labelling dead band, applied as a floor *beneath* the
cost hurdle.

A forward move smaller than the threshold is dropped rather than forced into UP or
DOWN. Without this the model is asked to classify the ambiguous middle where
labelling is essentially arbitrary.

On a scalping timescale the cost hurdle usually dominates, making this rarely
binding. It matters more on longer horizons or with very low costs.

### Validation

#### `n_splits`
**Default:** `4`

Purged walk-forward folds.

More folds mean more out-of-sample coverage and a longer training run. `4` gives
roughly 60% of the data as out-of-sample while keeping training near 3 minutes.

#### `embargo_bars`
**Default:** `20`

Extra bars dropped from the end of each training fold, on top of the purge that
removes overlapping labels.

Guards against serial correlation: bars adjacent across a fold boundary share
nearly identical features, so a model can memorise the regime around the split
and score well on both sides without learning anything general.

#### `min_train_bars`
**Default:** `4000`

Minimum labelled samples before a horizon is considered trainable.

Below this, raise history, lower `hurdle_multiple`, or accept that the horizon is
untradeable.

#### `train_n_jobs`
**Default:** `4`

Folds trained in parallel.

Estimators are pinned to one thread each while this is `> 1`. Leaving both levels
at `-1` oversubscribes the CPU and runs **measurably slower** than either level
alone. Set to `1` to disable fold parallelism.

#### `dir`
**Default:** `artifacts`

Where models and reports are written.

---

## `backtest`

### `slippage_bps`
**Default:** `0.5`

Slippage **per side**, in basis points. A round trip pays it twice.

Usually the largest real cost on a scalping timescale and the one most often
underestimated. A one basis point increase moves the hurdle by 25% and can
eliminate a marginal edge outright.

These defaults are plausible for a liquid NIFTY future in normal conditions. In a
fast market they are optimistic. **If you have real fill data, use it.**

Override per-run with `--slippage`.

### `cost_bps`
**Default:** `0.6`

Reference value only. The live cost calculation uses `CostModel`, not this.

---

## Environment variables

Loaded from `.env` via `python-dotenv`. Environment wins over the file.

| Variable | Purpose |
|---|---|
| `UPSTOX_CLIENT_ID` | API key |
| `UPSTOX_CLIENT_SECRET` | API secret |
| `UPSTOX_REDIRECT_URI` | Registered redirect URI |
| `UPSTOX_ACCESS_TOKEN` | Optional. Overrides the stored token — useful for CI or a scheduled job that injects a fresh one |
| `NIFTYPULSE_HOME` | Overrides the project root, so `data/`, `artifacts/`, and `logs/` relocate together |

`NIFTYPULSE_HOME` is the way to keep several configurations side by side.

---

## Cost model

The cost model itself is **not** in the YAML — it is constructed in code
(`backtest/costs.py`) with documented defaults, and `slippage_bps` is threaded in
from configuration.

| Component | Default | Basis |
|---|---|---|
| `brokerage_per_order` | ₹20 | Flat per order, both legs |
| `stt_sell_bps` | 2.00 | Securities Transaction Tax, sell leg only |
| `exchange_txn_bps` | 0.19 | NSE index futures |
| `sebi_bps` | 0.01 | SEBI turnover fee |
| `stamp_duty_buy_bps` | 0.20 | Buy leg only |
| `gst_rate` | 0.18 | On brokerage + exchange charges |
| `lot_size` | 75 | Converts flat brokerage into a rate |

Giving **~3.90 bps** round trip at ₹2,000,000 notional.

Rates change with each Union Budget and are set per exchange, so nothing is
hard-coded: the structure is fixed, the numbers are parameters. To change one,
edit `CostModel` directly or pass a configured instance to `BacktestConfig`.

---

## Recipes

### Realistic-cost research

Conservative assumptions, the right way to sanity-check any result:

```yaml
backtest:
  slippage_bps: 1.5      # 3x the default
model:
  reference_notional: 500000   # smaller size, higher relative cost
  hurdle_multiple: 2.0
```

If a strategy survives this, it is worth a closer look. Most do not.

### Study the raw signal

See what the model finds with no cost filter. **The backtest still charges
costs**, so this isolates the signal, it does not flatter it:

```yaml
model:
  enforce_cost_hurdle: false
  horizons: [1, 2, 3, 5]
```

Expectably, `train` will now produce 1m and 2m models. Expect the backtest to
show gross expectancy near zero against a 3.9 bp cost.

### Trade the futures

Activates the volume and order-flow features:

```yaml
data:
  instrument_key: "NSE_FO|<token>"
  symbol: "NIFTY FUT"
```

### Faster iteration

```yaml
model:
  n_splits: 2
  min_train_bars: 1500
```

Roughly halves training time at the cost of a noisier out-of-sample estimate. Use
it while iterating, then go back to defaults before drawing conclusions.

### Multiple configurations

```bash
NIFTYPULSE_HOME=~/nifty-conservative uv run niftypulse train
NIFTYPULSE_HOME=~/nifty-optimistic  uv run niftypulse train
```

Each gets its own `data/`, `artifacts/`, and `logs/`.

---

## Resolution order

1. Explicit function arguments (`load_settings(config_path=...)`)
2. `config/default.yaml`
3. Built-in defaults in `Settings`

Credentials additionally check the environment before the stored token file.

Invalid or missing keys fall back to defaults rather than raising — a partially
written config file should not stop the platform from starting.
