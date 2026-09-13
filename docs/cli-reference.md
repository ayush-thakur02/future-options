# CLI reference

```bash
uv run niftypulse <command> [options]
```

Alternatively, after `uv sync` the entry point is on the venv path:

```bash
.venv/bin/niftypulse <command>
```

```
╭─ Commands ───────────────────────────────────────────────────────╮
│ login       Authenticate with Upstox and store an access token.  │
│ fetch       Download historical candles into the local store.    │
│ train       Train direction models for every forecast horizon.   │
│ models      List trained model artifacts and their metrics.      │
│ backtest    Backtest a strategy against historical bars.         │
│ dashboard   Run the live terminal dashboard.                     │
│ strategies  List every available strategy.                       │
│ doctor      Check the environment, credentials, and cached data. │
╰──────────────────────────────────────────────────────────────────╯
```

---

## `doctor`

**Check the environment, credentials, and cached data.** Has no options.

Run this first whenever something seems wrong.

```bash
uv run niftypulse doctor
```

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ check          ┃ value           ┃ status ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ python         │ 3.14.7          │ ok     │
│ lightgbm       │ available       │ ok     │
│ upstox token   │ missing         │ warn   │
│ cached bars    │ 45,000          │ ok     │
│ trained models │ 1m, 2m, 3m, 5m  │ ok     │
│ market         │ closed          │ dim    │
└────────────────┴─────────────────┴────────┘
```

| Row | Meaning |
|---|---|
| `lightgbm` | `unavailable` means `libomp` is missing — `brew install libomp` |
| `upstox token` | Checks the env var, then the stored token, honouring expiry |
| `cached bars` | Row count in `data/candles.parquet` |
| `market` | Whether the NSE session is currently open, per the trading calendar |

---

## `login`

**Authenticate with Upstox and store an access token.**

| Option | Default | Description |
|---|---|---|
| `--open-browser` / `--no-open-browser` | `--open-browser` | Open the login page in a browser |

```bash
uv run niftypulse login
uv run niftypulse login --no-open-browser   # over SSH
```

Requires `UPSTOX_CLIENT_ID`, `UPSTOX_CLIENT_SECRET`, and `UPSTOX_REDIRECT_URI`.
Missing any of them prints setup instructions and exits 1.

The flow opens the Upstox authorization page, then prompts for the `code`
parameter from the redirect URL. The resulting token is written to
`data/upstox_token.json` with `chmod 600`.

**Tokens expire at 03:30 IST the next morning.** This is a once-per-morning step.

---

## `fetch`

**Download historical candles into the local store.**

| Option | Default | Description |
|---|---|---|
| `--days <int>` | `120` | Calendar days of 1-minute history |
| `--offline` | off | Generate synthetic data instead |

```bash
uv run niftypulse fetch --days 180
uv run niftypulse fetch --offline --days 120
```

Prints bars, sessions, date range, and the observed price range on completion.

**Behaviour notes:**

- The v3 endpoint caps 1–15 minute candles at one month per request, so a long
  fetch is chunked into roughly `days / 28` requests.
- A single failed window is logged and skipped rather than aborting the backfill.
- The current session is fetched separately via the intraday endpoint, because the
  historical one excludes it.
- Re-running is **idempotent** — writes deduplicate on the timestamp index, keeping
  the newest value. Safe to run repeatedly.
- Rate limiting enforces 8 requests/second and 180/minute internally.

Offline mode is deterministic: `--offline --days 120` produces the same data every
time.

---

## `train`

**Train direction models for every forecast horizon.**

| Option | Default | Description |
|---|---|---|
| `--days <int>` | `0` | Re-fetch this many days first; `0` uses the cache |
| `--horizons <str>` | config | Comma-separated, e.g. `3,5` |
| `--offline` | off | Train on synthetic data |

```bash
uv run niftypulse train
uv run niftypulse train --horizons 3,5
uv run niftypulse train --offline
```

Prints, in order:

1. Feature count and group breakdown
2. Per-horizon sample count and base rate
3. Calibration decision and reasoning
4. **The cost hurdle table** — how often a scalp could pay for itself
5. Walk-forward performance metrics
6. Confidence buckets and top features per horizon
7. Artifact paths

**Horizons too short to be tradeable are skipped with an explanation, not
trained.** See [Scalping economics](scalping-economics.md).

Exit code 1 only if *no* horizon could be trained.

Runtime: roughly 3 minutes for 4 horizons on 45,000 bars, with folds in parallel.

---

## `models`

**List trained artifacts and their metrics.** Has no options.

```bash
uv run niftypulse models
```

```
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ horizon ┃      trained at ┃ samples ┃ accuracy ┃    auc ┃    lift ┃ features ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│     15m │ 2026-09-13T17:… │  26,532 │   0.4971 │ 0.4991 │ -0.0211 │      117 │
└─────────┴─────────────────┴─────────┴──────────┴────────┴─────────┴──────────┘
```

**Lift is the column to read** — accuracy above the majority-class baseline. A
value near zero means the model is not beating "always predict UP".

---

## `backtest`

**Backtest a strategy against historical bars.**

| Option | Default | Description |
|---|---|---|
| `--strategy <str>` | `ensemble` | A strategy name, `ensemble`, or `ml` |
| `--horizon <int>` | `5` | Holding period in bars |
| `--threshold <float>` | `0.15` | Minimum conviction to enter |
| `--notional <float>` | `2000000` | Target position notional in rupees |
| `--lot-size <int>` | `75` | Contract lot size — **verify against the current spec** |
| `--slippage <float>` | `0.5` | Slippage per side, in basis points |
| `--sweep` / `--no-sweep` | off | Sweep entry thresholds |
| `--offline` | off | Use synthetic data |

```bash
uv run niftypulse backtest --strategy ensemble --horizon 5
uv run niftypulse backtest --strategy ml --horizon 5 --threshold 0.15
uv run niftypulse backtest --strategy vwap_reversion --sweep
uv run niftypulse backtest --slippage 1.5 --notional 1000000
```

The header panel shows the derived position size and round-trip cost, so you can
sanity-check the assumptions before reading the results:

```
╭────────────────────────────── backtest ──────────────────────────────╮
│ strategy    ml_5m                                                    │
│ bars        45,000                                                   │
│ horizon     5 bars                                                   │
│ threshold   0.15                                                     │
│ position    1 lot(s) = Rs 1,661,306                                  │
│ round-trip  3.95 bps                                                 │
╰──────────────────────────────────────────────────────────────────────╯
```

**`--strategy ml`** replays the saved walk-forward predictions rather than running
fresh inference, and prints a note saying so. Without a trained model it exits
with a message pointing at `train`.

**`--sweep`** runs the full threshold range and prints a comparison table. Read the
note in [Backtesting](backtesting.md#threshold-sweeps) on how to interpret it.

---

## `dashboard`

**Run the live terminal dashboard.**

| Option | Default | Description |
|---|---|---|
| `--offline` | off | Replay synthetic ticks instead of connecting |
| `--timeframe <int>` | `1` | Bar size in minutes |
| `--speed <float>` | `0.02` | Replay delay between bars, in seconds |
| `--refresh <float>` | `1.0` | Dashboard refresh interval |

```bash
uv run niftypulse dashboard
uv run niftypulse dashboard --offline
uv run niftypulse dashboard --timeframe 5
```

**Falls back to offline automatically** when no access token is available, rather
than failing.

Prints loaded model metrics on startup, so you can see what is driving the
forecasts.

**Note on `--timeframe`:** feature windows are expressed in bars, so changing the
timeframe invalidates models trained at another one. See
[Dashboard](dashboard.md#changing-the-timeframe).

Quit with `q` or `Ctrl-C`.

---

## `strategies`

**List every available strategy.** Has no options.

```bash
uv run niftypulse strategies
```

```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ name                ┃ category   ┃ fires ┃ description                    ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ema_trend           │ trend      │ 48.6% │ EMA 9/21 alignment confirmed … │
│ order_flow          │ flow       │  0.0% │ Order-book and traded-value …  │
└─────────────────────┴────────────┴───────┴────────────────────────────────┘
```

"Fires" is the share of bars with conviction above the entry threshold, measured
on synthetic data.

**"Fires 0%" is a finding, not a bug.** `order_flow` and `activity_spike` report
zero against NIFTY 50 because the index has no order book and no volume. Both
activate against a futures instrument.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Missing credentials, no data, no trained model, or no trainable horizon |

---

## Common invocations

```bash
# first time, no credentials
uv run niftypulse fetch --offline --days 120
uv run niftypulse train
uv run niftypulse dashboard --offline

# daily live session
uv run niftypulse doctor
uv run niftypulse login
uv run niftypulse fetch --days 5
uv run niftypulse dashboard

# research
uv run niftypulse strategies
uv run niftypulse models
uv run niftypulse backtest --strategy ml --horizon 5 --sweep
uv run niftypulse backtest --strategy ensemble --slippage 1.5

# sanity-check the cost assumption
uv run niftypulse backtest --strategy ml --notional 5000000 --slippage 2.0
```
