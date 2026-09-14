# CLI reference

Every command, every flag. `niftypulse <command> --help` prints the same thing
from the source of truth.

```bash
uv run niftypulse <command> [options]
```

Alternatively, after `uv sync` the entry point is on the venv path:

```bash
.venv/bin/niftypulse <command>
```

```
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ login       Authenticate with Upstox and store an access token.              │
│ fetch       Download historical candles into the local store.                │
│ train       Train direction models for every forecast horizon.               │
│ models      List trained model artifacts and their metrics.                  │
│ backtest    Backtest a strategy against historical bars.                     │
│ dashboard   Run the dashboard: call, index and put, with the next three      │
│             candles projected.                                               │
│ snapshot    Render one frame to stdout, with the projected candles, and      │
│             exit.                                                            │
│ strategies  List every available strategy.                                   │
│ plugins     List every plugin the kernel discovers, bundled and installed.   │
│ data        Show what is stored locally, without touching the network.       │
│ sync        Pull everything the account can give, and store it. Never        │
│             fetches twice.                                                   │
│ doctor      Check the environment, credentials, and cached data.             │
│ research    Print durable forecast, AI, strategy, P&L and trust scorecards.  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Commands take `--offline` where they touch the network, and will then run on
cached or generated data. Nothing requires credentials except `login` and the
live paths of `dashboard` and `sync`.

---

## `login`

**Authenticate with Upstox and store an access token.**

| Flag | Default | Meaning |
|---|---|---|
| `--open-browser / --no-open-browser` | on | Open the login URL in a browser |

Requires `UPSTOX_CLIENT_ID`, `UPSTOX_CLIENT_SECRET` and `UPSTOX_REDIRECT_URI` in
the environment or `.env`. The token is written to `data/upstox_token.json` and
expires at 03:30 IST the next morning, so this is a once-a-day step.

---

## `fetch`

**Download historical candles into the local store.**

| Flag | Default | Meaning |
|---|---|---|
| `--days` | `120` | Calendar days of 1-minute history |
| `--offline` | off | Use cached or generated data instead of pulling |

Only the missing tail is requested. What is already stored is never fetched again.

---

## `sync`

**Pull everything the account can give, and store it.**

| Flag | Default | Meaning |
|---|---|---|
| `--days` | `400` | Calendar days of 1-minute history |
| `--offline` | off | Generate instead of pulling |

Bars come from the cache first, then the provider for whatever is missing. With
credentials it also reads the option chain and writes a starting snapshot, and
reports what is now stored.

It says plainly that **ticks and chain history cannot be back-filled**: the API
publishes candles, not the tape that produced them, so both are recorded while a
live session runs.

---

## `data`

**Show what is stored locally, without touching the network.**

| Flag | Default | Meaning |
|---|---|---|
| `--verbose` | off | List every partition file with its size |

Answers "do I have to fetch this again?" from the manifest, so it is instant on a
store of any size. See [Storage](storage.md).

---

## `train`

**Train direction models for every forecast horizon.**

| Flag | Default | Meaning |
|---|---|---|
| `--days` | `0` | Re-fetch this many days first (`0` = use the cache) |
| `--horizons` | configured | Comma-separated, e.g. `1,3,5` |
| `--offline` | off | Train on generated data |

A horizon whose moves cannot clear the cost hurdle is **skipped with an
explanation**, not trained into a losing model. Artifacts land in `artifacts/`.

---

## `models`

**List trained model artifacts and their metrics.** Has no options.

---

## `backtest`

**Backtest a strategy against historical bars.**

| Flag | Default | Meaning |
|---|---|---|
| `--strategy` | `ensemble` | A strategy name, `ensemble`, or `ml` |
| `--horizon` | `5` | Holding period in bars |
| `--threshold` | `0.15` | Minimum conviction to enter |
| `--notional` | `2000000` | Target position notional in rupees |
| `--lot-size` | `75` | Contract lot size — verify against the current NIFTY spec |
| `--slippage` | `0.5` | Slippage per side, in basis points |
| `--sweep / --no-sweep` | off | Sweep entry thresholds |
| `--offline` | off | Backtest on generated data |

`--strategy ml` replays the walk-forward out-of-sample predictions rather than
re-predicting, so the decisions come from models that never saw the bars being
traded.

---

## `dashboard`

**Run the dashboard: call, index and put, with the next three candles projected.**

| Flag | Default | Meaning |
|---|---|---|
| `--offline` | off | Replay generated ticks instead of live |
| `--timeframe` | `1` | Bar size in minutes |
| `--speed` | `1.0` | Replay speed against the clock (`1.0` = real time) |
| `--refresh` | `1.0` | Seconds between frames |
| `--nowcast` | `1.0` | Seconds between projection refreshes |
| `--bars-ahead` | `3` | How many candles to project |
| `--legs / --no-legs` | on | Chart the at-the-money call and put beside the index |
| `--workers` | configured | CPU workers; `-1` uses all available CPUs |
| `--learn / --no-learn` | on | Update per-instrument online return learners |
| `--view` | `ai` | Initial panel: `research`, `costs`, `indicators`, `ai`, or `prediction` |
| `--web` | off | Render through the local Flask browser dashboard |
| `--web-host` | plugin config | Override the web bind host |
| `--web-port` | plugin config | Override the web bind port |
| `--open-browser / --no-open-browser` | plugin config (`on`) | Override automatic browser opening in web mode |

Offline, the feed is **paced against the wall clock** — a one-minute bar takes a
minute — so the projected candles have seconds to move in. `--speed 60` makes a
minute take a second.

With credentials it recovers the local tape, fills only missing history, then
streams the live WebSocket and records ticks and chain samples while it runs.
Keys `1`–`5` select results, positions, indicators, AI and prediction graphs;
`j`/`k` scroll the strategy grid in terminal mode. Web mode shows all views at
once at `http://127.0.0.1:5050`; `Ctrl+C` in the owning terminal stops it. See
[Dashboard](dashboard.md).

---

## `snapshot`

**Render one frame to stdout, with the projected candles, and exit.**

| Flag | Default | Meaning |
|---|---|---|
| `--rows` | `60` | Bars to show per chart |
| `--bars-ahead` | `3` | How many candles to project |
| `--offline / --live` | offline | Use generated or cached data |
| `--timeframe` | `1` | Bar size in minutes |
| `--legs / --no-legs` | on | Show the call/index/put board |

Useful in a pipe, in CI, or to see what a board looks like without waiting for a
clock.

---

## `strategies`

**List every available strategy.** Has no options.

Prints the share of bars each strategy fires on, the pack it came from, and the
reason any pack could not be built — the model-backed strategy is skipped offline
until artifacts exist.

---

## `plugins`

**List every plugin the kernel discovers, bundled and installed.**

| Flag | Default | Meaning |
|---|---|---|
| `--kind` | all | Filter: `source`, `aggregator`, `features`, `strategy`, `forecast`, `advisory`, `renderer` |
| `--capabilities` | off | Also print which plugin provides which capability |

Unresolved requirements and packs that failed to load are reported at the bottom.
See [Plugins](plugins.md).

---

## `doctor`

**Check the environment, credentials, and cached data.** Has no options.

```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ check                 ┃ value                 ┃ status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ python                │ 3.14.7                │ ok     │
│ lightgbm              │ available             │ ok     │
│ upstox token          │ missing               │ warn   │
│ partitioned store     │ 45,000 bars, 120 shards│ ok    │
│ recorded ticks        │ 1,204,331             │ ok     │
│ chain samples         │ 8,442                 │ ok     │
│ trained models        │ 1m, 2m, 3m, 5m        │ ok     │
│ market                │ closed                │ dim    │
└───────────────────────┴───────────────────────┴────────┘
```

| Row | Meaning |
|---|---|
| `lightgbm` | `unavailable` means `libomp` is missing — `brew install libomp` |
| `upstox token` | Checks the env var, then the stored token, honouring expiry |
| `legacy candle file` | A pre-partitioning `data/candles.parquet` waiting for `sync` to import |
| `partitioned store` | Bars on disk, from the manifest — no scanning |
| `recorded ticks` / `chain samples` | Data that cannot be re-fetched, so its volume is worth knowing |
| `market` | Whether the NSE session is currently open, per the trading calendar |

---

## `research`

**Print persisted research tables into normal terminal scrollback.**

| Flag | Default | Meaning |
|---|---|---|
| `--offline` | off | Read simulation ledgers instead of live ledgers |
| `--failures` | off | Show recent projected-candle misses with their issue context |
| `--limit` | `100` | Maximum rows per report section, up to 1,000 |
| `--section` | `all` | `all`, `forecasts`, `ai`, `predictions`, or `strategies` |
| `--json` | off | Emit the selected report as structured JSON |

The report separates statistical quality from economics so the grids remain
readable in an ordinary terminal. It includes exact-target accuracy, rolling
accuracy, baseline lift, calibration, profit, loss, costs, net P&L, drawdown,
pending/expired counts and trust for each online algorithm. Strategy tables show
the same cost-aware outcome view per instrument and rule. Latest AI rows include
the one-, two-, and three-bar BUY/SELL/HOLD research classification.

P&L is a hypothetical directional return in basis points after configured cost
assumptions. It is not an account statement or evidence of an executed fill.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — including "nothing to do" |
| `1` | Something was required and could not be produced: no bars, no horizon trainable, no credentials where they were needed |
