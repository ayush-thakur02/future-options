# Getting started

## Requirements

- **Python 3.11+**. Verified on 3.14.7.
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- **macOS only:** `libomp`, because LightGBM links against OpenMP.

```bash
brew install libomp
```

Without `libomp` the platform detects it and drops LightGBM from the ensemble
rather than crashing, so you still get three working learners — but install it if
you can.

## Install

```bash
uv sync --all-extras
```

`--all-extras` pulls in the dev dependencies, including `grpcio-tools`, which is
needed only to regenerate the protobuf stubs.

---

## First run — no credentials needed

The dashboard runs on generated data, so the whole pipeline can be exercised
before wiring up an API key:

```bash
uv run niftypulse --offline --speed 60
```

That warms three instruments on a generated series and serves the board at
<http://127.0.0.1:5050>. Nothing generated is recorded, so the tape and the chain
stay clean. `--speed 60` makes a replay minute take a second; the default
`--speed 1.0` is paced against the wall clock.

**What to expect from training:** accuracy at the base rate and AUC near 0.50 on
generated data. That is the correct result, not a bug. The generated series is a
near-random walk with no persistent structure, and its value is confirming the
pipeline is wired correctly. The tests include a positive control that injects a
deterministic pattern and asserts the pipeline learns it, so "found nothing" is
distinguishable from "broken".

The work that is not a live view — logging in, building history, training,
backtesting, diagnostics — is library code, and every entry point is collected in
[Operations](operations.md).

---

## With credentials

Create an app at <https://account.upstox.com/developer/apps>, then:

```bash
cp .env.example .env      # fill in client id, secret, redirect URI
```

Log in with `plugins.sources.upstox.auth.interactive_login`: it opens the Upstox
login and stores the token, and [Operations](operations.md) has the exact call.
Tokens expire at 03:30 IST the next day, so it is a once-per-morning step. If an
environment token is rejected, the app tries the saved login token automatically.
API key/secret values alone cannot authorize the market feed.

Then build history and train on real bars — `runtime.bars.BarLoader` and the
ensemble trainer, both shown in [Operations](operations.md) — and start the
dashboard:

```bash
uv run niftypulse          # live: websocket feed, recording as it goes
```

History is checked first and only the missing tail is requested, so loading it
every morning costs one request when nothing has changed and none at all when the
cache is current.

### What gets stored, and what cannot be

| Dataset | Where | Re-fetchable? |
|---|---|---|
| 1-minute candles | `data/candles/<instrument>/<year>/<month>/<date>.parquet` | Yes |
| Ticks | `data/ticks/<instrument>/<date>/<HH>.parquet` | **No** |
| Option chain samples | `data/chain/...` | **No** |

A provider publishes candles, not the tape that produced them. Ticks and chain
snapshots are recorded while a live session runs, so anything not recorded is
gone. Start a `dashboard` session to begin collecting. See
[Storage](storage.md).

---

## Reading the dashboard

The board shows three charts — the index, the at-the-money call and the
at-the-money put — each with the next three candles projected in **blue** after
the last printed one. Green and red are printed bars; blue is never a printed bar.

Everything is on one page: the cost-gated decision desk, the prediction matrix,
the realtime auto-AI scorecards, the strategy matrix, the indicator state, the
forward score and the system/cost state. Click any strategy row, AI row or
prediction cell to open the live calculation behind it. Historical warm-up samples
are shown separately from live wins and misses. See
[Live research](live-research.md) for the scoring and learning definitions and
[Dashboard](dashboard.md) for the page itself.

The printed local URL stays live until you press `Ctrl+C` in the command
terminal. Use `--no-open-browser` when running headlessly.

The legacy advisory calculations, available for separate research, use:

```
leg            do       needs   projected      edge  why
INDEX          ▲F          4bp      +0.3bp      -4bp  needs 3.9bp, projected 0.3bp
CALL 24,050    ▲F        60bp       +0.3bp     -60bp  needs 60bp, projected 0bp
PUT 24,050     ▲S        62bp       +0.3bp     -62bp  IV 10.0% over realised 5.6%, ...
```

`needs` is what the underlying must do for that leg to pay for itself, costs and
theta included. `projected` is what the platform thinks it will do. Most of the
time the first is far larger than the second, and the correct output is **no
trade** — printed as such rather than left blank.

See [Dashboard](dashboard.md) for the rest of the screen, [Projection](projection.md)
for what the blue candles are, and [Options](options.md) for the verdicts.

---

## Beyond the dashboard

Everything that is not a live view is a library call, and
[Operations](operations.md) has a runnable snippet for each:

| Job | Entry point |
|---|---|
| Authenticate with Upstox | `plugins.sources.upstox.auth.interactive_login` |
| Build or refresh history | `runtime.bars.BarLoader` |
| Train the horizons | `plugins.forecasts.ml_ensemble.Trainer` |
| List trained models | `plugins.forecasts.ml_ensemble.trainer.list_artifacts` |
| Backtest a rule or the model | `backtest.run_backtest`, `run_threshold_sweep` |
| Inspect the plugin graph | `kernel.Kernel` |
| Inspect the strategies | `plugins.strategies.StrategyCatalog` |
| Check the environment | `kernel.Kernel`, `plugins.sources.history.StoreManifest` |
| Read the research stores | `plugins.forecasts.online_research.OnlineResearchLab`, `plugins.advisory.performance_ledger` |

The ML backtest replays walk-forward out-of-sample predictions, so the decisions
come from models that never saw the bars being traded.

---

## Tuning for your costs

The hurdle is only as good as the cost inputs. Adjust these in
`config/default.yaml` before drawing conclusions:

```yaml
model:
  reference_notional: 2_000_000   # position size used to derive the hurdle
  lot_size: 75                    # verify against the current NIFTY spec
  hurdle_multiple: 1.5            # margin required above break-even
  enforce_cost_hurdle: true       # set false to study signal without the gate
```

`CostModel(slippage_bps=…)` and `BacktestConfig(notional=…)` override them for a
single run.

---

## Next

| If you want to… | Read |
|---|---|
| Understand why the platform refuses most trades | [Scalping economics](scalping-economics.md) |
| Add or replace a component | [Plugins](plugins.md) |
| See how the pieces fit | [Architecture](architecture.md) |
| Do anything but watch | [Operations](operations.md) |
| Fix something broken | [Troubleshooting](troubleshooting.md) |
