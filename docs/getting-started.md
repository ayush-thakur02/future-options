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

Commands that load market data accept `--offline`, so the whole pipeline can be
exercised before wiring up an API key.

```bash
uv run niftypulse doctor                       # environment check
uv run niftypulse plugins                      # what is installed, and how it is wired
uv run niftypulse sync --offline --days 120    # generate and store history
uv run niftypulse train                        # train the scalping horizons
uv run niftypulse snapshot                     # one frame of the board, then exit
uv run niftypulse dashboard --offline --speed 60
```

`snapshot` is the fastest way to see the whole thing work: it prints one
call/index/put board with the projected candles and exits.

**What to expect from `train`:** accuracy at the base rate and AUC near 0.50.
That is the correct result, not a bug. The generated series is a near-random walk
with no persistent structure, and its value is confirming the pipeline is wired
correctly. The tests include a positive control that injects a deterministic
pattern and asserts the pipeline learns it, so "found nothing" is distinguishable
from "broken".

---

## With credentials

Create an app at <https://account.upstox.com/developer/apps>, then:

```bash
cp .env.example .env      # fill in client id, secret, redirect URI
uv run niftypulse login   # opens the Upstox login, stores the token
```

Tokens expire at 03:30 IST the next day, so `login` is a once-per-morning step.

Run `niftypulse doctor --live` to validate the token and feed authorization. If an
environment token is rejected, the app tries the saved login token automatically.
API key/secret values alone cannot authorize the market feed.

Then:

```bash
uv run niftypulse sync --days 400          # pull everything the account can give
uv run niftypulse data                     # what is now stored, without a network call
uv run niftypulse train                    # train on real bars
uv run niftypulse dashboard                # live: websocket feed, recording as it goes
```

`sync` checks the cache first and requests only the missing tail, so running it
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

The default panel shows all ten research rules and forward prediction outcomes
for each instrument/horizon. Press `2` for the premium-scalping cost calculation,
`3` for indicators, or `1` to return to research. `q` exits. Historical warm-up
samples are shown separately from live wins and misses. See
[Live research](live-research.md) for the scoring and learning definitions.

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

## Common first commands

```bash
uv run niftypulse strategies                  # every rule, and how often it fires
uv run niftypulse models                      # trained artifacts and their metrics
uv run niftypulse backtest --strategy ensemble --horizon 5
uv run niftypulse backtest --strategy ml --horizon 5 --sweep
uv run niftypulse plugins --capabilities      # the capability index
```

`--strategy ml` replays walk-forward out-of-sample predictions, so the decisions
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

`backtest --slippage` and `--notional` override them for a single run.

---

## Next

| If you want to… | Read |
|---|---|
| Understand why the platform refuses most trades | [Scalping economics](scalping-economics.md) |
| Add or replace a component | [Plugins](plugins.md) |
| See how the pieces fit | [Architecture](architecture.md) |
| Look up a command | [CLI reference](cli-reference.md) |
| Fix something broken | [Troubleshooting](troubleshooting.md) |
