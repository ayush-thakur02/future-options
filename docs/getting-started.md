# Getting started

## Requirements

- **Python 3.11+**. Verified on 3.14.7.
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- **macOS only:** `libomp`, because LightGBM links against OpenMP.

```bash
brew install libomp
```

Without `libomp`, LightGBM fails to import with a `dlopen` error. The platform
detects this and silently drops LightGBM from the ensemble rather than crashing,
so you still get three working learners — but install it if you can.

## Install

```bash
cd "Future & Options"
uv sync --all-extras
```

`--all-extras` pulls in the dev dependencies, including `grpcio-tools` which is
needed only to regenerate the protobuf stubs.

## First run — no credentials needed

Every command accepts `--offline`, which runs on generated data. You can exercise
the whole pipeline before wiring up an API key.

```bash
uv run niftypulse doctor                      # environment check
uv run niftypulse fetch --offline --days 120  # generate synthetic history
uv run niftypulse train                       # train the scalping horizons
uv run niftypulse dashboard --offline         # live dashboard on replayed ticks
```

**What to expect:** accuracy at the base rate and AUC near 0.50. This is the
correct result, not a bug. The generated series is a near-random walk with no
persistent structure to find. Its purpose is to prove the pipeline is wired
correctly.

The tests include a positive control that injects a deterministic pattern and
asserts the pipeline learns it (AUC > 0.80) alongside a negative control asserting
no spurious edge appears on a random walk (AUC < 0.60). Those two together are
what let you trust that "found nothing" means the same thing as "there was
nothing to find".

## Going live with Upstox

### 1. Create an app

Go to <https://account.upstox.com/developer/apps> and create an app. You need:

- **API key** (`client_id`)
- **API secret** (`client_secret`)
- **Redirect URI** — must match exactly what you send in the login request

### 2. Store the credentials

```bash
cp .env.example .env
```

Fill in:

```bash
UPSTOX_CLIENT_ID=your-api-key
UPSTOX_CLIENT_SECRET=your-api-secret
UPSTOX_REDIRECT_URI=https://your-redirect-uri
```

### 3. Authenticate

```bash
uv run niftypulse login
```

This opens the Upstox login page, then asks you to paste the `code` parameter
from the redirect URL. The token is stored at `data/upstox_token.json` with
`chmod 600`.

**Upstox tokens expire at 03:30 IST the next morning**, so this is a
once-per-morning step. `doctor` shows whether the stored token is still valid.

If you would rather not run the browser flow, Upstox also lets you generate a
token manually from the developer dashboard and set it as `UPSTOX_ACCESS_TOKEN`
in `.env`, which takes precedence over the stored file.

### 4. Fetch history and train

```bash
uv run niftypulse fetch --days 180
uv run niftypulse train
```

The v3 historical endpoint caps a single request at one month of 1-minute
candles, so a 180-day fetch is chunked into roughly 7 requests. Rate limiting is
handled internally.

### 5. Run the dashboard

```bash
uv run niftypulse dashboard
```

Without `--offline`, this bootstraps from cached history and then streams live
ticks over the Upstox v3 WebSocket.

---

## A complete session

```bash
# morning
uv run niftypulse doctor          # confirm token is valid
uv run niftypulse login           # if the token expired overnight
uv run niftypulse fetch --days 5  # top up the cache
uv run niftypulse dashboard       # scalp

# research, any time
uv run niftypulse strategies
uv run niftypulse models
uv run niftypulse backtest --strategy ml --horizon 5 --sweep
```

---

## Where things live

| Path | Contents |
|---|---|
| `data/candles.parquet` | Cached 1-minute OHLCV bars |
| `data/upstox_token.json` | Access token (chmod 600) |
| `artifacts/direction_*m.joblib` | Trained models |
| `artifacts/oof_*m.parquet` | Walk-forward predictions, used by `--strategy ml` |
| `artifacts/training_report.json` | Last training run summary |
| `config/default.yaml` | Settings |
| `.env` | Credentials (gitignored) |

Set `NIFTYPULSE_HOME` to relocate all of it, useful if you want to keep several
configurations side by side.

---

## Next

- Read [Scalping economics](scalping-economics.md) before tuning anything. Most
  changes that look like improvements are not.
- [Configuration](configuration.md) if you want to change the cost assumptions —
  they matter more than any modelling choice.
- [CLI reference](cli-reference.md) for everything the commands can do.
