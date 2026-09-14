# Configuration

Platform-wide settings live in `config/default.yaml`. Each plugin has a separate
file at `config/plugins/<kind>/<name>.yaml`, mirroring its handle and keeping its
tuning next to related plugin configuration. Every setting has a working default,
so you only need to edit what you want to change.

Environment variables override credential fields. Each plugin YAML mapping is
merged into that plugin's build parameters.

---

## `data`

### `instrument_key`
**Default:** `NSE_INDEX|Nifty 50`

The instrument to stream and store.

> **Prefer the front-month future for scalping.** An index publishes no traded
> volume — there is no consolidated tape for NIFTY 50 — and no order book, so
> seven activity features fall back to tick count and both order-flow strategies
> return no opinion on it. An `NSE_FO|...` key activates them, and order flow is
> where short-timescale edge actually lives.

Resolve a key with the instrument search API (see
`.agents/skills/upstox/references/instruments.md`), which supports ATM-relative
option lookup and returns the lot size:

```python
instruments.search_instrument("Nifty 50", exchanges="NSE", segments="FO",
                              instrument_types="FUT", expiry="current_week")
```

Do not hardcode an F&O key: the numeric token changes every expiry.

### `symbol`
**Default:** `NIFTY 50`

Display name only.

### `bar_minutes`
**Default:** `1`

Base bar size. `--timeframe` on the `dashboard` and `snapshot` commands overrides
it for one run.

> **Changing this invalidates trained models.** Feature windows are expressed in
> bars, so a model trained on 1-minute bars cannot be applied to 5-minute ones.

### `dir`
**Default:** `data`

Where the partitioned store, the manifest and the token live. See
[Storage](storage.md) for the layout inside it.

---

## `model`

### The cost hurdle

These three settings decide whether the platform is honest. Read
[Scalping economics](scalping-economics.md) before changing them.

#### `enforce_cost_hurdle`
**Default:** `true`

Require labelled training moves to clear the round trip needed to capture them.
With this on, the model is only taught to classify moves it could have traded.

#### `hurdle_multiple`
**Default:** `1.5`

Multiplier on the round trip when setting the labelling threshold. At `1.0` a
labelled move merely breaks even, which is not worth taking.

#### `deadband_atr`
**Default:** `0.12`

Fraction of ATR used as the labelling dead band, applied as a floor beneath the
cost hurdle.

**Derived:** with the defaults, `cost_hurdle_bps()` is **3.904 bp** and
`labelling_hurdle_bps()` is **5.857 bp**.

### Horizons and validation

| Setting | Default | Meaning |
|---|---|---|
| `horizons` | `[1, 2, 3, 5]` | Forecast horizons in bars. Deliberately short — a 15-minute forecast is position trading |
| `n_splits` | `4` | Purged walk-forward folds |
| `embargo_bars` | `20` | Extra bars dropped after each fold, on top of the purge |
| `min_train_bars` | `4000` | Labelled samples required before a horizon is trainable |
| `train_n_jobs` | `4` | Folds trained in parallel |

### Position sizing

| Setting | Default | Meaning |
|---|---|---|
| `reference_notional` | `2_000_000` | Position size used to derive the cost hurdle |
| `lot_size` | `75` | NIFTY lot size — verify against the current contract spec |
| `dir` | `artifacts` | Where trained models and reports are written |

### Other model settings

| Setting | Default | Meaning |
|---|---|---|
| `expiry_weekday` | `1` (Tuesday) | NSE moved NIFTY weekly expiry from Thursday to Tuesday. Kept configurable |
| `session_open` / `session_close` | `09:15` / `15:30` | Session bounds, in IST |
| `pre_open_start` | `09:00` | Pre-open window start |

---

## `backtest`

| Setting | Default | Meaning |
|---|---|---|
| `slippage_bps` | `0.5` | Slippage per side, in basis points |
| `cost_bps` | `0.6` | Explicit cost component |

**Slippage is the setting that matters most.** Raising it by a basis point moves
the hurdle more than most modelling choices will. `backtest --slippage` overrides
it for one run.

---

## `plugins`

Per-plugin parameters are nested by plugin kind. The first directory becomes the
kind and the filename becomes the name, so
`config/plugins/forecast/projection.yaml` configures
`forecast:projection`. This is what makes a plugin tunable without editing code:

```yaml
# config/plugins/forecast/projection.yaml
bars_ahead: 5
```

Every mapping is merged into the kwargs the plugin's `build` receives, and the
`params` block in a plugin's manifest is applied underneath it. Run `niftypulse
plugins` to see which handles exist. See [Plugins](plugins.md).

Strategy packs use two common sections:

```yaml
# config/plugins/strategy/momentum.yaml
weights:
  macd_momentum: 0.8
  stochastic: 0.4       # 0 excludes it from the default ensemble
parameters:
  stochastic:
    oversold: 20.0
    overbought: 80.0
    hold_bars: 5
```

Weights control the default ensemble without removing direct access to a
strategy. Parameters are passed to that rule's constructor. Unknown strategy
names, non-mapping parameter blocks, and non-finite weights fail at startup
instead of being silently ignored.

Folders may be nested more deeply for organization. The first folder is still
the kind and the filename is still the plugin name. Both `.yaml` and `.yml` are
accepted. Defining two files that resolve to the same handle is rejected.

The old `plugins:` block in `default.yaml` remains supported for compatibility.
Its values override the separate file, but new settings should go in the plugin
tree.

| Handle | Parameter | Default | Meaning |
|---|---|---|---|
| `forecast:projection` | `bars_ahead` | `3` | How many candles to project |
| `source:simulated` | `seed`, `days` | `7`, `120` | Reproducibility and length of the generated series |
| `source:upstox` | `feed_mode` | `full` | `ltpc`, `full`, `full_d30`, `option_greeks` |
| `source:history` | `offline` | `false` | Serve the cache without fetching |
| `features:technical` | `include_context` | `true` | Session and calendar features |
| `advisory:breakeven_gate` | `cost_rate` | `0.012` | Round-trip cost as a fraction of premium |
| `advisory:breakeven_gate` | `horizon_bars` | `3` | Horizon the requirement is quoted over |
| `forecast:ml_ensemble` | `models`, `weights`, `parameters` | six learners | Batch model selection, soft-vote influence, and estimator hyperparameters |
| `forecast:online_research` | `algorithms` | all six | Mapping of learner names to parameters, including FTRL, adaptive KNN, and strategy combinations |
| `forecast:online_research` | `strategy_feature_limit` | `8` | Strongest strategy scores eligible for explicit interactions |
| `forecast:online_research` | `interaction_order` | `3` | Generate bounded pair/triple strategy interaction features |
| `forecast:online_research` | `rolling_window` | `100` | Matured predictions retained in the rolling score |
| `forecast:online_research` | `min_trust_samples` | `50` | Evidence needed before trust reaches full sample weight |
| `forecast:online_research` | `buy_probability`, `sell_probability` | `0.56`, `0.44` | Probability gates for research BUY and SELL classifications |
| `forecast:online_research` | `min_trust_for_action` | `0.15` | Below this trust, the research classification stays HOLD |
| `advisory:performance_ledger` | `min_trust_samples` | `50` | Strategy outcomes needed before trust reaches full sample weight |
| `strategy:ml_forecast` | `horizon`, `min_edge` | `1`, `0.04` | Which trained model, and the noise band |
| `renderer:terminal` | `refresh` | `1.0` | Seconds between frames |
| `renderer:terminal` | `chart_ratio`, panel heights | see YAML | Single-view layout sizing |
| `renderer:terminal` | `show_model_diagnostics` | `true` | Show the agreement/range prediction card |
| `renderer:web` | `host`, `port` | `127.0.0.1`, `5050` | Flask bind address for `dashboard --web` |
| `renderer:web` | `refresh_ms` | `1000` | Browser snapshot polling interval |
| `renderer:web` | `open_browser` | `true` | Open the local URL when the renderer starts |
| `renderer:web` | `max_candles` | `160` | Maximum printed candles sent per instrument |

`--web-host`, `--web-port`, and `--no-open-browser` override the corresponding
web renderer settings for one dashboard run. Keep the default loopback host unless
you intentionally want the unauthenticated, read-only view reachable from a
trusted network.

---

## Environment variables

| Variable | Meaning |
|---|---|
| `UPSTOX_CLIENT_ID` | From your app at account.upstox.com/developer/apps |
| `UPSTOX_CLIENT_SECRET` | — |
| `UPSTOX_REDIRECT_URI` | Must match the app registration exactly |
| `UPSTOX_ACCESS_TOKEN` | Optional; takes precedence over the stored token |
| `NIFTYPULSE_HOME` | Overrides the project root for `data/`, `artifacts/` and `logs/` |

Credentials are read from the environment first, then `.env`. The stored token in
`data/upstox_token.json` is used only if both are absent, and only if it has not
expired.

---

## Precedence

For a plugin parameter, highest first:

1. An explicit argument (`kernel.build("forecast:projection", bars_ahead=5)`)
2. The legacy `plugins:` override in `config/default.yaml`
3. `config/plugins/<kind>/<name>.yaml`
4. The `params` block in the plugin's `MANIFEST`
5. The default in the `build` signature

For credentials: environment → `.env` → stored token.

For paths: `NIFTYPULSE_HOME` → the current working directory.
