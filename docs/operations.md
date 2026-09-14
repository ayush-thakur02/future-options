# Operations

The browser dashboard is the only interface this platform has. Everything else
it used to offer as a command — logging in, building history, training,
backtesting, diagnostics — is **library code with no command of its own**, and
the two are deliberately separate: a live view answers "what is happening now",
and these answer "what is on disk and what does it say".

The snippets below are the entry points. Run them from the project root with
`uv run python -` and paste the body, or save them to a file and `uv run python
path/to/file.py`. Nothing here is hidden behind the dashboard — these are the
same functions the session itself calls.

---

## Start the dashboard

```bash
uv run niftypulse                                   # live Upstox feed, browser opens
uv run niftypulse --offline --speed 60              # replay generated ticks, 60x
uv run niftypulse --offline --no-open-browser       # serve without opening a browser
uv run niftypulse --port 8080 --host 0.0.0.0        # bind elsewhere (see the warning below)
uv run niftypulse --no-legs --refresh 1.0           # index only, one frame a second
uv run niftypulse --no-learn --workers -1           # freeze online learning, use every CPU
```

Refresh and nowcast are capped at one second: a live market view that is slower
than that is stale before it is drawn. The full flag list is in
[Dashboard](dashboard.md); the bind address defaults to the one in
`config/plugins/renderer/web.yaml`.

---

## Authenticate with Upstox

Tokens expire at 03:30 IST the following morning, so this is a once-per-day step.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from plugins.sources.upstox.auth import interactive_login

settings = load_settings()
credentials = settings.credentials
if not credentials.has_oauth:
    raise SystemExit("set UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET and UPSTOX_REDIRECT_URI in .env")

settings.ensure_dirs()
interactive_login(
    credentials.client_id,
    credentials.client_secret,
    credentials.redirect_uri,
    settings.token_path,
    open_browser=True,
)
PY
```

The redirect URI must match what is registered on the app exactly; `start.sh`
hosts a local callback for it through an Azure Dev Tunnel when the Upstox app
points at a public HTTPS URL.

---

## Build history

Candles come from the broker and are kept in the partitioned store. The store is
checked first, so a second run costs one API call when there is nothing new and
nothing at all when the cache is current.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from kernel import Kernel
from runtime.bars import BarLoader

settings = load_settings()
kernel = Kernel.bootstrap(settings)
bars = BarLoader(kernel).load(days=180, refresh=True, quiet=False, offline=False)
print(f"{len(bars):,} bars  {bars.index[0]} -> {bars.index[-1]}")
PY
```

`offline=True` generates a synthetic series instead of calling Upstox, which is
how the pipeline is exercised without credentials. Ticks and chain samples are
not back-fillable: the API publishes candles, not the tape that produced them,
and both are recorded only while a live session runs.

---

## Train the horizons

Training refuses to learn a move that cannot pay for its own round trip, so a
horizon where too few bars clear the hurdle is skipped with a reason rather than
trained into a losing model. `trainer.skipped` holds those reasons.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from kernel import Kernel
from plugins.features.technical import build_features, feature_columns
from plugins.forecasts.ml_ensemble.trainer import (
    Trainer,
    render_report,
    render_scalping_hurdle,
    save_training_metadata,
)
from runtime.bars import BarLoader

settings = load_settings()
bars = BarLoader(Kernel.bootstrap(settings)).load(days=400, refresh=False, quiet=True, offline=False)

features = build_features(bars, bar_minutes=settings.bar_minutes, expiry_weekday=settings.expiry_weekday)
columns = feature_columns(features)
print(f"{len(features):,} rows x {len(columns)} features")

trainer = Trainer(settings, n_jobs=settings.train_n_jobs)
results = trainer.train_all(features, bars, columns, horizons=settings.horizons)

render_scalping_hurdle(settings, bars)
render_report(results)

paths = trainer.save(results)
report_path = save_training_metadata(settings, results, skipped=trainer.skipped)
print(f"saved {len(paths)} artifacts to {settings.model_dir}")
print(f"saved the training report to {report_path}")
PY
```

Pass `horizons=(1, 3, 5)` to `train_all` to train a subset. The report prints to
the terminal as Rich tables; the artifact on disk carries the same numbers.

### List what is trained

```bash
uv run python - <<'PY'
from core.settings import load_settings
from plugins.forecasts.ml_ensemble.trainer import list_artifacts

for artifact in list_artifacts(load_settings()):
    metrics = artifact.get("metrics", {})
    print(
        f"{artifact['horizon']:>3}m  acc {metrics.get('accuracy', 0):.4f}  "
        f"auc {metrics.get('auc', 0):.4f}  lift {metrics.get('lift', 0):+.4f}  "
        f"n {metrics.get('n', 0):,}"
    )
PY
```

---

## Backtest a strategy

Execution enters on the next bar's open and positions size in whole lots, so a
backtest reports the notional you can actually reach rather than the one asked
for.

```bash
uv run python - <<'PY'
from backtest import BacktestConfig, CostModel, run_backtest
from core.settings import load_settings
from kernel import Kernel
from plugins.features.technical import build_features
from plugins.strategies import StrategyCatalog, StrategyContext
from runtime.bars import BarLoader

settings = load_settings()
kernel = Kernel.bootstrap(settings)
bars = BarLoader(kernel).load(days=200, refresh=False, quiet=True, offline=False)

features = build_features(bars, bar_minutes=settings.bar_minutes, expiry_weekday=settings.expiry_weekday)
context = StrategyContext(bars=bars, features=features)
strategy = StrategyCatalog.from_kernel(kernel).get("ensemble")

config = BacktestConfig(
    horizon=5,
    entry_threshold=0.15,
    notional=2_000_000,
    lot_size=75,
    cost_model=CostModel(slippage_bps=0.5),
)
result = run_backtest(bars, strategy.score(context), config, strategy=strategy.name)
for line in result.report.lines():
    print(line)
print("net expectancy", result.report.expectancy_bps, "bps")
PY
```

Swap `strategy = ...` for the walk-forward path to backtest the model instead of
a rule. The ML backtest trades out-of-sample predictions only:

```python
import pandas as pd

forecast = pd.read_parquet(settings.model_dir / "oof_5m.parquet")
scores = 2.0 * forecast["calibrated"] - 1.0
```

### Sweep the entry threshold

```bash
uv run python - <<'PY'
from backtest import BacktestConfig, CostModel, run_threshold_sweep
# ... load bars and scores exactly as above ...

config = BacktestConfig(horizon=5, entry_threshold=0.15, cost_model=CostModel(slippage_bps=0.5))
print(run_threshold_sweep(bars, scores, config, strategy="ensemble").to_string())
PY
```

Net expectancy should improve as the threshold rises. If it does not, the signal
carries no information about its own confidence. Raise slippage before believing
any of it: one basis point moves the hurdle more than most modelling choices.

---

## Check the environment

The checks the dashboard depends on, without starting one.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from kernel import Kernel
from plugins.forecasts.ml_ensemble.models import lightgbm_available
from plugins.sources.history import StoreManifest

settings = load_settings()
kernel = Kernel.bootstrap(settings)
broker = kernel.build("source:upstox")

print("lightgbm     ", "available" if lightgbm_available() else "unavailable (brew install libomp)")
print("upstox token ", "present" if broker.token else "missing")
print("cpu workers  ", settings.worker_count)
print("plugins      ", len(kernel.registry), "providing", len(kernel.registry.capabilities()), "capabilities")
print("load errors  ", kernel.load_report.errors or "none")
for problem in kernel.validate():
    print("unresolved   ", problem)

manifest = StoreManifest(settings.data_dir / "manifest.json").read()
for dataset, instruments in manifest.get("datasets", {}).items():
    for key, entry in sorted(instruments.items()):
        print(f"{dataset:>8}  {key}  {int(entry.get('rows', 0)):,} rows")
PY
```

Add `broker.validate_auth()` and `authorize_feed_url(broker.validate_auth())`
(from `plugins.sources.upstox.feed`) to check the live feed as well, which is
what `--live` used to do.

---

## Inspect the plugin graph

Everything is a plugin, so this is the wiring report: what was discovered, what
it provides, and what is missing.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from kernel import Kernel

kernel = Kernel.bootstrap(load_settings())
for entry in kernel.entries():
    origin = "installed" if entry.is_third_party else "bundled"
    print(f"{entry.handle:28} {origin:9} {', '.join(entry.manifest.provides) or '—'}")

print()
for capability, handle in sorted(kernel.registry.capabilities().items()):
    print(f"{capability:24} {handle}")
PY
```

---

## Inspect the strategies

Firing rate is the share of bars where the strategy holds a conviction above the
entry threshold. A strategy that never fires is not a signal.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from kernel import Kernel
from plugins.features.technical import build_features
from plugins.strategies import StrategyCatalog, StrategyContext
from runtime.bars import BarLoader

settings = load_settings()
kernel = Kernel.bootstrap(settings)
bars = BarLoader(kernel).load(days=20, refresh=False, quiet=True, offline=True)
features = build_features(bars, bar_minutes=settings.bar_minutes)
context = StrategyContext(bars=bars, features=features)

catalog = StrategyCatalog.from_kernel(kernel)
for instance in catalog.all():
    fires = (instance.score(context).abs() > 0.15).mean() * 100
    print(f"{instance.name:28} {fires:5.1f}%  {instance.description}")
for handle, reason in catalog.skipped.items():
    print(f"skipped {handle}: {reason}")
PY
```

---

## Read the research stores

The dashboard writes what it forecasts and what actually happened; these are the
readers for those files. They are ordinary SQLite databases under
`data/research/`, and they are the durable record — nothing expires except the
targets that never printed.

```bash
uv run python - <<'PY'
from core.settings import load_settings
from plugins.advisory.performance_ledger import StrategyPerformanceLedger
from plugins.forecasts.online_research import OnlineResearchLab

settings = load_settings()
research = settings.data_dir / "research"

for state_path in sorted((research / "online_ai" / "live").glob("*.sqlite3")):
    lab = OnlineResearchLab(state_path)
    for card in lab.scorecards():
        print(
            f"{state_path.stem:10} {card.algorithm:18} n {card.all_time.sample_count:>6}  "
            f"acc {card.all_time.accuracy * 100:5.1f}%  trust {card.trust_score * 100:5.1f}%"
        )

strategy_path = research / "strategies.sqlite3"
if strategy_path.exists():
    ledger = StrategyPerformanceLedger(strategy_path)
    print(ledger.counts())
    for card in ledger.scorecards():
        print(
            f"{card.instrument:10} {card.strategy:28} n {card.samples:>6}  "
            f"net {card.net_pnl_bps:+.2f} bp  trust {card.trust_score * 100:5.1f}%"
        )
PY
```

Use `"simulation"` in place of `"live"` to read what the offline runs recorded.
The tables themselves are plain SQL — `data/research/live.sqlite3` holds issued
forecasts and their scored outcomes, keyed by instrument, timeframe and horizon.

---

## Where the rest lives

| You want to… | Look at |
|---|---|
| Read the dashboard | [Dashboard](dashboard.md) |
| Know what a setting does | [Configuration](configuration.md) |
| Know what is on disk | [Storage](storage.md) |
| Interpret a training run | [ML pipeline](ml-pipeline.md) |
| Change or add a component | [Plugins](plugins.md) |
| Fix something broken | [Troubleshooting](troubleshooting.md) |
