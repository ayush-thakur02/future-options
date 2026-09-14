# Development

Conventions, tests, and how to extend the platform.

---

## Setup

```bash
uv sync --all-extras
uv run pytest
```

`--all-extras` includes `grpcio-tools` (for regenerating protobuf stubs) and
`ruff`.

```bash
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix
```

Line length 100, target Python 3.11. Rules: `E`, `F`, `I`, `UP`, `B`.

---

## Testing philosophy

The full suite covers every layer. The interesting tests are not the ones that
check a function returns a value; they are designed to **fail in a specific way if something
subtly breaks.**

### Positive and negative controls

`test_pipeline_learns_deterministic_pattern`
`test_pipeline_finds_no_edge_in_random_walk`

Together these bracket the ML pipeline. The first injects an alternating-drift
series where direction over the next 5 bars is almost perfectly determined by the
recent past, and asserts **AUC > 0.80**. The second feeds a driftless random walk
and asserts **AUC < 0.60**.

Neither is interesting alone. Together they are the only thing that lets you
interpret a null result on real data: if the negative control passes and the
positive control passes, then "found nothing" means there was nothing to find,
rather than that the pipeline is broken.

Without them, an AUC of 0.50 on real data is indistinguishable from a bug.

### Causality

`test_features_do_not_use_future_data`

Truncates the input to a prefix and asserts features on the overlapping rows are
**bit-for-bit unchanged**. Any indicator that peeks forward fails.

This replaced a manual review process and immediately caught a real leak: the
prior-session features originally used a whole-day maximum, which told a 09:30
bar where the day's high would end up.

### Regression guards

Several tests exist because a specific bug shipped and was found later. Each
documents the failure it prevents:

| Test | The bug it catches |
|---|---|
| `test_the_render_loop_draws_inside_the_live_block` | Frames published without opening the renderer's live block, so the server was never bound and nothing answered on the port |
| `test_probability_series_is_not_valid_conviction` | Probabilities passed to the backtester where it expected signed conviction, so `abs(0.48)` cleared every threshold |
| `test_position_sizing_targets_notional_when_affordable` | Position sized as `price × notional` rather than `price × units`, inflating exposure to ₹24 billion |
| `test_costs_are_plausible_magnitude` | Same bug, caught via an implausible 3.6-million-bps cost drag |
| `test_calibrator_rejects_spread_collapse` | Isotonic calibration collapsing 85% of predictions onto 0.50 |
| `test_donchian_prior_channel_excludes_current_bar` | Breakout channel included the bar under test, so the feature could never fire |
| `test_hurdle_makes_short_horizons_untrainable` | A sub-cost horizon being trained into a losing model |

> **A regression test that cannot fail is worthless.** Each guard above was
> verified by reintroducing the original bug and confirming the test fails —
> including the render-loop one, which has to reproduce the bug exactly to fail at
> all. A guard that passes against the broken code is worse than no guard, because
> it gives false confidence. Always confirm yours fails first.

### What is not tested

- **The live WebSocket path.** It needs Upstox credentials and a market session.
  The decoder is exercised via a synthetic protobuf round-trip instead, and the
  loader's import of every plugin catches structural breakage (it found a
  generated protobuf stub that could not be imported standalone).
- **Real market data.** Everything runs on synthetic data. See
  [Getting started](getting-started.md#first-run--no-credentials-needed).

---

## Test layout

```
tests/
├── conftest.py          Shared fixtures: bars, long_bars, trend_bars, rng, offline_session
├── test_kernel.py       Discovery, registry, capabilities, the bus, the loader
├── test_data.py         Aggregation, resampling, ingest round trips, the calendar
├── test_storage.py      Partitioning, the manifest, recorders, legacy migration
├── test_features.py     Causality, indicator correctness, feature assembly
├── test_strategies.py   Pack discovery, manifest/class agreement, firing rates, gates
├── test_backtest.py     Splits, execution timing, sizing, costs, reporting
├── test_ml.py           Labelling, CV, models, calibration, hurdle, training
├── test_option_chain.py Pricing identities, the chain, the clock, premium candles
├── test_projection.py   Path geometry, decay, tick momentum, the scoreboard
├── test_board.py        Multi-instrument composition and verdicts
├── test_runtime.py      Conviction, the engine, the refresh clock, the feed
├── test_web_dashboard.py  The web renderer, its JSON boundary and its API
└── test_webapp.py       The launcher: flags, validation, where they land
```

The modules follow the tree: one test file per area that has behaviour worth
pinning, and a plugin's behaviour belongs in its own module rather than in the
kernel's.

Fixtures in `conftest.py` are session-scoped where the data is expensive to
generate, and seeded so results are reproducible.

---

## Conventions

### Comments explain *why*

The codebase has a high comment density, but they explain reasoning rather than
mechanics. A comment saying what the next line does is noise; a comment saying
why a non-obvious choice was made is what stops someone "fixing" it later.

```python
# GOOD
# The floor matters. RSI extremes and high ADX occur together almost by
# construction — a sustained decline is what produces both — so a dampener that
# can reach zero silences the strategy on exactly the bars it exists to trade.

# BAD
# Apply the dampener
```

Several comments in this codebase document a bug that was fixed. Those are worth
keeping: they are the difference between a future reader understanding the
constraint and removing it.

### Fail loudly, or degrade gracefully — deliberately

Choose one and be explicit about which.

- **Degrade:** missing `libomp` drops LightGBM from the ensemble. The state
  problem is an install issue, not a code issue.
- **Refuse:** a horizon whose moves cannot cover costs is skipped. Training it
  would produce something worse than nothing.
- **Return zero:** `order_flow` reports no opinion against an index because the
  index publishes no order book. Fabricating a view would be a lie.

### Vectorise, don't loop

Strategies and indicators operate on whole Series. A per-bar Python loop is
usually 10–100x slower and makes the live and research paths diverge.

When a loop is unavoidable (SuperTrend's ratcheting bands are path-dependent),
keep it in `indicators.py` with a comment saying why.

### No silent `inf`

`build_features` replaces `inf` with `NaN`. `NaN` is a value LightGBM and the
imputer handle correctly; `inf` is not, and it propagates through arithmetic in
ways that are hard to trace.

---

## Extending

### Add a plugin

The whole installation procedure:

1. Make a folder under the right kind in `src/plugins/`, with `__init__.py` at
   every level.
2. Write `plugin.py` with a `MANIFEST` and a `build(ctx, **params)`.
3. Ask the kernel — `Kernel.entries()` should list it with its capabilities.
4. Add tests in a module of its own.

If it fills a slot something else already fills, either declare a *different*
capability or give it none and let the runtime pick it by handle. Two providers of
one capability is rejected at registration, deliberately: a capability that
resolved to different plugins on different runs would make a composition
impossible to reason about. See [Plugins](plugins.md).

### Add an indicator

1. Implement in `plugins/features/technical/indicators.py`, causal and vectorised.
2. Add it in the relevant `_add_*` function in `pipeline.py`, appending the name
   to that group's `columns` list.
3. New group? Call `_register(group, columns)`.
4. `uv run pytest tests/test_features.py` — the causality test catches lookahead
   automatically.
5. Retrain. Feature count changes invalidate artifacts.

### Add a strategy

Add the class to the right `rules.py`, then add its name to that module's
`STRATEGIES` tuple. The pack's manifest derives one `strategy:<name>` capability
per class from that tuple, so a rule cannot exist in code without being
advertised — and a test asserts both directions.

See [Strategies § Adding a strategy](strategies.md#adding-a-strategy).

Key guidance: gate on trend quality, dampen rather than veto, and check the firing
rate through `StrategyCatalog` — see
[Operations § Inspect the strategies](operations.md#inspect-the-strategies).
**A strategy that holds an opinion on nearly every bar will dominate any weighted
blend** — SuperTrend did exactly that before its quality gate was added.

### Add a cost component

Add a field to `CostModel` with a documented default, include it in `per_leg_bps`
on the correct leg, and note whether GST applies.

### Add a dashboard section

Sections are plain HTML in
`src/plugins/renderers/web/templates/dashboard.html`, each a numbered heading and
an empty container:

```html
<section class="section-block">
  <article class="terminal-card">
    <div class="card-title">
      <h2><span>09</span> MY PANEL</h2>
      <small>WHAT IT SHOWS</small>
    </div>
    <div id="my-panel"></div>
  </article>
</section>
```

The client fills it from the payload `serialize_snapshot` builds, so a new value
has to exist in `core.types` — or under a snapshot's `research` map — before it
can be drawn. Change the browser poll cadence, the bind address or the candle
limit in `config/plugins/renderer/web.yaml`. New data belongs in `_routes` in
`server.py`, beside `/api/snapshot` and `/api/health`, and the payload must stay
JSON-safe: `tests/test_web_dashboard.py` serializes it with `allow_nan=False`.

### Add a metric

Extend `PerformanceReport` as a dataclass field, compute it in `build_report`,
and add it to both `lines()` and `as_row()`. Keep per-trade figures in basis
points and gross and net separate — a report that only shows net PnL hides why a
strategy is losing.

---

## Performance notes

Measure before optimising. Two examples where the obvious culprit was wrong:

**`hurst_exponent` was 71% of feature time.** It used `rolling.apply` with a
Python callback that looped over the lags — 5,703 calls per recompute. Vectorised
across windows with `sliding_window_view`, verified against the reference to
2.2e-16, roughly 7x faster. Feature computation went from ~360 ms to ~50 ms.

**Fold parallelism needed `n_jobs=1` inside.** Setting both the fold level and
the estimator level to `-1` oversubscribes the CPU and runs *slower* than either
alone, because LightGBM spawns threads that contend for the same cores.

### Profiling

```bash
uv run python -c "
import cProfile, pstats, sys
sys.path.insert(0, 'src')
from plugins.sources.simulated.series import generate_candles
from plugins.features.technical import build_features

bars = generate_candles(days=6, seed=1).tail(2000)
build_features(bars)
pr = cProfile.Profile(); pr.enable()
for _ in range(3): build_features(bars)
pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(12)
"
```

---

## Repository hygiene

- **Commit the generated protobuf stubs.** The package must install without a
  protoc toolchain.
- **Never commit `.env`, `data/`, or `artifacts/`.** The `.gitignore` entries are
  anchored to the repository root (`/data/`) — unanchored, `data/` also matches
  `src/…/data/`, which is how the entire data layer once stayed out of version
  control without anyone noticing.
- **The layout is flat.** Everything lives directly under `src/`, so the
  top-level packages are `core`, `kernel`, `plugins`, `runtime` and `backtest`,
  alongside the `webapp` launcher module. Any import that crosses a top-level
  boundary is written absolutely; intra-package imports stay relative.
- **Verify rate assumptions before changing the cost model.** A one basis point
  change in slippage moves the hurdle by 25%.

---

## Before opening a PR

```bash
uv run ruff check src/ tests/
uv run pytest
uv run niftypulse --offline --speed 60   # eyeball the board
```

If you changed anything structural — feature count, horizons, cost model, bar
size — say so in the description, because artifacts trained before the change are
no longer valid.

If you fixed a bug, add a regression test **and verify it fails against the
broken code.** A guard that cannot fail is worse than no guard, because it
produces confidence that is not earned.
