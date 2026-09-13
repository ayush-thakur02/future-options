# Plugins

The platform is built as a plugin tree. This document is the contract: what a
plugin is, how the kernel finds it, how plugins link to each other, and what to
do when you want to add one.

The short version: **a plugin is a folder containing a `plugin.py`.** Drop it in
the tree and the platform can use it. Nothing else registers it.

---

## What the kernel promises

| Property | What it means in practice |
|---|---|
| Discovery is automatic | Every `plugin.py` under `plugins/` is found, at any depth. Installed distributions advertising the `niftypulse.plugins` entry point are found too. |
| Nothing imports a plugin | Plugins link through **capabilities**, so one can be replaced by another that provides the same name. |
| A broken plugin is data | A pack that fails to import is reported, not fatal. One bad third-party pack cannot stop the platform loading its own. |
| A composition can be checked | `kernel.validate()` reports every unsatisfied requirement before anything runs. |
| Building is explicit | `build(handle)` memoises; `new(handle)` gives you another instance. |

---

## The contract

```python
# plugins/sources/simulated/plugin.py
from kernel import PluginContext, PluginKind, PluginManifest

MANIFEST = PluginManifest(
    name="simulated",
    kind=PluginKind.SOURCE,
    description="Generated NIFTY-like bars and ticks, replayable in real time",
    tags=("simulated", "offline", "replay"),
    params={"seed": 7, "days": 120},
)


def build(ctx: PluginContext, seed: int = 7, days: int = 120, **params) -> SimulatedMarket:
    return SimulatedMarket(settings=ctx.settings, seed=seed, days=days)
```

Two names, `MANIFEST` and `build`. No base class, no decorator, no registration
call. `MANIFEST` can also be a plain dict if you prefer.

`build` receives a [PluginContext](#the-context) and whatever parameters the
kernel merged for it, and returns anything the rest of the platform expects for
that kind.

---

## Kinds

A kind is the slot a plugin fills. It exists so a handle can read
`source:upstox` rather than a bare name that collides across roles.

| Kind | Provides | Ships as |
|---|---|---|
| `source` | Data from outside: ticks, bars, the chain | `upstox`, `simulated`, `history`, `option_chain` |
| `aggregator` | Tick → bar | `candle_builder` |
| `features` | Bar → model inputs | `technical` |
| `strategy` | Bar → signed conviction | `trend`, `momentum`, `reversion`, `volatility`, `ml_forecast` |
| `forecast` | Conviction → probabilities and projected paths | `ml_ensemble`, `projection` |
| `advisory` | A whole board → a verdict per leg | `breakeven_gate` |
| `renderer` | A snapshot → something a human reads | `terminal` |
| `tool` | Offline research that is not part of the live graph | (reserved) |

Forecast and advisory are deliberately separate. A forecast is a statement about
the *market*; advice is a statement about a *position*, which also has to account
for what holding it costs.

---

## Capabilities

Capabilities are how plugins link without importing each other. A manifest
declares what it `provides` and what it `requires`.

```
aggregator:candle_builder   provides  bars
features:technical          provides  features
forecast:projection         provides  projection
advisory:breakeven_gate     provides  advisory      requires projection
source:history              provides  history       requires broker
source:upstox               provides  broker
```

Three rules make this work:

1. **A capability resolves to exactly one plugin.** Two providers is a wiring
   bug, and it is rejected at registration rather than resolved by a coin toss.
2. **Requiring something nobody provides is reported before the run.** The
   strategy pack for the model declares `requires=("forecast",)`, so a build with
   the forecast pack disabled says so at composition time instead of failing at
   09:20 on a trading day.
3. **`provides` is for roles, not for alternatives.** Two sources both stream
   ticks, and only one can be active — so sources declare no capability at all,
   and the runtime picks one by handle or tag. The rule of thumb: if two
   implementations could sensibly both be registered, do not give them the same
   capability.

`niftypulse plugins --capabilities` prints the live index.

---

## The context

`build` is handed a `PluginContext` — the only object that crosses from the
kernel into plugin code:

```python
def build(ctx, **params):
    ctx.settings          # resolved Settings: instruments, horizons, cost hurdle
    ctx.logger            # a logger already namespaced to this plugin
    ctx.bus               # the event bus
    ctx.build("source:upstox")     # build another plugin by handle
    ctx.capability("features")     # build whichever plugin provides a name
    ctx.handle_for("features")     # just resolve, do not build
    ctx.has_capability("forecast")
```

A plugin can compose its own parts and nothing else. It cannot mutate the
registry, which is what keeps the graph a tree built from the leaves up.

---

## The event bus

Plugins also talk sideways over topics, for things that are pushed rather than
asked for:

| Topic | Payload |
|---|---|
| `tick` | `Tick` |
| `bar` | closed bar dict |
| `forecast` | `Prediction` |
| `projection` | `ForecastCandle` list |
| `snapshot` | `MarketSnapshot` or `BoardSnapshot` |
| `status` | feed status dict |
| `error` | exception |

The bus is synchronous and defensive: a handler that raises is reported once and
skipped, and every other subscriber still receives the event. It sits directly
under a live market feed, so a bad handler must not take the feed down.

---

## Parameters

Anything a plugin declares in `params`, or that appears under `plugins:` in
`config/default.yaml`, is merged into the kwargs `build` receives:

```yaml
plugins:
  forecast:projection:
    bars_ahead: 5
  source:simulated:
    days: 30
```

So tuning a plugin is a config edit, not a code change. Explicit parameters win
over configured ones:

```python
kernel.build("forecast:projection")                    # configured, memoised
kernel.build("forecast:projection", bars_ahead=5)      # fresh instance
kernel.new("forecast:projection")                      # fresh, ignores the cache
```

That last one matters: `build` memoises because a feed or a model must not be
constructed twice, but the options board needs three projections with three
independent states sharing one kernel — without `new` they would share one
forecaster and move in perfect lockstep.

---

## Writing a plugin

1. Make a folder under the right kind, with an `__init__.py` at every level.
2. Write `plugin.py` with a `MANIFEST` and a `build`.
3. Run `niftypulse plugins` — it should appear.
4. Add a test. `tests/test_kernel.py` covers the kernel; a plugin's own
   behaviour belongs in its own test module (`tests/test_board.py` for the
   advisory pack, `tests/test_projection.py` for the projection).

A third-party plugin is the same thing in an installed package:

```toml
[project.entry-points."niftypulse.plugins"]
my_feed = "my_package.plugin"
```

---

## What the runtime adds

The kernel finds and builds plugins. The *runtime* decides which ones a given
run wants and drives them:

| Module | Job |
|---|---|
| `runtime/session.py` | Composes a run: source, bars, engine, board, renderer |
| `runtime/bars.py` | Where the bar series comes from, decided in one place |
| `runtime/engine.py` | Rolling state per instrument; bar close and tick |
| `runtime/board.py` | Several instruments on one clock |
| `runtime/nowcast.py` | The refresh clock behind the projected candles |
| `runtime/conviction.py` | Where the projected path gets its direction |

Nothing in `runtime/` is a plugin. Keeping composition out of the plugin tree is
what lets any pack be replaced without touching the application that uses it.
