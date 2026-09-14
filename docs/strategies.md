# Strategies

**30 rule-based strategies**, plus the trained model as a strategy, plus a
composite ensemble. All share one interface.

```
src/plugins/strategies/
├── base.py       Strategy ABC, CompositeStrategy, StrategyPack, helpers
├── catalog.py    StrategyCatalog + DEFAULT_WEIGHTS
├── trend/        5 rules  + plugin.py    provides strategy:ema_trend, ...
├── momentum/     5 rules  + plugin.py
├── reversion/    5 rules  + plugin.py
├── volatility/   4 rules  + plugin.py
├── channels/     3 rules  + plugin.py
├── flow/         3 rules  + plugin.py
├── regime/       3 rules  + plugin.py
├── statistical_anchor/  2 paired rules + plugin.py
└── ml_forecast/  1 rule   + plugin.py    requires "forecast"
```

Each pack's `rules.py` exposes a `STRATEGIES` tuple and its `plugin.py` derives one
`strategy:<name>` capability per class from it, so a rule cannot exist in code
without being advertised — or be advertised without existing. A test asserts both
directions.

---

## The contract

```python
class Strategy(ABC):
    name: ClassVar[str]
    category: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def score(self, context: StrategyContext) -> pd.Series:
        """Signed conviction in [-1, 1] for every bar."""
```

Strategies are **vectorised**: given the full feature frame, they return a
conviction *series* for every bar at once.

That is a deliberate departure from the event-driven style common in trading
frameworks. A rule evaluated per-bar in a loop cannot be validated until it has
been looped over, which makes research slow and mistakes expensive. With a
vectorised form:

- a backtest is a single pass over a Series
- live inference is `.iloc[-1]`
- **the same code produces both**, so a strategy that looks good in research is
  literally the strategy running live, not a re-implementation of it

Conventions:

- `+1` is maximum bullish conviction, `−1` maximum bearish, `0` no opinion
- Conviction is *graded*, not binary. A weak setup produces a small number.
- Every strategy returns zeros when its inputs are unavailable, rather than a
  fabricated opinion
- `_empty(context)` produces that zero series

### Helpers

| Helper | Purpose |
|---|---|
| `squash(series, scale)` | Maps unbounded scores into `[-1, 1]` via `tanh` |
| `gaussian_bell(series, center, width)` | Peaked weighting for threshold bands |
| `_trend_dampener(features)` | Scales a reversion signal down as ADX rises |

---

## Trend

### `ema_trend`
> EMA 9/21 alignment confirmed by ADX trend strength

**Category:** trend · **Fires:** ~49% of bars

Three-way EMA stack alignment (9 vs 21, 21 vs 50, 9 vs 50), graded rather than
binary, scaled by DI spread and gated on ADX. The gate matters: a cross in a
range-bound market is noise.

### `supertrend`
> ATR-based SuperTrend direction with distance-scaled conviction

**Category:** trend · **Fires:** ~55% of bars

Direction from the SuperTrend flip, scaled by how long the trend has held.

**SuperTrend is always either long or short**, so without a quality gate it holds
a strong opinion in dead chop and dominates any blend it sits in. This version
gates on efficiency ratio *and* ADX, which brought it from firing on 99.9% of bars
to 55%.

### `donchian_breakout`
> 20-bar channel breakout, filtered by Bollinger squeeze release

**Category:** trend · **Fires:** ~12% of bars

Breakout measured against the **prior** 20-bar channel, then gated on squeeze
release — a breakout from compression is worth more than one in an already-wide
range.

### `orb`
> Break of the first-15-minute range, active until midday

**Category:** trend · **Fires:** ~59% of bars

Opening range breakout. The opening range is computed per session, and the
strategy is only active between 15 and 240 minutes from the open — outside that
window the pattern has no edge and firing anyway would just add noise.

### `efficiency_trend`
> Kaufman efficiency ratio: follows only clean, low-noise moves

**Category:** trend · **Fires:** ~54% of bars

Efficiency ratio near 1 means price travelled in a straight line. Scales a
direction taken from the 20- and 50-bar regression slopes.

---

## Reversion

All five use `_trend_dampener`, and the **floor on that dampener is important**.
RSI extremes and high ADX occur together almost by construction — a sustained
decline produces both. A dampener that can reach zero silences the strategy on
exactly the bars it exists to trade, which is what an earlier version did:
`rsi_reversion` fired on 0.0% of bars. The current version shrinks the position
rather than vetoing it.

### `rsi_reversion`
> Fade RSI extremes, damped by trend strength

**Category:** reversion · **Fires:** ~2% of bars

Fades RSI below 30 or above 70, scaled by how extreme the reading is.

### `bollinger_reversion`
> Fade closes outside the Bollinger band, damped by trend strength

**Category:** reversion · **Fires:** ~15% of bars

Fades `%B` beyond 0.05 / 0.95.

### `vwap_reversion`
> Fade extension away from session VWAP

**Category:** reversion · **Fires:** ~21% of bars

Only the deviation *beyond* a threshold counts, so small distances near VWAP do
not accumulate into a signal.

### `zscore_reversion`
> Fade price deviations from its 100-bar mean

**Category:** reversion · **Fires:** ~5% of bars

Fades moves beyond 2σ.

### `range_fade`
> Fade extremes of the day's running range in low-volatility chop

**Category:** reversion · **Fires:** ~16% of bars

Fades the top and bottom decile of the session's *running* range, and requires a
quiet market (ADX < 25).

---

## Momentum

### `macd_momentum`
> MACD histogram level and slope, normalised by price

**Category:** momentum · **Fires:** ~33% of bars

Level says where we are; slope says whether it is still building. Weighted 60/40.

### `stochastic`
> Stochastic %K/%D crossover at range extremes

**Category:** momentum · **Fires:** ~36% of bars

A crossover only counts below 20 or above 80, and the signal is held for 5 bars
because a crossover is a one-bar event and otherwise untradeable.

### `activity_spike`
> Directional move on unusual activity (volume, or tick count for an index)

**Category:** momentum · **Fires:** **0% against the index**

A move backed by abnormal participation is more likely to continue.

**Returns no opinion for NIFTY 50.** The index has no volume, so the activity
features fall back to tick count, and the z-score does not reach the threshold
reliably. This is correct behaviour, not a dead strategy — it activates against a
futures instrument.

### `order_flow`
> Order-book and traded-value imbalance from futures depth

**Category:** flow · **Fires:** **0% against the index**

Blends `depth_imbalance` (bid/ask quantity), `trade_imbalance` (total buy vs sell
quantity), and their moving average.

**Returns no opinion for NIFTY 50**, because the index publishes no order book.
Returning zeros is the right behaviour: it lets the ensemble include the strategy
for instruments that support it without pretending it has a view when the inputs
are absent.

> **This is where short-timescale edge actually lives.** Order flow is the
> primary signal at scalping horizons. To use it, point `instrument_key` at the
> front-month NIFTY future.

---

## Volatility

### `squeeze_release`
> Trade the expansion after a Bollinger squeeze, in the breakout's direction

**Category:** volatility · **Fires:** ~25% of bars

Requires the band to have been compressed recently *and* to be expanding now, with
Keltner position disambiguating which edge broke.

### `vol_breakout`
> Range expansion beyond ATR, in the direction of the break

**Category:** volatility · **Fires:** ~9% of bars

Conviction scales with how far past the threshold the expansion ran.

### `vol_reversion`
> Fade an outsized candle once volatility starts to subside

**Category:** volatility · **Fires:** ~0.2% of bars

The opposite bet to `vol_breakout`, which is what makes the two useful together.
Deliberately rare.

### `vol_regime`
> Volatility regime tilt combined with trend direction

**Category:** volatility · **Fires:** ~40% of bars

Volatility alone says nothing about direction, so this combines an expanding-vol
tilt with SuperTrend direction and a cleanliness filter.

---

## Channels

| Strategy | Mathematics |
|---|---|
| `keltner_continuation` | Follows acceptance beyond a Keltner envelope when ADX and directional movement agree |
| `regression_channel_breakout` | Trades a least-squares channel break, scaled by regression fit and slope agreement |
| `channel_pressure` | Measures repeated closes near one side of a rolling range and gates them on directional strength |

## Flow

| Strategy | Mathematics |
|---|---|
| `chaikin_flow_trend` | Confirms directional momentum with Chaikin money flow and participation |
| `money_flow_reversal` | Requires MFI to turn out of an extreme with candle-location confirmation |
| `obv_divergence` | Compares price and OBV slopes and requires sufficient activity |

The flow pack returns zero when an instrument has neither volume nor a tick-count
proxy. Absence of activity data is not interpreted as neutral flow evidence.

## Adaptive regime

| Strategy | Mathematics |
|---|---|
| `hurst_adaptive` | Follows persistent markets and fades anti-persistent markets using rolling Hurst |
| `directional_entropy` | Follows sign imbalance only when binary entropy says recent direction is predictable |
| `volatility_state_rotation` | Uses momentum in volatility expansion and fades z-score stretch in contraction |

## Statistical anchor

| Strategy | Mathematics |
|---|---|
| `anchor_spread_reversion` | Estimates rolling beta, standardizes the log-price spread and fades an extreme |
| `relative_strength_rotation` | Follows beta-adjusted relative strength against an aligned reference series |

These paired rules require `anchor_close` in `StrategyContext.extras`. Option
engines receive the aligned index close from the board. A standalone instrument
without an anchor gets a zero series, keeping the rule causal and explicit.

All eleven additions are independent plugin classes, bounded to `[-1, 1]`, and
covered by finite-output and prefix-causality tests.

---

## ML strategy

### `ml`
> Gradient-boosted ensemble direction forecast

**Category:** ml

Wraps a trained model. Conviction is `2p − 1`, scaled by a confidence gate, so a
0.52 probability produces a small signal rather than a full-size trade.

Two gates:

1. **Confidence** — below `min_edge` (default 0.04) the model is within noise and
   reports no opinion.
2. **Cost** — when a fitted conviction-to-move curve is supplied, the strategy
   stays silent unless the expected move clears the round-trip hurdle. Most bars
   do not. That is the point.

See [Scalping economics](scalping-economics.md).

---

## Composition

### `CompositeStrategy`

Weighted blend of component scores, normalised by total weight. A component
returning all zeros (its inputs unavailable) simply contributes nothing.

### `catalog.ensemble()`

Blends every strategy the loaded packs offer. The original rules use the priors
in `DEFAULT_WEIGHTS`:

```python
DEFAULT_WEIGHTS = {
    "ema_trend": 1.0,        "macd_momentum": 0.8,
    "supertrend": 0.9,       "roc_momentum": 0.7,
    "efficiency_trend": 0.8, "squeeze_release": 0.7,
    "donchian_breakout": 0.7,"vol_breakout": 0.6,
    "orb": 0.6,              "vol_regime": 0.5,
    "rsi_reversion": 0.7,    "stochastic": 0.4,
    "bollinger_reversion":0.7,"activity_spike": 0.5,
    "vwap_reversion": 0.6,   "order_flow": 0.9,
    "zscore_reversion": 0.5, "vol_reversion": 0.4,
    "range_fade": 0.4,
}
```

These encode fallback priors. Each bundled pack overrides them from its
`config/plugins/strategy/<pack>.yaml` file, keeping ownership with the plugin.
Set a configured weight to `0` to leave the strategy directly available while
excluding it from the default ensemble. A newly discovered strategy falls back
to its class-level `default_weight` (0.5 unless the plugin declares another
value), so adding a pack does not require changing the central map.

Weights naming a strategy this build does not have are dropped silently, which is
what makes the default map usable offline where the model pack has no artifacts.

---

## The catalog

```python
from core.settings import Settings
from kernel import Kernel
from plugins.strategies import StrategyCatalog

kernel = Kernel.bootstrap(Settings())
catalog = StrategyCatalog.from_kernel(kernel)

catalog.summary()             # "30 strategies (channels 3, flow 3, ..., volatility 4)"
catalog.names()               # sorted rule names
catalog.get("ema_trend")      # one instance
catalog.get("ensemble")       # the default composite
catalog.all()                 # one instance of each
catalog.pack_of("order_flow") # which pack offers it
catalog.skipped               # packs that could not be built, and why
```

`from_kernel` builds every `kind=strategy` plugin and records rather than raises
the ones that fail. That is not defensive for its own sake: offline, with no
trained artifacts, the model pack legitimately cannot be built, and the run must
still produce a working rule-based engine.

### Persistent forward scorecards

The `advisory:performance_ledger` plugin freezes each active rule signal for a
three-bar target. It scores only the exact target bar and stores accuracy, wins,
losses, gross return, configured cost, profit, loss, net return, mean net return,
maximum drawdown and conservative trust per strategy and instrument. Missing the
target expires the record instead of scoring a later price. The ledger survives
restarts in `data/research/strategies.sqlite3` and is shown both on the dashboard
and by `StrategyPerformanceLedger` —
[Operations § Read the research stores](operations.md#read-the-research-stores).

---

## Inspecting behaviour

`StrategyCatalog` lists every strategy with its category and the share of bars
where conviction exceeds the entry threshold — see
[Operations § Inspect the strategies](operations.md#inspect-the-strategies) for
the runnable form.

**"Fires 0%" is a finding, not a bug.** Two strategies report it against the
index because the index has no order book. Both activate against a futures
instrument.

To inspect raw scores:

```python
from core.settings import Settings
from kernel import Kernel
from plugins.features.technical import build_features
from plugins.sources.simulated.series import generate_candles
from plugins.strategies import StrategyCatalog, StrategyContext

bars = generate_candles(days=20, seed=7)
features = build_features(bars)
context = StrategyContext(bars=bars, features=features)
catalog = StrategyCatalog.from_kernel(Kernel.bootstrap(Settings()))

for strategy in catalog.all():
    scores = strategy.score(context).dropna()
    print(f"{strategy.name:22s} std={scores.std():.3f} last={scores.iloc[-1]:+.3f}")
```

NaN in the warm-up window is correct — a 200-bar EMA has no value on bar 3 — and
the consumers drop or fill it. What must never happen is a NaN *after* warm-up:
the ensemble fills those with zero, so a broken rule would look like a rule with
no opinion. A test asserts exactly that.

A useful sanity check: **a strategy whose scores are non-zero on nearly every bar
will dominate any weighted blend.** SuperTrend did exactly that before its quality
gate was added.

---

## Adding a strategy

1. Implement the class in the right `rules.py`, subclassing `Strategy`.
2. Define `name`, `category`, `description` as class attributes.
3. Implement `score()`, returning a Series in `[-1, 1]`. Return `self._empty(context)`
   if required columns are missing — never raise on absent inputs, because the
   strategy may legitimately be run against an instrument that lacks them.
4. Add the class to that module's `STRATEGIES` tuple. The manifest derives its
   capabilities from there, so nothing else needs registering.
5. Override `default_weight` on the class only if the standard 0.5 ensemble weight
   is inappropriate.
6. Check its firing rate is sane —
   [Operations § Inspect the strategies](operations.md#inspect-the-strategies).

`roc_momentum` sat unregistered through several versions of the old hand-written
registry, weighting a strategy that never ran. The tuple exists so that cannot
happen again.

Guidance from experience on this codebase:

- **Gate on trend quality.** Strategies that always hold an opinion crowd out the
  rest of the blend.
- **Dampen, don't veto.** A multiplicative gate that can reach zero will silence
  a strategy on precisely the bars it was built for.
- **Prefer `sign()` of a spread to raw magnitudes** where the scale varies, so the
  signal is comparable across regimes.
