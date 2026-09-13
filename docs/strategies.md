# Strategies

18 rule-based strategies, plus an ML strategy, a composite ensemble, and a hybrid
blend. All share one interface.

```
src/niftypulse/strategies/
├── base.py         Strategy ABC, CompositeStrategy, helpers
├── trend.py        5 strategies
├── reversion.py    5 strategies
├── momentum.py     4 strategies
├── volatility.py   4 strategies
└── registry.py     Registry, MLStrategy, HybridStrategy, weights
```

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

### `default_ensemble()`

Blends all 18 strategies using `DEFAULT_WEIGHTS`:

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

These encode a prior — trend and flow carry more weight than oscillator fades —
and are meant to be replaced by measured weights after a backtest run.

### `HybridStrategy`

Blends rule-based conviction with the model probability at a configurable weight
(default 55% model). Rules are legible but brittle; models adapt but are opaque.
Keeping both in the loop means disagreement between them is itself visible.

---

## Registry

```python
from niftypulse.strategies import available_strategies, build_strategy, all_strategies

available_strategies()        # ['activity_spike', 'bollinger_reversion', ...]
build_strategy("ema_trend")   # instantiate by name
build_strategy("ensemble")    # the default composite
all_strategies()              # one instance of each
```

---

## Inspecting behaviour

```bash
uv run niftypulse strategies
```

Prints every strategy with its category and the share of bars where conviction
exceeds the entry threshold.

**"Fires 0%" is a finding, not a bug.** Two strategies report it against the
index because the index has no order book. Both activate against a futures
instrument.

To inspect raw scores:

```python
from niftypulse.data import MarketDataSource
from niftypulse.features import build_features
from niftypulse.strategies import all_strategies, StrategyContext

bars = MarketDataSource(settings, offline=True).load_history(days=20)
context = StrategyContext(bars=bars, features=build_features(bars))

for strategy in all_strategies():
    scores = strategy.score(context)
    print(f"{strategy.name:22s} std={scores.std():.3f} last={scores.iloc[-1]:+.3f}")
```

A useful sanity check: **a strategy whose scores are non-zero on nearly every bar
will dominate any weighted blend.** SuperTrend did exactly that before its quality
gate was added.

---

## Adding a strategy

1. Implement the class in the appropriate module, subclassing `Strategy`.
2. Define `name`, `category`, `description` as class attributes.
3. Implement `score()`, returning a Series in `[-1, 1]`. Return `self._empty(context)`
   if required columns are missing — never raise on absent inputs, because the
   strategy may legitimately be run against an instrument that lacks them.
4. Register it in `_registry()` in `registry.py`.
5. Add a weight to `DEFAULT_WEIGHTS` if it should join the ensemble.
6. Run `uv run niftypulse strategies` and check its firing rate is sane.

Guidance from experience on this codebase:

- **Gate on trend quality.** Strategies that always hold an opinion crowd out the
  rest of the blend.
- **Dampen, don't veto.** A multiplicative gate that can reach zero will silence
  a strategy on precisely the bars it was built for.
- **Prefer `sign()` of a spread to raw magnitudes** where the scale varies, so the
  signal is comparable across regimes.
