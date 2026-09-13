# Backtesting

Simulating execution honestly, which is harder than it sounds.

```
src/niftypulse/backtest/
├── costs.py     Indian derivatives cost model
├── engine.py    Event-driven execution
└── report.py    Performance statistics
```

---

## Execution model

`engine.py`

Execution timing is the detail that matters most, and it is where most backtests
quietly cheat.

### Enter on the next bar's open

A signal computed from bar *t*'s close **cannot be filled at bar *t*'s close**.
By the time you know the signal, that price is gone.

So the engine always enters at the **next bar's open** — the first price actually
available to a decision made at the close.

```
bar t-1        bar t          bar t+1        bar t+2 ... t+h
                 │              │
    signal ──────┘              │
                 decision       │
                                │
    entry ──────────────────────┘  (open)
                                │
    exit ───────────────────────┴──────────────▶ (close at t+h)
```

This is asserted by `test_engine_enters_at_next_bar_open`.

### One position at a time

A signal arriving while a position is open is **ignored**, and the engine resumes
scanning from the bar after the exit.

Overlapping trades are how a strategy appears to have far more edge than it does,
because the same move gets counted repeatedly.

### No overnight holds

A trade whose exit would fall on a different session is **skipped entirely**
rather than carried overnight.

The platform forecasts intraday moves. Holding through the gap would be trading a
different phenomenon with different risk, and the label set was built to exclude
exactly that.

### Position sizing

Positions size in **whole lots**, which is how index futures trade:

```python
lots = max(round(notional / (price × lot_size)), 1)
units = lots × lot_size
```

The realised exposure is therefore the nearest whole lot to the target, and
**at least one**.

> **This is a binding constraint, not a rounding nicety.** NIFTY near 24,000 with
> a 75-unit lot makes the smallest possible position about ₹1.8 million. A target
> of ₹1 million is simply unreachable, and the engine floors at one lot rather
> than reporting costs and PnL for a position nobody can hold.

### Costs

Charged on both legs of every trade, computed against the **realised position
notional**:

```python
positions = entry_price × units
costs = cost_model.round_trip_cost(positions)
```

---

## Cost model

`costs.py`

| Component | Default | Basis |
|---|---|---|
| `brokerage_per_order` | ₹20 | Flat per order — relatively worse for small positions |
| `stt_sell_bps` | 2.00 | Securities Transaction Tax, sell leg only |
| `exchange_txn_bps` | 0.19 | NSE index futures |
| `sebi_bps` | 0.01 | SEBI turnover fee |
| `stamp_duty_buy_bps` | 0.20 | Buy leg only |
| `gst_rate` | 0.18 | On brokerage + exchange charges |
| `slippage_bps` | 0.50 | Per side, so twice per round trip |
| `lot_size` | 75 | Used to convert flat brokerage into a rate |

Giving a round trip of **~3.90 bps** at ₹2,000,000 notional.

### Everything is a parameter

Rates change with each Union Budget and are set per exchange. Nothing is
hard-coded: the structure is fixed, the numbers are configuration. **Verify them
against the current schedule before drawing conclusions.**

### Slippage is usually the largest real cost

And the one most often underestimated. It is modelled per side, so a round trip
pays it twice. The defaults assume 0.5 bps per side; in a fast market it can be
several times that.

`CostSchedule` exists for modelling historical regimes across long backtests.

### The hurdle

```python
model.breakeven_move_bps(notional)   # move required just to cover costs
```

This is the strategy's true hurdle, and it feeds into labelling, the signal gate,
and the UI. See [Scalping economics](scalping-economics.md).

---

## Running a backtest

```bash
uv run niftypulse backtest --strategy ensemble --horizon 5
uv run niftypulse backtest --strategy ml --horizon 5 --threshold 0.15
uv run niftypulse backtest --strategy vwap_reversion --sweep
```

```python
from niftypulse.backtest import BacktestConfig, CostModel, run_backtest

config = BacktestConfig(
    horizon=5,
    entry_threshold=0.15,
    notional=2_000_000,
    lot_size=75,
    cost_model=CostModel(slippage_bps=0.5),
)
result = run_backtest(bars, scores, config, strategy="ensemble")
print(result.report.lines())
```

### The conviction contract

The engine thresholds on **signed conviction in `[-1, 1]`**, not on a
probability.

> **This is a real trap.** Feeding probabilities instead means `abs(0.48)` — a
> model with no opinion — clears every threshold below 0.48, so the entry filter
> silently stops filtering and every bar trades. `_resolve_scores` in the CLI maps
> model output to `2p − 1` before handing it over, with no opinion at exactly
> zero. `test_probability_series_is_not_valid_conviction` guards this.

### `--strategy ml`

Uses the saved walk-forward predictions, not fresh inference:

```
artifacts/oof_5m.parquet
```

A model refit on all the data has already seen the bars it would be judged on.
Replaying the out-of-sample predictions is the only way to get an honest number.

---

## Reporting

`report.py`

```
trades               910 (475 long / 435 short)
avg position         Rs 1,740,879
win rate             46.92%
gross per trade      +0.02 bps
costs per trade      3.94 bps
net per trade        -3.92 bps
expectancy           -3.92 bps
profit factor        0.359
avg win / avg loss   +4.68 / -11.53 bps
net PnL              Rs -622,153  (costs Rs 624,100)
Sharpe (annualised)  -26.090
Sortino (annualised) -18.906
max drawdown         -3563.7 bps
time in market       10.1%
```

### Gross and net are always both shown

A strategy with positive gross expectancy and negative net expectancy is the
**normal outcome** on intraday data. A report that only showed net PnL would hide
why.

In the example above, costs (₹624,100) exceed the net loss (₹622,153). The
strategy is not losing money because it is wrong; it is losing money because it is
paying to be roughly right.

### Ratio statistics are annualised by trade frequency

Sharpe and Sortino use per-trade returns, annualised by trades-per-year derived
from the actual span. For a trade-based strategy this is more meaningful than
resampling a mostly-flat equity curve, which would understate risk by counting
the many bars with no position as zero-return periods.

### Drawdown is in basis points

Because position size is a parameter, rupee drawdown is not comparable across
runs. Basis points are.

### Empty backtests

If no trades fire, `build_report` returns a fully-populated zero report rather
than raising. A strategy with no signals is a legitimate result.

---

## Threshold sweeps

```bash
uv run niftypulse backtest --strategy ml --horizon 5 --sweep
```

Runs the same backtest across entry thresholds from 0.05 to 0.50.

```
threshold  trades  win_rate  gross_bps  net_bps
0.05         1585    0.3211      0.044   -3.897
0.10         1383    0.3174     -0.148   -4.089
0.15         1086    0.3306     -0.137   -4.077
0.20          808    0.3193     -0.296   -4.236
0.30          334    0.3084     -0.564   -4.503
```

**How to read it:**

Trade count falls monotonically — the filter is working. But net expectancy gets
*worse*. Being more selective does not help.

That is the diagnostic. **If expectancy does not improve as the threshold rises,
the signal carries no information about its own confidence.** No amount of
threshold tuning will rescue it.

Conversely, a signal that *does* improve with threshold is one where the model
knows when it is right, which is the actual prerequisite for profitable trading.

`breakeven_threshold()` finds the smallest threshold with positive expectancy, if
one exists.

---

## Worked example

From the bundled synthetic data, `--strategy ml --horizon 5`:

| Metric | Value | Interpretation |
|---|---|---|
| Trades | 910 | ~2% of bars — the cost gate is filtering hard |
| Win rate | 46.9% | Better than the pre-gate 20.5%, because sub-cost trades are excluded |
| Gross | +0.02 bps | Effectively zero. No edge |
| Costs | 3.94 bps | The hurdle, as designed |
| Net | −3.92 bps | Costs dominate completely |

The honest conclusion: **on this data, at this cost assumption, there is no
tradeable edge.** The model is not wrong — it is being asked a question the data
does not answer.

That is the expected result for synthetic data, and it is a reasonable prior for
real intraday NIFTY at 1–5 minute horizons.

---

## Extending

### A new metric

Add to `PerformanceReport` as a field, compute it in `build_report`, and add it to
`lines()` and `as_row()`. Keep the existing convention: express per-trade figures
in basis points and keep gross and net separate.

### A new cost component

Add a field to `CostModel` with a documented default, include it in `per_leg_bps`
on the correct leg (buy, sell, or both), and note whether GST applies.

### A different execution model

`run_backtest` is deliberately a plain loop for clarity. Realistic extensions:

- **Limit orders** — fill only if price trades through the level, and model
  non-fills. This usually reduces reported edge substantially and is worth having.
- **Partial fills** — relevant above a certain size.
- **Market impact** — for sizes where your own order moves the price.
- **Intrabar stops** — needs tick or higher-resolution data to be honest; with
  1-minute bars any assumption about ordering within the bar is a guess.

If you add intrabar logic, be explicit about the assumption. Assuming the
favourable path within a bar is one of the most common ways a backtest becomes
fiction.
