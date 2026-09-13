# Scalping economics

This is the most important document in the project. Read it before tuning
anything, because most changes that look like improvements are not.

---

## The premise

Scalping is a cost game before it is a prediction game.

A strategy's edge is not its accuracy. It is:

```
edge = (probability of being right × average win) − (probability of being wrong × average loss) − costs
```

On a scalping timescale the cost term is not a detail. It is frequently larger
than the entire edge, and it is the one component you cannot improve by modelling
better.

For NIFTY index futures, a round trip is roughly **3.9 basis points**:

| Component | Basis points | Notes |
|---|---|---|
| Slippage, per side | 0.50 | Two legs |
| Brokerage | ~0.10 at ₹2M | Flat ₹20 per order, so relatively worse when small |
| STT (sell leg) | 2.00 | The largest single component |
| Exchange transaction | 0.19 | NSE index futures |
| Stamp duty (buy leg) | 0.20 | |
| SEBI turnover | 0.01 | |
| GST | 18% on brokerage + exchange | |
| **Round trip** | **~3.90** | at ₹2,000,000 notional, 75-unit lot |

Run `uv run niftypulse backtest` to see the hurdle computed for your own
assumptions.

---

## The measurement that matters

Given a 3.90 bp hurdle, how often does NIFTY actually move that far in the time
you intend to hold?

```
Cost hurdle — round trip 3.90 bps at Rs 2,000,000
┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ horizon ┃ median move ┃     p75 ┃     p90 ┃ clears hurdle ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│      1m │     1.30 bp │ 2.25 bp │ 3.31 bp │          5.7% │
│      2m │     1.80 bp │ 3.13 bp │ 4.65 bp │         16.0% │
│      3m │     2.20 bp │ 3.85 bp │ 5.72 bp │         24.3% │
│      5m │     2.87 bp │ 4.99 bp │ 7.35 bp │         36.3% │
└─────────┴─────────────┴─────────┴─────────┴───────────────┘
```

Read the 1-minute row carefully. The **median** move is 1.30 bp. Even the **90th
percentile** move is 3.31 bp — still below the hurdle.

So on a 1-minute horizon, fewer than **6% of bars** produce a move large enough to
pay for the trade. That is the ceiling on how often a 1-minute scalp can profit.
It is set by market structure and transaction costs, not by the quality of your
model. A perfect forecaster, right every single time, would still be limited to
those 6% of bars — and would need to be right about *which* ones.

This is the single most useful number in the platform, and it is why
`uv run niftypulse train` prints it before the model metrics.

---

## How the platform acts on it

### 1. Labelling requires a tradeable move

`ml/dataset.py`

```python
threshold = max(deadband_atr × ATR, labelling_hurdle_bps / 10_000)
```

The ATR-scaled band sets a baseline; the cost hurdle raises it where costs bind.
With `hurdle_multiple: 1.5`, labels require a move above **5.86 bp** — comfortably
past break-even, because a trade that merely breaks even is not worth taking.

Why this matters: without it, the model spends its entire capacity learning to
classify moves of 0.3 bp. Such a model can report 55% accuracy, look excellent in
every summary statistic, and lose money on every correct call. That is the
default failure mode of naive scalping research.

### 2. Horizons that cannot work are refused

If too few bars clear the hurdle, `train` **skips the horizon and explains why**:

```
Training horizons 1m, 2m, 3m, 5m...
  skipped 1m — 1m: only 387 of the bars have a forward move above the 5.86 bp
  labelling hurdle (4,000 needed).
    This horizon is untradeable at the current cost assumption — the moves are
    too small to pay for the round trip. Options:
      - train a longer horizon, where moves are larger
      - lower model.hurdle_multiple if you want the raw signal without a cost margin
      - set model.enforce_cost_hurdle false to study signal alone, remembering the
        backtest will still charge costs
```

Producing a model here would be worse than useless, because it would create the
impression of a working 1-minute scalper.

### 3. Signals carry a cost gate

`plugins/strategies/catalog.py`

The ML strategy stays silent unless the expected move clears the hurdle. Most
bars do not. That is the point.

The expected move is not a guess. It comes from a **conviction-to-move curve**
fitted on out-of-sample predictions (`ml/metrics.py:expected_move_curve`), which
buckets predictions by confidence and measures how far price actually travelled.
An earlier version used `edge × ATR × √horizon`, which has no basis in the data
and happily promised moves that never materialised.

### 4. The UI leads with the cost decision

```
╭────── scalp forecasts — does the move pay for the trade? ───────╮
│      h  view      P(up)  distribution      exp move    vs cost  │
│     3m  ▼D        0.461  ██████░░░░░░        -8.0bp     +4.1bp  │
│     5m  ▲U        0.537  ██████░░░░░░        +8.5bp     +4.6bp  │
│ round trip  3.90bp  tradeable  5m UP (+4.6bp)                   │
╰─────────────────────────────────────────────────────────────────╯
```

`vs cost` is the difference that decides whether to act. A `+` value is green. A
directionally correct forecast of a sub-cost move is a losing trade, so the
direction column is deliberately secondary.

---

## Consequences worth internalising

### Higher accuracy is not the goal

Because costs are fixed and the move distribution is fixed, the binding constraint
is **selectivity**, not accuracy. The platform's threshold sweep makes this
visible:

```
threshold  trades  win_rate  gross_bps  net_bps
0.05         1585    0.3211      0.044   -3.897
0.10         1383    0.3174     -0.148   -4.089
0.15         1086    0.3306     -0.137   -4.077
0.20          808    0.3193     -0.296   -4.236
0.30          334    0.3084     -0.564   -4.503
```

Trade count falls as the threshold rises — the filter works. But net expectancy
gets *worse*. Being more selective does not help, because the model's confidence
carries no information about which bars are tradeable. That is the diagnostic:
**if expectancy does not improve as the threshold rises, the signal says nothing
about its own confidence.**

### Longer horizons are structurally easier

Not because they are harder to predict — the 5m base rate (0.491) is no further
from 0.5 than the 1m base rate. Because the moves are bigger relative to a fixed
cost. A 36% tradeable share at 5m versus 5.7% at 1m is a sixfold difference in
how often the exercise is even possible.

This is counterintuitive if you think of scalping as "shorter is better". Past a
point, shorter is just more expensive.

### Costs are the highest-leverage parameter

`--slippage` and `--notional` in `config/default.yaml` change conclusions more
than any modelling choice. A one basis point increase in slippage — entirely
plausible in fast markets — moves the hurdle by 25% and can eliminate a marginal
edge outright.

Before trusting any result, sanity-check the cost inputs. If you have real fill
data, use it. Do not trust the defaults.

---

## Working in a low-signal regime

Practical guidance, in rough order of leverage:

1. **Get the costs right first.** Everything downstream is conditional on them.
2. **Trade the future, not the index.** NIFTY 50 publishes no volume and no order
   book. The front-month future has depth, traded value, and open interest. Most
   short-timescale edge lives in order flow, which the index simply does not
   expose (#17 and #13 in [Strategies](strategies.md) currently return no
   opinion for exactly this reason).
3. **Prefer longer scalping horizons.** 3m and 5m over 1m.
4. **Expect no edge.** Markets are close to efficient at this timescale. A model
   finding nothing is the common, honest outcome.
5. **Distrust good results.** A 55% accurate next-candle model is far more likely
   to be a leak than a discovery. [Development](development.md) describes the
   controls this project uses to tell the difference.

---

## Related

- [ML pipeline](ml-pipeline.md) — how the hurdle enters labelling and validation
- [Backtesting](backtesting.md) — the cost model in full, and how execution is simulated
- [Configuration](configuration.md) — every cost parameter
- [Strategies](strategies.md) — the cost-aware ML strategy
