# Features

147 columns, of which **141 are model-usable** against the bundled data. Every
column is tagged with a group so the training report can show where a model is
actually drawing its signal from.

```
src/plugins/features/technical/
├── indicators.py   classic indicators, pure pandas
├── advanced.py     adaptive trend, risk, entropy and liquidity mathematics
├── session.py      Session, calendar, prior-session, regime
├── pipeline.py     Matrix assembly and the FeaturePipeline object
└── plugin.py       The pack: provides "features"
```

The whole set is one plugin (`features:technical`, providing the `features`
capability) rather than one per indicator family. Features are assembled
together and consumed together, and splitting them into packs that must always be
used as a pair would be fake modularity — the real seam is the capability, and a
different feature set replaces the pack.

---

## Design rules

**Every function is causal.** The value at bar *t* uses bars up to and including
*t*. Short-window statistics are guarded so the first `n − 1` rows are `NaN`
rather than silently wrong.

This is enforced by a test rather than by discipline:
`test_features_do_not_use_future_data` truncates the input and asserts features on
the overlapping rows are unchanged. Any indicator that peeks forward fails it.

**No TA-Lib.** It needs a C toolchain and is the most common reason a Python
trading project fails to install. Implementing directly on pandas also makes the
smoothing conventions explicit — Wilder's for RSI, ATR, and ADX, recursive EMA
elsewhere — rather than hidden behind a library's defaults.

**Two deliberate exclusions.**

*Ichimoku's chikou span.* It is the close shifted backwards in time, so including
it hands the model the future. The other four Ichimoku components are used.

*Whole-session high and low.* Intraday range references use **running** cumulative
extremes within the session. Using the full-day maximum would tell a 09:30 bar
where the day's high ends up.

---

## Groups

| Group | Count | Purpose |
|---|---|---|
| [momentum](#momentum) | 16 | Recent returns and persistence |
| [trend](#trend) | 30 | Direction and trend quality |
| [oscillator](#oscillator) | 18 | Overbought/oversold state |
| [volatility](#volatility) | 19 | Range, bands, regime |
| [activity](#activity) | 4–7 | Volume, or tick count for an index |
| [candle](#candle) | 4–5 | Bar anatomy |
| [regime](#regime) | 2 | Statistical character of the series |
| [quantitative](#quantitative) | 24 | Adaptive filters, tail risk, dependence and liquidity |
| [context](#context) | 24 | Clock, calendar, prior session |

---

## momentum

*16 usable*

```
ret_1, ret_2, ret_3, ret_5, ret_10, ret_15, ret_30, ret_60, ret_z_10, ret_z_20,
roc_5, roc_10, roc_20, roc_60, direction_streak, up_bar_ratio_20
```

Log returns over eight horizons, plus their z-scores and rate-of-change
equivalents. `direction_streak` counts consecutive same-direction bars, signed.
`up_bar_ratio_20` is the share of up bars in the last 20.

Eight horizons rather than one because the useful timescale is unknown a priori,
and tree models handle redundant inputs better than a hand-picked wrong one.

---

## trend

*30 usable*

```
ema_dist_9, ema_dist_21, ema_dist_50, ema_dist_200, ema_9_21_spread,
ema_21_50_spread, ema_50_200_spread, ema_cross_9_21, ema_cross_21_50,
ema_cross_50_200, adx_14, plus_di, minus_di, di_spread, supertrend_dir,
supertrend_dist, aroon_up, aroon_down, aroon_osc, vortex_spread, slope_20,
slope_50, r2_20, efficiency_ratio_10, ichimoku_tenkan_dist, ichimoku_kijun_dist,
ichimoku_span_a_dist, ichimoku_span_b_dist, ichimoku_cloud_top,
ichimoku_cloud_bottom
```

EMA distances are expressed as `close / EMA − 1`, so they are scale-free and
comparable across price levels.

`di_spread` (plus DI minus minus DI) is often more informative than ADX alone —
ADX says how strong a trend is, the spread says which way.

`r2_20` is the R² of a rolling 20-bar regression, measuring how *linear* recent
price action is, distinguishing a clean trend from a choppy drift with the same
slope. `efficiency_ratio_10` is Kaufman's: net move over total path travelled.

**SuperTrend** bands ratchet in the trend direction and never loosen — the
standard formulation, implemented with an explicit loop in `indicators.py`
because it is path-dependent.

**Ichimoku** contributes four distance columns plus the cloud's upper and lower
edges. Chikou is excluded (see above).

---

## oscillator

*18 usable*

```
rsi_7, rsi_14, rsi_21, rsi_14_slope, rsi_overbought, rsi_oversold, macd_line,
macd_signal, macd_hist, macd_cross, stoch_k, stoch_d, stoch_cross, williams_r,
cci_20, ultimate_osc, mfi_14, cmf_20
```

RSI uses **Wilder's smoothing** (`ewm(alpha=1/n)`), not a simple EMA — the
distinction matters and is a common source of subtly wrong indicators.

MACD components are divided by price so they are comparable across levels.
`macd_cross` is `sign(histogram)`, giving the model a clean state variable.

---

## volatility

*19 usable*

```
atr_norm, atr_ratio, bb_pct_b, bb_bandwidth, bb_squeeze, keltner_position,
donchian_position, donchian_width, breakout_high_20, breakout_low_20,
realized_vol_10, realized_vol_20, realized_vol_60, parkinson_vol_20,
garman_klass_vol_20, vol_ratio_10_60, vol_ratio_20_60, return_skew_30,
return_kurt_30
```

`atr_norm` is ATR divided by price — the single most useful volatility feature,
and the basis for the labelling dead band.

`atr_ratio` compares ATR to its own 50-bar mean, so >1 means volatility is
expanding relative to its recent norm.

Three volatility estimators with different efficiency:

| Estimator | Uses | Notes |
|---|---|---|
| `realized_vol_*` | Close-to-close | Noisy but standard |
| `parkinson_vol_20` | High-low range | More efficient |
| `garman_klass_vol_20` | OHLC | Most efficient |

`bb_squeeze` is bandwidth relative to its own 100-bar mean: a value well below 1
means compression, which precedes expansion.

> **Breakout definition.** `breakout_high_20` is `close > prior 20-bar high`,
> where *prior* excludes the current bar. An earlier version compared against a
> channel that already contained the bar being tested — which requires the close
> to also be the bar's own high, and so almost never fires. The feature looked
> reasonable and was silently dead. `donchian()` now returns `prior_upper` and
> `prior_lower` explicitly for this reason.

---

## activity

*4–7 usable depending on the instrument*

```
activity_ratio_20, activity_z_20, activity_slope_20,
vwap_dist, vwap_above, obv_slope_20, obv_dist
```

**This group behaves differently for an index.** NIFTY 50 has no traded volume,
so `activity_proxy` falls back to `tick_count`, which is a genuine but coarser
signal. The group is registered as `activity(tick-proxy)` in that case, and
`activity(unavailable)` if neither volume nor tick count exists — in which case
the columns collapse to constants and `feature_columns()` drops them.

That naming is deliberate: the training report should not imply a model is using
volume when it is using tick count.

`vwap_dist` is distance from session-anchored VWAP, reset daily.
`obv_slope_20` is the rolling regression slope of On-Balance Volume.

---

## candle

*4–5 usable*

```
body_ratio, upper_wick_ratio, lower_wick_ratio, close_location, gap_from_prev_close
```

Each bar decomposed into proportions of its own range, so a hammer looks the same
at 24,000 as at 12,000. `close_location` is where the close sits within the range
— 0 at the low, 1 at the high.

---

## regime

*2 usable*

```
hurst_100, price_z_100
```

`hurst_100` is the rolling Hurst exponent, estimated by the variance-of-differences
method. H > 0.5 suggests trending, H < 0.5 mean-reverting.

> **Performance note.** This was originally implemented with `rolling.apply`, and
> the per-window Python callback ran 5,703 times per recompute — **71% of total
> feature time**. It is now vectorised across windows with
> `sliding_window_view`, verified against the reference to 2.2e-16, and roughly
> 7x faster. Feature computation went from ~360ms to ~50ms per recompute.

`price_z_100` is the z-score of price against its own 100-bar mean and standard
deviation, a scale-free measure of stretch.

---

## quantitative

*24 usable*

```
kama_dist_10, dema_dist_20, tema_dist_20, trix_15, ppo_line, ppo_signal,
ppo_hist, fisher_10, fisher_signal, rvi_10, rvi_signal, elder_bull_13,
elder_bear_13, choppiness_14, ulcer_14, downside_dev_30, sortino_30,
autocorr_1_50, variance_ratio_5_60, direction_entropy_50, mass_index_25,
ad_slope_20, ease_of_movement_14, amihud_20
```

KAMA adapts its smoothing speed to price efficiency; DEMA and TEMA reduce lag;
TRIX and PPO measure multi-scale momentum. Fisher, relative vigor and Elder
pressure describe where closes sit inside recent price structure.

The risk block separates volatility from harmful volatility: ulcer index tracks
drawdown depth, downside deviation and rolling Sortino isolate negative returns,
and mass index measures range expansion. Autocorrelation, variance ratio and
directional entropy distinguish persistence, mean reversion and random sign
sequences. Accumulation/distribution slope, ease of movement and Amihud
illiquidity add participation and price-impact context when activity exists.

Every implementation is causal and covered by prefix-invariance tests: computing
on a longer series cannot change a value already produced for an earlier bar.

---

## context

*24 usable*

```
session_progress, minutes_from_open, minutes_to_close, is_opening_30m,
is_closing_30m, is_midday_lull, time_sin, time_cos, day_of_week, is_monday,
is_friday, days_to_expiry, is_expiry_day, is_expiry_eve, bars_into_session,
overnight_gap, dist_from_day_high, dist_from_day_low, day_range_position,
dist_from_prior_high, dist_from_prior_low, prior_day_range, prior_day_return,
prior_day_body
```

Intraday index behaviour is strongly time-dependent. The first and last half hour
carry most of the day's range; midday is largely noise. Without clock features a
model will happily apply an opening-range pattern to a 13:00 bar.

`time_sin` / `time_cos` are a cyclical encoding of session progress, so 09:20 and
15:25 are not treated as far apart.

**All prior-session references are shifted by one day.** `dist_from_prior_high`
uses yesterday's high, not today's. `dist_from_day_high` and `dist_from_day_low`
use *running* cumulative extremes within the current session — a 09:30 bar does
not know where the day ends up.

`days_to_expiry` uses `expiry_weekday` from configuration. NSE moved NIFTY weekly
expiry from Thursday to Tuesday; the default is Tuesday (Python weekday 1), and
it is configurable because the exchange has changed this before.

---

## Going from bars to a matrix

```python
from plugins.features.technical import build_features, feature_columns

features = build_features(bars, bar_minutes=1, expiry_weekday=1)
columns = feature_columns(features)   # 141, numeric and non-constant on bundled data
```

`build_features` returns `inf` replaced by `NaN` — some ratios can blow up on a
zero-denominator bar, and `NaN` is a value LightGBM and the imputer handle
correctly whereas `inf` is not.

`feature_columns` excludes:

- `bar_minutes` and other metadata
- columns prefixed `_` (internal helpers)
- non-numeric columns
- **constant columns** — anything with one unique value carries no information
  and would only add noise to importance rankings

Columns are accumulated in a dict and concatenated once. Assigning 147 columns to
a DataFrame individually triggers pandas' fragmentation warning and is markedly
slower.

---

## Warm-up

Rolling windows mean the first ~200 rows are `NaN` for the longer indicators.
With 45,000 bars that is 0.4% and irrelevant. With a short history it is not —
`min_train_bars` guards against training on a dataset that is mostly warm-up.

The live engine keeps a 2,000-bar window, sized so EMA-200 has converged: after
2,000 bars the residual influence of the seed value is around 1e-8. A shorter
window introduces a train/serve skew that degrades every prediction quietly. See
[Architecture](architecture.md#features-are-computed-on-a-rolling-window-in-live-but-full-history-in-training).

---

## Adding a feature

1. Implement the indicator in `indicators.py` or `advanced.py`, causal and vectorised.
2. Add it in the appropriate `_add_*` function in `pipeline.py`, appending its
   name to that group's `columns` list.
3. If it opens a new group, call `_register(group, columns)`.
4. Run `uv run pytest tests/test_features.py` — the causality test will catch
   lookahead automatically.
5. Retrain. Feature count changes invalidate artifacts.
