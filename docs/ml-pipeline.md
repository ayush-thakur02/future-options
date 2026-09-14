# ML pipeline

Turning 45,000 bars into a calibrated probability that the next few minutes go up
— without fooling yourself in the process.

```
src/plugins/forecasts/ml_ensemble/
├── dataset.py       Dead-banded, hurdle-aware labelling
├── splits.py        Purged, embargoed walk-forward validation
├── models.py        6 base learners + configurable soft-voting ensemble
├── calibration.py   Twice-gated Platt / isotonic calibration
├── metrics.py       Lift-aware evaluation, expected-move curve
├── trainer.py       Orchestration and persistence
└── predictor.py     Inference
```

---

## The problem

Forecasting the sign of the next 5-minute return is close to unlearnable. The
signal is tiny relative to the noise, and a model that tries will spend its
capacity fitting microstructure noise that transaction costs would never let you
trade.

Worse, short-horizon financial ML is unusually good at producing results that
look excellent and are entirely fictitious. Overlapping labels leak between
training and test. Costs get applied at the end, after the conclusion is already
drawn. Probabilities turn out to be uncalibrated scores.

Each of the four sections below addresses one of those.

---

## 1. Labelling

`dataset.py`

### Dead-banding

A forward return smaller than the threshold is **dropped**, not forced into UP or
DOWN.

```python
threshold = max(deadband_atr × ATR, labelling_hurdle_bps / 10_000)
```

Two components:

- **`deadband_atr`** (default 0.12) — the ATR-scaled baseline
- **`labelling_hurdle_bps`** — the cost hurdle, `cost_hurdle_bps × hurdle_multiple`

The cost hurdle is a **floor**, so where costs bind it wins. On a scalping
timescale it almost always does. See [Scalping economics](scalping-economics.md)
for why this matters more than any modelling choice.

Without dead-banding, the model is asked to classify the ambiguous middle where
labelling is essentially arbitrary, and it learns that region's noise.

### Session masking

A label whose forward window crosses the overnight gap is **dropped**.

The close-to-open move is driven by global markets and news, a different
volatility regime entirely. Mixing those labels with intraday ones teaches the
model to forecast something that will never be traded.

### Result

```python
Dataset(
    X=...,                    # features, labelled rows only
    y=...,                    # 1 = up, 0 = down
    forward_return=...,       # retained in full, including dropped rows
    horizon=5,
    feature_names=[...],
)
```

`forward_return` is kept for **every** row, not just labelled ones, so the
backtest can compute realised PnL on the same bars — including those the model
was not asked about.

### Class weighting

`sample_weights` weights samples inversely to label frequency. Without it, a
dataset that is 54% UP produces a model that predicts UP almost always and looks
accurate while being useless.

---

## 2. Validation

`splits.py`

**Standard k-fold is invalid for financial labels.** Two distinct failures:

### Overlap

A label at bar *t* is computed from prices through bar *t + h*. If *t* is in the
training set and *t + h* falls inside the test set, the model has been shown the
answer. Accuracy inflates, and the effect is largest exactly where the horizon is
long relative to the fold size.

### Serial correlation

Even without direct label overlap, bars adjacent across a boundary share nearly
identical features. A model that memorises the regime around the split scores
well on both sides without learning anything general.

### The fix

```python
purged_walk_forward(n_samples, n_splits, horizon, embargo, min_train)
```

- **Expanding window.** Training always precedes testing in time, so each fold
  answers the question that matters: given everything up to here, what happens
  next?
- **Purge.** Drop training rows whose label window reaches into the test block:
  `purge_cut = test_start − horizon − embargo`
- **Embargo.** An additional 20 bars of margin beyond the purge.

The gap between the last training row and the first test row is therefore at
least `horizon + embargo`, which is asserted by a test.

Method follows López de Prado, *Advances in Financial Machine Learning*, ch. 7.

---

## 3. Models

`models.py`

Six base learners with genuinely different inductive biases, blended by **soft
voting**:

| Model | Why it is here |
|---|---|
| **LightGBM** | Usually the strongest single model on tabular data of this kind, and fast enough to retrain in a walk-forward loop |
| **HistGradientBoosting** | sklearn's boosting. Same family, different binning — correlated with LightGBM but not identical |
| **ExtraTrees** | Randomised bagging. Decorrelated from the boosted models because it fits deep, high-variance trees independently. **This is where most of the ensemble's diversity comes from** |
| **RandomForest** | Bootstrap aggregation with conventional split selection. Kept below ExtraTrees in the default vote because the two are correlated |
| **Shrinkage LDA** | A regularised generative covariance model: a smooth, non-tree boundary that tolerates numerous correlated indicators |
| **LogisticRegression** | A linear baseline. Included partly as a regulariser on the blend, and mostly because if the linear model is competitive with the boosted ones, the features probably are not carrying much signal — which is worth knowing |

The selected models, vote weights, and estimator hyperparameters live in
`config/plugins/forecast/ml_ensemble.yaml`. Unknown names fail before training;
LightGBM alone may be skipped when its native runtime is unavailable.

### Soft voting, not hard

Probabilities are averaged rather than votes counted, because the strategy layer
needs the confidence information. A 0.55 and a 0.95 both count as "UP" under hard
voting; only the probability carries how much the model actually believes it.

### Imputation

Non-tree models cannot consume NaN, so every pipeline imputes with a
train-fitted median.

**The imputer lives inside the pipeline and is refit per fold.** Imputing before
splitting would leak test-set statistics into training.

### Graceful degradation

```python
lightgbm_available()   # attempts an import and native load
```

On macOS, LightGBM's wheel links against OpenMP and importing it fails with a
`dlopen` error when `libomp` is missing. That is an install problem, not a code
problem, so the right response is to drop to the remaining five learners rather
than take the platform down.

### Feature importance

`DirectionEnsemble.feature_importance()` aggregates across models, normalised to
sum to 1. Tree models contribute `feature_importances_`; logistic regression and
shrinkage LDA contribute `|coef_|`.

### Disagreement

`DirectionEnsemble.disagreement()` returns the standard deviation of member
probabilities. High values mean the models split — visible in the dashboard's
prediction-quality card as agreement and member range.

---

## 4. Calibration

`calibration.py`

A boosted tree's "0.7" is a score, not a probability. Gradient boosting optimises
log loss and produces overconfident, often bimodal outputs, so a bar scored 0.7
may historically resolve upward only 54% of the time.

That matters directly here because the strategy sizes positions from probability.
An overconfident 0.7 that is really a 0.54 would be traded far too large.

### Fitted on out-of-fold predictions

The calibrator is fitted on the walk-forward **out-of-sample** predictions — never
on in-sample ones, which would calibrate against the model's own optimism.

### Platt is the default, not isotonic

Isotonic regression is non-parametric and can fit any monotone shape, which sounds
strictly better until you use it on weak signal. There it **collapses most inputs
to nearly the same output** — on this project's data it mapped 85% of predictions
into a single bucket at 0.50, destroying the ranking resolution downstream code
depends on.

Platt is a one-parameter logistic fit. It cannot collapse, it degrades gracefully,
and it strictly preserves ordering.

### Two gates

Calibration is applied **only if it clears both**:

**Gate 1 — meaningful Brier improvement.**

```python
improvement >= raw_brier × min_improvement_ratio   # 0.005
```

Testing on already-calibrated input shows a one-parameter fit still nudges the
holdout Brier by roughly 1e-5 in either direction — pure noise. Accepting *any*
improvement would accept noise half the time.

**Gate 2 — spread preservation.**

```python
calibrated_std / raw_std >= min_spread_ratio   # 0.10
```

A Brier improvement alone is not sufficient. When a model has no real signal,
mapping everything toward the base rate **always** improves Brier — the honest
probability genuinely is the base rate — while flattening the distribution to
near-constant. That looks like a scoreboard win while destroying the score spread
a threshold-based strategy needs.

Measured on synthetic data:

| Case | Shrinkage | Ratio |
|---|---|---|
| No signal at all | 138x | 0.007 |
| Genuinely informative, 2x overconfident | 1.9x | 0.51 |
| ... 3x overconfident | 3.0x | 0.34 |
| ... 5x overconfident | 5.0x | 0.21 |

A threshold of 0.10 sits well clear of both groups.

When either gate rejects, `transform()` is a pass-through and the model's raw
ranking survives untouched.

---

## 5. Metrics

`metrics.py`

**Accuracy is reported but is never the headline.** On a dataset with a 53% UP
base rate, predicting UP unconditionally scores 53%. Accuracy alone cannot
distinguish a model from a constant.

Every report includes:

| Metric | Why |
|---|---|
| **Lift** | Accuracy minus the majority-class baseline. The only number that separates a model from a constant |
| ROC AUC | Rank quality, threshold-free |
| Brier, log loss | Probability quality |
| MCC | Balanced single-number summary, robust to class imbalance |
| Per-class precision / recall | Which direction the model is actually good at |
| High-confidence accuracy and coverage | Performance in the regime the strategy actually trades |
| **Confidence buckets** | Accuracy by model confidence. **The single most useful diagnostic** |

### Confidence buckets

If accuracy does not rise with confidence, the probabilities carry no information
about *when* to trade — even if overall accuracy looks respectable.

```
        samples  accuracy  mean_confidence  mean_probability  realised_up_rate
bucket
Q1         2431    0.4850           0.0568            0.4989            0.4665
Q2         1724    0.4675           0.1699            0.4949            0.4727
Q3          731    0.4911           0.2840            0.5249            0.4856
Q4          253    0.4941           0.4007            0.5944            0.5217
Q5           53    0.6604           0.5132            0.6638            0.5849
```

Reading this: accuracy is flat across the first four buckets, so the model's
confidence is not discriminating. The Q5 headline (0.66) rests on 53 samples and
is not evidence of anything. **The flat column is the finding.**

### Expected-move curve

`expected_move_curve()` maps conviction to the realised size of the move it
preceded, by bucketing out-of-sample predictions and measuring how far price
actually travelled. Monotonicity is enforced with a cumulative maximum.

This is what makes the cost-hurdle check honest. A heuristic like
`edge × ATR × √horizon` has no basis in the data and will happily promise moves
that never materialise.

---

## Training

`trainer.py`

For each horizon:

1. Build a labelled dataset
2. Run purged walk-forward, collecting out-of-sample predictions from every fold
3. Fit the calibrator on those out-of-sample predictions
4. Fit the expected-move curve
5. Refit on the full history
6. Persist

### Parallelism

Folds train in parallel with joblib's **threading** backend. Threads rather than
processes because LightGBM and sklearn release the GIL during fitting, and threads
avoid pickling a 30,000-row feature frame once per worker.

Estimators are pinned to one thread each while folds run in parallel — leaving
both levels at `-1` oversubscribes the CPU and runs measurably slower than either
level alone.

### Refused horizons

```python
class InsufficientSamples(ValueError)
```

Raised when too few bars clear the labelling hurdle. This is a **finding, not a
failure**, and `train_all` catches it, records the reason, and continues with the
horizons that do work. Aborting the run would hide those.

### Artifacts

```
artifacts/direction_5m.joblib    model, calibrator, feature names, metrics,
                                 move curve, hurdle, importance
artifacts/oof_5m.parquet         walk-forward predictions
artifacts/training_report.json   summary of the last run
```

The OOF file exists so the backtester can replay genuinely out-of-sample
decisions. A model refit on all the data has already seen the bars it would be
judged on, so scoring it on its own training window would produce a misleadingly
good result.

### Configuration

| Setting | Default | Effect |
|---|---|---|
| `horizons` | `[1, 2, 3, 5]` | Forecast horizons in bars |
| `deadband_atr` | 0.12 | ATR-scaled labelling baseline |
| `enforce_cost_hurdle` | true | Require labels to clear costs |
| `hurdle_multiple` | 1.5 | Margin above break-even |
| `n_splits` | 4 | Walk-forward folds |
| `embargo_bars` | 20 | Extra bars dropped at each fold boundary |
| `min_train_bars` | 4000 | Floor for a trainable horizon |
| `train_n_jobs` | 4 | Folds trained in parallel |

---

## Inference

`predictor.py`

```python
predictor = Predictor(settings).load()
predictions = predictor.predict(features)      # list[Prediction], one per horizon
series = predictor.predict_frame(features, 5)  # vectorised, for backtests
```

`Prediction` carries:

| Field | Meaning |
|---|---|
| `p_up` | Calibrated probability |
| `direction` | UP / DOWN |
| `confidence` | `abs(p_up − 0.5) × 2` |
| `expected_move_bps` | From the fitted conviction curve |
| `hurdle_bps` | Round-trip cost at the reference notional |
| `clears_hurdle` | **Whether the move pays for the trade** |
| `edge_after_cost_bps` | Expected move minus hurdle |

It is deliberately thin — no state between calls beyond the loaded models — so
the same instance serves both backtest replay and live dashboard with no
behavioural drift.

---

## Verifying the pipeline

Two tests bracket the ML pipeline. Together they are what allow "found nothing"
to mean the same thing as "there was nothing to find".

### Positive control

`test_pipeline_learns_deterministic_pattern`

Injects an alternating-drift series: up 2.5 points per bar for 40 bars, then down
for 40. Direction over the next 5 bars is almost perfectly determined by the
recent past.

**Asserts AUC > 0.80 and accuracy > 0.70.** If the pipeline cannot learn this, it
is broken, and no result on real data would be trustworthy.

### Negative control

`test_pipeline_finds_no_edge_in_random_walk`

Feeds a driftless random walk. **Asserts AUC < 0.60.** This is the leakage
tripwire — if features or labels contained future information, this test would
show a spuriously high AUC.

### Causality

`test_features_do_not_use_future_data` truncates input and asserts features on
overlapping rows are bit-for-bit unchanged.

---

## Practical guidance

**Expect no edge.** Markets are close to efficient at this timescale. A model
finding nothing is the common, honest outcome.

**Distrust good results.** A 55% accurate next-candle model is far more likely to
be a leak than a discovery. Check the confidence buckets; if accuracy rises
monotonically with confidence *and* the positive control still passes, you may
have something.

**Feature importance is not causality.** The top features here are consistently
`ema_21_50_spread`, `realized_vol_60`, `hurst_100`, and time-of-day columns. That
is plausible — trend state and volatility regime — but importance only says the
model found the column useful, not that it means anything.

**Retrain after changing anything structural.** Feature count, `bar_minutes`,
horizon set, or the cost model all invalidate artifacts.
