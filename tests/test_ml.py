"""Tests for the ML pipeline.

Two controls matter here.

The **positive control** feeds the pipeline data with an obvious, deterministic
pattern. If the models cannot learn that, the pipeline is broken — and no amount
of good-looking metrics on real data would be trustworthy.

The **negative control** feeds it a pure random walk. If it reports a meaningful
edge there, something is leaking. Together they bracket the pipeline: able to
find real structure, unable to invent fake structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.settings import Settings
from features import build_features, feature_columns
from ml.calibration import ProbabilityCalibrator, expected_calibration_error
from ml.dataset import (
    build_dataset,
    directional_labels,
    sample_weights,
    tradeable_fraction,
)
from ml.metrics import edge_by_confidence, evaluate, expected_move_curve
from ml.models import DirectionEnsemble, build_models
from ml.trainer import Trainer


@pytest.fixture
def fast_settings(tmp_path) -> Settings:
    """Settings for testing training mechanics quickly.

    The cost hurdle is disabled here on purpose. These tests are about whether
    the pipeline works — labelling, validation, calibration, persistence — and
    the synthetic data they use produces moves far smaller than a realistic
    round trip, so the hurdle would reject nearly every bar and the tests would
    be measuring the hurdle rather than the machinery. Hurdle behaviour has its
    own tests below.
    """
    return Settings(
        horizons=(5,),
        n_splits=3,
        embargo_bars=20,
        min_train_bars=800,
        enforce_cost_hurdle=False,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
    )


def make_bars(close: np.ndarray, span_hours: float = 6.0) -> pd.DataFrame:
    """Wrap a close series in session-aligned OHLCV bars."""
    minutes = (span_hours * 60) / len(close)
    index = pd.date_range(
        "2026-01-05 09:15",
        periods=len(close),
        freq=pd.Timedelta(minutes=minutes),
        tz="Asia/Kolkata",
    )
    frame = pd.DataFrame({"close": close}, index=index)
    frame["open"] = np.concatenate([[close[0]], close[:-1]])
    frame["high"] = np.maximum(frame["open"], frame["close"]) + 0.5
    frame["low"] = np.minimum(frame["open"], frame["close"]) - 0.5
    frame["volume"] = 0.0
    frame["oi"] = 0.0
    return frame


# ------------------------------------------------------------ positive control


def test_pipeline_learns_deterministic_pattern() -> None:
    """Alternating drift runs must be learned to a high AUC.

    The series moves up 2.5 points per bar for 40 bars, then down for 40, and
    repeats. Direction over the next 5 bars is therefore almost perfectly
    determined by the recent past, and a working pipeline should find it.
    """
    n = 4000
    direction = np.resize(np.repeat([1.0, -1.0], 40), n)
    close = 24_000.0 + np.cumsum(direction * 2.5)
    bars = make_bars(close)

    features = build_features(bars)
    columns = feature_columns(features)
    dataset = build_dataset(features, bars, horizon=5, feature_names=columns, deadband=0.12)

    assert len(dataset) > 1500

    # Single split for speed and clarity.
    cut = int(len(dataset) * 0.6)
    train = dataset.X.iloc[:cut]
    train_y = dataset.y.iloc[:cut]
    test = dataset.X.iloc[cut:]
    test_y = dataset.y.iloc[cut:].to_numpy()

    model = DirectionEnsemble(build_models(random_state=0))
    subset = type(dataset)(
        X=train,
        y=train_y,
        forward_return=dataset.forward_return.iloc[:cut],
        horizon=5,
        feature_names=columns,
    )
    model.fit(train, train_y, sample_weight=sample_weights(subset))
    probabilities = model.predict_proba(test)

    report = evaluate(test_y, probabilities)
    assert report.auc > 0.80, f"pipeline failed to learn a deterministic pattern (AUC {report.auc:.3f})"
    assert report.accuracy > 0.70
    assert report.accuracy_lift > 0.15


# ------------------------------------------------------------ negative control


def test_pipeline_finds_no_edge_in_random_walk() -> None:
    """A driftless random walk must not yield a large apparent edge.

    This is the leakage tripwire. If features or labels contained future
    information, this test would show a spuriously high AUC.
    """
    rng = np.random.default_rng(99)
    n = 6000
    steps = rng.standard_normal(n) * 0.00008
    close = 24_000.0 * np.exp(np.cumsum(steps))
    bars = make_bars(close)

    features = build_features(bars)
    columns = feature_columns(features)
    dataset = build_dataset(features, bars, horizon=5, feature_names=columns, deadband=0.12)
    assert len(dataset) > 2000

    cut = int(len(dataset) * 0.6)
    train, train_y = dataset.X.iloc[:cut], dataset.y.iloc[:cut]
    test, test_y = dataset.X.iloc[cut:], dataset.y.iloc[cut:].to_numpy()

    model = DirectionEnsemble(build_models(random_state=0))
    subset = type(dataset)(
        X=train,
        y=train_y,
        forward_return=dataset.forward_return.iloc[:cut],
        horizon=5,
        feature_names=columns,
    )
    model.fit(train, train_y, sample_weight=sample_weights(subset))
    probabilities = model.predict_proba(test)

    report = evaluate(test_y, probabilities)
    assert report.auc < 0.60, f"spurious edge on a random walk (AUC {report.auc:.3f}) suggests leakage"


# ---------------------------------------------------------------- labelling


def test_deadband_drops_ambiguous_labels() -> None:
    """Moves inside the dead band must be dropped, not forced into a class."""
    rng = np.random.default_rng(3)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.4)
    bars = make_bars(close)
    features = build_features(bars)

    wide = directional_labels(bars, 5, features["atr_norm"], deadband=0.5)[0]
    narrow = directional_labels(bars, 5, features["atr_norm"], deadband=0.01)[0]

    assert wide.notna().sum() < narrow.notna().sum(), "a wider dead band must drop more labels"
    assert wide.notna().sum() > 0


def test_labels_are_balanced_by_weighting() -> None:
    rng = np.random.default_rng(11)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.4) + np.arange(n) * 0.05
    bars = make_bars(close)
    features = build_features(bars)
    dataset = build_dataset(features, bars, 5, feature_columns(features))

    weights = sample_weights(dataset)
    assert len(weights) == len(dataset)

    # Weighted class mass should be close to even after balancing.
    up = weights[dataset.y.to_numpy() == 1].sum()
    down = weights[dataset.y.to_numpy() == 0].sum()
    assert up == pytest.approx(down, rel=0.05)


# ---------------------------------------------------------------- metrics


def test_evaluate_perfect_predictions() -> None:
    labels = np.array([1, 1, 0, 0, 1, 0] * 30)
    probabilities = np.where(labels == 1, 0.95, 0.05)
    report = evaluate(labels, probabilities)
    assert report.accuracy == pytest.approx(1.0)
    assert report.auc == pytest.approx(1.0)
    assert report.mcc == pytest.approx(1.0)


def test_accuracy_lift_detects_majority_classifier() -> None:
    """Always predicting UP must show zero lift, not a flattering accuracy."""
    labels = np.array([1] * 80 + [0] * 20)
    probabilities = np.full(len(labels), 0.9)
    report = evaluate(labels, probabilities)
    assert report.accuracy == pytest.approx(0.8)
    assert report.accuracy_lift == pytest.approx(0.0, abs=1e-9)


def test_edge_by_confidence_buckets() -> None:
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, 2000)
    probabilities = rng.uniform(0, 1, 2000)
    table = edge_by_confidence(labels, probabilities, bins=4)
    assert not table.empty
    assert table["samples"].sum() == 2000


# ----------------------------------------------------------- calibration


def test_calibrator_identity_on_small_samples() -> None:
    """With too little data the calibrator must pass probabilities through."""
    calibrator = ProbabilityCalibrator(method="platt", min_samples=5000)
    probabilities = np.array([0.2, 0.5, 0.8])
    calibrator.fit(probabilities, np.array([0, 1, 1]))
    assert not calibrator.is_fitted
    assert calibrator.transform(probabilities) == pytest.approx(probabilities)


def test_calibrator_accepts_genuine_overconfidence() -> None:
    """A model that is real but too extreme must be shrunk, not discarded.

    Scores here carry genuine information — truth is drawn from a probability
    that moves with the score — but the model claims far more range than it has.
    Calibration should pull the spread in while keeping the ranking, and must be
    accepted rather than mistaken for a collapse. This is the case that
    distinguishes a working gate from one that simply rejects everything.
    """
    rng = np.random.default_rng(11)
    n = 40_000
    raw = rng.uniform(0.05, 0.95, n)
    true_probability = 0.5 + 0.33 * (raw - 0.5)  # roughly 3x overconfident
    truth = (rng.uniform(0, 1, n) < true_probability).astype(int)

    calibrator = ProbabilityCalibrator(method="platt", min_samples=500).fit(raw, truth)
    assert calibrator.is_fitted, calibrator.reason

    calibrated = calibrator.transform(raw)
    assert np.std(calibrated) < np.std(raw), "should shrink an overconfident model"
    assert calibrator.validation["spread_ratio"] > calibrator.min_spread_ratio

    before = expected_calibration_error(raw, truth)
    after = expected_calibration_error(calibrated, truth)
    assert after < before, "calibration should reduce the reliability gap"


def test_calibrator_rejects_noise_level_gain() -> None:
    """A gain indistinguishable from noise must not enable calibration.

    Input here is already calibrated by construction, so the best any monotone
    map can do is break even. Measured on this data a one-parameter fit moves the
    holdout Brier by roughly 1e-5 in either direction — accepting that would make
    the gate a coin flip, so calibration must stay off.
    """
    rng = np.random.default_rng(2)
    n = 40_000
    raw = rng.uniform(0.03, 0.97, n)
    truth = (rng.uniform(0, 1, n) < raw).astype(int)

    calibrator = ProbabilityCalibrator(method="platt", min_samples=500).fit(raw, truth)
    assert not calibrator.is_fitted
    assert "rejected" in calibrator.reason
    # Pass-through leaves the ranking information completely untouched.
    assert calibrator.transform(raw) == pytest.approx(raw)
    assert calibrator.validation["improvement"] < calibrator.validation["required"]


def test_platt_preserves_ranking() -> None:
    """Calibration must not collapse distinct scores onto one value.

    Isotonic regression can map a wide range of inputs to a single output when
    the signal is weak, which destroys the ordering the strategy layer relies on.
    Platt scaling is strictly monotone, so ordering always survives.
    """
    rng = np.random.default_rng(12)
    n = 30_000
    raw = rng.uniform(0.02, 0.98, n)
    truth = (rng.uniform(0, 1, n) < raw).astype(int)  # raw is already well calibrated

    calibrator = ProbabilityCalibrator(method="platt", min_samples=500).fit(raw, truth)
    calibrated = calibrator.transform(raw)

    # Distinct inputs must not be flattened onto a single output.
    assert len(np.unique(np.round(calibrated, 6))) > n * 0.5

    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= -1e-9), "calibration must be monotone"


def test_calibrator_rejects_spread_collapse() -> None:
    """Calibration must not flatten the model's opinion into a constant.

    When the model has no real signal, mapping toward the base rate always
    improves the Brier score — the honest probability genuinely is the base rate
    — while shrinking the spread to nearly nothing. That would trade away the
    score spread a threshold-based strategy needs in exchange for a scoreboard
    win, so it has to be rejected.
    """
    rng = np.random.default_rng(8)
    n = 30_000
    # Wide raw spread, labels unrelated to it: the collapse scenario.
    raw = rng.uniform(0.20, 0.80, n)
    truth = rng.integers(0, 2, n)

    calibrator = ProbabilityCalibrator(method="platt", min_samples=500).fit(raw, truth)
    assert not calibrator.is_fitted
    assert "collapse" in calibrator.reason
    assert calibrator.validation["spread_ratio"] < calibrator.min_spread_ratio
    # Rejection means the raw scores survive untouched.
    assert calibrator.transform(raw) == pytest.approx(raw)


def test_calibrated_outputs_stay_in_range() -> None:
    """Even extreme inputs must map to a valid open interval, never 0 or 1."""
    rng = np.random.default_rng(11)
    n = 40_000
    raw = rng.uniform(0.05, 0.95, n)
    true_probability = 0.5 + 0.33 * (raw - 0.5)
    truth = (rng.uniform(0, 1, n) < true_probability).astype(int)
    calibrator = ProbabilityCalibrator(min_samples=200).fit(raw, truth)
    assert calibrator.is_fitted, calibrator.reason

    out = calibrator.transform(np.array([0.0, 0.001, 0.5, 0.999, 1.0]))
    assert ((out > 0) & (out < 1)).all()
    assert np.all(np.isfinite(out))


# ------------------------------------------------------ cost hurdle (scalping)


def test_cost_hurdle_raises_labelling_threshold() -> None:
    """Labelling must ignore moves too small to pay for the round trip.

    On a scalping timescale this is the difference between a usable model and an
    expensive one: a model trained on sub-cost moves can be consistently right
    and still lose money on every correct call.
    """
    rng = np.random.default_rng(23)
    n = 4000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.4)
    bars = make_bars(close)
    features = build_features(bars)
    atr_norm = features["atr_norm"]

    plain, _ = directional_labels(bars, 5, atr_norm, deadband=0.12, hurdle_bps=0.0)
    hurdled, _ = directional_labels(bars, 5, atr_norm, deadband=0.12, hurdle_bps=20.0)

    assert plain.notna().sum() > 0
    assert hurdled.notna().sum() < plain.notna().sum(), "a hurdle must remove labels"
    # Every surviving label must clear the hurdle on its own forward return.
    returns = directional_labels(bars, 5, atr_norm, deadband=0.12, hurdle_bps=20.0)[1]
    surviving = returns.loc[hurdled.dropna().index]
    assert (surviving.abs() > 20.0 / 10_000.0).all()


def test_hurdle_never_lowers_the_threshold() -> None:
    """The hurdle is a floor. A large ATR band must still win when it is bigger."""
    rng = np.random.default_rng(29)
    n = 2000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.4)
    bars = make_bars(close)
    atr_norm = build_features(bars)["atr_norm"]

    # A tiny hurdle must not make labelling stricter than the ATR band alone.
    without, _ = directional_labels(bars, 5, atr_norm, deadband=0.5, hurdle_bps=0.0)
    with_tiny, _ = directional_labels(bars, 5, atr_norm, deadband=0.5, hurdle_bps=0.01)
    assert with_tiny.notna().sum() == without.notna().sum()


def test_tradeable_fraction_reports_the_ceiling() -> None:
    """The share of bars clearing the hurdle is the hard ceiling on scalping."""
    rng = np.random.default_rng(31)
    n = 5000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    bars = make_bars(close)

    low = tradeable_fraction(bars, 5, hurdle_bps=0.1)
    high = tradeable_fraction(bars, 5, hurdle_bps=50.0)

    assert 0.0 <= high["tradeable_share"] <= low["tradeable_share"] <= 1.0
    assert high["tradeable_share"] < 0.05
    assert high["p90_move_bps"] >= high["median_move_bps"]


def test_hurdle_makes_short_horizons_untrainable() -> None:
    """A horizon whose moves cannot cover costs must be skipped, not trained.

    Skipping is the correct outcome, not an error: it reports that the timescale
    is structually unprofitable rather than producing a model that will lose
    money reliably.
    """
    rng = np.random.default_rng(37)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.05)  # very quiet
    bars = make_bars(close)
    features = build_features(bars)
    columns = feature_columns(features)

    settings = Settings(
        horizons=(1,),
        n_splits=2,
        min_train_bars=500,
        enforce_cost_hurdle=True,
        hurdle_multiple=1.5,
        data_dir=Path("/tmp"),
        model_dir=Path("/tmp"),
        log_dir=Path("/tmp"),
    )
    trainer = Trainer(settings, quiet=True, n_jobs=1)
    results = trainer.train_all(features, bars, columns)

    assert results == {}, "a sub-cost horizon must not produce a model"
    assert 1 in trainer.skipped
    assert "untradeable" in trainer.skipped[1]


def test_expected_move_curve_is_monotone() -> None:
    """Conviction must map to a non-decreasing expected move."""
    rng = np.random.default_rng(41)
    n = 20_000
    probabilities = rng.uniform(0.3, 0.7, n)
    # Bigger conviction -> bigger realised move, with noise.
    forward = pd.Series(
        rng.standard_normal(n) * 0.0002 * (1.0 + 8.0 * np.abs(probabilities - 0.5))
    )

    curve = expected_move_curve(forward, probabilities, bins=6)
    moves = curve["moves"]
    assert len(moves) >= 3
    assert moves == sorted(moves), "expected move must not decrease with conviction"

    from ml.metrics import apply_expected_move_curve

    assert apply_expected_move_curve(0.0, curve) <= apply_expected_move_curve(0.4, curve)


def test_expected_move_curve_falls_back_when_sparse() -> None:
    """Too little data must fall back to a flat estimate, not invent a curve."""
    probabilities = np.array([0.51, 0.52, 0.49, 0.50])
    forward = pd.Series([0.0001, -0.0002, 0.0001, -0.0001])
    curve = expected_move_curve(forward, probabilities, bins=8, min_per_bin=100)
    assert curve["edges"] == []
    assert curve["mean_move_bps"] > 0


# ---------------------------------------------------------------- training


def test_trainer_produces_out_of_sample_predictions(fast_settings) -> None:
    """Every out-of-sample row must come from a model that had not seen it."""
    rng = np.random.default_rng(21)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    bars = make_bars(close)
    features = build_features(bars)
    columns = feature_columns(features)

    trainer = Trainer(fast_settings, quiet=True, n_jobs=1)
    result = trainer.train_horizon(features, bars, 5, columns)

    assert len(result.oof) > 0
    assert result.oof["probability"].between(0, 1).all()
    assert result.oof["label"].isin([0, 1]).all()
    # OOF coverage should be a large fraction of the dataset, minus purged rows.
    assert len(result.oof) > len(result.dataset) * 0.4
    assert result.report.n == len(result.oof)


def test_trainer_saves_and_reloads(fast_settings) -> None:
    rng = np.random.default_rng(31)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    bars = make_bars(close)
    features = build_features(bars)
    columns = feature_columns(features)

    trainer = Trainer(fast_settings, quiet=True, n_jobs=1)
    results = trainer.train_all(features, bars, columns)
    paths = trainer.save(results)

    assert paths
    assert trainer.artifact_path(5).exists()
    assert trainer.oof_path(5).exists()

    from ml.predictor import Predictor

    predictor = Predictor(fast_settings).load()
    assert predictor.is_ready

    predictions = predictor.predict(features)
    assert len(predictions) == 1
    assert 0.0 <= predictions[0].p_up <= 1.0
    assert predictions[0].horizon_min == 5


def test_predictor_frame_is_vectorised(fast_settings) -> None:
    rng = np.random.default_rng(41)
    n = 3000
    close = 24_000.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    bars = make_bars(close)
    features = build_features(bars)
    columns = feature_columns(features)

    trainer = Trainer(fast_settings, quiet=True, n_jobs=1)
    results = trainer.train_all(features, bars, columns)
    trainer.save(results)

    from ml.predictor import Predictor

    predictor = Predictor(fast_settings).load()
    series = predictor.predict_frame(features, 5)
    assert len(series) == len(features)
    assert series.between(0, 1).all()
