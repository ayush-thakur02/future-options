"""Training orchestration.

For each forecast horizon the trainer:

1. Builds a dead-banded labelled dataset.
2. Runs purged, embargoed walk-forward validation, collecting out-of-sample
   predictions from every fold.
3. Fits the probability calibrator on those out-of-sample predictions — never on
   in-sample ones, which would calibrate against the model's own optimism.
4. Refits on the full history and persists the artifact.

The out-of-sample predictions are retained in the result so the backtester can
replay the model's actual historical decisions rather than re-predicting with a
model that has already seen the future.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.table import Table

from ..config import Settings
from ..trading_calendar import IST
from .calibration import ProbabilityCalibrator, expected_calibration_error
from .dataset import (
    Dataset,
    build_dataset,
    class_balance,
    sample_weights,
    tradeable_fraction,
)
from .metrics import ClassificationReport, edge_by_confidence, evaluate, expected_move_curve
from .models import DirectionEnsemble, build_models
from .splits import Fold, purged_walk_forward

console = Console()

ARTIFACT_VERSION = 1


class InsufficientSamples(ValueError):
    """A horizon cannot be trained because too few bars clear the cost hurdle.

    This is a finding, not a bug. On a scalping timescale most bars move less
    than the round trip needed to capture them, so a short horizon can be
    structurally untradeable no matter how much history is available.
    """

    def __init__(self, horizon: int, available: int, required: int, hurdle_bps: float) -> None:
        self.horizon = horizon
        self.available = available
        self.required = required
        self.hurdle_bps = hurdle_bps
        super().__init__(self.explain())

    def explain(self) -> str:
        return (
            f"{self.horizon}m: only {self.available:,} of the bars have a forward move "
            f"above the {self.hurdle_bps:.2f} bp labelling hurdle "
            f"({self.required:,} needed).\n"
            f"    This horizon is untradeable at the current cost assumption — the moves "
            f"are too small to pay for the round trip. Options:\n"
            f"      - train a longer horizon, where moves are larger\n"
            f"      - lower model.hurdle_multiple if you want the raw signal without a "
            f"cost margin\n"
            f"      - set model.enforce_cost_hurdle false to study signal alone, "
            f"remembering the backtest will still charge costs"
        )


@dataclass
class HorizonResult:
    """Everything produced by training one horizon."""

    horizon: int
    report: ClassificationReport
    fold_reports: list[ClassificationReport]
    oof: pd.DataFrame
    importance: pd.Series
    calibration_error: float
    dataset: Dataset
    model: DirectionEnsemble
    calibrator: ProbabilityCalibrator
    move_curve: dict = field(default_factory=dict)
    fold_details: list[Fold] = field(default_factory=list)

    @property
    def oof_accuracy(self) -> float:
        return self.report.accuracy

    def summary(self) -> dict:
        return {
            "horizon": self.horizon,
            "samples": self.report.n,
            "base_rate": round(self.report.base_rate, 4),
            "accuracy": round(self.report.accuracy, 4),
            "lift": round(self.report.accuracy_lift, 4),
            "auc": round(self.report.auc, 4),
            "brier": round(self.report.brier, 4),
            "mcc": round(self.report.mcc, 4),
            "calibration_error": round(self.calibration_error, 4),
            "hi_conf_accuracy": round(self.report.high_conf_accuracy, 4),
            "hi_conf_coverage": round(self.report.high_conf_coverage, 4),
        }


class Trainer:
    """Trains and persists multi-horizon direction models."""

    def __init__(self, settings: Settings, quiet: bool = False, n_jobs: int = 4) -> None:
        self.settings = settings
        self.quiet = quiet
        self.n_jobs = n_jobs
        self.skipped: dict[int, str] = {}

    # ------------------------------------------------------------------ single

    def train_horizon(
        self,
        features: pd.DataFrame,
        bars: pd.DataFrame,
        horizon: int,
        feature_names: list[str],
    ) -> HorizonResult:
        dataset = build_dataset(
            features,
            bars,
            horizon=horizon,
            feature_names=feature_names,
            deadband=self.settings.deadband_atr,
            hurdle_bps=self.settings.labelling_hurdle_bps(),
        )

        if len(dataset) < self.settings.min_train_bars:
            raise InsufficientSamples(
                horizon=horizon,
                available=len(dataset),
                required=self.settings.min_train_bars,
                hurdle_bps=self.settings.labelling_hurdle_bps(),
            )

        folds = purged_walk_forward(
            len(dataset),
            n_splits=self.settings.n_splits,
            horizon=horizon,
            embargo=self.settings.embargo_bars,
            min_train=max(int(len(dataset) * 0.35), 500),
        )

        if not self.quiet:
            console.print(
                f"  [dim]horizon {horizon}m:[/] {len(dataset):,} samples, "
                f"{len(folds)} folds, base rate {dataset.base_rate:.3f}"
            )

        oof_probabilities = np.full(len(dataset), np.nan)
        fold_reports: list[ClassificationReport] = []

        # Folds are independent, so they train in parallel. The threading backend
        # is the right choice here rather than processes: LightGBM and sklearn
        # release the GIL during fitting, and threads avoid pickling a 30k-row
        # feature frame once per worker. Estimators are pinned to one thread each
        # so the two levels of parallelism do not fight over the same cores.
        folds_parallel = min(self.n_jobs if self.n_jobs > 0 else 4, len(folds))
        workers = 1 if folds_parallel <= 1 else folds_parallel
        inner_jobs = -1 if workers == 1 else 1

        def _run_fold(fold: Fold) -> tuple[Fold, np.ndarray, ClassificationReport]:
            train_X = dataset.X.iloc[fold.train]
            train_y = dataset.y.iloc[fold.train]
            test_X = dataset.X.iloc[fold.test]
            test_y = dataset.y.iloc[fold.test].to_numpy()

            subset = Dataset(
                X=train_X,
                y=train_y,
                forward_return=dataset.forward_return.iloc[fold.train],
                horizon=horizon,
                feature_names=feature_names,
            )
            model = DirectionEnsemble(
                build_models(random_state=42 + horizon, n_jobs=inner_jobs)
            )
            model.fit(train_X, train_y, sample_weight=sample_weights(subset))
            probabilities = model.predict_proba(test_X)
            return fold, probabilities, evaluate(test_y, probabilities)

        results = Parallel(n_jobs=workers, backend="threading")(
            delayed(_run_fold)(fold) for fold in folds
        )

        for fold, probabilities, report in results:
            oof_probabilities[fold.test] = probabilities
            fold_reports.append(report)

        mask = ~np.isnan(oof_probabilities)
        oof_probabilities_filled = oof_probabilities[mask]
        oof_truth = dataset.y.to_numpy()[mask]

        report = evaluate(oof_truth, oof_probabilities_filled)

        calibrator = ProbabilityCalibrator(method="platt", min_samples=400)
        calibrator.fit(oof_probabilities_filled, oof_truth)
        calibrated = calibrator.transform(oof_probabilities_filled)
        calibration_error = expected_calibration_error(calibrated, oof_truth)

        if not self.quiet:
            console.print(f"    [dim]calibration: {calibrator.describe()}[/]")

        oof = pd.DataFrame(
            {
                "probability": oof_probabilities_filled,
                "calibrated": calibrated,
                "label": oof_truth,
                "forward_return": dataset.forward_return.to_numpy()[mask],
            },
            index=dataset.X.index[mask],
        )

        # Fit the conviction -> realised-move curve on out-of-sample data only.
        move_curve = expected_move_curve(oof["forward_return"], calibrated)

        final_model = DirectionEnsemble(build_models(random_state=42 + horizon), feature_names=feature_names)
        final_model.fit(dataset.X, dataset.y, sample_weight=sample_weights(dataset))

        return HorizonResult(
            horizon=horizon,
            report=report,
            fold_reports=fold_reports,
            oof=oof,
            importance=final_model.feature_importance(top=30),
            calibration_error=calibration_error,
            dataset=dataset,
            model=final_model,
            calibrator=calibrator,
            move_curve=move_curve,
            fold_details=folds,
        )

    # ------------------------------------------------------------------- all

    def train_all(
        self,
        features: pd.DataFrame,
        bars: pd.DataFrame,
        feature_names: list[str],
        horizons: tuple[int, ...] | None = None,
    ) -> dict[int, HorizonResult]:
        """Train every horizon, skipping any that the cost hurdle rules out.

        A horizon too short to produce tradeable moves is a legitimate outcome,
        not a failure — aborting the run would hide the horizons that do work.
        Skips are reported so the reason is never silent.
        """
        horizons = horizons or self.settings.horizons
        results: dict[int, HorizonResult] = {}
        self.skipped: dict[int, str] = {}

        for horizon in horizons:
            try:
                results[horizon] = self.train_horizon(features, bars, horizon, feature_names)
            except InsufficientSamples as exc:
                self.skipped[horizon] = exc.explain()
                if not self.quiet:
                    console.print(f"  [yellow]skipped {horizon}m[/] — {exc.explain()}")

        return results

    # ---------------------------------------------------------------- persist

    def save(self, results: dict[int, HorizonResult]) -> list[Path]:
        self.settings.ensure_dirs()
        paths: list[Path] = []
        for horizon, result in results.items():
            payload = {
                "version": ARTIFACT_VERSION,
                "horizon": horizon,
                "trained_at": datetime.now(IST).isoformat(),
                "symbol": self.settings.symbol,
                "feature_names": result.dataset.feature_names,
                "model": result.model,
                "calibrator": result.calibrator,
                "metrics": result.report.as_row(),
                "calibration_error": result.calibration_error,
                "importance": result.importance.to_dict(),
                "deadband_atr": self.settings.deadband_atr,
                "move_curve": result.move_curve,
                "hurdle_bps": self.settings.cost_hurdle_bps(),
                "labelling_hurdle_bps": self.settings.labelling_hurdle_bps(),
            }
            path = self.artifact_path(horizon)
            joblib.dump(payload, path)
            paths.append(path)

            # Persist the walk-forward predictions separately. The backtester
            # replays these rather than re-predicting: a model refit on all the
            # data has already seen the bars it would be judged on, so scoring it
            # on its own training window would produce a misleadingly good result.
            result.oof.to_parquet(self.oof_path(horizon))
            paths.append(self.oof_path(horizon))
        return paths

    def artifact_path(self, horizon: int) -> Path:
        return self.settings.model_dir / f"direction_{horizon}m.joblib"

    def oof_path(self, horizon: int) -> Path:
        return self.settings.model_dir / f"oof_{horizon}m.parquet"


def load_artifact(settings: Settings, horizon: int) -> dict:
    """Load a persisted model bundle."""
    path = settings.model_dir / f"direction_{horizon}m.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model for horizon {horizon}m at {path}. Run `niftypulse train` first."
        )
    return joblib.load(path)


def list_artifacts(settings: Settings) -> list[dict]:
    """Metadata for every trained artifact on disk."""
    out: list[dict] = []
    if not settings.model_dir.exists():
        return out
    for path in sorted(settings.model_dir.glob("direction_*m.joblib")):
        try:
            payload = joblib.load(path)
        except Exception:
            continue
        out.append(
            {
                "path": str(path),
                "horizon": payload.get("horizon"),
                "trained_at": payload.get("trained_at"),
                "metrics": payload.get("metrics", {}),
                "features": len(payload.get("feature_names", [])),
            }
        )
    return out


def render_scalping_hurdle(settings: Settings, bars: pd.DataFrame) -> None:
    """Show how often a scalp could pay for itself at each horizon.

    The ceiling on any scalping strategy is set by how large the moves are
    relative to the cost of capturing them. If most bars move less than the
    round trip, no model can fix that — so this table is worth reading before
    the model metrics, not after.
    """
    hurdle = settings.cost_hurdle_bps()
    table = Table(
        title=f"Cost hurdle — round trip {hurdle:.2f} bps at Rs {settings.reference_notional:,.0f}",
        header_style="bold cyan",
    )
    for column in ("horizon", "median move", "p75", "p90", "clears hurdle"):
        table.add_column(column, justify="right")

    for horizon in settings.horizons:
        stats = tradeable_fraction(bars, horizon, hurdle)
        if not stats.get("bars"):
            continue
        share = stats["tradeable_share"]
        colour = "green" if share > 0.3 else "yellow" if share > 0.1 else "red"
        table.add_row(
            f"{horizon}m",
            f"{stats['median_move_bps']:.2f} bp",
            f"{stats['p75_move_bps']:.2f} bp",
            f"{stats['p90_move_bps']:.2f} bp",
            f"[{colour}]{share:.1%}[/]",
        )
    console.print(table)
    console.print(
        "\n[dim]'clears hurdle' is the share of bars whose forward move exceeds the "
        "round-trip cost. That is the hard ceiling on how often a scalp over that "
        "horizon can profit, regardless of how good the model is.\n"
        f"Labelling requires a move above "
        f"{settings.labelling_hurdle_bps():.2f} bps "
        f"({settings.hurdle_multiple}x the hurdle).[/]"
    )


def render_report(results: dict[int, HorizonResult]) -> None:
    """Print the training report."""
    table = Table(title="Walk-forward out-of-sample performance", header_style="bold cyan")
    table.add_column("Horizon", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Base rate", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Lift", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("MCC", justify="right")
    table.add_column("Calib. err", justify="right")
    table.add_column("Hi-conf", justify="right")

    for horizon, result in results.items():
        row = result.summary()
        lift = row["lift"]
        table.add_row(
            f"{horizon}m",
            f"{row['samples']:,}",
            f"{row['base_rate']:.3f}",
            f"{row['accuracy']:.4f}",
            f"[{'green' if lift > 0 else 'red'}]{lift:+.4f}[/]",
            f"{row['auc']:.4f}",
            f"{row['mcc']:+.4f}",
            f"{row['calibration_error']:.4f}",
            f"{row['hi_conf_accuracy']:.4f}",
        )
    console.print(table)

    console.print(
        "\n[dim]Lift is accuracy over the majority-class baseline. A value near zero means "
        "the model is not beating 'always predict UP'.[/]"
    )

    for horizon, result in results.items():
        console.print(f"\n[bold]Confidence buckets — {horizon}m[/]")
        buckets = edge_by_confidence(
            result.oof["label"].to_numpy(), result.oof["calibrated"].to_numpy()
        )
        console.print(buckets.to_string())

        top = result.importance.head(8)
        if len(top):
            console.print(f"[bold]Top features — {horizon}m[/]")
            for name, value in top.items():
                console.print(f"  {name:28s} {value:.4f}")


def save_training_metadata(
    settings: Settings,
    results: dict[int, HorizonResult],
    skipped: dict[int, str] | None = None,
) -> Path:
    settings.ensure_dirs()
    payload = {
        "generated_at": datetime.now(IST).isoformat(),
        "symbol": settings.symbol,
        "cost_hurdle_bps": settings.cost_hurdle_bps(),
        "labelling_hurdle_bps": settings.labelling_hurdle_bps(),
        "horizons": {str(h): r.summary() for h, r in results.items()},
        "calibration": {str(h): r.calibration_error for h, r in results.items()},
        "skipped": {str(h): reason for h, reason in (skipped or {}).items()},
    }
    path = settings.model_dir / "training_report.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def render_balance(features: pd.DataFrame, bars: pd.DataFrame, settings: Settings) -> None:
    table = Table(title="Label balance by horizon", header_style="bold cyan")
    table.add_column("Horizon", justify="right")
    table.add_column("Up", justify="right")
    table.add_column("Down", justify="right")
    table.add_column("Up share", justify="right")
    for horizon in settings.horizons:
        dataset = build_dataset(
            features, bars, horizon, [c for c in features.columns if c != "bar_minutes"],
            deadband=settings.deadband_atr,
        )
        balance = class_balance(dataset)
        table.add_row(
            f"{horizon}m",
            f"{balance['up']:,}",
            f"{balance['down']:,}",
            f"{balance['up_share']:.3f}",
        )
    console.print(table)
