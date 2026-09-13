"""Purged, embargoed walk-forward cross-validation.

Standard k-fold is invalid for financial labels. Two failures:

**Overlap.** A label at bar *t* is computed from prices through bar *t + h*.
If *t* sits in the training set and *t + h* falls inside the test set, the model
has been shown the answer. Accuracy inflates, and the effect is largest exactly
where the horizon is long relative to the fold size.

**Serial correlation.** Even without direct label overlap, bars adjacent across
the boundary share nearly identical features. A model that memorises the regime
around the split scores well on both sides without learning anything general.

Purging removes training samples whose label window reaches into the test set.
Embargoing additionally drops a margin after the boundary. Together they make the
test score an estimate of what the model would have done on data it had genuinely
never seen. Method follows López de Prado, *Advances in Financial Machine
Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward split."""

    index: int
    train: np.ndarray
    test: np.ndarray
    purged: int
    embargoed: int

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    def describe(self) -> str:
        return (
            f"fold {self.index}: train={self.train_size:,} test={self.test_size:,} "
            f"(purged {self.purged}, embargoed {self.embargoed})"
        )


def purged_walk_forward(
    n_samples: int,
    n_splits: int = 6,
    horizon: int = 5,
    embargo: int = 20,
    min_train: int = 500,
) -> list[Fold]:
    """Expanding-window walk-forward folds with purging and an embargo.

    Training always precedes testing in time, so each fold answers the question
    that actually matters: given everything up to here, what happens next?
    """
    if n_samples < min_train + n_splits:
        raise ValueError(
            f"need at least {min_train + n_splits} samples for {n_splits} folds, got {n_samples}"
        )

    # Hold back an initial training block, then split the rest into test blocks.
    test_span = max((n_samples - min_train) // n_splits, 1)
    folds: list[Fold] = []

    for fold_index in range(n_splits):
        test_start = min_train + fold_index * test_span
        test_end = min(test_start + test_span, n_samples)
        if test_start >= n_samples:
            break

        # Purge training rows whose label window reaches into the test block.
        purge_cut = max(test_start - horizon - embargo, 1)
        train = np.arange(0, purge_cut)
        test = np.arange(test_start, test_end)

        purged = max(min(test_start, horizon + embargo) - max(0, test_start - purge_cut), 0)
        embargoed = min(embargo, test_start)

        if len(train) < 100 or len(test) == 0:
            break

        folds.append(
            Fold(
                index=fold_index,
                train=train,
                test=test,
                purged=int(purged),
                embargoed=int(embargoed),
            )
        )

    if not folds:
        raise ValueError("no valid folds produced; check min_train and n_splits")
    return folds


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 6,
    horizon: int = 5,
    embargo: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convenience form returning plain (train, test) index pairs."""
    min_train = max(int(n_samples * 0.35), 500)
    folds = purged_walk_forward(
        n_samples, n_splits=n_splits, horizon=horizon, embargo=embargo, min_train=min_train
    )
    return [(fold.train, fold.test) for fold in folds]


def describe_folds(folds: list[Fold]) -> str:
    lines = [fold.describe() for fold in folds]
    covered = sum(fold.test_size for fold in folds)
    lines.append(f"total out-of-sample rows: {covered:,}")
    return "\n".join(lines)
