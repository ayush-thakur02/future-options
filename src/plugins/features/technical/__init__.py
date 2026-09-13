"""The technical feature set: indicators, session context, and the matrix.

The pack is a package rather than one module because the three concerns are
genuinely separate and each is long:

* :mod:`~plugins.features.technical.indicators` — 40+ indicators, no TA-Lib
* :mod:`~plugins.features.technical.session` — time of day, calendar, regime
* :mod:`~plugins.features.technical.pipeline` — assembly and the ``FeaturePipeline``

Everything here is pure and causal: given the same bars it returns the same
matrix, and no column ever reads a bar that had not closed yet.
"""

from .pipeline import (
    FEATURE_GROUPS,
    FeaturePipeline,
    build_features,
    feature_columns,
    regime_labels,
)
from .session import classify_regime

__all__ = [
    "FEATURE_GROUPS",
    "FeaturePipeline",
    "build_features",
    "classify_regime",
    "feature_columns",
    "regime_labels",
]
