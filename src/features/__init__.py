"""Feature engineering: indicators, context, and matrix assembly."""

from .pipeline import FEATURE_GROUPS, build_features, feature_columns, regime_labels

__all__ = ["FEATURE_GROUPS", "build_features", "feature_columns", "regime_labels"]
