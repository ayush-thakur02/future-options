"""Session-anchored time bars.

Unlike every other pack, this one is a plugin because a bar *definition* is a
choice, not an implementation detail: time bars, tick bars, and volume bars are
the same slot from the feature layer's point of view, and the ``bars`` capability
is what lets one replace another.
"""

from .aggregator import CandleAggregator, bar_start

__all__ = ["CandleAggregator", "bar_start"]
