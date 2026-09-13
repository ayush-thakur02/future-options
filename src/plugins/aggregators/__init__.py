"""Aggregators: the stage between a tick stream and the bar series.

An aggregator decides what a "bar" is. Time bars are the default, but the same
slot makes tick bars, volume bars, or a resampled higher timeframe a drop-in
replacement — anything that provides the ``bars`` capability can take the slot,
and every consumer downstream is unaffected.
"""
