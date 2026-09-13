"""Feature packs: bars in, model inputs out.

A feature pack is the bar-to-matrix stage. It is a plugin because the feature set
is the single largest influence on what a model can learn, and swapping it should
be a matter of dropping in a folder rather than editing an assembly function.
Any pack that provides the ``features`` capability can take the slot.
"""
