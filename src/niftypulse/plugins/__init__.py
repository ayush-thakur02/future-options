"""Bundled plugins.

Every folder below this one is a plugin pack. A pack is any package containing a
``plugin.py`` that exposes a ``MANIFEST`` and a ``build(ctx, **params)``. The
loader walks this tree, so nothing here needs to be listed, registered, or
imported anywhere else — adding a folder is the whole installation procedure.

    sources/        where ticks and bars come from
    aggregators/    tick -> bar
    features/       bar -> model inputs
    strategies/     bar -> signed conviction
    forecasts/      conviction -> predictions and projected candles
    renderers/      snapshot -> something a human reads
    tools/          offline research that is not part of the live graph

Third-party packs work the same way from an installed distribution advertising
the ``niftypulse.plugins`` entry-point group.
"""
