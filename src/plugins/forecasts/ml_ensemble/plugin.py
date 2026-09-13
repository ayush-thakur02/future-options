"""The ML ensemble as a plugin.

Provides the ``forecast`` capability: calibrated probabilities per horizon. The
strategy layer consumes it through that capability rather than importing this
pack, which is why the ML-backed strategy can be dropped from a build simply by
disabling this pack.

Building never fails on a machine with no trained artifacts. A missing model is a
state, not an error — ``Predictor.is_ready`` is False and the platform runs with
rule-based strategies alone.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .predictor import Predictor

MANIFEST = PluginManifest(
    name="ml_ensemble",
    kind=PluginKind.FORECAST,
    description="Calibrated P(up) per horizon from a gradient-boosted ensemble",
    provides=("forecast",),
    tags=("ml", "supervised", "offline-trained"),
)


def build(ctx: PluginContext, horizons=None, **params) -> Predictor:
    """Load whatever artifacts exist for the configured horizons."""
    selected = tuple(int(h) for h in horizons) if horizons else ctx.settings.horizons
    return Predictor(ctx.settings, horizons=selected).load()
