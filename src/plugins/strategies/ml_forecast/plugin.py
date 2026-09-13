"""The model-backed strategy as a plugin.

The only strategy pack that declares a requirement: it cannot exist without a
``forecast`` capability, so a build with the ML pack disabled simply does not
offer ``strategy:ml`` — and the catalog records why instead of failing.

Model parameters are read from the artifact rather than duplicated in config,
because the expected-move curve and the cost hurdle belong to the model that was
measured, not to whoever is configuring the dashboard.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .strategy import MLStrategy

MANIFEST = PluginManifest(
    name="ml_forecast",
    kind=PluginKind.STRATEGY,
    description="The trained model as a strategy, gated by edge and the cost hurdle",
    provides=("strategies:ml_forecast", "strategy:ml"),
    requires=("forecast",),
    tags=("ml", "model", "cost-gated"),
    params={"horizon": 1, "min_edge": 0.04},
)


def build(
    ctx: PluginContext,
    horizon: int = 1,
    min_edge: float = 0.04,
    **params,
) -> StrategyPack:
    """Bind the ML strategy to the model trained for ``horizon``.

    Raises if that model has not been trained, which the catalog records as a
    skipped pack: an untrained model is a normal state offline, not a fault.
    """
    predictor = ctx.capability("forecast")
    horizon = int(horizon)
    artifact = getattr(predictor, "artifacts", {}).get(horizon)
    if artifact is None:
        raise RuntimeError(
            f"no trained model for horizon {horizon}m; run `niftypulse train` "
            f"to enable the ML strategy"
        )

    return StrategyPack(
        name="ml_forecast",
        category="ml",
        description=MANIFEST.description,
        instances=(
            MLStrategy(
                predictor=predictor,
                horizon=horizon,
                feature_names=list(artifact.get("feature_names", [])),
                min_edge=min_edge,
                move_curve=artifact.get("move_curve") or {},
                hurdle_bps=float(artifact.get("hurdle_bps", 0.0)),
            ),
        ),
    )
