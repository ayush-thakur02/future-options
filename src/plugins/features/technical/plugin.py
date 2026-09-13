"""The technical feature set as a plugin.

Provides the ``features`` capability. The engine asks for that capability rather
than importing the pack, so a different feature set — order-flow, microstructure,
learned embeddings — takes the slot by providing the same name.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .pipeline import FeaturePipeline

MANIFEST = PluginManifest(
    name="technical",
    kind=PluginKind.FEATURES,
    description="40+ causal indicators plus session, calendar and regime context",
    provides=("features",),
    tags=("indicators", "technical", "context"),
)


def build(
    ctx: PluginContext,
    expiry_weekday: int | None = None,
    include_context: bool = True,
    **params,
) -> FeaturePipeline:
    """Build the pipeline. Expiry weekday defaults to the configured one."""
    return FeaturePipeline(
        expiry_weekday=(
            ctx.settings.expiry_weekday if expiry_weekday is None else int(expiry_weekday)
        ),
        include_context=include_context,
    )
