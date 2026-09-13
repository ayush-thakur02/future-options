"""The terminal renderer as a plugin.

Provides the ``frame`` capability — one rendered view of the market. The
dashboard is built from the kernel's ``snapshot`` subscription rather than being
imported by the CLI, so a different renderer can be put in its place without the
CLI knowing which one it is talking to.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .dashboard import TerminalRenderer

MANIFEST = PluginManifest(
    name="terminal",
    kind=PluginKind.RENDERER,
    description="Full-screen terminal dashboard: candles, projections, forecasts",
    provides=("frame",),
    tags=("terminal", "rich", "tui"),
    params={"refresh": 1.0},
)


def build(ctx: PluginContext, refresh: float = 1.0, forecaster=None, **params) -> TerminalRenderer:
    """Build the dashboard. The forecaster is injected so the panel can show how
    the projection has scored, rather than recomputing it for display.
    """
    return TerminalRenderer(
        symbol=ctx.settings.symbol,
        timeframe=f"{ctx.settings.bar_minutes}m",
        refresh=refresh,
        forecaster=forecaster,
    )
