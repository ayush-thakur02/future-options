"""Web dashboard renderer plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .server import WebRenderer

MANIFEST = PluginManifest(
    name="web",
    kind=PluginKind.RENDERER,
    description="Local Flask dashboard with realtime JSON snapshots and terminal styling",
    provides=("web_frame",),
    tags=("web", "flask", "dashboard", "realtime"),
    params={
        "host": "127.0.0.1",
        "port": 5050,
        "refresh_ms": 1000,
        "open_browser": True,
        "max_candles": 160,
    },
)


def build(
    ctx: PluginContext,
    host: str = "127.0.0.1",
    port: int = 5050,
    refresh_ms: int = 1000,
    open_browser: bool = True,
    max_candles: int = 160,
    **params,
) -> WebRenderer:
    return WebRenderer(
        host=host,
        port=port,
        refresh_ms=refresh_ms,
        open_browser=open_browser,
        max_candles=max_candles,
    )


__all__ = ["MANIFEST", "build"]
