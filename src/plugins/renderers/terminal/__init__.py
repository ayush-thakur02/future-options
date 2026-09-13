"""The terminal renderer.

Small modules, each with one job:

* :mod:`~plugins.renderers.terminal.theme`    colours and glyphs
* :mod:`~plugins.renderers.terminal.canvas`   the character grid and markup runs
* :mod:`~plugins.renderers.terminal.widgets`  bars, meters, sparklines, the axis
* :mod:`~plugins.renderers.terminal.candles`  printed and projected candles
* :mod:`~plugins.renderers.terminal.panels`   the panels themselves
* :mod:`~plugins.renderers.terminal.dashboard` layout and the live loop
"""

from .candles import Chart, render_candles, render_chart
from .dashboard import TerminalRenderer, render_once

__all__ = ["Chart", "TerminalRenderer", "render_candles", "render_chart", "render_once"]
