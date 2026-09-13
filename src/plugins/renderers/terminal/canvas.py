"""A character canvas that emits Rich markup.

Everything the chart draws lands here first as a character grid, and is colourised
afterwards by grouping runs of identically styled characters. Emitting markup per
character instead would produce a string many times the size of the visible
output and make the dashboard lag on a busy feed.

The grid also keeps the two properties the chart depends on: every cell is
exactly one character, so a row's visible width is known without measuring, and
leading blanks are preserved rather than trimmed, because they are what
right-aligns a chart inside its panel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Cell:
    char: str
    style: str


class Grid:
    """A fixed-size character canvas."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells: list[list[Cell | None]] = [[None] * width for _ in range(height)]

    def put(self, row: int, column: int, char: str, style: str = "") -> None:
        if 0 <= row < self.height and 0 <= column < self.width and char:
            self.cells[row][column] = Cell(char, style)

    def fill_column(self, column: int, top: int, bottom: int, char: str, style: str = "") -> None:
        """Draw a vertical run. Used for wicks, and for a bar with no body."""
        for row in range(min(top, bottom), max(top, bottom) + 1):
            self.put(row, column, char, style)

    def render(self) -> list[str]:
        """Render each row, exactly ``width`` visible characters wide."""
        return [render_row(row) for row in self.cells]


def render_row(row: list[Cell | None]) -> str:
    """Render one row to markup, grouping runs of identical style.

    Leading blanks are **preserved**: they position the chart inside the panel.
    Stripping them shifts every candle to the left edge, which is exactly the bug
    that made the chart look broken once already.
    """
    parts: list[str] = []
    run_style: str | None = None
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        text = "".join(run)
        parts.append(f"[{run_style}]{text}[/]" if run_style else text)
        run.clear()

    for cell in row:
        if cell is None:
            char, style = " ", None
        else:
            char, style = cell.char, (cell.style or None)
        if style != run_style:
            flush()
            run_style = style
        run.append(char)
    flush()
    return "".join(parts)


__all__ = ["Cell", "Grid", "render_row"]
