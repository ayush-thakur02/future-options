"""Flask web dashboard renderer."""

from .serialize import serialize_snapshot
from .server import WebRenderer

__all__ = ["WebRenderer", "serialize_snapshot"]
