"""A small synchronous event bus.

Plugins do not call each other; they publish and subscribe. That is the second
half of the plug-and-play story: a source publishes ticks without knowing that a
projection forecaster or a renderer is listening, so a new consumer is added by
dropping in a folder.

Deliberately synchronous, and deliberately defensive. The bus sits directly
underneath a live market feed, so a handler that raises must not take the feed
down with it — the offending handler is reported once and skipped, and every
other subscriber still receives the event.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Handler = Callable[[Any], None]


class Topic(StrEnum):
    """The event vocabulary plugins exchange."""

    TICK = "tick"
    BAR = "bar"
    FORECAST = "forecast"
    PROJECTION = "projection"
    SNAPSHOT = "snapshot"
    STATUS = "status"
    ERROR = "error"


@dataclass(slots=True)
class Subscription:
    """A live subscription. Cancel it to stop receiving events."""

    topic: str
    handler: Handler
    owner: str = ""
    active: bool = True

    def cancel(self) -> None:
        self.active = False

    def __call__(self, payload: Any) -> None:
        self.handler(payload)


class EventBus:
    """Publish/subscribe with per-handler fault isolation."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._handlers: dict[str, list[Subscription]] = {}
        self._log = logger or logging.getLogger("niftypulse.bus")
        self._failures: dict[str, int] = {}

    def subscribe(self, topic: Topic | str, handler: Handler, owner: str = "") -> Subscription:
        key = _topic_key(topic)
        subscription = Subscription(topic=key, handler=handler, owner=owner)
        self._handlers.setdefault(key, []).append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        subscription.cancel()
        handlers = self._handlers.get(subscription.topic, [])
        if subscription in handlers:
            handlers.remove(subscription)

    def publish(self, topic: Topic | str, payload: Any = None) -> int:
        """Deliver ``payload`` to every subscriber. Returns how many succeeded."""
        key = _topic_key(topic)
        delivered = 0
        for subscription in list(self._handlers.get(key, [])):
            if not subscription.active:
                continue
            try:
                subscription(payload)
                delivered += 1
            except Exception as exc:  # noqa: BLE001 — one bad handler must not stop the feed
                self._record_failure(subscription, key, exc)
        return delivered

    def _record_failure(self, subscription: Subscription, topic: str, exc: Exception) -> None:
        name = subscription.owner or getattr(subscription.handler, "__qualname__", "handler")
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        # Report the first failure loudly and the rest as a count: a handler that
        # raises on every tick would otherwise flood the log at tick rate.
        if count == 1:
            self._log.warning("handler %s failed on %s: %s", name, topic, exc)

    def listeners(self, topic: Topic | str | None = None) -> int:
        if topic is None:
            return sum(len(items) for items in self._handlers.values())
        return len(self._handlers.get(_topic_key(topic), []))

    @property
    def failures(self) -> dict[str, int]:
        return dict(self._failures)

    def clear(self) -> None:
        for handlers in self._handlers.values():
            for subscription in handlers:
                subscription.cancel()
        self._handlers.clear()

    def __repr__(self) -> str:
        return f"<EventBus {self.listeners()} handlers across {len(self._handlers)} topics>"


def _topic_key(topic: Topic | str) -> str:
    return topic.value if isinstance(topic, Topic) else str(topic)


__all__ = ["EventBus", "Handler", "Subscription", "Topic"]
