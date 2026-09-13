"""Render packs: a MarketSnapshot in, something a human can read out.

A renderer is a plugin like any other, which is what makes the terminal
dashboard one implementation of an interface rather than the only way to look at
the platform. A web renderer, a JSON feed for another process, or an alerting
renderer that only speaks when something clears the cost hurdle all take the same
slot by providing the ``frame`` capability.
"""
