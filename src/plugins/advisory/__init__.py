"""Advisory packs: everything in, a decision out.

An advisory plugin is the only kind allowed to say what to *do*. It consumes the
whole board — the underlying, the legs, the projection, the chain — and produces a
verdict per leg with its arithmetic attached.

It is a separate kind from a forecast on purpose. A forecast is a statement about
the market; a verdict is a statement about a position, and it has to account for
what the position costs to hold. Folding the second into the first is how a
platform ends up recommending trades that cannot pay for themselves.
"""
