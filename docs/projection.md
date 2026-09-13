# Projection — the next three candles

Every second, for every instrument on the board, the platform projects the next
three candles and draws them in blue after the forming one. This document is what
that path is, what it is not, and how it is scored.

Pack: `forecast:projection` · capability: `projection` · clock: `runtime/nowcast.py`

---

## What it is

A **path estimate**, not a price prediction. Given where price is now, how strong
the current view is, and how much the market is moving, where are the next few
bars likely to travel?

Three properties make it useful rather than decorative.

**It is anchored on the live price.** The first projected bar opens at the last
traded price. Between bar closes the path therefore moves with the tape instead of
drifting away from it — a projection drawn from the last *closed* bar would sit
still for a minute while the market walked away from it.

**It decays with horizon.** Conviction for the bar after next is worth less than
conviction for the next one, and the third less again
(`HORIZON_DECAY = 0.72`). A projection that kept full conviction three bars out
would be claiming to know more than it does.

**It widens with horizon.** Uncertainty compounds, so the projected range grows
per bar (`BASE_RANGE = 0.85`, `RANGE_GROWTH = 0.30` per bar, both in ATR). Far
bars are drawn taller because they are genuinely less certain.

Everything is deterministic given its inputs. That is what makes it scorable: a
projection that jittered randomly each second could never be evaluated.

---

## Where the direction comes from

Three inputs at three cadences, each earning its place:

| Input | Cadence | What it is |
|---|---|---|
| Strategy ensemble + model | bar close | The considered view: nineteen rules blended, times the model's probability when trained |
| Bar-scale trend | every refresh | Fast against slow EMA over recent bars, in ATR units, saturated to ±1 |
| Tick momentum | every tick | Fast against slow exponential average of *tick* prices, saturated to ±1 |

`runtime/conviction.py` blends the first two (`FAST_WEIGHT = 0.35`); the
projection pack blends in the third (`MICRO_WEIGHT = 0.35`). The weights are
simple and monotone on purpose: a more elaborate scheme would be untestable
against a target that is itself noisy, and the property that matters is only that
the path moves when the market moves.

Tick momentum is weighted by *elapsed time*, not tick count. A burst of twenty
prints inside 100ms and twenty spread over twenty seconds cover the same distance;
only the second is a move the market travelled. Without that, one large print
would swing the whole path.

---

## The arithmetic

For projected bar *i* = 1, 2, 3:

```
decay    = HORIZON_DECAY ^ (i - 1)
drift    = view × decay × DRIFT_SCALE × ATR
close_i  = open_i + drift
open_i   = close_(i-1),  with open_1 = live price
span     = ATR × (BASE_RANGE + RANGE_GROWTH × (i - 1)) × vol_scale
high     = max(open, close) + span × WICK_FRACTION
low      = min(open, close) - span × WICK_FRACTION
confidence = |view| × decay
```

`vol_scale` is current ATR against its own recent median, clamped to
`[0.55, 2.20]` — elevated volatility should widen a projection, and one violent
bar should not produce an absurd range.

Constants live at the top of `plugins/forecasts/projection/candles.py`. They are
prior beliefs about decay and range, not fitted parameters, and they are named so
they can be argued with.

---

## Reading it on the chart

| Colour | Meaning |
|---|---|
| Green / red | A printed bar. Green up, red down |
| Blue, brightest | The next bar |
| Blue, dimmer | Two bars out, three bars out |
| Fill density | Confidence: `▓` above 0.55, `▒` above 0.25, `░` below |

Projected bars are **never** drawn in green or red. Colour is the only thing
distinguishing them from printed bars at a glance, and a rising projection drawn
green would be actively misleading. A test asserts this through the rendered
style spans, not by eye.

The `┊` divider marks the boundary between what happened and what is expected, so
"now" is a position on the chart rather than a caption.

All three of the board's charts share one layout but **not** one price scale. A
premium and an index level on a single axis would be the most misleading thing the
screen could do.

---

## Is it any good?

A projected candle nobody checks is decoration. The projection pack keeps a
scoreboard.

When a projected bar opens, its projection is recorded against the target
timestamp and the live price it was anchored on. When that bar closes, the
projection is scored:

- **hit** — did the path lean the right way? (projected close versus anchor, and
  realised close versus anchor, same sign)
- **error** — how far, in basis points, did the projected close land from the
  real one?

Scores are kept **per horizon**, because the three answer different questions. A
one-bar-ahead projection is nearly a nowcast — the forming bar already contains
most of it. A three-bar-ahead projection is a real claim and will be wrong more
often. Averaging them into one number would hide exactly the decay the design is
honest about.

The projection panel prints the one-bar score: hit rate and mean error over the
session. `forecaster.describe()` returns all three as JSON.

Two details that keep the scoreboard honest:

- The **latest** projection for a given (target bar, horizon) is the one scored —
  it is the freshest estimate and the one a user was looking at when the bar
  closed.
- A **roll** resets the tracker. A projection made about one strike must never be
  scored against another strike's bars, which is a lie that would otherwise be
  invisible.

---

## Refresh cadence

`runtime/nowcast.py` rebuilds the path on a fixed interval (default 1s,
`--nowcast` to change it). Ticks drive the anchor and the momentum; the clock
drives the path. Keeping them apart is what makes the projection tick every
second on a quiet tape instead of only when a print arrives — and quiet stretches
of several seconds are normal on an index feed.

A bar close also triggers a refresh immediately, because a bar close changes the
volatility, the trend and the ensemble view all at once.

Measured cost of one refresh across a three-leg board: **~10–17 ms**, against a
1,000 ms budget.

A failed refresh is logged and skipped. A projection is not worth taking the
dashboard down for.

---

## Turning it off

Projections are optional. With the pack disabled, `engine.refresh_projection()`
returns `False`, the chart draws printed bars only, and nothing else changes:

```yaml
plugins:
  forecast:projection:
    bars_ahead: 3      # or disable the pack in your composition
```
