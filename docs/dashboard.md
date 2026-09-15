# Dashboard

One live session, served to the browser. The **board** puts the at-the-money call
and put beside the index; `--no-legs` gives the single-instrument view with the
strategy and forecast panels where the chart space was.

```bash
uv run niftypulse                      # live websocket, board, recording as it runs
uv run niftypulse --offline --speed 60 # replay generated ticks, 60x
uv run niftypulse --no-legs            # index only
uv run niftypulse --no-learn           # freeze online learning
uv run niftypulse --workers -1         # use every CPU
```

The process stays the owner of the market session: history is warmed, the online
learners start, the browser opens, and `Ctrl+C` in the terminal stops the feed and
the server together. Use `--no-open-browser` on a headless machine and open the
printed URL yourself.

---

## What the page shows

A white, responsive, monospace research terminal. It polls `/api/snapshot` once a
second and redraws; `/api/health` reports whether a snapshot has arrived yet.

| Section | Contents |
|---|---|
| Overview | Spot, change, regime, feed freshness and today's aggregate AI accuracy across index, call and put |
| 01 Decision desk | A cost-gated BUY/HOLD/SELL card per instrument, with the current reason, target and invalidation range. Instrument meaning is explicit: a long put is bearish on the index but bullish on put premium |
| 02 Money simulator | The funded paper book: equity, net result, reserve drawdown, then a card per leg with its open lot, the blended view behind it, and the decision log and closed trades |
| 03 Live instruments | Zoom-sharp SVG candlesticks for index, call and put, with blue projected candles on the same price scale as printed ones |
| 04 Prediction matrix | Index/call/put × horizon: P(up), trust, target time, projected close, expected move, cost gate and measured hit rate |
| 05 Realtime auto-AI | Learner sample counts, hit/miss results, all-time and rolling accuracy, net basis points, drawdown and trust |
| 06 Strategy matrix | Every discovered rule with per-leg state, confidence, trust, hit/miss and net result, plus three summary rows aggregating UP, DOWN, HOLD and net breadth. Filterable by name |
| 07 Indicator state | EMA 9/21/50/200, VWAP, SuperTrend, trend, momentum, volatility, flow, channel, regime and the quantitative indicators the strategies and models consume |
| 08 Forward score | Frozen projections against the close that actually printed |
| 09 System state | Feed freshness, queues, the cost assumptions the verdicts use, and what the pre-open warm-up managed |

Click any strategy row, AI row, prediction cell or indicator to open the live
calculation inspector: current values, the formula or policy used, the exact
threshold comparison, and matured performance evidence. That is also where the
distinction between probability, confidence, trust and post-cost edge is made
explicit.

The money simulator's view line opens the same inspector, listing every source
that fed the blended view — its value, its weight and its weighted contribution —
beside the projected move, the round-trip hurdle and the resulting edge.
[Money simulator](money-simulator.md) has the arithmetic.

Projected SVG bars carry only their blue OHLC candles and a subtle solid
printed/projected boundary — there is no dotted target path. Hover a bar for its
open, high, low, close and confidence, or read those values in the strip below the
chart.

---

## Colours

| Colour | Meaning |
|---|---|
| Green / red | A printed bar |
| Blue | The **next three projected** candles — brightest nearest |

Blue is never used for a printed bar and green/red are never used for a projected
one. That is enforced by test, not by convention.

---

## The board

Each instrument is a candlestick chart with its own price scale, and the strip
beneath it carries the numbers that decide whether the chart matters: for the
index, the conviction and a sparkline; for a leg, its delta, theta per minute,
implied vol, and the move the underlying must make for the leg to pay for itself.

The verdicts below the charts are one row per leg. `needs` is the requirement,
`projected` is the index's projected move over the same horizon, `edge` is the
difference, `do` is the action. The headline either names a trade or says plainly
that nothing clears its own breakeven — the correct output most of the time. See
[Options](options.md).

---

## Refresh cadence

| What | How often | Flag |
|---|---|---|
| Ticks | as they arrive | — |
| Projected path | every second | `--nowcast` |
| Snapshot redraw | every second | `--refresh` |
| Browser poll | every second | `config/plugins/renderer/web.yaml` |
| Model forecasts | on bar close | — |

The redraw is deliberately decoupled from the projection: a faster `--refresh`
gives a smoother-looking page while the projection still recomputes on its own
second, which is what the blue candles are promised to do. Both are capped at one
second, because a live market view that is slower than that is stale before it is
drawn.

---

## Binding and access

The server is read-only and defaults to `127.0.0.1:5050`. Binding it to the
network is explicit:

```bash
uv run niftypulse --host 0.0.0.0 --port 8080 --no-open-browser
```

Only do that on a trusted network; Flask adds no authentication to this local
research view. The bind defaults and candle history limit live in
`config/plugins/renderer/web.yaml`.

---

## Offline mode

`--offline` replays a generated series **paced against the wall clock**: one minute
of wall clock per one-minute bar at `--speed 1.0`. That is deliberate — a replay
that finishes a trading day in two seconds gives the projected candles no seconds
to move in, which is the one thing they exist to do.

`--speed 60` makes a minute take a second, which is the practical setting for
watching the board work.

The first pass is re-based onto the current time so a session opens on the present;
later passes continue forward rather than rewinding, because a tick stream that
moves backwards is not something the aggregator is built to survive.

Nothing generated is recorded: the tape and the chain are only written when the
session is live, so invented prices can never mix into the datasets that cannot be
re-fetched.
