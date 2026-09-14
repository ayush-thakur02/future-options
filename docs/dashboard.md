# Dashboard

The same live session can render as a native terminal screen or a local Flask web
dashboard. The **board** puts the call and the put beside the index;
`--no-legs` gives the single-instrument view with the strategy and forecast panels
where the chart space was.

```bash
niftypulse dashboard                  # live websocket, board, recording as it runs
niftypulse dashboard --web            # same session at http://127.0.0.1:5050
niftypulse dashboard --web --offline --speed 60
niftypulse dashboard --offline --speed 60
niftypulse dashboard --no-legs --refresh 0.25
niftypulse snapshot                   # one frame to stdout, then exit
```

---

## Web terminal

`dashboard --web` starts the Flask renderer after history and the online learners
are warm, opens the default browser, and keeps the terminal process as the owner
of the market session. Press `Ctrl+C` in that terminal to stop the feed and web
server. Use `--no-open-browser` on a headless machine and open the printed URL
yourself.

The browser is a white, responsive, monospace research terminal. It polls the
local snapshot API and shows:

- zoom-sharp SVG candlestick graphs for index, call, and put, with blue projected
  candles on the same price scale as printed candles;
- a separate cost-gated BUY/HOLD/SELL decision card for index, call, and put,
  including the current reason, target, invalidation range, and the exact
  probability/trust rules that must be satisfied;
- an index/call/put-by-horizon prediction matrix with P(up), trust, target time,
  projected close, expected move, cost gate, and measured hit rate;
- realtime learner sample counts, hit/miss results, all-time and rolling accuracy,
  net basis points, drawdown, and trust;
- a searchable strategy matrix whose top row aggregates UP, DOWN, HOLD, average
  confidence, average trust, and net breadth across all 30 strategies; each rule
  then shows per-leg state, current confidence, trust, hit/miss, and net result;
- EMA 9/21/50/200, VWAP, SuperTrend, trend, momentum, volatility, flow, channel,
  regime, and quantitative indicators used by the strategies and online models;
- frozen forward-score results, feed freshness, queues, and cost assumptions.

Click any strategy row, online AI row, prediction cell, or indicator to open the
live calculation inspector. It shows the current values, the formula/policy used,
the exact action threshold comparison, and matured performance evidence. This is
also where the distinction between probability, confidence, trust, and post-cost
edge is made explicit.

Projected SVG bars contain only their blue OHLC candles and a subtle solid
printed/projected boundary; there is no dotted target path. Hover a bar for its
open, high, low, close, and confidence, or read those values in the strip below
the chart.

The server is read-only and defaults to `127.0.0.1:5050`. Binding it to the
network is explicit:

```bash
niftypulse dashboard --web --web-host 0.0.0.0 --web-port 8080 --no-open-browser
```

Only do that on a trusted network; Flask adds no authentication to this local
research view. The bind defaults and candle history limit live in
`config/plugins/renderer/web.yaml`.

Web snapshots and projected paths refresh at least once per second. Setting a
faster `--refresh` or `--nowcast` cadence is supported; slower values are capped
at one second in web mode so the browser does not silently show stale state.

---

## The board

```
╭───────────────────────────────────────────────────────────────────────────╮
│  NIFTY 50 · 1m   22,150.75  +102.36 (+0.46%)  regime trending  view -0.21 │
│  expiry Tue 15 Sep   PCR 1.07                     21:02:06 IST  live       │
╰───────────────────────────────────────────────────────────────────────────╯
╭──── INDEX · 22,150.75 · 1m ────╮╭─── CALL 22,150 · 79.15 · 1m ───╮╭─── PUT ───╮
│ █                           ┊   ││ █                           ┊  ││         │
│ █▄▄█│                       ┊   ││ █▄▄█                        ┊  ││    ██   │
│    █▄█│                     ┊   ││    █▄█                      ┊  ││  █▄█ █▄▄ │
│        ██ │                 ┊   ││       █▄█ │                 ┊  ││         │
│   conv -0.21  ▇▇▆█▇▆▅▄▃▂▁▁▁▂    ││  Δ +0.50  θ/min -0.052  IV 10.0%  needs 71bp │
╰────────────────────────────────╯╰─────────────────────────────╯╰───────────╯
╭──────────────── does the move pay for the position? ──────────────────────╮
│  leg            do       needs   projected      edge  why                  │
│  INDEX          ▼F         4bp      -0.5bp      -3bp  needs 3.9bp, …      │
│  CALL 22,150    ▼F        60bp      -0.5bp     -60bp  projection points…   │
│  PUT 22,150     ▼F        62bp      -0.5bp     -62bp  needs 62bp, proj…    │
│   no leg clears its own breakeven — the correct output here is no trade    │
╰───────────────────────────────────────────────────────────────────────────╯
╭──── active strategies ────╮╭───────────── indicator state ──────────────────╮
│  ema_trend        ▲ ▮▮▮▮  ││  RSI(14)        41.7  neutral                  │
╰───────────────────────────╯╰───────────────────────────────────────────────╯
 █ actual  ▓ projected  │ next INDEX 3 CALL 3 PUT 3 bars │ q quit  r refresh
```

### Header

Symbol, timeframe, spot, change, regime, the index's current view, the expiry the
legs are trading, the put/call ratio, the clock, and the feed status.

### The three charts

Each is a candlestick chart of that instrument, with its own price scale.

| Colour | Meaning |
|---|---|
| Green / red | A printed bar |
| Blue | The **next three projected** candles — brightest nearest |
| `┊` | The divider between what happened and what is expected |
| `░ ▒ ▓` | Projection confidence, from low to high |

Blue is never used for a printed bar and green/red are never used for a projected
one. That is enforced by test, not by convention.

The strip under each chart carries the numbers that decide whether the chart
matters: for the index, the conviction and a sparkline; for a leg, its delta,
theta per minute, implied vol, and the move the underlying must make for the leg
to pay for itself.

### Verdicts

One row per leg. `needs` is the requirement, `projected` is the index's projected
move over the same horizon, `edge` is the difference, `do` is the action. The
headline either names a trade or says plainly that nothing clears its own
breakeven. See [Options](options.md).

### Scrollable research grids

The strategy grid contains every discovered rule for index, call and put. It
shows ACTIVE, WAIT or N/A, the rule explanation, and measured three-bar hits, net
return and trust when outcomes exist. Use `j` and `k` to move through its rows.

The right panel changes with the numbered views:

The realtime AI scorecard (`4`) is the default. It shows the index learners'
matured hit/miss counts, accuracy, net result and trust while every chart carries its
current one-bar action and probability.

| Key | View |
|---|---|
| `1` | Frozen projected-candle results by leg and horizon |
| `2` | CALL and PUT scenarios for each of the next three bars: projected premium, gross move, estimated cost and net value per lot |
| `3` | Indicator state and plain-language readings such as `overbought`, `strong trend` and `above upper band` |
| `4` | Each online classifier's BUY/SELL/HOLD view, probability and trust, plus sample count, accuracy, rolling accuracy, P&L and drawdown |
| `5` | Actual-to-projected mini-graphs for every leg and a one-bar P(up) matrix for every online algorithm |

The full-screen dashboard uses internal row scrolling. `niftypulse research`
prints the durable scorecards as a sequence of regular tables when native
terminal scrollback is more useful.

---

## The single-instrument view

```bash
niftypulse dashboard --no-legs
```

```
╭──── NIFTY 50 · 1m · candles and next 3 projected ────╮
│  █▄████▄▄▄┊░░░ │  22,148.55                          │
│    │████ ██│ │  22,143.68                            │
│      ██│     ┊    │  22,138.81                       │
╰──────────────────────────────────────────────────────╯
╭──── forecasts ────╮╭─ projected candles — rebuilt every second ─╮
│  h  view  P(up)   ││  bar      at      close      move  confid… │
│  1m  ▲U   0.612   ││   +1   15:30  22,151.43    +1.9bp  ▮▮▮▮▮ │
╰───────────────────╯╰────────────────────────────────────────────╯
```

The **realtime auto-AI** panel is the default forecast: probability, confidence,
trust and BUY/SELL/HOLD research classification at each horizon. It learns when
exact future target bars close and does not require offline training. If batch
artifacts exist, their cost-aware forecast remains available as a fallback. The
**projected candles** panel is the continuously rebuilt price path, bar by bar.
The **prediction quality** card shows member agreement and the full member
probability range at each horizon. A strong headline with a wide model range is
visible disagreement, not false certainty.

---

## Refresh cadence

| What | How often | Flag |
|---|---|---|
| Ticks | as they arrive | — |
| Projected path | every second | `--nowcast` |
| Frame redraw | every second | `--refresh` |
| Model forecasts | on bar close | — |

The redraw is deliberately decoupled from the projection. `--refresh 0.25` gives a
smoother-looking screen while the projection still recomputes on its own second,
which is what the blue candles are promised to do.

---

## Offline mode

`--offline` replays a generated series **paced against the wall clock**: one
minute of wall clock per one-minute bar at `--speed 1.0`. That is deliberate —
a replay that finishes a trading day in two seconds gives the projected candles no
seconds to move in, which is the one thing they exist to do.

`--speed 60` makes a minute take a second, which is the practical setting for
watching the board work.

The first pass is re-based onto the current time so a session opens on the
present; later passes continue forward rather than rewinding, because a tick
stream that moves backwards is not something the aggregator is built to survive.

---

## Small terminals

Panels degrade rather than wrap. Three charts across a narrow terminal is where
the price axis gives way first: below roughly 130 columns the axis is dropped
entirely and the candle field takes the space, because a nine-column chart with
labels beside it is harder to read than a nineteen-column chart without them. A
test renders the board at four terminal sizes and asserts no line exceeds the
console width.
