# Dashboard

Two layouts, one clock. The **board** puts the call and the put beside the index;
`--no-legs` gives the single-instrument view with the strategy and forecast panels
where the chart space was.

```bash
niftypulse dashboard                  # live websocket, board, recording as it runs
niftypulse dashboard --offline --speed 60
niftypulse dashboard --no-legs --refresh 0.25
niftypulse snapshot                   # one frame to stdout, then exit
```

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

| Key | View |
|---|---|
| `1` | Frozen projected-candle results by leg and horizon |
| `2` | CALL and PUT scenarios for each of the next three bars: projected premium, gross move, estimated cost and net value per lot |
| `3` | Indicator state and plain-language readings such as `overbought`, `strong trend` and `above upper band` |
| `4` | Each online classifier's BUY/SELL/HOLD view, probability and trust, plus sample count, accuracy, rolling accuracy, P&L and drawdown |

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

The **forecasts** panel is the trained model: probability, expected move, and
whether the move clears the round trip. The **projected candles** panel is the
path, bar by bar, with the hit rate the projection has scored so far. They are
different objects and they disagree by design — one is a calibrated probability on
a fixed horizon, the other a live path that includes the last twenty seconds.

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
