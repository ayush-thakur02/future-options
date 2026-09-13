# Dashboard

A full-screen terminal UI built with [Rich](https://rich.readthedocs.io/).

```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│  NIFTY 50 · 1m   23,999.42  +7.54 (+0.03%)   regime: volatile   live                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────── NIFTY 50 · 1m · candles ───────────────────────────────────────────────╮
│                                                                                                ██  │    24,005.83    │
│                                                                                              ████▄ │    23,999.47    │
│                                                              ██▄█                         █▄▄██    │    23,993.11    │
│                                                       █▄▄█ █▄██ █                        │█        │    23,986.75    │
│                                                    █▄▄█  █▄█    █▄▄█                    █▄█        │    23,980.40    │
│                                                  █▄█               ██▄▄▄▄▄▄▄██▄██  ██  ██          │    23,974.04    │
│                                                 ██                 ██       ██ ███▄██▄▄█           │    23,967.68    │
│ █  ███                              █▄██▄█   █▄▄█                               ██                 │    23,961.32    │
│ █▄▄███ ██            █     │▄█ ██ █▄█ ██ █▄█ █                                                     │    23,954.96    │
│      █▄███ █▄▄█    █│██▄▄▄██ █▄██▄█│       █▄█                                                     │    23,948.60    │
│          █▄█  ██████▄██   ██                                                                       │    23,942.24    │
│                ████│                                                                               │    23,935.88    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────── scalp forecasts — does the move pay for the trade? ─────────╮╭───────────── active strategies ──────────────╮
│       h  view        P(up)  distribution       exp move     vs cost  ││  strategy                  view    strength  │
│      3m  ▼D          0.412  █████░░░░░░░         -8.0bp      +4.1bp  ││  squeeze_release           ▲       ▮▮▮▮▮▮▮▯  │
│      5m  ▼D          0.462  ██████░░░░░░         -8.5bp      +4.6bp  ││  supertrend                ▲       ▮▮▯▯▯▯▯▯  │
│ round trip  3.90bp  tradeable  5m DOWN (+4.6bp)                      ││  macd_momentum             ▲       ▮▮▯▯▯▯▯▯  │
╰──────────────────────────────────────────────────────────────────────╯╰──────────────────────────────────────────────╯
╭─────────────────────── indicator state ────────────────────────╮╭──────────────── session ─────────────────╮
│  indicator                     value  read                     ││ session                                  │
│  RSI(14)                        37.4  neutral                  ││   17:27:11 IST                           │
│  ADX(14)                        49.0  strong trend             ││                                          │
│  ATR norm                      2.8bp                           ││ cost hurdle                              │
│  MACD hist                   +1.62bp  above                    ││   3.90 bp round trip                     │
│  Stoch %K                       93.7  overbought               ││                                          │
│  Bollinger %B                   0.79  mid band                 ││ model members                            │
│  vs VWAP                      -35.8bp  below                    ││   3m  hist_gbm up 0.61                   │
│  Efficiency                     0.58  clean directional move   ││   5m  lightgbm up 0.61                   │
│                                                                 ││                                          │
│                                                                 ││ q quit   r refresh                       │
╰─────────────────────────────────────────────────────────────────╯╰──────────────────────────────────────────╯
```

---

## Panels

### Header

Symbol, timeframe, last price, change from previous close, detected regime, and
engine status.

**Regime** is one of `trending`, `choppy`, `volatile`, `normal`, or `unknown`,
derived from ADX rank and volatility rank over a trailing 100-bar window.
Labels are assigned causally — the volatility percentile uses only trailing data.

**Status** reflects the feed and engine:

| Status | Meaning |
|---|---|
| `initialising` | Bootstrap in progress |
| `warming up` | Fewer than 60 bars; indicators not yet available |
| `ready` | History loaded, waiting for ticks |
| `connecting` | Opening the WebSocket |
| `connected` | Socket open, awaiting first frame |
| `live` | Processing ticks |
| `reconnecting` | Connection dropped, retrying with backoff |
| `NORMAL_OPEN` etc. | Market segment status from the feed |
| `feed error: ...` | Connection failed |

### Candlestick chart

Up to ~100 bars, newest on the right.

| Glyph | Meaning |
|---|---|
| `█` | Candle body (open to close) |
| `▄` | Doji hairline (body is a single row) |
| `│` | Wick (high to low) |
| `│` (coloured) | Separator before the price axis |

Bodies and wicks are coloured: **green** for up bars, **red** for down. Wicks use
a dimmer shade of the same colour rather than a neutral grey — on a dense chart
most bodies are only one row tall, so without the tint you cannot tell direction
from the wick rows.

The footer sparkline summarises the same window and is coloured by net change.

**Right-alignment.** When fewer bars are available than the chart is wide, they
sit against the right edge so the newest bar is always in the same place. This is
easy to break and produces a chart that looks subtly wrong; see
[below](#a-note-on-chart-layout).

### Forecasts

The panel that answers the only question that matters: **does the move pay for
the trade?**

| Column | Meaning |
|---|---|
| `h` | Horizon in minutes |
| `view` | ▲ UP / ▼ DOWN |
| `P(up)` | Calibrated probability |
| `distribution` | Probability as a bar |
| `exp move` | Expected move in basis points, from the fitted conviction curve |
| `vs cost` | **Expected move minus the round-trip hurdle** |

`vs cost` is the decision column. **Positive and green means tradeable.** A
directionally correct forecast of a sub-cost move is a losing trade, so the
direction column is deliberately secondary.

The footer shows the round-trip hurdle and the best tradeable horizon, or
`no horizon clears cost`.

When no models are trained, the panel says so and points at `niftypulse train`.

### Active strategies

Strategies currently above the entry threshold, sorted by conviction strength.

`ensemble` is the blended view of all 18 rule-based strategies plus the ML model.
The individual rows are a sample of named strategies, so you can see *why* the
ensemble holds the view it does.

Disagreement between rows is informative — when the ensemble says UP but two
momentum strategies say DOWN, that is worth knowing.

### Indicator state

Eight indicators with their values and a plain-English reading.

| Indicator | Reading |
|---|---|
| RSI(14) | `overbought` ≥ 70, `oversold` ≤ 30 |
| ADX(14) | strong trend ≥ 40, trending ≥ 25, developing ≥ 18, else range-bound |
| ATR norm | Volatility as a fraction of price, in bps |
| MACD hist | Signed, in bps |
| Stoch %K | overbought ≥ 80, oversold ≤ 20 |
| Bollinger %B | Position within the bands |
| vs VWAP | Distance from session VWAP, in bps |
| Efficiency | clean directional move > 0.5, choppy < 0.25 |

### Session

Clock, **cost hurdle**, per-horizon model agreement, and key bindings.

"Model members" shows which base learner holds the strongest view for each
horizon, and in which direction. When members disagree, the ensemble probability
sits near 0.5 — which is itself a useful signal that the models do not know.

---

## Keys

| Key | Action |
|---|---|
| `q` | Quit |
| `Ctrl-C` | Quit |
| `r` | Refresh (currently a no-op — the dashboard refreshes on a timer) |

---

## Live versus replay

### Live

```bash
uv run niftypulse dashboard
```

Bootstraps from cached history, then streams live ticks over the Upstox v3
WebSocket. The engine reconnects automatically with exponential backoff capped at
30 seconds.

If no access token is available, this **falls back to replay** rather than
failing, and prints a notice.

### Replay

```bash
uv run niftypulse dashboard --offline
uv run niftypulse dashboard --offline --speed 0
```

Generates synthetic bars and streams them through the full pipeline.

`--speed` is the delay between bars, in seconds. The engine recomputes features
on every bar close, so replay is **compute-bound** at roughly 200ms per bar
regardless of `--speed`. Setting it to `0` removes the artificial delay but the
replay still takes about 30 seconds per 150 bars.

Live operation is unaffected — bars close once per minute, so 200ms is about 0.3%
of the interval.

---

## Changing the timeframe

```bash
uv run niftypulse dashboard --timeframe 5
```

> **Feature windows are expressed in bars.** A model trained on 1-minute bars
> cannot meaningfully be applied to 5-minute ones — its indicators have entirely
> different meanings. Changing the timeframe displays the 5-minute chart but the
> forecasts come from models trained on 1-minute data, which is a mismatch.

To run at another timeframe properly, refetch and retrain with
`bar_minutes: 5` in configuration.

---

## A note on chart layout

The chart is rendered onto a character grid, then colourised by grouping runs of
identically styled characters. Emitting Rich markup per character would produce a
string many times the size of the visible output and make the dashboard lag on a
busy feed.

Two failure modes are worth knowing about, because both render as "the chart
looks broken" and neither is obvious from the code:

**Losing the left pad.** A shorter series is right-aligned by filling the left of
the grid with blanks. Right-stripping each style run deletes that leading blank
run, so every candle jumps to the left edge.

**Overflowing the panel interior.** A row is
`candles + separator + price axis`. If that exceeds the panel's *interior* width
— borders and padding included — Rich wraps it, and the price labels land on
their own lines interleaved with the candles. Note this does **not** exceed the
*console* width, so a naive width check misses it entirely.

Both are covered by `tests/test_ui.py`, and the guards were verified to fail
against the original broken code:

- `test_grid_preserves_leading_blanks`
- `test_render_candles_right_aligns_short_series`
- `test_chart_axis_is_on_every_candle_row` — parametrised across four terminal
  sizes

---

## Performance

| Stage | Cost |
|---|---|
| `build_features` (2,000 bars) | ~50 ms |
| `_collect_signals` (24 strategies, 600-bar tail) | ~150 ms |
| Render | ~3 ms |
| **Per bar close** | **~204 ms** |

The strategy layer is evaluated on a 600-bar tail rather than the full 2,000-bar
feature window. Strategies only report their latest reading, the indicators they
consume are already computed over full history, and the longest lookback any
strategy applies internally is about 100 bars. Evaluating all 24 of them across
the full window cost roughly 460ms and dominated the loop.

---

## Extending

### Add a panel

```python
layout["middle"].split_row(
    Layout(name="predictions", ratio=3),
    Layout(name="signals", ratio=2),
)
```

Add a `Layout`, then update it in `build()` with a Panel. The layout is rebuilt
each refresh — at a few rows of state the cost is negligible, and it avoids a
class of stale-panel bugs that incremental updates invite.

### Add an indicator

Add the key to the `spec` dict in `_indicators`, with a value formatter and an
interpretation function:

```python
"my_indicator": ("Label", lambda v: f"{v:.2f}", _read_my_indicator),
```

The value must be exposed in `MarketSnapshot.indicators` by
`LiveEngine.snapshot()`.

### Test it

Any new panel needs a width guard. Use the parametrised width tests in
`tests/test_ui.py` — an off-by-one in row arithmetic is invisible until it wraps.
