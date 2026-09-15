# Money simulator — a funded book, scored in rupees

Everything else in the advisory layer says what a leg is *worth doing*. This says
what it costs to have done it, at a fixed lot, against the tape.

Pack: `advisory:money_simulator` · Capability: `money_simulator`

It is not a strategy and not a forecast. It has no opinion the platform does not
already have; it consumes the strategy ensemble, the online learners, the batch
model and the projected path, and answers a different question about them. If the
platform is right about the move being there, what happens to money?

---

## The book

Three legs, each with its own wallet, each trading **exactly one lot**.

| | Index | Call | Put |
|---|---|---|---|
| Instrument | `NSE_INDEX|Nifty 50` | ATM call | ATM put |
| Opening balance | ₹25,000 | ₹25,000 | ₹25,000 |
| Lot | 65 units | 65 units | 65 units |
| Paid on | index points | premium | premium |

Plus a shared **reserve of ₹2,00,000**, so ₹2,75,000 is committed in total.

### The wallet holds risk capital, not notional

A 65-unit index leg at 24,000 is about **₹1.56 million of exposure** and it never
appears in the balance. What enters the wallet is the rupee-per-point the position
is worth — ₹65 a point — and the wallet is what absorbs being wrong about it.

```
wallet   ₹25,000          position  1 lot × 65 units
                          worth     ₹65 per index point

  24,000 → 24,020    +₹1,300
  24,000 → 23,980    −₹1,300
```

This is the only model in which the index leg can trade at all. A cash account
that had to fund ₹1.56 million of notional would refuse every trade at these
balances, and the reserve would be decoration. The choice is stated rather than
hidden because it is the single assumption the whole panel rests on: the numbers
are what a 65-unit position makes, not what a fully funded one would.

### Why ₹25,000

Sized so a leg survives a normal bad session before it has to borrow. A 65-unit
index lot moves about **₹12,000 on a 190-point day**, and an at-the-money premium
is roughly **₹7,000–10,500 a lot**. ₹25,000 absorbs a bad session and a bit more;
₹10,000 per leg would have the index wallet borrowing after a single stop.

Set `opening_balance` and `reserve` in
`config/plugins/advisory/money_simulator.yaml` to whatever you want to study.

---

## The decision

On each decision tick the policy blends **every source the platform already
computes** into one signed view per leg.

| Source | What it contributes | Weight |
|---|---|---|
| `strategies` | the rule ensemble's blended score, each rule weighted by its measured trust | 1.0 |
| `online_ai` | the online learners' P(up) edge, shrunk by their trust score | 0.9 |
| `model` | the batch ensemble's edge, averaged over the horizons it reports | 0.6 |
| `projection` | the projected path over the horizon, squashed to [−1, 1] | 1.2 |
| `momentum` | the last seconds of tape | 0.4 |

The blend is renormalised over the sources **actually present**. Offline, with no
trained artifacts, the four remaining sources read as a full opinion rather than
as three quarters of one — the alternative silently dilutes every reading by the
weight of a model that is not there.

Each contribution is kept alongside the total and shipped to the panel, so a
decision can be argued with. Click the view line on any wallet.

### A leg's view is not the index's view

A call gains when the index gains. A put gains when the index falls. The sign is
carried as `index_beta`, and it is why a put that every source agrees on reads as
unanimous agreement rather than unanimous dissent — contributions are compared in
**index** space, never against the leg's own view.

The same distinction applies to the projection. An option leg cannot report the
underlying's path: its own projected series is a levered premium path that moves
by whole percents a bar. Read as though it were the index it would announce a
five-percent move in the underlying every three minutes. The board passes the
index engine's path to every leg.

---

## The edge, and how it differs from the gate

[Options](options.md) documents the breakeven gate:

```
required move (bps) = (premium + round-trip cost + theta × horizon)
                      ÷ (|delta| × spot) × 10,000
```

That asks whether the option can **recover the premium it was bought for** — a
240-point move on a 120-rupee ATM premium. For a position held to expiry, that is
the right question.

A scalp does not hold to expiry. It buys at 120 and sells at 123, and the 120
comes back. So the simulator asks the round-trip question instead:

```
expected_move_bps = the projection, restated in the leg's own price
required_move_bps = what one round trip costs
edge_bps          = |expected_move_bps| − required_move_bps
```

**Both are correct answers to two different questions**, and the simulator shows
its own rather than borrowing the stricter one. At a 24,000 index with a 3-bar
projection of 8 bp:

| | Index | ATM call, ₹72 premium, Δ 0.54 |
|---|---|---|
| Projected move | 8.0 bp of index | ~890 bp of premium |
| Round trip | ~4.0 bp | 120 bp spread + ~30 bp decay |
| **Edge** | **+4.0 bp ≈ ₹620** | **+740 bp ≈ ₹355** |

An option is a levered claim on the same move, and quoting both legs in index
basis points would make the option look like a worse index position rather than a
different instrument. The gate's warning stands: the projection has to be *right*,
and the platform's own scoreboard is the place to check whether it has been.

For options the hurdle includes **the decay paid while held**. Spread alone would
quote a hurdle a position clears by simply refusing to wait.

### The entry gates

An entry needs all of:

1. `|view| ≥ entry_view` (0.15, the same threshold the strategy panel calls ACTIVE)
2. the projection to point the same way as the blend
3. at least `min_confirmations` sources leaning the same way
4. `edge_bps ≥ min_edge_bps` (0.0 — the round trip must be expected to pay)
5. no cooldown running from the leg's last close

Each refusal is recorded with its own reason, and the binding one is reported
rather than the first that happened to fail.

---

## The reserve

The reserve is what keeps a drawn-down leg investing.

```
wallet cash < opening × min_equity_fraction (0.5)
        ↓
borrow (opening − cash) from the pool, capped by what is left
        ↓
the leg can fund another lot
```

Repayment runs the other way and is deliberately ordered: whenever a leg is flat
and its cash exceeds its opening balance, **the excess goes back to the pool until
the draw is cleared**. Profits count as the leg's own only after the reserve has
been made whole. That makes the reserve temporary in fact and not only in name:

- a leg that recovers returns what it borrowed,
- a leg that keeps losing **exhausts the pool**, and the book stops inflating —
  the reserve is finite and there is no second reserve.

Watch `reserve.remaining` and `wallets[].reserve_drawn` on the panel: net P&L that
is positive while the reserve is drawn is a book that has not yet paid for itself.

---

## Two clocks

The split is the point.

| | Cadence | Why |
|---|---|---|
| Risk — stops, targets, ruin | every refresh (~1 s) | a stop that is only tested once a minute is not a stop, it is a hope |
| Entries | `decision_seconds` (30 s) | an entry is a judgement about the next few minutes; re-taking it every second buys nothing but noise |

Exits, in priority order: **stop → target → time → reversal → session close**.

- Stops and targets are **resting orders and fill at their level**. Filling at
  whatever the next refresh happened to see would let a two-second gap turn a
  20-point stop into a 300-point rout and make every stop look like a catastrophe.
- `time` and `reversal` are market orders and fill at the tape.
- A reversal waits out `min_hold_seconds`; a stop does not.
- Nothing is carried overnight — the session close flattens everything.
- A leg whose equity reaches zero is closed and then topped up on its next flat
  step, which is the "recover the loss so it can keep going" case.

---

## Costs

Costs are the reason the book is interesting, so they are charged both ways and
never netted into the price.

**Index** — through the platform's own `CostModel`, the same model that sets the
cost hurdle. On a ₹1.56 million lot:

| Leg | bps | Rupees |
|---|---|---|
| Entry (buy) | 1.085 | ₹169 |
| Exit (sell) | 2.885 | ₹450 |
| **Round trip** | **3.97** | **₹619** |

The two sides are deliberately not the same price: **STT falls on the sell leg**
at 2 bp and stamp duty on the buy leg at 0.2 bp. Slippage is charged per side.

**Options** — `option_cost_rate` (1.2%) of the premium, round trip, split evenly
across entry and exit. A futures basis-point charge would be meaningless on a
₹7,800 premium: spread, STT on the sell leg and charges are a fraction of the
premium, not a fraction of the index. The rate matches the breakeven gate's, kept
separately so neither pack depends on the other.

---

## Persistence

Trades, wallet balances and the reserve live in
`data/research/money_simulator/<mode>.sqlite3`, written after **every close**
rather than at shutdown, so a kill −9 costs at most the position that was open.

`mode` is `live` or `simulation` and the two books never mix: a replay's losses
must not appear in the live record. Wallet state is restored on the next start,
so the headline figure is a property of the trading and not of how often the
laptop was rebooted.

The book does **not** auto-flatten on restart — an open lot is simply not carried
across, and the next decision re-enters if the view still justifies it.

---

## The panel

Dashboard section **02**, above the charts.

- Four summary metrics: paper equity, net result, reserve left, lot size.
- One card per leg: equity and P&L, the open lot with its stop and target, the
  view line, and win rate / costs paid / reserve used / drawdown.
- **Click the view line** for the full calculation — every source, its value, its
  weight and its weighted contribution, next to the projected move, the round-trip
  hurdle and the resulting edge.
- Two tables: the decision log (including every refusal and its reason) and the
  closed trades.

---

## Configuration

`config/plugins/advisory/money_simulator.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `opening_balance` | `25000.0` | risk capital per leg |
| `reserve` | `200000.0` | the shared pool |
| `lot_size` | `65` | units per lot, all three legs |
| `decision_seconds` | `30.0` | entry cadence; risk is checked every refresh |
| `horizon_bars` | `3` | forward horizon the projection is read over |
| `entry_view` | `0.15` | minimum blended view |
| `min_edge_bps` | `0.0` | expected move must beat the round trip by this much |
| `min_confirmations` | `1` | sources that must lean the way the trade does |
| `min_equity_fraction` | `0.5` | drawdown at which a leg borrows |
| `stop_atr` / `target_atr` | `1.5` / `2.5` | stops and targets in leg-own ATRs |
| `fallback_stop_bps` | `10.0` | used until the ATR warms up |
| `max_hold_minutes` | `5.0` | give up on the thesis after this long |
| `min_hold_seconds` | `45.0` | a reversal waits this out |
| `cooldown_seconds` | `45.0` | no re-entry this soon after a close |
| `reversal_view` | `-0.15` | view crossing this closes an open lot |
| `option_cost_rate` | `0.012` | round trip as a fraction of premium |
| `projection_scale_bps` | `4.0` | the move that reads as a fully-convicted projection |
| `weights.*` | see above | how much each source counts |
| `state_path` | `null` | override the ledger location |

Stops and targets are ATR multiples so they scale with the instrument rather than
being a fixed distance that means something different on the index than on a
premium. `lot_size` here is the simulator's own; the platform-wide
`model.lot_size` still governs the cost hurdle and backtests.

---

## What to expect

The honest expectation is set by the platform's own numbers, not by the panel's
green ink.

The index round trip is **3.97 bp** and the median 3-minute move is **2.20 bp** —
the median move does not pay for the trade. Clearing ~4 bp happens on roughly
**24% of 3-minute bars**. So the index leg trades selectively and loses on the
majority of the trades it takes, which is the same finding the cost-hurdle table
in the main README reports. An option leg, being levered, clears its own hurdle
far more often, and its problem is different: it is levered in both directions.

The simulator exists to make that measurable in rupees over real arriving data
rather than argued about in basis points. If the book is down, the useful reading
is not "the simulator is broken" — check whether the projection has been right
(the Forward score and Realtime Auto-AI panels), and whether the costs paid exceed
the gross made.

**Nothing here places an order.** It is a bookkeeping layer over the research
platform, and the platform is not connected to any broker's order routing.

---

## Tests

`tests/test_money_simulator.py` — 30 tests, aimed at the failures that produce a
plausible number rather than an error:

- a point on the index staying worth exactly ₹65, and the wallet never holding the notional
- a stop filling at its level, not at the next print
- the exit leg costing more than the entry leg, because STT falls on the sell
- a put reading as unanimous agreement rather than dissent
- a premium's own projection being ignored in favour of the underlying's
- an option hurdle that includes the decay paid while held
- the reserve topping a wallet up, being repaid before profits belong to the leg,
  and running out rather than going negative
- live and simulated books staying separate in one ledger file
