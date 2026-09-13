# Options — calls, the index and puts

The board charts three instruments on one clock: the index, the at-the-money
call, and the at-the-money put. Under them sits a verdict per leg with the
arithmetic that produced it.

Packs: `source:option_chain`, `advisory:breakeven_gate` ·
Capabilities: `option_chain`, `advisory`

---

## Why options are in a scalping platform

Two reasons, and the first is structural.

**An index has no volume and no order book.** NIFTY 50 is an index; the exchange
publishes no traded volume and no depth for it. The flow-based strategies —
`order_flow`, `activity_spike` — return no opinion on the index and fire on 0% of
bars. That is correct behaviour, not a broken strategy. **An option leg has both
volume and open interest**, so adding the legs switches those strategies back on.

**An option makes the platform's central question explicit.** The platform has
always argued that a signal is only worth acting on when the move it forecasts
clears the cost of capturing it. A long option is the purest form of that: you pay
the premium up front, delta converts a move in the underlying into premium, and
theta bills you for every minute you hold.

---

## The number that decides it

```
required move (bps) = (premium + round-trip cost + theta × horizon)
                      ÷ (|delta| × spot) × 10,000
```

Measured on the board at a 24,000 index, ATM strike, 12% implied vol:

| Time to expiry | Premium | Delta | θ/min | Required move |
|---|---|---|---|---|
| 1 day (375 min) | ₹72.38 | +0.502 | −₹0.097 | **60 bp** |
| 4 days (1,500 min) | ₹121.82 | +0.504 | −₹0.040 | **100 bp** |
| Index / future | — | 1.00 | — | **3.9 bp** (round trip) |

So the requirement is roughly **60–100 bp** for a long ATM leg, depending on how
much time is left — against a measured median 1-minute move of **1.3 bp** and a
3-minute projection usually under 10 bp. That is the finding, and it is why the
panel's default answer is *no trade*: a long ATM leg on a scalping horizon needs
a move roughly fifty times the median.

Theta is what makes the shorter-dated case worse than it looks: a one-day ATM
option loses about ₹0.10 a minute, which over a three-minute horizon is the same
order as the entire median move.

The panel prints the requirement next to the projection, so a reader can disagree
with the inputs rather than the conclusion.

---

## Pricing

`plugins/sources/option_chain/pricing.py` — Black-Scholes with two deliberate
simplifications, stated rather than hidden:

- **The risk-free rate is zero.** On a same-day or next-day NIFTY option the carry
  term is worth a small fraction of a basis point of premium, and an invented rate
  would be a guess dressed up as a parameter.
- **Time is measured in trading minutes.** An option does not decay at the
  weekend, and a chain priced off the wall clock would show Friday's theta spread
  across a week of closed hours. `trading_minutes_to` counts only session time.

Both were verified by test: put-call parity holds at zero rates, delta is bounded
and meets ±0.5 at the money, theta is negative and worsens as expiry approaches,
and a flat index produces a *falling* premium across a session — which is the
strongest single check on the clock.

---

## The chain

`plugins/sources/option_chain/chain.py` builds the strikes around the money:

| Field | Notes |
|---|---|
| Premium, delta, gamma, theta/min, vega | Per strike, per side |
| Implied vol | Parabolic smile in moneyness: wings richer than the middle |
| Open interest | Peaked at the money, calls leaning above spot, puts below |
| Put/call ratio | From aggregated open interest |

The open-interest shape is not decoration: a symmetric blob would make PCR and
positioning readings meaningless, and calls-above/puts-below is where writers and
hedgers actually sit.

**Leg candles** are priced off the index bars beside them — each index bar's open,
high, low and close re-priced at the same implied vol — so the premium candle is a
monotone transform of the index candle and the two charts cannot tell different
stories. A test asserts the premium and the index agree on direction on more than
90% of bars.

### Live chains

Everything above is currently computed rather than read from an account. The live
chain is the same shape from the REST API — expiries, then
`get_put_call_option_chain(instrument_key, expiry_date)` per strike, with
`market_data` (ltp, volume, oi, prev_oi, bid/ask) and `option_greeks` (delta,
gamma, theta, vega, iv, pop). Adding it is a reader module in the same pack plus
the `broker` requirement; the leg interface does not change.

The API surface, verified against the official SDK, is in
`.agents/skills/upstox/references/option-chain.md`.

---

## The verdict

`plugins/advisory/breakeven_gate/gate.py` turns the board into per-leg actions.
Three outcomes, and the conservative one is the default:

| Action | When |
|---|---|
| `LONG` | The leg is on the side of the projection **and** the projected move clears the requirement |
| `SHORT` | The leg is on the *opposite* side, implied vol exceeds realised by 15%, and the strike is at least one step out of the money |
| `FLAT` | Everything else — which is almost always |

`SHORT` is deliberately narrow. Selling premium has unbounded risk, so it is only
suggested when the seller is being paid for that tail (IV richness), the
projection says the move is not coming, and there is a premium cushion between
spot and the strike. At the money the gate refuses to write it at all.

Longs are ranked by edge and reported first. A short's edge is **not** a negative
long's — it profits from travel that does not happen — so ranking the two together
by `edge_bps` would quietly pick the short every time.

The headline reads either `<LEG> LONG — edge +Nbp (reason)`, `<LEG> SHORT — IV
over realised, projection points away`, or:

```
no leg clears its own breakeven — the correct output here is no trade
```

---

## Strikes and rolling

Both legs take the same strike, so the call and the put are comparable rather than
silently one step apart. When spot drifts more than **two strikes** from a leg,
that leg is re-struck: its premium history is regenerated from the index history
at the new strike, its projected path is cleared, and its scoreboard is reset. A
roll is a new instrument, and scoring a projection made about one strike against
another strike's bars would be a quiet lie.

A leg two strikes from the money is no longer the trade the board is about — it is
a directional bet with a small delta.

---

## Reading the panel

```
leg            do       needs   projected      edge  why
INDEX          ▲F          4bp      +0.3bp      -4bp  needs 3.9bp, projected 0.3bp
CALL 24,050    ▲F        60bp       +0.3bp     -60bp  needs 60bp, projected 0bp
PUT 24,050     ▲S        62bp       +0.3bp     -62bp  IV 10.0% over realised 5.6%, ...
```

- **needs** — the move the underlying must make for this leg to pay for itself,
  costs and theta included
- **projected** — the index's projected move over the same horizon
- **edge** — the difference; negative means the leg cannot pay for itself
- **do** — the action and the direction the projection points

The chart strip above the verdicts carries the same leg's live greeks: delta,
theta per minute, implied vol, and the requirement in basis points.

---

## Costs

The gate charges 1.2% of premium round trip by default
(`DEFAULT_COST_RATE`): brokerage both ways, exchange charges, STT on the sell leg,
GST, and a spread — which on an option is usually the largest single item.
Deliberately pessimistic, and configurable:

```yaml
plugins:
  advisory:breakeven_gate:
    cost_rate: 0.015
    horizon_bars: 3
```

`backtest/costs.py` already models Indian derivatives properly (STT on the sell
leg, exchange transaction charges, GST on brokerage and exchange fees, SEBI
turnover fee, stamp duty on the buy leg, flat brokerage per order). An options
schedule differs in one respect — STT is charged on **premium** rather than on
notional — which is the piece to add before backtesting a verdict.

---

## What this board is not

It is not a spread selector, and it does not suggest multi-leg structures. Single
legs are what the arithmetic above speaks to directly, and the honest answer for
most of them is that they cannot pay for themselves. A spread — sell the wing the
projection says will not be reached, buy the one it might — is where the same
analysis would point next, and it belongs in a second advisory plugin rather than
in this one.
