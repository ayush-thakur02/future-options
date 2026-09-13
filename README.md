# NIFTY Pulse

A scalping platform for NIFTY 50. Streams live ticks from Upstox, runs nineteen
technical strategies and a trained model, projects the **next three candles every
second**, and charts the call, the index and the put side by side — with
transaction costs treated as the primary constraint rather than an afterthought.

Everything runs locally. No data leaves your machine except the Upstox API calls.
Everything that arrives is stored, partitioned so it is never fetched twice.

---

## The number that decides everything

Scalping is a cost game before it is a prediction game. At a typical NIFTY round
trip of roughly **3.9 basis points**, a signal is only worth acting on if the move
it forecasts is *larger* than that. This platform measures the consequence
directly, and the answer is uncomfortable:

```
Cost hurdle — round trip 3.90 bps at Rs 2,000,000
┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ horizon ┃ median move ┃     p75 ┃     p90 ┃ clears hurdle ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│      1m │     1.30 bp │ 2.25 bp │ 3.31 bp │          5.7% │
│      2m │     1.80 bp │ 3.13 bp │ 4.65 bp │         16.0% │
│      3m │     2.20 bp │ 3.85 bp │ 5.72 bp │         24.3% │
│      5m │     2.87 bp │ 4.99 bp │ 7.35 bp │         36.3% │
└─────────┴─────────────┴─────────┴─────────┴───────────────┘
```

On the 1-minute horizon fewer than **6% of bars** produce a move large enough to
pay for the trade. That is the hard ceiling on how often a 1-minute scalp can
profit, and no model, however good, can lift it.

So the platform acts on it:

- **Training refuses to learn untradeable moves.** Labelling requires a forward
  move above the cost hurdle, so the model is never taught to classify noise it
  could not have traded. A horizon where too few bars clear the hurdle is
  **skipped with an explanation**, not trained into a money-losing model.
- **Signals carry a cost gate.** A forecast stays silent unless the move it
  historically preceded is large enough to cover the round trip.
- **Expected move is measured, not assumed.** The conviction-to-move curve is
  fitted from out-of-sample predictions, so "expected move" reflects what the
  model actually demonstrated rather than a volatility heuristic.
- **The dashboard shows the cost decision first.** Every horizon displays its
  expected move next to its edge after cost, and says plainly when nothing is
  worth trading.

Most of the time the correct output is **no trade**. That is the design working.

## Install

Python 3.11+ is required.

```bash
uv sync --all-extras
```

On macOS, LightGBM links against OpenMP and needs it present:

```bash
brew install libomp
```

Without it LightGBM is skipped automatically and the ensemble trains on the
remaining three learners.

## Authenticate with Upstox

Create an app at <https://account.upstox.com/developer/apps>.

```bash
cp .env.example .env      # fill in your credentials
uv run niftypulse login   # opens the Upstox login, stores the token
```

Tokens expire at 03:30 IST the next day, so `login` is a once-per-morning step.

**No credentials yet?** Every command accepts `--offline` and runs on generated
data, so you can exercise the whole pipeline first.

## Usage

```bash
uv run niftypulse doctor                  # check environment and credentials
uv run niftypulse fetch --days 180        # download 1-minute history
uv run niftypulse train                   # train the scalping horizons
uv run niftypulse dashboard               # live terminal dashboard

uv run niftypulse strategies              # list strategies and how often they fire
uv run niftypulse models                  # list trained artifacts and metrics
uv run niftypulse backtest --strategy ensemble --horizon 5
uv run niftypulse backtest --strategy ml --horizon 5 --sweep
```

## Tuning for your costs

The hurdle is only as good as the cost inputs. Adjust these in
`config/default.yaml` before drawing conclusions:

```yaml
model:
  reference_notional: 2_000_000   # position size used to derive the hurdle
  lot_size: 75                    # verify against the current NIFTY spec
  hurdle_multiple: 1.5            # margin required above break-even
  enforce_cost_hurdle: true       # set false to study signal without the gate
```

`niftypulse backtest --slippage` and `--notional` override them for a single run.
Raising slippage by a basis point moves the hurdle more than most modelling
choices will.

## Modelling notes

**Labels are dead-banded, with the hurdle as a floor.** The ATR-scaled band
(`deadband_atr`) sets the baseline, and the cost hurdle raises it where costs are
the binding constraint — which on scalping timescales is almost always.

**Validation is purged and embargoed.** Overlapping label windows leak between
adjacent folds and inflate accuracy in a way that never shows up in testing. Each
fold purges training samples whose label window reaches into the test set and
embargoes a further 20 bars. Follows López de Prado, *Advances in Financial
Machine Learning*, ch. 7.

**Calibration is gated twice.** Platt scaling is applied only if it improves the
Brier score by a meaningful margin on a chronological holdout *and* does not
collapse the score spread. Brier alone is a trap: when a model has no real signal,
mapping everything toward the base rate always improves Brier while flattening
the distribution a threshold-based strategy needs to act on.

**Execution enters on the next bar's open.** A signal computed from a bar's close
cannot be filled at that close. Trades that would cross the session close are
skipped rather than carried overnight. Positions size in whole lots.

## Reading the results

Short-horizon index direction is close to unpredictable, and a model reporting
55% accuracy on the next 5-minute candle is usually reporting a leak rather than
an edge.

- **Lift, not accuracy.** The training report shows accuracy *minus the
  majority-class baseline*. On a dataset that is 53% UP, predicting UP always
  scores 53%.
- **Confidence buckets.** Accuracy by model confidence. If it does not rise with
  confidence, the probabilities carry no information about *when* to trade.
- **Gross and net, always.** A strategy with positive gross expectancy and
  negative net expectancy is the normal outcome, and the report is shaped to make
  that obvious.
- **Threshold sweeps.** `--sweep` raises the entry threshold. Net expectancy
  should improve as it rises. If it does not, the signal says nothing about its
  own confidence.

Run on `--offline` data you will see accuracy at the base rate and AUC near 0.50.
**That is the correct result**, not a bug: the generated series is a near-random
walk with no persistent structure. Its value is confirming the pipeline is wired
correctly. The tests include a positive control that injects a deterministic
pattern and asserts the pipeline learns it, so "found nothing" is distinguishable
from "broken".

## Documentation

Full documentation is in [`docs/`](docs/README.md).

| If you want to... | Read |
|---|---|
| Get it running | [Getting started](docs/getting-started.md) |
| Understand why it behaves as it does | [Scalping economics](docs/scalping-economics.md) |
| See how the pieces fit | [Architecture](docs/architecture.md) |
| Add or replace a component | [Plugins](docs/plugins.md) |
| Know what the blue candles are | [Projection](docs/projection.md) |
| Understand the call/index/put board | [Options](docs/options.md) |
| Know what is on disk | [Storage](docs/storage.md) |
| Look up a command | [CLI reference](docs/cli-reference.md) |
| Look up a setting | [Configuration](docs/configuration.md) |
| Interpret a training run | [ML pipeline](docs/ml-pipeline.md) |
| Read the dashboard | [Dashboard](docs/dashboard.md) |
| Fix something | [Troubleshooting](docs/troubleshooting.md) |

Reference docs also cover the [data layer](docs/data-layer.md),
[features](docs/features.md), [strategies](docs/strategies.md),
[backtesting](docs/backtesting.md), and [development](docs/development.md).

## Project layout

Three layers, and one rule: `core` knows nothing, `kernel` knows plugins,
`plugins` know their job, `runtime` decides which ones to use.

```
src/
├── core/          # domain types, settings, calendar, the OHLCV schema
├── kernel/        # discovery, capability wiring, the event bus
├── plugins/       # everything that does work, as small packs
│   ├── sources/       upstox · simulated · history · option_chain
│   ├── aggregators/   candle_builder
│   ├── features/      technical
│   ├── strategies/    trend · momentum · reversion · volatility · ml_forecast
│   ├── forecasts/     ml_ensemble · projection
│   ├── advisory/      breakeven_gate
│   └── renderers/     terminal
├── runtime/       # session, engine, board, nowcast, conviction, bars
├── backtest/      # costs, execution, reporting
└── cli.py

docs/                      # full documentation, see docs/README.md
tests/                     # 387 tests
config/default.yaml        # settings
data/                      # partitioned store: candles, ticks, chain
```

**Everything is a plugin.** A capability is added by dropping a folder with a
`plugin.py` into the tree — no registry to edit, no factory to extend. Plugins
link through declared capabilities rather than imports, so any one can be replaced
by another that provides the same name. `niftypulse plugins` lists what is wired
to what: **15 packs, 30 capabilities**. See [Plugins](docs/plugins.md).

## Tests

```bash
uv run pytest
```

387 tests covering indicator correctness, feature causality, split purging,
execution timing, position sizing, cost accounting, the cost hurdle, plugin
discovery and capability wiring, option pricing identities, projection geometry
and its scoreboard, storage partitioning, terminal rendering, and the ML pipeline
end to end.

Some are worth knowing about:

- `test_features_do_not_use_future_data` truncates the input and asserts features
  on the overlapping rows are unchanged. Any indicator that peeks forward fails.
- `test_pipeline_learns_deterministic_pattern` and
  `test_pipeline_finds_no_edge_in_random_walk` bracket the ML pipeline: it must
  find structure when it exists and must not invent it when it does not.
- `test_chart_axis_is_on_every_candle_row` renders the dashboard at four terminal
  sizes and asserts no row wrapped. The bug it guards did not exceed the console
  width, only the panel interior — so a naive width check would have missed it.
- `test_hurdle_makes_short_horizons_untrainable` asserts a horizon whose moves
  cannot cover costs is refused rather than trained into a losing model.
- `test_every_instrument_gets_its_own_projection` asserts the three charts on the
  board do not share one forecaster. They would have, if the kernel's memoisation
  had not been bypassed — and the failure looks like three charts moving in
  perfect lockstep rather than like a bug.
- `test_leg_premium_decays_when_the_index_does_not_move` holds spot flat and
  asserts the premium falls across a session, which is the strongest single check
  that the options clock counts trading time rather than wall-clock time.

## Things that trip people up

**The index has no volume and no order book.** NIFTY 50 is an index; the exchange
publishes no traded volume or depth for it. Volume features fall back to tick
count, and the order-flow strategies return no opinion. **For real scalping,
subscribe to the front-month NIFTY future instead** — set `instrument_key` to its
`NSE_FO|...` key and you get depth, traded value, and open interest, which is
where most short-timescale edge actually lives.

**Lot size sets your minimum position.** NIFTY near 24,000 with a 75-unit lot
means the smallest possible trade is about ₹1.8 million of notional. Any smaller
target is unreachable, and the backtest floors at one lot rather than pretending
otherwise.

**Changing `bar_minutes` invalidates trained models.** Feature windows are in
bars, so a model trained on 1-minute bars cannot be applied to 5-minute ones.

**Weekly expiry day is configurable.** NSE moved NIFTY weekly expiry from Thursday
to Tuesday; it is `expiry_weekday` in `config/default.yaml`.

**Ticks and chain history cannot be re-fetched.** The API publishes candles, not
the tape that produced them. Both are recorded while a live session runs, so
whatever is not recorded is gone. Everything else is pulled once and kept.

**An at-the-money leg needs 60–100bp to pay for itself** — premium, costs and
theta — against a median 1-minute move of 1.3bp. That is why the board says *no
trade* most of the time, and it is the correct output rather than a failure of the
search.

## Disclaimer

This is research software. It places no orders and is not connected to any
broker's order routing. Nothing here is investment advice.

Note also that algorithmic trading in India is regulated by SEBI, with
registration and order-tagging requirements that apply as soon as you connect to
a broker's order API. This platform deliberately stops short of that.
