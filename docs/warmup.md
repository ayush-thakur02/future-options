# Warm-up — being ready before the market opens

Every session used to open with the same two facts: nothing had been scored yet,
and no trust had been earned.

That is not a cosmetic problem. Trust is what gates action, and trust needs
matured outcomes — so the strategies ran at their plain prior weights and the
online learners returned HOLD for everything until enough target bars had closed
under them. The first hour of every day went on learning what the previous
session already knew.

The warm-up replays recent bars through the same ledgers the live loop writes to,
before the feed starts.

---

## What it does

Three stages, in the order the live loop runs them:

| Stage | What it fills in |
|---|---|
| **Strategies** | Each rule issues on the bars it would have fired on and is scored at its exact target bar, so it opens the day with a hit rate, a post-cost result and a trust score |
| **Online learners** | A forecast per bar, trained as each target closes, so their weights and their calibration are already earned |
| **Projections** | Recorded and scored, so the forward scorecard has a history behind it rather than a blank |

Measured on generated data, 600 bars per instrument across the index, call and put:

```
warming research (up to 600 bars per instrument)
    INDEX: 600 bars · 3,486 signals (3,469 scored) · 599 AI forecasts · 1 rules trusted · 4.4s
    CALL: 600 bars · 5,940 signals (5,908 scored) · 599 AI forecasts · 6 rules trusted · 5.6s
    PUT: 600 bars · 6,134 signals (6,092 scored) · 599 AI forecasts · 5 rules trusted · 5.6s
  1,800 bars replayed (cold start) · 6 strategies trusted · 15.8s
```

---

## Trust is computed as of each bar

This is the part that matters, and it is easy to get wrong in a way that looks
better than it is.

The number fed to a learner at bar *i* comes from the outcomes that had **matured
by bar i** — not from the scorecard of the whole window. Using the final scorecard
for every bar would be both unrealistic and a leak: a trust value that summarises
the window encodes the outcome of the very bar it is describing, and a learner
handed it can learn to read the answer.

`tests/test_warmup.py` asserts the consequence directly: in the opening bars of a
replay, nothing a rule does can have been scored, so trust there is exactly zero.

That second call site is why the trust arithmetic lives in `core/scoring.py`
rather than in either pack. The batch scorecard and the incremental replay have to
reach the same number, and two implementations of one formula is exactly where it
drifts.

---

## Two guards

**A signal is never issued across a session boundary.** The target bar has to be
exactly `h` bars ahead; a gap is a session close. Warming across one would
manufacture the overnight outcome the live loop deliberately never takes, and its
score would be a statement about a move nobody could have traded.

**Nothing is projected where its target falls outside the replay.** A projection
left pending past the end would be pruned as a missing bar when the first live bar
closed, and the forward scorecard would report gaps that never happened.

---

## The checkpoint

`data/research/warmup.sqlite3` records how far each instrument has been warmed,
per mode and bar size.

```
1,800 bars replayed (cold start)   15.4s
already current — nothing new      2.5s
```

That is what makes this get **better** over time rather than merely get ready. The
state persists across runs and every session adds to it, so a fresh install
replays the configured window once and every later start replays only what has
appeared since.

It is also the whole of the resume logic, which is worth stating plainly: deleting
the checkpoint replays the window again. The strategy ledger absorbs that
harmlessly — its rows are keyed by instrument, rule, issue time and target, so the
same signal is never counted twice — while the learners' ledger is append-only, so
a lost checkpoint costs a redundant training pass rather than a wrong one.

To start a learner over properly, delete its state under
`data/research/online_ai/`. To start the strategies over, delete
`data/research/strategies.sqlite3`. The two are independent, and so is the
checkpoint.

The `live` and `simulation` books never mix: a replay's outcomes must not appear
in the live record, and they are keyed separately.

---

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `--warmup` / `--no-warmup` | on | Replay before the session starts |
| `--warmup-bars` | `600` | Bars a **cold** instrument replays |

`SessionConfig(warmup=..., warmup_bars=..., warmup_projections=...)` is the same
set for library use.

A cold instrument replays `warmup_bars` bars, capped at the engine's own strategy
window — 600 bars, because that is all a rule is ever evaluated on live, and a
score series longer than the live one would be a different measurement wearing the
same name.

600 bars gives roughly 6× the 50 outcomes a scorecard needs for full evidence
weight on the learners, and a few hundred per rule on the strategies that fire.

---

## Making it affordable

A replay asks the learners for thousands of predictions in a row, which turned
three long-standing hot spots into an unusable startup cost. Measured over 600
bars, per instrument:

| | Before | After |
|---|---|---|
| Whole replay | 39.4s | 8.3s |

Each of the three is a scan replaced by a carried total, and each is as much a
live-loop cost as a replay cost — a session that runs for a week hits all of them:

- **Scorecards were rescanned on every issue.** Publishing a forecast got slower
  the more forecasts had been published. `MetricAccumulator` maintains every
  quantity `summarise` reports as records arrive, and `summarise` is now a thin
  wrapper over it so the streaming and batch paths cannot disagree — asserted over
  every prefix of a 300-record series.
- **The three-horizon fan-out predicted three times.** Every learner is
  deterministic and stateless between issues, so all three calls computed the same
  answer. `issue_horizons` computes it once.
- **The nearest-neighbour search re-standardised its window every call** — 160
  neighbours × 48 features × a square root each. The statistics do not depend on
  the neighbour, so they come out of the loop, and only the selected features are
  standardised at all.

The projections needed the same treatment: `atr_of` built three aligned Series and
a frame per call, once per projected bar per instrument, and is now array
arithmetic. Its trap is the first bar, which has no previous close —
`DataFrame.max(axis=1)` skips that NaN while `numpy.maximum` propagates it, so it
needs `fmax`. Each of these is guarded by a test that replays it against the
original implementation.

---

## The dashboard

The **System / cost state** panel carries four warm-up rows: the headline, whether
this run was a cold start or a resume, the bars and seconds, and the per-instrument
counts.

Without them on screen, a trust score that is not zero before the market opens
looks like it came from nowhere — and "resumed from checkpoint" is the visible
evidence that the thing is cumulative.

---

## What to expect

On generated data the warm-up finishes with only a handful of trusted rules and
learner trust near zero. **That is the correct result**, not a failure of the
replay: the generated series is a near-random walk, and a rule with no edge should
not earn trust however much evidence it is given.

The tests carry a positive control for exactly that reason — a deterministic
alternating trend is predictable three bars ahead, and the replay has to notice.
It does.

On live data the same machinery has real structure to find, and the record it
builds is scored at the exact target bars the live loop uses.

---

## Tests

`tests/test_warmup.py` — 21 tests:

- a cold replay earns evidence for every instrument, and the engine reads the
  outcomes back (without that step the replay happens and the ensemble never
  notices)
- no rule is trusted before its first outcome matures
- a series with real structure earns trust — the positive control
- a signal is never warmed across a session boundary
- a second run replays nothing and keeps what the first earned
- only the bars after the checkpoint are replayed
- a lost checkpoint replays the window again rather than silently doing nothing
- a replay with research disabled, or too little history, is skipped rather than
  raising
- a failing replay does not take the session with it
