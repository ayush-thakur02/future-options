"""Forecast packs: a view in, an expectation out.

A forecaster answers two different questions, and keeping them in separate packs
is deliberate:

* ``ml_ensemble`` — "what is the probability the next *h* bars close up?", from
  models trained offline and validated walk-forward. It only updates when a bar
  closes, because that is when there are new features to score.
* ``projection`` — "where are the next few candles likely to travel?", rebuilt
  continuously from the live price and current conviction. It updates every
  second, because a projected candle that only moves once a minute would just be
  a slower chart.

The two disagree by design. One is a calibrated probability on a fixed horizon;
the other is a live path estimate that includes the last twenty seconds of tape.
"""
