# Data layer

Sources, transport, aggregation, and storage.

```
src/niftypulse/data/
├── upstox_auth.py    OAuth 2.0 flow and token lifecycle
├── upstox_rest.py    Historical, intraday, and quote endpoints
├── upstox_feed.py    v3 WebSocket client with protobuf decoding
├── aggregator.py     Tick → OHLCV bar
├── resample.py       Timeframe aggregation
├── store.py          Parquet persistence
├── synthetic.py      Offline data source
└── proto/            MarketDataFeed.proto + compiled stubs
```

---

## Authentication

`upstox_auth.py`

Upstox uses the OAuth 2.0 authorization-code flow.

```
GET  https://api.upstox.com/v2/login/authorization/dialog   → user logs in
     redirects with ?code=...
POST https://api.upstox.com/v2/login/authorization/token    → access_token
```

**Tokens expire at 03:30 IST the following morning.** The token is stored at
`data/upstox_token.json` (chmod 600) alongside its generation time, and
`TokenStore.load()` returns `None` once past expiry rather than letting a stale
token produce confusing API errors.

Token resolution order:

1. `UPSTOX_ACCESS_TOKEN` environment variable
2. `data/upstox_token.json`, if unexpired

Two alternative flows exist for automation. Upstox's **semi-automated** flow
pushes a token to a notifier URL after you approve on your phone, and the
**manual** flow lets you copy a token from the developer dashboard. Both are
supported by Upstox; this platform implements the interactive flow and accepts a
token from the environment, which covers scheduled jobs adequately.

---

## REST client

`upstox_rest.py`

| Method | Endpoint | Use |
|---|---|---|
| `fetch_historical` | `/v3/historical-candle/{key}/{unit}/{interval}/{to}/{from}` | One window |
| `fetch_history_range` | — | Chunked, stitched, rate-limited |
| `fetch_minute_history` | — | Convenience: last *N* calendar days |
| `fetch_intraday` | `/v3/historical-candle/intraday/{key}/{unit}/{interval}` | Current session only |
| `fetch_quote` | `/v2/market-quote/quotes` | Latest snapshot |

### Window limits

The v3 endpoint caps a single request:

| Unit | Available from | Max per request |
|---|---|---|
| minutes 1–15 | Jan 2022 | 1 month |
| minutes 16–300 | Jan 2022 | 1 quarter |
| hours 1–5 | Jan 2022 | 1 quarter |
| days | Jan 2000 | 1 decade |
| weeks, months | Jan 2000 | unlimited |

`_chunk_days()` encodes these, and `fetch_history_range` walks the range
accordingly. A single failed window is logged and skipped rather than aborting a
long backfill — over 180 days, one bad request should not cost you the other
twenty-five.

### Rate limiting

`RateLimiter` is a sliding window enforcing both published caps: **8 requests per
second** and **180 per minute**, comfortably under the provider's 25/s and
250/min. It blocks with a short sleep rather than raising, so callers do not need
retry logic for the common case.

`_get` additionally retries on 429 and 5xx with exponential backoff up to 4
attempts. Other status codes raise immediately with the response body attached,
because a malformed instrument key will not fix itself on retry.

### Intraday versus historical

The intraday endpoint is a separate call because the historical one excludes the
current session. `load_history` fetches both and merges them — without this, a
morning run would be missing the most recent bars, which are exactly the ones
that matter.

---

## WebSocket feed

`upstox_feed.py`

### Connection

```
GET  /v3/feed/market-data-feed/authorize     → wss:// URI with single-use code
WSS  connect, send binary subscription
```

The authorized URI contains a single-use `code`, so the connection is
self-authenticating. The `Authorization` header is sent as well, matching the
documentation.

### Subscription

Sent as a **binary** frame (not text), JSON-encoded:

```json
{
  "guid": "a1b2c3d4e5f6a7b8c9d0",
  "method": "sub",
  "data": { "mode": "full", "instrumentKeys": ["NSE_INDEX|Nifty 50"] }
}
```

Modes: `ltpc`, `full`, `full_d30`, `option_greeks`. The platform uses `full`,
which for a futures instrument carries depth, traded value, open interest, and
average traded price.

### Decoding

Frames are Protobuf, decoded against `MarketDataFeed.proto`. Two shapes matter:

| Instrument type | Feed variant | Contains |
|---|---|---|
| Index | `IndexFullFeed` | LTPC + rolled-up OHLC |
| Equity / futures | `MarketFullFeed` | LTPC + depth + ATP + OI + buy/sell quantities |

The decoder normalises both into a single `Tick` type, leaving depth fields at
zero for instruments that have no order book.

The first frame is `market_info` (segment status), the second a snapshot, and
subsequent frames are live updates.

### Resilience

`run()` reconnects with exponential backoff capped at 30s. A 45-second receive
timeout sends a ping rather than treating silence as a failure — a quiet market
is not a broken connection, and disconnecting on it would churn connections
through lunchtime.

### Regenerating the stubs

Only needed if Upstox publishes a new `.proto`:

```bash
uv run python -m grpc_tools.protoc \
  -I src/niftypulse/data/proto \
  -I "$(uv run python -c 'import grpc_tools,os;print(os.path.join(os.path.dirname(grpc_tools.__file__),"_proto"))')" \
  --python_out=src/niftypulse/data/proto \
  src/niftypulse/data/proto/MarketDataFeed.proto
```

The generated `MarketDataFeed_pb2.py` is **committed deliberately**, so the
package installs without a protoc toolchain.

---

## Tick aggregation

`aggregator.py`

Converts a tick stream into OHLCV bars.

### Session-anchored buckets

```python
def bar_start(moment, bar_minutes):
    elapsed = (moment - session_open).total_seconds()
    return session_open + timedelta(minutes=(elapsed // (bar_minutes*60)) * bar_minutes)
```

Buckets anchor to the 09:15 session open, not the top of the hour. Naive
`resample("5min")` would offset every bar by 45 minutes and leave a 15-minute
stub at the close.

### The volume problem

**An index has no traded volume.** There is no consolidated tape for NIFTY 50, so
`volume` arrives as zero.

Feeding a column of zeros into volume indicators produces constant, meaningless
signals that look like features and carry nothing. So the aggregator also records
`tick_count` — how many updates landed in the bar — which is a genuine proxy for
activity.

`activity_proxy()` in `features/indicators.py` encapsulates the fallback:

```python
volume  if the instrument reports any
tick_count  otherwise
constant 1.0  if neither (which makes the features constant, so they get dropped)
```

Returns a Series in every case. An earlier version returned a scalar `1.0`,
which broke `.rolling()` with an `AttributeError` — see
[Development](development.md#regression-guards).

### Session gaps

If a new bar's timestamp is on a different day than the partial bar, the partial
is **discarded rather than emitted**. Emitting it would produce a bar containing
an overnight gap, and every indicator downstream would inherit the artefact.

---

## Resampling

`resample.py`

Aggregates 1-minute bars to longer timeframes for the dashboard's `--timeframe`
flag.

The same session-anchoring applies: each row's bar timestamp is computed directly
and grouped on that. Because the bucket counter restarts at each session open,
bars can never span two days and no separate session guard is needed.

> **Note on a pandas behaviour:** passing a `MultiIndex` to `DataFrame.groupby`
> treats it as a flat array of tuple keys, not as two grouping levels. An earlier
> implementation relied on that and silently produced a single-level index. The
> current code groups on a computed timestamp instead, which sidesteps the issue
> entirely.

Changing `bar_minutes` invalidates trained models, because feature windows are
expressed in bars.

---

## Storage

`store.py`

`CandleStore` is an append-only Parquet store.

```python
store.append(frame)   # merge, deduplicate on index, keep="last"
store.load(start, end)
store.coverage()      # (first, last) timestamps
```

Deduplication keeps the **last** occurrence, so re-fetching an overlapping window
overwrites rather than duplicating. This matters because the current session is
re-pulled on every run.

All indices are normalised to `Asia/Kolkata`. Timezone bugs in this layer are
particularly nasty — a naive index silently shifts every session boundary.

---

## Offline source

`synthetic.py`

Generates NIFTY-like 1-minute bars with:

- regime-switching drift and volatility
- volatility clustering via an AR(1) process on the shock scale
- a U-shaped intraday volume curve
- a weak mean-reverting component

The mean reversion exists so the series has *some* structure to find. Without it,
a model finding nothing would be uninformative; with it, "found nothing" is a
statement about the pipeline.

### Why not pure noise

`generate_candles` produces a series whose AUC comes out near 0.50 — the honest
answer for something close to a random walk. The test suite separately constructs
a **deterministic alternating-drift series** as a positive control, so the two
cases are distinguishable. See [Development](development.md#testing-philosophy).

`generate_ticks` expands bars into a plausible tick stream for replay mode, and
`generate_quote` shapes a payload like the REST response.

Both are seeded and deterministic — `generate_candles(seed=42)` returns an
identical frame every time, which the tests rely on.

---

## Instrument keys

| Instrument | Key |
|---|---|
| NIFTY 50 | `NSE_INDEX\|Nifty 50` |
| NIFTY Bank | `NSE_INDEX\|Nifty Bank` |
| India VIX | `NSE_INDEX\|India VIX` |
| Front-month future | `NSE_FO\|<token>` — look up in the instrument master |

The full instrument master is published as gzipped JSON:

```
https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
```

Filter on `segment` and `instrument_type` to find a key — `NSE_FO` + `FUT` for
futures, `NSE_INDEX` + `INDEX` for indices.

**Prefer the future for scalping.** The index publishes no volume and no order
book, which disables seven activity features and both order-flow strategies.
