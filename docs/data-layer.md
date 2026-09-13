# Data layer

Sources, transport, aggregation, and where it all lands.

```
src/plugins/sources/
├── upstox/           the account-backed adapter
│   ├── auth.py       OAuth 2.0 flow and token lifecycle
│   ├── rest.py       Historical, intraday, and quote endpoints
│   ├── feed.py       v3 WebSocket client with protobuf decoding
│   ├── broker.py     the object the "broker" capability builds
│   └── proto/        MarketDataFeed.proto + compiled stubs
├── simulated/        generated series, clock-paced feed, tick expansion
├── history/          the partitioned store, manifest, recorders, resampling
└── option_chain/     pricing, the chain, the money legs

src/plugins/aggregators/candle_builder/    tick → session-anchored time bars
```

The Upstox client is not imported by anything above it. The history store asks
for the `broker` capability; the session asks for a source. That is what makes
the provider a pack rather than a dependency.

The API surface implemented here is documented — with signatures verified against
the official SDK — in the installed skill at
`.agents/skills/upstox/references/`. Read those before changing a URL: `HistoryV3Api`
takes no `api_version` and the v2 classes take one, and mixing the two is the most
common source of errors.

---

## Authentication

`plugins/sources/upstox/auth.py`

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

`UpstoxBroker` wraps all three concerns — credential, REST client, live feed — so
a plugin that needs market data depends on one capability (`broker`) rather than
on three modules. `broker.is_configured` never raises: a missing token is a
state, and it is what the session uses to decide between live and simulated.

---

## REST client

`plugins/sources/upstox/rest.py`

| Method | Endpoint | Use |
|---|---|---|
| `fetch_historical` | `/v3/historical-candle/{key}/{unit}/{interval}/{to}/{from}` | One window |
| `fetch_history_range` | — | Chunked, stitched, rate-limited |
| `fetch_minute_history` | — | Convenience: last *N* calendar days |
| `fetch_intraday` | `/v3/historical-candle/intraday/{key}/{unit}/{interval}` | Current session only |
| `fetch_quote` | `/v2/market-quote/quotes` | Latest snapshot |

Note the historical endpoint takes `to_date` **before** `from_date`. It is not a
typo here, and it is an easy thing to "fix" into a broken request.

### Window limits

The v3 endpoint caps a single request:

| Unit | Available from | Max per request |
|---|---|---|
| minutes 1–15 | Jan 2022 | ~1 month |
| minutes 16–300 | Jan 2022 | ~1 quarter |
| hours 1–5 | Jan 2022 | ~1 quarter |
| days | Jan 2000 | ~1 decade |
| weeks, months | Jan 2000 | unlimited |

`_chunk_days()` encodes these and `fetch_history_range` walks the range
accordingly. A single failed window is logged and skipped rather than aborting a
long backfill — over 180 days, one bad request should not cost you the other
twenty-five.

Intervals are validated locally (`VALID_INTERVALS`) before the request goes out,
so an unsupported unit fails with a clear message instead of as a 400 halfway
through a backfill.

### Rate limiting

`RateLimiter` is a sliding window enforcing **8 requests per second** and **180 per
minute**. The published caps are 50/s and 500/min for market data; a backfill is a
bulk operation and there is nothing to gain from running at the ceiling, which
also leaves headroom for a live feed in the same process. It blocks with a short
sleep rather than raising, so callers do not need retry logic for the common case.

`_get` retries on 429 and 5xx with exponential backoff up to 4 attempts. Other
status codes raise immediately with the response body attached, because a
malformed instrument key will not fix itself on retry.

### Intraday versus historical

The intraday endpoint is a separate call because the historical one excludes the
current session. `load_history` fetches both and merges them — without this, a
morning run would be missing the most recent bars, which are exactly the ones
that matter.

### Endpoints not yet used

Orders, GTT, portfolio, margins, the option chain, and instrument search exist in
the API and are deliberately not called. This platform places no orders. See the
chain reader note in [Options](options.md) for the one that is next.

---

## WebSocket feed

`plugins/sources/upstox/feed.py`

```
GET  /v3/feed/market-data-feed/authorize   → single-use socket URI
WS   connect, send {"method": "sub", "data": {...}}, receive protobuf frames
```

Two feed shapes matter. Index instruments (`NSE_INDEX|Nifty 50`) arrive as
`IndexFullFeed` and carry no order book — only LTPC plus rolled-up OHLC. Equity
and futures instruments arrive as `MarketFullFeed` with depth, traded value, open
interest, and bid/ask. The decoder normalises both into one `Tick` and leaves the
depth fields at zero when the instrument has none.

| Mode | Data |
|---|---|
| `ltpc` | LTP, last-traded quantity, close, volume |
| `full` | LTP + OHLC + market depth + OI |
| `full_d30` | `full` with 30 levels of depth |
| `option_greeks` | IV, delta, gamma, theta, vega |

Reconnection is automatic with exponential backoff to 30s, and status is reported
through a callback so the dashboard can show it. A quiet market is handled
explicitly: no frames for 45 seconds sends a ping rather than assuming the
connection is dead.

The generated protobuf stub declares a dependency on
`google/protobuf/wrappers.proto` that it does not import, which makes it
unimportable on its own. `proto/__init__.py` imports the dependency first, so the
generated file stays regenerable and untouched. The bug was latent until the
plugin loader — which imports every plugin, as it must — imported the feed; the
live path would have failed on its first connection with the same error.

Regenerating the stubs:

```bash
uv run python -m grpc_tools.protoc \
  -I src/plugins/sources/upstox/proto \
  --python_out=src/plugins/sources/upstox/proto \
  src/plugins/sources/upstox/proto/MarketDataFeed.proto
```

---

## The simulated source

`plugins/sources/simulated/`

Two shapes of the same generated market:

- **`SimulatedFeed`** — a tick stream paced by the wall clock, for the live
  dashboard. One minute of wall clock per one-minute bar at `speed=1.0`. The
  first pass is re-based onto the current time so a session opens on the present;
  later passes continue forward rather than rewinding, because a tick stream that
  moves backwards is not something the aggregator is built to survive.
- **`generate_ticks`** — the same expansion as a frame, for headless replay.

The generated series has volatility clustering (an AR(1) process on the shock
scale), intraday volume seasonality, regime shifts, and a weak mean-reverting
component. That is deliberate: a series with *no* structure would let a broken
feature pipeline look identical to a working one, because "found nothing" and
"found nothing because it is broken" would be indistinguishable.

---

## Aggregation

`plugins/aggregators/candle_builder/`

`CandleAggregator` accumulates ticks into OHLCV bars and emits each bar once it
closes. Two details:

**Buckets are anchored to the 09:15 open, not the hour.** A naive
`resample("5min")` offsets every bar by 45 minutes and turns the last bar of the
day into a 15-minute stub.

**`tick_count` is recorded alongside volume.** An index has no consolidated tape,
so `volume` is zero for NIFTY 50. Rather than feed a column of zeros into volume
indicators — which produces constant, meaningless signals — the aggregator counts
updates per bar and the feature layer falls back to that.

A session gap drops a stale partial bar rather than emitting one that spans the
overnight break.

---

## Storage

See [Storage](storage.md). The short version: one parquet per instrument per
trading day for bars, one per hour for the tape, and a manifest that answers
"what have I got?" without a scan.
