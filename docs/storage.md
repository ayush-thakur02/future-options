# Storage

Everything the platform has ever pulled is on disk, partitioned so that a write
costs one file and a read costs only the range it asks for.

```
data/
├── manifest.json
├── candles/NSE_INDEX_Nifty_50/2026/09/2026-09-11.parquet
├── ticks/NSE_INDEX_Nifty_50/2026-09-11/09.parquet
├── ticks/NSE_INDEX_Nifty_50/.pending.jsonl
├── chain/NSE_INDEX_Nifty_50-chain/2026-09-11/09.parquet
└── research/
    ├── strategies.sqlite3
    └── online_ai/{live|simulation}/<instrument>.sqlite3
```

`niftypulse data` prints the inventory; `niftypulse data --verbose` lists every
partition file.

---

## Why partitioned

The previous layout kept every bar in one `candles.parquet`. Three things were
wrong with it, and they compound as a store grows:

| | Single file | Partitioned |
|---|---|---|
| Append a session | Rewrite every bar ever stored — 100k rows read, concatenated, written back to add 375 | Rewrite one day |
| Read yesterday | Scan the whole history | Open one file |
| Session boundaries | Mergeable by construction | The date *is* the path |

That last row is the one that is easy to miss. A naive `resample("5min")` anchors
buckets to the hour rather than the 09:15 open and will happily put a 15:29 bar
next to the next morning's 09:16 bar. Here a session cannot be split across two
files or merged with another, because the store has no way to express it.

Compression is **zstd**. On sorted numeric columns it is several times smaller
than CSV and decompresses far faster than it compresses, which is the right trade
for data written once a day and read constantly.

---

## What is stored

| Dataset | Shard | Written by | Can it be re-fetched? |
|---|---|---|---|
| `candles` | trading day | `fetch`, `sync`, `dashboard` | Yes — it is the provider's historical endpoint |
| `ticks` | hour | a live session | **No.** A provider publishes candles, not the tape that produced them |
| `chain` | hour | a live session, sampled once a minute | **No** — and historical option chains are not published either |

The asymmetry is the reason recording exists at all. Candles can always be pulled
again; the tape cannot. Whatever is not recorded while it streams is gone.

Ticks are buffered and flushed on a row count (4,000) **or** the hour boundary —
the boundary matters because the hour is the partition key, and a late flush would
reopen a closed shard. A write per tick would spend the whole session in the
filesystem: a parquet write costs milliseconds and a NIFTY tick arrives every few
milliseconds at the open.

Before entering that memory buffer, every tick is appended to
`.pending.jsonl`. The journal is flushed and synced on a short cadence, recovered
at recorder startup, and truncated only after its rows are safely compacted into
Parquet. A partial last line from a killed process is ignored while every complete
event remains recoverable.

Tick identity hashes the full event, including quote/depth/greek values when
present. Two WebSocket updates with the same exchange timestamp are retained if
their payload differs; replaying an identical event is idempotent. Parquet shards
and the manifest use temporary files, `fsync`, and atomic rename under per-file
locks, so a crash cannot expose a half-written replacement.

Chain snapshots are sampled once a minute rather than streamed. Open interest and
implied vol move on a scale of minutes, and polling faster mostly records the same
numbers again.

---

## The manifest

`data/manifest.json` is a small index of what is on disk:

```json
{
  "version": 1,
  "updated_at": "2026-09-13T21:08:29+05:30",
  "datasets": {
    "candles": {
      "NSE_INDEX_Nifty_50": {
        "instrument": "NSE_INDEX|Nifty 50",
        "rows": 45000,
        "sessions": 120,
        "first": "2026-03-30T09:15:00+05:30",
        "last": "2026-09-11T15:29:00+05:30",
        "bar_minutes": 1
      }
    }
  }
}
```

It exists so that "what have I got?" is instant on a store of any size. The
alternative — scanning the tree — gets slower every day the platform runs, and the
question is asked on startup, by `doctor`, and by every tool that wants coverage.

It is **advisory, never authoritative**. Delete it and the next write rebuilds it;
corrupt it and the store still reads correctly, it just takes the slow path to
find out. The files are the truth.

---

## Never fetching twice

`niftypulse sync` and dashboard startup recover the journal first, convert the
local tick tape into complete session-anchored candles, then check candle
coverage. The broker is asked only for the missing prefix or tail:

```
niftypulse sync --days 400
```

- Cache current → no API calls at all.
- Cache two days stale → one request covering two days.
- No cache → the configured window, chunked to respect the provider's request
  caps and stitched.

The coverage check includes the requested start. A small recent tail cannot be
mistaken for a complete warm-up window. Candle manifest entries record whether
their source was `broker`, `websocket`, `manual`, or `simulation`; simulation
uses its own instrument key and never suppresses a live backfill.

Open `niftypulse sync --offline` and it will use whatever is cached, or generate a
series when the cache is empty — which is also how the whole pipeline can be
exercised without credentials.

---

## Migrating from the single-file store

If `data/candles.parquet` exists it is imported into partitions on first use and
renamed to `candles.parquet.migrated`. Silently starting from an empty store
would look exactly like losing your history, which is why the import is automatic
and the old file is kept rather than deleted.

`doctor` reports a legacy file as `legacy candle file — run niftypulse sync to
import` until it has been migrated.

---

## Reading it back

The store is a plugin (`source:history`, providing the `history` capability), so
application code asks for bars rather than for files:

```python
from plugins.sources.history import PartitionedStore

store = PartitionedStore(settings.data_dir, settings.instrument_key)
store.load()                                   # everything
store.load(start, end)                         # one range, one file each
store.coverage()                               # from the manifest
store.write(frame)                             # merge, deduplicated on the index
```

A candle write is idempotent on timestamp. A tick write is idempotent on
timestamp plus event identity, preserving distinct same-time updates while
preventing replay duplication.

---

## Sizing

Measured on 45,000 one-minute bars (120 sessions): **2.2 MB** on disk across 120
partitions. A tick hour of a busy index feed lands around 1–4 MB compressed
depending on how much of the depth fields the feed fills in.

If you want to bound it, delete old `ticks/` and `chain/` partitions — they are
recorded, not fetched, so nothing else depends on them.
