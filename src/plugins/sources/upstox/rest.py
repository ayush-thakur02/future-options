"""Upstox REST client for historical, intraday, and snapshot market data.

The v3 historical endpoint caps a single request at about a month of 1–15 minute
candles, so anything longer is transparently chunked and stitched. A token bucket
keeps us well inside the published limits.

The API surface implemented here is documented, with signatures verified against
the official SDK, in the installed skill at
``.agents/skills/upstox/references/market-data.md`` (endpoints, units, intervals)
and ``.../errors.md`` (status and UDAPI codes). Read those before changing a URL
or adding an endpoint — ``HistoryV3Api`` takes no ``api_version`` and the v2
classes take one, and mixing the two is the most common source of errors.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import date, datetime, timedelta
from threading import Lock

import httpx
import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST

from .auth import AuthenticationError

API_BASE = "https://api.upstox.com"
HISTORY_URL = f"{API_BASE}/v3/historical-candle"
INTRADAY_URL = f"{API_BASE}/v3/historical-candle/intraday"
QUOTE_URL = f"{API_BASE}/v2/market-quote/quotes"

HISTORY_START = date(2022, 1, 1)

# Intervals the historical endpoint accepts, per unit. Passing anything else is a
# 400, so it is checked here rather than discovered as a failed backfill.
VALID_INTERVALS: dict[str, range] = {
    "minutes": range(1, 301),
    "hours": range(1, 6),
    "days": range(1, 2),
    "weeks": range(1, 2),
    "months": range(1, 2),
}

# Published limits are 50 requests/second and 500/minute for market data, but a
# backfill is a bulk operation and there is nothing to gain from running at the
# ceiling. These stay far underneath, which also leaves headroom for a live feed
# running in the same process.
_MAX_PER_SECOND = 8
_MAX_PER_MINUTE = 180


class RateLimiter:
    """Sliding-window limiter covering both the per-second and per-minute caps."""

    def __init__(self, per_second: int = _MAX_PER_SECOND, per_minute: int = _MAX_PER_MINUTE) -> None:
        self.per_second = per_second
        self.per_minute = per_minute
        self._second_window: deque[float] = deque()
        self._minute_window: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        with self._lock:
            self._acquire()

    def _acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self._second_window and now - self._second_window[0] > 1.0:
                self._second_window.popleft()
            while self._minute_window and now - self._minute_window[0] > 60.0:
                self._minute_window.popleft()

            if len(self._second_window) < self.per_second and len(self._minute_window) < self.per_minute:
                self._second_window.append(now)
                self._minute_window.append(now)
                return
            time.sleep(0.05)


class UpstoxREST:
    """Thin, retrying wrapper over the Upstox v2/v3 REST APIs."""

    def __init__(self, access_token: str, timeout: float = 30.0, max_retries: int = 4) -> None:
        self.access_token = access_token
        self.max_retries = max_retries
        self._limiter = RateLimiter()
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> UpstoxREST:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _get(self, url: str, params: dict | None = None) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code == 401:
                raise AuthenticationError("Upstox rejected the access token (401). Run `niftypulse login`.")

            # Retry on rate limiting and transient server faults.
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2.0**attempt, 15.0))
                last_error = RuntimeError(f"{response.status_code}: {response.text[:200]}")
                continue

            raise RuntimeError(
                f"Upstox request failed ({response.status_code}) for {url}: {response.text[:300]}"
            )

        raise RuntimeError(f"Upstox request exhausted retries for {url}: {last_error}")

    # ------------------------------------------------------------------ history

    def ltp(self, instrument_key: str) -> dict:
        return self._get(f"{API_BASE}/v3/market-quote/ltp", {"instrument_key": instrument_key}).get("data", {})

    def market_status(self, exchange: str = "NSE") -> str:
        return self._get(f"{API_BASE}/v2/market/status/{exchange}").get("data", {}).get("status", "unknown")

    def option_contracts(self, instrument_key: str) -> list[dict]:
        return self._get(f"{API_BASE}/v2/option/contract", {"instrument_key": instrument_key}).get("data", [])

    def option_chain(self, instrument_key: str, expiry: str) -> list[dict]:
        return self._get(f"{API_BASE}/v2/option/chain", {"instrument_key": instrument_key, "expiry_date": expiry}).get("data", [])

    def fetch_historical(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """One raw historical request (subject to the provider's window limits).

        Note the endpoint takes ``to_date`` before ``from_date``. It is not a typo
        here, and it is an easy thing to "fix" into a broken request.
        """
        _check_interval(unit, interval)
        url = f"{HISTORY_URL}/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
        payload = self._get(url)
        return _parse_candles(payload)

    def fetch_intraday(self, instrument_key: str, unit: str = "minutes", interval: int = 1) -> pd.DataFrame:
        """Candles for the current trading day only."""
        _check_interval(unit, interval)
        url = f"{INTRADAY_URL}/{instrument_key}/{unit}/{interval}"
        payload = self._get(url)
        return _parse_candles(payload)

    def fetch_history_range(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch an arbitrarily long range, chunking to respect API window caps."""
        chunk_days = _chunk_days(unit, interval)
        frames: list[pd.DataFrame] = []
        cursor = from_date

        while cursor <= to_date:
            window_end = min(cursor + timedelta(days=chunk_days - 1), to_date)
            try:
                frame = self.fetch_historical(instrument_key, unit, interval, cursor, window_end)
                if not frame.empty:
                    frames.append(frame)
            except AuthenticationError:
                raise
            except RuntimeError as exc:
                # A single empty/over-limit window should not abort a long backfill.
                print(f"  ! window {cursor} -> {window_end} failed: {str(exc)[:140]}")
            cursor = window_end + timedelta(days=1)

        if not frames:
            return normalize_candles(pd.DataFrame())
        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")]
        return normalize_candles(merged)

    def fetch_minute_history(
        self,
        instrument_key: str,
        days: int = 120,
        end: date | None = None,
    ) -> pd.DataFrame:
        """1-minute candles for the last ``days`` calendar days."""
        end = end or datetime.now(IST).date()
        start = max(end - timedelta(days=days), HISTORY_START)
        return self.fetch_history_range(instrument_key, "minutes", 1, start, end)

    # ------------------------------------------------------------------ quotes

    def fetch_quote(self, instrument_key: str) -> dict:
        """Latest snapshot for an instrument (LTP, OHLC, volume, depth)."""
        payload = self._get(QUOTE_URL, params={"instrument_key": instrument_key})
        data = payload.get("data", {})
        return next(iter(data.values()), {})


def _check_interval(unit: str, interval: int) -> None:
    """Fail early on an interval the endpoint will reject with a 400."""
    allowed = VALID_INTERVALS.get(unit)
    if allowed is None:
        raise ValueError(f"unsupported unit {unit!r}; expected one of {sorted(VALID_INTERVALS)}")
    if interval not in allowed:
        raise ValueError(
            f"interval {interval} is not valid for unit {unit!r} "
            f"(allowed {allowed.start}..{allowed.stop - 1})"
        )


def _chunk_days(unit: str, interval: int) -> int:
    """Largest safe request window for a given unit/interval."""
    if unit == "minutes":
        if interval <= 15:
            return 28
        return 85
    if unit == "hours":
        return 85
    if unit == "days":
        return 3000
    return 3000


def _parse_candles(payload: dict) -> pd.DataFrame:
    """Convert the nested candle array into a typed OHLCV frame."""
    rows = payload.get("data", {}).get("candles", []) or []
    if not rows:
        return normalize_candles(pd.DataFrame())

    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert(IST)
    frame = frame.set_index("ts").sort_index()
    for column in ("open", "high", "low", "close", "volume", "oi"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return normalize_candles(frame.dropna(subset=["close"]))
