"""Upstox REST client for historical, intraday, and snapshot market data.

Reference: https://upstox.com/developer/api-documentation/v3/get-historical-candle-data

The v3 historical endpoint caps a single request at one month of 1–15 minute
candles, so anything longer is transparently chunked and stitched. A token
bucket keeps us inside the published per-minute rate limit.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST

API_BASE = "https://api.upstox.com"
HISTORY_URL = f"{API_BASE}/v3/historical-candle"
INTRADAY_URL = f"{API_BASE}/v3/historical-candle/intraday"
QUOTE_URL = f"{API_BASE}/v2/market-quote/quotes"

HISTORY_START = date(2022, 1, 1)

# Published limits: 25 req/s, 250 req/min. Stay comfortably underneath.
_MAX_PER_SECOND = 8
_MAX_PER_MINUTE = 180


class RateLimiter:
    """Sliding-window limiter covering both the per-second and per-minute caps."""

    def __init__(self, per_second: int = _MAX_PER_SECOND, per_minute: int = _MAX_PER_MINUTE) -> None:
        self.per_second = per_second
        self.per_minute = per_minute
        self._second_window: deque[float] = deque()
        self._minute_window: deque[float] = deque()

    def acquire(self) -> None:
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

    def fetch_historical(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """One raw historical request (subject to the provider's window limits)."""
        url = f"{HISTORY_URL}/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
        payload = self._get(url)
        return _parse_candles(payload)

    def fetch_intraday(self, instrument_key: str, unit: str = "minutes", interval: int = 1) -> pd.DataFrame:
        """Candles for the current trading day only."""
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
