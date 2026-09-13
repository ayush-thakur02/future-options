"""Upstox Market Data Feed v3 WebSocket client.

The feed contract — authorization flow, the four subscription modes, the
``Ticker``/``FeedResponse`` shapes — is documented in the installed skill at
``.agents/skills/upstox/references/websocket.md``. The one detail worth checking
against your account before going live is the authorize path: this client calls
``/v3/feed/market-data-feed/authorize``, which is where the v3 proto feed is
issued, while the SDK's own ``WebsocketApi.get_market_data_feed_authorize`` is
the v2 endpoint. A 404 here means the account is on the other one.

Flow: fetch a single-use authorized socket URI, connect, subscribe, then decode
protobuf frames into :class:`Tick` objects.

Two feed shapes matter here. Index instruments (``NSE_INDEX|Nifty 50``) arrive as
``IndexFullFeed`` and carry no order book — only LTPC plus rolled-up OHLC. Equity
and futures instruments arrive as ``MarketFullFeed`` with depth, traded value,
open interest, and bid/ask. The decoder normalises both into one Tick type and
leaves depth fields at zero when the instrument has none.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from datetime import datetime

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from core.calendar import IST
from core.types import Tick

from .proto import MarketDataFeed_pb2 as pb

FEED_AUTHORIZE_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"

TickHandler = Callable[[Tick], None]
StatusHandler = Callable[[dict], None]

MARKET_STATUS_NAMES = {
    0: "PRE_OPEN_START",
    1: "PRE_OPEN_END",
    2: "NORMAL_OPEN",
    3: "NORMAL_CLOSE",
    4: "CLOSING_START",
    5: "CLOSING_END",
}


def authorize_feed_url(access_token: str, timeout: float = 30.0) -> str:
    """Exchange the access token for a single-use WebSocket URI."""
    response = httpx.get(
        FEED_AUTHORIZE_URL,
        headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"feed authorization failed ({response.status_code}): {response.text[:300]}"
        )
    payload = response.json()
    uri = payload.get("data", {}).get("authorized_redirect_uri")
    if not uri:
        raise RuntimeError(f"feed authorization returned no URI: {payload}")
    return uri


class UpstoxFeed:
    """Async WebSocket client for real-time ticks with automatic reconnection."""

    def __init__(
        self,
        access_token: str,
        instrument_keys: list[str],
        on_tick: TickHandler,
        on_status: StatusHandler | None = None,
        mode: str = "full",
        max_reconnect_delay: float = 30.0,
    ) -> None:
        if mode not in {"ltpc", "full", "full_d30", "option_greeks"}:
            raise ValueError(f"unsupported feed mode: {mode}")
        self.access_token = access_token
        self.instrument_keys = instrument_keys
        self.on_tick = on_tick
        self.on_status = on_status
        self.mode = mode
        self.max_reconnect_delay = max_reconnect_delay
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Connect and stream until :meth:`stop` is called."""
        delay = 1.0
        while not self._stop.is_set():
            try:
                await self._stream_once()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._emit_status({"type": "connection_error", "error": str(exc)[:200]})
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, self.max_reconnect_delay)

    async def _stream_once(self) -> None:
        uri = await asyncio.to_thread(authorize_feed_url, self.access_token)

        async with websockets.connect(
            uri,
            additional_headers={"Authorization": f"Bearer {self.access_token}"},
            max_size=16 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=20,
        ) as socket:
            self._emit_status({"type": "connected", "uri_host": _host_of(uri)})
            await socket.send(json.dumps(self._subscription_payload()))

            while not self._stop.is_set():
                try:
                    message = await asyncio.wait_for(socket.recv(), timeout=45.0)
                except TimeoutError:
                    # No frames during a quiet market; nudge with a ping to keep
                    # the connection from being reaped by an intermediary.
                    await socket.ping()
                    continue
                except ConnectionClosed:
                    self._emit_status({"type": "disconnected"})
                    return

                if isinstance(message, bytes):
                    self._handle_frame(message)

    def _subscription_payload(self) -> dict:
        return {
            "guid": uuid.uuid4().hex[:20],
            "method": "sub",
            "data": {"mode": self.mode, "instrumentKeys": self.instrument_keys},
        }

    def _handle_frame(self, raw: bytes) -> None:
        response = pb.FeedResponse()
        try:
            response.ParseFromString(raw)
        except Exception:
            return

        kind = response.type
        if kind == pb.market_info:
            self._emit_status(
                {
                    "type": "market_info",
                    "segments": {
                        name: MARKET_STATUS_NAMES.get(status, str(status))
                        for name, status in response.marketInfo.segmentStatus.items()
                    },
                }
            )
            return

        received_at = _frame_time(response.currentTs)
        for instrument_key, feed in response.feeds.items():
            tick = _decode_tick(instrument_key, feed, received_at)
            if tick is not None:
                self.on_tick(tick)

    def _emit_status(self, payload: dict) -> None:
        if self.on_status is not None:
            self.on_status(payload)


def _decode_tick(instrument_key: str, feed, received_at: datetime) -> Tick | None:
    """Normalise any feed variant into a Tick."""
    variant = feed.WhichOneof("FeedUnion")
    if variant is None:
        return None

    if variant == "ltpc":
        return _tick_from_ltpc(instrument_key, feed.ltpc, received_at)

    if variant == "fullFeed":
        full = feed.fullFeed
        inner = full.WhichOneof("FullFeedUnion")
        if inner == "indexFF":
            return _tick_from_ltpc(instrument_key, full.indexFF.ltpc, received_at)
        if inner == "marketFF":
            return _tick_from_market_feed(instrument_key, full.marketFF, received_at)
        return None

    if variant == "firstLevelWithGreeks":
        first = feed.firstLevelWithGreeks
        return _tick_from_ltpc(
            instrument_key,
            first.ltpc,
            received_at,
            bid_p=first.firstDepth.bidP,
            bid_q=first.firstDepth.bidQ,
            ask_p=first.firstDepth.askP,
            ask_q=first.firstDepth.askQ,
            volume_traded=first.vtt,
            open_interest=first.oi,
        )

    return None


def _tick_from_ltpc(
    instrument_key: str,
    ltpc,
    received_at: datetime,
    bid_p: float = 0.0,
    bid_q: int = 0,
    ask_p: float = 0.0,
    ask_q: int = 0,
    volume_traded: int = 0,
    open_interest: float = 0.0,
) -> Tick:
    return Tick(
        ts=received_at,
        ltp=ltpc.ltp,
        ltq=int(ltpc.ltq),
        close_prev=ltpc.cp,
        bid_p=bid_p,
        bid_q=int(bid_q),
        ask_p=ask_p,
        ask_q=int(ask_q),
        volume_traded=int(volume_traded),
        open_interest=float(open_interest),
    )


def _tick_from_market_feed(instrument_key: str, market, received_at: datetime) -> Tick:
    depth = market.marketLevel.bidAskQuote
    bid_p = depth[0].bidP if depth else 0.0
    bid_q = depth[0].bidQ if depth else 0
    ask_p = depth[0].askP if depth else 0.0
    ask_q = depth[0].askQ if depth else 0

    return Tick(
        ts=received_at,
        ltp=market.ltpc.ltp,
        ltq=int(market.ltpc.ltq),
        close_prev=market.ltpc.cp,
        bid_p=bid_p,
        bid_q=int(bid_q),
        ask_p=ask_p,
        ask_q=int(ask_q),
        volume_traded=int(market.vtt),
        avg_traded_price=float(market.atp),
        open_interest=float(market.oi),
        total_buy_qty=float(market.tbq),
        total_sell_qty=float(market.tsq),
    )


def _frame_time(current_ts: int) -> datetime:
    if not current_ts:
        return datetime.now(IST)
    return datetime.fromtimestamp(current_ts / 1000.0, tz=IST)


def _host_of(uri: str) -> str:
    try:
        return uri.split("//", 1)[1].split("/", 1)[0]
    except IndexError:
        return "unknown"
