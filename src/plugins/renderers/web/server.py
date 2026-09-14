"""Threaded local Flask server implementing the live renderer contract."""

from __future__ import annotations

import threading
import webbrowser
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template
from werkzeug.serving import WSGIRequestHandler, make_server

from core.types import BoardSnapshot, MarketSnapshot

from .serialize import serialize_snapshot


class _QuietHandler(WSGIRequestHandler):
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        return None


class WebRenderer:
    """Cache live snapshots for Flask while the market session owns cadence."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5050,
        refresh_ms: int = 1000,
        open_browser: bool = True,
        max_candles: int = 160,
    ) -> None:
        if not host.strip():
            raise ValueError("web host cannot be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("web port must be between 1 and 65535")
        self.host = host.strip()
        self.port = int(port)
        # A live market view should never become more than one second stale in
        # the browser. Faster polling remains configurable for local use.
        self.refresh_ms = min(max(int(refresh_ms), 100), 1000)
        self.open_browser = bool(open_browser)
        self.max_candles = max(int(max_candles), 20)
        self._lock = threading.RLock()
        self._payload: dict[str, Any] = {
            "ready": False,
            "kind": "waiting",
            "status": "warming market state",
            "updated_at": None,
        }
        self._server = None
        self._thread: threading.Thread | None = None

        package = Path(__file__).resolve().parent
        self.app = Flask(
            __name__,
            template_folder=str(package / "templates"),
            static_folder=str(package / "static"),
        )
        self.app.config.update(JSON_SORT_KEYS=False)
        self._routes()

    @property
    def url(self) -> str:
        browser_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{browser_host}:{self.port}"

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _routes(self) -> None:
        @self.app.get("/")
        def index():
            return render_template("dashboard.html", refresh_ms=self.refresh_ms)

        @self.app.get("/api/snapshot")
        def api_snapshot():
            with self._lock:
                return jsonify(self._payload)

        @self.app.get("/api/health")
        def health():
            with self._lock:
                return jsonify(
                    {
                        "ok": True,
                        "ready": bool(self._payload.get("ready")),
                        "updated_at": self._payload.get("updated_at"),
                    }
                )

    @contextmanager
    def live(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

    def start(self) -> None:
        if self.running:
            return
        self._server = make_server(
            self.host,
            self.port,
            self.app,
            threaded=True,
            request_handler=_QuietHandler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="niftypulse-web",
            daemon=True,
        )
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self.url, new=2)

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def live_update(
        self,
        snapshot: BoardSnapshot | MarketSnapshot,
        status: str = "",
    ) -> None:
        payload = serialize_snapshot(snapshot, status=status, max_candles=self.max_candles)
        payload["updated_at"] = datetime.now(UTC).isoformat()
        with self._lock:
            self._payload = payload


__all__ = ["WebRenderer"]
