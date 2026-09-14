"""Web renderer API and JSON boundary tests."""

from __future__ import annotations

import json

import pytest

from core.settings import Settings
from kernel import Kernel
from plugins.renderers.web import WebRenderer, serialize_snapshot


def test_web_renderer_is_a_separate_plugin_capability() -> None:
    kernel = Kernel.bootstrap(Settings(), with_entry_points=False)
    assert kernel.provider("web_frame") == "renderer:web"
    renderer = kernel.build("renderer:web", open_browser=False)
    assert isinstance(renderer, WebRenderer)
    assert kernel.provider("frame") == "renderer:terminal"


def test_board_snapshot_serializes_without_nan(offline_session) -> None:
    frame = offline_session.snapshot()
    payload = serialize_snapshot(frame, status="SIMULATION", max_candles=40)

    assert payload["ready"] is True
    assert payload["kind"] == "board"
    assert payload["status"] == "SIMULATION"
    assert [leg["label"] for leg in payload["legs"]] == ["INDEX", "CALL", "PUT"]
    assert all(len(leg["market"]["candles"]) <= 40 for leg in payload["legs"])
    assert all("research" in leg["market"] for leg in payload["legs"])
    json.dumps(payload, allow_nan=False)


def test_api_exposes_health_and_latest_snapshot(offline_session) -> None:
    renderer = WebRenderer(open_browser=False, max_candles=25)
    client = renderer.app.test_client()

    waiting = client.get("/api/health")
    assert waiting.status_code == 200
    assert waiting.get_json()["ready"] is False

    renderer.live_update(offline_session.snapshot(), "SIMULATION")
    response = client.get("/api/snapshot")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ready"] is True
    assert payload["updated_at"]
    assert payload["status"] == "SIMULATION"
    assert len(payload["legs"][0]["market"]["candles"]) <= 25


@pytest.mark.parametrize(("host", "port"), [("", 5050), ("127.0.0.1", 0), ("127.0.0.1", 70000)])
def test_invalid_web_bind_configuration_fails_early(host: str, port: int) -> None:
    with pytest.raises(ValueError):
        WebRenderer(host=host, port=port, open_browser=False)
