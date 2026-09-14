"""Launcher tests: the flags a run is started with, and where they land.

The dashboard is started by one command, so its flags are the whole run
interface — they have to reach the session exactly as written, and a bad value
has to stop the run before it warms an engine.
"""

from __future__ import annotations

import pytest

from webapp import build_parser, main, session_config


def test_defaults_describe_a_live_dashboard() -> None:
    config = session_config(build_parser().parse_args([]))

    assert config.offline is False
    assert config.refresh == 1.0
    assert config.nowcast_interval == 1.0
    assert config.legs is True
    assert config.web_host is None
    assert config.web_port is None
    assert config.open_browser is None


def test_flags_reach_the_session_config() -> None:
    args = build_parser().parse_args(
        ["--offline", "--speed", "60", "--timeframe", "5", "--no-legs", "--port", "8080"]
    )
    config = session_config(args)

    assert config.offline is True
    assert config.speed == 60
    assert config.timeframe == 5
    assert config.legs is False
    assert config.web_port == 8080


def test_the_view_never_goes_staler_than_one_second() -> None:
    args = build_parser().parse_args(["--refresh", "30", "--nowcast", "30"])
    config = session_config(args)

    assert config.refresh == 1.0
    assert config.nowcast_interval == 1.0


@pytest.mark.parametrize(
    "argument",
    [
        ["--timeframe", "0"],
        ["--bars-ahead", "0"],
        ["--speed", "0"],
        ["--refresh", "0"],
        ["--port", "70000"],
        ["--host", ""],
    ],
)
def test_an_impossible_run_stops_before_it_warms_anything(argument: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argument)

    assert exit_info.value.code == 2
