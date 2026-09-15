"""Web dashboard launcher.

The browser dashboard is the platform's only interface. Everything it needs to
come up is here: the run knobs on the command line, the session they configure,
and the renderer that serves it. The work that is not a live view — logging in,
building history, training, backtesting — is library code with no command of its
own; docs/operations.md shows how to call it.

Absolute imports rather than relative ones: the layout is flat, so this module is
a top-level module and has no parent package to be relative to.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from core.settings import load_settings
from kernel import Kernel
from runtime.session import Session, SessionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="niftypulse",
        description="Serve the NIFTY 50 research dashboard: call, index and put, with the next three candles projected.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Replay simulated ticks instead of the live Upstox feed",
    )
    parser.add_argument("--timeframe", type=int, default=1, help="Bar size in minutes")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Replay speed against the clock (1.0 = real time)"
    )
    parser.add_argument("--refresh", type=float, default=1.0, help="Seconds between dashboard frames")
    parser.add_argument(
        "--nowcast", type=float, default=1.0, help="Seconds between projection refreshes"
    )
    parser.add_argument("--bars-ahead", type=int, default=3, help="How many candles to project")
    parser.add_argument(
        "--legs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chart the at-the-money call and put beside the index",
    )
    parser.add_argument(
        "--learn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update instrument models as candles close",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay recent bars into the research ledgers before the session starts",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=600,
        help="Bars a cold-start warm-up replays per instrument (later runs resume instead)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU workers (-1 = all available CPUs; default from config)",
    )
    parser.add_argument(
        "--host", default=None, help="Web bind host (default from renderer config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Web bind port (default from renderer config)"
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether the dashboard opens in the default browser",
    )
    return parser


def session_config(args: argparse.Namespace) -> SessionConfig:
    """The run the flags describe.

    Refresh and nowcast are capped at one second: a live market view that is
    slower than that is stale before it is drawn, and the browser polls at most
    once a second anyway.
    """
    return SessionConfig(
        offline=args.offline,
        timeframe=args.timeframe,
        refresh=min(float(args.refresh), 1.0),
        nowcast_interval=min(float(args.nowcast), 1.0),
        speed=args.speed,
        legs=args.legs,
        warmup=args.warmup,
        warmup_bars=args.warmup_bars,
        web_host=args.host.strip() if args.host else None,
        web_port=args.port,
        open_browser=args.open_browser,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeframe < 1 or args.bars_ahead < 1 or args.refresh <= 0 or args.nowcast <= 0 or args.speed <= 0:
        parser.error("timeframe, bars-ahead, refresh, nowcast and speed must be positive")
    if args.warmup_bars < 1:
        parser.error("--warmup-bars must be at least 1")
    if args.host is not None and not args.host.strip():
        parser.error("--host cannot be empty")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    settings = load_settings()
    if args.workers is not None:
        settings.workers = args.workers
    settings.online_learning = args.learn
    settings.plugin_config.setdefault("forecast:projection", {})["bars_ahead"] = args.bars_ahead

    session = Session(kernel=Kernel.bootstrap(settings), config=session_config(args))
    print(
        "live Upstox feed — ticks stream as they print"
        if session.live
        else "simulated feed — generated ticks, paced against the clock. Set Upstox credentials for live data.",
        flush=True,
    )
    try:
        session.bootstrap(progress=lambda message: print(f"  {message}", flush=True))
        print(session.describe(), flush=True)
        if session.renderer is not None:
            print(f"web dashboard  {session.renderer.url}", flush=True)
        print("The browser refreshes automatically. Press Ctrl+C here to stop it.", flush=True)
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        print("dashboard stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
