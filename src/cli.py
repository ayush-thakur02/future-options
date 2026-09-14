"""Command line interface.

Absolute imports rather than relative ones: the layout is flat, so this module is
a top-level module and has no parent package to be relative to.
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backtest import BacktestConfig, CostModel, run_backtest, run_threshold_sweep
from core.calendar import TradingCalendar
from core.settings import Settings, load_settings
from core.types import BoardSnapshot
from core.version import __version__
from kernel import BUILTIN_PACKAGE, Kernel
from plugins.features.technical import FEATURE_GROUPS, build_features, feature_columns
from plugins.forecasts.ml_ensemble.trainer import (
    Trainer,
    list_artifacts,
    render_report,
    render_scalping_hurdle,
    save_training_metadata,
)
from plugins.sources.history import StoreManifest
from plugins.strategies import StrategyCatalog, StrategyContext
from runtime.bars import BarLoader
from runtime.session import Session, SessionConfig

app = typer.Typer(
    name="niftypulse",
    help="Real-time NIFTY 50 analytics, strategy signals, and short-horizon ML forecasts.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _settings() -> Settings:
    return load_settings()


def _kernel() -> Kernel:
    """A kernel with every bundled plugin discovered and registered."""
    return Kernel.bootstrap(_settings())


def _bars(offline: bool, days: int, refresh: bool = True, quiet: bool = False):
    """The bar series to work on, wherever it comes from."""
    kernel = _kernel()
    return BarLoader(kernel).load(days=days, refresh=refresh, quiet=quiet, offline=offline)


# --------------------------------------------------------------------- login


@app.command()
def login(
    open_browser: bool = typer.Option(True, help="Open the login page in a browser"),
) -> None:
    """Authenticate with Upstox and store an access token."""
    from plugins.sources.upstox.auth import interactive_login

    settings = _settings()
    credentials = settings.credentials
    if not credentials.has_oauth:
        console.print(
            Panel(
                "Set [bold]UPSTOX_CLIENT_ID[/], [bold]UPSTOX_CLIENT_SECRET[/] and "
                "[bold]UPSTOX_REDIRECT_URI[/] in your environment or .env file.\n\n"
                "Create an app at [link=https://account.upstox.com/developer/apps]"
                "account.upstox.com/developer/apps[/link] to get these.",
                title="[red]missing credentials[/]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    settings.ensure_dirs()
    interactive_login(
        credentials.client_id,
        credentials.client_secret,
        credentials.redirect_uri,
        settings.token_path,
        open_browser=open_browser,
    )


# --------------------------------------------------------------------- fetch


@app.command()
def fetch(
    days: int = typer.Option(120, help="Calendar days of 1-minute history to fetch"),
    offline: bool = typer.Option(False, "--offline", help="Use synthetic data instead of Upstox"),
) -> None:
    """Download historical candles into the local store."""
    frame = _bars(offline=offline, days=days, refresh=True)

    if frame.empty:
        console.print("[red]no candles retrieved[/]")
        raise typer.Exit(code=1)

    first, last = frame.index[0], frame.index[-1]
    sessions = frame.index.normalize().nunique()
    console.print(
        Panel(
            f"bars       [bold]{len(frame):,}[/]\n"
            f"sessions   [bold]{sessions:,}[/]\n"
            f"range      {first} -> {last}\n"
            f"price      {frame['close'].min():,.2f} .. {frame['close'].max():,.2f}",
            title="[green]history stored[/]",
            border_style="green",
        )
    )


# --------------------------------------------------------------------- train


@app.command()
def train(
    days: int = typer.Option(0, help="Re-fetch this many days before training (0 = use cache)"),
    horizons: str = typer.Option("", help="Comma-separated horizons, e.g. 1,3,5,15"),
    offline: bool = typer.Option(False, "--offline", help="Train on synthetic data"),
) -> None:
    """Train direction models for every forecast horizon."""
    settings = _settings()
    bars = _bars(offline=offline, days=days or 400, refresh=bool(days))

    if bars.empty:
        console.print("[red]no bars available[/]")
        raise typer.Exit(code=1)

    console.print("Computing features...")
    features = build_features(bars, bar_minutes=settings.bar_minutes, expiry_weekday=settings.expiry_weekday)
    columns = feature_columns(features)
    console.print(f"  {len(features):,} rows x {len(columns)} features across {len(FEATURE_GROUPS)} groups")

    selected = tuple(int(h) for h in horizons.split(",") if h.strip()) if horizons else settings.horizons

    console.print(f"\nTraining horizons {', '.join(f'{h}m' for h in selected)}...")
    trainer = Trainer(settings, n_jobs=settings.train_n_jobs)
    results = trainer.train_all(features, bars, columns, horizons=selected)

    if not results:
        console.print(
            "\n[red]no horizon could be trained[/]. Every horizon is untradeable at the "
            "current cost assumption. See the reasons above."
        )
        raise typer.Exit(code=1)

    console.print()
    render_scalping_hurdle(settings, bars)
    console.print()
    render_report(results)

    paths = trainer.save(results)
    report_path = save_training_metadata(settings, results, skipped=trainer.skipped)
    console.print(f"\n[green]saved[/] {len(paths)} artifacts to {settings.model_dir}")
    console.print(f"[green]saved[/] training report to {report_path}")


@app.command()
def models() -> None:
    """List trained model artifacts and their metrics."""
    settings = _settings()
    artifacts = list_artifacts(settings)
    if not artifacts:
        console.print("[yellow]no trained models[/]. Run `niftypulse train` first.")
        raise typer.Exit(code=0)

    table = Table(title="Trained models", header_style="bold cyan")
    for column in ("horizon", "trained at", "samples", "accuracy", "auc", "lift", "features"):
        table.add_column(column, justify="right")
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        table.add_row(
            f"{artifact['horizon']}m",
            str(artifact.get("trained_at", ""))[:19],
            f"{metrics.get('n', 0):,}",
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('auc', 0):.4f}",
            f"{metrics.get('lift', 0):+.4f}",
            str(artifact.get("features", 0)),
        )
    console.print(table)


# ------------------------------------------------------------------ backtest


@app.command()
def backtest(
    strategy: str = typer.Option("ensemble", help="Strategy name, 'ensemble', or 'ml'"),
    horizon: int = typer.Option(5, help="Holding period in bars"),
    threshold: float = typer.Option(0.15, help="Minimum conviction to enter"),
    notional: float = typer.Option(2_000_000.0, help="Target position notional in rupees"),
    lot_size: int = typer.Option(75, help="Contract lot size. Verify against the current NIFTY spec"),
    slippage: float = typer.Option(0.5, help="Slippage per side in basis points"),
    sweep: bool = typer.Option(False, help="Sweep entry thresholds"),
    offline: bool = typer.Option(False, "--offline", help="Backtest on synthetic data"),
) -> None:
    """Backtest a strategy against historical bars."""
    settings = _settings()
    bars = _bars(offline=offline, days=200, refresh=False, quiet=True)

    if bars.empty:
        console.print("[red]no bars available[/]. Run `niftypulse fetch` first.")
        raise typer.Exit(code=1)

    cost_model = CostModel(slippage_bps=slippage)
    config = BacktestConfig(
        horizon=horizon,
        entry_threshold=threshold,
        notional=notional,
        lot_size=lot_size,
        cost_model=cost_model,
    )

    scores, label = _resolve_scores(strategy, settings, bars, horizon)
    if scores is None or scores.empty:
        raise typer.Exit(code=1)

    lots = config.size_for(bars["close"].iloc[-1])[0]
    realised = lots * lot_size * bars["close"].iloc[-1]

    console.print(
        Panel(
            f"strategy    [bold]{label}[/]\n"
            f"bars        {len(bars):,}\n"
            f"horizon     {horizon} bars\n"
            f"threshold   {threshold}\n"
            f"position    {lots} lot(s) = Rs {realised:,.0f}\n"
            f"round-trip  [bold]{cost_model.round_trip_bps(realised):.2f} bps[/]",
            title="[cyan]backtest[/]",
            border_style="cyan",
        )
    )

    if sweep:
        table = run_threshold_sweep(bars, scores, config, strategy=label)
        console.print("\n[bold]Threshold sweep[/]")
        console.print(table.to_string())
        console.print(
            "\n[dim]Net expectancy should improve as the threshold rises. If it does not, "
            "the signal carries no information about its own confidence.[/]"
        )

    result = run_backtest(bars, scores, config, strategy=label)
    report = result.report

    console.print()
    for line in report.lines():
        console.print(f"  {line}")

    if report.trades:
        verdict = (
            "[green]positive net expectancy[/]" if report.expectancy_bps > 0
            else "[red]negative net expectancy[/]"
        )
        console.print(f"\n  verdict: {verdict}\n")


def _resolve_scores(strategy: str, settings, bars, horizon: int):
    """Build the conviction series to trade."""
    if strategy == "ml":
        oof_path = settings.model_dir / f"oof_{horizon}m.parquet"
        if not oof_path.exists():
            console.print(
                f"[red]no walk-forward predictions for {horizon}m[/]. "
                f"Run `niftypulse train` first."
            )
            return None, "ml"
        import pandas as pd

        oof = pd.read_parquet(oof_path)
        console.print(
            "[dim]Note: ML backtest uses walk-forward out-of-sample predictions, "
            "so these decisions come from models that never saw the bars being traded.[/]"
        )
        # The engine thresholds on signed conviction in [-1, 1], not on a
        # probability. Passing probabilities straight through would make every
        # bar clear any threshold below 0.5, since abs(0.48) looks like a strong
        # signal. Mapping to 2p - 1 also puts "no opinion" at exactly zero, which
        # is what the bars outside the walk-forward window get.
        conviction = 2.0 * oof["calibrated"] - 1.0
        return conviction, f"ml_{horizon}m"

    features = build_features(bars, bar_minutes=settings.bar_minutes, expiry_weekday=settings.expiry_weekday)
    context = StrategyContext(bars=bars, features=features)
    catalog = StrategyCatalog.from_kernel(_kernel())
    try:
        instance = catalog.get(strategy)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        return None, strategy

    return instance.score(context), instance.name


# ----------------------------------------------------------------- dashboard


@app.command()
def dashboard(
    offline: bool = typer.Option(False, "--offline", help="Replay simulated ticks instead of live"),
    timeframe: int = typer.Option(1, help="Bar size in minutes"),
    speed: float = typer.Option(1.0, help="Replay speed against the clock (1.0 = real time)"),
    refresh: float = typer.Option(1.0, help="Seconds between frames"),
    nowcast: float = typer.Option(1.0, help="Seconds between projection refreshes"),
    bars_ahead: int = typer.Option(3, help="How many candles to project"),
    legs: bool = typer.Option(
        True, "--legs/--no-legs", help="Chart the at-the-money call and put beside the index"
    ),
    workers: int | None = typer.Option(None, help="CPU workers (-1 = all available CPUs; default from config)"),
    learn: bool = typer.Option(True, "--learn/--no-learn", help="Update instrument models as candles close"),
    view: str = typer.Option("ai", help="Initial panel: research, costs, indicators, ai, or prediction"),
) -> None:
    """Run the dashboard: call, index and put, with the next three candles projected."""
    settings = _settings()
    if timeframe < 1 or bars_ahead < 1 or refresh <= 0 or nowcast <= 0 or speed <= 0:
        raise typer.BadParameter("Timeframe, bars ahead, refresh, nowcast and speed must be positive.")
    if view not in {"research", "costs", "indicators", "ai", "prediction"}:
        raise typer.BadParameter("View must be research, costs, indicators, ai or prediction.")
    if workers is not None:
        settings.workers = workers
    settings.online_learning = learn
    settings.bar_minutes = timeframe
    if bars_ahead:
        settings.plugin_config.setdefault("forecast:projection", {})["bars_ahead"] = bars_ahead

    session = Session(
        kernel=Kernel.bootstrap(settings),
        config=SessionConfig(
            offline=offline,
            timeframe=timeframe,
            refresh=refresh,
            nowcast_interval=nowcast,
            speed=speed,
            legs=legs,
        ),
    )
    session.renderer.view = view

    if session.live:
        console.print("[cyan]live Upstox feed[/] — ticks stream as they print\n")
    else:
        console.print(
            "[cyan]simulated feed[/] — generated ticks, paced against the clock. "
            "Set Upstox credentials for live data.\n"
        )
    try:
        session.bootstrap(progress=lambda message: console.print(f"[dim]{message}[/]"))
        console.print(f"[dim]{session.describe()}[/]")
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from None
    finally:
        console.print("[green]dashboard stopped[/]")


@app.command()
def snapshot(
    rows: int = typer.Option(60, help="Bars to show in the chart"),
    bars_ahead: int = typer.Option(3, help="How many candles to project"),
    offline: bool = typer.Option(True, "--offline/--live", help="Use simulated bars"),
    timeframe: int = typer.Option(1, help="Bar size in minutes"),
    legs: bool = typer.Option(True, "--legs/--no-legs", help="Show the call/index/put board"),
) -> None:
    """Render one frame to stdout, with the projected candles, and exit.

    Useful when there is no interactive terminal — in a pipe, in CI, or when
    checking what a board looks like without waiting for a clock.
    """
    settings = _settings()
    settings.bar_minutes = timeframe
    if bars_ahead:
        settings.plugin_config.setdefault("forecast:projection", {})["bars_ahead"] = bars_ahead

    session = Session(
        kernel=Kernel.bootstrap(settings),
        config=SessionConfig(offline=offline, timeframe=timeframe, days=3, legs=legs, record=False, research=False),
    )
    session.bootstrap()

    frame = session.snapshot()
    if isinstance(frame, BoardSnapshot):
        for leg in frame.legs:
            leg.snapshot.candles = leg.snapshot.candles.tail(rows)
        session.renderer.show_board(frame, session.engine.status)
    else:
        frame.candles = frame.candles.tail(rows)
        session.renderer.show(frame, session.engine.status)

    console.print(f"[dim]{session.describe()}[/]")
    stats = session.engine.forecaster
    if stats is not None:
        console.print(f"[dim]projection: {stats.summary} · {stats.tracker.summary()}[/]")
    if session.board:
        session.board.close()
    else:
        session.engine.close()
    session.broker.close()


# ---------------------------------------------------------------- strategies


@app.command()
def strategies() -> None:
    """List every available strategy."""
    settings = _settings()
    kernel = _kernel()
    bars = BarLoader(kernel).load(days=20, refresh=False, quiet=True, offline=True)

    features = build_features(bars, bar_minutes=settings.bar_minutes)
    context = StrategyContext(bars=bars, features=features)

    catalog = StrategyCatalog.from_kernel(kernel)

    table = Table(title="Strategies", header_style="bold cyan")
    for column in ("name", "pack", "category", "fires", "description"):
        table.add_column(column)

    for instance in catalog.all():
        scores = instance.score(context)
        fires = (scores.abs() > 0.15).mean() * 100
        pack = catalog.pack_of(instance.name)
        table.add_row(
            instance.name,
            pack.name if pack else "—",
            instance.category,
            f"{fires:.1f}%",
            instance.description,
        )
    console.print(table)

    for handle, reason in catalog.skipped.items():
        console.print(f"[yellow]skipped[/] {handle}: {reason}")
    console.print("[dim]'fires' is the share of bars with conviction above the entry threshold.[/]")


# ------------------------------------------------------------------- plugins


@app.command()
def plugins(
    kind: str = typer.Option("", help="Filter by kind: source, strategy, forecast, renderer, ..."),
    capabilities: bool = typer.Option(
        False, "--capabilities", help="Show which plugin provides which capability"
    ),
) -> None:
    """List every plugin the kernel discovers, bundled and installed."""
    kernel = Kernel.bootstrap(_settings())
    entries = kernel.entries(kind or None)

    if not entries:
        console.print(
            f"[yellow]no plugins found[/] under {BUILTIN_PACKAGE}"
            + (f" for kind {kind!r}" if kind else "")
        )
    else:
        table = Table(title="Plugins", header_style="bold cyan")
        for column in ("handle", "origin", "provides", "description"):
            table.add_column(column)
        for entry in entries:
            manifest = entry.manifest
            table.add_row(
                f"[bold]{entry.handle}[/]",
                "[grey62]bundled[/]" if not entry.is_third_party else "[bright_cyan]installed[/]",
                ", ".join(manifest.provides) or "[grey42]—[/]",
                manifest.description,
            )
        console.print(table)

    if capabilities:
        table = Table(title="Capabilities", header_style="bold cyan")
        table.add_column("capability")
        table.add_column("provided by")
        for name, handle in sorted(kernel.registry.capabilities().items()):
            table.add_row(name, handle)
        console.print(table)

    problems = kernel.validate()
    if problems:
        console.print("\n[yellow]unresolved requirements[/]")
        for problem in problems:
            console.print(f"  [yellow]·[/] {problem}")

    if kernel.load_report.errors:
        console.print("\n[red]plugins that failed to load[/]")
        for module, error in kernel.load_report.errors:
            console.print(f"  [red]·[/] {module}: {error}")

    console.print(
        f"\n[dim]{len(kernel.registry)} plugins, "
        f"{len(kernel.registry.capabilities())} capabilities. "
        f"Drop a folder containing plugin.py into {BUILTIN_PACKAGE} to add one.[/]"
    )


# ---------------------------------------------------------------------- data


@app.command()
def data(
    verbose: bool = typer.Option(False, "--verbose", help="List every partition file"),
) -> None:
    """Show what is stored locally, without touching the network.

    Answers "do I have to fetch this again?" from the manifest, so it is instant
    on a store of any size.
    """
    settings = _settings()
    manifest = StoreManifest(settings.data_dir / "manifest.json")
    payload = manifest.read()
    datasets = payload.get("datasets", {})

    if not datasets:
        console.print(
            f"[yellow]nothing stored[/] under {settings.data_dir}\n"
            "[dim]run `niftypulse fetch` for historical bars, or `niftypulse sync` "
            "for everything a live account can give.[/]"
        )
        return

    table = Table(title=f"Stored data — {settings.data_dir}", header_style="bold cyan")
    for column in ("dataset", "instrument", "rows", "shards", "from", "to"):
        table.add_column(column)

    for name in sorted(datasets):
        for key, entry in sorted(datasets[name].items()):
            table.add_row(
                name,
                str(entry.get("instrument", key)),
                f"{int(entry.get('rows', 0)):,}",
                str(entry.get("sessions", entry.get("shards", ""))),
                str(entry.get("first", ""))[:16],
                str(entry.get("last", ""))[:16],
            )
    console.print(table)

    if verbose:
        for path in sorted(settings.data_dir.rglob("*.parquet")):
            size_kb = path.stat().st_size / 1024
            console.print(f"  [grey62]{path.relative_to(settings.data_dir)}[/]  {size_kb:,.0f} KB")

    total = sum(path.stat().st_size for path in settings.data_dir.rglob("*.parquet"))
    console.print(
        f"\n[dim]{len(list(settings.data_dir.rglob('*.parquet')))} partition files, "
        f"{total / 1_048_576:.1f} MB on disk. Manifest updated {payload.get('updated_at', 'never')}.[/]"
    )


@app.command()
def sync(
    days: int = typer.Option(400, help="Calendar days of 1-minute history to pull"),
    offline: bool = typer.Option(False, "--offline", help="Generate instead of pulling"),
) -> None:
    """Pull everything the account can give, and store it. Never fetches twice.

    The cache is checked first and only the missing tail is requested, so running
    this on every start costs one API call when there is nothing new, and nothing
    at all when the cache is current.
    """
    settings = _settings()
    kernel = Kernel.bootstrap(settings)
    broker = kernel.build("source:upstox")
    loader = BarLoader(kernel)

    if not offline and not broker.is_configured:
        console.print(
            "[yellow]no Upstox credentials[/] — pulling nothing. "
            "Run `niftypulse login` to fetch real data, or pass --offline to generate."
        )

    frames = loader.load(days=days, refresh=True, offline=offline)
    if frames.empty:
        console.print("[red]no candles retrieved[/]")
        raise typer.Exit(code=1)

    store = loader.kernel.capability("history").store
    first, last = store.coverage()
    console.print(
        Panel(
            f"bars       [bold]{len(frames):,}[/]\n"
            f"sessions   [bold]{store.sessions():,}[/]\n"
            f"range      {first:%Y-%m-%d %H:%M} -> {last:%Y-%m-%d %H:%M}\n"
            f"partitions {store.base.relative_to(settings.data_dir)}/YYYY/MM/DD.parquet",
            title="[green]candles stored[/]",
            border_style="green",
        )
    )

    if broker.is_configured and not offline:
        source = kernel.capability("option_chain")
        chain = source.chain(frames["close"].iloc[-1], bars=frames)
        totals = chain.totals()
        console.print(
            Panel(
                f"expiry     [bold]{chain.expiry:%a %d %b %Y}[/]\n"
                f"strikes    {len(chain.strikes())} around {chain.atm_strike:,.0f}\n"
                f"PCR        {totals['pcr']:.2f}  "
                f"(CE OI {totals['call_oi']:,.0f} / PE OI {totals['put_oi']:,.0f})\n"
                f"snapshot   written to data/chain/...",
                title="[green]option chain read[/]",
                border_style="green",
            )
        )
        _store_chain_snapshot(kernel, chain)

    console.print(
        "\n[dim]Ticks and chain history cannot be back-filled: the API publishes "
        "candles, not the tape that produced them. Both are recorded while a live "
        "session runs, so whatever is missing is missing for good — start a "
        "session with `niftypulse dashboard` to begin collecting.[/]"
    )


def _store_chain_snapshot(kernel, chain) -> None:
    """Write one chain snapshot, so the store has a starting point."""
    import time as _time

    from plugins.sources.history import (
        CHAIN,
        HOUR,
        ChainRecorder,
        PartitionedStore,
        normalize_frames,
    )

    settings = kernel.settings
    recorder = ChainRecorder(
        store=PartitionedStore(
            settings.data_dir,
            f"{settings.instrument_key}-chain",
            dataset=CHAIN,
            shard=HOUR,
            normalizer=normalize_frames,
        )
    )
    recorder.record(chain, _time.monotonic())
    recorder.close()


# -------------------------------------------------------------------- doctor


@app.command()
def doctor(live: bool = typer.Option(False, "--live", help="Validate credentials and feed authorization with Upstox")) -> None:
    """Check the environment, credentials, and cached data."""
    settings = _settings()
    rows: list[tuple[str, str, str]] = []
    provider_market_status = None

    rows.append(("python", sys.version.split()[0], "ok"))

    from plugins.forecasts.ml_ensemble.models import lightgbm_available

    rows.append(
        (
            "lightgbm",
            "available" if lightgbm_available() else "unavailable (brew install libomp)",
            "ok" if lightgbm_available() else "warn",
        )
    )

    kernel = _kernel()
    broker = kernel.build("source:upstox")
    token = broker.token
    rows.append(
        ("upstox token", "present" if token else "missing", "ok" if token else "warn")
    )
    if live:
        from plugins.sources.upstox.feed import authorize_feed_url

        try:
            selected = broker.validate_auth()
            authorize_feed_url(selected)
            provider_market_status = broker.rest().market_status()
            rows.append(("live authorization", broker.auth_status + "; feed accepted", "ok"))
        except Exception as exc:
            rows.append(("live authorization", str(exc), "warn"))
        finally:
            broker.close()
    rows.append(("CPU workers", f"{settings.worker_count} / all available", "ok"))

    if settings.candles_path.exists():
        import pandas as pd

        legacy = len(pd.read_parquet(settings.candles_path, columns=["close"]))
        rows.append(
            ("legacy candle file", f"{legacy:,} bars — run `niftypulse sync` to import", "warn")
        )

    manifest = StoreManifest(settings.data_dir / "manifest.json")
    stored = manifest.read().get("datasets", {})
    candles = stored.get("candles", {})
    entry = next(iter(candles.values()), {})
    if entry:
        rows.append(
            (
                "partitioned store",
                f"{int(entry.get('rows', 0)):,} bars, {entry.get('sessions', 0)} shards",
                "ok",
            )
        )
    if stored.get("ticks"):
        tick_entry = next(iter(stored["ticks"].values()), {})
        rows.append(("recorded ticks", f"{int(tick_entry.get('rows', 0)):,}", "ok"))
    if stored.get("chain"):
        chain_entry = next(iter(stored["chain"].values()), {})
        rows.append(("chain samples", f"{int(chain_entry.get('rows', 0)):,}", "ok"))

    manifest = StoreManifest(settings.data_dir / "manifest.json")
    stored = manifest.read().get("datasets", {})
    candles = stored.get("candles", {})
    entry = next(iter(candles.values()), {})
    rows.append(
        (
            "stored bars",
            f"{int(entry.get('rows', 0)):,} in {entry.get('sessions', 0)} shards"
            if entry
            else "none",
            "ok" if entry else "warn",
        )
    )
    if stored.get("ticks"):
        tick_entry = next(iter(stored["ticks"].values()), {})
        rows.append(("recorded ticks", f"{int(tick_entry.get('rows', 0)):,}", "ok"))

    artifacts = list_artifacts(settings)
    rows.append(
        (
            "trained models",
            ", ".join(f"{a['horizon']}m" for a in artifacts) if artifacts else "none",
            "ok" if artifacts else "warn",
        )
    )

    calendar = TradingCalendar()
    open_now = calendar.is_open()
    rows.append(("market (Upstox)", provider_market_status, "ok" if provider_market_status == "NORMAL_OPEN" else "dim")
                if provider_market_status else
                ("session clock", "open hours (holiday status not checked)" if open_now else "outside open hours", "dim"))

    table = Table(title=f"niftypulse {__version__} — environment", header_style="bold cyan")
    for column in ("check", "value", "status"):
        table.add_column(column)
    for name, value, status in rows:
        colour = {"ok": "green", "warn": "yellow", "dim": "grey62"}[status]
        table.add_row(name, value, f"[{colour}]{status}[/]")
    console.print(table)


@app.command()
def research(
    offline: bool = typer.Option(False, "--offline", help="Inspect simulation outcomes instead of live"),
    failures: bool = typer.Option(False, "--failures", help="Show recent misses and their original context"),
    limit: int = typer.Option(100, min=1, max=1000),
    section: str = typer.Option(
        "all", help="Report section: all, forecasts, ai, predictions, or strategies"
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable report for further research"),
) -> None:
    """Print durable forecast, AI, and strategy research into terminal scrollback."""
    import json
    import sqlite3

    selected = section.strip().lower()
    allowed = {"all", "forecasts", "ai", "predictions", "strategies"}
    if selected not in allowed:
        raise typer.BadParameter(f"section must be one of: {', '.join(sorted(allowed))}")

    settings = _settings()
    mode = "simulation" if offline else "live"
    research_dir = settings.data_dir / "research"
    payload: dict[str, object] = {"mode": mode}

    if selected in {"all", "forecasts"}:
        path = research_dir / f"{mode}.sqlite3"
        records: list[dict] = []
        if path.exists():
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                if failures:
                    query = """SELECT instrument,timeframe,target,horizon,
                        predicted_close,actual_close,error_bps,context FROM forecasts
                        WHERE status='miss' ORDER BY target DESC LIMIT ?"""
                    records = [dict(row) for row in db.execute(query, (limit,))]
                else:
                    query = """SELECT instrument,timeframe,horizon,
                        SUM(status IN ('hit','miss')) AS scored, SUM(status='hit') AS wins,
                        SUM(status='miss') AS misses, ROUND(100.0*SUM(status='hit')/
                        NULLIF(SUM(status IN ('hit','miss')),0),2) AS hit_pct,
                        ROUND(AVG(error_bps),3) AS mae_bps,
                        SUM(status IN ('missing_bar','unresolved','pending')) AS unscored
                        FROM forecasts GROUP BY instrument,timeframe,horizon
                        ORDER BY instrument,timeframe,horizon"""
                    records = [dict(row) for row in db.execute(query)]
        payload["forecast_failures" if failures else "forecasts"] = records

    if selected in {"all", "ai", "predictions"}:
        from plugins.forecasts.online_research import OnlineResearchLab

        scorecards: list[dict] = []
        latest: list[dict] = []
        for state_path in sorted((research_dir / "online_ai" / mode).glob("*.sqlite3")):
            lab = OnlineResearchLab(state_path)
            ledger = lab.records()
            instrument = ledger[-1].instrument if ledger else state_path.stem
            for card in lab.scorecards():
                all_time = card.all_time
                rolling = card.rolling
                scorecards.append(
                    {
                        "instrument": instrument,
                        "algorithm": card.algorithm,
                        "samples": all_time.sample_count,
                        "accuracy": round(all_time.accuracy * 100.0, 2),
                        "lift": round(all_time.accuracy_lift * 100.0, 2),
                        "rolling_accuracy": round(rolling.accuracy * 100.0, 2),
                        "brier": round(all_time.brier_score, 4),
                        "calibration_error": round(all_time.calibration_error, 4),
                        "trades": all_time.trade_count,
                        "wins": all_time.wins,
                        "losses": all_time.losses,
                        "profit_bps": round(all_time.profit_bps, 2),
                        "loss_bps": round(all_time.loss_bps, 2),
                        "net_pnl_bps": round(all_time.net_pnl_bps, 2),
                        "drawdown_bps": round(all_time.max_drawdown_bps, 2),
                        "trust": round(card.trust_score * 100.0, 2),
                        "pending": lab.pending_count,
                        "expired": lab.expired_count,
                    }
                )
            seen: set[tuple[str, int]] = set()
            for record in reversed(ledger):
                key = (record.algorithm, record.horizon_min)
                if key in seen:
                    continue
                seen.add(key)
                latest.append(
                    {
                        "instrument": record.instrument or instrument,
                        "algorithm": record.algorithm,
                        "horizon": record.horizon_min,
                        "issued_at": record.issued_at.isoformat(),
                        "target_at": record.target_at.isoformat(),
                        "action": record.action.value,
                        "p_up": round(record.p_up * 100.0, 2),
                        "confidence": round(record.confidence * 100.0, 2),
                        "trust": round(record.trust_at_issue * 100.0, 2),
                        "status": record.status,
                    }
                )
        if selected in {"all", "ai"}:
            payload["ai_scorecards"] = scorecards[:limit]
        if selected in {"all", "ai", "predictions"}:
            payload["latest_ai_predictions"] = latest[:limit]

    if selected in {"all", "strategies"}:
        from plugins.advisory.performance_ledger import StrategyPerformanceLedger

        strategy_path = research_dir / "strategies.sqlite3"
        strategy_rows: list[dict] = []
        counts: dict[str, int] = {}
        if strategy_path.exists():
            ledger = StrategyPerformanceLedger(strategy_path)
            counts = ledger.counts()
            for card in ledger.scorecards():
                strategy_rows.append(
                    {
                        "instrument": card.instrument,
                        "strategy": card.strategy,
                        "samples": card.samples,
                        "accuracy": round(card.accuracy * 100.0, 2),
                        "wins": card.wins,
                        "losses": card.losses,
                        "gross_pnl_bps": round(card.gross_pnl_bps, 2),
                        "cost_bps": round(card.cost_paid_bps, 2),
                        "profit_bps": round(card.profit_bps, 2),
                        "loss_bps": round(card.loss_bps, 2),
                        "net_pnl_bps": round(card.net_pnl_bps, 2),
                        "mean_net_bps": round(card.mean_net_pnl_bps, 2),
                        "drawdown_bps": round(card.max_drawdown_bps, 2),
                        "trust": round(card.trust_score * 100.0, 2),
                    }
                )
        strategy_rows.sort(key=lambda row: (str(row["instrument"]), -float(row["trust"])))
        payload["strategy_scorecards"] = strategy_rows[:limit]
        payload["strategy_outcomes"] = counts

    if json_output:
        console.print_json(json.dumps(payload))
        return

    rendered = False
    views = (
        (
            "forecasts",
            "Forward outcomes · first-issued projected candles",
            (
                ("instrument", "instrument"),
                ("timeframe", "tf"),
                ("horizon", "horizon"),
                ("scored", "n"),
                ("wins", "wins"),
                ("misses", "misses"),
                ("hit_pct", "hit %"),
                ("mae_bps", "MAE bp"),
                ("unscored", "unscored"),
            ),
        ),
        ("forecast_failures", "Forecast misses · frozen issue context", None),
        (
            "ai_scorecards",
            "Online AI · causal quality and trust",
            (
                ("instrument", "instrument"),
                ("algorithm", "algorithm"),
                ("samples", "n"),
                ("accuracy", "acc %"),
                ("lift", "lift %"),
                ("rolling_accuracy", "roll %"),
                ("brier", "Brier"),
                ("calibration_error", "ECE"),
                ("trust", "trust %"),
            ),
        ),
        (
            "ai_scorecards",
            "Online AI · hypothetical economics after costs",
            (
                ("instrument", "instrument"),
                ("algorithm", "algorithm"),
                ("trades", "trades"),
                ("wins", "W"),
                ("losses", "L"),
                ("profit_bps", "profit bp"),
                ("loss_bps", "loss bp"),
                ("net_pnl_bps", "net bp"),
                ("drawdown_bps", "DD bp"),
                ("pending", "pending"),
                ("expired", "expired"),
            ),
        ),
        (
            "latest_ai_predictions",
            "Latest AI research predictions · no broker execution",
            (
                ("instrument", "instrument"),
                ("algorithm", "algorithm"),
                ("horizon", "horizon"),
                ("target_at", "target"),
                ("action", "view"),
                ("p_up", "up %"),
                ("confidence", "conf %"),
                ("trust", "trust %"),
                ("status", "status"),
            ),
        ),
        (
            "strategy_scorecards",
            "Strategies · causal quality and trust",
            (
                ("instrument", "instrument"),
                ("strategy", "strategy"),
                ("samples", "n"),
                ("accuracy", "acc %"),
                ("wins", "W"),
                ("losses", "L"),
                ("trust", "trust %"),
            ),
        ),
        (
            "strategy_scorecards",
            "Strategies · hypothetical one-unit economics after costs",
            (
                ("instrument", "instrument"),
                ("strategy", "strategy"),
                ("gross_pnl_bps", "gross bp"),
                ("cost_bps", "cost bp"),
                ("profit_bps", "profit bp"),
                ("loss_bps", "loss bp"),
                ("net_pnl_bps", "net bp"),
                ("mean_net_bps", "mean bp"),
                ("drawdown_bps", "DD bp"),
            ),
        ),
    )
    for name, title, columns in views:
        rows = payload.get(name)
        if not isinstance(rows, list) or not rows:
            continue
        table = Table(title=title, header_style="bold cyan", show_lines=name.endswith("scorecards"))
        selected_columns = columns or tuple((column, column) for column in rows[0])
        for column, label in selected_columns:
            table.add_column(
                label,
                overflow="fold",
                no_wrap=column not in {"context", "instrument", "strategy", "algorithm"},
            )
        for row in rows:
            table.add_row(
                *(
                    str(row.get(column)) if row.get(column) is not None else "—"
                    for column, _label in selected_columns
                )
            )
        console.print(table)
        rendered = True

    counts = payload.get("strategy_outcomes")
    if isinstance(counts, dict) and counts:
        console.print(
            "[cyan]strategy ledger[/] "
            + " · ".join(f"{status} {count:,}" for status, count in sorted(counts.items()))
        )
        rendered = True
    if not rendered:
        console.print("No recorded research outcomes yet. Start niftypulse dashboard.")
        return
    console.print(
        "[dim]Accuracy and trust use only exact target bars; missed targets expire. "
        "P&L is a hypothetical directional return in basis points after configured costs, "
        "not broker fills or guaranteed profit. Use --section and --limit for terminal scrollback, "
        "or --json for analysis.[/]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
