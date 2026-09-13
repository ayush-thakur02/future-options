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
from core.version import __version__
from kernel import BUILTIN_PACKAGE, Kernel
from live import LiveEngine
from plugins.features.technical import FEATURE_GROUPS, build_features, feature_columns
from plugins.forecasts.ml_ensemble.predictor import Predictor
from plugins.forecasts.ml_ensemble.trainer import (
    Trainer,
    list_artifacts,
    render_report,
    render_scalping_hurdle,
    save_training_metadata,
)
from runtime.bars import BarLoader
from strategies import build_strategy, default_ensemble
from strategies.base import StrategyContext

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
    try:
        instance = build_strategy(strategy)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        return None, strategy

    return instance.score(context), instance.name


# ----------------------------------------------------------------- dashboard


@app.command()
def dashboard(
    offline: bool = typer.Option(False, "--offline", help="Replay synthetic ticks instead of live"),
    timeframe: int = typer.Option(1, help="Bar size in minutes"),
    speed: float = typer.Option(0.02, help="Replay delay between bars, in seconds"),
    refresh: float = typer.Option(1.0, help="Dashboard refresh interval"),
) -> None:
    """Run the live terminal dashboard."""
    settings = _settings()
    settings.bar_minutes = timeframe
    kernel = Kernel.bootstrap(settings)

    calendar = TradingCalendar()
    predictor = Predictor(settings).load()
    if predictor.is_ready:
        console.print(f"[dim]models loaded: {predictor.describe()}[/]")
    else:
        console.print("[yellow]no trained models — forecasts disabled. Run `niftypulse train`.[/]")

    engine = LiveEngine(
        settings,
        predictor=predictor,
        strategy=default_ensemble(),
        bar_minutes=timeframe,
        calendar=calendar,
    )

    broker = kernel.build("source:upstox")
    if offline or not broker.is_configured:
        _run_offline_dashboard(kernel, engine, speed, refresh, timeframe)
    else:
        _run_live_dashboard(kernel, engine, refresh, timeframe)


def _run_offline_dashboard(kernel, engine, speed: float, refresh: float, timeframe: int) -> None:
    from ui.dashboard import Dashboard

    console.print("[cyan]offline replay mode[/] — set up Upstox credentials for live data\n")
    market = kernel.build("source:simulated", days=5)
    bars = market.candles(days=5, bar_minutes=timeframe)
    engine.bootstrap(history=bars)
    ticks = market.ticks(bars, ticks_per_bar=6)

    dash = Dashboard(symbol=engine.settings.symbol, timeframe=f"{timeframe}m", refresh=refresh)
    asyncio.run(_replay_loop(engine, ticks, dash, speed))


async def _replay_loop(engine, ticks, dash, speed: float) -> None:
    from rich.live import Live

    with Live(console=dash.console, screen=True, refresh_per_second=8, transient=False) as live:
        try:
            for snapshot in engine.replay(ticks):
                live.update(dash.build(snapshot, engine.status))
                await asyncio.sleep(speed)
        except KeyboardInterrupt:
            pass
    console.print("[green]replay finished[/]")


def _run_live_dashboard(kernel, engine, refresh: float, timeframe: int) -> None:
    from ui.dashboard import Dashboard

    bars = BarLoader(kernel).load(days=15, refresh=True)
    engine.bootstrap(history=bars)
    dash = Dashboard(symbol=engine.settings.symbol, timeframe=f"{timeframe}m", refresh=refresh)
    asyncio.run(_live_loop(engine, dash, refresh))


async def _live_loop(engine, dash, refresh: float) -> None:
    from rich.live import Live

    feed_task = asyncio.create_task(engine.run_live())
    with Live(console=dash.console, screen=True, refresh_per_second=2, transient=False) as live:
        try:
            while True:
                live.update(dash.build(engine.snapshot(), engine.status))
                await asyncio.sleep(refresh)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            feed_task.cancel()


# ---------------------------------------------------------------- strategies


@app.command()
def strategies() -> None:
    """List every available strategy."""
    settings = _settings()
    bars = _bars(offline=True, days=20, refresh=False, quiet=True)

    features = build_features(bars, bar_minutes=settings.bar_minutes)
    context = StrategyContext(bars=bars, features=features)

    table = Table(title="Strategies", header_style="bold cyan")
    for column in ("name", "category", "fires", "description"):
        table.add_column(column)

    from strategies import all_strategies

    for instance in all_strategies():
        scores = instance.score(context)
        fires = (scores.abs() > 0.15).mean() * 100
        table.add_row(
            instance.name,
            instance.category,
            f"{fires:.1f}%",
            instance.description,
        )
    console.print(table)
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


# -------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check the environment, credentials, and cached data."""
    settings = _settings()
    rows: list[tuple[str, str, str]] = []

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

    store_bars = 0
    if settings.candles_path.exists():
        import pandas as pd

        store_bars = len(pd.read_parquet(settings.candles_path, columns=["close"]))
    rows.append(("cached bars", f"{store_bars:,}", "ok" if store_bars else "warn"))

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
    rows.append(
        ("market", "open" if open_now else "closed", "ok" if open_now else "dim")
    )

    table = Table(title=f"niftypulse {__version__} — environment", header_style="bold cyan")
    for column in ("check", "value", "status"):
        table.add_column(column)
    for name, value, status in rows:
        colour = {"ok": "green", "warn": "yellow", "dim": "grey62"}[status]
        table.add_row(name, value, f"[{colour}]{status}[/]")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
