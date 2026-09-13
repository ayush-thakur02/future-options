"""Configuration and settings resolution.

Settings come from (in order of precedence): explicit arguments, environment
variables, an optional YAML file at ``config/default.yaml``, then defaults.

Part of ``core``: the vocabulary and configuration every plugin shares. Nothing
here imports a plugin, which is what lets the plugin tree depend on ``core``
without any risk of a cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# NIFTY 50 index instrument keys as published by Upstox.
NIFTY50_KEY = "NSE_INDEX|Nifty 50"
NIFTYBANK_KEY = "NSE_INDEX|Nifty Bank"
INDIA_VIX_KEY = "NSE_INDEX|India VIX"

IST_OFFSET_HOURS = 5.5


def project_root() -> Path:
    """Directory that holds data/ and artifacts/, overridable via env."""
    env = os.getenv("NIFTYPULSE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


@dataclass
class UpstoxCredentials:
    """OAuth credentials for the Upstox API."""

    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str | None = None
    access_token: str | None = None

    @property
    def has_oauth(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.access_token)


@dataclass
class Settings:
    """Runtime settings for the whole platform."""

    instrument_key: str = NIFTY50_KEY
    symbol: str = "NIFTY 50"
    vix_key: str = INDIA_VIX_KEY
    banknifty_key: str = NIFTYBANK_KEY

    bar_minutes: int = 1
    # Scalping horizons. Everything is a few minutes at most — a 15-minute
    # forecast is position trading, not scalping, and averaging it into the same
    # view would blur the only timescale this platform is built for.
    horizons: tuple[int, ...] = (1, 2, 3, 5)

    # NSE moved NIFTY weekly expiry from Thursday to Tuesday. Kept configurable
    # because the exchange has changed this before and will change it again.
    expiry_weekday: int = 1

    session_open: time = time(9, 15)
    session_close: time = time(15, 30)
    pre_open_start: time = time(9, 0)

    # Model / labelling parameters
    deadband_atr: float = 0.12
    # Require labelled moves to clear the round-trip cost of trading them. A
    # scalping model trained on sub-cost moves is being taught to predict noise:
    # it can be right most of the time and still lose money on every correct
    # call. Set False only to study the raw signal in isolation from costs.
    enforce_cost_hurdle: bool = True
    # Multiplier on the cost hurdle when setting the labelling dead band. At 1.0
    # a labelled move merely breaks even; some margin is required for the trade
    # to be worth taking.
    hurdle_multiple: float = 1.5
    n_splits: int = 4
    embargo_bars: int = 20
    min_train_bars: int = 4000
    train_n_jobs: int = 4

    # Position sizing used for the cost hurdle and for backtests.
    reference_notional: float = 2_000_000.0
    lot_size: int = 75

    # Backtest assumptions (round-trip cost in basis points of notional)
    cost_bps: float = 0.6
    slippage_bps: float = 0.5

    root: Path = field(default_factory=project_root)
    data_dir: Path = field(default_factory=lambda: project_root() / "data")
    model_dir: Path = field(default_factory=lambda: project_root() / "artifacts")
    log_dir: Path = field(default_factory=lambda: project_root() / "logs")

    credentials: UpstoxCredentials = field(default_factory=UpstoxCredentials)

    # Per-plugin parameters, keyed by handle ("forecast:projection"). The kernel
    # merges these into the kwargs a plugin's build() receives, so tuning a
    # plugin is a config edit rather than a code change.
    plugin_config: dict[str, dict] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.model_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def candles_path(self) -> Path:
        return self.data_dir / "candles.parquet"

    def cost_hurdle_bps(self) -> float:
        """Round-trip cost of a scalp, in basis points.

        The number that decides everything on a scalping timescale. A signal is
        only worth acting on if the move it forecasts is larger than this, and a
        model is only worth training if the moves it is asked to classify are
        larger than this too.
        """
        from backtest.costs import CostModel

        return CostModel(slippage_bps=self.slippage_bps).round_trip_bps(self.reference_notional)

    def labelling_hurdle_bps(self) -> float:
        """Minimum move size eligible for a training label, in basis points."""
        if not self.enforce_cost_hurdle:
            return 0.0
        return self.cost_hurdle_bps() * self.hurdle_multiple

    @property
    def ticks_path(self) -> Path:
        return self.data_dir / "ticks.parquet"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "upstox_token.json"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def load_settings(config_path: Path | None = None) -> Settings:
    """Build :class:`Settings` from env, YAML, and defaults."""
    load_dotenv(override=False)
    root = project_root()

    raw = _load_yaml(config_path or root / "config" / "default.yaml")
    data_cfg = raw.get("data", {})
    model_cfg = raw.get("model", {})
    backtest_cfg = raw.get("backtest", {})
    plugins_cfg = raw.get("plugins", {}) or {}

    credentials = UpstoxCredentials(
        client_id=os.getenv("UPSTOX_CLIENT_ID"),
        client_secret=os.getenv("UPSTOX_CLIENT_SECRET"),
        redirect_uri=os.getenv("UPSTOX_REDIRECT_URI"),
        access_token=os.getenv("UPSTOX_ACCESS_TOKEN"),
    )

    horizons = model_cfg.get("horizons")
    settings = Settings(
        instrument_key=data_cfg.get("instrument_key", NIFTY50_KEY),
        symbol=data_cfg.get("symbol", "NIFTY 50"),
        bar_minutes=int(data_cfg.get("bar_minutes", 1)),
        horizons=tuple(int(h) for h in horizons) if horizons else Settings.horizons,
        deadband_atr=float(model_cfg.get("deadband_atr", Settings.deadband_atr)),
        enforce_cost_hurdle=_as_bool(
            model_cfg.get("enforce_cost_hurdle", Settings.enforce_cost_hurdle)
        ),
        hurdle_multiple=float(model_cfg.get("hurdle_multiple", Settings.hurdle_multiple)),
        n_splits=int(model_cfg.get("n_splits", Settings.n_splits)),
        embargo_bars=int(model_cfg.get("embargo_bars", Settings.embargo_bars)),
        min_train_bars=int(model_cfg.get("min_train_bars", Settings.min_train_bars)),
        train_n_jobs=int(model_cfg.get("train_n_jobs", Settings.train_n_jobs)),
        reference_notional=float(
            model_cfg.get("reference_notional", Settings.reference_notional)
        ),
        lot_size=int(model_cfg.get("lot_size", Settings.lot_size)),
        cost_bps=float(backtest_cfg.get("cost_bps", Settings.cost_bps)),
        slippage_bps=float(backtest_cfg.get("slippage_bps", Settings.slippage_bps)),
        root=root,
        data_dir=root / data_cfg.get("dir", "data"),
        model_dir=root / model_cfg.get("dir", "artifacts"),
        log_dir=root / "logs",
        credentials=credentials,
        plugin_config={
            str(handle): dict(values or {}) for handle, values in plugins_cfg.items()
        },
    )
    return settings


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
