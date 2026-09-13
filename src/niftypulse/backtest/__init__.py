"""Backtesting: execution, cost model, and performance reporting."""

from .costs import CostModel, CostSchedule, default_model, net_expectancy
from .engine import (
    BacktestConfig,
    BacktestResult,
    breakeven_threshold,
    run_backtest,
    run_threshold_sweep,
)
from .report import PerformanceReport, build_report, equity_curve

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostModel",
    "CostSchedule",
    "PerformanceReport",
    "breakeven_threshold",
    "build_report",
    "default_model",
    "equity_curve",
    "net_expectancy",
    "run_backtest",
    "run_threshold_sweep",
]
