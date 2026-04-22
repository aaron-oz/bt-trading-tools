"""
bt_trading_tools.backtest — Generic backtesting engine for Bittensor strategies.

The engine is data-agnostic: callers provide a sequence of TickData,
and a Strategy that decides what to do each tick. The engine handles
AMM execution, position tracking, P&L, and writes the same TradeLog /
PortfolioLog as live bots — one analysis pipeline for both.

Usage::

    from bt_trading_tools.backtest import BacktestEngine, TickData, Order

    engine = BacktestEngine(capital=100.0, bot_name="my_strategy")
    results = engine.run(ticks, strategy)
    print(results.stats)
"""

from bt_trading_tools.backtest.engine import BacktestEngine, BacktestResults
from bt_trading_tools.backtest.types import (
    Order,
    Position,
    TickData,
    SubnetTick,
    Strategy,
)
from bt_trading_tools.backtest.stats import (
    compute_stats,
    compute_regime_stats,
    validate_cv_results,
    BacktestStats,
    RegimeStats,
    ValidationCheck,
    ValidationReport,
)
from bt_trading_tools.backtest.cv import PurgedWalkForwardCV, CVFold
from bt_trading_tools.backtest.scheduled import ScheduledStrategy

__all__ = [
    "BacktestEngine",
    "BacktestResults",
    "Order",
    "Position",
    "TickData",
    "SubnetTick",
    "Strategy",
    "BacktestStats",
    "compute_stats",
    "compute_regime_stats",
    "validate_cv_results",
    "RegimeStats",
    "ValidationCheck",
    "ValidationReport",
    "PurgedWalkForwardCV",
    "CVFold",
    "ScheduledStrategy",
]
