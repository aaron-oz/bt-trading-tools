"""
bt_trading_tools.tracking — Auditability layer for Bittensor trading bots.

Provides four complementary logs:
    TradeLog        SQLite trade records with cost-basis and P&L.
    DecisionLog     JSON-lines per-tick decision trail (full replay).
    PortfolioLog    SQLite time-series of portfolio value each tick.
    EventLog        JSON-lines significant events (trades, errors, restarts).
"""

from bt_trading_tools.tracking.trade_log import TradeLog
from bt_trading_tools.tracking.decision_log import DecisionLog
from bt_trading_tools.tracking.portfolio_log import PortfolioLog
from bt_trading_tools.tracking.event_log import EventLog

__all__ = ["TradeLog", "DecisionLog", "PortfolioLog", "EventLog"]
