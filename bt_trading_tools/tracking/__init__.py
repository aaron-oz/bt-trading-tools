"""
bt_trading_tools.tracking — Auditability layer for Bittensor trading bots.

Canonical v1 trade log schema (see docs/trade_log_schema.md):
    TradeLogWriter       Non-blocking JSONL writer with queued validation.
    TradeRecord,
    MTMSample            Pydantic models for the two record types.
    Side, Status,
    Category             Enums used in trade records.
    SCHEMA_VERSION       int constant (currently 1).
    load_trade_log       Read a JSONL log into a DataFrame.
    iter_trade_log       Stream raw dicts from a JSONL log.
    validate_trade_log   Strict validation; returns list of ValidationIssue.
    compute_pnl          Downstream realized P&L honoring position_model.

Orthogonal logs (pre-existing, unchanged):
    DecisionLog          JSON-lines per-tick strategy replay.
    PortfolioLog         SQLite equity-curve time series.
    EventLog             JSON-lines significant events.

Legacy:
    TradeLog             Pre-v1 SQLite trade log with embedded cost-basis.
                         Kept for backward compatibility. New bots should use
                         TradeLogWriter; downstream P&L should use compute_pnl.
"""

from bt_trading_tools.tracking.schema import (
    SCHEMA_VERSION,
    AlphaYieldSource,
    Category,
    FeeSource,
    MTMSample,
    PortfolioSnapshot,
    Side,
    Status,
    TradeRecord,
)
from bt_trading_tools.tracking.writer import TradeLogWriter
from bt_trading_tools.tracking.reader import (
    ValidationIssue,
    iter_trade_log,
    load_trade_log,
    validate_trade_log,
)
from bt_trading_tools.tracking.pnl import (
    PnLBasis,
    PositionModel,
    PositionPnL,
    compute_pnl,
)
from bt_trading_tools.tracking.metrics import (
    drawdown_series,
    equity_series,
    max_drawdown,
    sharpe,
)
from bt_trading_tools.tracking.fee_receipt import (
    FEE_RECEIPT_SCHEMA_VERSION,
    FeeReceipt,
    FeeReceiptWriter,
    iter_fee_receipts,
    validate_fee_receipt_log,
)

from bt_trading_tools.tracking.trade_log import TradeLog
from bt_trading_tools.tracking.decision_log import DecisionLog
from bt_trading_tools.tracking.portfolio_log import PortfolioLog
from bt_trading_tools.tracking.event_log import EventLog

__all__ = [
    # v1 canonical
    "SCHEMA_VERSION",
    "TradeLogWriter",
    "TradeRecord",
    "MTMSample",
    "PortfolioSnapshot",
    "Side",
    "Status",
    "Category",
    "FeeSource",
    "AlphaYieldSource",
    "load_trade_log",
    "iter_trade_log",
    "validate_trade_log",
    "ValidationIssue",
    "compute_pnl",
    "PositionPnL",
    "PositionModel",
    "PnLBasis",
    "equity_series",
    "drawdown_series",
    "max_drawdown",
    "sharpe",
    "FEE_RECEIPT_SCHEMA_VERSION",
    "FeeReceipt",
    "FeeReceiptWriter",
    "iter_fee_receipts",
    "validate_fee_receipt_log",
    # Orthogonal
    "DecisionLog",
    "PortfolioLog",
    "EventLog",
    # Legacy
    "TradeLog",
]
