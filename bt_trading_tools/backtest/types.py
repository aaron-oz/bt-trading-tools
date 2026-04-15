"""
Shared types for the backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SubnetTick:
    """Per-subnet market data for one tick."""
    netuid: int
    price: float          # TAO per alpha
    tao_pool: float       # TAO liquidity in AMM pool
    alpha_pool: float     # Alpha liquidity in AMM pool
    # Strategy-specific signal columns (emission_share, pct_change, etc.)
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class TickData:
    """All market data for one point in time."""
    timestamp: int                           # unix timestamp
    subnets: dict[int, SubnetTick]           # netuid → SubnetTick
    # Global signals (TAO/USD, market regime, etc.)
    global_signals: dict[str, float] = field(default_factory=dict)


@dataclass
class Position:
    """An open position."""
    netuid: int
    entry_price: float
    alpha_qty: float
    tao_cost: float          # total TAO spent
    entry_time: int          # unix timestamp
    entry_fees: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """A trade instruction from a strategy.

    The engine will execute it using the AMM model.
    """
    netuid: int
    side: str                # "buy" or "sell"
    tao_amount: float = 0.0  # TAO to spend (for buys)
    alpha_amount: float = 0.0 # Alpha to sell (for sells); 0 = sell all
    reason: str = ""
    signal_data: dict[str, Any] | None = None
    # Price caps — critical for preventing TP overshoot with execution delay.
    # Without these, a TP sell delayed to a later tick would execute at the
    # market price instead of the TP target.  See backtest_bugs_mar20.md
    # Bug #4: this bug caused +164,605% phantom returns.
    limit_price: float | None = None  # For sells: cap effective price (TP overshoot prevention)
                                       # For buys: max price willing to pay (slippage protection)


@runtime_checkable
class Strategy(Protocol):
    """Protocol that all backtest strategies must implement.

    The engine calls ``on_tick`` each timestep. The strategy returns
    a list of Orders to execute (can be empty).
    """

    def on_tick(
        self,
        tick: TickData,
        positions: dict[int, Position],
        capital: float,
        portfolio_value: float,
    ) -> list[Order]:
        """Evaluate market data and return trade orders.

        Args:
            tick: Current market data for all subnets.
            positions: Current open positions {netuid: Position}.
            capital: Available cash (TAO).
            portfolio_value: Total value (cash + positions MTM).

        Returns:
            List of Order objects to execute this tick.
        """
        ...
