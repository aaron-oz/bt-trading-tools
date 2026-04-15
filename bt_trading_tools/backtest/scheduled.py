"""
ScheduledStrategy — strategy wrapper with time-scheduled parameter changes.

Enables rolling walk-forward optimization with a single continuous backtest:
positions carry over between parameter windows naturally (the engine tracks
positions, not the strategy).

Supports two exit modes:
- "adopt" (default): all decisions use the current window's params.
- "sticky": exits use the birth-window's params (the params that were active
  when the position was opened). Entries always use current params.

Usage::

    from bt_trading_tools.backtest import BacktestEngine
    from bt_trading_tools.backtest.scheduled import ScheduledStrategy

    # Phase 1: optimize per window → build schedule
    schedule = [
        (window1_ts, params1),
        (window2_ts, params2),
        ...
    ]

    # Phase 2: one continuous backtest with position carry-over
    wrapped = ScheduledStrategy(
        strategy_factory=MyStrategy,
        schedule=schedule,
        exit_mode="adopt",
    )
    results = engine.run(all_ticks, wrapped)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from bt_trading_tools.backtest.types import Order, Position, Strategy, TickData


class ScheduledStrategy:
    """Strategy wrapper with time-scheduled parameter changes.

    Wraps a parameterized strategy so that parameters change at scheduled
    timestamps while positions carry over continuously between windows.

    Args:
        strategy_factory: Callable that takes a params object and returns
            a Strategy instance. Called once per schedule entry.
        schedule: List of (unix_timestamp, params) tuples. The first
            entry's params apply from the start of the backtest.
            Will be sorted by timestamp.
        exit_mode: How to evaluate exits for carried-over positions:
            - ``"adopt"`` — all decisions use the current window's params.
              Simple and usually the right default.
            - ``"sticky"`` — each position uses its birth-window's params
              for exit evaluation. Entries always use current params.
              Useful for comparing whether adaptive exits help or hurt.
    """

    def __init__(
        self,
        strategy_factory: Callable[[Any], Strategy],
        schedule: list[tuple[int, Any]],
        exit_mode: str = "adopt",
    ):
        if not schedule:
            raise ValueError("schedule must be non-empty")
        if exit_mode not in ("adopt", "sticky"):
            raise ValueError(f"exit_mode must be 'adopt' or 'sticky', got {exit_mode!r}")

        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.exit_mode = exit_mode
        self._strategies = [strategy_factory(params) for _, params in self.schedule]
        self._birth_idx: dict[int, int] = {}  # netuid → schedule index at entry

    def _current_idx(self, timestamp: int) -> int:
        """Find which schedule entry is active at this timestamp."""
        idx = 0
        for i, (ts, _) in enumerate(self.schedule):
            if ts <= timestamp:
                idx = i
            else:
                break
        return idx

    def on_tick(
        self,
        tick: TickData,
        positions: dict[int, Position],
        capital: float,
        portfolio_value: float,
    ) -> list[Order]:
        cur_idx = self._current_idx(tick.timestamp)

        if self.exit_mode == "adopt":
            orders = self._strategies[cur_idx].on_tick(
                tick, positions, capital, portfolio_value,
            )
        else:
            orders = self._sticky_on_tick(
                tick, positions, capital, portfolio_value, cur_idx,
            )

        # Track birth/exit of positions for sticky mode bookkeeping.
        # Safe to run in adopt mode too (just unused overhead).
        for o in orders:
            if o.side == "buy":
                self._birth_idx[o.netuid] = cur_idx
            elif o.side == "sell":
                self._birth_idx.pop(o.netuid, None)

        return orders

    def _sticky_on_tick(
        self,
        tick: TickData,
        positions: dict[int, Position],
        capital: float,
        portfolio_value: float,
        cur_idx: int,
    ) -> list[Order]:
        """Sticky mode: exits use birth-window params, entries use current.

        1. Group positions by their birth-window index.
        2. For each group, ask the birth-window strategy to evaluate only
           those positions → collect sell orders.
        3. Determine remaining positions after sells.
        4. Ask the current strategy for entries, seeing only remaining
           positions (so it correctly counts open slots).
        """
        # Step 1: group positions by birth window
        groups: dict[int, dict[int, Position]] = defaultdict(dict)
        for netuid, pos in positions.items():
            birth_idx = self._birth_idx.get(netuid, cur_idx)
            groups[birth_idx][netuid] = pos

        # Step 2: evaluate exits per birth-window
        sell_orders: list[Order] = []
        for birth_idx, group_positions in groups.items():
            strat = self._strategies[birth_idx]
            orders = strat.on_tick(
                tick, group_positions, capital, portfolio_value,
            )
            sell_orders.extend(o for o in orders if o.side == "sell")

        # Step 3: remaining positions after sells
        selling = {o.netuid for o in sell_orders}
        remaining = {k: v for k, v in positions.items() if k not in selling}

        # Step 4: entries with current window's strategy
        cur_strat = self._strategies[cur_idx]
        entry_orders = cur_strat.on_tick(
            tick, remaining, capital, portfolio_value,
        )
        buy_orders = [o for o in entry_orders if o.side == "buy"]

        return sell_orders + buy_orders
