"""
BacktestEngine — generic event-driven backtester for Bittensor strategies.

Data-agnostic: the caller provides a sequence of TickData and a Strategy.
The engine handles AMM execution, position tracking, equity recording,
and optionally writes to the same TradeLog / PortfolioLog used by live bots.

Known-bug prevention (see backtest_bugs_mar20.md):
  - Entry price uses cost-weighted average, not overwrite (Bug #1)
  - Entry timestamp preserved on accumulation (Bug #2)
  - Watch timeout is strategy-level, engine uses tick-based delay (Bug #3)
  - Order.limit_price prevents TP overshoot on delayed execution (Bug #4)
  - Delayed orders correctly update capital (Bug #5-like)
  - Positions passed to strategy as a shallow copy (defensive)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from bt_trading_tools.amm import amm_buy, amm_sell, slippage_pct, spot_price
from bt_trading_tools.backtest.stats import BacktestStats, compute_stats
from bt_trading_tools.backtest.types import Order, Position, Strategy, SubnetTick, TickData


@dataclass
class BacktestResults:
    """Output of a backtest run."""
    stats: BacktestStats
    trades: list[dict]
    equity_curve: list[dict]
    positions_at_end: dict[int, Position]


# Fees matching emission-bot's model (can be overridden)
DEFAULT_SWAP_FEE_RATE = 0.0005   # 0.05% per swap
DEFAULT_GAS_FEE_TAO = 0.00001   # gas per transaction


class BacktestEngine:
    """Generic backtesting engine.

    Args:
        capital: Starting capital in TAO.
        bot_name: Name for tracking logs (if using TradeLog/PortfolioLog).
        swap_fee_rate: Proportional swap fee (default 0.05%).
        gas_fee_tao: Fixed gas fee per transaction (default 0.00001 TAO).
        max_pool_pct: Max fraction of pool depth per trade (default 5%).
        execution_delay: Number of ticks to delay execution (0 = instant).
        trade_log: Optional TradeLog instance for recording trades.
        portfolio_log: Optional PortfolioLog instance for equity curve.
        ticks_per_year: For Sharpe annualization. 365=daily, 8760=hourly,
            2628000=12-second blocks.

    Usage::

        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, strategy)
        print(results.stats)
    """

    def __init__(
        self,
        capital: float = 100.0,
        bot_name: str = "backtest",
        swap_fee_rate: float = DEFAULT_SWAP_FEE_RATE,
        gas_fee_tao: float = DEFAULT_GAS_FEE_TAO,
        max_pool_pct: float = 0.05,
        execution_delay: int = 0,
        trade_log: Any = None,
        portfolio_log: Any = None,
        ticks_per_year: float = 365.0,
    ):
        self.starting_capital = capital
        self.bot_name = bot_name
        self.swap_fee_rate = swap_fee_rate
        self.gas_fee_tao = gas_fee_tao
        self.max_pool_pct = max_pool_pct
        self.execution_delay = execution_delay
        self.trade_log = trade_log
        self.portfolio_log = portfolio_log
        self.ticks_per_year = ticks_per_year

    def run(
        self,
        ticks: list[TickData],
        strategy: Strategy,
    ) -> BacktestResults:
        """Run the backtest.

        Args:
            ticks: Chronologically ordered market data. Each tick has
                subnets dict with price, pool depth, and signals.
            strategy: Implements Strategy protocol (on_tick method).

        Returns:
            BacktestResults with stats, trades, equity curve.
        """
        capital = self.starting_capital
        positions: dict[int, Position] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        # Pending delayed orders: (execute_at_tick_idx, order, decision_tick_snapshot)
        # The snapshot is the SubnetTick at decision time — used as fallback
        # price when the execution tick has no data for this subnet (because
        # nobody traded between decision and execution, so price didn't change).
        # This avoids the need to forward-fill sparse data to block level.
        pending_orders: list[tuple[int, Order, SubnetTick | None]] = []

        for tick_idx, tick in enumerate(ticks):
            # ── Execute delayed orders ───────────────────────────
            if self.execution_delay > 0:
                ready = [
                    (idx, order, snap) for idx, order, snap in pending_orders
                    if idx <= tick_idx
                ]
                pending_orders = [
                    (idx, order, snap) for idx, order, snap in pending_orders
                    if idx > tick_idx
                ]
                for _, order, decision_snap in ready:
                    # Build execution tick: use current tick data if available,
                    # otherwise fall back to the price at decision time.
                    # Between transactions, AMM pool state doesn't change —
                    # using the decision-time price IS ground truth, not
                    # approximation. This avoids inflating the dataset 15x
                    # by forward-filling to block level.
                    exec_tick = tick
                    if order.netuid not in tick.subnets and decision_snap is not None:
                        exec_tick = TickData(
                            timestamp=tick.timestamp,
                            subnets={**tick.subnets, order.netuid: decision_snap},
                            global_signals=tick.global_signals,
                        )
                    result = self._execute_order(
                        order, exec_tick, capital, positions, trades,
                    )
                    if result is not None:
                        capital = result

            # ── Ask strategy for orders ──────────────────────────
            pv = self._portfolio_value(capital, positions, tick)
            # FIX (defensive): pass a shallow copy of positions so strategy
            # can't corrupt engine state by mutating the dict.
            pos_snapshot = {k: copy.copy(v) for k, v in positions.items()}
            orders = strategy.on_tick(tick, pos_snapshot, capital, pv)

            for order in orders:
                if self.execution_delay > 0:
                    # Snapshot the subnet state at decision time
                    snap = tick.subnets.get(order.netuid)
                    pending_orders.append(
                        (tick_idx + self.execution_delay, order, snap)
                    )
                else:
                    result = self._execute_order(
                        order, tick, capital, positions, trades,
                    )
                    if result is not None:
                        capital = result

            # ── Record equity ────────────────────────────────────
            pv = self._portfolio_value(capital, positions, tick)
            eq_point = {
                "timestamp": tick.timestamp,
                "capital": round(capital, 6),
                "positions_value": round(pv - capital, 6),
                "total_equity": round(pv, 6),
                "n_positions": len(positions),
            }
            equity_curve.append(eq_point)

            if self.portfolio_log:
                self.portfolio_log.record(
                    total_value=pv, cash=capital,
                    staked_value=pv - capital,
                    n_positions=len(positions),
                    timestamp=tick.timestamp,
                )

        # ── Force-close remaining positions at last tick ─────────
        if ticks:
            last_tick = ticks[-1]
            for netuid in list(positions.keys()):
                pos = positions[netuid]
                st = last_tick.subnets.get(netuid)
                if st and st.tao_pool > 0 and st.alpha_pool > 0:
                    tao_out, _, _ = amm_sell(
                        pos.alpha_qty, st.tao_pool, st.alpha_pool,
                    )
                    fee = tao_out * self.swap_fee_rate + self.gas_fee_tao
                    tao_received = tao_out - fee
                else:
                    price = st.price if st else pos.entry_price
                    tao_received = pos.alpha_qty * price
                    fee = 0.0

                pnl = tao_received - pos.tao_cost
                capital += tao_received

                trade = {
                    "netuid": netuid,
                    "entry_time": pos.entry_time,
                    "exit_time": last_tick.timestamp,
                    "entry_price": pos.entry_price,
                    "exit_price": tao_received / pos.alpha_qty if pos.alpha_qty > 0 else 0,
                    "alpha_qty": pos.alpha_qty,
                    "tao_cost": pos.tao_cost,
                    "tao_received": tao_received,
                    "pnl": pnl,
                    "fees": pos.entry_fees + fee,
                    "hold_seconds": last_tick.timestamp - pos.entry_time,
                    "reason": "end_of_data",
                }
                trades.append(trade)
                self._record_trade(trade, "sell")
                del positions[netuid]

        stats = compute_stats(
            trades, equity_curve, self.starting_capital, self.ticks_per_year,
        )

        return BacktestResults(
            stats=stats,
            trades=trades,
            equity_curve=equity_curve,
            positions_at_end=positions,
        )

    # ── Order execution ──────────────────────────────────────────

    def _execute_order(
        self,
        order: Order,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
        trades: list[dict],
    ) -> float | None:
        """Execute an order. Returns updated capital if changed, else None."""
        st = tick.subnets.get(order.netuid)
        if st is None:
            return None

        if order.side == "buy":
            return self._execute_buy(order, st, tick, capital, positions)
        elif order.side == "sell":
            return self._execute_sell(order, st, tick, capital, positions, trades)
        return None

    def _execute_buy(
        self,
        order: Order,
        st: SubnetTick,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
    ) -> float | None:
        """Execute a buy order. Returns updated capital."""
        spend = order.tao_amount
        if spend <= 0 or spend > capital:
            return None

        # FIX (Bug #4 analog for buys): if the order has a limit_price and
        # the current price exceeds it, skip — the market moved away.
        if order.limit_price is not None and st.price > order.limit_price:
            return None

        # Cap at max pool percentage
        max_by_pool = st.tao_pool * self.max_pool_pct
        if max_by_pool > 0:
            spend = min(spend, max_by_pool)

        if st.tao_pool <= 0 or st.alpha_pool <= 0:
            return None

        # AMM execution with fees
        fee = spend * self.swap_fee_rate + self.gas_fee_tao
        spend_net = spend - fee
        if spend_net <= 0:
            return None

        alpha_received, _, _ = amm_buy(spend_net, st.tao_pool, st.alpha_pool)
        if alpha_received <= 0:
            return None

        eff_price = spend / alpha_received  # total cost basis

        # Accumulate into existing position or create new.
        # FIX (Bug #1): cost-weighted average, not overwrite.
        # FIX (Bug #2): entry_time preserved from first buy.
        if order.netuid in positions:
            pos = positions[order.netuid]
            total_alpha = pos.alpha_qty + alpha_received
            total_cost = pos.tao_cost + spend
            pos.entry_price = total_cost / total_alpha
            pos.alpha_qty = total_alpha
            pos.tao_cost = total_cost
            pos.entry_fees += fee
        else:
            positions[order.netuid] = Position(
                netuid=order.netuid,
                entry_price=eff_price,
                alpha_qty=alpha_received,
                tao_cost=spend,
                entry_time=tick.timestamp,
                entry_fees=fee,
                metadata=order.signal_data or {},
            )

        trade = {
            "netuid": order.netuid,
            "entry_time": tick.timestamp,
            "exit_time": None,
            "entry_price": eff_price,
            "exit_price": None,
            "alpha_qty": alpha_received,
            "tao_cost": spend,
            "tao_received": 0,
            "pnl": 0,
            "fees": fee,
            "hold_seconds": 0,
            "reason": order.reason,
        }
        self._record_trade(trade, "buy")

        return capital - spend

    def _execute_sell(
        self,
        order: Order,
        st: SubnetTick,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
        trades: list[dict],
    ) -> float | None:
        """Execute a sell order. Returns updated capital."""
        pos = positions.get(order.netuid)
        if pos is None:
            return None

        alpha_to_sell = order.alpha_amount if order.alpha_amount > 0 else pos.alpha_qty

        # Cap at max pool percentage (in alpha equivalent)
        if st.alpha_pool > 0:
            max_alpha_by_pool = st.alpha_pool * self.max_pool_pct
            alpha_to_sell = min(alpha_to_sell, max_alpha_by_pool)

        if alpha_to_sell <= 0:
            return None

        # ── FIX (Bug #4): TP overshoot prevention via limit_price ──
        # If the order has a limit_price (e.g. TP target), and the current
        # market price exceeds it, execute at the limit price instead.
        # This prevents delayed sells from profiting off extreme prices
        # that exceed the TP target — the bug that caused +164,605% phantom
        # returns in backtest_realistic.py.
        use_pool_tao = st.tao_pool
        use_pool_alpha = st.alpha_pool
        if order.limit_price is not None and st.price > order.limit_price:
            # Reconstruct pool state at the limit price.
            # k = tao * alpha, price = tao/alpha → tao = sqrt(k * price)
            k = st.tao_pool * st.alpha_pool
            if k > 0 and order.limit_price > 0:
                use_pool_tao = (k * order.limit_price) ** 0.5
                use_pool_alpha = (k / order.limit_price) ** 0.5

        # AMM execution
        if use_pool_tao > 0 and use_pool_alpha > 0:
            tao_out, _, _ = amm_sell(alpha_to_sell, use_pool_tao, use_pool_alpha)
            fee = tao_out * self.swap_fee_rate + self.gas_fee_tao
            tao_received = max(0, tao_out - fee)
        else:
            tao_received = alpha_to_sell * st.price
            fee = 0.0

        # Proportional cost basis
        fraction = alpha_to_sell / pos.alpha_qty if pos.alpha_qty > 0 else 1.0
        fraction = min(fraction, 1.0)
        cost_of_sold = pos.tao_cost * fraction
        pnl = tao_received - cost_of_sold

        trade = {
            "netuid": order.netuid,
            "entry_time": pos.entry_time,
            "exit_time": tick.timestamp,
            "entry_price": pos.entry_price,
            "exit_price": tao_received / alpha_to_sell if alpha_to_sell > 0 else 0,
            "alpha_qty": alpha_to_sell,
            "tao_cost": cost_of_sold,
            "tao_received": tao_received,
            "pnl": pnl,
            "fees": pos.entry_fees * fraction + fee,
            "hold_seconds": tick.timestamp - pos.entry_time,
            "reason": order.reason,
        }
        trades.append(trade)
        self._record_trade(trade, "sell")

        # Update or remove position
        remaining = pos.alpha_qty - alpha_to_sell
        if remaining < 1e-9:
            del positions[order.netuid]
        else:
            pos.alpha_qty = remaining
            pos.tao_cost -= cost_of_sold
            pos.entry_fees -= pos.entry_fees * fraction

        return capital + tao_received

    # ── Helpers ──────────────────────────────────────────────────

    def _portfolio_value(
        self,
        capital: float,
        positions: dict[int, Position],
        tick: TickData,
    ) -> float:
        """Mark-to-market portfolio value."""
        value = capital
        for netuid, pos in positions.items():
            st = tick.subnets.get(netuid)
            if st:
                # Cap MTM at 10x cost to handle price anomalies
                mtm = min(pos.alpha_qty * st.price, pos.tao_cost * 10)
                value += mtm
            else:
                value += pos.tao_cost  # fallback: assume flat
        return value

    def _record_trade(self, trade: dict, trade_type: str) -> None:
        """Record to TradeLog if configured."""
        if self.trade_log and trade.get("exit_time") is not None:
            try:
                self.trade_log.record_trade(
                    trade_type=trade_type,
                    netuid=trade["netuid"],
                    tao_amount=trade.get("tao_received", trade.get("tao_cost", 0)),
                    alpha_amount=trade["alpha_qty"],
                    price=trade.get("exit_price", trade.get("entry_price", 0)),
                    slippage=0,
                    hotkey="backtest",
                    reason=trade.get("reason", ""),
                    signal_data=None,
                    timestamp=trade["exit_time"],
                )
            except Exception:
                pass  # Don't let logging failures crash the backtest
