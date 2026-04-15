"""
TradeExecutor — non-blocking async trade execution.

Consolidates trade execution from doubledip (background tasks with
reservation tracking) and autobot/bagbot (direct execution). Wallet
is passed per trade call, not stored on the executor.

Usage::

    executor = TradeExecutor(client, tracking=trade_log, events=event_log)

    # Non-blocking (returns immediately, trade executes in background):
    executor.submit_buy(wallet, hotkey, netuid=107, tao_amount=0.5,
                        pool_tao=150.0, reason="inventory_build",
                        signal_data={"pct_change": -1.2})

    # Each tick, harvest completed trades:
    completed = executor.process_pending()

    # Blocking (waits for result):
    result = await executor.buy(wallet, hotkey, netuid=107, ...)
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bt_trading_tools.amm import slippage_pct

if TYPE_CHECKING:
    from bt_trading_tools.network.client import SubtensorClient
    from bt_trading_tools.network.wallet import ProxyWalletManager
    from bt_trading_tools.tracking.trade_log import TradeLog
    from bt_trading_tools.tracking.event_log import EventLog

logger = logging.getLogger(__name__)

# Reserve enough free TAO to cover transaction fees.  Each extrinsic costs
# ~0.0001–0.001 TAO; 0.01 TAO provides a comfortable margin even when
# multiple trades fire in the same tick.
FEE_RESERVE_TAO: float = 0.01


@dataclass
class TradeResult:
    """Outcome of a trade attempt."""
    success: bool
    netuid: int
    trade_type: str               # "buy" or "sell"
    tao_amount: float             # TAO spent (buy) or received (sell)
    alpha_amount: float           # Alpha received (buy) or sold (sell)
    price: float                  # Effective price
    slippage: float               # Actual slippage percentage
    reason: str                   # Why this trade was made
    error: str | None = None      # Error message if failed
    signal_data: dict | None = None


class TradeExecutor:
    """Async trade executor with optional background (non-blocking) mode.

    Args:
        client: SubtensorClient for chain access.
        tracking: Optional TradeLog for automatic trade recording.
        events: Optional EventLog for automatic event logging.
        buy_timeout: Timeout for buy operations in seconds.
        sell_timeout: Timeout for sell operations in seconds.
        default_sell_rate_tolerance: Rate tolerance for sells (fraction).
            Sells use a permissive tolerance because we always want to exit.
        default_slippage_buffer: Extra buffer added to computed slippage
            for rate_tolerance, in percentage points.
    """

    def __init__(
        self,
        client: SubtensorClient,
        proxy_manager: ProxyWalletManager | None = None,
        tracking: TradeLog | None = None,
        events: EventLog | None = None,
        buy_timeout: float = 45.0,
        sell_timeout: float = 60.0,
        default_sell_rate_tolerance: float = 0.50,
        default_slippage_buffer: float = 2.0,
        fee_reserve: float = FEE_RESERVE_TAO,
    ):
        self.client = client
        self.proxy_manager = proxy_manager
        self.tracking = tracking
        self.events = events
        self.buy_timeout = buy_timeout
        self.sell_timeout = sell_timeout
        self.default_sell_rate_tolerance = default_sell_rate_tolerance
        self.default_slippage_buffer = default_slippage_buffer
        self.fee_reserve = fee_reserve

        # Background task tracking
        self._pending_tasks: list[asyncio.Task] = []
        self._pending_trades: dict[int, str] = {}  # netuid → "buying"|"selling"
        self._reserved_balance: float = 0.0

    # ── Balance reservation (for non-blocking buys) ──────────────

    @property
    def reserved_balance(self) -> float:
        """TAO reserved for in-flight buy orders."""
        return self._reserved_balance

    def available_balance(self, total_balance: float) -> float:
        """Balance minus TAO reserved for pending buys and fee reserve."""
        return total_balance - self._reserved_balance - self.fee_reserve

    def is_pending(self, netuid: int) -> bool:
        """Check if a trade is in-flight for this subnet."""
        return netuid in self._pending_trades

    def pending_direction(self, netuid: int) -> str | None:
        """Returns 'buying' or 'selling' if pending, else None."""
        return self._pending_trades.get(netuid)

    # ── Process completed background trades ──────────────────────

    def process_pending(self) -> list[TradeResult]:
        """Harvest completed background trade tasks.

        Call once per tick. Returns list of completed TradeResults.
        """
        completed = []
        still_pending = []
        for task in self._pending_tasks:
            if task.done():
                try:
                    result = task.result()
                    if isinstance(result, TradeResult):
                        completed.append(result)
                except Exception as e:
                    logger.error(f"Background trade task failed: {e}")
            else:
                still_pending.append(task)
        self._pending_tasks = still_pending
        return completed

    # ── Non-blocking submit ──────────────────────────────────────

    def submit_buy(
        self,
        wallet: Any,
        hotkey: str,
        netuid: int,
        tao_amount: float,
        pool_tao: float,
        total_balance: float = float("inf"),
        reason: str = "",
        signal_data: dict | None = None,
        rate_tolerance: float | None = None,
    ) -> bool:
        """Submit a buy to execute in the background.

        Reserves balance immediately. Updates pending status.
        Call process_pending() each tick to harvest results.

        Args:
            total_balance: Current wallet balance. If the available balance
                (after reservations and fee reserve) is insufficient, the
                buy is skipped and ``False`` is returned.

        Returns:
            True if the buy was submitted, False if skipped due to
            insufficient balance.
        """
        if self.available_balance(total_balance) < tao_amount:
            logger.warning(
                f"BUY SN{netuid} SKIPPED: need {tao_amount:.4f} TAO but only "
                f"{self.available_balance(total_balance):.4f} available "
                f"(balance={total_balance:.4f}, reserved={self._reserved_balance:.4f}, "
                f"fee_reserve={self.fee_reserve:.4f})"
            )
            return False
        self._reserved_balance += tao_amount
        self._pending_trades[netuid] = "buying"
        task = asyncio.create_task(
            self._buy_background(
                wallet, hotkey, netuid, tao_amount, pool_tao,
                reason, signal_data, rate_tolerance,
            )
        )
        self._pending_tasks.append(task)
        return True

    def submit_sell(
        self,
        wallet: Any,
        hotkey: str,
        netuid: int,
        alpha_amount: float,
        price: float,
        pool_tao: float,
        reason: str = "",
        signal_data: dict | None = None,
        rate_tolerance: float | None = None,
    ) -> None:
        """Submit a sell to execute in the background."""
        self._pending_trades[netuid] = "selling"
        task = asyncio.create_task(
            self._sell_background(
                wallet, hotkey, netuid, alpha_amount, price, pool_tao,
                reason, signal_data, rate_tolerance,
            )
        )
        self._pending_tasks.append(task)

    # ── Blocking execution ───────────────────────────────────────

    async def buy(
        self,
        wallet: Any,
        hotkey: str,
        netuid: int,
        tao_amount: float,
        pool_tao: float,
        reason: str = "",
        signal_data: dict | None = None,
        rate_tolerance: float | None = None,
    ) -> TradeResult:
        """Execute a buy (add_stake) and wait for result."""
        import bittensor as bt

        price_before = pool_tao  # We'll compute effective price from result
        slip = slippage_pct(tao_amount, pool_tao)

        if rate_tolerance is None:
            rate_tolerance = slip + self.default_slippage_buffer / 100.0

        bt_amount = bt.utils.balance.tao(tao_amount)

        logger.info(
            f"BUY SN{netuid}: {tao_amount:.4f} TAO "
            f"(impact {slip:.2%}, tol {rate_tolerance:.2%}, pool {pool_tao:.0f} TAO)"
        )

        try:
            if self.proxy_manager:
                from bittensor.core.extrinsics.pallets import SubtensorModule
                from bittensor.core.chain_data.proxy import ProxyType

                call = await SubtensorModule(self.client.sub).add_stake(
                    netuid=netuid,
                    hotkey=hotkey,
                    amount_staked=bt_amount.rao,
                )
                result = await asyncio.wait_for(
                    self.client.sub.proxy(
                        wallet=self.proxy_manager.proxy_wallet,
                        real_account_ss58=self.proxy_manager.real_account_ss58,
                        force_proxy_type=ProxyType.Staking,
                        call=call,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                    ),
                    timeout=self.buy_timeout,
                )
            else:
                result = await asyncio.wait_for(
                    self.client.sub.add_stake(
                        wallet=wallet,
                        hotkey_ss58=hotkey,
                        netuid=netuid,
                        amount=bt_amount,
                        rate_tolerance=rate_tolerance,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                        safe_staking=True,
                        allow_partial_stake=False,
                    ),
                    timeout=self.buy_timeout,
                )
            success = result is True or (hasattr(result, "success") and result.success)

            if success:
                # Estimate alpha received from AMM math
                from bt_trading_tools.amm import amm_buy
                stats = await self.client.get_all_subnets()
                sinfo = stats.get(netuid)
                if sinfo:
                    alpha_received, _, _ = amm_buy(tao_amount, sinfo.tao_in, sinfo.alpha_in)
                    eff_price = tao_amount / alpha_received if alpha_received > 0 else sinfo.price
                else:
                    alpha_received = tao_amount / slip if slip < 1 else 0
                    eff_price = 0

                tr = TradeResult(
                    success=True, netuid=netuid, trade_type="buy",
                    tao_amount=tao_amount, alpha_amount=alpha_received,
                    price=eff_price, slippage=slip * 100,
                    reason=reason, signal_data=signal_data,
                )
                logger.info(
                    f"BUY SN{netuid} SUCCESS: ~{alpha_received:.1f} alpha "
                    f"for {tao_amount:.4f} TAO"
                )
            else:
                tr = TradeResult(
                    success=False, netuid=netuid, trade_type="buy",
                    tao_amount=tao_amount, alpha_amount=0, price=0,
                    slippage=slip * 100, reason=reason,
                    error=str(result), signal_data=signal_data,
                )
                logger.warning(f"BUY SN{netuid} FAILED: {result}")

        except asyncio.TimeoutError:
            tr = TradeResult(
                success=False, netuid=netuid, trade_type="buy",
                tao_amount=tao_amount, alpha_amount=0, price=0,
                slippage=slip * 100, reason=reason,
                error=f"Timeout after {self.buy_timeout}s",
                signal_data=signal_data,
            )
            logger.error(f"BUY SN{netuid} TIMEOUT after {self.buy_timeout}s")
        except Exception as e:
            tr = TradeResult(
                success=False, netuid=netuid, trade_type="buy",
                tao_amount=tao_amount, alpha_amount=0, price=0,
                slippage=slip * 100, reason=reason,
                error=f"{type(e).__name__}: {e}",
                signal_data=signal_data,
            )
            logger.error(f"BUY SN{netuid} ERROR: {e}")
            logger.debug(traceback.format_exc())

        self._record(tr)
        return tr

    async def sell(
        self,
        wallet: Any,
        hotkey: str,
        netuid: int,
        alpha_amount: float,
        price: float,
        pool_tao: float,
        reason: str = "",
        signal_data: dict | None = None,
        rate_tolerance: float | None = None,
    ) -> TradeResult:
        """Execute a sell (unstake) and wait for result."""
        import bittensor as bt

        approx_tao = alpha_amount * price
        if rate_tolerance is None:
            rate_tolerance = self.default_sell_rate_tolerance

        # Subtract 1 rao safety margin to handle rounding (emissions shift stake)
        bt_alpha = bt.utils.balance.tao(alpha_amount, netuid)
        if bt_alpha.rao > 0:
            bt_alpha = bt.Balance.from_rao(bt_alpha.rao - 1, netuid)

        logger.info(
            f"SELL SN{netuid}: ~{alpha_amount:.1f} alpha (~{approx_tao:.4f} TAO) "
            f"reason={reason}"
        )

        try:
            # Balance queries use the real account (Ledger) when proxy is active
            balance_ss58 = (
                self.proxy_manager.real_account_ss58
                if self.proxy_manager
                else wallet.coldkey.ss58_address
            )

            balance_before = await self.client.get_balance(
                balance_ss58, timeout=10.0,
            )

            if self.proxy_manager:
                from bittensor.core.extrinsics.pallets import SubtensorModule
                from bittensor.core.chain_data.proxy import ProxyType

                call = await SubtensorModule(self.client.sub).remove_stake(
                    netuid=netuid,
                    hotkey=hotkey,
                    amount_unstaked=bt_alpha.rao,
                )
                result = await asyncio.wait_for(
                    self.client.sub.proxy(
                        wallet=self.proxy_manager.proxy_wallet,
                        real_account_ss58=self.proxy_manager.real_account_ss58,
                        force_proxy_type=ProxyType.Staking,
                        call=call,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                    ),
                    timeout=self.sell_timeout,
                )
            else:
                result = await asyncio.wait_for(
                    self.client.sub.unstake(
                        wallet=wallet,
                        hotkey_ss58=hotkey,
                        netuid=netuid,
                        amount=bt_alpha,
                        rate_tolerance=rate_tolerance,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                        safe_unstaking=True,
                        allow_partial_stake=False,
                    ),
                    timeout=self.sell_timeout,
                )
            success = result is True or (hasattr(result, "success") and result.success)

            if success:
                # Measure actual TAO received via balance delta
                try:
                    balance_after = await self.client.get_balance(
                        balance_ss58, timeout=10.0,
                    )
                    actual_tao = balance_after - balance_before
                    if actual_tao <= 0:
                        actual_tao = approx_tao  # Fallback if concurrent trades
                except Exception:
                    actual_tao = approx_tao

                actual_slip = (1 - actual_tao / approx_tao) * 100 if approx_tao > 0 else 0

                tr = TradeResult(
                    success=True, netuid=netuid, trade_type="sell",
                    tao_amount=actual_tao, alpha_amount=alpha_amount,
                    price=actual_tao / alpha_amount if alpha_amount > 0 else price,
                    slippage=actual_slip, reason=reason, signal_data=signal_data,
                )
                logger.info(
                    f"SELL SN{netuid} SUCCESS: {actual_tao:.4f} TAO, "
                    f"reason={reason}"
                )
            else:
                tr = TradeResult(
                    success=False, netuid=netuid, trade_type="sell",
                    tao_amount=0, alpha_amount=alpha_amount, price=price,
                    slippage=0, reason=reason, error=str(result),
                    signal_data=signal_data,
                )
                logger.warning(f"SELL SN{netuid} FAILED: {result}")

        except asyncio.TimeoutError:
            tr = TradeResult(
                success=False, netuid=netuid, trade_type="sell",
                tao_amount=0, alpha_amount=alpha_amount, price=price,
                slippage=0, reason=reason,
                error=f"Timeout after {self.sell_timeout}s",
                signal_data=signal_data,
            )
            logger.error(f"SELL SN{netuid} TIMEOUT after {self.sell_timeout}s")
        except Exception as e:
            tr = TradeResult(
                success=False, netuid=netuid, trade_type="sell",
                tao_amount=0, alpha_amount=alpha_amount, price=price,
                slippage=0, reason=reason,
                error=f"{type(e).__name__}: {e}",
                signal_data=signal_data,
            )
            logger.error(f"SELL SN{netuid} ERROR: {e}")
            logger.debug(traceback.format_exc())

        self._record(tr)
        return tr

    # ── Background task wrappers ─────────────────────────────────

    async def _buy_background(
        self, wallet, hotkey, netuid, tao_amount, pool_tao,
        reason, signal_data, rate_tolerance,
    ) -> TradeResult:
        """Background buy task. Releases reservation on completion."""
        try:
            return await self.buy(
                wallet, hotkey, netuid, tao_amount, pool_tao,
                reason, signal_data, rate_tolerance,
            )
        finally:
            self._reserved_balance = max(0, self._reserved_balance - tao_amount)
            self._pending_trades.pop(netuid, None)

    async def _sell_background(
        self, wallet, hotkey, netuid, alpha_amount, price, pool_tao,
        reason, signal_data, rate_tolerance,
    ) -> TradeResult:
        """Background sell task. Clears pending status on completion."""
        try:
            return await self.sell(
                wallet, hotkey, netuid, alpha_amount, price, pool_tao,
                reason, signal_data, rate_tolerance,
            )
        finally:
            self._pending_trades.pop(netuid, None)

    # ── Recording ────────────────────────────────────────────────

    def _record(self, tr: TradeResult) -> None:
        """Record trade to tracking systems if configured."""
        if tr.success and self.tracking:
            try:
                self.tracking.record_trade(
                    trade_type=tr.trade_type,
                    netuid=tr.netuid,
                    tao_amount=tr.tao_amount,
                    alpha_amount=tr.alpha_amount,
                    price=tr.price,
                    slippage=tr.slippage,
                    hotkey="",  # filled by caller's state
                    reason=tr.reason,
                    signal_data=tr.signal_data,
                )
            except Exception as e:
                logger.error(f"Failed to record trade: {e}")

        if self.events:
            if tr.success:
                self.events.trade(
                    f"{tr.trade_type}_success",
                    netuid=tr.netuid,
                    tao=tr.tao_amount,
                    detail={
                        "alpha": round(tr.alpha_amount, 2),
                        "slippage": round(tr.slippage, 2),
                        "reason": tr.reason,
                    },
                )
            else:
                self.events.error(
                    f"{tr.trade_type}_failed",
                    detail={
                        "netuid": tr.netuid,
                        "error": tr.error,
                        "reason": tr.reason,
                    },
                )
