"""
SubtensorClient — wallet-agnostic async Bittensor chain connection.

Consolidates retry/reconnect logic from autobot, doubledip, and bagbot
into a single robust implementation. Any bot passes its own wallet when
it needs to trade; the client itself just talks to the chain.

Usage::

    client = SubtensorClient(network="finney")
    await client.connect()

    stats = await client.get_all_subnets()
    balance = await client.get_balance(coldkey_ss58)
    stakes = await client.get_stakes(coldkey_ss58, hotkeys)

    await client.wait_for_block()
    await client.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubnetInfo:
    """Parsed subnet data from all_subnets()."""
    netuid: int
    name: str
    price: float          # TAO per alpha
    tao_in: float         # TAO liquidity in pool
    alpha_in: float       # Alpha liquidity in pool


class SubtensorClient:
    """Async Bittensor chain connection with auto-reconnect.

    This class owns the subtensor connection and provides all read operations.
    It is wallet-agnostic — no wallet is stored or required for reads.
    Trade operations (add_stake, unstake) are handled by TradeExecutor,
    which takes a wallet per call.

    Args:
        network: Bittensor network name (default "finney").
        max_connect_retries: Max retries for initial connection (default 20).
        max_reconnect_retries: Max retries for reconnection (default 5).
        reconnect_base_wait: Base wait in seconds for exponential backoff (default 10).
        reconnect_max_wait: Max wait in seconds between retries (default 60).
    """

    def __init__(
        self,
        network: str = "finney",
        max_connect_retries: int = 20,
        max_reconnect_retries: int = 5,
        reconnect_base_wait: float = 10.0,
        reconnect_max_wait: float = 60.0,
    ):
        self.network = network
        self.max_connect_retries = max_connect_retries
        self.max_reconnect_retries = max_reconnect_retries
        self.reconnect_base_wait = reconnect_base_wait
        self.reconnect_max_wait = reconnect_max_wait
        self._sub = None

    @property
    def sub(self):
        """Raw subtensor object for advanced usage."""
        if self._sub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._sub

    # ── Connection management ────────────────────────────────────

    async def connect(self) -> None:
        """Establish initial connection with retry."""
        from bittensor.core.async_subtensor import get_async_subtensor

        for attempt in range(1, self.max_connect_retries + 1):
            try:
                self._sub = await get_async_subtensor(self.network)
                logger.info(f"Connected to {self.network}")
                return
            except (
                asyncio.TimeoutError,
                ConnectionResetError,
                AttributeError,
                OSError,
            ) as e:
                wait = min(attempt * 2, 30)
                logger.warning(
                    f"Connect attempt {attempt}/{self.max_connect_retries} failed: "
                    f"{type(e).__name__}: {e}, retrying in {wait}s"
                )
                if attempt >= self.max_connect_retries:
                    raise
                await asyncio.sleep(wait)

    async def reconnect(self) -> None:
        """Reconnect after a connection failure. Exponential backoff."""
        from bittensor.core.async_subtensor import get_async_subtensor

        for attempt in range(1, self.max_reconnect_retries + 1):
            # Close existing connection
            try:
                if self._sub is not None:
                    await self._sub.close()
            except Exception:
                pass
            self._sub = None

            try:
                self._sub = await get_async_subtensor(self.network)
                logger.info(f"Reconnected to {self.network} (attempt {attempt})")
                return
            except Exception as e:
                wait = min(attempt * self.reconnect_base_wait, self.reconnect_max_wait)
                logger.warning(
                    f"Reconnect attempt {attempt}/{self.max_reconnect_retries} failed: "
                    f"{type(e).__name__}: {e}, retrying in {wait}s"
                )
                await asyncio.sleep(wait)

        logger.error(
            f"All {self.max_reconnect_retries} reconnect attempts failed, "
            f"will retry next tick"
        )

    async def close(self) -> None:
        """Close the connection."""
        if self._sub is not None:
            try:
                await self._sub.close()
            except Exception:
                pass
            self._sub = None

    # ── Chain reads ──────────────────────────────────────────────

    async def get_all_subnets(self, timeout: float = 30.0) -> dict[int, SubnetInfo]:
        """Fetch all subnet data. Returns {netuid: SubnetInfo}.

        Retries internally on transient errors (websocket, attribute).
        """
        for attempt in range(6):
            try:
                raw = await asyncio.wait_for(self._sub.all_subnets(), timeout=timeout)
                stats = {}
                for subnet in raw:
                    netuid = subnet.netuid
                    price = float(subnet.price)
                    if price <= 0:
                        continue
                    name = str(subnet.subnet_name) if hasattr(subnet, "subnet_name") else ""
                    stats[netuid] = SubnetInfo(
                        netuid=netuid,
                        name=name,
                        price=price,
                        tao_in=float(subnet.tao_in.tao) if hasattr(subnet.tao_in, 'tao') else float(subnet.tao_in),
                        alpha_in=float(subnet.alpha_in.tao) if hasattr(subnet.alpha_in, 'tao') else float(subnet.alpha_in),
                    )
                return stats
            except asyncio.TimeoutError:
                logger.error(f"get_all_subnets timeout after {timeout}s")
                raise
            except (AttributeError, OSError, ConnectionError) as e:
                logger.error(f"get_all_subnets attempt {attempt+1} failed: {e}")
                await asyncio.sleep(3)
                await self.reconnect()
            except Exception as e:
                # Catch websocket errors and other transient failures
                err_type = type(e).__name__
                if "websocket" in err_type.lower() or "invalid" in err_type.lower():
                    logger.error(f"get_all_subnets attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(3)
                    await self.reconnect()
                else:
                    raise

        raise RuntimeError("get_all_subnets failed after 6 attempts")

    async def get_stake(
        self, coldkey_ss58: str, hotkey_ss58: str, timeout: float = 20.0,
        retries: int = 10,
    ) -> dict[int, Any]:
        """Get stake for a specific coldkey+hotkey pair.

        Returns {netuid: StakeObject} where StakeObject has .stake.rao.
        """
        for attempt in range(retries):
            try:
                return await asyncio.wait_for(
                    self._sub.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=hotkey_ss58,
                        coldkey_ss58=coldkey_ss58,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"get_stake timeout (attempt {attempt+1}/{retries})"
                )
                await asyncio.sleep(min(10, attempt * 3))
        raise TimeoutError(f"get_stake failed after {retries} attempts")

    async def get_stakes(
        self, coldkey_ss58: str, hotkeys: list[str],
        timeout: float = 20.0,
    ) -> dict[str, dict[int, Any]]:
        """Get stake for multiple hotkeys in parallel.

        Returns {hotkey: {netuid: StakeObject}}.
        """
        results = await asyncio.gather(
            *(self.get_stake(coldkey_ss58, hk, timeout=timeout) for hk in hotkeys),
            return_exceptions=True,
        )
        stakes = {}
        for hotkey, result in zip(hotkeys, results):
            if isinstance(result, Exception):
                logger.warning(f"get_stake({hotkey[:8]}...) failed: {result}")
                stakes[hotkey] = {}
            else:
                stakes[hotkey] = result
        return stakes

    async def get_balance(self, coldkey_ss58: str, timeout: float = 20.0) -> float:
        """Get wallet balance in TAO."""
        balance = await asyncio.wait_for(
            self._sub.get_balance(address=coldkey_ss58),
            timeout=timeout,
        )
        return float(balance)

    async def discover_validators(
        self, coldkey_ss58: str, timeout: float = 30.0,
    ) -> list[str] | None:
        """Find all validators where this coldkey has stake.

        Returns list of hotkey addresses, or None if discovery fails.
        """
        try:
            stake_info_list = await asyncio.wait_for(
                self._sub.get_stake_info_for_coldkey(coldkey_ss58=coldkey_ss58),
                timeout=timeout,
            )

            if stake_info_list is None:
                logger.warning("get_stake_info_for_coldkey returned None")
                return None

            validators = set()
            if isinstance(stake_info_list, list):
                for info in stake_info_list:
                    hotkey = getattr(info, "hotkey_ss58", None) or getattr(info, "hotkey", None)
                    if hotkey:
                        validators.add(hotkey)
            else:
                logger.warning(
                    f"Unexpected type from get_stake_info_for_coldkey: "
                    f"{type(stake_info_list)}"
                )
                return None

            if validators:
                logger.info(f"Discovered {len(validators)} validators with stake")
                return list(validators)
            else:
                logger.warning("No validators found with stake")
                return None

        except asyncio.TimeoutError:
            logger.warning("Timeout discovering validators")
            return None
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error parsing stake info: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not discover validators: {e}")
            return None

    async def wait_for_block(self, timeout: float = 30.0) -> None:
        """Wait for next block. Reconnects on timeout."""
        try:
            await asyncio.wait_for(self._sub.wait_for_block(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("wait_for_block timed out, reconnecting...")
            await self.reconnect()
        except (OSError, KeyError):
            # Transient network error — just sleep one block interval
            await asyncio.sleep(12)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def total_stake(
        stake_info: dict[str, dict[int, Any]], netuid: int,
    ) -> float:
        """Sum stake across all hotkeys for a subnet.

        Args:
            stake_info: {hotkey: {netuid: StakeObject}} from get_stakes().
            netuid: Subnet to sum.

        Returns alpha tokens staked (float).
        """
        total = 0.0
        for hotkey_stakes in stake_info.values():
            obj = hotkey_stakes.get(netuid)
            if obj is not None:
                total += float(obj.stake)
        return total

    @staticmethod
    def find_hotkey_with_stake(
        stake_info: dict[str, dict[int, Any]],
        netuid: int,
        preferred_hotkey: str | None = None,
    ) -> str | None:
        """Find a hotkey that has stake on a given subnet.

        Checks preferred_hotkey first, then falls back to any hotkey with stake.
        """
        if preferred_hotkey and preferred_hotkey in stake_info:
            obj = stake_info[preferred_hotkey].get(netuid)
            if obj is not None and float(obj.stake) > 0:
                return preferred_hotkey

        for hotkey, stakes in stake_info.items():
            obj = stakes.get(netuid)
            if obj is not None and float(obj.stake) > 0:
                return hotkey

        return None
