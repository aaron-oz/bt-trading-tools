"""
WalletManager — thin wrapper for Bittensor wallet setup.

Each bot creates its own wallet. This helper standardizes the
load/unlock/env-var pattern used across all bots.

Also exposes two on-chain wallet utilities that don't fit on
``SubtensorClient`` directly because they compose multiple client calls:

* ``valuate_wallet`` — total TAO value of a wallet (free balance + alpha
  positions valued at current AMM spot). Read-only.
* ``unstake_all`` — drain every open stake position from a coldkey. Write
  operation; defaults to ``dry_run=True`` so it returns the plan without
  signing. Pass ``dry_run=False`` to actually unstake.

Usage::

    wm = WalletManager(name="doubledip", password_env="DD_WALLET_PW")
    wallet = wm.setup()
    # wallet.coldkey.ss58_address is now available
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bt_trading_tools.network.client import SubtensorClient

logger = logging.getLogger(__name__)


class WalletManager:
    """Bittensor wallet setup helper.

    Args:
        name: Wallet name (as created by ``btcli w create``).
        password_env: Environment variable name holding the wallet password.
            If the env var is not set, falls back to ``password`` arg.
        password: Fallback password if env var is not set. Prefer env var
            for security.
        create_if_missing: Whether to create the wallet if it doesn't exist.
    """

    def __init__(
        self,
        name: str,
        password_env: str | None = None,
        password: str | None = None,
        create_if_missing: bool = False,
    ):
        self.name = name
        self.password_env = password_env
        self.password = password
        self.create_if_missing = create_if_missing
        self._wallet = None

    def setup(self) -> Any:
        """Load and unlock the wallet. Returns the bittensor Wallet object."""
        import bittensor as bt

        wallet = bt.Wallet(name=self.name)

        if self.create_if_missing:
            wallet.create_if_non_existent()

        # Resolve password: env var takes precedence
        pw = None
        if self.password_env:
            pw = os.environ.get(self.password_env)
        if pw is None:
            pw = self.password

        if pw:
            wallet.coldkey_file.save_password_to_env(pw)

        wallet.unlock_coldkey()
        self._wallet = wallet

        logger.info(
            f"Wallet '{self.name}' unlocked "
            f"(coldkey={wallet.coldkey.ss58_address[:8]}...)"
        )
        return wallet

    @property
    def wallet(self) -> Any:
        """The loaded wallet, or raises if setup() hasn't been called."""
        if self._wallet is None:
            raise RuntimeError("Wallet not set up. Call setup() first.")
        return self._wallet

    @property
    def coldkey_ss58(self) -> str:
        """Coldkey SS58 address."""
        return self.wallet.coldkey.ss58_address


class ProxyWalletManager:
    """Wallet manager for proxy-based trading.

    The proxy wallet signs all transactions, but trades execute on behalf
    of ``real_account_ss58`` (the Ledger account). The proxy must have been
    granted Staking proxy rights on-chain.

    Args:
        proxy_name: Wallet name of the proxy (e.g., "autobot-proxy").
        real_account_ss58: The Ledger account SS58 address.
        password_env: Env var holding the proxy wallet password.
        password: Fallback password if env var is not set.
    """

    def __init__(
        self,
        proxy_name: str,
        real_account_ss58: str,
        password_env: str | None = None,
        password: str | None = None,
    ):
        self._wm = WalletManager(
            name=proxy_name, password_env=password_env, password=password,
        )
        self._real_account_ss58 = real_account_ss58

    def setup(self) -> Any:
        """Load and unlock the proxy wallet. Returns the bittensor Wallet object."""
        wallet = self._wm.setup()
        logger.info(
            f"Proxy wallet '{self._wm.name}' configured for "
            f"real account {self._real_account_ss58[:8]}..."
        )
        return wallet

    @property
    def proxy_wallet(self) -> Any:
        """The proxy wallet object (for signing)."""
        return self._wm.wallet

    @property
    def real_account_ss58(self) -> str:
        """The Ledger account address (trades execute on this account)."""
        return self._real_account_ss58

    @property
    def proxy_ss58(self) -> str:
        """The proxy wallet's coldkey address."""
        return self._wm.coldkey_ss58

    @property
    def coldkey_ss58(self) -> str:
        """The account to query for balance/stake (the Ledger account).

        This lets code that uses ``manager.coldkey_ss58`` for on-chain
        queries work transparently with both WalletManager and
        ProxyWalletManager.
        """
        return self._real_account_ss58


# ── On-chain valuation ───────────────────────────────────────────────


@dataclass(frozen=True)
class WalletPosition:
    """Single (hotkey, netuid) stake position with TAO valuation."""
    netuid: int
    hotkey_ss58: str
    alpha: float                # alpha tokens currently staked (units: alpha)
    spot_price_tao: float       # current AMM spot price tao_in / alpha_in
    tao_value: float            # alpha * spot_price_tao

    @classmethod
    def from_alpha(cls, netuid: int, hotkey_ss58: str, alpha: float,
                   spot_price_tao: float) -> "WalletPosition":
        return cls(
            netuid=netuid, hotkey_ss58=hotkey_ss58, alpha=alpha,
            spot_price_tao=spot_price_tao, tao_value=alpha * spot_price_tao,
        )


@dataclass
class WalletValuation:
    """Total on-chain TAO value of a wallet, with per-position breakdown.

    ``stake_coldkey_ss58`` is the address that holds stake (the Ledger
    principal under proxy patterns). ``free_balance_ss58s`` is the list of
    addresses whose free TAO is summed into ``free_tao``; defaults to just
    the stake coldkey but can include a separate proxy wallet that holds tx
    fees (autobot pattern: principal Ledger holds stakes, proxy holds free
    TAO for gas).
    """
    stake_coldkey_ss58: str
    free_balance_ss58s: list[str]
    free_tao: float                       # sum across free_balance_ss58s
    free_tao_per_address: dict[str, float]
    staked_tao: float                     # sum of position.tao_value
    total_tao: float                      # free_tao + staked_tao
    positions: list[WalletPosition] = field(default_factory=list)
    n_subnets_with_stake: int = 0

    def __str__(self) -> str:
        lines = [
            f"WalletValuation: {self.total_tao:.4f} TAO total "
            f"(free {self.free_tao:.4f} + staked {self.staked_tao:.4f})",
            f"  stake coldkey:   {self.stake_coldkey_ss58}",
            f"  balance sources: {self.free_balance_ss58s}",
            f"  per-address free TAO:",
        ]
        for ss, t in self.free_tao_per_address.items():
            lines.append(f"    {ss[:14]}…  {t:.4f}")
        lines.append(f"  positions: {len(self.positions)} across "
                     f"{self.n_subnets_with_stake} subnets")
        for p in sorted(self.positions, key=lambda x: -x.tao_value):
            lines.append(
                f"    netuid={p.netuid:>3}  α={p.alpha:>10.3f}  "
                f"price={p.spot_price_tao:.6f}  "
                f"value={p.tao_value:>8.4f} TAO  "
                f"hk={p.hotkey_ss58[:14]}…"
            )
        return "\n".join(lines)


async def valuate_wallet(
    client: "SubtensorClient",
    stake_coldkey_ss58: str,
    free_balance_ss58s: "list[str] | str | None" = None,
    timeout_per_call_s: float = 20.0,
) -> WalletValuation:
    """Return the total on-chain TAO value of a wallet.

    Composes three client calls:

    1. ``get_balance(ss58)`` for each free-balance address (parallel).
    2. ``_sub.get_stake_info_for_coldkey(stake_coldkey_ss58)`` for the
       full set of (hotkey, netuid, alpha) positions on the principal.
    3. ``get_all_subnets()`` once for the spot-price table to value each
       position at current AMM mid.

    Args:
        client: A connected ``SubtensorClient``.
        stake_coldkey_ss58: The coldkey whose stake we sum. For proxy
            patterns this is the Ledger principal, NOT the proxy.
        free_balance_ss58s: Address(es) whose free TAO contributes to the
            total. Defaults to ``[stake_coldkey_ss58]``. For proxy patterns
            pass ``[principal, proxy]`` so the proxy's tx-fee balance is
            counted too.
        timeout_per_call_s: Per-RPC timeout. Total wall time is bounded
            by ``2 * timeout_per_call_s + 30`` (balance + stake + subnets).

    Returns:
        ``WalletValuation`` — total + per-position breakdown.

    Raises:
        Whatever the underlying client calls raise on permanent failure.
        Per-position parse errors are logged and skipped (fail-soft so a
        valuation is still produced for the rest).
    """
    # Normalize free_balance_ss58s
    if free_balance_ss58s is None:
        free_addrs = [stake_coldkey_ss58]
    elif isinstance(free_balance_ss58s, str):
        free_addrs = [free_balance_ss58s]
    else:
        free_addrs = list(free_balance_ss58s)

    # 1. Free balances in parallel
    bal_tasks = [
        client.get_balance(coldkey_ss58=ss, timeout=timeout_per_call_s)
        for ss in free_addrs
    ]
    bal_results = await asyncio.gather(*bal_tasks, return_exceptions=True)
    free_per_addr: dict[str, float] = {}
    for ss, r in zip(free_addrs, bal_results):
        if isinstance(r, Exception):
            logger.warning("get_balance(%s...) failed: %s", ss[:12], r)
            free_per_addr[ss] = 0.0
        else:
            free_per_addr[ss] = float(r)
    free_tao = sum(free_per_addr.values())

    # 2. Stake info for principal — one RPC returns every (hotkey, netuid).
    #    Use the underlying SDK directly because SubtensorClient.get_stakes
    #    requires a hotkey list up-front; we want full discovery.
    stake_infos = await asyncio.wait_for(
        client._sub.get_stake_info_for_coldkey(coldkey_ss58=stake_coldkey_ss58),
        timeout=timeout_per_call_s,
    )

    # 3. Spot prices for all subnets — needed to value alpha at TAO.
    subnet_table = await client.get_all_subnets(timeout=timeout_per_call_s)

    positions: list[WalletPosition] = []
    netuids_with_stake: set[int] = set()
    for info in stake_infos or []:
        try:
            netuid = int(getattr(info, "netuid"))
            hk = getattr(info, "hotkey_ss58", None) or getattr(info, "hotkey", None)
            stake_obj = getattr(info, "stake", None)
            # SDK returns a Balance-like object; .rao is in the chain's
            # rao integer units (1 alpha = 1e9 rao for alpha tokens).
            if hasattr(stake_obj, "rao"):
                alpha = float(stake_obj.rao) / 1e9
            else:
                alpha = float(stake_obj) if stake_obj is not None else 0.0
            if alpha <= 0 or hk is None:
                continue
            sub_info = subnet_table.get(netuid)
            if sub_info is None:
                logger.warning(
                    "no subnet_info for netuid=%d; using price=0", netuid,
                )
                price = 0.0
            else:
                price = float(sub_info.price)
            positions.append(WalletPosition.from_alpha(
                netuid=netuid, hotkey_ss58=hk,
                alpha=alpha, spot_price_tao=price,
            ))
            netuids_with_stake.add(netuid)
        except Exception as e:
            logger.warning("skipping stake record: %s (%r)", e, info)

    staked_tao = sum(p.tao_value for p in positions)
    total_tao = free_tao + staked_tao

    return WalletValuation(
        stake_coldkey_ss58=stake_coldkey_ss58,
        free_balance_ss58s=free_addrs,
        free_tao=free_tao,
        free_tao_per_address=free_per_addr,
        staked_tao=staked_tao,
        total_tao=total_tao,
        positions=positions,
        n_subnets_with_stake=len(netuids_with_stake),
    )


# ── Drain (unstake all) ──────────────────────────────────────────────


@dataclass
class UnstakePlanItem:
    """One position scheduled for unstaking."""
    netuid: int
    hotkey_ss58: str
    alpha: float                 # amount to unstake (in alpha tokens)


@dataclass
class UnstakeResult:
    """Outcome of attempting to unstake one position."""
    netuid: int
    hotkey_ss58: str
    alpha_attempted: float
    success: bool
    error: "str | None" = None


@dataclass
class UnstakeAllResult:
    """Aggregated result of unstake_all()."""
    dry_run: bool
    plan: list[UnstakePlanItem]
    results: list[UnstakeResult] = field(default_factory=list)

    @property
    def n_planned(self) -> int:
        return len(self.plan)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


async def unstake_all(
    client: "SubtensorClient",
    coldkey_ss58: str,
    *,
    wallet: Any = None,
    use_proxy: bool = False,
    rate_tolerance: float = 0.005,
    timeout_per_unstake_s: float = 60.0,
    dry_run: bool = True,
    inter_unstake_delay_s: float = 1.0,
) -> UnstakeAllResult:
    """Drain every open stake position from a coldkey.

    Discovery: queries ``get_stake_info_for_coldkey(coldkey_ss58)`` for the
    full position list, then issues one unstake extrinsic per (hotkey,
    netuid) pair. Sequential by default (with ``inter_unstake_delay_s``
    between calls) so a flaky chain doesn't pile parallel timeouts.

    Default is ``dry_run=True`` — returns the plan without signing
    anything. Pass ``dry_run=False`` to actually unstake. **The caller is
    responsible for confirming this is intended** — once submitted,
    extrinsics are irreversible.

    Args:
        client: Connected ``SubtensorClient``.
        coldkey_ss58: Coldkey whose stakes will be drained. Under proxy
            patterns this is the Ledger principal; the proxy wallet does
            the signing (passed via ``wallet``).
        wallet: Bittensor Wallet for signing. Required when
            ``dry_run=False``. Ignored in dry-run.
        use_proxy: If True, wraps each unstake in
            ``sub.proxy(real_account_ss58=coldkey_ss58, ...)`` (Substrate
            Proxy pallet). If False, the wallet's coldkey IS the principal
            and unstakes happen directly.
        rate_tolerance: Slippage tolerance for safe unstake.
        timeout_per_unstake_s: Per-extrinsic timeout.
        dry_run: When True (default), returns the plan without signing.
        inter_unstake_delay_s: Sleep between extrinsics. Avoids overwhelming
            the chain RPC and gives space for the previous extrinsic to
            settle into a block.

    Returns:
        ``UnstakeAllResult`` with the plan and (in non-dry-run mode) one
        ``UnstakeResult`` per position.
    """
    # Discover positions
    stake_infos = await asyncio.wait_for(
        client._sub.get_stake_info_for_coldkey(coldkey_ss58=coldkey_ss58),
        timeout=timeout_per_unstake_s,
    )
    plan: list[UnstakePlanItem] = []
    for info in stake_infos or []:
        try:
            netuid = int(getattr(info, "netuid"))
            hk = getattr(info, "hotkey_ss58", None) or getattr(info, "hotkey", None)
            stake_obj = getattr(info, "stake", None)
            if hasattr(stake_obj, "rao"):
                alpha = float(stake_obj.rao) / 1e9
            else:
                alpha = float(stake_obj) if stake_obj is not None else 0.0
            if alpha <= 0 or hk is None:
                continue
            plan.append(UnstakePlanItem(
                netuid=netuid, hotkey_ss58=hk, alpha=alpha,
            ))
        except Exception as e:
            logger.warning("skipping stake record during plan build: %s (%r)", e, info)

    result = UnstakeAllResult(dry_run=dry_run, plan=plan)
    if dry_run:
        logger.info("unstake_all DRY RUN: %d positions identified, no signing",
                    len(plan))
        return result

    if wallet is None:
        raise ValueError("unstake_all(dry_run=False) requires wallet=...")

    # Execute
    import bittensor as bt
    for i, item in enumerate(plan):
        logger.info(
            "unstake %d/%d: netuid=%d hk=%s α=%.4f",
            i + 1, len(plan), item.netuid, item.hotkey_ss58[:14], item.alpha,
        )
        try:
            if use_proxy:
                from bittensor.core.extrinsics.pallets import SubtensorModule
                from bittensor.core.chain_data.proxy import ProxyType
                # Build the inner remove_stake call, then wrap in Proxy
                call = await SubtensorModule(client._sub).remove_stake(
                    netuid=item.netuid,
                    hotkey=item.hotkey_ss58,
                    amount_unstaked=int(item.alpha * 1e9),
                )
                await asyncio.wait_for(
                    client._sub.proxy(
                        wallet=wallet,
                        real_account_ss58=coldkey_ss58,
                        force_proxy_type=ProxyType.Staking,
                        call=call,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                    ),
                    timeout=timeout_per_unstake_s,
                )
            else:
                await asyncio.wait_for(
                    client._sub.unstake(
                        wallet=wallet,
                        hotkey_ss58=item.hotkey_ss58,
                        netuid=item.netuid,
                        amount=bt.Balance.from_tao(item.alpha),
                        rate_tolerance=rate_tolerance,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                        safe_staking=True,
                        allow_partial_stake=False,
                    ),
                    timeout=timeout_per_unstake_s,
                )
            result.results.append(UnstakeResult(
                netuid=item.netuid, hotkey_ss58=item.hotkey_ss58,
                alpha_attempted=item.alpha, success=True,
            ))
        except Exception as e:
            logger.warning("unstake netuid=%d failed: %s", item.netuid, e)
            result.results.append(UnstakeResult(
                netuid=item.netuid, hotkey_ss58=item.hotkey_ss58,
                alpha_attempted=item.alpha, success=False, error=str(e),
            ))
        if i < len(plan) - 1:
            await asyncio.sleep(inter_unstake_delay_s)

    logger.info(
        "unstake_all complete: %d succeeded, %d failed",
        result.n_succeeded, result.n_failed,
    )
    return result
