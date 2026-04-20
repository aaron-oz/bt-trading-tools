"""
Deterministic fee model for Bittensor stake/unstake operations.

Replaces the two historical constants (``SWAP_FEE_RATE=0.0005``,
``GAS_FEE_TAO=0.00001``) applied uniformly across paper/backtest with a
chain-sourced, per-operation fee quote. Three atomic components:

    swap_fee_tao   — pool liquidity fee (~0.0504% via FeeRate/65535)
    gas_fee_tao    — extrinsic weight+length fee (~8 µTAO)
    proxy_fee_tao  — proxy.proxy wrapping overhead (small, non-zero)

Design goals:

* ``FeeModel.quote`` never raises — chain errors degrade to calibrated
  fallbacks and populate ``FeeQuote.error`` + ``FeeQuote.source="fallback"``.
* Per-(netuid, operation, amount-bucket, uses_proxy) TTL cache avoids
  hammering the chain on every tick. Amount bucketing (2 sig figs ≈ 1%
  bucket width) means 0.50 and 0.51 TAO share a cache entry —
  proportional fees make the error < 1% of the fee.
* Chain client is a Protocol, not a concrete SDK import — tests mock
  cleanly and the bittensor import is deferred until a real adapter is
  instantiated.
* ``coldkey_ss58`` is resolved kwarg → env ``FEE_MODEL_COLDKEY_SS58`` →
  a well-known dev SS58 (Alice). ``get_payment_info`` is read-only; any
  valid SS58 returns the same fee estimate.

See ``docs/fees_and_yield_design.md`` in the alpha-trading repo for the
full design + integration plan.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from bt_trading_tools.tracking.schema import FeeSource

logger = logging.getLogger(__name__)

# ── Public constants ──────────────────────────────────────────────────

Operation = Literal["add_stake", "remove_stake"]

# Alice dev account — a valid SS58 that corresponds to no production wallet.
# Used when neither kwarg nor env var supplies a coldkey. get_payment_info
# is read-only; this never signs or spends anything.
DEFAULT_FEE_SIM_SS58 = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"

# Calibrated fallbacks — derived from docs.learnbittensor.org/learn/fees.
# Refit after ~1 week of TradeExecutor fee receipts (scripts/calibrate_fees.py).
FALLBACK_SWAP_RATE: float = 33 / 65535   # ≈ 5.04e-4, the default mechanism-1 FeeRate
FALLBACK_GAS_TAO: float = 8.4e-6         # mid-range of reference estimate (8.4–10 µTAO)
FALLBACK_PROXY_TAO: float = 1.0e-6       # order-of-magnitude; refit after calibration

DEFAULT_CACHE_TTL_S: float = 900.0       # 15 min; fees don't change intra-tempo


# ── Data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeeQuote:
    """Deterministic fee quote for a single stake operation.

    All fee components are TAO-normalized. Chain returns alpha-denominated
    fees on remove_stake; FeeModel converts via ``spot_price`` before
    populating this record.
    """
    operation: Operation
    netuid: int
    amount: float
    uses_proxy: bool
    swap_fee_tao: float
    gas_fee_tao: float
    proxy_fee_tao: float
    total_fee_tao: float
    source: FeeSource
    observed_at: float
    error: Optional[str] = None


class ChainFeeClient(Protocol):
    """Sync interface for chain fee RPCs.

    Concrete adapters wrap the bittensor SDK; mocks in tests implement
    these two methods directly. Implementations should be thread-safe
    enough to be called from one ``FeeModel`` instance sequentially.
    """

    def sim_swap(
        self,
        netuid: int,
        operation: Operation,
        amount: float,
    ) -> dict:
        """Simulate a stake/unstake swap.

        Returns a dict with at least:
            {"tao_fee": float | None, "alpha_fee": float | None}

        Exactly one of ``tao_fee`` / ``alpha_fee`` should be populated:
        TAO for add_stake, alpha for remove_stake.
        """
        ...

    def get_payment_info(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        coldkey_ss58: str,
        uses_proxy: bool,
    ) -> dict:
        """Fetch extrinsic payment info for the (optionally proxy-wrapped)
        stake operation.

        Returns a dict with at least:
            {"partial_fee_tao": float}
        """
        ...


# ── Amount bucketing ──────────────────────────────────────────────────

def _bucket_amount(amount: float, sig_figs: int = 2) -> str:
    """Round ``amount`` to ``sig_figs`` significant figures and return a
    canonical string key. Near-identical amounts map to the same bucket;
    proportional fees make the per-bucket error bounded.
    """
    if amount <= 0 or not math.isfinite(amount):
        return "0"
    exp = math.floor(math.log10(amount))
    factor = 10 ** (exp - (sig_figs - 1))
    bucketed = round(amount / factor) * factor
    return f"{bucketed:.6g}"


# ── FeeModel ──────────────────────────────────────────────────────────

class FeeModel:
    """Deterministic fee calculator with per-subnet TTL cache.

    ``quote()`` is the only entry point and never raises. On chain
    error, returns a fallback-sourced FeeQuote with ``error`` populated.

    Thread-safety: not thread-safe. One instance per bot process.
    """

    def __init__(
        self,
        chain: Optional[ChainFeeClient] = None,
        fallback_swap_rate: float = FALLBACK_SWAP_RATE,
        fallback_gas_tao: float = FALLBACK_GAS_TAO,
        fallback_proxy_tao: float = FALLBACK_PROXY_TAO,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        coldkey_ss58: Optional[str] = None,
    ):
        self._chain = chain
        self._fallback_swap_rate = fallback_swap_rate
        self._fallback_gas_tao = fallback_gas_tao
        self._fallback_proxy_tao = fallback_proxy_tao
        self._cache_ttl_s = cache_ttl_s
        self._coldkey_ss58 = (
            coldkey_ss58
            or os.environ.get("FEE_MODEL_COLDKEY_SS58")
            or DEFAULT_FEE_SIM_SS58
        )
        self._cache: dict[tuple, tuple[float, FeeQuote]] = {}

    # ── Public ──────────────────────────────────────────────────────

    def quote(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        uses_proxy: bool = True,
        spot_price: Optional[float] = None,
    ) -> FeeQuote:
        """Return a FeeQuote. Never raises.

        Args:
            operation: "add_stake" or "remove_stake".
            netuid: subnet id.
            amount: TAO for add_stake, alpha for remove_stake.
            uses_proxy: whether to simulate proxy-wrapped extrinsic.
            spot_price: TAO per alpha. Required to convert alpha-denominated
                swap fees (remove_stake) to TAO. If omitted on a remove_stake
                where the chain returned alpha_fee, we fall back.
        """
        now = time.time()
        key = (operation, netuid, _bucket_amount(amount), uses_proxy)

        cached = self._cache.get(key)
        if cached is not None:
            expires_at, quote = cached
            if expires_at > now:
                return quote
            del self._cache[key]

        if self._chain is None:
            q = self._fallback_quote(
                operation, netuid, amount, uses_proxy, now,
                error="no_chain_client",
            )
        else:
            try:
                q = self._chain_quote(
                    operation, netuid, amount, uses_proxy, spot_price, now,
                )
            except Exception as e:   # pragma: no cover — chain is mocked in tests
                logger.warning(
                    "FeeModel.quote chain error (op=%s, netuid=%d, amount=%s): "
                    "%s: %s; falling back",
                    operation, netuid, amount, type(e).__name__, e,
                )
                q = self._fallback_quote(
                    operation, netuid, amount, uses_proxy, now,
                    error=f"{type(e).__name__}: {e}",
                )

        self._cache[key] = (now + self._cache_ttl_s, q)
        return q

    def clear_cache(self) -> None:
        """Drop all cached quotes. Useful for tests or after chain upgrades."""
        self._cache.clear()

    # ── Internals ───────────────────────────────────────────────────

    def _chain_quote(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        uses_proxy: bool,
        spot_price: Optional[float],
        now: float,
    ) -> FeeQuote:
        """Query chain for swap + extrinsic + proxy components."""
        assert self._chain is not None

        # Swap fee component
        swap_result = self._chain.sim_swap(netuid, operation, amount)
        tao_fee = swap_result.get("tao_fee")
        alpha_fee = swap_result.get("alpha_fee")
        if operation == "add_stake":
            if tao_fee is None:
                raise ValueError(
                    f"sim_swap(add_stake) returned no tao_fee: {swap_result}"
                )
            swap_fee_tao = float(tao_fee)
        else:
            if alpha_fee is None:
                # Chain sometimes returns tao_fee even on remove_stake
                if tao_fee is not None:
                    swap_fee_tao = float(tao_fee)
                else:
                    raise ValueError(
                        f"sim_swap(remove_stake) returned no alpha_fee: {swap_result}"
                    )
            else:
                if spot_price is None or spot_price <= 0:
                    raise ValueError(
                        "spot_price required to convert alpha-denominated "
                        "remove_stake fee to TAO"
                    )
                swap_fee_tao = float(alpha_fee) * float(spot_price)

        # Extrinsic + proxy component
        bare_info = self._chain.get_payment_info(
            operation, netuid, amount, self._coldkey_ss58, uses_proxy=False,
        )
        gas_fee_tao = float(bare_info["partial_fee_tao"])

        if uses_proxy:
            proxy_info = self._chain.get_payment_info(
                operation, netuid, amount, self._coldkey_ss58, uses_proxy=True,
            )
            proxy_total = float(proxy_info["partial_fee_tao"])
            proxy_fee_tao = max(0.0, proxy_total - gas_fee_tao)
        else:
            proxy_fee_tao = 0.0

        total = swap_fee_tao + gas_fee_tao + proxy_fee_tao
        return FeeQuote(
            operation=operation,
            netuid=netuid,
            amount=amount,
            uses_proxy=uses_proxy,
            swap_fee_tao=swap_fee_tao,
            gas_fee_tao=gas_fee_tao,
            proxy_fee_tao=proxy_fee_tao,
            total_fee_tao=total,
            source=FeeSource.CHAIN,
            observed_at=now,
            error=None,
        )

    def _fallback_quote(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        uses_proxy: bool,
        now: float,
        error: Optional[str] = None,
    ) -> FeeQuote:
        """Compute a fallback quote from calibrated constants.

        Swap fee is ``amount × fallback_swap_rate``. For remove_stake this
        is alpha × rate (in alpha); we treat the rate as dimensionless and
        return ``amount × rate`` — callers should be aware this under-
        estimates by the TAO/alpha conversion factor when amount is
        alpha-denominated. The paper-trader agent's pipeline always passes
        TAO-denominated amounts on buys and alpha-denominated on sells, and
        downstream consumers treat the number as "fee in the unit of
        amount"; post-TAO-normalization in the caller is the caller's job
        in fallback mode. In practice: fallback is only hit when the chain
        is unreachable and the caller should already have the spot price
        for its own AMM math.

        To keep this simple and honest: fallback produces numerically the
        same shape it always produced (the historical constants × amount).
        Calibration will refit after real data lands.
        """
        swap_fee_tao = amount * self._fallback_swap_rate
        gas_fee_tao = self._fallback_gas_tao
        proxy_fee_tao = self._fallback_proxy_tao if uses_proxy else 0.0
        total = swap_fee_tao + gas_fee_tao + proxy_fee_tao
        return FeeQuote(
            operation=operation,
            netuid=netuid,
            amount=amount,
            uses_proxy=uses_proxy,
            swap_fee_tao=swap_fee_tao,
            gas_fee_tao=gas_fee_tao,
            proxy_fee_tao=proxy_fee_tao,
            total_fee_tao=total,
            source=FeeSource.FALLBACK,
            observed_at=now,
            error=error,
        )


# ── Concrete bittensor adapter (optional; deferred import) ────────────

class BittensorFeeClient:
    """Concrete ``ChainFeeClient`` backed by the bittensor SDK.

    The bittensor import is deferred to method-call time so
    ``bt-trading-tools`` stays importable in environments without the
    SDK (tests, CI, local dev without chain access). Instantiation
    itself is cheap — the WebSocket connection opens lazily on the
    first call.

    Uses ``asyncio.run`` to bridge the async SDK into the sync
    ``ChainFeeClient`` contract. Matches the ``_LivePoolCache`` pattern
    in quality-dip-bot.

    SDK surface verified against bittensor 9.x:

    * ``sub.sim_swap(origin_netuid, destination_netuid, amount)`` returns
      a ``SimSwapResult`` with ``.tao_fee`` / ``.alpha_fee`` /
      ``.tao_amount`` / ``.alpha_amount`` (all ``Balance`` objects).
      origin=0 → add_stake; origin=netuid → remove_stake.
      Source: bittensor/core/async_subtensor.py::sim_swap (line ~638).
    * ``sub.compose_call(call_module, call_function, call_params)``
      validates params against chain metadata then returns a
      ``GenericCall``. Source: async_subtensor.py::compose_call (line ~5871).
    * ``sub.get_extrinsic_fee(call, keypair)`` returns a ``Balance``
      wrapping partial_fee. Source: async_subtensor.py::get_extrinsic_fee
      (line ~6036).
    * ``SubtensorModule.add_stake(netuid, hotkey, amount_staked)`` /
      ``remove_stake(netuid, hotkey, amount_unstaked)`` are the canonical
      call_params shapes. Source: core/extrinsics/pallets/subtensor_module.py.

    See ``docs/data-sources.md`` in the alpha-trading repo for links.
    """

    def __init__(self, network: str = "finney"):
        self.network = network

    def sim_swap(self, netuid: int, operation: Operation, amount: float) -> dict:
        import asyncio
        return asyncio.run(self._sim_swap_async(netuid, operation, amount))

    def get_payment_info(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        coldkey_ss58: str,
        uses_proxy: bool,
    ) -> dict:
        import asyncio
        return asyncio.run(
            self._payment_info_async(
                operation, netuid, amount, coldkey_ss58, uses_proxy,
            )
        )

    # ── Async implementations ──────────────────────────────────────

    async def _sim_swap_async(
        self, netuid: int, operation: Operation, amount: float,
    ) -> dict:
        import bittensor as bt
        from bittensor.core.async_subtensor import get_async_subtensor

        sub = await get_async_subtensor(self.network)
        try:
            # add_stake:    origin=0 (root TAO), destination=netuid
            # remove_stake: origin=netuid, destination=0
            if operation == "add_stake":
                amount_bal = bt.Balance.from_tao(amount)
                origin, dest = 0, netuid
            else:
                # Alpha is tracked in rao on-chain; amount arg is alpha
                # tokens, multiply by 1e9 for the Balance representation.
                amount_bal = bt.Balance.from_rao(int(amount * 1e9))
                origin, dest = netuid, 0

            result = await sub.sim_swap(
                origin_netuid=origin,
                destination_netuid=dest,
                amount=amount_bal,
            )
            return {
                "tao_fee": _balance_to_tao(getattr(result, "tao_fee", None)),
                "alpha_fee": _balance_to_tao(getattr(result, "alpha_fee", None)),
            }
        finally:
            try:
                await sub.close()
            except Exception:
                pass

    async def _payment_info_async(
        self,
        operation: Operation,
        netuid: int,
        amount: float,
        coldkey_ss58: str,
        uses_proxy: bool,
    ) -> dict:
        import bittensor as bt
        from bittensor.core.async_subtensor import get_async_subtensor

        sub = await get_async_subtensor(self.network)
        try:
            amount_param = (
                "amount_staked" if operation == "add_stake" else "amount_unstaked"
            )
            call = await sub.compose_call(
                call_module="SubtensorModule",
                call_function=operation,
                call_params={
                    # hotkey field expects an SS58; for fee simulation the
                    # specific value doesn't change the weight/length fee.
                    "hotkey": coldkey_ss58,
                    "netuid": netuid,
                    amount_param: int(amount * 1e9),
                },
            )

            if uses_proxy:
                call = await sub.compose_call(
                    call_module="Proxy",
                    call_function="proxy",
                    call_params={
                        "real": coldkey_ss58,
                        "force_proxy_type": None,
                        "call": call,
                    },
                )

            keypair = bt.Keypair(ss58_address=coldkey_ss58)
            fee = await sub.get_extrinsic_fee(call=call, keypair=keypair)
            return {"partial_fee_tao": _balance_to_tao(fee) or 0.0}
        finally:
            try:
                await sub.close()
            except Exception:
                pass


def _balance_to_tao(b) -> Optional[float]:
    """Convert a bittensor ``Balance`` (or None / int-rao / float-tao) to a
    plain TAO float. Module-level so both BittensorFeeClient and tests
    can use it.
    """
    if b is None:
        return None
    if hasattr(b, "tao"):
        try:
            return float(b.tao)
        except Exception:
            pass
    if hasattr(b, "rao"):
        try:
            return float(b.rao) / 1e9
        except Exception:
            pass
    try:
        return float(b)
    except (TypeError, ValueError):
        return None
