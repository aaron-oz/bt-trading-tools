"""
Alpha yield accretion model.

When a staker holds alpha tokens in a subnet, the position accrues
additional alpha over time via subnet emissions distributed to
delegators each epoch (tempo × 360 blocks ≈ 72 min). Paper bots and
backtest engines that freeze ``alpha_qty`` from buy to sell under-
report position value.

This module provides:

* ``AlphaYieldModel.accrued_yield(netuid, alpha_qty, entry_time, now)``
  — a **pure** function returning additional alpha accrued. Calling it
  N times with the same inputs returns the same value. No state
  mutation. Safe to call inside ``mark_to_market`` twice per tick.
* ``AlphaYieldModel.rate(netuid)`` — cached per-subnet daily yield rate.
* ``YieldRateProvider`` Protocol + ``CascadingYieldProvider`` that tries
  taostats → chain → empirical → fallback in order, caching the first
  success.

**Realization happens only at sell**, inside ``PaperBotBase.apply_actions``
(or equivalent). The model itself never writes to position state.

See ``docs/fees_and_yield_design.md`` in the alpha-trading repo.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from bt_trading_tools.tracking.schema import AlphaYieldSource

logger = logging.getLogger(__name__)


DEFAULT_YIELD_CACHE_TTL_S: float = 3600.0   # 1 hr; yield is set by emission config


# ── Data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class YieldRate:
    """Per-day alpha-per-alpha yield rate for a subnet.

    ``rate_per_day`` is a fractional growth rate: ``accrued = alpha_qty
    × rate × days``. A rate of 0.003 means an alpha position grows
    0.3%/day absent any trading.
    """
    netuid: int
    rate_per_day: float
    source: AlphaYieldSource
    observed_at: float
    error: Optional[str] = None


class YieldRateProvider(Protocol):
    """Sync provider of per-subnet yield rates.

    May raise on transient errors; the caller is expected to fall back
    or cache. Concrete implementations below: Taostats (primary),
    Chain (secondary), Empirical (tertiary).
    """
    source: AlphaYieldSource

    def fetch_rate(self, netuid: int) -> float:
        """Return alpha-per-alpha per-day yield rate for ``netuid``.

        Must return a non-negative float or raise an exception.
        """
        ...


# ── Cascading provider ────────────────────────────────────────────────

class CascadingYieldProvider:
    """Try providers in order; first success wins. Logs failures.

    Final tier is a built-in zero-rate fallback (``rate_per_day=0.0``,
    source=FALLBACK) so yield accrual becomes a no-op rather than
    crashing the paper bot.
    """
    source = AlphaYieldSource.FALLBACK   # only used if every tier fails

    def __init__(self, providers: list[YieldRateProvider]):
        self._providers = list(providers)

    def fetch_rate_with_source(self, netuid: int) -> tuple[float, AlphaYieldSource, Optional[str]]:
        """Return (rate, source, error). Never raises."""
        last_error: Optional[str] = None
        for p in self._providers:
            try:
                rate = float(p.fetch_rate(netuid))
                if rate < 0 or not _finite(rate):
                    raise ValueError(f"invalid rate {rate!r} from {p.source}")
                return rate, p.source, None
            except Exception as e:
                last_error = f"{p.source}:{type(e).__name__}:{e}"
                logger.debug(
                    "YieldProvider %s failed for netuid=%d: %s",
                    p.source, netuid, e,
                )
        return 0.0, AlphaYieldSource.FALLBACK, last_error


# ── AlphaYieldModel ───────────────────────────────────────────────────

class AlphaYieldModel:
    """Pure, idempotent yield accrual with cached per-subnet rate.

    Thread-safety: not thread-safe. One instance per bot process.
    """

    def __init__(
        self,
        provider: CascadingYieldProvider | YieldRateProvider,
        cache_ttl_s: float = DEFAULT_YIELD_CACHE_TTL_S,
    ):
        # Accept either a CascadingYieldProvider or a single YieldRateProvider.
        # Wrap bare providers so fetch_rate_with_source is always available.
        if isinstance(provider, CascadingYieldProvider):
            self._provider = provider
        else:
            self._provider = CascadingYieldProvider([provider])
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[int, tuple[float, YieldRate]] = {}

    # ── Public ──────────────────────────────────────────────────────

    def rate(self, netuid: int) -> YieldRate:
        """Return cached or freshly-fetched YieldRate for a subnet.

        Never raises. On provider failure, returns a YieldRate with
        ``rate_per_day=0.0``, ``source=FALLBACK``, and ``error`` populated.
        """
        now = time.time()
        cached = self._cache.get(netuid)
        if cached is not None:
            expires_at, yr = cached
            if expires_at > now:
                return yr
            del self._cache[netuid]

        rate_val, source, error = self._provider.fetch_rate_with_source(netuid)
        yr = YieldRate(
            netuid=netuid,
            rate_per_day=rate_val,
            source=source,
            observed_at=now,
            error=error,
        )
        self._cache[netuid] = (now + self._cache_ttl_s, yr)
        return yr

    def accrued_yield(
        self,
        netuid: int,
        alpha_qty: float,
        entry_time: float,
        now: float,
    ) -> float:
        """Return alpha accrued on a position since ``entry_time``.

        PURE FUNCTION — no state mutation, idempotent. Same
        ``(netuid, alpha_qty, entry_time, now)`` always returns the same
        value within a rate-cache window.

        ``entry_time`` and ``now`` are unix timestamps (seconds); the
        backtest engine passes simulated-time timestamps, paper bots
        pass ``time.time()``.

        Returns 0.0 if ``alpha_qty`` is non-positive, if ``now`` is at or
        before ``entry_time``, or if the rate is zero.
        """
        if alpha_qty <= 0 or not _finite(alpha_qty):
            return 0.0
        if now <= entry_time or not _finite(now) or not _finite(entry_time):
            return 0.0

        yr = self.rate(netuid)
        if yr.rate_per_day <= 0:
            return 0.0

        elapsed_days = (now - entry_time) / 86400.0
        return alpha_qty * yr.rate_per_day * elapsed_days

    def clear_cache(self) -> None:
        """Drop all cached rates."""
        self._cache.clear()


# ── Concrete providers ────────────────────────────────────────────────

class ZeroYieldProvider:
    """Always returns 0.0. Useful as a deterministic fallback or for
    tests that need to verify yield-is-disabled behavior."""
    source = AlphaYieldSource.FALLBACK

    def fetch_rate(self, netuid: int) -> float:
        return 0.0


class TaostatsYieldProvider:
    """Fetch per-subnet daily yield rate from taostats.

    Uses the ``/api/dtao/validator/yield/latest/v1`` endpoint which
    exposes per-validator APY directly. We take the **stake-weighted
    mean** of ``thirty_day_apy`` across all validators on the subnet
    and divide by 365 to convert to a simple-interest daily rate.

    Stake-weighted is the right aggregation: APY is per-validator and
    reflects that validator's performance + take; weighting by stake
    gives the rate a typical delegator on the subnet would observe.

    Endpoint reference: https://docs.taostats.io/reference
    Verified live 2026-04-20. Returns fresh per-block data (stake in rao,
    APY as a fraction e.g. 0.54 = 54%/yr).

    Conversion note: APY values are interpreted as simple-interest
    annual rate — ``rate_per_day = apy / 365``. That matches how the
    rest of the yield model accrues (linear in days). The continuous-
    compounding alternative (``log(1+apy)/365``) would give slightly
    lower daily rates but over-complicates the model for no gain in
    accuracy.

    Reads ``TAOSTATS_API_KEY`` from the environment if ``api_key`` not
    passed. Rate limits: 5 req/min free / 240 req/min Pro. With
    ``cache_ttl_s=3600`` on the model, one call per subnet per hour —
    well within either tier.
    """
    source = AlphaYieldSource.TAOSTATS

    ENDPOINT_PATH = "/api/dtao/validator/yield/latest/v1"
    APY_FIELD = "thirty_day_apy"
    DEFAULT_PAGE_LIMIT = 200   # subnets typically have <100 validators

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.taostats.io",
        timeout_s: float = 10.0,
    ):
        import os
        self._api_key = api_key or os.environ.get("TAOSTATS_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def fetch_rate(self, netuid: int) -> float:
        """Return alpha-per-alpha per-day yield rate.

        Raises on network / schema errors so the cascade falls through
        cleanly to ChainYieldProvider / EmpiricalYieldProvider / fallback.
        """
        import requests

        if not self._api_key:
            raise RuntimeError("TAOSTATS_API_KEY not set")

        headers = {"Authorization": self._api_key, "accept": "application/json"}

        resp = requests.get(
            f"{self._base_url}{self.ENDPOINT_PATH}",
            params={"netuid": netuid, "limit": self.DEFAULT_PAGE_LIMIT},
            headers=headers,
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        validators = payload.get("data") or []
        if not validators:
            raise ValueError(
                f"taostats returned no validators for netuid={netuid}"
            )

        # Stake-weighted mean of 30-day APY.
        total_stake = 0.0
        weighted_apy = 0.0
        for v in validators:
            stake = _as_float(v.get("stake"))
            apy = _as_float(v.get(self.APY_FIELD))
            if stake is None or apy is None or stake <= 0 or apy < 0:
                continue
            total_stake += stake
            weighted_apy += apy * stake

        if total_stake <= 0:
            raise ValueError(
                f"taostats returned no usable (stake, apy) pairs for "
                f"netuid={netuid}"
            )

        apy_mean = weighted_apy / total_stake
        return apy_mean / 365.0


class ChainYieldProvider:
    """Fetch per-subnet daily yield rate directly from the bittensor SDK.

    Fallback for when taostats is unreachable. Derives an approximate
    delegator yield rate from on-chain alpha emission and staked alpha.
    Bittensor import is deferred.

    SDK surface (verified against bittensor 9.x):

    * ``sub.subnet(netuid)`` → ``DynamicInfo`` with ``.alpha_out_emission``
      (Balance, **per-block**), ``.alpha_out`` (Balance, total alpha
      circulating / staked outside the pool), ``.alpha_in`` (Balance,
      alpha inside the AMM pool — NOT used here), ``.tempo`` (int).
      Source: bittensor/core/chain_data/dynamic_info.py.

    Derivation:

        blocks_per_day        = 7200                # ~12s per block
        daily_alpha_emission  = alpha_out_emission × blocks_per_day
        daily_to_delegators   = daily_alpha_emission × DELEGATOR_SHARE
        rate_per_day          = daily_to_delegators / alpha_out_staked

    ``DELEGATOR_SHARE = 0.82`` is a coarse project-wide assumption
    (Bittensor doesn't expose a per-subnet validator take on
    DynamicInfo and emission splits vary by consensus). For accurate
    per-subnet rates use TaostatsYieldProvider — it measures realized
    delegator APY directly from chain observations.
    """
    source = AlphaYieldSource.CHAIN

    # Coarse project-wide delegator share. Refine only if DynamicInfo
    # gains a per-subnet validator-take field in a future SDK release.
    DELEGATOR_SHARE: float = 0.82
    BLOCKS_PER_DAY: float = 7200.0

    def __init__(self, network: str = "finney"):
        self._network = network

    def fetch_rate(self, netuid: int) -> float:
        import asyncio

        async def _run():
            from bittensor.core.async_subtensor import get_async_subtensor
            sub = await get_async_subtensor(self._network)
            try:
                subnet = await sub.subnet(netuid=netuid)
                if subnet is None:
                    raise ValueError(f"sub.subnet({netuid}) returned None")

                # Balance fields: .tao gives the float in token units.
                alpha_out_emission_per_block = _balance_to_float(
                    getattr(subnet, "alpha_out_emission", None)
                )
                alpha_out = _balance_to_float(
                    getattr(subnet, "alpha_out", None)
                )

                if (
                    alpha_out_emission_per_block is None
                    or alpha_out is None
                    or alpha_out <= 0
                ):
                    raise ValueError(
                        f"DynamicInfo missing fields for netuid={netuid}: "
                        f"alpha_out_emission={alpha_out_emission_per_block}, "
                        f"alpha_out={alpha_out}"
                    )

                daily_alpha_emission = (
                    alpha_out_emission_per_block * self.BLOCKS_PER_DAY
                )
                daily_to_delegators = daily_alpha_emission * self.DELEGATOR_SHARE
                return daily_to_delegators / alpha_out
            finally:
                try:
                    await sub.close()
                except Exception:
                    pass

        return asyncio.run(_run())


class EmpiricalYieldProvider:
    """Fetch per-subnet daily yield rate from local historical CSVs.

    Computes a rolling mean from ``subnet_history.csv`` and
    ``pool_history.csv`` over the last ``window_days``. Third-tier
    fallback: only reached if both taostats and the chain RPC are
    unreachable. Useful as a deterministic default for backtests
    against historical data.

    Derivation (from ``docs/data-dictionary.md`` §"Data Notes"):

        subnet_history.csv ``emission``      = rao per tempo
        pool_history.csv ``alpha_staked``    = rao (alpha outside pool)
        blocks_per_day                       = 7200
        tempos_per_day                       = blocks_per_day / tempo
        daily_alpha_emission (rao)           = emission × tempos_per_day
        daily_to_delegators (rao)            = daily_alpha_emission × DELEGATOR_SHARE
        rate_per_day (dimensionless)         = daily_to_delegators / alpha_staked

    Both numerator and denominator are in rao so units cancel.
    ``DELEGATOR_SHARE = 0.82`` matches ``ChainYieldProvider`` — see
    notes there.

    Files resolved via the ``data_dir`` constructor arg; override for
    tests or alternate layouts.
    """
    source = AlphaYieldSource.EMPIRICAL

    DELEGATOR_SHARE: float = 0.82
    BLOCKS_PER_DAY: float = 7200.0

    def __init__(
        self,
        data_dir: str = "data/taostats",
        window_days: int = 7,
    ):
        self._data_dir = data_dir
        self._window_days = window_days
        self._cache: dict[int, float] = {}
        self._loaded_at: float = 0.0

    def fetch_rate(self, netuid: int) -> float:
        self._maybe_reload()
        if netuid not in self._cache:
            raise KeyError(f"no empirical yield data for netuid={netuid}")
        return self._cache[netuid]

    def _maybe_reload(self) -> None:
        """Reload from disk at most once per hour."""
        now = time.time()
        if now - self._loaded_at < 3600.0 and self._cache:
            return

        import os

        import pandas as pd

        subnet_path = os.path.join(self._data_dir, "subnet_history.csv")
        pool_path = os.path.join(self._data_dir, "pool_history.csv")

        subnet = pd.read_csv(subnet_path)
        pool = pd.read_csv(pool_path)

        # Normalize date columns
        for df in (subnet, pool):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"])

        cutoff = subnet["date"].max() - pd.Timedelta(days=self._window_days)
        recent_subnet = subnet[subnet["date"] >= cutoff]
        recent_pool = pool[pool["date"] >= cutoff]

        tempo = (
            recent_subnet["tempo"].mean()
            if "tempo" in recent_subnet.columns
            else 360.0
        )
        tempos_per_day = self.BLOCKS_PER_DAY / max(float(tempo), 1.0)

        agg_emission = recent_subnet.groupby("netuid")["emission"].mean()
        agg_alpha_staked = recent_pool.groupby("netuid")["alpha_staked"].mean()

        out: dict[int, float] = {}
        for netuid, em_rao_per_tempo in agg_emission.items():
            alpha_staked_rao = agg_alpha_staked.get(netuid)
            if alpha_staked_rao is None or alpha_staked_rao <= 0:
                continue
            daily_alpha = em_rao_per_tempo * tempos_per_day
            daily_to_delegators = daily_alpha * self.DELEGATOR_SHARE
            # Both sides rao → dimensionless rate_per_day.
            out[int(netuid)] = float(daily_to_delegators / alpha_staked_rao)

        self._cache = out
        self._loaded_at = now


# ── Helpers ───────────────────────────────────────────────────────────

def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _balance_to_float(v) -> Optional[float]:
    """Coerce a bittensor Balance (or None / raw numeric) to a float in
    token units. Balance stores rao internally; ``.tao`` returns the
    float token value regardless of which subnet the Balance is tagged
    with (the attribute is named ``.tao`` historically; for alpha
    Balances it holds alpha tokens).
    """
    if v is None:
        return None
    if hasattr(v, "tao"):
        try:
            return float(v.tao)
        except Exception:
            pass
    if hasattr(v, "rao"):
        try:
            return float(v.rao) / 1e9
        except Exception:
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_record(payload) -> dict:
    """Taostats often wraps single items in {'data': [...]} — unwrap."""
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            if not payload["data"]:
                raise ValueError("empty taostats data array")
            return payload["data"][0]
        return payload
    if isinstance(payload, list):
        if not payload:
            raise ValueError("empty taostats list")
        return payload[0]
    raise ValueError(f"unexpected taostats payload shape: {type(payload)}")
