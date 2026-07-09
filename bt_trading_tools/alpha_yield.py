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

    Instrumentation (added 2026-07-09): every call to
    ``fetch_rate_with_source`` updates a per-tier hit counter accessible via
    ``tier_hits()``. The 2026-07-08 yield audit measured 4-11 per-bot silent
    fall-throughs to zero per quarter using log-mtime heuristics; this
    counter makes future audits a `provider.tier_hits()` call instead of
    log-mining. Also tracks per-tier failure counts + last error message.
    """
    source = AlphaYieldSource.FALLBACK   # only used if every tier fails

    def __init__(self, providers: list[YieldRateProvider]):
        self._providers = list(providers)
        # {AlphaYieldSource: int} — one bucket per real tier + FALLBACK.
        # Bucket keys are the sources exposed by the providers; add the
        # fallback bucket explicitly since it isn't a "provider" object.
        self._tier_hits: dict[AlphaYieldSource, int] = {
            p.source: 0 for p in self._providers
        }
        self._tier_hits[AlphaYieldSource.FALLBACK] = 0
        self._tier_failures: dict[AlphaYieldSource, int] = {
            p.source: 0 for p in self._providers
        }
        # Last error per tier, useful for post-mortem after a spike in
        # failures. Keep small (just the type + repr, not full traceback).
        self._tier_last_error: dict[AlphaYieldSource, Optional[str]] = {
            p.source: None for p in self._providers
        }

    def fetch_rate_with_source(self, netuid: int) -> tuple[float, AlphaYieldSource, Optional[str]]:
        """Return (rate, source, error). Never raises."""
        last_error: Optional[str] = None
        for p in self._providers:
            try:
                rate = float(p.fetch_rate(netuid))
                if rate < 0 or not _finite(rate):
                    raise ValueError(f"invalid rate {rate!r} from {p.source}")
                self._tier_hits[p.source] = self._tier_hits.get(p.source, 0) + 1
                return rate, p.source, None
            except Exception as e:
                last_error = f"{p.source}:{type(e).__name__}:{e}"
                self._tier_failures[p.source] = self._tier_failures.get(p.source, 0) + 1
                self._tier_last_error[p.source] = f"{type(e).__name__}: {e!s:.200}"
                logger.debug(
                    "YieldProvider %s failed for netuid=%d: %s",
                    p.source, netuid, e,
                )
        self._tier_hits[AlphaYieldSource.FALLBACK] = (
            self._tier_hits.get(AlphaYieldSource.FALLBACK, 0) + 1
        )
        return 0.0, AlphaYieldSource.FALLBACK, last_error

    # ── Instrumentation accessors (added 2026-07-09) ────────────────────

    def tier_hits(self) -> dict[AlphaYieldSource, int]:
        """Return a copy of the per-tier hit counter.

        A "hit" = the tier successfully returned a rate. The FALLBACK
        bucket counts calls where every real tier failed (silent zero
        events — the case the 2026-07-08 audit flagged). Copies so
        callers can't mutate internal state.
        """
        return dict(self._tier_hits)

    def tier_failures(self) -> dict[AlphaYieldSource, int]:
        """Return a copy of the per-tier failure counter.

        A "failure" = the tier raised or returned an invalid rate. Same
        call can produce a failure for tier N (counted here) and a hit
        for tier N+1 (counted in tier_hits) — the cascade tried multiple.
        """
        return dict(self._tier_failures)

    def tier_last_error(self) -> dict[AlphaYieldSource, Optional[str]]:
        """Return a copy of the last-error-per-tier map.

        None means the tier has never failed since construction (or has
        never been called). Errors are truncated to ~200 chars so a
        blown-up traceback doesn't sit in memory forever.
        """
        return dict(self._tier_last_error)

    def instrumentation_snapshot(self) -> dict:
        """Combined single-object snapshot for logging / diagnostic dumps.

        Shape:
            {
              "hits":     {"validator_cache": 12345, "taostats": 42, ...},
              "failures": {"validator_cache": 3, "taostats": 8, ...},
              "last_error": {"validator_cache": null, "taostats": "...", ...},
              "total_calls": 12345,
              "fallback_rate": 0.0001,
            }
        Fallback rate = FALLBACK hits / total calls; the audit-relevant
        metric is fallback_rate stays close to zero.
        """
        hits = self.tier_hits()
        total = sum(hits.values())
        fallback = hits.get(AlphaYieldSource.FALLBACK, 0)
        return {
            "hits": {s.value: n for s, n in hits.items()},
            "failures": {s.value: n for s, n in self.tier_failures().items()},
            "last_error": {s.value: e for s, e in self.tier_last_error().items()},
            "total_calls": total,
            "fallback_rate": (fallback / total) if total else 0.0,
        }

    def reset_instrumentation(self) -> None:
        """Zero every counter + clear last-error. Useful for windowed audits."""
        for k in list(self._tier_hits.keys()):
            self._tier_hits[k] = 0
        for k in list(self._tier_failures.keys()):
            self._tier_failures[k] = 0
        for k in list(self._tier_last_error.keys()):
            self._tier_last_error[k] = None


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


class ValidatorCacheYieldProvider:
    """Fetch per-subnet yield rate from the validator-selection cache file.

    The cache (written by the daily ``update_validator_selection.py`` cron in
    the alpha-trading repo) contains per-validator ``apy_after_fee`` (30d-
    smoothed) plus ``stake_alpha`` per candidate per subnet. We compute the
    stake-weighted mean across the subnet's candidates and divide by 365 to
    express a simple-interest daily rate — same convention as
    ``TaostatsYieldProvider``.

    **Why this exists:** ``TaostatsYieldProvider`` makes a live network call
    on cache miss (~250ms-5s per call). When a paper bot suddenly has many
    fresh positions (e.g. a yield-carry rebalance settling), the first MTM
    iteration hits N cold cache slots → N live API calls → minutes of tick
    latency. The validator-selection cron already pulls per-validator APY
    daily, so reading from that cache gives O(1) per-netuid lookup at sub-ms
    cost. 24h-stale by design — yield rates don't move minute-to-minute
    meaningfully; the freshness loss is dominated by the speedup.

    **Cache schema:** requires schema_version >= 2 (the 7d/30d split). v3
    candidates carry chain-fresh metadata too but the yield computation only
    needs apy_after_fee + stake_alpha, both present since v1.

    **Staleness:** if the file's ``generated_at`` is older than ``max_age_s``
    (default 36h), raises so the surrounding ``CascadingYieldProvider`` falls
    through to live providers. 36h matches
    ``ValidatorSelector.DEFAULT_MAX_AGE_S`` — same convention.

    Strategies that need fresher yield data (none currently planned) can opt
    out by constructing their own ``AlphaYieldModel`` with
    ``TaostatsYieldProvider`` first in the cascade.
    """
    source = AlphaYieldSource.VALIDATOR_CACHE

    DEFAULT_CACHE_PATH = "/root/.validator_selection/best_validators.json"
    DEFAULT_MAX_AGE_S: float = 36 * 3600

    def __init__(
        self,
        cache_path: "str | None" = None,
        max_age_s: float = DEFAULT_MAX_AGE_S,
    ):
        from pathlib import Path
        self._cache_path = Path(cache_path or self.DEFAULT_CACHE_PATH)
        self._max_age_s = max_age_s
        self._cache: Optional[dict] = None
        self._cache_mtime: float = 0.0

    def fetch_rate(self, netuid: int) -> float:
        """Return alpha-per-alpha per-day yield rate, derived from the
        validator-selection cache file.

        Raises on missing file / stale file / missing netuid / no usable
        candidates so the cascade falls through to live providers.
        """
        self._maybe_reload()
        if self._cache is None:
            raise RuntimeError(
                f"validator-selection cache not loadable at {self._cache_path}"
            )

        # Staleness gate
        from datetime import datetime, timezone
        try:
            gen = datetime.fromisoformat(
                self._cache["generated_at"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError) as e:
            raise RuntimeError(
                f"validator-selection cache missing/invalid generated_at: {e}"
            )
        age_s = (datetime.now(timezone.utc) - gen).total_seconds()
        if age_s > self._max_age_s:
            raise RuntimeError(
                f"validator-selection cache stale: {age_s:.0f}s old "
                f"(max {self._max_age_s:.0f}s)"
            )

        subnet = self._cache.get("subnets", {}).get(str(netuid))
        if not subnet:
            raise KeyError(
                f"no validator-selection cache entry for netuid={netuid}"
            )
        cands = subnet.get("candidates") or []
        if not cands:
            raise ValueError(
                f"no candidates in validator-selection cache for netuid={netuid}"
            )

        # Stake-weighted mean of 30-day APY (same convention as
        # TaostatsYieldProvider). apy_after_fee is already 30d-smoothed +
        # post-take in the cache file.
        total_stake = 0.0
        weighted_apy = 0.0
        for c in cands:
            stake = _as_float(c.get("stake_alpha"))
            apy = _as_float(c.get("apy_after_fee"))
            if stake is None or apy is None or stake <= 0 or apy < 0:
                continue
            total_stake += stake
            weighted_apy += apy * stake

        if total_stake <= 0:
            raise ValueError(
                f"no usable (stake, apy) pairs in validator-selection cache "
                f"for netuid={netuid}"
            )
        apy_mean = weighted_apy / total_stake
        return apy_mean / 365.0

    def _maybe_reload(self) -> None:
        if not self._cache_path.exists():
            self._cache = None
            return
        mtime = self._cache_path.stat().st_mtime
        if self._cache is not None and mtime == self._cache_mtime:
            return
        import json
        try:
            with open(self._cache_path) as f:
                self._cache = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._cache = None
            raise RuntimeError(
                f"failed to read validator-selection cache "
                f"{self._cache_path}: {e}"
            )
        self._cache_mtime = mtime


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

    Derivation (Bittensor participant-alpha distribution, see
    ``docs/bittensor-mechanics-primer.md`` §6 — NOT the TAO-denominated
    ``emission`` column):

        participant_alpha/block = PARTICIPANT_ALPHA_PER_BLOCK   (0.5 post-halving)
        daily_participant_alpha = participant_alpha/block × BLOCKS_PER_DAY
        subnet_staker_alpha/day = daily_participant_alpha
                                  × VALIDATOR_STAKER_SHARE       (0.41)
                                  × (1 − root_prop)              (per-subnet, pool_history)
        delegator_alpha/day     = subnet_staker_alpha/day × DELEGATOR_SHARE (0.82)
        rate_per_day            = delegator_alpha/day ÷ (alpha_staked_rao / 1e9)

    Prior bug (fixed 2026-06): the formula used ``subnet_history.emission``,
    which is the subnet's *TAO* Taoflow share (rao of TAO), as if it were
    alpha emission, and divided by alpha-rao staked. The units did not
    cancel and the result was ~0 yield for every subnet (~1000× too low);
    a separate ISO8601 date-parse crash sent the whole cascade to the zero
    fallback. Both are corrected here.

    CAVEAT: ``PARTICIPANT_ALPHA_PER_BLOCK`` is fixed at the post-halving
    0.5 alpha/block default. Subnets that have not yet halved emit 1.0
    alpha/block, so this UNDERSTATES their yield by up to 2×. The taostats
    CSVs carry no per-subnet alpha-emission or halving state; for
    per-subnet-accurate emission use ``ChainYieldProvider`` (reads
    ``alpha_out_emission`` from chain DynamicInfo) or a live APY source
    (``ValidatorCacheYieldProvider`` / ``TaostatsYieldProvider``). The
    rate is also a trailing-window mean (time-invariant within a backtest),
    so it does not capture a subnet's yield changing over a position's hold.

    Files resolved via the ``data_dir`` constructor arg; override for
    tests or alternate layouts.
    """
    source = AlphaYieldSource.EMPIRICAL

    DELEGATOR_SHARE: float = 0.82            # 1 − ≤18% validator commission (§6)
    VALIDATOR_STAKER_SHARE: float = 0.41     # validator+staker share of participant alpha (§6)
    PARTICIPANT_ALPHA_PER_BLOCK: float = 0.5  # post-halving default (see CAVEAT)
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

        pool_path = os.path.join(self._data_dir, "pool_history.csv")
        pool = pd.read_csv(pool_path, low_memory=False)

        # Normalize date column. Parse with ISO8601 explicitly: the CSVs mix
        # microsecond and whole-second timestamps, which crashes a
        # format-inferred parse (and previously sent the cascade to zero).
        if "date" in pool.columns:
            pool["date"] = pd.to_datetime(pool["date"], format="ISO8601", utc=True)
        elif "timestamp" in pool.columns:
            pool["date"] = pd.to_datetime(pool["timestamp"], format="ISO8601", utc=True)

        cutoff = pool["date"].max() - pd.Timedelta(days=self._window_days)
        recent = pool[pool["date"] >= cutoff]

        agg_alpha_staked = recent.groupby("netuid")["alpha_staked"].mean()
        agg_root_prop = (
            recent.groupby("netuid")["root_prop"].mean()
            if "root_prop" in recent.columns
            else None
        )

        daily_participant_alpha = self.PARTICIPANT_ALPHA_PER_BLOCK * self.BLOCKS_PER_DAY

        out: dict[int, float] = {}
        for netuid, alpha_staked_rao in agg_alpha_staked.items():
            if alpha_staked_rao is None or alpha_staked_rao <= 0:
                continue
            root_prop = (
                float(agg_root_prop.get(netuid, 0.0)) if agg_root_prop is not None else 0.0
            )
            if not (0.0 <= root_prop <= 1.0):
                root_prop = 0.0
            delegator_alpha_day = (
                daily_participant_alpha
                * self.VALIDATOR_STAKER_SHARE
                * (1.0 - root_prop)
                * self.DELEGATOR_SHARE
            )
            alpha_staked_tokens = float(alpha_staked_rao) / 1e9
            out[int(netuid)] = delegator_alpha_day / alpha_staked_tokens

        self._cache = out
        self._loaded_at = now


# ── Default model factory ─────────────────────────────────────────────

def _validator_cache_fallback_paths() -> tuple[str, ...]:
    """Fallback paths searched (in order) when VALIDATOR_CACHE_PATH env is
    unset. First existing file wins. Production VPS first; local-user and
    research-snapshot paths exist for dev / backtest sessions where the
    operator has copied the validator cache out of bot-vps for analysis.
    """
    from pathlib import Path
    return (
        "/root/.validator_selection/best_validators.json",
        str(Path.home() / ".validator_selection" / "best_validators.json"),
        "/tmp/autobot_live_data/best_validators.json",
    )


def _resolve_validator_cache_path():
    """Return the first existing validator-cache path from env → fallback list,
    or None if none exist. Returns ``pathlib.Path`` or ``None``.
    """
    import os
    from pathlib import Path
    cache_path_env = os.environ.get("VALIDATOR_CACHE_PATH")
    if cache_path_env:
        p = Path(cache_path_env)
        return p if p.exists() else None
    for candidate in _validator_cache_fallback_paths():
        p = Path(candidate)
        if p.exists():
            return p
    return None


def build_default_yield_model() -> "AlphaYieldModel":
    """Construct an ``AlphaYieldModel`` with an env-driven cascade.

    Auto-selects providers based on environment variables that the
    production fleet always sets. Each provider is included only when
    its required configuration is present, so test environments
    (no env vars set) get a cascade that fast-fails to zero without
    making slow chain RPC attempts.

    Configuration:

        VALIDATOR_CACHE_PATH → enables ValidatorCacheYieldProvider (explicit override)
        TAOSTATS_API_KEY    → enables TaostatsYieldProvider (live API)
        BT_NETWORK          → enables ChainYieldProvider (live chain RPC)
        TAOSTATS_DATA_DIR   → enables EmpiricalYieldProvider (historical CSV)

    Validator-cache path fallback search order (when VALIDATOR_CACHE_PATH
    env is unset; first existing file wins):
      1. /root/.validator_selection/best_validators.json (production VPS)
      2. ~/.validator_selection/best_validators.json (local dev)
      3. /tmp/autobot_live_data/best_validators.json (research snapshot path
         used by Phase 0 autobot calibration; convention for paper-trading
         snapshots pulled from bot-vps)

    Cascade order: validator-cache (sub-ms, daily-fresh) > live taostats >
    live chain > historical CSV > built-in zero fallback. The validator
    cache is preferred first because it's the fastest path that produces
    correct values; the live providers handle cache-stale or cache-missing
    cases via the cascade fall-through.

    This function is the canonical default for both ``PaperBotBase`` and
    ``BacktestEngine``. Production paper bots have all three env vars
    set in their systemd units, so they get full live yield. Research
    sessions on dev / laptop typically get the validator cache via the
    fallback search above (one of those paths is usually present).

    Returns:
        ``AlphaYieldModel`` with a ``CascadingYieldProvider`` whose tier
        list reflects the current environment. The cascade itself never
        raises (last-tier built-in fallback is zero with source=FALLBACK).
    """
    import os
    from pathlib import Path

    providers: list[YieldRateProvider] = []

    # Validator-selection cache: sub-ms per-netuid lookup, daily-fresh.
    # Resolved via env override OR fallback paths. None when no cache present.
    cache_path = _resolve_validator_cache_path()
    if cache_path is not None:
        providers.append(ValidatorCacheYieldProvider(cache_path=str(cache_path)))

    # Taostats live — fast-fails when no API key is set, so this is safe in
    # test environments. In production it's the freshest source.
    providers.append(TaostatsYieldProvider())

    # Chain RPC: only include when BT_NETWORK is explicitly configured. Skip
    # in test environments to avoid slow chain-connect timeouts on every
    # cascade tier-fall-through.
    bt_network = os.environ.get("BT_NETWORK")
    if bt_network:
        providers.append(ChainYieldProvider(network=bt_network))

    # Empirical CSV: only include when TAOSTATS_DATA_DIR points at a real
    # directory. Backtests set this to the data dir; paper bots may also
    # set it for fall-through if both API and chain are down.
    data_dir = os.environ.get("TAOSTATS_DATA_DIR")
    if data_dir and Path(data_dir).exists():
        providers.append(EmpiricalYieldProvider(data_dir=data_dir))

    return AlphaYieldModel(CascadingYieldProvider(providers))


# ── Validator dilution ────────────────────────────────────────────────

def dilute_apy(
    headline_apy: float,
    our_alpha: float,
    validator_stake_alpha: float,
) -> float:
    """Validator yield diluted by adding our position to the validator's stake pool.

    A validator's per-stake yield is split pro-rata among delegators. When we
    delegate ``our_alpha`` to a validator already holding ``validator_stake_alpha``,
    our slice of the per-epoch alpha bucket is::

        our_share = our_alpha / (validator_stake_alpha + our_alpha)

    For the *headline_apy* (our_alpha=0 reference, ignoring our own contribution),
    the validator's per-stake yield is approximately:

        per_stake_apy = headline_apy / our_alpha     # NOT directly useful

    The realized APY for our position after dilution is::

        diluted_apy = headline_apy × validator_stake_alpha / (validator_stake_alpha + our_alpha)

    For ``our_alpha << validator_stake_alpha`` this reduces to ``headline_apy``
    (no dilution). For ``our_alpha == validator_stake_alpha`` we get half.

    Args:
        headline_apy: validator's reported APY before our position joins.
        our_alpha: our position size in alpha tokens.
        validator_stake_alpha: validator's existing stake in alpha tokens.

    Returns:
        Realized APY after dilution. Returns 0.0 if validator_stake_alpha ≤ 0
        (no pool to dilute into → cannot earn yield).

    Raises:
        ValueError: if our_alpha < 0.
    """
    if our_alpha < 0:
        raise ValueError(f"our_alpha must be non-negative, got {our_alpha}")
    if validator_stake_alpha <= 0:
        return 0.0
    return headline_apy * validator_stake_alpha / (validator_stake_alpha + our_alpha)


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
