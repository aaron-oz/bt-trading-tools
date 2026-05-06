"""Tests for bt_trading_tools.alpha_yield — pure yield accrual.

Covers:
* AlphaYieldModel.accrued_yield is pure (idempotent under repeated calls)
* accrued scales linearly in alpha_qty and elapsed time
* rate cache honors TTL
* CascadingYieldProvider tries tiers in order, falls through cleanly
* Degenerate inputs (neg alpha, now <= entry) return 0.0
* Invalid rates (negative, NaN, non-numeric) skip their tier
"""

from __future__ import annotations

import math
import time
import unittest

from bt_trading_tools.alpha_yield import (
    AlphaYieldModel,
    CascadingYieldProvider,
    ValidatorCacheYieldProvider,
    YieldRate,
    ZeroYieldProvider,
    build_default_yield_model,
    dilute_apy,
)
from bt_trading_tools.tracking.schema import AlphaYieldSource


class _StubProvider:
    """Return a fixed rate; count fetch_rate calls."""

    def __init__(self, rate, source=AlphaYieldSource.TAOSTATS):
        self.rate = rate
        self.source = source
        self.calls = 0

    def fetch_rate(self, netuid):
        self.calls += 1
        if callable(self.rate):
            return self.rate(netuid)
        return self.rate


class _FailingProvider:
    """Always raises."""

    def __init__(self, source=AlphaYieldSource.TAOSTATS, exc=RuntimeError("boom")):
        self.source = source
        self._exc = exc
        self.calls = 0

    def fetch_rate(self, netuid):
        self.calls += 1
        raise self._exc


class TestAccruedYield(unittest.TestCase):

    def test_purity_repeated_calls(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p, cache_ttl_s=60.0)
        entry = 1_000_000.0
        now = entry + 86400.0  # 1 day
        vals = [m.accrued_yield(23, 100.0, entry, now) for _ in range(200)]
        self.assertEqual(len(set(vals)), 1, "accrued_yield must be idempotent")

    def test_linear_in_alpha_qty(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        entry = 0.0
        now = 86400.0
        a = m.accrued_yield(1, 100.0, entry, now)
        b = m.accrued_yield(1, 200.0, entry, now)
        self.assertAlmostEqual(b, 2 * a)

    def test_linear_in_time(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        entry = 0.0
        a = m.accrued_yield(1, 100.0, entry, 86400.0)
        b = m.accrued_yield(1, 100.0, entry, 2 * 86400.0)
        self.assertAlmostEqual(b, 2 * a)

    def test_known_value(self):
        # 100 alpha × 0.003 rate × 2.5 days = 0.75 alpha
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        out = m.accrued_yield(1, 100.0, 0.0, 2.5 * 86400.0)
        self.assertAlmostEqual(out, 0.75)

    def test_degenerate_inputs(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        # Zero / negative alpha
        self.assertEqual(m.accrued_yield(1, 0.0, 0.0, 86400.0), 0.0)
        self.assertEqual(m.accrued_yield(1, -5.0, 0.0, 86400.0), 0.0)
        # now <= entry
        self.assertEqual(m.accrued_yield(1, 100.0, 100.0, 100.0), 0.0)
        self.assertEqual(m.accrued_yield(1, 100.0, 100.0, 50.0), 0.0)
        # Zero rate
        zero = AlphaYieldModel(ZeroYieldProvider())
        self.assertEqual(zero.accrued_yield(1, 100.0, 0.0, 86400.0), 0.0)

    def test_nonfinite_guard(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        self.assertEqual(m.accrued_yield(1, float("nan"), 0.0, 86400.0), 0.0)
        self.assertEqual(m.accrued_yield(1, 100.0, 0.0, float("inf")), 0.0)


class TestRateCache(unittest.TestCase):

    def test_cache_hit_within_ttl(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p, cache_ttl_s=60.0)
        for _ in range(10):
            m.rate(netuid=1)
        self.assertEqual(p.calls, 1)

    def test_distinct_netuids_miss(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p, cache_ttl_s=60.0)
        m.rate(netuid=1)
        m.rate(netuid=2)
        m.rate(netuid=3)
        self.assertEqual(p.calls, 3)

    def test_cache_expiry(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p, cache_ttl_s=0.01)
        m.rate(netuid=1)
        time.sleep(0.02)
        m.rate(netuid=1)
        self.assertEqual(p.calls, 2)

    def test_clear_cache(self):
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p, cache_ttl_s=60.0)
        m.rate(netuid=1)
        m.clear_cache()
        m.rate(netuid=1)
        self.assertEqual(p.calls, 2)


class TestCascadingProvider(unittest.TestCase):

    def test_first_success_wins(self):
        p1 = _StubProvider(rate=0.01, source=AlphaYieldSource.TAOSTATS)
        p2 = _StubProvider(rate=0.02, source=AlphaYieldSource.CHAIN)
        cascade = CascadingYieldProvider([p1, p2])
        rate, source, err = cascade.fetch_rate_with_source(netuid=1)
        self.assertAlmostEqual(rate, 0.01)
        self.assertEqual(source, AlphaYieldSource.TAOSTATS)
        self.assertIsNone(err)
        # Second provider is not consulted
        self.assertEqual(p1.calls, 1)
        self.assertEqual(p2.calls, 0)

    def test_falls_through_on_failure(self):
        p1 = _FailingProvider(source=AlphaYieldSource.TAOSTATS)
        p2 = _StubProvider(rate=0.02, source=AlphaYieldSource.CHAIN)
        cascade = CascadingYieldProvider([p1, p2])
        rate, source, err = cascade.fetch_rate_with_source(netuid=1)
        self.assertAlmostEqual(rate, 0.02)
        self.assertEqual(source, AlphaYieldSource.CHAIN)
        # Error was recorded but overridden by success
        self.assertIsNone(err)

    def test_all_fail_to_zero_fallback(self):
        p1 = _FailingProvider(source=AlphaYieldSource.TAOSTATS)
        p2 = _FailingProvider(source=AlphaYieldSource.CHAIN)
        p3 = _FailingProvider(source=AlphaYieldSource.EMPIRICAL)
        cascade = CascadingYieldProvider([p1, p2, p3])
        rate, source, err = cascade.fetch_rate_with_source(netuid=1)
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, AlphaYieldSource.FALLBACK)
        self.assertIsNotNone(err)

    def test_negative_rate_skipped(self):
        p1 = _StubProvider(rate=-0.01, source=AlphaYieldSource.TAOSTATS)
        p2 = _StubProvider(rate=0.02, source=AlphaYieldSource.CHAIN)
        cascade = CascadingYieldProvider([p1, p2])
        rate, source, _err = cascade.fetch_rate_with_source(netuid=1)
        self.assertAlmostEqual(rate, 0.02)
        self.assertEqual(source, AlphaYieldSource.CHAIN)

    def test_nan_rate_skipped(self):
        p1 = _StubProvider(rate=float("nan"), source=AlphaYieldSource.TAOSTATS)
        p2 = _StubProvider(rate=0.02, source=AlphaYieldSource.CHAIN)
        cascade = CascadingYieldProvider([p1, p2])
        rate, source, _err = cascade.fetch_rate_with_source(netuid=1)
        self.assertAlmostEqual(rate, 0.02)
        self.assertEqual(source, AlphaYieldSource.CHAIN)


class TestModelAcceptsBareProvider(unittest.TestCase):

    def test_bare_provider_wrapped(self):
        """Passing a single provider (not a cascade) should still work."""
        p = _StubProvider(rate=0.003)
        m = AlphaYieldModel(p)
        self.assertAlmostEqual(
            m.accrued_yield(1, 100.0, 0.0, 86400.0), 0.3,
        )

    def test_bare_provider_failure_falls_back(self):
        p = _FailingProvider()
        m = AlphaYieldModel(p)
        yr = m.rate(netuid=1)
        self.assertEqual(yr.rate_per_day, 0.0)
        self.assertEqual(yr.source, AlphaYieldSource.FALLBACK)


class TestDiluteApy(unittest.TestCase):
    """dilute_apy: validator's APY after our position joins their stake pool."""

    def test_no_dilution_when_our_position_is_tiny(self):
        # 1 alpha into a 10000-alpha validator → effectively no dilution
        diluted = dilute_apy(headline_apy=0.50, our_alpha=1.0, validator_stake_alpha=10000.0)
        self.assertAlmostEqual(diluted, 0.50, places=3)

    def test_half_apy_when_we_match_validator_stake(self):
        # our_alpha == validator_stake → diluted to half
        diluted = dilute_apy(headline_apy=0.50, our_alpha=1000.0, validator_stake_alpha=1000.0)
        self.assertAlmostEqual(diluted, 0.25)

    def test_quarter_apy_when_we_triple_validator_stake(self):
        # our_alpha = 3 × validator_stake → split is 1:3, validator-side share = 1/4
        diluted = dilute_apy(headline_apy=0.40, our_alpha=3000.0, validator_stake_alpha=1000.0)
        self.assertAlmostEqual(diluted, 0.10)

    def test_zero_validator_stake_returns_zero(self):
        # No existing stake to delegate to → cannot earn yield
        self.assertEqual(
            dilute_apy(headline_apy=0.50, our_alpha=100.0, validator_stake_alpha=0.0),
            0.0,
        )

    def test_zero_our_position_returns_headline(self):
        # Reference case: our position = 0 means we ARE the headline
        self.assertEqual(
            dilute_apy(headline_apy=0.42, our_alpha=0.0, validator_stake_alpha=10000.0),
            0.42,
        )

    def test_negative_our_alpha_raises(self):
        with self.assertRaises(ValueError):
            dilute_apy(headline_apy=0.50, our_alpha=-1.0, validator_stake_alpha=1000.0)

    def test_negative_validator_stake_returns_zero(self):
        # Garbage in (negative stake) → 0 rather than crash
        self.assertEqual(
            dilute_apy(headline_apy=0.50, our_alpha=100.0, validator_stake_alpha=-1.0),
            0.0,
        )


# ── ValidatorCacheYieldProvider (Scope C Phase 2) ──────────────────────

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _make_validator_cache(
    subnets: dict,
    generated_at: datetime | None = None,
    schema_version: int = 3,
) -> dict:
    """Build a minimal validator-selection cache payload for tests."""
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    return {
        "schema_version": schema_version,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "data_source": "taostats+chain",
        "subnets": {
            str(netuid): {
                "n_validators_total": len(cands),
                "n_validators_passing_filters": len(cands),
                "candidates": cands,
            }
            for netuid, cands in subnets.items()
        },
    }


class TestValidatorCacheYieldProvider(unittest.TestCase):
    """Phase 2: cache-backed yield provider for fast MTM."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.tmpdir.name) / "best_validators.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, payload: dict) -> None:
        self.cache_path.write_text(json.dumps(payload))

    def test_returns_stake_weighted_mean_apy_per_day(self):
        """fetch_rate returns stake-weighted mean apy / 365.

        Two validators on netuid 44:
          - apy=0.50, stake=10000  →  contributes 0.50 × 10000 = 5000
          - apy=0.20, stake=40000  →  contributes 0.20 × 40000 = 8000
        weighted mean = 13000 / 50000 = 0.26
        per-day = 0.26 / 365 ≈ 7.123e-4
        """
        payload = _make_validator_cache({
            44: [
                {"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0},
                {"hotkey": "hk_B", "apy_after_fee": 0.20, "stake_alpha": 40000.0},
            ]
        })
        self._write(payload)
        prov = ValidatorCacheYieldProvider(cache_path=str(self.cache_path))
        rate = prov.fetch_rate(netuid=44)
        self.assertAlmostEqual(rate, 0.26 / 365.0, places=8)

    def test_zero_or_negative_stake_skipped(self):
        """Candidates with stake_alpha <= 0 don't contribute to the mean."""
        payload = _make_validator_cache({
            44: [
                {"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0},
                {"hotkey": "hk_B", "apy_after_fee": 9.99, "stake_alpha": 0.0},
                {"hotkey": "hk_C", "apy_after_fee": 9.99, "stake_alpha": -1.0},
            ]
        })
        self._write(payload)
        prov = ValidatorCacheYieldProvider(cache_path=str(self.cache_path))
        rate = prov.fetch_rate(netuid=44)
        # Only hk_A contributes → mean = 0.50, daily = 0.50/365
        self.assertAlmostEqual(rate, 0.50 / 365.0, places=8)

    def test_raises_on_stale_cache(self):
        """Cache older than max_age_s raises so cascade falls through."""
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        payload = _make_validator_cache(
            {44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0}]},
            generated_at=old,
        )
        self._write(payload)
        prov = ValidatorCacheYieldProvider(
            cache_path=str(self.cache_path), max_age_s=24 * 3600,
        )
        with self.assertRaises(RuntimeError) as ctx:
            prov.fetch_rate(netuid=44)
        self.assertIn("stale", str(ctx.exception))

    def test_raises_on_missing_netuid(self):
        """Missing subnet → KeyError so cascade falls through."""
        payload = _make_validator_cache({
            44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0}]
        })
        self._write(payload)
        prov = ValidatorCacheYieldProvider(cache_path=str(self.cache_path))
        with self.assertRaises(KeyError):
            prov.fetch_rate(netuid=99)

    def test_raises_on_no_usable_candidates(self):
        """Subnet with only zero-stake candidates → ValueError."""
        payload = _make_validator_cache({
            44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 0.0}]
        })
        self._write(payload)
        prov = ValidatorCacheYieldProvider(cache_path=str(self.cache_path))
        with self.assertRaises(ValueError):
            prov.fetch_rate(netuid=44)

    def test_raises_when_file_missing(self):
        prov = ValidatorCacheYieldProvider(
            cache_path=str(self.cache_path / "does_not_exist"),
        )
        with self.assertRaises(RuntimeError):
            prov.fetch_rate(netuid=44)

    def test_cascade_falls_through_on_stale_cache(self):
        """End-to-end: stale cache provider in front of a working stub →
        cascade falls to the stub. Verifies the integration contract."""
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        payload = _make_validator_cache(
            {44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0}]},
            generated_at=old,
        )
        self._write(payload)
        cache_prov = ValidatorCacheYieldProvider(
            cache_path=str(self.cache_path), max_age_s=24 * 3600,
        )
        backup = _StubProvider(rate=0.001, source=AlphaYieldSource.TAOSTATS)
        cascade = CascadingYieldProvider([cache_prov, backup])
        rate, source, _err = cascade.fetch_rate_with_source(netuid=44)
        self.assertEqual(source, AlphaYieldSource.TAOSTATS)
        self.assertEqual(rate, 0.001)
        self.assertEqual(backup.calls, 1)

    def test_reload_on_mtime_change(self):
        """Updating the cache file is reflected on the next fetch."""
        import os
        payload1 = _make_validator_cache({
            44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0}]
        })
        self._write(payload1)
        prov = ValidatorCacheYieldProvider(cache_path=str(self.cache_path))
        rate1 = prov.fetch_rate(netuid=44)
        self.assertAlmostEqual(rate1, 0.50 / 365.0, places=8)
        # Rewrite with different APY + bump mtime
        time.sleep(0.05)
        payload2 = _make_validator_cache({
            44: [{"hotkey": "hk_A", "apy_after_fee": 0.10, "stake_alpha": 10000.0}]
        })
        self._write(payload2)
        new_mtime = time.time()
        os.utime(self.cache_path, (new_mtime, new_mtime))
        rate2 = prov.fetch_rate(netuid=44)
        self.assertAlmostEqual(rate2, 0.10 / 365.0, places=8)


class TestBuildDefaultYieldModelCacheFirst(unittest.TestCase):
    """build_default_yield_model puts ValidatorCacheYieldProvider FIRST when
    VALIDATOR_CACHE_PATH points at an existing file."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.tmpdir.name) / "best_validators.json"
        self.cache_path.write_text(json.dumps(_make_validator_cache({
            44: [{"hotkey": "hk_A", "apy_after_fee": 0.50, "stake_alpha": 10000.0}]
        })))

    def tearDown(self):
        import os
        for k in ("VALIDATOR_CACHE_PATH", "BT_NETWORK", "TAOSTATS_DATA_DIR"):
            os.environ.pop(k, None)
        self.tmpdir.cleanup()

    def test_cache_provider_first_in_cascade(self):
        import os
        os.environ["VALIDATOR_CACHE_PATH"] = str(self.cache_path)
        # Ensure the chain + empirical providers are NOT included to keep
        # the cascade list short and easy to reason about.
        os.environ.pop("BT_NETWORK", None)
        os.environ.pop("TAOSTATS_DATA_DIR", None)
        model = build_default_yield_model()
        # CascadingYieldProvider stores tiers internally; rely on the public
        # behavior — cache provider at netuid 44 should produce a rate
        # matching the synthetic cache, sourced as VALIDATOR_CACHE.
        yr = model.rate(netuid=44)
        self.assertEqual(yr.source, AlphaYieldSource.VALIDATOR_CACHE)
        self.assertAlmostEqual(yr.rate_per_day, 0.50 / 365.0, places=8)

    def test_cache_provider_skipped_when_file_missing(self):
        """If VALIDATOR_CACHE_PATH points at a non-existent file, the cache
        provider is omitted (no spurious failure on every cascade)."""
        import os
        os.environ["VALIDATOR_CACHE_PATH"] = "/nonexistent/path/best_validators.json"
        os.environ.pop("BT_NETWORK", None)
        os.environ.pop("TAOSTATS_DATA_DIR", None)
        model = build_default_yield_model()
        # Without taostats key + with no cache, taostats-tier raises and
        # the cascade falls through to the built-in zero fallback.
        yr = model.rate(netuid=44)
        self.assertEqual(yr.source, AlphaYieldSource.FALLBACK)


if __name__ == "__main__":
    unittest.main()
