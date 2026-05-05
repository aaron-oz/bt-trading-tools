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
    YieldRate,
    ZeroYieldProvider,
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


if __name__ == "__main__":
    unittest.main()
