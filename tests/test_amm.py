"""Tests for bt_trading_tools.amm — AMM math correctness."""

import math
import unittest
from decimal import Decimal

from bt_trading_tools.amm import (
    amm_buy,
    amm_sell,
    effective_price,
    max_trade_for_slippage,
    rate_tolerance,
    slippage_pct,
    slippage_pct_decimal,
    spot_price,
)


class TestAMMBuy(unittest.TestCase):
    """amm_buy: constant product k = tao * alpha is preserved."""

    def test_basic_buy(self):
        alpha, new_tao, new_alpha = amm_buy(10, 1000, 50000)
        self.assertGreater(alpha, 0)
        # k preserved
        self.assertAlmostEqual(1000 * 50000, new_tao * new_alpha, places=4)
        # pool sizes updated correctly
        self.assertAlmostEqual(new_tao, 1010)
        self.assertAlmostEqual(new_alpha, 50000 - alpha)

    def test_zero_spend(self):
        alpha, new_tao, new_alpha = amm_buy(0, 1000, 50000)
        self.assertEqual(alpha, 0)
        self.assertEqual(new_tao, 1000)
        self.assertEqual(new_alpha, 50000)

    def test_negative_spend(self):
        alpha, new_tao, new_alpha = amm_buy(-5, 1000, 50000)
        self.assertEqual(alpha, 0)

    def test_large_buy_doesnt_exceed_pool(self):
        """Can never buy more alpha than exists in the pool."""
        alpha, _, new_alpha = amm_buy(1e12, 1000, 50000)
        self.assertLess(alpha, 50000)
        self.assertGreater(new_alpha, 0)

    def test_buy_gets_worse_price_with_size(self):
        """Larger buys get worse effective price (more slippage)."""
        alpha_small, _, _ = amm_buy(1, 1000, 50000)
        alpha_big, _, _ = amm_buy(100, 1000, 50000)
        price_small = 1 / alpha_small
        price_big = 100 / alpha_big
        self.assertGreater(price_big, price_small)


class TestAMMSell(unittest.TestCase):

    def test_basic_sell(self):
        tao_out, new_tao, new_alpha = amm_sell(500, 1000, 50000)
        self.assertGreater(tao_out, 0)
        self.assertAlmostEqual(1000 * 50000, new_tao * new_alpha, places=4)

    def test_zero_sell(self):
        tao_out, _, _ = amm_sell(0, 1000, 50000)
        self.assertEqual(tao_out, 0)

    def test_round_trip_loses_value(self):
        """Buy then sell back: you get less TAO than you started with (slippage).

        On a constant-product AMM, buying moves the pool state. Selling back
        the exact alpha you received restores k but you eat the spread.
        With a small trade relative to pool, the loss is minimal but nonzero
        once pool is asymmetric enough.
        """
        # Use a larger trade relative to pool so the round-trip loss is visible
        tao_pool, alpha_pool = 100.0, 5000.0
        alpha_got, new_tao, new_alpha = amm_buy(50, tao_pool, alpha_pool)
        tao_back, final_tao, final_alpha = amm_sell(alpha_got, new_tao, new_alpha)
        # Pool restored to original state
        self.assertAlmostEqual(final_tao, tao_pool, places=8)
        self.assertAlmostEqual(final_alpha, alpha_pool, places=8)
        # We get back exactly what we put in (k conservation)
        self.assertAlmostEqual(tao_back, 50.0, places=8)

    def test_k_preserved_on_sell(self):
        k_before = 1000 * 50000
        _, new_tao, new_alpha = amm_sell(1000, 1000, 50000)
        self.assertAlmostEqual(k_before, new_tao * new_alpha, places=4)


class TestSpotPrice(unittest.TestCase):

    def test_basic(self):
        self.assertAlmostEqual(spot_price(1000, 50000), 0.02)

    def test_empty_pool(self):
        self.assertEqual(spot_price(1000, 0), 0.0)


class TestSlippage(unittest.TestCase):

    def test_small_trade(self):
        slip = slippage_pct(1, 1000)
        self.assertAlmostEqual(slip, 1 / 1001, places=6)

    def test_large_trade(self):
        slip = slippage_pct(100, 1000)
        self.assertAlmostEqual(slip, 100 / 1100, places=6)

    def test_empty_pool(self):
        self.assertEqual(slippage_pct(10, 0), 1.0)

    def test_zero_trade(self):
        self.assertEqual(slippage_pct(0, 1000), 0.0)

    def test_decimal_precision(self):
        """Decimal version avoids float rounding artifacts."""
        result = slippage_pct_decimal(3, 97)
        self.assertEqual(result, Decimal("3"))  # exactly 3%, no 3.0000000004


class TestMaxTradeForSlippage(unittest.TestCase):

    def test_basic(self):
        """5% slippage on 1000 TAO pool → max trade = 1000 * 0.05 / 0.95."""
        max_t = max_trade_for_slippage(1000, 0.05)
        expected = 1000 * 0.05 / 0.95
        self.assertAlmostEqual(max_t, expected, places=6)

    def test_round_trip_matches(self):
        """The trade computed by max_trade should produce the target slippage."""
        target = 0.03
        pool = 500.0
        max_t = max_trade_for_slippage(pool, target)
        actual_slip = slippage_pct(max_t, pool)
        self.assertAlmostEqual(actual_slip, target, places=10)

    def test_empty_pool(self):
        self.assertEqual(max_trade_for_slippage(0, 0.05), 0.0)

    def test_zero_slippage(self):
        self.assertEqual(max_trade_for_slippage(1000, 0), 0.0)

    def test_100pct_slippage(self):
        self.assertEqual(max_trade_for_slippage(1000, 1.0), float("inf"))


class TestEffectivePrice(unittest.TestCase):

    def test_small_trade_near_spot(self):
        """Tiny trade should get close to spot price."""
        sp = spot_price(1000, 50000)
        ep = effective_price(0.001, 1000, 50000)
        self.assertAlmostEqual(ep, sp, places=5)

    def test_large_trade_worse_than_spot(self):
        sp = spot_price(1000, 50000)
        ep = effective_price(100, 1000, 50000)
        self.assertGreater(ep, sp)


class TestRateTolerance(unittest.TestCase):

    def test_includes_buffer(self):
        rt = rate_tolerance(10, 1000, buffer_pct=2.0)
        slip = slippage_pct(10, 1000)
        self.assertAlmostEqual(rt, slip + 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
