"""Tests for bt_trading_tools.ledger (shared position bookkeeping)."""

import unittest

import pandas as pd

from bt_trading_tools import ledger
from bt_trading_tools.amm import amm_sell
from bt_trading_tools.backtest.types import SubnetTick, TickData
from bt_trading_tools.utils.lifecycle import (
    find_reregistrations,
    reregistrations_from_pool_history,
    stamp_generations,
)


def flat_yield(rate_per_day):
    """Yield fn: alpha_qty * rate * days since anchor."""
    def fn(netuid, alpha_qty, since, now):
        return alpha_qty * rate_per_day * max(0.0, now - since) / 86400.0
    return fn


class TestBuy(unittest.TestCase):
    def test_open_then_top_up_is_cost_weighted(self):
        pos = {}
        ledger.book_buy(pos, 7, alpha_received=100, tao_spent=1.0, fee=0.01, now=0)
        ledger.book_buy(pos, 7, alpha_received=50, tao_spent=1.0, fee=0.01, now=10)
        p = pos[7]
        self.assertAlmostEqual(p.alpha_qty, 150)
        self.assertAlmostEqual(p.tao_cost, 2.0)
        self.assertAlmostEqual(p.entry_price, 2.0 / 150)
        self.assertEqual(p.entry_time, 0)          # first fill kept
        self.assertAlmostEqual(p.entry_fees, 0.02)

    def test_top_up_does_not_give_new_lot_back_yield(self):
        # 10%/day. Lot A: 100 alpha at t=0. Lot B: 100 alpha at day 1.
        # At day 2 accrued should be A: 100*0.1*1 folded at day 1 = 10,
        # then (110 + 100) * 0.1 * 1 = 21. Old behavior (anchor at first
        # entry) would give 200 * 0.1 * 2 = 40.
        y = flat_yield(0.10)
        pos = {}
        ledger.book_buy(pos, 1, 100, 1.0, 0.0, now=0, yield_fn=y)
        ledger.book_buy(pos, 1, 100, 1.0, 0.0, now=86400, yield_fn=y)
        self.assertAlmostEqual(pos[1].alpha_qty, 210)
        self.assertAlmostEqual(ledger.accrued_yield(pos[1], 2 * 86400, y), 21.0)

    def test_generation_mismatch_raises(self):
        pos = {}
        ledger.book_buy(pos, 3, 10, 1.0, 0.0, now=0, generation=0)
        with self.assertRaises(ledger.GenerationMismatch):
            ledger.book_buy(pos, 3, 10, 1.0, 0.0, now=1, generation=1)


class TestSell(unittest.TestCase):
    def test_partial_sell_keeps_remainder(self):
        # Regression for the engine bug where a capped/partial close booked
        # the full cost and deleted the whole position.
        pos = {}
        ledger.book_buy(pos, 5, 1000, 10.0, 0.0, now=0)
        rec = ledger.book_sell(pos, 5, alpha_sold=250, tao_received=3.0, fee=0.0, now=1)
        self.assertFalse(rec["position_closed"])
        self.assertAlmostEqual(rec["tao_cost"], 2.5)
        self.assertAlmostEqual(rec["pnl"], 0.5)
        self.assertAlmostEqual(pos[5].alpha_qty, 750)
        self.assertAlmostEqual(pos[5].tao_cost, 7.5)

    def test_full_sell_closes(self):
        pos = {}
        ledger.book_buy(pos, 5, 1000, 10.0, 0.0, now=0)
        rec = ledger.book_sell(pos, 5, alpha_sold=1000, tao_received=9.0, fee=0.0, now=1)
        self.assertTrue(rec["position_closed"])
        self.assertNotIn(5, pos)
        self.assertAlmostEqual(rec["pnl"], -1.0)

    def test_sell_includes_accrued_yield_at_zero_cost(self):
        y = flat_yield(0.01)
        pos = {}
        ledger.book_buy(pos, 2, 100, 1.0, 0.0, now=0, yield_fn=y)
        rec = ledger.book_sell(pos, 2, alpha_sold=101, tao_received=1.2, fee=0.0,
                               now=86400, yield_fn=y)
        self.assertAlmostEqual(rec["alpha_yield_accrued"], 1.0)
        self.assertTrue(rec["position_closed"])
        self.assertAlmostEqual(rec["tao_cost"], 1.0)


class TestValuation(unittest.TestCase):
    def test_liquidation_value_is_amm_sale_minus_fee(self):
        tao_out, _, _ = amm_sell(100, 1000, 100000)
        self.assertAlmostEqual(ledger.liquidation_value(100, 1000, 100000, 0.01),
                               tao_out - 0.01)

    def test_liquidation_value_floored_and_guarded(self):
        self.assertEqual(ledger.liquidation_value(1e-9, 1000, 1e6, 1.0), 0.0)
        self.assertEqual(ledger.liquidation_value(10, 0, 100, 0.0), 0.0)

    def test_dereg_refund_is_spot_times_alpha(self):
        self.assertAlmostEqual(ledger.dereg_refund(500, 200, 100000), 1.0)

    def test_close_deregistered_stops_yield_at_last_seen(self):
        y = flat_yield(0.10)
        pos = {}
        ledger.book_buy(pos, 9, 100, 1.0, 0.0, now=0, generation=0, yield_fn=y)
        rec = ledger.close_deregistered(pos, 9, last_tao_pool=10, last_alpha_pool=1000,
                                        last_seen_ts=86400, yield_fn=y)
        self.assertNotIn(9, pos)
        self.assertAlmostEqual(rec["alpha_qty"], 110)
        self.assertAlmostEqual(rec["tao_received"], 110 * 0.01)
        self.assertEqual(rec["generation"], 0)


class TestLifecycle(unittest.TestCase):
    def test_detects_collapse_and_ignores_mass_unstake(self):
        ts = pd.date_range("2026-05-01", periods=4, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": list(ts) * 2,
            "netuid": [1] * 4 + [2] * 4,
            # netuid 1: re-registration at step 2 (both sides collapse)
            # netuid 2: mass unstake at step 2 (staked collapses, pool RISES)
            "alpha_in": [1e6, 1e6, 7e4, 7e4, 1e6, 1e6, 1.9e6, 1.9e6],
            "alpha_out": [1.5e6, 1.5e6, 5e4, 5e4, 1e6, 1e6, 1e5, 1e5],
        })
        ev = find_reregistrations(df)
        self.assertEqual(list(ev), [1])
        self.assertEqual(ev[1][0], ts[2])

    def test_startup_flip_counts_once(self):
        days = pd.date_range("2026-08-18", periods=6, freq="D", tz="UTC")
        rows = []
        for i, d in enumerate(days):
            for h in (0, 6, 12):          # several rows per day, as taostats has
                rows.append({"netuid": 36, "timestamp": d + pd.Timedelta(hours=h),
                             "alpha_in_pool": 5e5 if i < 2 else 9e4,
                             "alpha_staked": 5e5 if i < 2 else 0,
                             "startup_mode": i >= 2})
        ev = reregistrations_from_pool_history(pd.DataFrame(rows))
        self.assertEqual(len(ev[36]), 1)
        self.assertEqual(ev[36][0].normalize(), days[2])

    def test_stamp_generations(self):
        t0 = int(pd.Timestamp("2026-05-01", tz="UTC").timestamp())
        ticks = [TickData(timestamp=t0 + k * 86400,
                          subnets={4: SubnetTick(4, 1.0, 10, 10),
                                   5: SubnetTick(5, 1.0, 10, 10)})
                 for k in range(3)]
        stamp_generations(ticks, {4: [pd.Timestamp("2026-05-02 08:00", tz="UTC")]},
                          floor_to_day=True)
        self.assertEqual([t.subnets[4].generation for t in ticks], [0, 1, 1])
        self.assertEqual([t.subnets[5].generation for t in ticks], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
