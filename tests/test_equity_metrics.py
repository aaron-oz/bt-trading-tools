"""Tests for the shared equity_metrics (backtest and paper use the same one)."""

import math
import unittest

from bt_trading_tools.backtest import equity_metrics

DAY = 86400


class TestEquityMetrics(unittest.TestCase):
    def test_cadence_independent(self):
        # Same daily path sampled hourly vs once a day -> same metrics.
        daily = [100, 101, 99, 103, 102]
        pts_daily = [(d * DAY + 23 * 3600, v) for d, v in enumerate(daily)]
        pts_hourly = [(d * DAY + h * 3600, v) for d, v in enumerate(daily)
                      for h in range(24)]
        a, b = equity_metrics(pts_daily), equity_metrics(pts_hourly)
        self.assertAlmostEqual(a["sharpe"], b["sharpe"])
        self.assertAlmostEqual(a["max_drawdown_pct"], b["max_drawdown_pct"])
        self.assertAlmostEqual(a["total_return_pct"], b["total_return_pct"])

    def test_values(self):
        pts = [(d * DAY, v) for d, v in enumerate([100, 110, 99, 121])]
        m = equity_metrics(pts)
        self.assertAlmostEqual(m["total_return_pct"], 21.0)
        self.assertAlmostEqual(m["max_drawdown_pct"], 10.0)
        rets = [0.10, -0.10, 121 / 99 - 1]
        mean = sum(rets) / 3
        sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / 2)   # sample stdev
        self.assertAlmostEqual(m["sharpe"], mean / sd * math.sqrt(365))
        self.assertAlmostEqual(m["annualized_return_pct"], 21.0 * 365 / 3)

    def test_starting_equity_is_baseline(self):
        pts = [(0, 98.0), (DAY, 99.0)]
        m = equity_metrics(pts, starting_equity=100.0)
        self.assertAlmostEqual(m["total_return_pct"], -1.0)
        self.assertAlmostEqual(m["max_drawdown_pct"], 2.0)

    def test_drops_nonpositive_and_handles_short(self):
        self.assertIsNone(equity_metrics([(0, 100.0)])["sharpe"])
        m = equity_metrics([(0, 100.0), (DAY, 0.0), (2 * DAY, 105.0)])
        self.assertAlmostEqual(m["total_return_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()
