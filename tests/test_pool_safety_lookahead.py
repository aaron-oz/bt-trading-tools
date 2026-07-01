"""Regression tests for the backtest pool-safety lookahead fix.

Bug (confirmed 2026-06-25): BacktestEngine called
``pool_safety_checker.check(netuid)`` without ``as_of``. The checker then
defaulted its reference date to the latest row in the entire pool_history
(``df.index.max()``), so every simulated tick's safety verdict reflected
end-of-dataset pool state. Appending future data silently changed historical
results (a fixed window's Sharpe drifted as the dataset grew).

Fix: the engine now passes ``as_of = <current tick timestamp>`` so safety is
evaluated point-in-time. These tests lock that contract using lightweight stub
checkers (no bt-strategy dependency, matching this package's test isolation).
"""

import unittest
from datetime import datetime, timezone

from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, SubnetTick, TickData


# ── Stub checkers ────────────────────────────────────────────────────────────

class _Flags:
    def __init__(self, unsafe: bool):
        self._unsafe = unsafe

    @property
    def any_unsafe(self) -> bool:
        return self._unsafe


class _RecordingChecker:
    """Records the as_of passed to check(); never flags anything unsafe."""

    def __init__(self):
        self.as_of_calls = []

    def refresh(self):
        pass

    def check(self, netuid, *, as_of=None):
        self.as_of_calls.append(as_of)
        return None


class _CrashChecker:
    """Flags one netuid unsafe once as_of reaches a crash time.

    Faithfully emulates the real point-in-time contract. With as_of=None it
    returns unsafe whenever the crash exists at all, emulating the buggy
    end-of-dataset behavior, so a test that relies on as_of being threaded
    will fail loudly if the engine ever regresses to passing None.
    """

    def __init__(self, netuid: int, crash_ts: int):
        self.netuid = netuid
        self.crash_dt = datetime.fromtimestamp(crash_ts, tz=timezone.utc)

    def refresh(self):
        pass

    def check(self, netuid, *, as_of=None):
        if netuid != self.netuid:
            return _Flags(False)
        if as_of is None:
            return _Flags(True)          # buggy path: sees crash regardless of date
        return _Flags(as_of >= self.crash_dt)


# ── Strategies ───────────────────────────────────────────────────────────────

class _BuyFirstTick:
    def __init__(self, netuid, spend):
        self.netuid, self.spend, self.done = netuid, spend, False

    def on_tick(self, tick, positions, capital, pv):
        if not self.done and capital >= self.spend:
            self.done = True
            return [Order(self.netuid, "buy", tao_amount=self.spend, reason="buy")]
        return []


class _BuyEveryTick:
    def __init__(self, netuid, spend):
        self.netuid, self.spend = netuid, spend

    def on_tick(self, tick, positions, capital, pv):
        if capital >= self.spend:
            return [Order(self.netuid, "buy", tao_amount=self.spend, reason="buy")]
        return []


# ── Helpers ──────────────────────────────────────────────────────────────────

NETUID = 107
START = 1_700_000_000
DAY = 86_400


def _ticks(n, price=0.01, tao_pool=5000.0, alpha_pool=500_000.0):
    out = []
    for i in range(n):
        out.append(TickData(
            timestamp=START + i * DAY,
            subnets={NETUID: SubnetTick(netuid=NETUID, price=price,
                                        tao_pool=tao_pool, alpha_pool=alpha_pool)},
        ))
    return out


def _failed_pool_safety(trades):
    return [t for t in trades
            if t.get("status") == "failed" and t.get("failure_reason") == "pool_safety"]


def _executed(trades):
    return [t for t in trades if t.get("status") != "failed"]


# ── Tests ────────────────────────────────────────────────────────────────────

class TestPoolSafetyLookahead(unittest.TestCase):

    def test_engine_passes_tick_timestamp_as_of(self):
        """The engine must call check() with as_of == the current tick's
        timestamp (tz-aware UTC), never None."""
        ticks = _ticks(3)
        rec = _RecordingChecker()
        engine = BacktestEngine(capital=10_000.0, pool_safety_checker=rec)
        engine.run(ticks, _BuyEveryTick(NETUID, spend=1.0))

        expected = [datetime.fromtimestamp(t.timestamp, tz=timezone.utc) for t in ticks]
        self.assertEqual(rec.as_of_calls, expected)
        self.assertTrue(all(a is not None for a in rec.as_of_calls))

    def test_future_crash_does_not_block_earlier_buys(self):
        """A pool crash AFTER the backtest window must not block buys during
        the window (no lookahead). Fails if the engine reverts to as_of=None."""
        ticks = _ticks(3)
        crash_ts = ticks[-1].timestamp + 10 * DAY   # strictly after the window
        checker = _CrashChecker(NETUID, crash_ts)
        engine = BacktestEngine(capital=100.0, pool_safety_checker=checker)
        results = engine.run(ticks, _BuyFirstTick(NETUID, spend=5.0))

        self.assertEqual(_failed_pool_safety(results.trades), [],
                         "future crash leaked into the window (lookahead)")
        self.assertGreaterEqual(len(_executed(results.trades)), 1,
                                "the in-window buy should have executed")
        # Sanity: the buggy as_of=None path WOULD have flagged it unsafe.
        self.assertTrue(checker.check(NETUID, as_of=None).any_unsafe)

    def test_crash_blocks_point_in_time_from_crash_onward(self):
        """With a crash at tick 1, the tick-0 buy executes but tick-1/2 buys
        are blocked — proving point-in-time evaluation, not all-or-nothing."""
        ticks = _ticks(3)
        crash_ts = ticks[1].timestamp            # crash at the 2nd tick
        checker = _CrashChecker(NETUID, crash_ts)
        engine = BacktestEngine(capital=10_000.0, pool_safety_checker=checker)
        results = engine.run(ticks, _BuyEveryTick(NETUID, spend=1.0))

        # tick 0 buy allowed; tick 1 and tick 2 buys blocked.
        self.assertEqual(len(_failed_pool_safety(results.trades)), 2)
        self.assertGreaterEqual(len(_executed(results.trades)), 1)


if __name__ == "__main__":
    unittest.main()
