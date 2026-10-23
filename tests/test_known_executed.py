"""Order.known_executed: the realism layer's random rejection is not applied
to replays of trades known to have executed (2026-10-22)."""

from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig, RealismSimulator

DAY = 86400


def _ticks(n):
    return [TickData(timestamp=i * 900, subnets={7: SubnetTick(7, 0.001, 1000.0, 1_000_000.0)})
            for i in range(n)]


class _Buys:
    def __init__(self, known):
        self.known = known

    def on_tick(self, tick, positions, capital, pv):
        return [Order(netuid=7, side="buy", tao_amount=0.1, known_executed=self.known)]


def _run(known, rate, seed=3, n=200):
    eng = BacktestEngine(capital=100.0, yield_model=None, realism_rng_seed=seed,
                         realism_config=RealismConfig(buy_failure_rate=rate))
    res = eng.run(_ticks(n), _Buys(known))
    failed = [t for t in res.trades if t.get("status") == "failed"]
    return res, failed


def test_default_still_rejects():
    _, failed = _run(False, 0.5)
    assert len(failed) > 50
    assert all(t.get("failure_reason") == "random_reject" for t in failed)


def test_known_executed_never_random_rejected():
    _, failed = _run(True, 0.5)
    assert not [t for t in failed if t.get("failure_reason") == "random_reject"]


def test_flag_equals_zero_failure_rate_draw_for_draw():
    """Same seed: known_executed at rate 0.5 gives exactly the fills of rate 0."""
    sim_a = RealismSimulator(RealismConfig(buy_failure_rate=0.0), rng_seed=11)
    sim_b = RealismSimulator(RealismConfig(buy_failure_rate=0.5), rng_seed=11)
    base = {"type": "buy", "tao_spent": 1.0, "alpha_qty": 1000.0, "price": 0.001,
            "decision_pool_tao": 1000.0, "decision_pool_alpha": 1e6}
    for _ in range(200):
        a = sim_a.simulate_fill(dict(base))
        b = sim_b.simulate_fill(dict(base, known_executed=True))
        assert (a.get("status"), a["alpha_qty"], a["latency_ms"]) == \
            (b.get("status"), b["alpha_qty"], b["latency_ms"])
