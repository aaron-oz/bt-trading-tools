"""Tests for bt_trading_tools.execution — shared realism layer.

The same module is used by PaperBotBase.simulate_execution and (after Step 3)
BacktestEngine. These tests cover the layer logic in isolation; PaperBotBase's
tests in bt-strategy cover the integration.
"""

from __future__ import annotations

import os
import unittest

from bt_trading_tools.execution import RealismConfig, RealismSimulator


def _buy_action(**overrides) -> dict:
    """Minimal buy action — same shape paper bots produce."""
    base = {
        "type": "buy",
        "netuid": 1,
        "tao_spent": 1.0,
        "alpha_qty": 100.0,
        "price": 0.01,
        "decision_pool_tao": 1000.0,
        "decision_pool_alpha": 100000.0,
    }
    base.update(overrides)
    return base


def _sell_action(**overrides) -> dict:
    base = {
        "type": "sell",
        "netuid": 1,
        "alpha_qty": 100.0,
        "tao_received": 0.99,
        "exit_price": 0.0099,
    }
    base.update(overrides)
    return base


class TestRealismConfigDefaults(unittest.TestCase):
    """Defaults must match the calibrated values used in production paper bots."""

    def test_default_buy_failure_rate(self):
        self.assertEqual(RealismConfig().buy_failure_rate, 0.03)

    def test_default_settlement_delay_is_one_block(self):
        self.assertEqual(RealismConfig().settlement_delay_s, 12.0)

    def test_default_partial_fill_off(self):
        self.assertEqual(RealismConfig().partial_fill_rate, 0.0)

    def test_default_enabled(self):
        self.assertTrue(RealismConfig().enabled)


class TestRealismConfigFromEnv(unittest.TestCase):
    """from_env applies overrides to the right fields with the right types."""

    def setUp(self):
        # Snapshot env so we can clean up
        self._saved = {}
        for k in ("REALISM_ENABLED", "REALISM_BUY_FAILURE_RATE", "REALISM_BUY_LATENCY_MEAN_S"):
            self._saved[k] = os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_env_disables_realism(self):
        os.environ["REALISM_ENABLED"] = "false"
        cfg = RealismConfig.from_env()
        self.assertFalse(cfg.enabled)

    def test_env_overrides_failure_rate(self):
        os.environ["REALISM_BUY_FAILURE_RATE"] = "0.10"
        cfg = RealismConfig.from_env()
        self.assertAlmostEqual(cfg.buy_failure_rate, 0.10)

    def test_env_overrides_latency(self):
        os.environ["REALISM_BUY_LATENCY_MEAN_S"] = "30.0"
        cfg = RealismConfig.from_env()
        self.assertEqual(cfg.buy_latency_mean_s, 30.0)

    def test_invalid_env_value_logs_and_keeps_default(self):
        os.environ["REALISM_BUY_FAILURE_RATE"] = "not-a-number"
        cfg = RealismConfig.from_env()
        # Default preserved on bad input — never crash.
        self.assertEqual(cfg.buy_failure_rate, 0.03)


class TestSimulateFillDeterminism(unittest.TestCase):
    """Same seed → same outcomes."""

    def test_two_simulators_same_seed_match(self):
        cfg = RealismConfig(buy_failure_rate=0.30)
        sim_a = RealismSimulator(cfg, rng_seed=1234)
        sim_b = RealismSimulator(cfg, rng_seed=1234)
        outcomes_a = [sim_a.simulate_fill(_buy_action())["status"] for _ in range(100)]
        outcomes_b = [sim_b.simulate_fill(_buy_action())["status"] for _ in range(100)]
        self.assertEqual(outcomes_a, outcomes_b)

    def test_different_seed_different_outcomes(self):
        cfg = RealismConfig(buy_failure_rate=0.30)
        sim_a = RealismSimulator(cfg, rng_seed=1)
        sim_b = RealismSimulator(cfg, rng_seed=2)
        outcomes_a = [sim_a.simulate_fill(_buy_action())["status"] for _ in range(100)]
        outcomes_b = [sim_b.simulate_fill(_buy_action())["status"] for _ in range(100)]
        # Should differ at least somewhat
        self.assertNotEqual(outcomes_a, outcomes_b)


class TestSimulateFillLayer1Failure(unittest.TestCase):
    """Layer 1: Bernoulli random_reject."""

    def test_high_failure_rate_produces_failures(self):
        cfg = RealismConfig(buy_failure_rate=1.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "random_reject")

    def test_zero_failure_rate_never_fails_random(self):
        cfg = RealismConfig(buy_failure_rate=0.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        # Without any other failure source (large tolerance), should pass
        cfg.rate_tolerance_buy_pp = 100.0
        for _ in range(50):
            result = sim.simulate_fill(_buy_action())
            self.assertNotEqual(result.get("failure_reason"), "random_reject")


class TestSimulateFillLayer2SlippageNoise(unittest.TestCase):
    """Layer 2: CSV-only Gaussian noise; skipped when live_spot_price set."""

    def test_skipped_when_live_spot_price_present(self):
        cfg = RealismConfig(slippage_noise_mean_pct=10.0, slippage_noise_std_pct=0.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _buy_action(live_spot_price=0.01)
        original_price = action["price"]
        result = sim.simulate_fill(action)
        self.assertEqual(result["price"], original_price)
        self.assertNotIn("exec_slippage_pct", result)

    def test_applied_when_live_spot_price_absent(self):
        cfg = RealismConfig(
            slippage_noise_mean_pct=1.0,
            slippage_noise_std_pct=0.0,  # deterministic noise = +1%
            buy_failure_rate=0.0,
            rate_tolerance_buy_pp=100.0,  # don't fail on the noise
        )
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _buy_action()
        result = sim.simulate_fill(action)
        # Noise applied — exec_slippage_pct stamped
        self.assertIn("exec_slippage_pct", result)
        self.assertAlmostEqual(result["exec_slippage_pct"], 1.0, places=2)


class TestSimulateFillLayer3RateTolerance(unittest.TestCase):
    """Layer 3: rate-tolerance breach → status=failed."""

    def test_breach_fails_with_rate_tolerance_reason(self):
        cfg = RealismConfig(
            buy_failure_rate=0.0,
            slippage_noise_mean_pct=10.0,
            slippage_noise_std_pct=0.0,
            rate_tolerance_buy_pp=2.0,  # exec slip ≈ 10% > 2% breach
        )
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "rate_tolerance")

    def test_within_tolerance_passes(self):
        cfg = RealismConfig(
            buy_failure_rate=0.0,
            slippage_noise_mean_pct=0.5,
            slippage_noise_std_pct=0.0,
            rate_tolerance_buy_pp=10.0,  # 0.5% slip < 10% buffer
        )
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertEqual(result["status"], "executed")


class TestSimulateFillLayer4PartialFill(unittest.TestCase):
    """Layer 4: Bernoulli partial fill — defaults off."""

    def test_default_off(self):
        cfg = RealismConfig(buy_failure_rate=0.0, rate_tolerance_buy_pp=100.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertEqual(result["status"], "executed")  # not "partial"

    def test_high_rate_produces_partial_fills(self):
        cfg = RealismConfig(
            buy_failure_rate=0.0,
            rate_tolerance_buy_pp=100.0,
            partial_fill_rate=1.0,  # always partial
        )
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _buy_action()
        original_qty = action["alpha_qty"]
        result = sim.simulate_fill(action)
        self.assertEqual(result["status"], "partial")
        self.assertLess(result["alpha_qty"], original_qty)
        self.assertGreaterEqual(result["alpha_qty"], original_qty * 0.5)


class TestSimulateFillLayer5Latency(unittest.TestCase):
    """Layer 5: latency always drawn, even on failure."""

    def test_latency_stamped_on_executed(self):
        cfg = RealismConfig(buy_failure_rate=0.0, rate_tolerance_buy_pp=100.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertIn("latency_ms", result)
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_latency_stamped_on_failed(self):
        cfg = RealismConfig(buy_failure_rate=1.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        result = sim.simulate_fill(_buy_action())
        self.assertEqual(result["status"], "failed")
        self.assertIn("latency_ms", result)


class TestSimulateFillDisabled(unittest.TestCase):
    """enabled=False short-circuits all layers."""

    def test_disabled_short_circuits(self):
        cfg = RealismConfig(enabled=False)
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _buy_action()
        result = sim.simulate_fill(action)
        self.assertEqual(result["status"], "executed")
        self.assertNotIn("latency_ms", result)
        self.assertNotIn("intended_slippage_tolerance_pct", result)


class TestSimulateFillSell(unittest.TestCase):
    """Sell-side handling — slippage noise scales tao_received, not alpha_qty."""

    def test_sell_noise_reduces_tao_received(self):
        cfg = RealismConfig(
            sell_failure_rate=0.0,
            rate_tolerance_sell_pct=100.0,
            slippage_noise_mean_pct=1.0,
            slippage_noise_std_pct=0.0,
        )
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _sell_action()
        original_tao = action["tao_received"]
        result = sim.simulate_fill(action)
        # Worse fill: less TAO for the same alpha
        self.assertLess(result["tao_received"], original_tao)


class TestRequestedAmountsPreserved(unittest.TestCase):
    """Action dict must carry requested_*_amount even when fully executed,
    so downstream can compute partial-fill ratio if desired."""

    def test_buy_stamps_requested_tao_amount(self):
        cfg = RealismConfig(buy_failure_rate=0.0, rate_tolerance_buy_pp=100.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _buy_action()
        result = sim.simulate_fill(action)
        self.assertIn("requested_tao_amount", result)
        self.assertEqual(result["requested_tao_amount"], 1.0)

    def test_sell_stamps_requested_alpha_amount(self):
        cfg = RealismConfig(sell_failure_rate=0.0, rate_tolerance_sell_pct=100.0)
        sim = RealismSimulator(cfg, rng_seed=42)
        action = _sell_action()
        result = sim.simulate_fill(action)
        self.assertIn("requested_alpha_amount", result)
        self.assertEqual(result["requested_alpha_amount"], 100.0)


if __name__ == "__main__":
    unittest.main()
