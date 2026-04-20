"""Tests for the v1 trade log schema, writer, reader, metrics, and P&L."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from bt_trading_tools.tracking import (
    SCHEMA_VERSION,
    FeeReceiptWriter,
    TradeLogWriter,
    compute_pnl,
    drawdown_series,
    equity_series,
    iter_fee_receipts,
    iter_trade_log,
    load_trade_log,
    max_drawdown,
    sharpe,
    validate_fee_receipt_log,
    validate_trade_log,
)


def _make_writer(tmpdir: str, bot_id: str = "testbot") -> TradeLogWriter:
    path = Path(tmpdir) / "tl.jsonl"
    return TradeLogWriter(path, bot_id=bot_id, flush_interval_s=0.05,
                          auto_close_on_exit=False)


class TestTradeLogWriter(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove files, tolerate if worker still draining
        for f in Path(self.tmpdir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def test_roundtrip_trade(self):
        w = _make_writer(self.tmpdir)
        w.log_trade({
            "netuid": 107, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "buy", "status": "executed", "category": "entry",
            "intent": "t_entry", "position_id": "sn107",
            "tao_amount": 0.5, "alpha_amount": 100.0,
            "requested_tao_amount": 0.5, "executed_price": 0.005,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
        })
        w.close()
        time.sleep(0.1)

        path = Path(self.tmpdir) / "tl.jsonl"
        records = list(iter_trade_log(path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_type"], "trade")
        self.assertEqual(records[0]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(records[0]["bot_id"], "testbot")

        self.assertEqual(validate_trade_log(path), [])

    def test_validation_failure_routes_to_errors_file(self):
        w = _make_writer(self.tmpdir)
        # requested_tao_amount missing for a buy → schema violation
        w.log_trade({
            "netuid": 107, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "buy", "status": "executed", "category": "entry",
            "intent": "bad", "position_id": "sn107",
            "tao_amount": 0.5, "alpha_amount": 100.0,
            "executed_price": 0.005,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
        })
        w.close()
        time.sleep(0.1)

        errs = Path(self.tmpdir) / "tl.errors.jsonl"
        self.assertTrue(errs.exists())
        err_rec = json.loads(errs.read_text().strip().splitlines()[0])
        self.assertIn("record", err_rec)
        self.assertIn("error", err_rec)
        self.assertIn("requested_tao_amount", err_rec["error"])
        self.assertGreaterEqual(w.flush_errors_since_start, 1)

    def test_record_types_coexist_in_same_file(self):
        w = _make_writer(self.tmpdir)
        w.log_trade({
            "netuid": 107, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "buy", "status": "executed", "category": "entry",
            "intent": "entry", "position_id": "sn107",
            "tao_amount": 0.5, "alpha_amount": 100.0,
            "requested_tao_amount": 0.5, "executed_price": 0.005,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
        })
        w.log_mtm({
            "netuid": 107, "is_paper": True,
            "pool_tao": 10500.0, "pool_alpha": 1900000.0,
            "position_id": "sn107",
        })
        w.log_portfolio_snapshot({
            "is_paper": True,
            "capital_tao": 99.5, "positions_value_tao": 0.52,
            "total_equity_tao": 100.02, "realized_pnl_to_date_tao": 0.02,
            "open_positions_count": 1,
        })
        w.close()
        time.sleep(0.1)

        df = load_trade_log(Path(self.tmpdir) / "tl.jsonl")
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["record_type"]),
                         {"trade", "mtm_sample", "portfolio_snapshot"})


class TestPnL(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "tl.jsonl"
        self.w = _make_writer(self.tmpdir)

    def tearDown(self):
        try:
            self.w.close()
        except Exception:
            pass
        for f in Path(self.tmpdir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def _buy(self, tao, alpha, price):
        self.w.log_trade({
            "netuid": 10, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "buy", "status": "executed", "category": "entry",
            "intent": "entry", "position_id": "sn10",
            "tao_amount": tao, "alpha_amount": alpha,
            "requested_tao_amount": tao, "executed_price": price,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
        })

    def _sell(self, tao, alpha, price):
        self.w.log_trade({
            "netuid": 10, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "sell", "status": "executed", "category": "exit",
            "intent": "exit", "position_id": "sn10",
            "tao_amount": tao, "alpha_amount": alpha,
            "requested_alpha_amount": alpha, "executed_price": price,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
        })

    def test_fifo_roundtrip(self):
        self._buy(1.0, 100.0, 0.01)
        self._buy(2.0, 100.0, 0.02)
        self._sell(1.5, 100.0, 0.015)
        self.w.close()
        time.sleep(0.1)

        results = compute_pnl(self.path, position_model="inventory", basis="fifo")
        pnl = results["sn10"]
        # FIFO: first 100 at 0.01 → cost basis 1.0, sold for 1.5 → realized 0.5
        self.assertAlmostEqual(pnl.realized_tao, 0.5, places=4)
        self.assertAlmostEqual(pnl.alpha_held, 100.0, places=6)
        self.assertTrue(pnl.is_open)

    def test_avg_cost(self):
        self._buy(1.0, 100.0, 0.01)
        self._buy(2.0, 100.0, 0.02)
        self._sell(1.5, 100.0, 0.015)
        self.w.close()
        time.sleep(0.1)

        results = compute_pnl(self.path, position_model="inventory", basis="avg_cost")
        pnl = results["sn10"]
        # Avg cost: (1+2)/200 = 0.015, sold 100 at avg cost 1.5; received 1.5 → realized 0.0
        self.assertAlmostEqual(pnl.realized_tao, 0.0, places=4)

    def test_yield_accrued_has_zero_cost_basis(self):
        # Buy 100 alpha for 1.0 TAO (basis 0.01/alpha).
        # Sell 102 alpha for 1.5 TAO, with 2 alpha declared as yield-accrued.
        # Expected: 100 alpha against cost basis 1.0, yielded 2 alpha at zero basis.
        # Realized P&L = 1.5 (received) - 1.0 (basis of traded) = 0.5 TAO.
        self._buy(1.0, 100.0, 0.01)
        self.w.log_trade({
            "netuid": 10, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "sell", "status": "executed", "category": "take_profit",
            "intent": "exit", "position_id": "sn10",
            "tao_amount": 1.5, "alpha_amount": 102.0,
            "requested_alpha_amount": 102.0, "executed_price": 0.014705882,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
            "alpha_yield_accrued": 2.0,
        })
        self.w.close()
        time.sleep(0.1)

        results = compute_pnl(self.path, position_model="inventory", basis="fifo")
        pnl = results["sn10"]
        self.assertAlmostEqual(pnl.realized_tao, 0.5, places=4)
        self.assertFalse(pnl.is_open)  # 100 alpha matched, 2 accrued → position closed

    def test_fees_use_atomic_components(self):
        # Verify compute_pnl sums fees from the three atomic fields, not legacy network_fee_tao.
        self.w.log_trade({
            "netuid": 10, "is_paper": True,
            "pool_tao": 10000.0, "pool_alpha": 2000000.0,
            "side": "buy", "status": "executed", "category": "entry",
            "intent": "entry", "position_id": "sn10",
            "tao_amount": 1.0, "alpha_amount": 100.0,
            "requested_tao_amount": 1.0, "executed_price": 0.01,
            "decision_pool_tao": 10000.0, "decision_pool_alpha": 2000000.0,
            "swap_fee_tao": 0.0005, "gas_fee_tao": 8.4e-6, "proxy_fee_tao": 1e-6,
        })
        self._sell(1.5, 100.0, 0.015)  # no explicit fees → 0
        self.w.close()
        time.sleep(0.1)

        results = compute_pnl(self.path, position_model="inventory", basis="fifo")
        pnl = results["sn10"]
        expected_fees = 0.0005 + 8.4e-6 + 1e-6
        self.assertAlmostEqual(pnl.tao_fees, expected_fees, places=8)


class TestFeeReceipt(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "fr.jsonl"
        self.w = FeeReceiptWriter(self.path, bot_id="bagbot",
                                  flush_interval_s=0.05, auto_close_on_exit=False)

    def tearDown(self):
        try:
            self.w.close()
        except Exception:
            pass
        for f in Path(self.tmpdir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def test_valid_receipt_roundtrip(self):
        self.w.log_receipt({
            "wallet_coldkey": "5Gproxy", "wallet_hotkey": "5Ghk",
            "extrinsic_type": "proxy.proxy(add_stake)",
            "netuid": 23, "tao_amount_requested": 0.5,
            "rate_tolerance": 0.034,
            "extrinsic_status": "success",
            "chain_tx_hash": "0xabc",
            "observed_swap_fee_tao": 0.000252,
            "observed_gas_fee_tao": 8.4e-6,
            "observed_proxy_fee_tao": 9e-7,
        })
        self.w.close()
        time.sleep(0.1)

        self.assertEqual(validate_fee_receipt_log(self.path), [])
        recs = list(iter_fee_receipts(self.path))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["chain_tx_hash"], "0xabc")

    def test_timeout_status_valid(self):
        self.w.log_receipt({
            "extrinsic_type": "proxy.proxy(remove_stake)",
            "netuid": 23,
            "extrinsic_status": "timeout",
            "parse_error": "TimeoutError after 45s",
        })
        self.w.close()
        time.sleep(0.1)
        self.assertEqual(validate_fee_receipt_log(self.path), [])

    def test_bad_status_rejected(self):
        self.w.log_receipt({
            "extrinsic_type": "add_stake",
            "netuid": 23,
            "extrinsic_status": "who_knows",
        })
        self.w.close()
        time.sleep(0.1)
        self.assertEqual(self.w.flush_errors_since_start, 1)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "tl.jsonl"
        self.w = _make_writer(self.tmpdir)

    def tearDown(self):
        try:
            self.w.close()
        except Exception:
            pass
        for f in Path(self.tmpdir).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def _snap(self, eq):
        self.w.log_portfolio_snapshot({
            "is_paper": True,
            "capital_tao": eq * 0.3,
            "positions_value_tao": eq * 0.7,
            "total_equity_tao": eq,
            "realized_pnl_to_date_tao": eq - 100.0,
            "open_positions_count": 2,
        })

    def test_equity_and_drawdown(self):
        for eq in [100.0, 110.0, 105.0, 108.0, 95.0, 100.0]:
            self._snap(eq)
            time.sleep(0.002)
        self.w.close()
        time.sleep(0.1)

        eq = equity_series(self.path)
        self.assertEqual(len(eq), 6)

        dd = drawdown_series(self.path)
        # Peak 110, trough 95, drawdown = -15 (-13.64%)
        self.assertAlmostEqual(dd["drawdown_tao"].min(), -15.0, places=4)

        mdd = max_drawdown(self.path)
        self.assertAlmostEqual(mdd["max_drawdown_tao"], -15.0, places=4)
        self.assertLess(mdd["max_drawdown_pct"], 0.0)

        s = sharpe(self.path, periods_per_year=252)
        self.assertIsNotNone(s)


if __name__ == "__main__":
    unittest.main()
