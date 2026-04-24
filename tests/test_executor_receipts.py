"""Tests for TradeExecutor fee-receipt parsing.

Covers the parser helpers against the real bittensor SDK
``ExtrinsicResponse`` shape (verified against bittensor 9.x source —
see docs/data-sources.md for pointers):

    ExtrinsicResponse:
        success: bool
        extrinsic_fee: Balance             (extrinsic weight+length fee)
        extrinsic_receipt: ExtrinsicReceipt (.extrinsic_hash)
        transaction_tao_fee: Balance        (swap fee for add_stake)
        transaction_alpha_fee: Balance      (swap fee for remove_stake)
        error: Exception | None

Also covers legacy shapes (raw substrate dicts, bool returns, unknown
objects) to confirm the parser degrades to None without raising.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from bt_trading_tools.network.executor import (
    TradeExecutor,
    TradeResult,
    _as_fee_alpha,
    _as_fee_tao,
    _coldkey_for_receipt,
    _extract_block_hash,
    _extract_block_number,
    _extract_partial_fee_tao,
    _extract_swap_fee_alpha,
    _extract_swap_fee_tao,
    _extract_tx_hash,
    _safe_raw_dump,
    _status_from_tr,
)


# ── Test doubles mimicking bittensor's real shapes ────────────────

class _FakeBalance:
    """Mimics bittensor.Balance — exposes .tao (TAO float) and .rao (int).

    The SDK's Balance stores as rao internally; .tao is computed.
    """
    def __init__(self, rao: int):
        self.rao = rao

    @property
    def tao(self) -> float:
        return self.rao / 1e9


class _FakeReceipt:
    """Mimics async_substrate_interface.AsyncExtrinsicReceipt."""
    def __init__(
        self,
        extrinsic_hash: str,
        block_hash: str | None = None,
        block_number: int | None = None,
    ):
        self.extrinsic_hash = extrinsic_hash
        self.block_hash = block_hash
        self.block_number = block_number


class _FakeExtrinsicResponse:
    """Mimics bittensor.core.types.ExtrinsicResponse."""
    def __init__(
        self,
        success: bool = True,
        extrinsic_fee: _FakeBalance | None = None,
        extrinsic_receipt: _FakeReceipt | None = None,
        transaction_tao_fee: _FakeBalance | None = None,
        transaction_alpha_fee: _FakeBalance | None = None,
        error: Exception | None = None,
    ):
        self.success = success
        self.extrinsic_fee = extrinsic_fee
        self.extrinsic_receipt = extrinsic_receipt
        self.transaction_tao_fee = transaction_tao_fee
        self.transaction_alpha_fee = transaction_alpha_fee
        self.error = error


# ── _extract_tx_hash ───────────────────────────────────────────────

class TestExtractTxHash(unittest.TestCase):

    def test_sdk_response_with_receipt(self):
        r = _FakeExtrinsicResponse(
            extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xabc123"),
        )
        self.assertEqual(_extract_tx_hash(r), "0xabc123")

    def test_sdk_response_no_receipt(self):
        r = _FakeExtrinsicResponse(extrinsic_receipt=None)
        self.assertIsNone(_extract_tx_hash(r))

    def test_direct_attr_fallback(self):
        obj = MagicMock(spec=["extrinsic_hash"])
        obj.extrinsic_hash = "0xdeadbeef"
        self.assertEqual(_extract_tx_hash(obj), "0xdeadbeef")

    def test_dict_shape(self):
        self.assertEqual(
            _extract_tx_hash({"extrinsic_hash": "0xfeed"}),
            "0xfeed",
        )

    def test_boolean_result_returns_none(self):
        self.assertIsNone(_extract_tx_hash(True))
        self.assertIsNone(_extract_tx_hash(False))

    def test_none_returns_none(self):
        self.assertIsNone(_extract_tx_hash(None))


# ── _extract_block_hash / _extract_block_number (v2) ───────────────

class TestExtractBlockHash(unittest.TestCase):

    def test_sdk_response_with_receipt_block_hash(self):
        r = _FakeExtrinsicResponse(
            extrinsic_receipt=_FakeReceipt(
                extrinsic_hash="0xabc", block_hash="0xblock123"
            ),
        )
        self.assertEqual(_extract_block_hash(r), "0xblock123")

    def test_sdk_response_receipt_no_block_hash(self):
        r = _FakeExtrinsicResponse(
            extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xabc"),
        )
        self.assertIsNone(_extract_block_hash(r))

    def test_sdk_response_no_receipt(self):
        r = _FakeExtrinsicResponse(extrinsic_receipt=None)
        self.assertIsNone(_extract_block_hash(r))

    def test_dict_shape(self):
        self.assertEqual(
            _extract_block_hash({"block_hash": "0xdict_block"}),
            "0xdict_block",
        )

    def test_boolean_and_none_safe(self):
        self.assertIsNone(_extract_block_hash(True))
        self.assertIsNone(_extract_block_hash(False))
        self.assertIsNone(_extract_block_hash(None))


class TestExtractBlockNumber(unittest.TestCase):

    def test_sdk_response_with_receipt_block_number(self):
        r = _FakeExtrinsicResponse(
            extrinsic_receipt=_FakeReceipt(
                extrinsic_hash="0xabc", block_number=4_200_000
            ),
        )
        self.assertEqual(_extract_block_number(r), 4_200_000)

    def test_sdk_response_receipt_no_block_number(self):
        r = _FakeExtrinsicResponse(
            extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xabc"),
        )
        self.assertIsNone(_extract_block_number(r))

    def test_dict_shape_with_int(self):
        self.assertEqual(
            _extract_block_number({"block_number": 1234}),
            1234,
        )

    def test_dict_shape_with_string_coerces(self):
        self.assertEqual(
            _extract_block_number({"block_number": "1234"}),
            1234,
        )

    def test_dict_shape_with_negative_returns_none(self):
        self.assertIsNone(_extract_block_number({"block_number": -1}))

    def test_dict_shape_with_garbage_returns_none(self):
        self.assertIsNone(_extract_block_number({"block_number": "not-a-number"}))

    def test_boolean_and_none_safe(self):
        self.assertIsNone(_extract_block_number(True))
        self.assertIsNone(_extract_block_number(False))
        self.assertIsNone(_extract_block_number(None))


# ── _extract_partial_fee_tao (gas/extrinsic fee) ───────────────────

class TestExtractPartialFee(unittest.TestCase):

    def test_sdk_extrinsic_fee_balance(self):
        """Primary path: SDK ExtrinsicResponse.extrinsic_fee."""
        r = _FakeExtrinsicResponse(
            extrinsic_fee=_FakeBalance(rao=8_400),  # 8.4e-6 TAO
        )
        fee = _extract_partial_fee_tao(r)
        self.assertIsNotNone(fee)
        self.assertAlmostEqual(fee, 8.4e-6, places=10)

    def test_sdk_extrinsic_fee_none(self):
        r = _FakeExtrinsicResponse(extrinsic_fee=None)
        self.assertIsNone(_extract_partial_fee_tao(r))

    def test_raw_substrate_dict(self):
        """Legacy path: raw substrate payment_info response."""
        self.assertAlmostEqual(
            _extract_partial_fee_tao({"partial_fee": 8_400}),
            8.4e-6, places=10,
        )

    def test_camelcase_variant(self):
        self.assertAlmostEqual(
            _extract_partial_fee_tao({"partialFee": 10_000}),
            1e-5, places=10,
        )

    def test_object_with_partial_fee_attr(self):
        obj = MagicMock(spec=["partial_fee", "extrinsic_fee"])
        obj.extrinsic_fee = None
        obj.partial_fee = 8_400
        self.assertAlmostEqual(
            _extract_partial_fee_tao(obj),
            8.4e-6, places=10,
        )

    def test_boolean_result(self):
        self.assertIsNone(_extract_partial_fee_tao(True))
        self.assertIsNone(_extract_partial_fee_tao(False))

    def test_unknown_shape(self):
        self.assertIsNone(_extract_partial_fee_tao("just a string"))


# ── _extract_swap_fee_tao / _extract_swap_fee_alpha ────────────────

class TestExtractSwapFees(unittest.TestCase):

    def test_add_stake_tao_fee(self):
        """Buys: swap fee comes in transaction_tao_fee."""
        r = _FakeExtrinsicResponse(
            transaction_tao_fee=_FakeBalance(rao=252_000),  # 2.52e-4 TAO
            transaction_alpha_fee=None,
        )
        self.assertAlmostEqual(
            _extract_swap_fee_tao(r), 2.52e-4, places=10,
        )
        self.assertIsNone(_extract_swap_fee_alpha(r))

    def test_remove_stake_alpha_fee(self):
        """Sells: swap fee comes in transaction_alpha_fee."""
        r = _FakeExtrinsicResponse(
            transaction_tao_fee=None,
            transaction_alpha_fee=_FakeBalance(rao=500_000_000),  # 0.5 alpha
        )
        self.assertIsNone(_extract_swap_fee_tao(r))
        self.assertAlmostEqual(
            _extract_swap_fee_alpha(r), 0.5, places=10,
        )

    def test_neither_set(self):
        r = _FakeExtrinsicResponse()
        self.assertIsNone(_extract_swap_fee_tao(r))
        self.assertIsNone(_extract_swap_fee_alpha(r))

    def test_dict_shape(self):
        self.assertAlmostEqual(
            _extract_swap_fee_tao({"transaction_tao_fee": _FakeBalance(rao=252_000)}),
            2.52e-4, places=10,
        )

    def test_boolean_result(self):
        self.assertIsNone(_extract_swap_fee_tao(True))
        self.assertIsNone(_extract_swap_fee_alpha(False))


# ── Numeric coercion helpers ──────────────────────────────────────

class TestFeeCoercion(unittest.TestCase):

    def test_balance_via_tao(self):
        self.assertAlmostEqual(_as_fee_tao(_FakeBalance(rao=10_000)), 1e-5, places=10)

    def test_plain_float_small(self):
        # < 1000 interpreted as TAO already
        self.assertAlmostEqual(_as_fee_tao(0.0001), 0.0001, places=10)

    def test_plain_int_large_as_rao(self):
        # > 1000 heuristically rao
        self.assertAlmostEqual(_as_fee_tao(10_000), 1e-5, places=10)

    def test_dict_with_partial_fee(self):
        self.assertAlmostEqual(_as_fee_tao({"partial_fee": 8_400}), 8.4e-6, places=10)

    def test_none(self):
        self.assertIsNone(_as_fee_tao(None))
        self.assertIsNone(_as_fee_alpha(None))

    def test_alpha_via_rao(self):
        self.assertAlmostEqual(_as_fee_alpha(_FakeBalance(rao=int(1e9))), 1.0, places=10)


# ── _safe_raw_dump ────────────────────────────────────────────────

class TestSafeRawDump(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_safe_raw_dump(None))

    def test_bool(self):
        self.assertEqual(_safe_raw_dump(True), {"bool": True})
        self.assertEqual(_safe_raw_dump(False), {"bool": False})

    def test_dict_passthrough(self):
        self.assertEqual(_safe_raw_dump({"a": 1, "b": "x"}), {"a": 1, "b": "x"})

    def test_sdk_response_captured(self):
        r = _FakeExtrinsicResponse(
            extrinsic_fee=_FakeBalance(rao=8_400),
            extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xabc"),
            transaction_tao_fee=_FakeBalance(rao=252_000),
        )
        dump = _safe_raw_dump(r)
        self.assertIsInstance(dump, dict)
        self.assertIn("success", dump)
        self.assertIn("extrinsic_fee", dump)
        self.assertEqual(dump["success"], True)

    def test_unserializable_value_becomes_repr(self):
        class Opaque:
            def __repr__(self): return "<Opaque>"
        dump = _safe_raw_dump({"k": Opaque()})
        self.assertEqual(dump["k"], "<Opaque>")


# ── _status_from_tr ───────────────────────────────────────────────

class TestStatusFromTR(unittest.TestCase):

    def _mk(self, success, error=None):
        return TradeResult(
            success=success, netuid=1, trade_type="buy",
            tao_amount=0, alpha_amount=0, price=0, slippage=0,
            reason="", error=error,
        )

    def test_success(self):
        self.assertEqual(_status_from_tr(self._mk(True)), "success")

    def test_timeout(self):
        tr = self._mk(False, error="Timeout after 45s")
        self.assertEqual(_status_from_tr(tr), "timeout")

    def test_failed(self):
        tr = self._mk(False, error="RuntimeError: bad")
        self.assertEqual(_status_from_tr(tr), "failed")

    def test_failed_no_error(self):
        self.assertEqual(_status_from_tr(self._mk(False)), "failed")


# ── _coldkey_for_receipt ──────────────────────────────────────────

class TestColdkeyForReceipt(unittest.TestCase):

    def test_proxy_preferred(self):
        pm = MagicMock()
        pm.real_account_ss58 = "5Real"
        wallet = MagicMock()
        wallet.coldkey.ss58_address = "5WalletColdkey"
        self.assertEqual(_coldkey_for_receipt(pm, wallet), "5Real")

    def test_wallet_fallback(self):
        wallet = MagicMock()
        wallet.coldkey.ss58_address = "5Wallet"
        self.assertEqual(_coldkey_for_receipt(None, wallet), "5Wallet")

    def test_both_missing(self):
        self.assertIsNone(_coldkey_for_receipt(None, None))

    def test_exception_safe(self):
        bad = MagicMock()
        type(bad).coldkey = property(lambda s: (_ for _ in ()).throw(Exception("x")))
        self.assertIsNone(_coldkey_for_receipt(None, bad))


# ── End-to-end: _emit_fee_receipt with verified shapes ────────────

class TestEmitReceiptEndToEnd(unittest.TestCase):

    def test_buy_success_receipt_fields(self):
        """Full SDK response → receipt with all three observed_* fields set."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fee_receipts.jsonl"
            te = TradeExecutor(
                client=MagicMock(), fee_log_path=p, bot_id="testbot",
            )
            tr = TradeResult(
                success=True, netuid=23, trade_type="buy",
                tao_amount=0.5, alpha_amount=100.0, price=0.005,
                slippage=0.1, reason="entry",
            )
            raw_result = _FakeExtrinsicResponse(
                success=True,
                extrinsic_fee=_FakeBalance(rao=8_400),          # 8.4e-6 TAO gas
                extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xbuy_hash"),
                transaction_tao_fee=_FakeBalance(rao=252_000),  # 2.52e-4 TAO swap
            )
            te._emit_fee_receipt(
                trade_type="buy", netuid=23,
                tao_amount_requested=0.5, alpha_amount_requested=None,
                rate_tolerance=0.03,
                extrinsic_status=_status_from_tr(tr),
                raw_result=raw_result,
                pool_tao_at_submit=1000.0, pool_alpha_at_submit=20000.0,
                pool_tao_post=1000.5, pool_alpha_post=19900.0,
                coldkey_ss58="5D_coldkey", hotkey_ss58="5G_hotkey",
            )
            te._fee_writer.close()
            time.sleep(0.1)

            import json
            content = p.read_text().strip().split("\n")
            self.assertEqual(len(content), 1)
            rec = json.loads(content[0])
            self.assertEqual(rec["extrinsic_status"], "success")
            self.assertEqual(rec["chain_tx_hash"], "0xbuy_hash")
            self.assertAlmostEqual(rec["observed_gas_fee_tao"], 8.4e-6)
            self.assertAlmostEqual(rec["observed_swap_fee_tao"], 2.52e-4)
            self.assertIsNone(rec["parse_error"])

    def test_sell_alpha_fee_converted_to_tao(self):
        """Sells write observed_swap_fee_tao as alpha_fee × pool spot price."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fee_receipts.jsonl"
            te = TradeExecutor(
                client=MagicMock(), fee_log_path=p, bot_id="testbot",
            )
            tr = TradeResult(
                success=True, netuid=23, trade_type="sell",
                tao_amount=0.5, alpha_amount=100.0, price=0.005,
                slippage=0.1, reason="exit",
            )
            raw_result = _FakeExtrinsicResponse(
                success=True,
                extrinsic_fee=_FakeBalance(rao=8_400),
                extrinsic_receipt=_FakeReceipt(extrinsic_hash="0xsell_hash"),
                transaction_alpha_fee=_FakeBalance(rao=500_000_000),  # 0.5 alpha
            )
            te._emit_fee_receipt(
                trade_type="sell", netuid=23,
                tao_amount_requested=None, alpha_amount_requested=100.0,
                rate_tolerance=0.5,
                extrinsic_status=_status_from_tr(tr),
                raw_result=raw_result,
                pool_tao_at_submit=1000.0, pool_alpha_at_submit=20000.0,
                pool_tao_post=999.5, pool_alpha_post=20100.0,
                coldkey_ss58="5D", hotkey_ss58="5G",
            )
            te._fee_writer.close()
            time.sleep(0.1)

            import json
            rec = json.loads(p.read_text().strip())
            # spot = pool_tao_post / pool_alpha_post ≈ 999.5/20100 = 0.0497
            # observed_swap_tao = 0.5 × 0.0497 ≈ 0.02486
            expected = 0.5 * (999.5 / 20100.0)
            self.assertAlmostEqual(rec["observed_swap_fee_tao"], expected, places=6)

    def test_timeout_status_receipt_still_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fee_receipts.jsonl"
            te = TradeExecutor(
                client=MagicMock(), fee_log_path=p, bot_id="testbot",
            )
            tr = TradeResult(
                success=False, netuid=1, trade_type="buy",
                tao_amount=0, alpha_amount=0, price=0, slippage=0,
                reason="", error="Timeout after 45s",
            )
            te._emit_fee_receipt(
                trade_type="buy", netuid=1,
                tao_amount_requested=0.5, alpha_amount_requested=None,
                rate_tolerance=0.03,
                extrinsic_status=_status_from_tr(tr),
                raw_result=None,
                pool_tao_at_submit=None, pool_alpha_at_submit=None,
                pool_tao_post=None, pool_alpha_post=None,
                coldkey_ss58=None, hotkey_ss58=None,
            )
            te._fee_writer.close()
            time.sleep(0.1)

            import json
            rec = json.loads(p.read_text().strip())
            self.assertEqual(rec["extrinsic_status"], "timeout")
            self.assertIsNone(rec["observed_gas_fee_tao"])
            self.assertIsNone(rec["chain_tx_hash"])

    def test_fee_writer_disabled_is_noop(self):
        """fee_log_path=None → no writer, no file."""
        te = TradeExecutor(client=MagicMock())
        self.assertIsNone(te._fee_writer)
        # Should not raise
        te._emit_fee_receipt(
            trade_type="buy", netuid=1,
            tao_amount_requested=0.5, alpha_amount_requested=None,
            rate_tolerance=0.03, extrinsic_status="success",
            raw_result=_FakeExtrinsicResponse(),
            pool_tao_at_submit=None, pool_alpha_at_submit=None,
            pool_tao_post=None, pool_alpha_post=None,
            coldkey_ss58=None, hotkey_ss58=None,
        )

    def test_v2_block_fields_populated_from_sdk_receipt(self):
        """v2: block_hash + block_number flow through from the SDK
        ExtrinsicReceipt into the emitted fee-receipt record."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fee_receipts.jsonl"
            te = TradeExecutor(
                client=MagicMock(), fee_log_path=p, bot_id="testbot",
            )
            tr = TradeResult(
                success=True, netuid=23, trade_type="buy",
                tao_amount=0.5, alpha_amount=100.0, price=0.005,
                slippage=0.1, reason="entry",
            )
            raw_result = _FakeExtrinsicResponse(
                success=True,
                extrinsic_fee=_FakeBalance(rao=8_400),
                extrinsic_receipt=_FakeReceipt(
                    extrinsic_hash="0xbuy_hash",
                    block_hash="0xdeadbeef_block",
                    block_number=4_200_001,
                ),
                transaction_tao_fee=_FakeBalance(rao=252_000),
            )
            te._emit_fee_receipt(
                trade_type="buy", netuid=23,
                tao_amount_requested=0.5, alpha_amount_requested=None,
                rate_tolerance=0.03,
                extrinsic_status=_status_from_tr(tr),
                raw_result=raw_result,
                pool_tao_at_submit=1000.0, pool_alpha_at_submit=20000.0,
                pool_tao_post=1000.5, pool_alpha_post=19900.0,
                coldkey_ss58="5D", hotkey_ss58="5G",
            )
            te._fee_writer.close()
            time.sleep(0.1)

            import json
            rec = json.loads(p.read_text().strip())
            self.assertEqual(rec["schema_version"], 2)
            self.assertEqual(rec["block_hash"], "0xdeadbeef_block")
            self.assertEqual(rec["block_number"], 4_200_001)
            # extrinsic_index deferred — populated by later parser work (F2)
            self.assertIsNone(rec["extrinsic_index"])

    def test_v2_block_fields_null_when_receipt_missing(self):
        """Timeouts (no receipt) still write a record; v2 block fields null."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fee_receipts.jsonl"
            te = TradeExecutor(
                client=MagicMock(), fee_log_path=p, bot_id="testbot",
            )
            te._emit_fee_receipt(
                trade_type="buy", netuid=1,
                tao_amount_requested=0.5, alpha_amount_requested=None,
                rate_tolerance=0.03, extrinsic_status="timeout",
                raw_result=None,
                pool_tao_at_submit=None, pool_alpha_at_submit=None,
                pool_tao_post=None, pool_alpha_post=None,
                coldkey_ss58=None, hotkey_ss58=None,
            )
            te._fee_writer.close()
            time.sleep(0.1)

            import json
            rec = json.loads(p.read_text().strip())
            self.assertEqual(rec["schema_version"], 2)
            self.assertIsNone(rec["block_hash"])
            self.assertIsNone(rec["block_number"])
            self.assertIsNone(rec["extrinsic_index"])


if __name__ == "__main__":
    unittest.main()
