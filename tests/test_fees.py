"""Tests for bt_trading_tools.fees — FeeModel behavior.

Covers:
* chain-sourced quotes with a mock ChainFeeClient
* fallback path on any chain error (never raises)
* per-(op, netuid, amount-bucket, proxy) TTL cache
* amount bucketing collapses near-identical amounts
* proxy on vs off: proxy_fee_tao non-negative
* alpha-fee → TAO conversion on remove_stake via spot_price
* coldkey_ss58 resolution (kwarg > env > default)
"""

from __future__ import annotations

import os
import time
import unittest
from unittest import mock

from bt_trading_tools.fees import (
    DEFAULT_FEE_SIM_SS58,
    FALLBACK_GAS_TAO,
    FALLBACK_PROXY_TAO,
    FALLBACK_SWAP_RATE,
    FeeModel,
    FeeQuote,
    _bucket_amount,
)
from bt_trading_tools.tracking.schema import FeeSource


class _FakeChainClient:
    """Deterministic mock implementing the ChainFeeClient Protocol.

    Returns fixed values per operation so tests can assert exact
    numbers. Tracks call counts so cache tests can verify we don't
    re-query.
    """

    def __init__(self, tao_fee=0.000252, alpha_fee=0.5, gas=0.0000084, proxy_extra=1e-6):
        self.tao_fee = tao_fee
        self.alpha_fee = alpha_fee
        self.gas = gas
        self.proxy_extra = proxy_extra
        self.sim_swap_calls = 0
        self.payment_info_calls = 0
        self.raise_on_swap = False
        self.raise_on_payment = False

    def sim_swap(self, netuid, operation, amount):
        self.sim_swap_calls += 1
        if self.raise_on_swap:
            raise RuntimeError("simulated swap RPC failure")
        if operation == "add_stake":
            return {"tao_fee": self.tao_fee, "alpha_fee": None}
        else:
            return {"tao_fee": None, "alpha_fee": self.alpha_fee}

    def get_payment_info(self, operation, netuid, amount, coldkey_ss58, uses_proxy):
        self.payment_info_calls += 1
        if self.raise_on_payment:
            raise RuntimeError("simulated payment_info RPC failure")
        fee = self.gas + (self.proxy_extra if uses_proxy else 0.0)
        return {"partial_fee_tao": fee}


class TestBucketAmount(unittest.TestCase):

    def test_zero_and_negative(self):
        self.assertEqual(_bucket_amount(0), "0")
        self.assertEqual(_bucket_amount(-1.5), "0")

    def test_near_values_collapse(self):
        # 0.500 and 0.501 at 3 sig figs should bucket identically
        self.assertEqual(_bucket_amount(0.500), _bucket_amount(0.501))

    def test_order_of_magnitude_distinct(self):
        self.assertNotEqual(_bucket_amount(0.5), _bucket_amount(5.0))


class TestChainQuote(unittest.TestCase):

    def test_buy_quote_components(self):
        chain = _FakeChainClient(tao_fee=0.000252, gas=0.0000084, proxy_extra=1e-6)
        fm = FeeModel(chain=chain, coldkey_ss58="5Test")
        q = fm.quote("add_stake", netuid=23, amount=0.5, uses_proxy=True)
        self.assertIsInstance(q, FeeQuote)
        self.assertEqual(q.source, FeeSource.CHAIN)
        self.assertAlmostEqual(q.swap_fee_tao, 0.000252)
        self.assertAlmostEqual(q.gas_fee_tao, 0.0000084)
        self.assertAlmostEqual(q.proxy_fee_tao, 1e-6)
        self.assertAlmostEqual(
            q.total_fee_tao, 0.000252 + 0.0000084 + 1e-6,
        )
        self.assertIsNone(q.error)

    def test_sell_fee_converted_to_tao(self):
        """remove_stake returns alpha_fee; must convert via spot_price."""
        chain = _FakeChainClient(alpha_fee=2.0)
        fm = FeeModel(chain=chain)
        q = fm.quote(
            "remove_stake", netuid=23, amount=100.0,
            uses_proxy=False, spot_price=0.001,
        )
        # swap_fee_tao = alpha_fee × spot_price = 2.0 × 0.001 = 0.002
        self.assertAlmostEqual(q.swap_fee_tao, 0.002)
        self.assertEqual(q.source, FeeSource.CHAIN)

    def test_sell_without_spot_price_falls_back(self):
        chain = _FakeChainClient(alpha_fee=2.0)
        fm = FeeModel(chain=chain)
        q = fm.quote(
            "remove_stake", netuid=23, amount=100.0,
            uses_proxy=False, spot_price=None,
        )
        self.assertEqual(q.source, FeeSource.FALLBACK)
        self.assertIsNotNone(q.error)

    def test_proxy_off_has_zero_proxy_fee(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain)
        q = fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=False)
        self.assertEqual(q.proxy_fee_tao, 0.0)

    def test_proxy_on_has_non_negative_delta(self):
        chain = _FakeChainClient(gas=1e-5, proxy_extra=2e-6)
        fm = FeeModel(chain=chain)
        q = fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertGreaterEqual(q.proxy_fee_tao, 0.0)


class TestFallback(unittest.TestCase):

    def test_no_chain_client_returns_fallback(self):
        fm = FeeModel(chain=None)
        q = fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(q.source, FeeSource.FALLBACK)
        self.assertEqual(q.error, "no_chain_client")
        self.assertAlmostEqual(q.swap_fee_tao, 0.5 * FALLBACK_SWAP_RATE)
        self.assertAlmostEqual(q.gas_fee_tao, FALLBACK_GAS_TAO)
        self.assertAlmostEqual(q.proxy_fee_tao, FALLBACK_PROXY_TAO)

    def test_chain_raises_falls_back(self):
        chain = _FakeChainClient()
        chain.raise_on_swap = True
        fm = FeeModel(chain=chain)
        q = fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(q.source, FeeSource.FALLBACK)
        self.assertIn("simulated swap RPC failure", q.error or "")

    def test_payment_info_raises_falls_back(self):
        chain = _FakeChainClient()
        chain.raise_on_payment = True
        fm = FeeModel(chain=chain)
        q = fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(q.source, FeeSource.FALLBACK)

    def test_never_raises_on_garbage(self):
        chain = mock.Mock()
        chain.sim_swap.side_effect = TypeError("bad arg")
        fm = FeeModel(chain=chain)
        # Should not raise
        q = fm.quote("add_stake", netuid=0, amount=0.0, uses_proxy=False)
        self.assertEqual(q.source, FeeSource.FALLBACK)


class TestCache(unittest.TestCase):

    def test_cache_hit_within_ttl(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=60.0)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(chain.sim_swap_calls, 1)
        # payment_info called twice the first time (bare + proxy),
        # zero additional on cache hits
        self.assertEqual(chain.payment_info_calls, 2)

    def test_near_amounts_share_bucket(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=60.0)
        fm.quote("add_stake", netuid=1, amount=0.500, uses_proxy=True)
        fm.quote("add_stake", netuid=1, amount=0.501, uses_proxy=True)
        # 3-sig-fig bucketing means 0.500 == 0.501
        self.assertEqual(chain.sim_swap_calls, 1)

    def test_distinct_amounts_miss(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=60.0)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        fm.quote("add_stake", netuid=1, amount=5.0, uses_proxy=True)
        self.assertEqual(chain.sim_swap_calls, 2)

    def test_cache_expiry(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=0.01)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        time.sleep(0.02)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(chain.sim_swap_calls, 2)

    def test_proxy_distinct_from_no_proxy(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=60.0)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=False)
        self.assertEqual(chain.sim_swap_calls, 2)

    def test_clear_cache(self):
        chain = _FakeChainClient()
        fm = FeeModel(chain=chain, cache_ttl_s=60.0)
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        fm.clear_cache()
        fm.quote("add_stake", netuid=1, amount=0.5, uses_proxy=True)
        self.assertEqual(chain.sim_swap_calls, 2)


class TestColdkeyResolution(unittest.TestCase):

    def setUp(self):
        self._prior = os.environ.pop("FEE_MODEL_COLDKEY_SS58", None)

    def tearDown(self):
        if self._prior is not None:
            os.environ["FEE_MODEL_COLDKEY_SS58"] = self._prior
        else:
            os.environ.pop("FEE_MODEL_COLDKEY_SS58", None)

    def test_kwarg_takes_priority(self):
        os.environ["FEE_MODEL_COLDKEY_SS58"] = "5Env"
        fm = FeeModel(coldkey_ss58="5Kwarg")
        self.assertEqual(fm._coldkey_ss58, "5Kwarg")

    def test_env_used_when_no_kwarg(self):
        os.environ["FEE_MODEL_COLDKEY_SS58"] = "5Env"
        fm = FeeModel()
        self.assertEqual(fm._coldkey_ss58, "5Env")

    def test_default_fallback(self):
        fm = FeeModel()
        self.assertEqual(fm._coldkey_ss58, DEFAULT_FEE_SIM_SS58)


if __name__ == "__main__":
    unittest.main()
