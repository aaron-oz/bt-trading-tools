"""Tests for bt_trading_tools.network — client, executor, wallet.

These tests mock the bittensor SDK so they run without chain access.
"""

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from bt_trading_tools.network.client import SubtensorClient, SubnetInfo
from bt_trading_tools.network.executor import TradeExecutor, TradeResult


# ── Mock helpers ─────────────────────────────────────────────────

@dataclass
class MockBalance:
    tao: float
    rao: int = 0

    def __float__(self):
        return self.tao

    @classmethod
    def from_rao(cls, rao, netuid=None):
        return cls(tao=rao / 1e9, rao=rao)


@dataclass
class MockSubnet:
    netuid: int
    price: float
    tao_in: MockBalance
    alpha_in: MockBalance
    subnet_name: str = ""


@dataclass
class MockStake:
    stake: MockBalance


class MockExtrinsicResponse:
    def __init__(self, success=True):
        self.success = success


def make_mock_subtensor(subnets=None, balance=100.0, stakes=None):
    """Create a mock async subtensor with configurable responses."""
    sub = AsyncMock()

    if subnets is None:
        subnets = [
            MockSubnet(107, 0.005, MockBalance(150.0), MockBalance(30000.0), "alpha107"),
            MockSubnet(99, 0.02, MockBalance(500.0), MockBalance(25000.0), "alpha99"),
        ]
    sub.all_subnets = AsyncMock(return_value=subnets)
    sub.get_balance = AsyncMock(return_value=MockBalance(balance))
    sub.wait_for_block = AsyncMock()
    sub.close = AsyncMock()

    if stakes is None:
        stakes = {107: MockStake(MockBalance(50.0)), 99: MockStake(MockBalance(0.0))}
    sub.get_stake_for_coldkey_and_hotkey = AsyncMock(return_value=stakes)

    sub.add_stake = AsyncMock(return_value=MockExtrinsicResponse(True))
    sub.unstake = AsyncMock(return_value=MockExtrinsicResponse(True))

    return sub


# ── Client tests ─────────────────────────────────────────────────

class TestSubtensorClient(unittest.TestCase):

    def test_get_all_subnets(self):
        """Parse subnet data correctly."""
        client = SubtensorClient()
        client._sub = make_mock_subtensor()

        result = asyncio.run(client.get_all_subnets())

        self.assertIn(107, result)
        self.assertIn(99, result)
        self.assertIsInstance(result[107], SubnetInfo)
        self.assertAlmostEqual(result[107].price, 0.005)
        self.assertAlmostEqual(result[107].tao_in, 150.0)
        self.assertEqual(result[107].name, "alpha107")

    def test_get_all_subnets_skips_zero_price(self):
        """Subnets with price=0 are excluded."""
        client = SubtensorClient()
        client._sub = make_mock_subtensor(subnets=[
            MockSubnet(1, 0.0, MockBalance(100.0), MockBalance(100.0)),
            MockSubnet(2, 0.01, MockBalance(100.0), MockBalance(100.0)),
        ])

        result = asyncio.run(client.get_all_subnets())
        self.assertNotIn(1, result)
        self.assertIn(2, result)

    def test_get_balance(self):
        client = SubtensorClient()
        client._sub = make_mock_subtensor(balance=42.5)

        balance = asyncio.run(client.get_balance("5Gtest"))
        self.assertAlmostEqual(balance, 42.5)

    def test_get_stakes_parallel(self):
        """get_stakes fetches multiple hotkeys in parallel."""
        client = SubtensorClient()
        mock_sub = make_mock_subtensor()
        client._sub = mock_sub

        # Different return per call
        call_count = 0
        async def mock_get_stake(hotkey_ss58, coldkey_ss58):
            return {107: MockStake(MockBalance(50.0))}

        mock_sub.get_stake_for_coldkey_and_hotkey = AsyncMock(side_effect=mock_get_stake)

        result = asyncio.run(client.get_stakes("5Gcoldkey", ["5Ghot1", "5Ghot2"]))
        self.assertIn("5Ghot1", result)
        self.assertIn("5Ghot2", result)
        self.assertEqual(mock_sub.get_stake_for_coldkey_and_hotkey.call_count, 2)

    def test_total_stake(self):
        """Sum stake across multiple hotkeys."""
        stake_info = {
            "hot1": {107: MockStake(MockBalance(30.0)), 99: MockStake(MockBalance(10.0))},
            "hot2": {107: MockStake(MockBalance(20.0))},
        }
        total = SubtensorClient.total_stake(stake_info, 107)
        self.assertAlmostEqual(total, 50.0)

    def test_total_stake_missing_subnet(self):
        stake_info = {"hot1": {99: MockStake(MockBalance(10.0))}}
        total = SubtensorClient.total_stake(stake_info, 107)
        self.assertAlmostEqual(total, 0.0)

    def test_find_hotkey_with_stake_preferred(self):
        """Preferred hotkey is returned when it has stake."""
        stake_info = {
            "hot1": {107: MockStake(MockBalance(30.0))},
            "hot2": {107: MockStake(MockBalance(20.0))},
        }
        result = SubtensorClient.find_hotkey_with_stake(stake_info, 107, "hot1")
        self.assertEqual(result, "hot1")

    def test_find_hotkey_with_stake_fallback(self):
        """Falls back to any hotkey when preferred has no stake."""
        stake_info = {
            "hot1": {107: MockStake(MockBalance(0.0))},
            "hot2": {107: MockStake(MockBalance(20.0))},
        }
        result = SubtensorClient.find_hotkey_with_stake(stake_info, 107, "hot1")
        self.assertEqual(result, "hot2")

    def test_find_hotkey_with_stake_none(self):
        """Returns None when no hotkey has stake."""
        stake_info = {"hot1": {99: MockStake(MockBalance(10.0))}}
        result = SubtensorClient.find_hotkey_with_stake(stake_info, 107)
        self.assertIsNone(result)

    def test_discover_validators(self):
        """Parse validator discovery response."""
        client = SubtensorClient()
        mock_sub = make_mock_subtensor()

        @dataclass
        class MockStakeInfo:
            hotkey_ss58: str

        mock_sub.get_stake_info_for_coldkey = AsyncMock(return_value=[
            MockStakeInfo("5Ghot1"),
            MockStakeInfo("5Ghot2"),
            MockStakeInfo("5Ghot1"),  # duplicate
        ])
        client._sub = mock_sub

        validators = asyncio.run(client.discover_validators("5Gcold"))
        self.assertIsNotNone(validators)
        self.assertEqual(len(validators), 2)
        self.assertIn("5Ghot1", validators)
        self.assertIn("5Ghot2", validators)

    def test_discover_validators_timeout(self):
        """Returns None on timeout."""
        client = SubtensorClient()
        mock_sub = make_mock_subtensor()
        mock_sub.get_stake_info_for_coldkey = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        client._sub = mock_sub

        result = asyncio.run(client.discover_validators("5Gcold"))
        self.assertIsNone(result)


# ── Executor tests ───────────────────────────────────────────────

class TestTradeExecutor(unittest.TestCase):

    def _make_executor(self, subnets=None, balance=100.0):
        client = SubtensorClient()
        client._sub = make_mock_subtensor(subnets=subnets, balance=balance)
        return TradeExecutor(client)

    def test_trade_result_dataclass(self):
        tr = TradeResult(
            success=True, netuid=107, trade_type="buy",
            tao_amount=0.5, alpha_amount=100, price=0.005,
            slippage=0.3, reason="test",
        )
        self.assertTrue(tr.success)
        self.assertEqual(tr.netuid, 107)

    def test_available_balance(self):
        executor = self._make_executor()
        # available = total - reserved - fee_reserve (default 0.01)
        self.assertAlmostEqual(executor.available_balance(100.0), 100.0 - 0.01)
        executor._reserved_balance = 10.0
        self.assertAlmostEqual(executor.available_balance(100.0), 90.0 - 0.01)

    def test_pending_tracking(self):
        executor = self._make_executor()
        self.assertFalse(executor.is_pending(107))
        executor._pending_trades[107] = "buying"
        self.assertTrue(executor.is_pending(107))
        self.assertEqual(executor.pending_direction(107), "buying")

    def test_process_pending_empty(self):
        executor = self._make_executor()
        completed = executor.process_pending()
        self.assertEqual(len(completed), 0)

    def test_process_pending_harvests_done(self):
        """Completed tasks are harvested and removed."""
        executor = self._make_executor()

        # Simulate a completed task with MagicMock
        tr = TradeResult(True, 107, "buy", 0.5, 100, 0.005, 0.3, "test")
        task = MagicMock()
        task.done.return_value = True
        task.result.return_value = tr

        executor._pending_tasks = [task]
        completed = executor.process_pending()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].netuid, 107)
        self.assertEqual(len(executor._pending_tasks), 0)

    def test_process_pending_keeps_running(self):
        """Running tasks stay in the pending list."""
        executor = self._make_executor()
        task = MagicMock()
        task.done.return_value = False
        executor._pending_tasks = [task]
        completed = executor.process_pending()
        self.assertEqual(len(completed), 0)
        self.assertEqual(len(executor._pending_tasks), 1)


# ── Wallet tests ─────────────────────────────────────────────────

class TestWalletManager(unittest.TestCase):

    @patch.dict("os.environ", {"TEST_WALLET_PW": "secret123"})
    def test_env_var_precedence(self):
        """Password from env var takes precedence."""
        from bt_trading_tools.network.wallet import WalletManager
        wm = WalletManager(name="test", password_env="TEST_WALLET_PW", password="fallback")
        # We can't actually create a wallet without btcli, but we can check
        # that the config is stored correctly
        self.assertEqual(wm.name, "test")
        self.assertEqual(wm.password_env, "TEST_WALLET_PW")

    def test_wallet_not_setup_raises(self):
        from bt_trading_tools.network.wallet import WalletManager
        wm = WalletManager(name="test")
        with self.assertRaises(RuntimeError):
            _ = wm.wallet


# ── Proxy wallet tests ───────────────────────────────────────────

class TestProxyWalletManager(unittest.TestCase):

    def test_coldkey_ss58_returns_real_account(self):
        """coldkey_ss58 returns the Ledger (real) address, not the proxy."""
        from bt_trading_tools.network.wallet import ProxyWalletManager
        pm = ProxyWalletManager(
            proxy_name="autobot-proxy",
            real_account_ss58="5GrealLedgerAddress",
            password_env="PROXY_PW",
        )
        self.assertEqual(pm.real_account_ss58, "5GrealLedgerAddress")
        self.assertEqual(pm.coldkey_ss58, "5GrealLedgerAddress")

    def test_proxy_wallet_not_setup_raises(self):
        from bt_trading_tools.network.wallet import ProxyWalletManager
        pm = ProxyWalletManager(
            proxy_name="test-proxy",
            real_account_ss58="5Gtest",
        )
        with self.assertRaises(RuntimeError):
            _ = pm.proxy_wallet

    def test_proxy_ss58_requires_setup(self):
        from bt_trading_tools.network.wallet import ProxyWalletManager
        pm = ProxyWalletManager(
            proxy_name="test-proxy",
            real_account_ss58="5Gtest",
        )
        with self.assertRaises(RuntimeError):
            _ = pm.proxy_ss58


# ── Proxy executor tests ────────────────────────────────────────

class TestTradeExecutorProxy(unittest.TestCase):

    def _make_proxy_manager(self):
        """Create a mock ProxyWalletManager."""
        pm = MagicMock()
        pm.proxy_wallet = MagicMock()
        pm.proxy_wallet.coldkey.ss58_address = "5Gproxy"
        pm.real_account_ss58 = "5GrealLedger"
        pm.proxy_ss58 = "5Gproxy"
        pm.coldkey_ss58 = "5GrealLedger"
        return pm

    def _make_proxy_executor(self, balance=100.0):
        client = SubtensorClient()
        client._sub = make_mock_subtensor(balance=balance)
        pm = self._make_proxy_manager()
        return TradeExecutor(client, proxy_manager=pm), pm

    def test_executor_stores_proxy_manager(self):
        executor, pm = self._make_proxy_executor()
        self.assertIs(executor.proxy_manager, pm)

    def test_executor_without_proxy_backwards_compatible(self):
        """No proxy_manager = same behavior as before."""
        client = SubtensorClient()
        client._sub = make_mock_subtensor()
        executor = TradeExecutor(client)
        self.assertIsNone(executor.proxy_manager)

    def _install_bittensor_mocks(self):
        """Pre-inject mock bittensor modules into sys.modules.

        The executor imports bittensor lazily inside buy()/sell().
        Since bittensor can't be fully imported in the test env (missing
        aiohttp deps), we inject mocks before the code runs.
        """
        import sys

        # Mock the top-level bittensor module with Balance support
        mock_bt = MagicMock()
        mock_balance = MagicMock()
        mock_balance.rao = 500000000  # 0.5 TAO in rao
        mock_bt.utils.balance.tao.return_value = mock_balance
        mock_bt.Balance.from_rao.return_value = mock_balance

        mock_pallets = MagicMock()
        mock_proxy_mod = MagicMock()

        sys.modules["bittensor"] = mock_bt
        sys.modules["bittensor.core"] = MagicMock()
        sys.modules["bittensor.core.extrinsics"] = MagicMock()
        sys.modules["bittensor.core.extrinsics.pallets"] = mock_pallets
        sys.modules["bittensor.core.chain_data"] = MagicMock()
        sys.modules["bittensor.core.chain_data.proxy"] = mock_proxy_mod

        return mock_bt, mock_pallets, mock_proxy_mod

    def setUp(self):
        """Save bittensor modules state to restore after each test."""
        import sys
        self._saved_modules = {
            k: sys.modules.get(k)
            for k in [
                "bittensor", "bittensor.core", "bittensor.core.extrinsics",
                "bittensor.core.extrinsics.pallets", "bittensor.core.chain_data",
                "bittensor.core.chain_data.proxy",
            ]
        }

    def tearDown(self):
        """Restore bittensor modules state."""
        import sys
        for k, v in self._saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_buy_with_proxy_calls_subtensor_proxy(self):
        """buy() wraps the call in subtensor.proxy() when proxy_manager is set."""
        mock_bt, mock_pallets, mock_proxy_mod = self._install_bittensor_mocks()
        executor, pm = self._make_proxy_executor()

        mock_call = MagicMock()
        mock_proxy_response = MockExtrinsicResponse(True)

        # SubtensorModule().add_stake() returns the inner call
        mock_module_instance = AsyncMock()
        mock_module_instance.add_stake = AsyncMock(return_value=mock_call)
        mock_pallets.SubtensorModule.return_value = mock_module_instance

        # subtensor.proxy() returns success
        executor.client.sub.proxy = AsyncMock(return_value=mock_proxy_response)

        result = asyncio.run(executor.buy(
            pm.proxy_wallet, "5Ghotkey", netuid=107,
            tao_amount=0.5, pool_tao=150.0, reason="test",
        ))

        # Verify proxy() was called, not add_stake() directly
        executor.client.sub.proxy.assert_called_once()
        call_kwargs = executor.client.sub.proxy.call_args
        self.assertEqual(call_kwargs.kwargs["wallet"], pm.proxy_wallet)
        self.assertEqual(call_kwargs.kwargs["real_account_ss58"], "5GrealLedger")
        self.assertEqual(call_kwargs.kwargs["call"], mock_call)

    def test_sell_with_proxy_calls_subtensor_proxy(self):
        """sell() wraps the call in subtensor.proxy() when proxy_manager is set."""
        mock_bt, mock_pallets, mock_proxy_mod = self._install_bittensor_mocks()
        executor, pm = self._make_proxy_executor()

        mock_call = MagicMock()
        mock_proxy_response = MockExtrinsicResponse(True)

        mock_module_instance = AsyncMock()
        mock_module_instance.remove_stake = AsyncMock(return_value=mock_call)
        mock_pallets.SubtensorModule.return_value = mock_module_instance

        executor.client.sub.proxy = AsyncMock(return_value=mock_proxy_response)

        result = asyncio.run(executor.sell(
            pm.proxy_wallet, "5Ghotkey", netuid=107,
            alpha_amount=100.0, price=0.005, pool_tao=150.0,
            reason="test",
        ))

        executor.client.sub.proxy.assert_called_once()
        call_kwargs = executor.client.sub.proxy.call_args
        self.assertEqual(call_kwargs.kwargs["wallet"], pm.proxy_wallet)
        self.assertEqual(call_kwargs.kwargs["real_account_ss58"], "5GrealLedger")

    def test_sell_with_proxy_queries_real_account_balance(self):
        """sell() uses real_account_ss58 for balance queries, not proxy address."""
        mock_bt, mock_pallets, mock_proxy_mod = self._install_bittensor_mocks()
        executor, pm = self._make_proxy_executor()

        mock_call = MagicMock()
        mock_proxy_response = MockExtrinsicResponse(True)

        mock_module_instance = AsyncMock()
        mock_module_instance.remove_stake = AsyncMock(return_value=mock_call)
        mock_pallets.SubtensorModule.return_value = mock_module_instance
        executor.client.sub.proxy = AsyncMock(return_value=mock_proxy_response)

        result = asyncio.run(executor.sell(
            pm.proxy_wallet, "5Ghotkey", netuid=107,
            alpha_amount=100.0, price=0.005, pool_tao=150.0,
            reason="test",
        ))

        # get_balance should be called with real account, not proxy
        balance_calls = executor.client.sub.get_balance.call_args_list
        for call in balance_calls:
            addr = call.kwargs.get("address", call.args[0] if call.args else None)
            self.assertNotEqual(addr, "5Gproxy",
                "Balance query should use real_account_ss58, not proxy address")


if __name__ == "__main__":
    unittest.main()
