"""
WalletManager — thin wrapper for Bittensor wallet setup.

Each bot creates its own wallet. This helper standardizes the
load/unlock/env-var pattern used across all bots.

Usage::

    wm = WalletManager(name="doubledip", password_env="DD_WALLET_PW")
    wallet = wm.setup()
    # wallet.coldkey.ss58_address is now available
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class WalletManager:
    """Bittensor wallet setup helper.

    Args:
        name: Wallet name (as created by ``btcli w create``).
        password_env: Environment variable name holding the wallet password.
            If the env var is not set, falls back to ``password`` arg.
        password: Fallback password if env var is not set. Prefer env var
            for security.
        create_if_missing: Whether to create the wallet if it doesn't exist.
    """

    def __init__(
        self,
        name: str,
        password_env: str | None = None,
        password: str | None = None,
        create_if_missing: bool = False,
    ):
        self.name = name
        self.password_env = password_env
        self.password = password
        self.create_if_missing = create_if_missing
        self._wallet = None

    def setup(self) -> Any:
        """Load and unlock the wallet. Returns the bittensor Wallet object."""
        import bittensor as bt

        wallet = bt.Wallet(name=self.name)

        if self.create_if_missing:
            wallet.create_if_non_existent()

        # Resolve password: env var takes precedence
        pw = None
        if self.password_env:
            pw = os.environ.get(self.password_env)
        if pw is None:
            pw = self.password

        if pw:
            wallet.coldkey_file.save_password_to_env(pw)

        wallet.unlock_coldkey()
        self._wallet = wallet

        logger.info(
            f"Wallet '{self.name}' unlocked "
            f"(coldkey={wallet.coldkey.ss58_address[:8]}...)"
        )
        return wallet

    @property
    def wallet(self) -> Any:
        """The loaded wallet, or raises if setup() hasn't been called."""
        if self._wallet is None:
            raise RuntimeError("Wallet not set up. Call setup() first.")
        return self._wallet

    @property
    def coldkey_ss58(self) -> str:
        """Coldkey SS58 address."""
        return self.wallet.coldkey.ss58_address


class ProxyWalletManager:
    """Wallet manager for proxy-based trading.

    The proxy wallet signs all transactions, but trades execute on behalf
    of ``real_account_ss58`` (the Ledger account). The proxy must have been
    granted Staking proxy rights on-chain.

    Args:
        proxy_name: Wallet name of the proxy (e.g., "autobot-proxy").
        real_account_ss58: The Ledger account SS58 address.
        password_env: Env var holding the proxy wallet password.
        password: Fallback password if env var is not set.
    """

    def __init__(
        self,
        proxy_name: str,
        real_account_ss58: str,
        password_env: str | None = None,
        password: str | None = None,
    ):
        self._wm = WalletManager(
            name=proxy_name, password_env=password_env, password=password,
        )
        self._real_account_ss58 = real_account_ss58

    def setup(self) -> Any:
        """Load and unlock the proxy wallet. Returns the bittensor Wallet object."""
        wallet = self._wm.setup()
        logger.info(
            f"Proxy wallet '{self._wm.name}' configured for "
            f"real account {self._real_account_ss58[:8]}..."
        )
        return wallet

    @property
    def proxy_wallet(self) -> Any:
        """The proxy wallet object (for signing)."""
        return self._wm.wallet

    @property
    def real_account_ss58(self) -> str:
        """The Ledger account address (trades execute on this account)."""
        return self._real_account_ss58

    @property
    def proxy_ss58(self) -> str:
        """The proxy wallet's coldkey address."""
        return self._wm.coldkey_ss58

    @property
    def coldkey_ss58(self) -> str:
        """The account to query for balance/stake (the Ledger account).

        This lets code that uses ``manager.coldkey_ss58`` for on-chain
        queries work transparently with both WalletManager and
        ProxyWalletManager.
        """
        return self._real_account_ss58
