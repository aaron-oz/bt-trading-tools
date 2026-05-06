"""
bt_trading_tools.network — Async Bittensor network interface.

Provides:
    SubtensorClient  Wallet-agnostic async chain connection with auto-reconnect.
    TradeExecutor     Non-blocking trade execution with background tasks.
    WalletManager     Wallet setup and unlock helper.
"""

from bt_trading_tools.network.client import SubtensorClient
from bt_trading_tools.network.executor import FEE_RESERVE_TAO, TradeExecutor, TradeResult
from bt_trading_tools.network.wallet import (
    ProxyWalletManager,
    UnstakeAllResult,
    UnstakePlanItem,
    UnstakeResult,
    WalletManager,
    WalletPosition,
    WalletValuation,
    unstake_all,
    valuate_wallet,
)

__all__ = [
    "FEE_RESERVE_TAO",
    "ProxyWalletManager",
    "SubtensorClient",
    "TradeExecutor",
    "TradeResult",
    "UnstakeAllResult",
    "UnstakePlanItem",
    "UnstakeResult",
    "WalletManager",
    "WalletPosition",
    "WalletValuation",
    "unstake_all",
    "valuate_wallet",
]
