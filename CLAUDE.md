# bt-trading-tools

Generic trading infrastructure for Bittensor alpha-token trading. Installed as editable package.

## Workspace location

**Canonical workspace:** `/var/home/aoz/code/bt-trading-tools/` (and worktrees under `/var/home/aoz/code/bt-trading-tools-worktrees/`). The Dropbox copy at `/var/home/aoz/Dropbox/bittensor/bt-trading-tools/` is an out-of-band sync mirror — files there can lag `origin/main` by hours and may contain stale or in-progress edits from another session. Always work from the `~/code/` clone, for both reads and writes. First action on every session: `cd /var/home/aoz/code/bt-trading-tools && git fetch origin && git status -sb`; pivot here if the session launched with cwd inside the Dropbox tree. See global CLAUDE.md "canonical workspace" rule for the full background.

## Modules

| Module | What it provides |
|--------|-----------------|
| `bt_trading_tools.amm` | AMM math: `amm_buy()`, `amm_sell()`, `spot_price()`, `slippage_pct()`, `max_trade_for_slippage()` |
| `bt_trading_tools.network` | `WalletManager`, `ProxyWalletManager`, `SubtensorClient` (async, auto-reconnect), `TradeExecutor` (non-blocking buy/sell) |
| `bt_trading_tools.backtest` | `BacktestEngine`, `PurgedWalkForwardCV`, `ScheduledStrategy`, types: `TickData`, `SubnetTick`, `Order`, `Position`, `Strategy` protocol |
| `bt_trading_tools.data` | `UnifiedDataLoader` + `LoaderConfig` -> `DataArrays` (aligned n_times x n_subnets grids) |
| `bt_trading_tools.tracking` | `TradeLog` (SQLite), `PortfolioLog`, `DecisionLog`, `EventLog` |
| `bt_trading_tools.utils` | `detect_lifecycle_boundaries()`, `apply_lifecycle_mask()` -- subnet rebirth/deregistration handling |
