# bt-trading-tools

Generic trading infrastructure for Bittensor alpha-token trading. Installed as editable package.

## Modules

| Module | What it provides |
|--------|-----------------|
| `bt_trading_tools.amm` | AMM math: `amm_buy()`, `amm_sell()`, `spot_price()`, `slippage_pct()`, `max_trade_for_slippage()` |
| `bt_trading_tools.network` | `WalletManager`, `ProxyWalletManager`, `SubtensorClient` (async, auto-reconnect), `TradeExecutor` (non-blocking buy/sell) |
| `bt_trading_tools.backtest` | `BacktestEngine`, `PurgedWalkForwardCV`, `ScheduledStrategy`, types: `TickData`, `SubnetTick`, `Order`, `Position`, `Strategy` protocol |
| `bt_trading_tools.data` | `UnifiedDataLoader` + `LoaderConfig` -> `DataArrays` (aligned n_times x n_subnets grids) |
| `bt_trading_tools.tracking` | `TradeLog` (SQLite), `PortfolioLog`, `DecisionLog`, `EventLog` |
| `bt_trading_tools.utils` | `detect_lifecycle_boundaries()`, `apply_lifecycle_mask()` -- subnet rebirth/deregistration handling |
