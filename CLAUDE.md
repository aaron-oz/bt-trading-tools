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

## Known traps (read before trusting a backtest number)

Two measured, silent ways a backtest result goes wrong. Details, measurements, and detection snippets are in `docs/known_traps.md`.

1. **A new engine per trade replays the same realism noise.** `BacktestEngine(realism_rng_seed=0)` is the default, and every engine with the same seed draws the same random numbers. Averaging many one-trade runs (one engine each) under one seed counts a single draw many times and overstated a 1 TAO round trip by about 0.4 percentage points. Use `panel_forward_returns` (distinct derived seed per window) or `seeded_engine_factory`. The engine warns after 25 near-single-trade runs under one seed.
2. **The engine fills from pool reserves, not from `SubnetTick.price`.** Hourly prices paired with stale (daily) reserves misprice fills; at surge entries the implied price averaged 0.92 x the tick price. `load_parquet_ticks` now rescales reserves to the hourly close by default (`reserves="rescale"`). Check any tick source with `reserve_price_gap`; `BacktestResults.orders_on_inconsistent_reserves` and a run-time warning flag the problem in a finished run.
