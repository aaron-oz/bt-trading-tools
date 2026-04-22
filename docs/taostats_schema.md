# Taostats Data Schema

Schema reference for the CSV files consumed by `bt_trading_tools.data.UnifiedDataLoader`. All data originates from the [Taostats API](https://taostats.io/developers/) and is stored in a local `data/taostats/` directory.

## Data Pipeline Overview

```
Taostats API ──> backfill scripts ──> CSV files ──> UnifiedDataLoader ──> DataArrays
                                          │
                              build_ohlcv_from_delegations.py
                              (derives OHLCV from delegation_full.csv)
```

**Scripts** (in `alpha-trading/data/`):
- `backfill_taostats.py` — Full backfill of delegation transactions (cursor-based, multiprocessing)
- `backfill_taostats_supplementary.py` — Backfill pool_history, subnet_history, tao_price
- `update_taostats.py` — Incremental updater (designed for systemd timer, every 6h)
- `build_ohlcv_from_delegations.py` — Reconstructs OHLCV bars from raw delegations

All scripts require the `TAOSTATS_API_KEY` environment variable. See `data/.env.example`.

---

## CSV Files

### delegation_full.csv

Raw delegation (stake/unstake) transactions. Primary source of truth for all price data.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `id` | int | — | Unique transaction ID |
| `block_number` | int | — | Bittensor block number |
| `timestamp` | datetime | UTC ISO8601 | Transaction timestamp |
| `action` | str | — | `DELEGATE` (buy/stake) or `UNDELEGATE` (sell/unstake) |
| `nominator` | str | SS58 | Wallet address initiating the action |
| `delegate` | str | SS58 | Validator hotkey being staked to |
| `amount` | float | TAO | TAO amount involved in the transaction |
| `alpha` | float | tokens | Alpha tokens involved |
| `netuid` | int | — | Subnet network UID |
| `alpha_price_in_tao` | float | TAO/alpha | Spot price at time of transaction |
| `slippage` | float | fraction | Price impact of this trade |
| `fee` | float | TAO | Transaction fee paid |

**Size:** ~8 GB, 22.7M+ rows. Sorted by `block_number` DESC.

**Source:** `api/delegation/v1`

---

### delegation_ohlcv_5m.csv / delegation_ohlcv_hourly.csv

OHLCV price bars reconstructed from delegation transactions. Between transactions, the AMM pool state doesn't change, so forward-fill is ground truth (not an approximation).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time` | datetime | UTC ISO8601 | Bar timestamp (period start) |
| `netuid` | int | — | Subnet network UID |
| `open` | float | TAO/alpha | Opening price |
| `high` | float | TAO/alpha | High price |
| `low` | float | TAO/alpha | Low price |
| `close` | float | TAO/alpha | Closing price |
| `volume` | float | TAO | Total TAO volume in bar |
| `n_trades` | int | — | Number of transactions in bar |
| `net_flow_tao` | float | TAO | Signed TAO flow (positive = inflow/buys, negative = outflow/sells) |

**Size:** ~563 MB (5m), ~141 MB (hourly).

**Built by:** `build_ohlcv_from_delegations.py` — not fetched from API directly.

**Gotcha:** Bars with `n_trades=0` are forward-filled from the previous close. This is accurate (AMM price doesn't change without trades), but be aware when computing volume-weighted metrics.

---

### pool_history.csv

Daily AMM pool state per subnet. One row per subnet per day.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `netuid` | int | — | Subnet network UID |
| `block_number` | int | — | Block number of snapshot |
| `timestamp` | datetime | UTC ISO8601 | Snapshot timestamp |
| `price` | float | TAO/alpha | Spot price |
| `market_cap` | float | rao | Market capitalization |
| `liquidity` | float | rao | Pool liquidity |
| `total_tao` | float | **rao** | TAO in pool (divide by 1e9 for TAO) |
| `total_alpha` | float | **rao** | Total alpha supply (divide by 1e9 for tokens) |
| `alpha_in_pool` | float | **rao** | Alpha in AMM pool (divide by 1e9 for tokens) |
| `alpha_staked` | float | **rao** | Alpha staked to validators (divide by 1e9 for tokens) |
| `root_prop` | float | fraction | Root network proportion |
| `rank` | int | — | Subnet rank |
| `startup_mode` | bool | — | Whether subnet is in bootstrap mode (prices are nonsensical) |
| `fee_rate` | float | fraction | Swap fee rate |
| `fee_global_alpha` | float | rao | Global alpha fees collected |
| `fee_global_tao` | float | rao | Global TAO fees collected |
| `swap_v3_initialized` | bool | — | Whether V3 concentrated liquidity is active |
| `liquidity_raw` | float | — | Raw liquidity value (V3) |
| `current_tick` | int | — | Current price tick (V3) |
| `protocol_provided_tao` | float | **rao** | Protocol-provided TAO liquidity (divide by 1e9) |
| `user_provided_tao` | float | rao | User-provided TAO liquidity |
| `protocol_provided_alpha` | float | **rao** | Protocol-provided alpha liquidity (divide by 1e9) |
| `user_provided_alpha` | float | rao | User-provided alpha liquidity |
| `enabled_user_liquidity` | bool | — | Whether user liquidity provision is enabled |

**Size:** ~19 MB, ~44K rows.

**Source:** `api/dtao/pool/history/v1`

**Gotchas:**
- Most numeric columns are in **rao** (1e-9 TAO/tokens). The `UnifiedDataLoader` divides `total_tao`, `alpha_in_pool`, `alpha_staked`, `protocol_provided_tao`, and `protocol_provided_alpha` by 1e9 automatically.
- **`startup_mode` filtering is critical:** ~53/129 subnets had nonsensical prices during bootstrap. Always filter via `startup_mode` before using price data. The loader does this automatically.
- The loader derives `k = (total_tao/1e9) * (alpha_in_pool/1e9)` for AMM invariant calculations.

---

### subnet_history.csv

Daily subnet configuration and emission data. One row per subnet per day.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `netuid` | int | — | Subnet network UID |
| `block_number` | int | — | Block number of snapshot |
| `timestamp` | datetime | UTC ISO8601 | Snapshot timestamp |
| `emission` | float | **rao per block** | Per-block TAO emission snapshot (`SubtensorModule::SubnetTaoInEmission` at the Taostats snapshot block). NOT per-tempo. Tempo is irrelevant to daily conversion. See the "Gotcha" note below. |
| `projected_emission` | float | fraction (dimensionless) | Taoflow share at the snapshot; sums to ~1.0 across active subnets. Preferred input for daily-rate calculations (see loader Key Derived Fields). |
| `ema_tao_flow` | float | rao | Exponential moving average of TAO flow |
| `tao_flow` | float | rao | Raw TAO flow |
| `excess_tao` | float | rao | Excess TAO in subnet |
| `active_keys` | int | — | Total registered keys |
| `validators` | int | — | Total validators |
| `active_validators` | int | — | Active validators |
| `active_miners` | int | — | Active miners |
| `max_neurons` | int | — | Maximum allowed neurons |
| `tempo` | int | blocks | Subnet tempo (blocks per epoch) |
| `immunity_period` | int | blocks | Immunity period for new neurons |
| `registration_cost` | float | rao | Cost to register a neuron |
| `neuron_registration_cost` | float | rao | Alternative registration cost field |
| `rho` | float | — | Rho parameter |
| `kappa` | float | — | Kappa parameter |
| `weights_version` | int | — | Weights version |
| `weights_rate_limit` | int | — | Weights rate limit |
| `recycled_lifetime` | float | rao | Lifetime recycled TAO |
| `recycled_24_hours` | float | rao | TAO recycled in last 24h |
| `recycled_since_registration` | float | rao | TAO recycled since registration |
| `fee_rate` | float | fraction | Fee rate |
| `bonds_penalty` | float | — | Bonds penalty parameter |

**Size:** ~13 MB, ~44K rows.

**Source:** `api/subnet/history/v1`

**Gotcha (corrected 2026-04-20):** the loader derives `daily_emission_tao`. An earlier version of this formula divided by `tempo` — that was wrong. The `emission` field is rao per *block*, not per *tempo*; dividing by tempo undercounted daily injection by exactly that factor (~360× for typical subnets).

Preferred formula, used by `UnifiedDataLoader` when `projected_emission` is available:

```
daily_emission_tao = projected_emission × block_reward(t) × 7200
```

where `block_reward(t) = 1.0` TAO/block pre-halving (before 2025-12-14) and `0.5` post-halving, and `7200` is blocks/day (12 s block time).

Fallback formula when `projected_emission` is missing, used for older data:

```
daily_emission_tao = emission × 7200 / 1e9
```

The fallback is correct in expectation but noisy at single-block granularity (~50% of subnets show `emission = 0` on any given snapshot block). See `docs/bittensor-mechanics-primer.md` §5 for the full derivation and the loader source `bt_trading_tools/data/loader.py` for implementation.

---

### tao_price.csv

TAO/USD price at 15-minute intervals.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | UTC ISO8601 | Price timestamp |
| `price` | float | USD | TAO price in USD |
| `volume_24h` | float | USD | 24h trading volume |
| `market_cap` | float | USD | Market capitalization |
| `circulating_supply` | float | TAO | Circulating supply |
| `total_supply` | float | TAO | Total supply |
| `percent_change_1h` | float | % | 1-hour price change |
| `percent_change_24h` | float | % | 24-hour price change |
| `percent_change_7d` | float | % | 7-day price change |
| `percent_change_30d` | float | % | 30-day price change |

**Size:** ~9 MB, ~65K rows.

**Source:** `api/price/history/v1`

---

## How UnifiedDataLoader Consumes These Files

The loader (`bt_trading_tools/data/loader.py`) reads CSVs and produces aligned `(n_times, n_subnets)` numpy arrays:

1. **OHLCV** — reads `delegation_ohlcv_5m.csv` (primary) or `delegation_ohlcv_hourly.csv` (fallback). Pivots long-format into arrays.
2. **Pool history** — reads `pool_history.csv`. Merges daily data onto the OHLCV time grid via `merge_asof` (backward fill). Derives `k`, `tao_pools`, `alpha_pools` from pool state.
3. **Subnet history** — reads `subnet_history.csv`. Same merge strategy. Derives `daily_emission_tao`.
4. **TAO/USD** — reads `tao_price.csv` (15-min) with fallback to `tao_ohlc_hourly.csv`. Produces `(n_times,)` array.
5. **Startup mode filtering** — subnets in `startup_mode=True` have their OHLCV data dropped (prices are meaningless during bootstrap).
6. **Lifecycle masking** — detects subnet birth/death/rebirth boundaries and masks invalid data windows.

### Key Derived Fields

| Field | Formula | Notes |
|-------|---------|-------|
| `tao_in` | `total_tao / 1e9` | Pool TAO in tokens |
| `alpha_in` | `alpha_in_pool / 1e9` | Pool alpha in tokens |
| `k` | `tao_in * alpha_in` | AMM constant product invariant |
| `tao_pools` | `sqrt(k * price)` | Derived from k and spot price |
| `alpha_pools` | `sqrt(k / price)` | Derived from k and spot price |
| `daily_emission_tao` | `projected_emission × block_reward(t) × 7200` (preferred) or `emission × 7200 / 1e9` (fallback) | Daily TAO injected into the pool. Corrected 2026-04-20 — see subnet_history "Gotcha" above |

### Optional Data Sources

These are loaded when configured but are not part of the core taostats pipeline:

- `dtao_validator_history.csv` — Validator dominance and nominator returns
- Extra feature CSVs — any `(time, netuid, ...)` CSV loaded via `extra_feature_paths` config
- External market data (BTC, ETH, Fear & Greed) from Binance backfills
