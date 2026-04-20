# Trade Log Schema — v1

**Status:** canonical data contract for every bot in the alpha-trading fleet.
**Schema version:** 1 (locked 2026-04-19).
**Home:** `bt_trading_tools.tracking` (public-safe; just a data contract — no strategy logic).

Every bot writes trade and mark-to-market (MTM) records through `bt_trading_tools.tracking` in the format defined here. The `meta-agent` and downstream analysis tools consume these logs. Keeping the format uniform across the fleet is a prerequisite for the signal × bot effect matrix and for apples-to-apples P&L across strategies.

---

## 1. Storage format

- **Format:** JSON Lines (JSONL) — one record per line, UTF-8.
- **Default path:** `{state_dir}/trade_log.jsonl` per bot. Bots may optionally shard by day (`trade_log_YYYY-MM-DD.jsonl`); the reader handles both.
- **Paper vs live:** a single file may contain both; each record carries `is_paper: bool`. Bots MAY separate paper and live into distinct files if they prefer.
- **Append-only.** No in-place edits. Corrections are a new record (use `status: "failed"` for retractions or a downstream audit trail; do not rewrite history).

Every record is a JSON object with the fields below.

---

## 2. Record types

The schema defines three record types, distinguished by the required `record_type` field:

| `record_type` | Meaning | Scope |
|---|---|---|
| `"trade"` | A trade intent that was executed, failed, or partially filled on-chain (or simulated, for paper/backtest). | Per subnet |
| `"mtm_sample"` | A mark-to-market observation of an open position at a bot's evaluation tick. | Per position (subnet-scoped) |
| `"portfolio_snapshot"` | A bot-wide equity snapshot at an evaluation tick. Drives equity curve, drawdown, and Sharpe. | Bot-wide (no subnet) |

Hard contracts:

- Bots MUST emit one `mtm_sample` per currently-held position on every evaluation tick (live, paper, and backtest). Zero-position ticks emit zero samples. Downstream per-position mark-to-market depends on it.
- Bots MUST emit one `portfolio_snapshot` per evaluation tick. One snapshot per tick, always — even when zero positions are held. Downstream equity-curve metrics (drawdown, Sharpe) depend on it.

---

## 3. Fields common to all records

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | `int` | yes | Start at 1. Bump on breaking changes. |
| `bot_id` | `str` | yes | Must match `bot_id` in the bot's manifest. |
| `timestamp` | `str` (ISO 8601 UTC) | yes | Event time. For trades: execution time. For mtm_sample and portfolio_snapshot: the bot's tick time. |
| `record_type` | `"trade"` \| `"mtm_sample"` \| `"portfolio_snapshot"` | yes | See §2. |
| `is_paper` | `bool` | yes | True for paper trading; false for live chain trades. |

### 3.1 Subnet-scoped records (trade, mtm_sample) additionally require:

| Field | Type | Required | Notes |
|---|---|---|---|
| `netuid` | `int` | yes | Subnet identifier. |
| `pool_tao` | `float` | yes | AMM pool TAO at the recorded time (execution for trades; sample time for mtm_sample), in TAO units. |
| `pool_alpha` | `float` | yes | AMM pool alpha at the recorded time, in alpha tokens. |

`portfolio_snapshot` is bot-wide — it does NOT carry `netuid`, `pool_tao`, or `pool_alpha`.

**Mark price is derived, not stored.** Downstream consumers compute `mark_price = pool_tao / pool_alpha` from `pool_tao` and `pool_alpha`. Do not add a redundant `mark_price` field — it would go stale relative to the pool state.

---

## 4. Fields for `record_type: "trade"`

In addition to §3 common fields, trade records require:

### 4.1 Trade identity and status

| Field | Type | Required | Notes |
|---|---|---|---|
| `side` | `"buy"` \| `"sell"` | yes | Trade direction. |
| `status` | `"executed"` \| `"failed"` \| `"partial"` | yes | Execution outcome. `partial` = filled < requested. `failed` = nothing filled. |
| `category` | enum (see §4.6) | yes | Shared vocabulary across fleet for rollup analysis. |
| `intent` | `str` | yes | Bot-local label (e.g., `"grid_buy_level_3"`, `"mr_entry"`). Must be stable within a bot — not free-form per trade. |
| `position_id` | `str` | yes | Linkage key, interpreted per the bot's declared `position_model` (see §6). |

### 4.2 Amounts — executed vs requested

Trades carry both **executed** amounts (what actually happened) and **requested** amounts (what the bot asked for). The gap tells execution quality.

| Field | Type | Required | Notes |
|---|---|---|---|
| `tao_amount` | `float` | yes | Actual TAO moved, unsigned. Sign is inferred from `side`. `0.0` for `status: "failed"`. |
| `alpha_amount` | `float` | yes | Actual alpha moved (acquired on buy; released on sell), unsigned, in alpha tokens. `0.0` for `status: "failed"`. |
| `requested_tao_amount` | `float` \| `null` | buys only | Populated when `side: "buy"`. The bot's requested spend. Equals `tao_amount` for clean executions; greater for partial/failed buys. Null for sells. |
| `requested_alpha_amount` | `float` \| `null` | sells only | Populated when `side: "sell"`. The bot's requested sell size in alpha tokens. Equals `alpha_amount` for clean executions; greater for partial/failed sells. Null for buys. |
| `executed_price` | `float` \| `null` | on fill | TAO per alpha at execution. Null for `status: "failed"`. |

### 4.3 Decision vs execution pool state

The price a bot saw when it decided to trade is not always the pool state at chain execution. The gap is a core slippage signal — capture both.

| Field | Type | Required | Notes |
|---|---|---|---|
| `pool_tao`, `pool_alpha` | `float` | yes | Pool state at **execution**. (From §3 common.) |
| `decision_pool_tao` | `float` | yes | Pool TAO the bot **saw when deciding** to place this order. |
| `decision_pool_alpha` | `float` | yes | Pool alpha the bot saw when deciding. |

For bots with decision ≈ execution simultaneity (e.g., block-subscription bots like BagBot), `decision_pool_*` equals execution pool state. Always populate — downstream code should never have to null-check these.

### 4.4 Fees, slippage tolerance, and yield

Fees are decomposed into three atomic components. A single summed field was considered and rejected — the three components have materially different semantics and are analytically different at downstream attribution time. Consumers that want a sum compute `swap_fee_tao + gas_fee_tao + proxy_fee_tao`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `intended_slippage_tolerance_pct` | `float` \| `null` | yes when live | The bot's configured slippage tolerance (%) at order time. Not reconstructable after the fact. Null permitted for paper/backtest where the concept is moot. |
| `swap_fee_tao` | `float` \| `null` | nullable | Swap-component fee in TAO (pool-rake). Baked into the AMM execution — `executed_price` already reflects it. Stored explicitly for forensic clarity. On sells, alpha-denominated swap fee is converted to TAO via spot at execution. |
| `gas_fee_tao` | `float` \| `null` | nullable | Extrinsic weight+length fee (TAO burned by the chain). Null permitted for paper when fees aren't simulated. |
| `proxy_fee_tao` | `float` \| `null` | nullable | Proxy-wrapping overhead. `0.0` when the extrinsic is not proxy-wrapped; null when not computed. |
| `fee_source` | `"chain"` \| `"fallback"` \| `"receipt"` \| `null` | nullable | Provenance of the fee numbers. `chain` = pre-trade `FeeModel.quote()` against live chain. `fallback` = calibrated constants (chain unavailable). `receipt` = observed from the post-execution chain receipt. |

Realized slippage is NOT stored — it is derived downstream from `decision_pool_*`, `pool_*`, and `executed_price`.

### 4.5 Alpha yield accrual (sell trades only)

Alpha tokens accrue yield to the holder via the chain's emission-and-distribution mechanism. For paper and backtest, yield accrual is simulated via `AlphaYieldModel.accrued_yield(entry_time, alpha_qty, netuid, now)`. For live, it's observed from chain state at entry and sell.

Yield is realized **only at sell** — never continuously mutating position state. The sell trade's `alpha_amount` includes both purchased alpha (with cost basis) and yield-accrued alpha (zero cost basis); `alpha_yield_accrued` decomposes the two so the downstream P&L util attributes them correctly.

| Field | Type | Required | Notes |
|---|---|---|---|
| `alpha_yield_accrued` | `float` (≥ 0) \| `null` | sell-only, nullable | Alpha accrued since entry via yield. Must satisfy `alpha_yield_accrued ≤ alpha_amount`. Null or omitted on buys. |
| `alpha_yield_rate_per_day` | `float` \| `null` | sell-only, nullable | Diagnostic: rate used at time of sell. |
| `alpha_yield_source` | `"taostats"` \| `"chain"` \| `"empirical"` \| `"fallback"` \| `null` | sell-only, nullable | Diagnostic: provenance of the yield rate. |

`compute_pnl` treats `alpha_yield_accrued` on a sell as zero-cost-basis alpha — its revenue is pure profit, and it does not draw down existing cost basis. Trading P&L and yield P&L are therefore cleanly separable at the trade level.

### 4.5 Chain identity

| Field | Type | Required | Notes |
|---|---|---|---|
| `chain_tx_hash` | `str` \| `null` | yes | Chain transaction hash for live trades. Null for paper. |
| `wallet_coldkey` | `str` (SS58) \| `null` | yes | The bot's proxy coldkey SS58 (signer). Null for paper. |
| `wallet_hotkey` | `str` (SS58) \| `null` | yes | The bot's proxy-wallet hotkey SS58 (from the bittensor `Wallet` object tied to the coldkey above). Null for paper. |
| `validator_hotkey` | `str` (SS58) \| `null` | yes | The validator hotkey this stake was delegated to (destination). Null for paper or when not applicable. |

Rationale for three identity fields:
- `wallet_coldkey` is the **signer** — the proxy coldkey that authorized the extrinsic.
- `wallet_hotkey` is the **bot's own** wallet hotkey (paired with the coldkey in a bittensor `Wallet`). Not the same as the validator hotkey.
- `validator_hotkey` is the **delegation destination** — the validator the stake was delegated to. A separate selection per trade.

Chain audit data (which sees coldkey, validator hotkey, netuid, amount — but no `bot_id`) can join back via `(wallet_coldkey, netuid, timestamp)` or `(validator_hotkey, netuid, timestamp)`.

### 4.6 `category` enum

One of:

| Value | Meaning |
|---|---|
| `entry` | Open or add to a position. |
| `exit` | Close or reduce a position for strategy reasons that aren't stop_loss/take_profit. |
| `rebalance` | Position resizing driven by allocation or signal change, not by P&L. |
| `stop_loss` | Triggered by a hard loss threshold. |
| `take_profit` | Triggered by a profit target. |
| `other` | Escape hatch. Use sparingly — if you find yourself reaching for it often, propose a new enum value. |

Shared across the fleet. `intent` (§4.1) is the bot-local refinement.

---

## 5. Fields for `record_type: "mtm_sample"`

In addition to §3 common fields (incl. §3.1 subnet-scoped):

| Field | Type | Required | Notes |
|---|---|---|---|
| `position_id` | `str` | yes | Refers to a specific open position (interpreted per `position_model`). Lets downstream P&L tie samples to positions under pair/level/cycle models. |
| `alpha_yield_accrued` | `float` (≥ 0) \| `null` | nullable | Alpha accrued on this position since entry, at sample time. Lets downstream mark-to-market use effective alpha (`alpha_qty + alpha_yield_accrued`) without re-running the yield model. Null or zero when the yield model is not active for this bot. |

Mark price = `pool_tao / pool_alpha` (derived).

All trade-specific fields (`side`, `status`, `category`, `tao_amount`, `executed_price`, `decision_pool_*`, fees, identity, chain hash, etc.) are omitted. A JSONL parser that tolerates missing keys is fine; a strict reader should check `record_type` first and dispatch.

**Backtest engines must emit `mtm_sample` records** on the same tick schedule as trade evaluations, using simulated pool state. This is a required consumer behavior of `bt_trading_tools.backtest` — it keeps methodology consistent across backtest/paper/live and makes backtest-vs-live P&L comparable.

---

## 5b. Fields for `record_type: "portfolio_snapshot"`

In addition to §3 common fields (bot-wide — NOT §3.1 subnet fields):

| Field | Type | Required | Notes |
|---|---|---|---|
| `capital_tao` | `float` (≥ 0) | yes | Uninvested cash TAO at tick time. |
| `positions_value_tao` | `float` (≥ 0) | yes | Sum of per-position marks at tick time (same pool state the bot used elsewhere). |
| `total_equity_tao` | `float` (≥ 0) | yes | `capital_tao + positions_value_tao`. Stored explicitly to avoid reconstruction ambiguity. |
| `realized_pnl_to_date_tao` | `float` | yes | Cumulative realized P&L since the bot's first snapshot. May be negative. |
| `open_positions_count` | `int` (≥ 0) | yes | Number of held positions at tick time. |

Backtest engines MUST emit `portfolio_snapshot` on the same tick schedule as trade evaluations. Methodology consistency across backtest/paper/live is the whole point — otherwise equity-curve comparability breaks.

Downstream metrics (see `bt_trading_tools.tracking.metrics`):
- `equity_series` — snapshots as a DataFrame
- `drawdown_series` — running-peak drawdown curve
- `max_drawdown` — peak-to-trough (value and %)
- `sharpe` — annualized from tick-to-tick returns (caller supplies `periods_per_year`)

---

## 6. `position_id` semantics by model

`position_id` is interpreted according to the bot's `position_model` declared in its manifest (`bots/<bot>/manifest.yaml`). The trade log schema is agnostic to which model a bot uses; it only requires `position_id` to be a consistent string under that model.

| `position_model` | `position_id` convention | Example |
|---|---|---|
| `pair` | UUID generated at open, referenced at close. Exactly one buy trade and one sell trade share a position_id. | `"a9c1...-4d7e"` |
| `level` | Stable per-grid-level identifier. Many buys/sells share the same id as the level accumulates and empties. | `"L3"` |
| `inventory` | Subnet-level key when one position per subnet (`sn<netuid>`). UUID per tranche if the bot opens overlapping positions in the same subnet. | `"sn107"` or `"sn107_t3"` |
| `cycle` | Per-emission-cycle identifier. All trades within a cycle share. | `"sn107_cycle_2026-03"` |

Each bot picks one model and sticks to it. Downstream P&L uses `position_model` from the manifest to decide accounting (FIFO for inventory, per-pair for pair, per-level for grid, per-cycle for cycle).

---

## 7. Async logging contract

`bt_trading_tools.tracking.log_trade(record)` and `log_mtm(record)` MUST be non-blocking from the bot's perspective. BagBot's block-subscription loop and other latency-sensitive bots cannot afford disk I/O in their hot path.

**Contract:**

- `log_trade(dict)` and `log_mtm(dict)` enqueue the raw dict onto an in-process queue and return immediately. The hot path pays ~queue-put cost, nothing more.
- A background flusher (thread for sync callers, asyncio task for async callers) drains the queue, validates each record against the schema (pydantic), and writes the line to the JSONL file.
- Schema validation happens **on the flusher, not on enqueue** — the hot path never pays validation cost.
- On validation failure, the raw original dict plus the validation error details are appended to a sibling `errors.jsonl` file. A counter (`flush_errors_since_start`) is exposed so the file cannot silently grow unbounded; long-running bots should log this counter to their event log periodically.
- On `SIGTERM` and clean shutdown, the flusher drains the queue before exiting. Records in flight are not lost.
- Both sync (`queue.Queue`) and async (`asyncio.Queue`) wrappers are provided. Thread-safe enqueue.

This is a first-class contract of `bt_trading_tools.tracking`, not an implementation detail.

---

## 8. Versioning policy

- `schema_version: 1` is the initial locked contract (2026-04-19).
- Adding a new optional field is NOT a breaking change — it does not bump `schema_version`. Readers MUST tolerate unknown fields.
- Adding a new required field, renaming a field, changing a field's type, or changing an enum's semantics IS a breaking change — bumps `schema_version` to 2.
- Each bumped version ships with a readable migration note in this file and a migration helper in `bt_trading_tools.tracking` where feasible.
- Readers inspect `schema_version` and dispatch accordingly. Unknown future versions should log a warning and attempt best-effort parsing, not crash.

---

## 9. Storage-cost note

At 5-minute evaluation cadence × 20 held positions × 6 months, `mtm_sample` records dominate (≈ 1M/bot, ~50–200 MB JSONL). `portfolio_snapshot` adds 1 row/tick (≈ 50K/bot at 5-min for 6mo — trivial). `trade` records are orders of magnitude fewer. Not a blocker at v1; document here so future optimization (Parquet rollup, coarser MTM cadence, periodic archival) is an informed decision rather than a panic.

---

## 10. Explicit exclusions (v1)

- **Universe-wide marks on unheld subnets.** Trade log stays tied to the bot's actual positions. Use `data/taostats/delegation_full.csv` (tick-level ground truth) for market-wide analysis.
- **Realized / unrealized P&L on the trade record itself.** Not a stored field. Computed downstream by a shared utility in `bt_trading_tools` that respects each bot's declared `position_model`. A bot may optionally record `bot_reported_pnl_tao` on a sell trade for internal dashboards, but it is informational — the canonical number comes from downstream. `realized_pnl_to_date_tao` on `portfolio_snapshot` is explicitly a running total, not a per-trade attribution.
- **Raw chain-receipt forensic data.** Fee receipts (raw extrinsic responses, observed fees, chain-parse errors) live in a separate `fee_receipts.jsonl` log emitted by TradeExecutor, not in this trade log. The two join on `(bot_id, chain_tx_hash)`. See `bt_trading_tools.tracking.FeeReceipt`.

---

## 11. Minimal worked examples

### 11.1 Live buy, clean execution

```json
{"schema_version": 1, "bot_id": "autobot", "timestamp": "2026-04-19T14:23:07Z",
 "record_type": "trade", "netuid": 107, "pool_tao": 13421.5, "pool_alpha": 2685432.1,
 "is_paper": false,
 "side": "buy", "status": "executed", "category": "entry", "intent": "mr_entry",
 "position_id": "sn107",
 "tao_amount": 0.5, "alpha_amount": 99.83, "requested_tao_amount": 0.5, "requested_alpha_amount": null,
 "executed_price": 0.005009,
 "decision_pool_tao": 13420.9, "decision_pool_alpha": 2685551.0,
 "intended_slippage_tolerance_pct": 1.0,
 "swap_fee_tao": 0.000252, "gas_fee_tao": 8.4e-6, "proxy_fee_tao": 1.0e-6, "fee_source": "chain",
 "chain_tx_hash": "0xabc...", "wallet_coldkey": "5Gproxy...", "wallet_hotkey": "5Gbothk...",
 "validator_hotkey": "5Gvalidator..."}
```

### 11.2 Live sell, partial fill, with yield accrued

```json
{"schema_version": 1, "bot_id": "autobot", "timestamp": "2026-04-19T15:10:22Z",
 "record_type": "trade", "netuid": 107, "pool_tao": 13480.2, "pool_alpha": 2673108.9,
 "is_paper": false,
 "side": "sell", "status": "partial", "category": "take_profit", "intent": "mr_exit",
 "position_id": "sn107",
 "tao_amount": 0.41, "alpha_amount": 79.9, "requested_tao_amount": null, "requested_alpha_amount": 99.83,
 "executed_price": 0.005131,
 "decision_pool_tao": 13478.5, "decision_pool_alpha": 2673450.0,
 "intended_slippage_tolerance_pct": 1.5,
 "swap_fee_tao": 0.000207, "gas_fee_tao": 8.4e-6, "proxy_fee_tao": 1.0e-6, "fee_source": "receipt",
 "alpha_yield_accrued": 1.5, "alpha_yield_rate_per_day": 0.003, "alpha_yield_source": "taostats",
 "chain_tx_hash": "0xdef...", "wallet_coldkey": "5Gproxy...", "wallet_hotkey": "5Gbothk...",
 "validator_hotkey": "5Gvalidator..."}
```

### 11.3 Paper MTM sample

```json
{"schema_version": 1, "bot_id": "emissions-drought-bot", "timestamp": "2026-04-19T16:00:00Z",
 "record_type": "mtm_sample", "netuid": 52, "pool_tao": 8742.3, "pool_alpha": 1923847.5,
 "is_paper": true,
 "position_id": "sn52_cycle_2026-04"}
```

### 11.3b Portfolio snapshot

```json
{"schema_version": 1, "bot_id": "emissions-drought-bot", "timestamp": "2026-04-19T16:00:00Z",
 "record_type": "portfolio_snapshot", "is_paper": true,
 "capital_tao": 42.5, "positions_value_tao": 57.2, "total_equity_tao": 99.7,
 "realized_pnl_to_date_tao": -0.3, "open_positions_count": 4}
```

### 11.4 Live trade that failed to execute

```json
{"schema_version": 1, "bot_id": "bagbot", "timestamp": "2026-04-19T17:02:11Z",
 "record_type": "trade", "netuid": 18, "pool_tao": 4120.9, "pool_alpha": 830121.4,
 "is_paper": false,
 "side": "buy", "status": "failed", "category": "entry", "intent": "grid_buy_level_5",
 "position_id": "L5",
 "tao_amount": 0.0, "alpha_amount": 0.0, "requested_tao_amount": 0.25, "requested_alpha_amount": null,
 "executed_price": null,
 "decision_pool_tao": 4121.1, "decision_pool_alpha": 830080.6,
 "intended_slippage_tolerance_pct": 0.5,
 "swap_fee_tao": null, "gas_fee_tao": 8.4e-6, "proxy_fee_tao": 1.0e-6, "fee_source": "receipt",
 "chain_tx_hash": null, "wallet_coldkey": "5Gproxy...", "wallet_hotkey": "5Gbothk...",
 "validator_hotkey": "5Gvalidator..."}
```

*(Failed buys still incur extrinsic and proxy fees — the chain charges for the attempt. `swap_fee_tao` is null because no swap occurred.)*
