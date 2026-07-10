"""
Trade log schema v2 — canonical data contract.

See bt-trading-tools/docs/trade_log_schema.md for the full spec.

Three record types live in a single JSONL log per bot:
    - `trade`              An executed, failed, or partially-filled order.
    - `mtm_sample`         A mark-to-market observation of an open position.
    - `portfolio_snapshot` A per-tick bot-wide equity snapshot
                           (drives equity curve, drawdown, Sharpe).

v2 (2026-04-23) promoted from extras to canonical on TradeRecord:
    `failure_reason`, `latency_ms`, `execution_mode`, and three
    meta-agent booleans (`meta_circuit_breaker_active`,
    `meta_novelty_gate_active`, `meta_stale_inputs`). The same three
    booleans are added to PortfolioSnapshot. MTMSample is unchanged.

v2 additive (2026-07-09): `meta_circuit_breaker_reasons: list[str]` added
    to TradeRecord and PortfolioSnapshot so a CQI audit can see WHY the
    breaker fired at emission time (not just that it did). Defaults to
    the empty list, so pre-2026-07-09 records that omit the field still
    parse cleanly. No SCHEMA_VERSION bump.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = 2


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Status(str, Enum):
    EXECUTED = "executed"
    FAILED = "failed"
    PARTIAL = "partial"


class Category(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    OTHER = "other"


class FeeSource(str, Enum):
    """Provenance of fee numbers on a trade record.

    - chain:    quoted from the chain via sim_swap / get_payment_info
    - fallback: computed from calibrated constants (no chain access)
    - receipt:  observed post-execution from the TradeExecutor fee receipt
    """
    CHAIN = "chain"
    FALLBACK = "fallback"
    RECEIPT = "receipt"


class AlphaYieldSource(str, Enum):
    """Provenance of the per-subnet alpha yield rate."""
    TAOSTATS = "taostats"
    CHAIN = "chain"
    EMPIRICAL = "empirical"
    VALIDATOR_CACHE = "validator_cache"
    FALLBACK = "fallback"


class ExecutionMode(str, Enum):
    """Execution context that produced a trade record.

    - csv_only:  paper bot with CSV pool history + realism layer.
    - hybrid:    paper bot with live pool-state fetch + realism layer.
    - live:      real chain-signing bot.
    - replay:    historical-replay harness (re-runs paper bot against past pools).
    - backtest:  backtest engine (simulated pools, simulated ticks).

    Optional in v2 schema (Commit 1a); promoted to required in v2-enforced
    (Commit 1b) after every trade-emission path is audited.
    """
    CSV_ONLY = "csv_only"
    HYBRID = "hybrid"
    LIVE = "live"
    REPLAY = "replay"
    BACKTEST = "backtest"


# ── Base classes ──────────────────────────────────────────────────

class _RootRecord(BaseModel):
    """Fields common to every record_type, including bot-wide snapshots."""
    model_config = ConfigDict(extra="allow")  # forward-compat + strategy-specific extras

    schema_version: int = Field(ge=1)
    bot_id: str
    timestamp: str
    is_paper: bool


class _SubnetRecord(_RootRecord):
    """Records that are scoped to a specific subnet's AMM pool."""
    netuid: int = Field(ge=0)
    pool_tao: float = Field(ge=0)
    pool_alpha: float = Field(ge=0)


# ── Record types ──────────────────────────────────────────────────

class TradeRecord(_SubnetRecord):
    record_type: Literal["trade"]

    side: Side
    status: Status
    category: Category
    intent: str
    position_id: str

    tao_amount: float = Field(ge=0)
    alpha_amount: float = Field(ge=0)
    requested_tao_amount: Optional[float] = Field(default=None, ge=0)
    requested_alpha_amount: Optional[float] = Field(default=None, ge=0)
    executed_price: Optional[float] = Field(default=None, ge=0)

    decision_pool_tao: float = Field(ge=0)
    decision_pool_alpha: float = Field(ge=0)

    intended_slippage_tolerance_pct: Optional[float] = Field(default=None, ge=0)

    # Fees — three atomic components instead of a single network_fee_tao.
    # See trade_log_schema.md §4.4 for semantics.
    swap_fee_tao: Optional[float] = Field(default=None, ge=0)
    gas_fee_tao: Optional[float] = Field(default=None, ge=0)
    proxy_fee_tao: Optional[float] = Field(default=None, ge=0)
    fee_source: Optional[FeeSource] = None

    # Alpha yield accrued since entry. Populated on sell trades when the
    # bot runs an AlphaYieldModel; null on buys and when yield isn't tracked.
    alpha_yield_accrued: Optional[float] = Field(default=None, ge=0)
    alpha_yield_rate_per_day: Optional[float] = None
    alpha_yield_source: Optional[AlphaYieldSource] = None

    chain_tx_hash: Optional[str] = None
    wallet_coldkey: Optional[str] = None
    wallet_hotkey: Optional[str] = None
    validator_hotkey: Optional[str] = None

    # v2 fields (2026-04-23). See trade_log_schema.md §4.7–§4.9.
    failure_reason: Optional[str] = None
    latency_ms: Optional[int] = Field(default=None, ge=0)
    execution_mode: Optional[ExecutionMode] = None
    meta_circuit_breaker_active: bool = False
    meta_novelty_gate_active: bool = False
    meta_stale_inputs: bool = False
    # v2 additive (2026-07-09): breaker reason list at emission time.
    # Empty when the breaker was inactive; also empty for pre-2026-07-09 records
    # that never carried the field. See trade_log_schema.md §4.7.
    meta_circuit_breaker_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_requested_pairing(self):
        if self.side == Side.BUY:
            if self.requested_tao_amount is None:
                raise ValueError("requested_tao_amount is required when side='buy'")
            if self.requested_alpha_amount is not None:
                raise ValueError("requested_alpha_amount must be null when side='buy'")
        else:  # SELL
            if self.requested_alpha_amount is None:
                raise ValueError("requested_alpha_amount is required when side='sell'")
            if self.requested_tao_amount is not None:
                raise ValueError("requested_tao_amount must be null when side='sell'")
        return self

    @model_validator(mode="after")
    def _validate_status_fields(self):
        if self.status == Status.FAILED:
            if self.tao_amount != 0.0:
                raise ValueError("tao_amount must be 0 when status='failed'")
            if self.alpha_amount != 0.0:
                raise ValueError("alpha_amount must be 0 when status='failed'")
            if self.executed_price is not None:
                raise ValueError("executed_price must be null when status='failed'")
        else:  # executed or partial
            if self.executed_price is None:
                raise ValueError(
                    f"executed_price is required when status='{self.status.value}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_failure_reason(self):
        """v2 enforcement: status='failed' records must carry a
        failure_reason. Gated on schema_version >= 2 so v1 records
        (which never had this field) continue to parse cleanly.

        Current canonical emission paths (all in paper_base.py as of
        2026-04-23) already set failure_reason on every failed path:
            - simulate_execution layer 1 (random_reject)
            - simulate_execution layer 3 (rate_tolerance)
            - _orphan_pending_order (orphaned)

        Live-bot emission via BaseBotLoop / TradeExecutor writes to
        the legacy SQLite trade_log, not the canonical
        TradeLogWriter — so this validator never fires on live paths
        today. If a future commit adds canonical live-side emission,
        this validator will force failure_reason to be set at that
        commit time.
        """
        if (self.schema_version >= 2
                and self.status == Status.FAILED
                and self.failure_reason is None):
            raise ValueError(
                "failure_reason is required when status='failed' "
                "under schema_version >= 2"
            )
        return self

    @model_validator(mode="after")
    def _validate_execution_mode_required(self):
        """v2 enforcement: execution_mode must be set. Gated on
        schema_version >= 2 so v1 records continue to parse.
        """
        if self.schema_version >= 2 and self.execution_mode is None:
            raise ValueError(
                "execution_mode is required under schema_version >= 2 "
                "(one of: csv_only, hybrid, live, replay, backtest)"
            )
        return self


class MTMSample(_SubnetRecord):
    record_type: Literal["mtm_sample"]
    position_id: str
    # Optional: alpha accrued on this position since entry, at sample time.
    # Lets downstream consumers mark positions at effective alpha without
    # re-running the yield model. Null or zero when the yield model is
    # not active for this bot.
    alpha_yield_accrued: Optional[float] = Field(default=None, ge=0)


class PortfolioSnapshot(_RootRecord):
    """Bot-wide equity snapshot at a tick. Drives equity curve, drawdown, Sharpe.

    Not subnet-scoped — no netuid or pool fields. Emit once per evaluation
    tick from the bot framework (PaperBotBase / BaseBotLoop).
    """
    record_type: Literal["portfolio_snapshot"]
    capital_tao: float = Field(ge=0)              # cash available
    positions_value_tao: float = Field(ge=0)      # sum of position marks at tick
    total_equity_tao: float = Field(ge=0)         # capital + positions_value
    realized_pnl_to_date_tao: float                # can be negative
    open_positions_count: int = Field(ge=0)

    # v2 fields (2026-04-23): meta-agent state at snapshot time. Lets the
    # fleet-dashboard audit whether bots honored halt/novelty/stale signals
    # on each tick without cross-joining to allocation.json by timestamp.
    meta_circuit_breaker_active: bool = False
    meta_novelty_gate_active: bool = False
    meta_stale_inputs: bool = False
    # v2 additive (2026-07-09): mirrored from TradeRecord for the same reason
    # — the snapshot is what the CQI dashboard aggregates per tick.
    meta_circuit_breaker_reasons: list[str] = Field(default_factory=list)


Record = Annotated[
    Union[TradeRecord, MTMSample, PortfolioSnapshot],
    Field(discriminator="record_type"),
]
