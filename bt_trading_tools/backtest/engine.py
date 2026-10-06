"""
BacktestEngine — generic event-driven backtester for Bittensor strategies.

Data-agnostic: the caller provides a sequence of TickData and a Strategy.
The engine handles AMM execution, position tracking, equity recording,
and optionally writes to the same TradeLog / PortfolioLog used by live bots.

Known-bug prevention (see backtest_bugs_mar20.md):
  - Entry price uses cost-weighted average, not overwrite (Bug #1)
  - Entry timestamp preserved on accumulation (Bug #2)
  - Watch timeout is strategy-level, engine uses tick-based delay (Bug #3)
  - Order.limit_price prevents TP overshoot on delayed execution (Bug #4)
  - Delayed orders correctly update capital (Bug #5-like)
  - Positions passed to strategy as a shallow copy (defensive)
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from bt_trading_tools.amm import amm_buy, amm_sell, slippage_pct, spot_price
from bt_trading_tools.backtest.stats import BacktestStats, compute_stats
from bt_trading_tools.backtest.types import Order, Position, Strategy, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig, RealismSimulator
from bt_trading_tools import ledger

if TYPE_CHECKING:
    from bt_trading_tools.alpha_yield import AlphaYieldModel
    from bt_trading_tools.fees import FeeModel


@dataclass
class BacktestResults:
    """Output of a backtest run."""
    stats: BacktestStats
    trades: list[dict]
    equity_curve: list[dict]
    positions_at_end: dict[int, Position]
    # True when the ticks carried subnet identity (SubnetTick.generation), so
    # positions in a re-registered netuid were closed at the deregistration
    # refund. False means the run could not tell subnets in the same slot
    # apart; see bt_trading_tools.utils.lifecycle.stamp_generations.
    identity_guard_active: bool = False
    # Reserve-versus-price consistency of the orders the strategy placed:
    # how many orders were checked, and how many were placed against a pool
    # whose reserves imply a price more than RESERVE_GAP_TOL away from the
    # tick's own price. The engine fills from the RESERVES, not from
    # ``SubnetTick.price``, so a large share here means the ticks pair a
    # price with stale reserves (for example hourly closes with daily pool
    # snapshots) and fills are mispriced. See docs/known_traps.md.
    orders_checked: int = 0
    orders_on_inconsistent_reserves: int = 0


# A run is flagged when more than this share of its orders (at least
# RESERVE_GAP_MIN_ORDERS of them) sit on reserves whose implied price differs
# from the tick price by more than RESERVE_GAP_TOL. On the SDK feed the
# share is under 1%; on hourly closes paired with daily reserves it is large
# around surges (measured 2026-10: pool-implied price averaged 0.92 x the
# tick price at surge entries).
RESERVE_GAP_TOL = 0.05
RESERVE_GAP_SHARE_WARN = 0.10
RESERVE_GAP_MIN_ORDERS = 10

# Same-seed trap: every engine built with the same ``realism_rng_seed``
# replays the same realism draws. Averaging many near-single-trade runs under
# one seed therefore averages one draw, not many. Heuristic guard: after this
# many runs with at most FEW_FILL_MAX fills each under one seed in a process,
# warn once for that seed (see docs/known_traps.md).
SAME_SEED_WARN_AFTER = 25
FEW_FILL_MAX = 4
_SEED_FEW_FILL_RUNS: dict[int, int] = {}
_SEED_WARNED: set[int] = set()


# Default flat-fee constants — used only when fee_model=None and no chain
# client is wired. Match the calibrated FeeModel fallbacks (2026-05-04
# empirical refit from 62 chain-source quotes; see bt_trading_tools/fees.py
# and docs/fees_and_yield_design.md §1).
DEFAULT_SWAP_FEE_RATE = 33 / 65535   # ≈ 5.04e-4, default mechanism-1 FeeRate
DEFAULT_GAS_FEE_TAO = 1.18e-3        # mean of buy (1.34e-3) + sell (9.15e-4)


class BacktestEngine:
    """Generic backtesting engine.

    Args:
        capital: Starting capital in TAO.
        bot_name: Name for tracking logs (if using TradeLog/PortfolioLog).
        swap_fee_rate: Proportional swap fee (default 0.05%).
        gas_fee_tao: Fixed gas fee per transaction (default 0.00001 TAO).
        max_pool_pct: Optional cap on order size as a fraction of pool depth
            (TAO side for buys, alpha side for sells). Default None = no
            cap. A pool cap is a per-bot choice; until 2026-10-01 this
            defaulted to 0.05 and silently capped every bot's trades.
        execution_delay: Number of ticks to delay execution (0 = instant).
        trade_log: Optional TradeLog instance for recording trades.
        portfolio_log: Optional PortfolioLog instance for equity curve.
        ticks_per_year: For Sharpe annualization. 365=daily, 8760=hourly,
            2628000=12-second blocks.

    Usage::

        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, strategy)
        print(results.stats)
    """

    def __init__(
        self,
        capital: float = 100.0,
        bot_name: str = "backtest",
        swap_fee_rate: float = DEFAULT_SWAP_FEE_RATE,
        gas_fee_tao: float = DEFAULT_GAS_FEE_TAO,
        max_pool_pct: float | None = None,
        execution_delay: int = 0,
        trade_log: Any = None,
        portfolio_log: Any = None,
        ticks_per_year: float = 365.0,
        fee_model: FeeModel | None = None,
        yield_model: AlphaYieldModel | None = None,
        uses_proxy: bool = True,
        realism_config: RealismConfig | None = None,
        realism_rng_seed: Optional[int] = 0,
        pool_safety_checker: Any = None,
    ):
        """
        Extra kwargs (back-compat preserved when both are None):

            fee_model:  When provided, replaces the flat
                ``swap_fee_rate + gas_fee_tao`` formula with a deterministic
                per-trade FeeModel.quote(...) call. Trade records gain
                ``swap_fee_tao`` / ``gas_fee_tao`` / ``proxy_fee_tao`` /
                ``fee_source`` fields.
            yield_model: When provided, applies alpha yield accretion to
                sells (``effective_alpha_qty``) and to MTM equity. Pure,
                idempotent; backtest-time simulated timestamps are used
                as ``now`` so reruns are deterministic. Trade records
                (sells) gain ``alpha_yield_accrued``.
            uses_proxy: When True (default), FeeModel quotes include the
                ``proxy_fee_tao`` component. The fleet always uses proxy
                wallets in production, so the backtest defaults to True
                — matches paper/live execution friction.
            realism_config: ``RealismConfig`` controlling the five
                realism layers (random_reject failures, latency, slippage
                noise, rate-tolerance breach, partial fills). When None,
                a default ``RealismConfig()`` (enabled=True with calibrated
                values matching paper bots) is used. **Defaults to ON
                because backtest friction must mirror paper / live.** Pass
                ``RealismConfig(enabled=False)`` for non-realism unit
                tests.
            realism_rng_seed: Seed for the realism RNG. Defaults to 0 so
                backtest runs are reproducible — same data + same strategy
                = same realised outcomes. Pass None for nondeterministic.
                **TRAP: the seed is per ENGINE, not per trade.** Every new
                engine with the same seed replays the same draws, so the
                realism noise of N independent one-trade runs is one draw
                counted N times, not N draws. Measured 2026-10: for a 1 TAO
                round trip on a deep pool at unchanged price, seed 0 gave
                -0.01% while 200 seeds averaged -0.45% (sd 0.28%), so
                one-trade runs under the default seed overstate returns by
                about 0.4 percentage points. A single long run with many
                trades is not affected (draws advance trade to trade). When
                you run many independent windows or trades, give each its
                own seed: use ``bt_trading_tools.backtest.panel.
                seeded_engine_factory`` or ``panel_forward_returns`` (which
                does this by default). See docs/known_traps.md.
            pool_safety_checker: Optional ``PoolSafetyChecker`` (from
                ``bt_strategy.regime.pool_safety``) that drops "buy"
                orders for subnets flagged by the slow_drain / fast_rug /
                A>S filters — same behaviour as paper bots get via
                PaperBotBase. Sells are never blocked. Defaults to None
                (no filter); pass an instance to align backtest filtering
                with paper. Construct via:
                ``PoolSafetyChecker(DataFramePoolHistoryProvider(pool_history_df))``.

        Compared to pre-integration: realism layers + proxy fee are now
        on by default. Existing tests that assert specific deterministic
        outcomes may need ``realism_config=RealismConfig(enabled=False)``
        if they're testing engine mechanics rather than realism behaviour.
        """
        self.starting_capital = capital
        self.bot_name = bot_name
        self.swap_fee_rate = swap_fee_rate
        self.gas_fee_tao = gas_fee_tao
        self.max_pool_pct = max_pool_pct
        self.execution_delay = execution_delay
        self.trade_log = trade_log
        self.portfolio_log = portfolio_log
        self.ticks_per_year = ticks_per_year
        self.fee_model = fee_model
        # Yield is on by default. When the caller doesn't pass a yield_model,
        # use the env-driven default cascade (TAOSTATS_API_KEY, BT_NETWORK,
        # TAOSTATS_DATA_DIR). In test environments with no env vars set, the
        # cascade fast-fails to zero — same effective behavior as the
        # historical default, but safe-by-default in production usage.
        if yield_model is None:
            from bt_trading_tools.alpha_yield import build_default_yield_model
            self.yield_model = build_default_yield_model()
        else:
            self.yield_model = yield_model
        self.uses_proxy = uses_proxy
        self.realism_rng_seed = realism_rng_seed
        self._realism = RealismSimulator(
            realism_config or RealismConfig(),
            rng_seed=realism_rng_seed,
        )
        self.pool_safety_checker = pool_safety_checker
        self._orders_checked = 0
        self._orders_gap = 0

    def run(
        self,
        ticks: list[TickData],
        strategy: Strategy,
    ) -> BacktestResults:
        """Run the backtest.

        Args:
            ticks: Chronologically ordered market data. Each tick has
                subnets dict with price, pool depth, and signals. Stamp
                ``SubnetTick.generation`` (see
                ``bt_trading_tools.utils.lifecycle.stamp_generations``) so the
                engine can tell successive subnets in one netuid apart.
            strategy: Implements Strategy protocol (on_tick method).

        Returns:
            BacktestResults with stats, trades, equity curve.
        """
        capital = self.starting_capital
        self._orders_checked = 0
        self._orders_gap = 0
        positions: dict[int, Position] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        # Pending delayed orders: (execute_at_tick_idx, order, decision_tick_snapshot)
        # The snapshot is the SubnetTick at decision time, used as fallback
        # price when the execution tick has no data for this subnet (nobody
        # traded between decision and execution, so the pool did not change).
        pending_orders: list[tuple[int, Order, SubnetTick | None]] = []
        # Last-seen state per (netuid, generation): (SubnetTick, timestamp).
        # Sparse data means a subnet can be absent from a tick although it
        # traded recently; between transactions AMM pool state does not
        # change, so the last-seen state IS ground truth. Used for valuation
        # of absent subnets, the end-of-data close, and the deregistration
        # refund (keyed by generation so a re-registration cannot overwrite
        # the dead subnet's last state).
        self._last_seen: dict[tuple[int, int | None], tuple[SubnetTick, int]] = {}

        identity_known = any(
            st.generation is not None for t in ticks for st in t.subnets.values()
        )
        if ticks and not identity_known:
            warnings.warn(
                "BacktestEngine: ticks carry no SubnetTick.generation, so a "
                "position held across a netuid re-registration will be valued "
                "and sold against the NEW subnet's pool. Stamp ticks with "
                "bt_trading_tools.utils.lifecycle.stamp_generations.",
                stacklevel=2,
            )

        for tick_idx, tick in enumerate(ticks):
            # ── Close positions whose subnet was replaced ───────────
            # Must run BEFORE last_seen is updated with this tick, and before
            # any order, so nothing is ever priced against the new subnet.
            capital = self._close_reregistered(tick, positions, trades, capital)

            for _netuid, _st in tick.subnets.items():
                self._last_seen[(_netuid, _st.generation)] = (_st, tick.timestamp)
            # ── Refresh pool-safety checker once per tick (clears its
            # per-subnet cache so subsequent check() calls see fresh data) ─
            if self.pool_safety_checker is not None:
                try:
                    self.pool_safety_checker.refresh()
                except Exception:
                    pass  # provider's refresh() may no-op or raise; non-fatal
            # ── Execute delayed orders ───────────────────────────
            if self.execution_delay > 0:
                ready = [
                    (idx, order, snap) for idx, order, snap in pending_orders
                    if idx <= tick_idx
                ]
                pending_orders = [
                    (idx, order, snap) for idx, order, snap in pending_orders
                    if idx > tick_idx
                ]
                for _, order, decision_snap in ready:
                    exec_tick = tick
                    if order.netuid not in tick.subnets and decision_snap is not None:
                        exec_tick = TickData(
                            timestamp=tick.timestamp,
                            subnets={**tick.subnets, order.netuid: decision_snap},
                            global_signals=tick.global_signals,
                        )
                    result = self._execute_order(
                        order, exec_tick, capital, positions, trades,
                    )
                    if result is not None:
                        capital = result

            # ── Ask strategy for orders ──────────────────────────
            pv = self._portfolio_value(capital, positions, tick)
            # Shallow copies so the strategy cannot corrupt engine state.
            pos_snapshot = {k: copy.copy(v) for k, v in positions.items()}
            orders = strategy.on_tick(tick, pos_snapshot, capital, pv)

            for order in orders:
                if self.execution_delay > 0:
                    snap = tick.subnets.get(order.netuid)
                    pending_orders.append(
                        (tick_idx + self.execution_delay, order, snap)
                    )
                else:
                    result = self._execute_order(
                        order, tick, capital, positions, trades,
                    )
                    if result is not None:
                        capital = result

            # ── Record equity ────────────────────────────────────
            pv = self._portfolio_value(capital, positions, tick)
            eq_point = {
                "timestamp": tick.timestamp,
                "capital": round(capital, 6),
                "positions_value": round(pv - capital, 6),
                "total_equity": round(pv, 6),
                "n_positions": len(positions),
            }
            equity_curve.append(eq_point)

            if self.portfolio_log:
                self.portfolio_log.record(
                    total_value=pv, cash=capital,
                    staked_value=pv - capital,
                    n_positions=len(positions),
                    timestamp=tick.timestamp,
                )

        # ── Close remaining positions at their last observed state ──
        if ticks:
            last_tick = ticks[-1]
            for netuid in list(positions.keys()):
                pos = positions[netuid]
                st, exit_source = self._state_for(pos, last_tick)
                eff_alpha = pos.alpha_qty + ledger.accrued_yield(
                    pos, last_tick.timestamp, self._yield_fn)
                if st is not None and st.tao_pool > 0 and st.alpha_pool > 0:
                    fee, fee_components = self._sell_fee(
                        netuid, eff_alpha, st.tao_pool, st.alpha_pool,
                    )
                    tao_received = ledger.liquidation_value(
                        eff_alpha, st.tao_pool, st.alpha_pool, fee)
                else:
                    # Defensive: an open position implies at least its entry
                    # tick was seen, so this needs zero pools or no state.
                    fee, fee_components = 0.0, {}
                    tao_received = eff_alpha * (st.price if st else pos.entry_price)
                    exit_source = "entry_fallback" if st is None else exit_source
                rec = ledger.book_sell(
                    positions, netuid, eff_alpha, tao_received, fee,
                    last_tick.timestamp, yield_fn=self._yield_fn,
                )
                capital += tao_received
                trade = {
                    "netuid": netuid,
                    "exit_time": last_tick.timestamp,
                    "exit_price": tao_received / eff_alpha if eff_alpha > 0 else 0,
                    "reason": "end_of_data",
                    "exit_source": exit_source,
                    **rec,
                    **fee_components,
                }
                trades.append(trade)
                self._record_trade(trade, "sell")

        stats = compute_stats(
            trades, equity_curve, self.starting_capital, self.ticks_per_year,
        )

        self._warn_known_traps(trades)

        return BacktestResults(
            stats=stats,
            trades=trades,
            equity_curve=equity_curve,
            positions_at_end=positions,
            identity_guard_active=identity_known,
            orders_checked=self._orders_checked,
            orders_on_inconsistent_reserves=self._orders_gap,
        )

    def _warn_known_traps(self, trades: list[dict]) -> None:
        """Warn about two measured, silent ways a backtest result goes wrong.

        1. Stale reserves: a large share of orders were placed on reserves whose
           implied price differs from the tick price (fills use the reserves).
        2. Same seed, few fills: many near-single-trade runs under one
           realism seed replay one set of random draws.
        Both are heuristics that only warn; nothing is changed.
        """
        if (self._orders_checked >= RESERVE_GAP_MIN_ORDERS
                and self._orders_gap / self._orders_checked > RESERVE_GAP_SHARE_WARN):
            warnings.warn(
                f"BacktestEngine: {self._orders_gap} of {self._orders_checked} orders "
                f"({100 * self._orders_gap / self._orders_checked:.0f}%) were placed on pool "
                f"reserves whose implied price differs from the tick price by more than "
                f"{100 * RESERVE_GAP_TOL:.0f}%. The engine fills from the reserves, so these "
                "fills are mispriced (typical cause: hourly prices paired with daily reserves, "
                "as in load_parquet_ticks with reserves='raw'). Use contemporaneous reserves "
                "(SDK ticks); load_parquet_ticks(reserves='rescale') matches the price but is not "
                "validated. See docs/known_traps.md.",
                RuntimeWarning, stacklevel=3,
            )
        seed = self.realism_rng_seed
        if seed is not None:
            fills = sum(1 for t in trades if t.get("status") != "failed")
            if fills <= FEW_FILL_MAX:
                n = _SEED_FEW_FILL_RUNS.get(seed, 0) + 1
                _SEED_FEW_FILL_RUNS[seed] = n
                if n >= SAME_SEED_WARN_AFTER and seed not in _SEED_WARNED:
                    _SEED_WARNED.add(seed)
                    warnings.warn(
                        f"BacktestEngine: {n} runs with at most {FEW_FILL_MAX} fills each have used "
                        f"realism_rng_seed={seed} in this process. Each engine replays the same "
                        "realism draws, so averaging such runs averages one draw, not many, and "
                        "biases returns (measured about +0.4 percentage points per one-trade run "
                        "under seed 0). Give each independent run its own seed: "
                        "bt_trading_tools.backtest.panel.seeded_engine_factory, or use "
                        "panel_forward_returns. See docs/known_traps.md.",
                        RuntimeWarning, stacklevel=3,
                    )

    # ── Subnet identity ──────────────────────────────────────────

    def _state_for(
        self, pos: Position, tick: TickData,
    ) -> tuple[SubnetTick | None, str]:
        """Pool state to value ``pos`` at: this tick if it shows the same
        subnet, else the last state seen for the position's generation."""
        st = tick.subnets.get(pos.netuid)
        if st is not None and (
            st.generation is None or pos.generation is None
            or st.generation == pos.generation
        ):
            return st, "last_tick"
        seen = self._last_seen.get((pos.netuid, pos.generation))
        if seen is not None:
            return seen[0], "last_seen"
        return None, "none"

    def _close_reregistered(
        self, tick: TickData, positions: dict[int, Position],
        trades: list[dict], capital: float,
    ) -> float:
        """Close every position whose netuid now shows a different subnet
        generation, at the deregistration refund (spot x alpha at the dead
        subnet's last observed state). Returns updated capital."""
        for netuid in list(positions.keys()):
            pos = positions[netuid]
            st = tick.subnets.get(netuid)
            if (st is None or st.generation is None or pos.generation is None
                    or st.generation == pos.generation):
                continue
            seen = self._last_seen.get((netuid, pos.generation))
            if seen is not None:
                old, old_ts = seen
                tao_pool, alpha_pool = old.tao_pool, old.alpha_pool
            else:  # defensive: position opened without a recorded state
                old_ts, tao_pool, alpha_pool = pos.entry_time, 0.0, 0.0
            rec = ledger.close_deregistered(
                positions, netuid, tao_pool, alpha_pool, old_ts,
                yield_fn=self._yield_fn,
            )
            capital += rec["tao_received"]
            trade = {
                "netuid": netuid,
                "exit_time": tick.timestamp,
                "reason": "dereg_refund",
                "exit_source": "last_seen_before_reregistration",
                "last_seen_time": old_ts,
                "new_generation": st.generation,
                **rec,
            }
            trades.append(trade)
            self._record_trade(trade, "sell")
        return capital

    # ── Order execution ──────────────────────────────────────────

    def _execute_order(
        self,
        order: Order,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
        trades: list[dict],
    ) -> float | None:
        """Execute an order. Returns updated capital if changed, else None."""
        st = tick.subnets.get(order.netuid)
        if st is None:
            return None

        # Reserve-vs-price consistency: the fill uses the reserves, so a tick
        # whose reserves imply a different price than ``st.price`` is mispriced.
        if st.price > 0 and st.alpha_pool > 0 and st.tao_pool > 0:
            self._orders_checked += 1
            if abs(st.tao_pool / st.alpha_pool / st.price - 1.0) > RESERVE_GAP_TOL:
                self._orders_gap += 1

        if order.side == "buy":
            return self._execute_buy(order, st, tick, capital, positions, trades)
        elif order.side == "sell":
            return self._execute_sell(order, st, tick, capital, positions, trades)
        return None

    def _failed(self, order: Order, side: str, tick: TickData, price: float,
                reason: str | None, latency_ms=None,
                pos: Position | None = None) -> dict:
        return {
            "netuid": order.netuid,
            "entry_time": pos.entry_time if pos else tick.timestamp,
            "exit_time": None,
            "entry_price": pos.entry_price if pos else price,
            "exit_price": None,
            "alpha_qty": 0,
            "tao_cost": 0,
            "tao_received": 0,
            "pnl": 0,
            "fees": 0,
            "hold_seconds": (tick.timestamp - pos.entry_time) if pos else 0,
            "reason": order.reason,
            "status": "failed",
            "failure_reason": reason,
            "latency_ms": latency_ms,
            "side": side,
        }

    def _execute_buy(
        self,
        order: Order,
        st: SubnetTick,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
        trades: list[dict],
    ) -> float | None:
        """Execute a buy order. Returns updated capital."""
        spend = order.tao_amount
        if spend <= 0 or spend > capital:
            return None

        # If the order has a limit_price and the price moved above it, skip.
        if order.limit_price is not None and st.price > order.limit_price:
            return None

        # ── Pool-safety filter (matches paper-bot behavior) ──────
        # Drop buys for flagged subnets; sells are never blocked. as_of is
        # the tick time (point-in-time; see bt-trading-tools 49bc38b).
        if self.pool_safety_checker is not None:
            try:
                as_of = datetime.fromtimestamp(tick.timestamp, tz=timezone.utc)
                flags = self.pool_safety_checker.check(order.netuid, as_of=as_of)
                if flags is not None and getattr(flags, "any_unsafe", False):
                    failed = self._failed(order, "buy", tick, st.price, "pool_safety")
                    trades.append(failed)
                    self._record_trade(failed, "buy")
                    return None
            except Exception:
                pass  # fail-open on data-access errors

        # Optional per-bot cap on order size relative to pool depth.
        if self.max_pool_pct is not None and st.tao_pool > 0:
            spend = min(spend, st.tao_pool * self.max_pool_pct)

        if st.tao_pool <= 0 or st.alpha_pool <= 0:
            return None

        fee, fee_components = self._buy_fee(
            order.netuid, spend, st.tao_pool, st.alpha_pool,
        )
        spend_net = spend - fee
        if spend_net <= 0:
            return None

        alpha_received, _, _ = amm_buy(spend_net, st.tao_pool, st.alpha_pool)
        if alpha_received <= 0:
            return None
        eff_price = spend / alpha_received

        # ── Realism layer (same simulator as PaperBotBase) ─────────
        action = {
            "type": "buy",
            "netuid": order.netuid,
            "tao_spent": spend,
            "alpha_qty": alpha_received,
            "price": eff_price,
            "decision_pool_tao": st.tao_pool,
            "decision_pool_alpha": st.alpha_pool,
        }
        self._realism.simulate_fill(action)
        if action.get("status") == "failed":
            failed = self._failed(order, "buy", tick, eff_price,
                                  action.get("failure_reason"),
                                  action.get("latency_ms"))
            trades.append(failed)
            self._record_trade(failed, "buy")
            return None
        spend = action["tao_spent"]
        alpha_received = action["alpha_qty"]
        if alpha_received <= 0:
            return None
        eff_price = spend / alpha_received

        ledger.book_buy(
            positions, order.netuid, alpha_received, spend, fee, tick.timestamp,
            generation=st.generation, yield_fn=self._yield_fn,
            metadata=order.signal_data,
        )

        trade = {
            "netuid": order.netuid,
            "entry_time": tick.timestamp,
            "exit_time": None,
            "entry_price": eff_price,
            "exit_price": None,
            "alpha_qty": alpha_received,
            "tao_cost": spend,
            "tao_received": 0,
            "pnl": 0,
            "fees": fee,
            "hold_seconds": 0,
            "reason": order.reason,
            "generation": st.generation,
            **fee_components,
        }
        self._record_trade(trade, "buy")

        return capital - spend

    def _execute_sell(
        self,
        order: Order,
        st: SubnetTick,
        tick: TickData,
        capital: float,
        positions: dict[int, Position],
        trades: list[dict],
    ) -> float | None:
        """Execute a sell order. Returns updated capital.

        ``order.alpha_amount <= 0`` means "sell the whole position, including
        accrued yield". Any cap, partial fill or realism truncation leaves the
        unsold remainder open (the ledger keeps it); nothing is deleted
        unless the remainder is dust.
        """
        pos = positions.get(order.netuid)
        if pos is None:
            return None

        held = pos.alpha_qty + ledger.accrued_yield(
            pos, tick.timestamp, self._yield_fn)
        if order.alpha_amount <= 0:
            alpha_to_sell = held
        else:
            alpha_to_sell = min(order.alpha_amount, held)

        if self.max_pool_pct is not None and st.alpha_pool > 0:
            alpha_to_sell = min(alpha_to_sell, st.alpha_pool * self.max_pool_pct)

        if alpha_to_sell <= 0:
            return None

        # ── TP overshoot prevention via limit_price (Bug #4) ──
        use_pool_tao = st.tao_pool
        use_pool_alpha = st.alpha_pool
        if order.limit_price is not None and st.price > order.limit_price:
            k = st.tao_pool * st.alpha_pool
            if k > 0 and order.limit_price > 0:
                use_pool_tao = (k * order.limit_price) ** 0.5
                use_pool_alpha = (k / order.limit_price) ** 0.5

        if use_pool_tao > 0 and use_pool_alpha > 0:
            fee, fee_components = self._sell_fee(
                order.netuid, alpha_to_sell, use_pool_tao, use_pool_alpha,
            )
            tao_received = ledger.liquidation_value(
                alpha_to_sell, use_pool_tao, use_pool_alpha, fee)
        else:
            tao_received = alpha_to_sell * st.price
            fee = 0.0
            fee_components = {}

        # ── Realism layer (sell) ───────────────────────────────────
        sell_action = {
            "type": "sell",
            "netuid": order.netuid,
            "alpha_qty": alpha_to_sell,
            "tao_received": tao_received,
            "exit_price": tao_received / alpha_to_sell if alpha_to_sell > 0 else 0,
            "decision_pool_tao": use_pool_tao,
            "decision_pool_alpha": use_pool_alpha,
        }
        self._realism.simulate_fill(sell_action)
        if sell_action.get("status") == "failed":
            failed = self._failed(order, "sell", tick, pos.entry_price,
                                  sell_action.get("failure_reason"),
                                  sell_action.get("latency_ms"), pos=pos)
            trades.append(failed)
            self._record_trade(failed, "sell")
            return None
        alpha_to_sell = sell_action["alpha_qty"]
        tao_received = sell_action["tao_received"]
        if alpha_to_sell <= 0:
            return None

        rec = ledger.book_sell(
            positions, order.netuid, alpha_to_sell, tao_received, fee,
            tick.timestamp, yield_fn=self._yield_fn,
        )
        trade = {
            "netuid": order.netuid,
            "exit_time": tick.timestamp,
            "exit_price": tao_received / alpha_to_sell if alpha_to_sell > 0 else 0,
            "reason": order.reason,
            **rec,
            **fee_components,
        }
        trades.append(trade)
        self._record_trade(trade, "sell")
        return capital + tao_received

    # ── Helpers ──────────────────────────────────────────────────

    # ── Fee + yield helpers (opt-in via fee_model / yield_model) ─────

    def _buy_fee(
        self, netuid: int, spend: float,
        pool_tao: float, pool_alpha: float,
    ) -> tuple[float, dict]:
        """Return (total_fee, extra_trade_fields)."""
        if self.fee_model is None:
            # Flat-rate path also includes a calibrated proxy fee component
            # when uses_proxy=True (default) — see fees.FALLBACK_PROXY_TAO.
            # Otherwise just swap + gas (back-compat for explicit no-proxy).
            from bt_trading_tools.fees import FALLBACK_PROXY_TAO
            fee = spend * self.swap_fee_rate + self.gas_fee_tao
            if self.uses_proxy:
                fee += FALLBACK_PROXY_TAO
            return fee, {}
        spot = pool_tao / pool_alpha if pool_alpha > 0 else None
        q = self.fee_model.quote(
            "add_stake", netuid=netuid, amount=spend,
            uses_proxy=self.uses_proxy, spot_price=spot,
        )
        return q.total_fee_tao, {
            "swap_fee_tao": q.swap_fee_tao,
            "gas_fee_tao": q.gas_fee_tao,
            "proxy_fee_tao": q.proxy_fee_tao,
            "fee_source": q.source.value,
        }

    def _sell_fee(
        self, netuid: int, alpha_qty: float,
        pool_tao: float, pool_alpha: float,
    ) -> tuple[float, dict]:
        """Return (total_fee, extra_trade_fields)."""
        if self.fee_model is None:
            # Historical engine computed fee from tao_out, not alpha_qty.
            # Reproduce that path exactly for back-compat. Add proxy fee
            # when uses_proxy=True (default).
            from bt_trading_tools.fees import FALLBACK_PROXY_TAO
            tao_out, _, _ = amm_sell(alpha_qty, pool_tao, pool_alpha)
            fee = tao_out * self.swap_fee_rate + self.gas_fee_tao
            if self.uses_proxy:
                fee += FALLBACK_PROXY_TAO
            return fee, {}
        spot = pool_tao / pool_alpha if pool_alpha > 0 else None
        q = self.fee_model.quote(
            "remove_stake", netuid=netuid, amount=alpha_qty,
            uses_proxy=self.uses_proxy, spot_price=spot,
        )
        return q.total_fee_tao, {
            "swap_fee_tao": q.swap_fee_tao,
            "gas_fee_tao": q.gas_fee_tao,
            "proxy_fee_tao": q.proxy_fee_tao,
            "fee_source": q.source.value,
        }

    def _yield_fn(self, netuid: int, alpha_qty: float,
                  since: float, now: float) -> float:
        """Ledger-compatible yield function over the engine's yield model."""
        if self.yield_model is None:
            return 0.0
        return self.yield_model.accrued_yield(
            netuid=netuid, alpha_qty=alpha_qty, entry_time=since, now=now,
        )

    def _portfolio_value(
        self,
        capital: float,
        positions: dict[int, Position],
        tick: TickData,
    ) -> float:
        """Cash plus each position's liquidation value net of the sell fee.

        Convention (2026-10-01): what an AMM sale of the whole position would
        return now, as PaperBotBase marks it. A subnet absent from this tick
        is valued at its last observed state for the position's generation
        (pools do not change between transactions), not at cost.
        """
        value = capital
        for netuid, pos in positions.items():
            st, _ = self._state_for(pos, tick)
            eff_alpha = pos.alpha_qty + ledger.accrued_yield(
                pos, tick.timestamp, self._yield_fn)
            if st is not None and st.tao_pool > 0 and st.alpha_pool > 0:
                fee, _ = self._sell_fee(netuid, eff_alpha, st.tao_pool, st.alpha_pool)
                value += ledger.liquidation_value(
                    eff_alpha, st.tao_pool, st.alpha_pool, fee)
            else:
                value += pos.tao_cost  # defensive: no state ever observed
        return value

    def _record_trade(self, trade: dict, trade_type: str) -> None:
        """Record to TradeLog if configured."""
        if self.trade_log and trade.get("exit_time") is not None:
            try:
                self.trade_log.record_trade(
                    trade_type=trade_type,
                    netuid=trade["netuid"],
                    tao_amount=trade.get("tao_received", trade.get("tao_cost", 0)),
                    alpha_amount=trade["alpha_qty"],
                    price=trade.get("exit_price", trade.get("entry_price", 0)),
                    slippage=0,
                    hotkey="backtest",
                    reason=trade.get("reason", ""),
                    signal_data=None,
                    timestamp=trade["exit_time"],
                )
            except Exception:
                pass  # Don't let logging failures crash the backtest
