"""
Downstream P&L computed from the raw trade log.

Bots write dumb raw trades; this module reconstructs realized P&L per
position, honoring each bot's declared `position_model` (from its manifest).

Supported models:
    pair       1 buy trade + 1 sell trade share a position_id.
    level      Many buys/sells share a stable per-level id; FIFO within level.
    cycle      Many trades within an emission cycle share an id; FIFO within cycle.
    inventory  Typically one position per subnet; accounting = FIFO or avg-cost.

Only `status ∈ {executed, partial}` trades contribute to P&L.
`mtm_sample` records are ignored — they drive mark-to-market, not realized P&L.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bt_trading_tools.tracking.reader import iter_trade_log


PositionModel = Literal["pair", "level", "inventory", "cycle"]
PnLBasis = Literal["fifo", "avg_cost"]


@dataclass
class PositionPnL:
    position_id: str
    netuid: int
    realized_tao: float       # realized P&L net of fees
    tao_invested: float       # total TAO spent on buys
    tao_received: float       # total TAO received from sells
    tao_fees: float           # sum of network_fee_tao across trades (null fees treated as 0)
    trades: int
    is_open: bool
    alpha_held: float


def compute_pnl(
    log_path: str | Path,
    *,
    position_model: PositionModel,
    basis: PnLBasis = "fifo",
) -> dict[str, PositionPnL]:
    """Compute realized P&L per position_id from a v1 trade log.

    `basis` only matters for inventory model; pair/level/cycle always use FIFO
    within the position (which collapses to trivial matching for pair).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in iter_trade_log(log_path):
        if rec.get("record_type") != "trade":
            continue
        if rec.get("status") not in ("executed", "partial"):
            continue
        pid = rec.get("position_id")
        if pid is None:
            continue
        groups[pid].append(rec)

    results: dict[str, PositionPnL] = {}
    for pid, trades in groups.items():
        trades.sort(key=lambda r: r.get("timestamp", ""))
        if position_model == "inventory" and basis == "avg_cost":
            results[pid] = _avg_cost_pnl(pid, trades)
        else:
            results[pid] = _fifo_pnl(pid, trades)
    return results


def _trade_fees_tao(t: dict) -> float:
    """Sum of atomic fee components on a trade record. Treats missing as 0."""
    return (
        (t.get("swap_fee_tao") or 0.0)
        + (t.get("gas_fee_tao") or 0.0)
        + (t.get("proxy_fee_tao") or 0.0)
    )


def _fifo_pnl(pid: str, trades: list[dict]) -> PositionPnL:
    """FIFO cost basis with yield-attributed zero-cost alpha on sells.

    `alpha_yield_accrued` on a sell record is treated as zero-cost-basis
    alpha (it was "earned," not bought). The remaining (alpha_amount -
    yield_accrued) is matched against existing lots FIFO-style for cost
    basis. Realized P&L therefore credits yield at the full executed price.
    """
    lots: deque[list[float]] = deque()  # each lot: [alpha_remaining, tao_per_alpha_cost]
    tao_invested = 0.0
    tao_received = 0.0
    tao_fees = 0.0
    realized = 0.0
    netuid = trades[0].get("netuid", -1) if trades else -1

    for t in trades:
        tao_fees += _trade_fees_tao(t)
        if t["side"] == "buy":
            alpha = t["alpha_amount"]
            tao = t["tao_amount"]
            if alpha > 0:
                lots.append([alpha, tao / alpha])
                tao_invested += tao
        else:  # sell
            alpha_to_sell = t["alpha_amount"]
            yield_accrued = t.get("alpha_yield_accrued") or 0.0
            yield_accrued = min(yield_accrued, alpha_to_sell)  # safety clamp
            traded_alpha = alpha_to_sell - yield_accrued       # alpha with real cost basis
            tao_out = t["tao_amount"]
            tao_received += tao_out

            remaining = traded_alpha
            cost_basis_of_sold = 0.0
            while remaining > 1e-12 and lots:
                lot_alpha, lot_cost = lots[0]
                if lot_alpha <= remaining + 1e-12:
                    cost_basis_of_sold += lot_alpha * lot_cost
                    remaining -= lot_alpha
                    lots.popleft()
                else:
                    cost_basis_of_sold += remaining * lot_cost
                    lots[0][0] -= remaining
                    remaining = 0.0
            # Yield-accrued portion has zero cost basis → its tao revenue is pure profit.
            realized += tao_out - cost_basis_of_sold

    alpha_held = sum(l[0] for l in lots)
    return PositionPnL(
        position_id=pid,
        netuid=netuid,
        realized_tao=realized - tao_fees,
        tao_invested=tao_invested,
        tao_received=tao_received,
        tao_fees=tao_fees,
        trades=len(trades),
        is_open=alpha_held > 1e-9,
        alpha_held=alpha_held,
    )


def _avg_cost_pnl(pid: str, trades: list[dict]) -> PositionPnL:
    """Average-cost basis with yield-attributed zero-cost alpha on sells.

    See `_fifo_pnl` for the yield-accrual semantics; same policy here —
    `alpha_yield_accrued` on a sell bypasses cost-basis draw-down.
    """
    total_alpha = 0.0
    total_tao_invested_running = 0.0
    tao_invested = 0.0
    tao_received = 0.0
    tao_fees = 0.0
    realized = 0.0
    netuid = trades[0].get("netuid", -1) if trades else -1

    for t in trades:
        tao_fees += _trade_fees_tao(t)
        if t["side"] == "buy":
            total_alpha += t["alpha_amount"]
            total_tao_invested_running += t["tao_amount"]
            tao_invested += t["tao_amount"]
        else:  # sell
            alpha_to_sell = t["alpha_amount"]
            yield_accrued = t.get("alpha_yield_accrued") or 0.0
            yield_accrued = min(yield_accrued, alpha_to_sell)
            traded_alpha = alpha_to_sell - yield_accrued
            tao_out = t["tao_amount"]
            tao_received += tao_out
            if total_alpha > 0:
                avg_cost = total_tao_invested_running / total_alpha
                covered = min(traded_alpha, total_alpha)
                cost_of_sold = avg_cost * covered
                realized += tao_out - cost_of_sold
                total_tao_invested_running -= cost_of_sold
                total_alpha = max(0.0, total_alpha - traded_alpha)
            else:
                realized += tao_out  # sells without prior inventory have zero basis

    return PositionPnL(
        position_id=pid,
        netuid=netuid,
        realized_tao=realized - tao_fees,
        tao_invested=tao_invested,
        tao_received=tao_received,
        tao_fees=tao_fees,
        trades=len(trades),
        is_open=total_alpha > 1e-9,
        alpha_held=total_alpha,
    )
