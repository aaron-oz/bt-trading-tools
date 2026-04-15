"""
Backtest statistics — compute summary metrics from trade and equity data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BacktestStats:
    """Summary statistics from a backtest run."""
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float                   # percentage
    total_pnl: float                  # TAO
    total_return_pct: float           # percentage
    total_fees: float                 # TAO
    avg_pnl: float                    # TAO per trade
    median_pnl: float                 # TAO per trade
    avg_winner: float                 # TAO
    avg_loser: float                  # TAO
    avg_hold_seconds: float           # seconds
    max_drawdown_pct: float           # percentage
    sharpe: float                     # annualized (assumes daily ticks)
    n_subnets_traded: int
    final_equity: float               # TAO
    starting_capital: float           # TAO
    by_reason: dict[str, dict]        # exit_reason → {count, total_pnl, avg_pnl}


def compute_stats(
    trades: list[dict],
    equity_curve: list[dict],
    starting_capital: float,
    ticks_per_year: float = 365.0,
) -> BacktestStats:
    """Compute summary statistics from backtest results.

    Args:
        trades: List of trade dicts with keys: pnl, fees, hold_seconds,
                netuid, reason.
        equity_curve: List of dicts with key: total_equity.
        starting_capital: Initial capital in TAO.
        ticks_per_year: Number of ticks per year (for Sharpe annualization).
            365 for daily, 8760 for hourly, 2628000 for 12-second blocks.
    """
    if not trades:
        return BacktestStats(
            n_trades=0, n_wins=0, n_losses=0, win_rate=0,
            total_pnl=0, total_return_pct=0, total_fees=0,
            avg_pnl=0, median_pnl=0, avg_winner=0, avg_loser=0,
            avg_hold_seconds=0, max_drawdown_pct=0, sharpe=0,
            n_subnets_traded=0, final_equity=starting_capital,
            starting_capital=starting_capital, by_reason={},
        )

    pnls = [t["pnl"] for t in trades]
    fees = [t.get("fees", 0) for t in trades]
    holds = [t.get("hold_seconds", 0) for t in trades]
    netuids = {t["netuid"] for t in trades}
    reasons = {}
    for t in trades:
        r = t.get("reason", "unknown")
        if r not in reasons:
            reasons[r] = {"count": 0, "total_pnl": 0.0}
        reasons[r]["count"] += 1
        reasons[r]["total_pnl"] += t["pnl"]
    for r in reasons:
        reasons[r]["avg_pnl"] = reasons[r]["total_pnl"] / reasons[r]["count"]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    # Median
    sorted_pnls = sorted(pnls)
    n = len(sorted_pnls)
    median = (sorted_pnls[n // 2] if n % 2 == 1
              else (sorted_pnls[n // 2 - 1] + sorted_pnls[n // 2]) / 2)

    # Max drawdown from equity curve
    max_dd = 0.0
    if equity_curve:
        peak = 0.0
        for pt in equity_curve:
            eq = pt["total_equity"]
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd

    # Sharpe from equity curve returns
    sharpe = 0.0
    if len(equity_curve) > 1:
        equities = [pt["total_equity"] for pt in equity_curve]
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append(equities[i] / equities[i - 1] - 1)
        if returns:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(var_r) if var_r > 0 else 0
            if std_r > 0:
                sharpe = mean_r / std_r * math.sqrt(ticks_per_year)

    total_pnl = sum(pnls)
    final_eq = equity_curve[-1]["total_equity"] if equity_curve else starting_capital

    return BacktestStats(
        n_trades=len(trades),
        n_wins=len(winners),
        n_losses=len(losers),
        win_rate=len(winners) / len(trades) * 100 if trades else 0,
        total_pnl=round(total_pnl, 6),
        total_return_pct=round(total_pnl / starting_capital * 100, 2) if starting_capital > 0 else 0,
        total_fees=round(sum(fees), 6),
        avg_pnl=round(total_pnl / len(trades), 6),
        median_pnl=round(median, 6),
        avg_winner=round(sum(winners) / len(winners), 6) if winners else 0,
        avg_loser=round(sum(losers) / len(losers), 6) if losers else 0,
        avg_hold_seconds=round(sum(holds) / len(holds), 1) if holds else 0,
        max_drawdown_pct=round(max_dd, 2),
        sharpe=round(sharpe, 3),
        n_subnets_traded=len(netuids),
        final_equity=round(final_eq, 6),
        starting_capital=starting_capital,
        by_reason=reasons,
    )
