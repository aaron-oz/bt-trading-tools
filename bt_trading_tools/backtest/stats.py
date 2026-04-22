"""
Backtest statistics — compute summary metrics from trade and equity data.

Includes:
- BacktestStats: aggregate metrics from a single run
- RegimeStats: per-regime breakdown of metrics
- ValidationReport: automated go/no-go checks from CV results
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Per-regime breakdown
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Performance breakdown for a single regime condition."""
    regime: str
    n_trades: int
    n_wins: int
    win_rate: float              # percentage
    total_pnl: float             # TAO
    avg_pnl: float               # TAO per trade
    sharpe: float                # annualized
    max_drawdown_pct: float      # percentage
    pct_of_time: float           # percentage of ticks in this regime


def compute_regime_stats(
    trades: list[dict],
    equity_curve: list[dict],
    starting_capital: float,
    tao_usd: list[float] | None = None,
    timestamps: list[int] | None = None,
    sma_short: int = 20,
    sma_long: int = 50,
    ticks_per_year: float = 365.0,
) -> dict[str, RegimeStats]:
    """Compute per-regime performance breakdown.

    Classifies each tick as 'tao_bear' or 'tao_bull' based on SMA crossover
    of TAO/USD prices, then splits trades and equity by regime.

    Args:
        trades: Trade dicts (must include 'entry_time' and 'pnl' keys).
        equity_curve: Equity dicts (must include 'timestamp' and 'total_equity').
        starting_capital: Initial capital in TAO.
        tao_usd: TAO/USD prices aligned with equity_curve timestamps.
            If None, attempts to read from equity_curve 'tao_usd' key,
            or from global_signals.
        timestamps: Unix timestamps aligned with tao_usd. If None, uses
            equity_curve timestamps.
        sma_short: Short SMA window for regime classification.
        sma_long: Long SMA window for regime classification.
        ticks_per_year: For Sharpe annualization.

    Returns:
        Dict mapping regime name to RegimeStats.
    """
    if not trades or not equity_curve:
        return {}

    # Build TAO/USD series and classify regime per tick
    if tao_usd is None:
        tao_usd = [pt.get("tao_usd", 0) for pt in equity_curve]

    if timestamps is None:
        timestamps = [pt.get("timestamp", i) for i, pt in enumerate(equity_curve)]

    n = len(tao_usd)
    if n < sma_long or all(p == 0 for p in tao_usd):
        return {}

    # Compute SMAs
    regimes = []  # one per tick: "tao_bear" or "tao_bull"
    for i in range(n):
        if i < sma_long - 1:
            regimes.append("warmup")
            continue
        sma_s = sum(tao_usd[i - sma_short + 1:i + 1]) / sma_short
        sma_l = sum(tao_usd[i - sma_long + 1:i + 1]) / sma_long
        regimes.append("tao_bear" if sma_s < sma_l else "tao_bull")

    # Build timestamp → regime lookup
    ts_to_regime = {}
    for i, ts in enumerate(timestamps):
        ts_to_regime[ts] = regimes[i]

    # Classify trades by regime at entry time
    regime_trades: dict[str, list[dict]] = {}
    for trade in trades:
        entry_ts = trade.get("entry_time", 0)
        # Find closest timestamp
        regime = "unknown"
        if entry_ts in ts_to_regime:
            regime = ts_to_regime[entry_ts]
        else:
            # Find nearest
            best_dist = float("inf")
            for ts, r in ts_to_regime.items():
                dist = abs(ts - entry_ts)
                if dist < best_dist:
                    best_dist = dist
                    regime = r
        if regime == "warmup":
            regime = "unknown"
        regime_trades.setdefault(regime, []).append(trade)

    # Split equity curve by regime for Sharpe/DD
    regime_equity: dict[str, list[dict]] = {}
    for i, pt in enumerate(equity_curve):
        r = regimes[i] if i < len(regimes) else "unknown"
        if r == "warmup":
            continue
        regime_equity.setdefault(r, []).append(pt)

    # Count ticks per regime (excluding warmup)
    regime_tick_counts: dict[str, int] = {}
    for r in regimes:
        if r != "warmup":
            regime_tick_counts[r] = regime_tick_counts.get(r, 0) + 1
    total_ticks = sum(regime_tick_counts.values())

    # Compute stats per regime
    result = {}
    for regime_name, r_trades in regime_trades.items():
        if regime_name == "unknown":
            continue
        pnls = [t["pnl"] for t in r_trades]
        winners = [p for p in pnls if p > 0]

        # Sharpe from regime equity slice
        r_sharpe = 0.0
        r_max_dd = 0.0
        r_eq = regime_equity.get(regime_name, [])
        if len(r_eq) > 1:
            equities = [pt["total_equity"] for pt in r_eq]
            returns = []
            for j in range(1, len(equities)):
                if equities[j - 1] > 0:
                    returns.append(equities[j] / equities[j - 1] - 1)
            if returns:
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                std_r = math.sqrt(var_r) if var_r > 0 else 0
                if std_r > 0:
                    r_sharpe = mean_r / std_r * math.sqrt(ticks_per_year)

            # Max drawdown
            peak = 0.0
            for pt in r_eq:
                eq = pt["total_equity"]
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak * 100
                    if dd > r_max_dd:
                        r_max_dd = dd

        result[regime_name] = RegimeStats(
            regime=regime_name,
            n_trades=len(r_trades),
            n_wins=len(winners),
            win_rate=round(len(winners) / len(r_trades) * 100, 1) if r_trades else 0,
            total_pnl=round(sum(pnls), 6),
            avg_pnl=round(sum(pnls) / len(r_trades), 6) if r_trades else 0,
            sharpe=round(r_sharpe, 3),
            max_drawdown_pct=round(r_max_dd, 2),
            pct_of_time=round(
                regime_tick_counts.get(regime_name, 0) / total_ticks * 100, 1
            ) if total_ticks > 0 else 0,
        )

    return result


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """Single validation criterion and result."""
    name: str
    value: float | str
    threshold: float | str
    passed: bool
    severity: str            # "critical", "warning", "info"
    detail: str = ""


@dataclass
class ValidationReport:
    """Automated go/no-go assessment of backtest CV results.

    Generated by validate_cv_results(). Each check is a ValidationCheck
    with pass/fail and severity. The overall verdict is one of:
    - PASS: all critical checks pass, no warnings
    - CONDITIONAL PASS: all critical checks pass, some warnings
    - FAIL: at least one critical check fails
    """
    checks: list[ValidationCheck]
    verdict: str              # "PASS", "CONDITIONAL PASS", "FAIL"
    regime_stats: dict[str, RegimeStats] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the validation report."""
        lines = [f"Validation Report: {self.verdict}", "=" * 50]
        for check in self.checks:
            icon = "PASS" if check.passed else "FAIL"
            if not check.passed and check.severity == "warning":
                icon = "WARN"
            lines.append(
                f"  [{icon}] {check.name}: {check.value} "
                f"(threshold: {check.threshold})"
            )
            if check.detail:
                lines.append(f"        {check.detail}")

        if self.regime_stats:
            lines.append("")
            lines.append("Per-Regime Breakdown:")
            lines.append("-" * 50)
            for regime, rs in sorted(self.regime_stats.items()):
                lines.append(
                    f"  {regime}: Sharpe={rs.sharpe:.3f}, "
                    f"WR={rs.win_rate:.1f}%, "
                    f"PnL={rs.total_pnl:.4f} TAO, "
                    f"Trades={rs.n_trades}, "
                    f"Time={rs.pct_of_time:.0f}%"
                )

        return "\n".join(lines)


def validate_cv_results(
    fold_stats: list[BacktestStats],
    train_stats: list[BacktestStats] | None = None,
    n_free_params: int = 0,
    tao_usd_per_fold: list[list[float]] | None = None,
    equity_per_fold: list[list[dict]] | None = None,
    trades_per_fold: list[list[dict]] | None = None,
    timestamps_per_fold: list[list[int]] | None = None,
    max_overfit_ratio: float = 1.5,
    min_trades_per_fold: int = 20,
    min_trades_per_param: int = 10,
    max_drawdown_pct: float = 25.0,
    min_sharpe: float = 1.0,
) -> ValidationReport:
    """Run automated validation checks on cross-validation results.

    Args:
        fold_stats: BacktestStats from each OOS fold.
        train_stats: BacktestStats from each training fold (for overfit ratio).
        n_free_params: Number of optimized parameters (for trades-per-param check).
        tao_usd_per_fold: TAO/USD prices per fold (for regime breakdown).
        equity_per_fold: Equity curves per fold (for regime breakdown).
        trades_per_fold: Trade lists per fold (for regime breakdown).
        timestamps_per_fold: Timestamps per fold (for regime breakdown).
        max_overfit_ratio: Maximum acceptable train/test Sharpe ratio.
        min_trades_per_fold: Minimum trades required per fold.
        min_trades_per_param: Minimum trades per free parameter.
        max_drawdown_pct: Maximum acceptable drawdown.
        min_sharpe: Minimum acceptable average OOS Sharpe.

    Returns:
        ValidationReport with checks, verdict, and regime breakdown.
    """
    checks: list[ValidationCheck] = []

    # --- Aggregate fold metrics ---
    avg_sharpe = sum(f.sharpe for f in fold_stats) / len(fold_stats)
    avg_return = sum(f.total_return_pct for f in fold_stats) / len(fold_stats)
    avg_wr = sum(f.win_rate for f in fold_stats) / len(fold_stats)
    min_trades = min(f.n_trades for f in fold_stats)
    total_trades = sum(f.n_trades for f in fold_stats)
    max_dd = max(f.max_drawdown_pct for f in fold_stats)

    # 1. Overfit ratio (critical)
    if train_stats and len(train_stats) == len(fold_stats):
        train_sharpes = [f.sharpe for f in train_stats]
        test_sharpes = [f.sharpe for f in fold_stats]
        avg_train = sum(train_sharpes) / len(train_sharpes)
        avg_test = sum(test_sharpes) / len(test_sharpes)
        overfit_ratio = avg_train / avg_test if avg_test != 0 else float("inf")
        checks.append(ValidationCheck(
            name="Overfit ratio (train/test Sharpe)",
            value=round(overfit_ratio, 2),
            threshold=f"<{max_overfit_ratio}x",
            passed=overfit_ratio < max_overfit_ratio,
            severity="critical",
            detail=f"Train Sharpe={avg_train:.3f}, Test Sharpe={avg_test:.3f}",
        ))
    else:
        checks.append(ValidationCheck(
            name="Overfit ratio",
            value="N/A",
            threshold=f"<{max_overfit_ratio}x",
            passed=True,
            severity="info",
            detail="No train stats provided — cannot compute overfit ratio",
        ))

    # 2. Min trades per fold (critical)
    checks.append(ValidationCheck(
        name="Min trades per fold",
        value=min_trades,
        threshold=f">={min_trades_per_fold}",
        passed=min_trades >= min_trades_per_fold,
        severity="critical",
        detail=f"Trades per fold: {[f.n_trades for f in fold_stats]}",
    ))

    # 3. Trades per parameter (critical if n_free_params > 0)
    if n_free_params > 0:
        trades_per_param = total_trades / n_free_params
        checks.append(ValidationCheck(
            name="Trades per free parameter",
            value=round(trades_per_param, 1),
            threshold=f">={min_trades_per_param}",
            passed=trades_per_param >= min_trades_per_param,
            severity="critical",
            detail=f"{total_trades} total trades / {n_free_params} params",
        ))

    # 4. Average OOS Sharpe (critical)
    checks.append(ValidationCheck(
        name="Avg OOS Sharpe",
        value=round(avg_sharpe, 3),
        threshold=f">={min_sharpe}",
        passed=avg_sharpe >= min_sharpe,
        severity="critical",
    ))

    # 5. Max drawdown (warning)
    checks.append(ValidationCheck(
        name="Max drawdown",
        value=f"{max_dd:.1f}%",
        threshold=f"<{max_drawdown_pct}%",
        passed=max_dd < max_drawdown_pct,
        severity="warning",
        detail=f"DD per fold: {[f'{f.max_drawdown_pct:.1f}%' for f in fold_stats]}",
    ))

    # 6. Average win rate (info)
    checks.append(ValidationCheck(
        name="Avg win rate",
        value=f"{avg_wr:.1f}%",
        threshold=">50%",
        passed=avg_wr > 50,
        severity="warning",
    ))

    # 7. Negative folds (warning)
    neg_folds = sum(1 for f in fold_stats if f.total_return_pct < 0)
    checks.append(ValidationCheck(
        name="Negative return folds",
        value=f"{neg_folds}/{len(fold_stats)}",
        threshold="0",
        passed=neg_folds == 0,
        severity="warning",
        detail="Folds with negative returns suggest regime dependence" if neg_folds > 0 else "",
    ))

    # 8. Sharpe consistency across folds (warning)
    sharpe_std = math.sqrt(
        sum((f.sharpe - avg_sharpe) ** 2 for f in fold_stats) / len(fold_stats)
    )
    sharpe_cv = sharpe_std / abs(avg_sharpe) if avg_sharpe != 0 else float("inf")
    checks.append(ValidationCheck(
        name="Sharpe consistency (CV)",
        value=round(sharpe_cv, 2),
        threshold="<1.0",
        passed=sharpe_cv < 1.0,
        severity="warning",
        detail=f"Sharpe per fold: {[f'{f.sharpe:.3f}' for f in fold_stats]}",
    ))

    # --- Per-regime breakdown ---
    regime_stats: dict[str, RegimeStats] = {}
    if tao_usd_per_fold and equity_per_fold and trades_per_fold:
        # Combine all folds for aggregate regime stats
        all_trades = []
        all_equity = []
        all_tao_usd = []
        all_timestamps = []
        for i in range(len(fold_stats)):
            all_trades.extend(trades_per_fold[i])
            all_equity.extend(equity_per_fold[i])
            all_tao_usd.extend(tao_usd_per_fold[i])
            if timestamps_per_fold:
                all_timestamps.extend(timestamps_per_fold[i])

        if all_trades and all_equity:
            regime_stats = compute_regime_stats(
                trades=all_trades,
                equity_curve=all_equity,
                starting_capital=fold_stats[0].starting_capital,
                tao_usd=all_tao_usd,
                timestamps=all_timestamps if all_timestamps else None,
            )

            # Add regime-specific check
            if regime_stats:
                regime_sharpes = {r: rs.sharpe for r, rs in regime_stats.items()}
                worst_regime = min(regime_sharpes, key=regime_sharpes.get)
                checks.append(ValidationCheck(
                    name=f"Worst regime ({worst_regime}) Sharpe",
                    value=round(regime_sharpes[worst_regime], 3),
                    threshold=">0",
                    passed=regime_sharpes[worst_regime] > 0,
                    severity="warning",
                    detail=(
                        f"Regime Sharpes: {', '.join(f'{r}={s:.3f}' for r, s in regime_sharpes.items())}. "
                        "Negative regime Sharpe = bot needs meta-agent gating."
                    ),
                ))

    # --- Verdict ---
    critical_fails = [c for c in checks if not c.passed and c.severity == "critical"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]

    if critical_fails:
        verdict = "FAIL"
    elif warnings:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    return ValidationReport(
        checks=checks,
        verdict=verdict,
        regime_stats=regime_stats,
    )
