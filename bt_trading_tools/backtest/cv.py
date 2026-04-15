"""
Purged walk-forward cross-validation for backtesting.

Implements de Prado's purged K-fold CV adapted for tick-based backtesting:
- Chronological splits (no future leakage)
- Purge gap around train/test boundaries to prevent position carry-over leakage
- Optional embargo after test folds to reduce serial correlation

Usage::

    from bt_trading_tools.backtest.cv import PurgedWalkForwardCV, CVFold
    from bt_trading_tools.backtest import TickData

    cv = PurgedWalkForwardCV(
        n_folds=5,
        purge_days=30,   # max hold period — positions from train can't leak
        embargo_days=0,
    )
    for fold in cv.split(ticks, ticks_per_day=1):
        # fold.train_ticks, fold.test_ticks are non-overlapping
        train_results = engine.run(fold.train_ticks, strategy)
        test_results = engine.run(fold.test_ticks, strategy)

Also provides expanding-window walk-forward splits::

    cv = PurgedWalkForwardCV(n_folds=5, purge_days=30)
    for fold in cv.expanding_split(ticks, min_train_days=60, ticks_per_day=1):
        # fold.train_ticks grows each fold, test_ticks is always the next block
        ...
"""

from __future__ import annotations

from dataclasses import dataclass

from bt_trading_tools.backtest.types import TickData


@dataclass
class CVFold:
    """One train/test split from cross-validation."""
    fold_idx: int
    train_ticks: list[TickData]
    test_ticks: list[TickData]
    # Metadata for reporting
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int


class PurgedWalkForwardCV:
    """Purged walk-forward cross-validation for tick-based backtesting.

    Args:
        n_folds: Number of folds to create.
        purge_days: Days to purge between train and test boundaries.
            Should be >= strategy's max hold period to prevent position
            carry-over leakage. Ticks within the purge window are dropped
            from training data.
        embargo_days: Days to skip after each test fold before the next
            training window starts. Reduces serial correlation between
            consecutive test results.
        ticks_per_day: How many ticks per calendar day (1 for daily,
            24 for hourly, etc.). Used to convert day-based purge/embargo
            to tick counts.
    """

    def __init__(
        self,
        n_folds: int = 5,
        purge_days: int = 30,
        embargo_days: int = 0,
        ticks_per_day: int = 1,
    ):
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.n_folds = n_folds
        self.purge_ticks = purge_days * ticks_per_day
        self.embargo_ticks = embargo_days * ticks_per_day
        self.ticks_per_day = ticks_per_day

    def split(self, ticks: list[TickData]) -> list[CVFold]:
        """Generate purged K-fold splits (chronological, non-expanding).

        Each fold uses one contiguous block as test, the rest (minus purge
        zones) as training. Unlike random holdout, this respects temporal
        ordering and prevents leakage from open positions.

        Returns:
            List of CVFold objects, one per fold.
        """
        n = len(ticks)
        if n < self.n_folds * 2:
            raise ValueError(f"Not enough ticks ({n}) for {self.n_folds} folds")

        fold_size = n // self.n_folds
        folds = []

        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_folds - 1 else n

            # Training = everything outside [test_start, test_end), minus purge zones
            train_ticks = []
            for j in range(n):
                # Skip test region
                if test_start <= j < test_end:
                    continue
                # Skip purge zone before test (positions opened here might still
                # be open when test starts)
                if test_start - self.purge_ticks <= j < test_start:
                    continue
                # Skip embargo zone after test
                if test_end <= j < test_end + self.embargo_ticks:
                    continue
                train_ticks.append(ticks[j])

            test_ticks = ticks[test_start:test_end]

            if not train_ticks or not test_ticks:
                continue

            folds.append(CVFold(
                fold_idx=i,
                train_ticks=train_ticks,
                test_ticks=test_ticks,
                train_start_ts=train_ticks[0].timestamp,
                train_end_ts=train_ticks[-1].timestamp,
                test_start_ts=test_ticks[0].timestamp,
                test_end_ts=test_ticks[-1].timestamp,
            ))

        return folds

    def expanding_split(
        self,
        ticks: list[TickData],
        min_train_days: int = 60,
    ) -> list[CVFold]:
        """Generate expanding-window walk-forward splits.

        Training window grows with each fold. Test window is always the
        next block after training + purge gap. This is the most realistic
        CV for live trading: you always train on all available history.

        Args:
            ticks: Chronologically ordered tick data.
            min_train_days: Minimum training days before first test fold.

        Returns:
            List of CVFold objects.
        """
        n = len(ticks)
        min_train_ticks = min_train_days * self.ticks_per_day

        if n <= min_train_ticks + self.purge_ticks:
            raise ValueError(
                f"Not enough ticks ({n}) for min_train={min_train_ticks} + purge={self.purge_ticks}"
            )

        # Divide remaining ticks (after min_train + purge) into n_folds test blocks
        available = n - min_train_ticks - self.purge_ticks
        if available < self.n_folds:
            raise ValueError(f"Only {available} ticks available for {self.n_folds} test folds")

        test_block_size = available // self.n_folds
        folds = []

        for i in range(self.n_folds):
            # Training: ticks[0 : train_end]
            train_end = min_train_ticks + i * test_block_size

            # Purge gap: skip purge_ticks after train_end
            test_start = train_end + self.purge_ticks

            # Test block
            if i < self.n_folds - 1:
                test_end = test_start + test_block_size
            else:
                test_end = n  # last fold gets remaining ticks

            if test_start >= n or test_end > n:
                break

            train_ticks = ticks[:train_end]
            test_ticks = ticks[test_start:test_end]

            if not train_ticks or not test_ticks:
                continue

            folds.append(CVFold(
                fold_idx=i,
                train_ticks=train_ticks,
                test_ticks=test_ticks,
                train_start_ts=train_ticks[0].timestamp,
                train_end_ts=train_ticks[-1].timestamp,
                test_start_ts=test_ticks[0].timestamp,
                test_end_ts=test_ticks[-1].timestamp,
            ))

        return folds
