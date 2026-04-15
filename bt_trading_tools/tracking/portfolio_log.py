"""
PortfolioLog — SQLite time-series of portfolio value per tick.

This is the "equity curve" log.  Every tick, the bot records its total
value.  Dead simple to query and plot::

    SELECT timestamp, total_value FROM portfolio WHERE bot_name='doubledip'

Can be shared across bots (same db file) for side-by-side comparison.
"""

import sqlite3
import time
from pathlib import Path


class PortfolioLog:
    """Per-tick portfolio value recording.

    Usage::

        plog = PortfolioLog("/shared/portfolio.db", bot_name="autobot")

        # Each tick:
        plog.record(total_value=52.3, cash=10.0, staked_value=42.3,
                     n_positions=5, n_pending=1)

        # Analysis:
        rows = plog.get_series(hours=24)
        # [(timestamp, total_value, cash, staked_value, n_positions, n_pending), ...]
    """

    def __init__(self, db_path: str | Path, bot_name: str):
        self.bot_name = bot_name
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio (
                timestamp INTEGER NOT NULL,
                bot_name TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                staked_value REAL NOT NULL,
                n_positions INTEGER NOT NULL,
                n_pending INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_portfolio_bot_ts
                ON portfolio(bot_name, timestamp);
        """)

    def record(
        self,
        total_value: float,
        cash: float,
        staked_value: float,
        n_positions: int,
        n_pending: int = 0,
        timestamp: int | None = None,
    ) -> None:
        """Record one portfolio snapshot."""
        if timestamp is None:
            timestamp = int(time.time())
        self._conn.execute(
            """INSERT INTO portfolio
               (timestamp, bot_name, total_value, cash, staked_value,
                n_positions, n_pending)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, self.bot_name, total_value, cash, staked_value,
             n_positions, n_pending),
        )
        self._conn.commit()

    def get_series(
        self,
        hours: float | None = None,
        since: int | None = None,
    ) -> list[tuple]:
        """Retrieve portfolio time-series.

        Args:
            hours: Last N hours of data. If None, return all.
            since: Unix timestamp to start from. Overrides hours.

        Returns list of (timestamp, total_value, cash, staked_value,
        n_positions, n_pending).
        """
        if since is not None:
            cutoff = since
        elif hours is not None:
            cutoff = int(time.time()) - int(hours * 3600)
        else:
            cutoff = 0

        return self._conn.execute(
            """SELECT timestamp, total_value, cash, staked_value,
                      n_positions, n_pending
               FROM portfolio
               WHERE bot_name = ? AND timestamp >= ?
               ORDER BY timestamp""",
            (self.bot_name, cutoff),
        ).fetchall()

    def get_latest(self) -> tuple | None:
        """Most recent portfolio snapshot."""
        row = self._conn.execute(
            """SELECT timestamp, total_value, cash, staked_value,
                      n_positions, n_pending
               FROM portfolio WHERE bot_name = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (self.bot_name,),
        ).fetchone()
        return row

    def get_all_bots(self) -> list[str]:
        """List all bot names with portfolio data (for comparison dashboards)."""
        rows = self._conn.execute(
            "SELECT DISTINCT bot_name FROM portfolio"
        ).fetchall()
        return [r[0] for r in rows]

    def cleanup(self, max_age_days: int = 90) -> None:
        """Remove entries older than max_age_days."""
        cutoff = int(time.time()) - max_age_days * 86400
        self._conn.execute(
            "DELETE FROM portfolio WHERE bot_name = ? AND timestamp < ?",
            (self.bot_name, cutoff),
        )
        self._conn.commit()

    def export_csv(self, path: str | Path, hours: float | None = None) -> None:
        """Export portfolio series to CSV for external plotting."""
        import csv
        rows = self.get_series(hours=hours)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "total_value", "cash", "staked_value",
                         "n_positions", "n_pending"])
            w.writerows(rows)

    def close(self) -> None:
        self._conn.close()
