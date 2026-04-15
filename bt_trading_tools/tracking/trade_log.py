"""
TradeLog — SQLite-backed trade recording with cost-basis tracking.

Each bot gets its own TradeLog instance pointed at a db file.
The same schema works for both live and backtesting, so one analysis
pipeline covers both.
"""

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TradeRecord:
    """A single executed trade."""
    timestamp: int
    bot_name: str
    trade_type: str           # "buy" or "sell"
    netuid: int
    tao_amount: float
    alpha_amount: float
    price: float              # effective price (tao/alpha)
    slippage: float           # actual slippage percentage
    hotkey: str               # validator hotkey used
    reason: str               # why: "rally_sell", "reload", "stop_loss", "inventory_build", ...
    signal_data: dict | None  # the signals that triggered it (pct_change, threshold, etc.)


class TradeLog:
    """SQLite trade log with cost-basis and P&L tracking.

    Usage::

        log = TradeLog("/path/to/trades.db", bot_name="doubledip")
        log.record_trade("buy", netuid=107, tao_amount=0.5, alpha_amount=100,
                         price=0.005, slippage=0.3, hotkey="5G3w...",
                         reason="inventory_build", signal_data={"pct_change": -1.2})
        basis = log.get_cost_basis(107)
    """

    def __init__(self, db_path: str | Path, bot_name: str):
        self.bot_name = bot_name
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                bot_name TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                netuid INTEGER NOT NULL,
                tao_amount REAL NOT NULL,
                alpha_amount REAL NOT NULL,
                price REAL NOT NULL,
                slippage REAL,
                hotkey TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                signal_data TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_bot_netuid
                ON trades(bot_name, netuid);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp);

            CREATE TABLE IF NOT EXISTS position_snapshots (
                timestamp INTEGER NOT NULL,
                bot_name TEXT NOT NULL,
                netuid INTEGER NOT NULL,
                alpha_held REAL NOT NULL,
                price REAL NOT NULL,
                tao_invested REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_bot_netuid_ts
                ON position_snapshots(bot_name, netuid, timestamp);
        """)

    # ── Trade recording ──────────────────────────────────────────

    def record_trade(
        self,
        trade_type: str,
        netuid: int,
        tao_amount: float,
        alpha_amount: float,
        price: float,
        slippage: float,
        hotkey: str,
        reason: str = "",
        signal_data: dict | None = None,
        timestamp: int | None = None,
    ) -> TradeRecord:
        """Record a trade. Returns the TradeRecord written."""
        if timestamp is None:
            timestamp = int(time.time())
        sig_json = json.dumps(signal_data) if signal_data else None
        self._conn.execute(
            """INSERT INTO trades
               (timestamp, bot_name, trade_type, netuid, tao_amount,
                alpha_amount, price, slippage, hotkey, reason, signal_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, self.bot_name, trade_type, netuid, tao_amount,
             alpha_amount, price, slippage, hotkey, reason, sig_json),
        )
        self._conn.commit()
        return TradeRecord(
            timestamp=timestamp, bot_name=self.bot_name, trade_type=trade_type,
            netuid=netuid, tao_amount=tao_amount, alpha_amount=alpha_amount,
            price=price, slippage=slippage, hotkey=hotkey,
            reason=reason, signal_data=signal_data,
        )

    def insert_baseline_buy(
        self, netuid: int, alpha_amount: float, price: float,
        hotkey: str, timestamp: int,
    ) -> None:
        """Synthetic buy to establish cost basis for pre-existing positions."""
        self.record_trade(
            "buy", netuid, tao_amount=alpha_amount * price,
            alpha_amount=alpha_amount, price=price, slippage=0.0,
            hotkey=hotkey, reason="baseline", timestamp=timestamp,
        )

    # ── Cost basis & P&L ─────────────────────────────────────────

    def get_cost_basis(self, netuid: int) -> dict[str, float] | None:
        """Weighted-average cost basis for a subnet.

        Returns dict with: avg_buy_price, total_tao_invested, total_alpha_held,
        realized_pnl, total_tao_received.  Or None if no trades.
        """
        rows = self._conn.execute(
            """SELECT trade_type, tao_amount, alpha_amount
               FROM trades WHERE bot_name = ? AND netuid = ?
               ORDER BY timestamp""",
            (self.bot_name, netuid),
        ).fetchall()

        if not rows:
            return None

        total_tao_invested = 0.0
        total_alpha_held = 0.0
        total_tao_received = 0.0
        realized_pnl = 0.0

        for trade_type, tao_amount, alpha_amount in rows:
            if trade_type == "buy":
                total_tao_invested += tao_amount
                total_alpha_held += alpha_amount
            elif trade_type == "sell":
                if total_alpha_held > 0:
                    fraction_sold = min(alpha_amount / total_alpha_held, 1.0)
                    cost_of_sold = total_tao_invested * fraction_sold
                    realized_pnl += tao_amount - cost_of_sold
                    total_tao_invested -= cost_of_sold
                    total_alpha_held -= alpha_amount
                    total_tao_received += tao_amount
                else:
                    realized_pnl += tao_amount
                    total_tao_received += tao_amount

        # Clamp float noise
        if total_alpha_held < 1e-9:
            total_alpha_held = 0.0
        if total_tao_invested < 1e-12:
            total_tao_invested = 0.0

        avg_buy_price = (
            (total_tao_invested / total_alpha_held) if total_alpha_held > 0
            else 0.0
        )
        return {
            "avg_buy_price": avg_buy_price,
            "total_tao_invested": total_tao_invested,
            "total_alpha_held": total_alpha_held,
            "realized_pnl": realized_pnl,
            "total_tao_received": total_tao_received,
        }

    def get_all_cost_bases(self) -> dict[int, dict[str, float]]:
        """Cost basis for every traded subnet."""
        rows = self._conn.execute(
            "SELECT DISTINCT netuid FROM trades WHERE bot_name = ?",
            (self.bot_name,),
        ).fetchall()
        result = {}
        for (netuid,) in rows:
            basis = self.get_cost_basis(netuid)
            if basis is not None:
                result[netuid] = basis
        return result

    def get_portfolio_summary(self) -> dict[str, float]:
        """Aggregate: total_invested, total_received, realized_pnl."""
        bases = self.get_all_cost_bases()
        total_invested = sum(b["total_tao_invested"] for b in bases.values())
        total_received = sum(b["total_tao_received"] for b in bases.values())
        realized_pnl = sum(b["realized_pnl"] for b in bases.values())
        return {
            "total_invested": total_invested,
            "total_received": total_received,
            "realized_pnl": realized_pnl,
        }

    # ── Snapshots & delta-P&L ────────────────────────────────────

    def record_snapshot(
        self, netuid: int, alpha_held: float, price: float,
        tao_invested: float, timestamp: int | None = None,
    ) -> None:
        """Record a position snapshot for delta-P&L tracking."""
        if timestamp is None:
            timestamp = int(time.time())
        self._conn.execute(
            """INSERT INTO position_snapshots
               (timestamp, bot_name, netuid, alpha_held, price, tao_invested)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (timestamp, self.bot_name, netuid, alpha_held, price, tao_invested),
        )
        self._conn.commit()

    def record_snapshots_bulk(
        self, entries: list[tuple[int, float, float, float]],
        timestamp: int | None = None,
    ) -> None:
        """Batch snapshots. entries: [(netuid, alpha_held, price, tao_invested), ...]."""
        if timestamp is None:
            timestamp = int(time.time())
        self._conn.executemany(
            """INSERT INTO position_snapshots
               (timestamp, bot_name, netuid, alpha_held, price, tao_invested)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(timestamp, self.bot_name, n, a, p, t) for n, a, p, t in entries],
        )
        self._conn.commit()

    def get_pnl_delta(self, netuid: int, hours: float) -> tuple[float | None, float | None]:
        """Change in unrealized P&L over a time window.

        Returns (delta_tao, pct_of_invested) or (None, None).
        """
        now = int(time.time())
        lookback_secs = int(hours * 3600)
        smooth_secs = max(lookback_secs // 4, 180)

        recent = self._conn.execute(
            """SELECT alpha_held * price - tao_invested, tao_invested
               FROM position_snapshots
               WHERE bot_name = ? AND netuid = ?
                 AND timestamp >= ? AND timestamp <= ?""",
            (self.bot_name, netuid, now - smooth_secs, now),
        ).fetchall()

        earlier_center = now - lookback_secs
        earlier = self._conn.execute(
            """SELECT alpha_held * price - tao_invested, tao_invested
               FROM position_snapshots
               WHERE bot_name = ? AND netuid = ?
                 AND timestamp >= ? AND timestamp <= ?""",
            (self.bot_name, netuid,
             earlier_center - smooth_secs // 2,
             earlier_center + smooth_secs // 2),
        ).fetchall()

        if not recent or not earlier:
            return None, None

        avg_pnl_now = sum(r[0] for r in recent) / len(recent)
        avg_invested_now = sum(r[1] for r in recent) / len(recent)
        avg_pnl_earlier = sum(r[0] for r in earlier) / len(earlier)

        delta = avg_pnl_now - avg_pnl_earlier
        pct = (delta / avg_invested_now * 100) if avg_invested_now > 0 else None
        return delta, pct

    def get_all_pnl_deltas(self, hours: float) -> dict[int, tuple[float, float | None]]:
        """Delta-P&L for all subnets with snapshot data."""
        rows = self._conn.execute(
            "SELECT DISTINCT netuid FROM position_snapshots WHERE bot_name = ?",
            (self.bot_name,),
        ).fetchall()
        result = {}
        for (netuid,) in rows:
            delta, pct = self.get_pnl_delta(netuid, hours)
            if delta is not None:
                result[netuid] = (delta, pct)
        return result

    def cleanup_old_snapshots(self, max_age_days: int = 10) -> None:
        """Remove snapshots older than max_age_days."""
        cutoff = int(time.time()) - max_age_days * 86400
        self._conn.execute(
            "DELETE FROM position_snapshots WHERE bot_name = ? AND timestamp < ?",
            (self.bot_name, cutoff),
        )
        self._conn.commit()

    # ── Query helpers ────────────────────────────────────────────

    def get_recent_trades(self, limit: int = 20) -> list[TradeRecord]:
        """Most recent trades for this bot."""
        rows = self._conn.execute(
            """SELECT timestamp, bot_name, trade_type, netuid, tao_amount,
                      alpha_amount, price, slippage, hotkey, reason, signal_data
               FROM trades WHERE bot_name = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (self.bot_name, limit),
        ).fetchall()
        return [
            TradeRecord(
                timestamp=r[0], bot_name=r[1], trade_type=r[2], netuid=r[3],
                tao_amount=r[4], alpha_amount=r[5], price=r[6], slippage=r[7],
                hotkey=r[8], reason=r[9],
                signal_data=json.loads(r[10]) if r[10] else None,
            )
            for r in rows
        ]

    def delete_subnet_trades(self, netuid: int) -> None:
        """Delete all trades for a subnet (for re-initialization)."""
        self._conn.execute(
            "DELETE FROM trades WHERE bot_name = ? AND netuid = ?",
            (self.bot_name, netuid),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
