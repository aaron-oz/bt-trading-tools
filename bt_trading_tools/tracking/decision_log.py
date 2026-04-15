"""
DecisionLog — JSON-lines per-tick decision trail for full replay.

Every tick the bot writes one line capturing what it saw, what it decided,
and why.  This is the primary debugging and iteration tool: given any trade
(win or loss), you can trace back to the exact tick context that produced it.

File format: one JSON object per line (JSON-lines / .jsonl).
"""

import json
import time
from pathlib import Path
from typing import Any


class DecisionLog:
    """Append-only JSON-lines decision log.

    Usage::

        dlog = DecisionLog("/path/to/decisions.jsonl", bot_name="autobot")

        # Each tick:
        dlog.record_tick(
            tick=42,
            portfolio_value=52.3,
            cash=10.0,
            n_positions=5,
            decisions=[
                {"netuid": 107, "action": "hold", "reason": "below_threshold",
                 "pct_change": 1.2, "threshold": 7.0},
                {"netuid": 99, "action": "sell", "reason": "rally_sell",
                 "pct_change": 8.4, "threshold": 7.0, "tao_amount": 0.5},
            ],
            market_snapshot={"n_subnets": 128, "universe_size": 15},
        )
    """

    def __init__(self, path: str | Path, bot_name: str):
        self.bot_name = bot_name
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", buffering=1)  # line-buffered

    def record_tick(
        self,
        tick: int,
        portfolio_value: float,
        cash: float,
        n_positions: int,
        decisions: list[dict[str, Any]],
        market_snapshot: dict[str, Any] | None = None,
        timestamp: int | None = None,
    ) -> None:
        """Write one tick's full decision context."""
        if timestamp is None:
            timestamp = int(time.time())
        record = {
            "ts": timestamp,
            "tick": tick,
            "bot": self.bot_name,
            "pv": round(portfolio_value, 6),
            "cash": round(cash, 6),
            "n_pos": n_positions,
            "decisions": decisions,
        }
        if market_snapshot:
            record["market"] = market_snapshot
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._file.close()

    # ── Replay helpers (for offline analysis) ────────────────────

    @staticmethod
    def iter_ticks(path: str | Path, bot_name: str | None = None):
        """Iterate over all ticks in a decision log file.

        Yields dicts. Optionally filter by bot_name.
        """
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if bot_name and record.get("bot") != bot_name:
                    continue
                yield record

    @staticmethod
    def find_tick(path: str | Path, tick: int, bot_name: str | None = None) -> dict | None:
        """Find a specific tick's decision record."""
        for record in DecisionLog.iter_ticks(path, bot_name):
            if record.get("tick") == tick:
                return record
        return None

    @staticmethod
    def find_trades_for_netuid(
        path: str | Path, netuid: int, bot_name: str | None = None,
    ) -> list[dict]:
        """Find all ticks where a trade was executed on a given subnet."""
        results = []
        for record in DecisionLog.iter_ticks(path, bot_name):
            for d in record.get("decisions", []):
                if d.get("netuid") == netuid and d.get("action") not in ("hold", "skip"):
                    results.append(record)
                    break
        return results
