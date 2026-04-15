"""
EventLog — JSON-lines log for significant events.

Unlike DecisionLog (every tick), EventLog records only notable events:
trades executed, errors, reconnections, universe refreshes, param changes,
bot start/stop.  This is what a human reads to understand what happened.
"""

import json
import time
from pathlib import Path
from typing import Any


# Severity levels (ascending)
DEBUG = "debug"
INFO = "info"
TRADE = "trade"
WARNING = "warning"
ERROR = "error"


class EventLog:
    """Append-only event log.

    Usage::

        elog = EventLog("/path/to/events.jsonl", bot_name="autobot")
        elog.trade("rally_sell", netuid=107, tao=0.5, pnl=0.12,
                   detail={"pct_change": 8.4, "hold_blocks": 3})
        elog.info("universe_refresh", detail={"new_size": 15, "added": [92, 100]})
        elog.error("subtensor_timeout", detail={"retry": 3, "wait_s": 8})
    """

    def __init__(self, path: str | Path, bot_name: str):
        self.bot_name = bot_name
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", buffering=1)

    def _write(
        self, level: str, event: str,
        detail: dict[str, Any] | None = None,
        timestamp: int | None = None,
    ) -> None:
        if timestamp is None:
            timestamp = int(time.time())
        record: dict[str, Any] = {
            "ts": timestamp,
            "bot": self.bot_name,
            "level": level,
            "event": event,
        }
        if detail:
            record["detail"] = detail
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def debug(self, event: str, **kwargs: Any) -> None:
        self._write(DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._write(INFO, event, **kwargs)

    def trade(
        self, event: str, *,
        netuid: int | None = None,
        tao: float | None = None,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a trade event with standard fields."""
        detail = kwargs.pop("detail", {}) or {}
        if netuid is not None:
            detail["netuid"] = netuid
        if tao is not None:
            detail["tao"] = round(tao, 6)
        if pnl is not None:
            detail["pnl"] = round(pnl, 6)
        self._write(TRADE, event, detail=detail, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._write(WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._write(ERROR, event, **kwargs)

    def close(self) -> None:
        self._file.close()

    # ── Replay helpers ───────────────────────────────────────────

    @staticmethod
    def iter_events(
        path: str | Path,
        bot_name: str | None = None,
        level: str | None = None,
    ):
        """Iterate over events. Optionally filter by bot and/or level."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if bot_name and record.get("bot") != bot_name:
                    continue
                if level and record.get("level") != level:
                    continue
                yield record
