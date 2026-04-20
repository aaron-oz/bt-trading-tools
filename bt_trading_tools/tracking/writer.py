"""
TradeLogWriter — non-blocking JSONL writer for the v1 trade log schema.

Hot-path API (`log_trade`, `log_mtm`) enqueues a dict and returns immediately.
A background thread drains the queue, validates each record, writes valid
records to the main JSONL file, and dumps invalid records (plus error
details) to a sibling `<name>.errors.jsonl` without stopping.

Non-blocking is a hard contract: bots running in block-subscription loops
cannot afford disk I/O on the hot path.
"""
from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from bt_trading_tools.tracking.schema import Record, SCHEMA_VERSION

_log = logging.getLogger(__name__)
_record_adapter = TypeAdapter(Record)

_STOP_SENTINEL = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class TradeLogWriter:
    """Non-blocking writer for the v1 trade log schema.

    Usage::

        writer = TradeLogWriter("/path/to/trade_log.jsonl", bot_id="autobot")
        writer.log_trade({...})   # returns immediately
        writer.log_mtm({...})
        writer.close()            # flushes and stops worker

    Missing `schema_version`, `bot_id`, `timestamp`, and `record_type` are
    auto-filled. Callers that want to override any of them can still provide
    them explicitly.
    """

    def __init__(
        self,
        path: str | Path,
        bot_id: str,
        *,
        batch_size: int = 128,
        flush_interval_s: float = 1.0,
        auto_close_on_exit: bool = True,
    ):
        self.path = Path(path)
        self.errors_path = self.path.with_name(
            self.path.stem + ".errors" + self.path.suffix
        )
        self.bot_id = bot_id
        self._queue: queue.Queue[object] = queue.Queue()
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._stop = threading.Event()
        self._flush_errors = 0
        self._flush_errors_lock = threading.Lock()

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._run,
            name=f"TradeLogWriter-{bot_id}",
            daemon=True,
        )
        self._worker.start()

        if auto_close_on_exit:
            atexit.register(self.close)

    # ── Hot-path API ──────────────────────────────────────────────

    def log_trade(self, record: dict[str, Any]) -> None:
        """Enqueue a trade record. Non-blocking."""
        rec = {**record, "record_type": "trade"}
        self._enqueue(rec)

    def log_mtm(self, record: dict[str, Any]) -> None:
        """Enqueue an MTM sample record. Non-blocking."""
        rec = {**record, "record_type": "mtm_sample"}
        self._enqueue(rec)

    def log_portfolio_snapshot(self, record: dict[str, Any]) -> None:
        """Enqueue a portfolio snapshot record. Non-blocking.

        Required fields (besides the auto-filled ones): is_paper,
        capital_tao, positions_value_tao, total_equity_tao,
        realized_pnl_to_date_tao, open_positions_count.
        """
        rec = {**record, "record_type": "portfolio_snapshot"}
        self._enqueue(rec)

    def _enqueue(self, record: dict[str, Any]) -> None:
        record.setdefault("schema_version", SCHEMA_VERSION)
        record.setdefault("bot_id", self.bot_id)
        record.setdefault("timestamp", _utc_now_iso())
        self._queue.put(record)

    # ── Lifecycle ─────────────────────────────────────────────────

    @property
    def flush_errors_since_start(self) -> int:
        """Count of records that failed validation since process start."""
        with self._flush_errors_lock:
            return self._flush_errors

    def flush(self, timeout: float | None = None) -> None:
        """Wait until the queue is drained. Does not stop the worker."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._queue.empty():
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def close(self, timeout: float = 5.0) -> None:
        """Flush and stop the background worker. Safe to call multiple times."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(_STOP_SENTINEL)
        self._worker.join(timeout=timeout)

    # ── Background worker ─────────────────────────────────────────

    def _run(self) -> None:
        buffer: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        stopping = False
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                item = None

            if item is _STOP_SENTINEL:
                stopping = True
                # Pull anything else still queued.
                while True:
                    try:
                        extra = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if extra is _STOP_SENTINEL:
                        continue
                    if isinstance(extra, dict):
                        buffer.append(extra)
                if buffer:
                    self._flush_batch(buffer)
                return

            if isinstance(item, dict):
                buffer.append(item)

            now = time.monotonic()
            should_flush = (
                len(buffer) >= self._batch_size
                or (now - last_flush) >= self._flush_interval_s
            )
            if should_flush and buffer:
                self._flush_batch(buffer)
                buffer = []
                last_flush = now

            if stopping and not buffer:
                return

    def _flush_batch(self, records: list[dict[str, Any]]) -> None:
        valid_lines: list[str] = []
        errors: list[tuple[dict[str, Any], str]] = []
        for rec in records:
            try:
                validated = _record_adapter.validate_python(rec)
                valid_lines.append(validated.model_dump_json(exclude_none=False))
            except ValidationError as e:
                errors.append((rec, e.json()))
            except Exception as e:  # defensive
                errors.append((rec, f"{type(e).__name__}: {e}"))

        if valid_lines:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(valid_lines) + "\n")

        if errors:
            with open(self.errors_path, "a", encoding="utf-8") as f:
                for raw, detail in errors:
                    f.write(
                        json.dumps(
                            {"record": raw, "error": detail},
                            default=str,
                        )
                        + "\n"
                    )
            with self._flush_errors_lock:
                self._flush_errors += len(errors)
            _log.warning(
                "[%s] TradeLogWriter rejected %d invalid record(s); "
                "total since start: %d",
                self.bot_id,
                len(errors),
                self._flush_errors,
            )
