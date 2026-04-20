"""
Fee receipt schema + writer.

A `FeeReceipt` is a raw chain-event forensic record produced by
TradeExecutor after every extrinsic submission (success, failure, or
timeout). It lives in a separate JSONL log from the trade log; the two
join on `(bot_id, chain_tx_hash)`.

Separation of concerns:
    - Trade log  = bot decision + execution summary (what the bot chose,
                   what the chain returned as a compact predicted-fee
                   snapshot via FeeModel.quote or the chain receipt).
    - Fee log    = raw chain-event data (observed fees, full extrinsic
                   response, parse errors). Used for calibration and deep
                   audit. Its schema evolves as our parser learns.

Reuses the non-blocking queue + validation-on-flush architecture of
`TradeLogWriter`.
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
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


FEE_RECEIPT_SCHEMA_VERSION = 1

_log = logging.getLogger(__name__)
_STOP_SENTINEL = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class FeeReceipt(BaseModel):
    """One chain-event receipt for a submitted extrinsic.

    Written by TradeExecutor after every submit (success / failure /
    timeout). Parsing may fail — in that case `parse_error` is populated
    and `raw_extrinsic_result` preserves the full response for re-derivation.
    """
    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(ge=1)
    bot_id: str
    timestamp: str                         # ISO 8601 UTC
    wallet_coldkey: Optional[str] = None
    wallet_hotkey: Optional[str] = None
    extrinsic_type: str                    # e.g. "proxy.proxy(add_stake)"
    netuid: int = Field(ge=0)
    tao_amount_requested: Optional[float] = Field(default=None, ge=0)
    alpha_amount_requested: Optional[float] = Field(default=None, ge=0)
    rate_tolerance: Optional[float] = None
    extrinsic_status: Literal["success", "failed", "timeout"]
    chain_tx_hash: Optional[str] = None    # join key to trade log
    observed_swap_fee_tao: Optional[float] = Field(default=None, ge=0)
    observed_gas_fee_tao: Optional[float] = Field(default=None, ge=0)
    observed_proxy_fee_tao: Optional[float] = Field(default=None, ge=0)
    pool_tao_at_submit: Optional[float] = Field(default=None, ge=0)
    pool_alpha_at_submit: Optional[float] = Field(default=None, ge=0)
    pool_tao_post: Optional[float] = Field(default=None, ge=0)
    pool_alpha_post: Optional[float] = Field(default=None, ge=0)
    raw_extrinsic_result: Optional[dict] = None
    parse_error: Optional[str] = None


_receipt_adapter = TypeAdapter(FeeReceipt)


class FeeReceiptWriter:
    """Non-blocking JSONL writer for FeeReceipt.

    Mirrors TradeLogWriter's contract: enqueue-and-return from the hot
    path, validate on a background flusher, route malformed records to a
    sibling `<name>.errors.jsonl`. Always safe to call from inside a
    trading loop — disk I/O never happens on the caller's thread.
    """

    def __init__(
        self,
        path: str | Path,
        bot_id: str,
        *,
        batch_size: int = 64,
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
            name=f"FeeReceiptWriter-{bot_id}",
            daemon=True,
        )
        self._worker.start()

        if auto_close_on_exit:
            atexit.register(self.close)

    # ── Hot-path API ──────────────────────────────────────────────

    def log_receipt(self, record: dict[str, Any]) -> None:
        """Enqueue a FeeReceipt dict. Non-blocking."""
        record = {**record}
        record.setdefault("schema_version", FEE_RECEIPT_SCHEMA_VERSION)
        record.setdefault("bot_id", self.bot_id)
        record.setdefault("timestamp", _utc_now_iso())
        self._queue.put(record)

    # ── Lifecycle ─────────────────────────────────────────────────

    @property
    def flush_errors_since_start(self) -> int:
        with self._flush_errors_lock:
            return self._flush_errors

    def flush(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._queue.empty():
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def close(self, timeout: float = 5.0) -> None:
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
                validated = _receipt_adapter.validate_python(rec)
                valid_lines.append(validated.model_dump_json(exclude_none=False))
            except ValidationError as e:
                errors.append((rec, e.json()))
            except Exception as e:
                errors.append((rec, f"{type(e).__name__}: {e}"))

        if valid_lines:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(valid_lines) + "\n")

        if errors:
            with open(self.errors_path, "a", encoding="utf-8") as f:
                for raw, detail in errors:
                    f.write(
                        json.dumps({"record": raw, "error": detail}, default=str) + "\n"
                    )
            with self._flush_errors_lock:
                self._flush_errors += len(errors)
            _log.warning(
                "[%s] FeeReceiptWriter rejected %d invalid receipt(s); "
                "total since start: %d",
                self.bot_id, len(errors), self._flush_errors,
            )


# ── Readers ───────────────────────────────────────────────────────

def iter_fee_receipts(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one raw receipt dict per line; skips blanks and JSON errors."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def validate_fee_receipt_log(path: str | Path) -> list[tuple[int, dict, str]]:
    """Strict validation. Returns (line_no, record, error) tuples; empty = clean."""
    issues = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append((line_no, {}, f"json decode: {e}"))
                continue
            try:
                _receipt_adapter.validate_python(rec)
            except ValidationError as e:
                issues.append((line_no, rec, e.json()))
    return issues
