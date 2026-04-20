"""
Readers and validators for the v1 trade log schema.

Readers are permissive (tolerate missing / extra fields, skip bad JSON).
The validator is strict — it reports every record that fails schema checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from bt_trading_tools.tracking.schema import Record

_record_adapter = TypeAdapter(Record)


def iter_trade_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one raw dict per line. Skips blanks and JSON decode failures."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_trade_log(path: str | Path) -> pd.DataFrame:
    """Load a JSONL trade log into a DataFrame. Tolerates missing fields.

    Timestamps are converted to pandas UTC datetimes; mis-formatted
    timestamps become NaT.
    """
    rows = list(iter_trade_log(path))
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


@dataclass
class ValidationIssue:
    line_no: int
    record: dict[str, Any]
    error: str

    def __str__(self) -> str:  # compact human repr
        return f"line {self.line_no}: {self.error}"


def validate_trade_log(path: str | Path) -> list[ValidationIssue]:
    """Validate every record. Returns issues in file order; empty list = clean."""
    issues: list[ValidationIssue] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(line_no, {}, f"json decode: {e}"))
                continue
            try:
                _record_adapter.validate_python(rec)
            except ValidationError as e:
                issues.append(ValidationIssue(line_no, rec, e.json()))
    return issues
