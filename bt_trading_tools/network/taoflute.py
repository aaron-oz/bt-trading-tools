"""Read-only client for TAOFlute (taoflute.com).

TAOFlute is a Grafana instance fronting a read-only PostgreSQL database named
`trading`, with Bittensor-specific tables (per-subnet hourly pool state, top
alpha holders, delegation history, Discord messages, tweets, registrations,
news, dev activity, ...). We access it through Grafana's documented HTTP API
(`POST /api/ds/query`) as a paid Viewer. All access is SELECT-only.

This is generic, strategy-agnostic data infrastructure: it knows how to talk to
TAOFlute, not what to trade. Any bot or research script needing TAOFlute data
should go through this one client so the owner-agreed rate limits are enforced
in a single place.

────────────────────────────────────────────────────────────────────────────
LIMITS WE MUST RESPECT (agreed with the site owner, 2026-06) — do not loosen
without re-confirming with the owner:

  * **≤ 1 request / second, single-threaded. No parallel fan-out.**
  * ≤ 1000 requests / day total.
  * ≤ 100,000 rows per response.
  * Cache locally; never refetch unchanged data (use incremental / keyset pulls).
  * Exponential backoff + jitter on 429 / 5xx.
  * Traceable User-Agent so the owner can attribute usage.
  * If the owner reports performance impact, cut cadence immediately.

These are the DEFAULTS below. Construct with the defaults unless you have a
specific, owner-blessed reason to change them.
────────────────────────────────────────────────────────────────────────────

Credentials: TAOFLUTE_EMAIL / TAOFLUTE_PASSWORD, resolved from (first hit wins)
the shell environment, ~/.bittensor_env, then the legacy data/.env in the
alpha-trading tree.

See alpha-trading/docs/taoflute_exploration_report.md for the full schema map.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import requests

# Authenticated "BAG" org Postgres datasource UID (see exploration report §2.2).
_DS = {"type": "grafana-postgresql-datasource", "uid": "fesucg1x21mv4d"}
_BASE = "https://taoflute.com"

# Traceable UA so the owner can attribute usage (per the agreed terms).
_UA = "bittensor-alpha-trading-bot/1.0 (oz.aaron@pm.me)"

# Owner-agreed ceilings. Exposed as module constants so callers can assert.
MAX_ROWS_PER_RESPONSE = 100_000
MIN_QUERY_INTERVAL_S = 1.1   # ≤ 1 req/s with margin
MAX_QUERIES_PER_DAY = 1000

# Credential search path (first hit wins).
_ENV_CANDIDATES = [
    Path.home() / ".bittensor_env",
    Path.home() / "Dropbox" / "bittensor" / "alpha-trading" / "data" / ".env",
    Path.home() / "code" / "alpha-trading" / "data" / ".env",
]


def _resolve_credentials() -> tuple[str, str]:
    """Return (email, password) from env, then known dotenv files."""
    email = os.environ.get("TAOFLUTE_EMAIL")
    pw = os.environ.get("TAOFLUTE_PASSWORD")
    if email and pw:
        return email, pw
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        for path in _ENV_CANDIDATES:
            if path.exists():
                load_dotenv(path)
                email = email or os.environ.get("TAOFLUTE_EMAIL")
                pw = pw or os.environ.get("TAOFLUTE_PASSWORD")
                if email and pw:
                    break
    if not email or not pw:
        raise ValueError(
            "TAOFLUTE_EMAIL / TAOFLUTE_PASSWORD not found. Set them in the "
            "shell env, ~/.bittensor_env, or alpha-trading/data/.env."
        )
    return email, pw


class TaofluteClient:
    """Minimal, polite, read-only Grafana SQL client for TAOFlute.

    The defaults encode the owner-agreed limits. A single client instance is
    single-threaded by construction; do NOT share one across threads or run
    multiple concurrently (that would breach the no-parallel-fan-out term).
    """

    def __init__(
        self,
        min_query_interval_s: float = MIN_QUERY_INTERVAL_S,
        max_queries: int = MAX_QUERIES_PER_DAY,
        timeout_s: float = 90.0,
        max_retries: int = 5,
    ):
        if min_query_interval_s < 1.0:
            raise ValueError(
                "min_query_interval_s < 1.0 would exceed the owner-agreed "
                "1 req/s limit. Re-confirm with the owner before lowering."
            )
        self.min_query_interval_s = min_query_interval_s
        self.max_queries = max_queries
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._n_queries = 0
        self._last_query_t = 0.0
        self._session: Optional[requests.Session] = None

    def login(self) -> "TaofluteClient":
        email, pw = _resolve_credentials()
        s = requests.Session()
        s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
        r = s.post(_BASE + "/login", json={"user": email, "password": pw},
                   timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"TAOFlute login failed: {r.status_code} {r.text[:200]}")
        self._session = s
        return self

    def sql(self, raw_sql: str) -> tuple[list[str], list[tuple]]:
        """Run one SELECT. Returns (column_names, rows).

        Enforces the rate floor and the per-run query cap, and retries with
        exponential backoff + jitter on 429 / 5xx.
        """
        if self._session is None:
            self.login()
        if self._n_queries >= self.max_queries:
            raise RuntimeError(
                f"TAOFlute query cap reached ({self.max_queries}/day). "
                "Raise max_queries only within the owner-agreed daily limit."
            )
        wait = self.min_query_interval_s - (time.monotonic() - self._last_query_t)
        if wait > 0:
            time.sleep(wait)

        body = {
            "queries": [{"refId": "A", "datasource": _DS,
                         "rawSql": raw_sql, "format": "table"}],
            "from": "now-1h", "to": "now",
        }
        resp = None
        for attempt in range(self.max_retries):
            resp = self._session.post(
                _BASE + "/api/ds/query", data=json.dumps(body),
                headers={"Content-Type": "application/json"}, timeout=self.timeout_s,
            )
            self._last_query_t = time.monotonic()
            if resp.status_code == 200:
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = min(60.0, 2 ** attempt) + random.uniform(0, 1.0)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"query failed {resp.status_code}: {resp.text[:300]}")
        self._n_queries += 1
        if resp is None or resp.status_code != 200:
            code = resp.status_code if resp is not None else "no-response"
            raise RuntimeError(f"query failed after {self.max_retries} retries: {code}")

        frames = resp.json()["results"]["A"].get("frames", [])
        if not frames:
            return [], []
        frame = frames[0]
        cols = [fd["name"] for fd in frame["schema"]["fields"]]
        vals = frame["data"]["values"]
        rows = list(zip(*vals)) if vals else []
        if len(rows) >= MAX_ROWS_PER_RESPONSE:
            # Soft guard: a full response at the ceiling means the query was
            # probably truncated. Callers should paginate with LIMIT/keyset.
            raise RuntimeError(
                f"response hit the {MAX_ROWS_PER_RESPONSE}-row ceiling; "
                "paginate this query (LIMIT + keyset on an indexed column)."
            )
        return cols, rows

    def query_dicts(self, raw_sql: str) -> list[dict]:
        """Convenience: run a SELECT and return a list of row dicts."""
        cols, rows = self.sql(raw_sql)
        return [dict(zip(cols, r)) for r in rows]

    @property
    def n_queries(self) -> int:
        return self._n_queries
