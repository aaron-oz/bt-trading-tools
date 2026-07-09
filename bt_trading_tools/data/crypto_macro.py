"""
Crypto macro data loaders for S11 (crypto macro regime signal).

Pure data fetch + parse. No thresholds, no signal logic (that belongs in
bt-strategy). These utilities are **public-safe** (no strategy encoded).

Exports:

- ``load_btc_usd_hourly``        — local CSV loader for Binance BTCUSDT klines
                                    (handles both hourly and 5-min inputs,
                                    resampling to hourly).
- ``load_funding_rates``         — local CSV loader for per-symbol perpetual
                                    funding rate history.
- ``fetch_btc_usd_klines``       — Binance US hourly BTC/USD klines
                                    (historical backfill; 1000 rows/page).
- ``fetch_funding_rate_history`` — Binance USD-M perpetual funding history
                                    for a symbol (BTCUSDT, ETHUSDT). Falls
                                    back to MEXC contract API on Binance 451
                                    geo-block (e.g. EU VPS hosts).
- ``fetch_funding_rate_history_mexc`` — MEXC contract funding history;
                                    geo-block-resistant alternative to
                                    Binance. ~1.5y depth.
- ``fetch_btc_dominance_snapshot`` — current BTC.D from CoinGecko or
                                      CoinPaprika (free, no key).
- ``append_btc_dominance_sample``  — sample BTC.D and append to a CSV (daily
                                      cron pattern; BTC.D moves on a weekly
                                      timescale per meta-agent plan §4.8).
- ``load_btc_dominance_series``    — local CSV loader for the accumulated BTC.D
                                      file.
- ``average_funding_rate``         — trailing-window mean over a funding-rate
                                      frame (convenience for S11).
- ``FetchProvenance``              — dataclass capturing source URL, fetch
                                      timestamp, row count, file SHA-256
                                      (satisfies experiment-reproducibility
                                      protocol).

Design notes:

* Network fetchers return ``list[dict]`` rows with ISO-8601 ``time`` fields.
  Callers persist. ``persist_rows_csv`` is a helper that writes + checksums.
* Loaders degrade gracefully — they return ``None`` or an empty DataFrame if
  the file is missing, so callers can decide how to surface staleness.
* No credentials required. Endpoint reachability varies by region:
    - Spot klines: ``api.binance.us``    (US, EU)
                   ``api.binance.com`` returns 451 from US.
    - Funding:     ``fapi.binance.com``  (US-reachable; some EU VPS hosts
                   like Contabo Germany return 451 — fallback to MEXC).
                   ``contract.mexc.com`` (US, EU; ~1.5y history depth).
    - BTC.D:       CoinGecko ``/global`` or CoinPaprika ``/global`` (free).
    - **NOT reachable from EU VPS:** ``api.bybit.com`` (CloudFront
      country-block), ``fapi.binance.com`` (Binance Futures geo-block).
* BTC.D historical backfill is not available on any audited free tier.
  For Phase 1 we sample daily going forward; that's sufficient given the
  signal only uses the trailing-7d direction.

See ``docs/s11_data_source_verification.md`` for the audit behind these choices.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import pandas as pd

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

BINANCE_US_SPOT_KLINES = "https://api.binance.us/api/v3/klines"
BINANCE_FUTURES_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"

# MEXC contract (perpetual) funding history — used as fallback when
# fapi.binance.com geo-blocks (e.g. from EU/Germany VPS providers like
# Contabo). Endpoint accepts only underscore-separated symbol form
# (BTC_USDT, ETH_USDT). Same 8h funding cadence as Binance USD-M.
# Page size up to 100; returns descending order by settleTime.
MEXC_FUNDING_HISTORY = "https://contract.mexc.com/api/v1/contract/funding_rate/history"

COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
COINPAPRIKA_GLOBAL = "https://api.coinpaprika.com/v1/global"

DEFAULT_REQUEST_TIMEOUT = 15.0  # seconds
DEFAULT_USER_AGENT = "bt-trading-tools/0.1 (+crypto_macro_loader)"
MAX_RETRIES = 3


@dataclass
class FetchProvenance:
    """Lightweight record of where data came from + when.

    Meta-agent reproducibility protocol expects us to log source URL, fetch
    timestamp, and for file outputs, a SHA-256 checksum. Callers can log this
    dataclass directly.
    """

    source_url: str
    fetched_at: datetime  # UTC
    row_count: int
    file_path: Optional[str] = None
    file_sha256: Optional[str] = None


def _http_get_json(
    url: str, params: Optional[dict] = None, timeout: float = DEFAULT_REQUEST_TIMEOUT
) -> dict | list:
    """Minimal JSON GET with retry. Raises on unrecoverable failure."""
    full = url
    if params:
        from urllib.parse import urlencode
        full = f"{url}?{urlencode(params)}"
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        req = urllib_request.Request(full, headers={"User-Agent": DEFAULT_USER_AGENT})
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                payload = resp.read()
            return json.loads(payload)
        except HTTPError as e:
            last_exc = e
            if e.code in (418, 429):
                time.sleep(5 * (2 ** attempt))
                continue
            if e.code in (451, 403):
                raise RuntimeError(f"Endpoint blocked from this IP ({e.code}): {url}") from e
            if 400 <= e.code < 500:
                raise
            time.sleep(2 * (2 ** attempt))
        except URLError as e:
            last_exc = e
            time.sleep(2 * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ms_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


# ---------------------------------------------------------------------------
# BTC/USD spot klines (Binance US)
# ---------------------------------------------------------------------------

KLINE_COLUMNS = [
    "time", "open", "high", "low", "close", "volume",
    "quote_volume", "n_trades", "taker_buy_volume",
]


def fetch_btc_usd_klines(
    start: datetime,
    end: datetime,
    interval: str = "1h",
    max_candles_per_req: int = 1000,
    request_delay_s: float = 0.3,
) -> tuple[list[dict], FetchProvenance]:
    """Fetch BTC/USD klines from Binance US.

    Pagination is handled internally. Returns rows + provenance; caller
    persists (use ``persist_rows_csv``).

    Args:
        start: inclusive UTC start. ``tzinfo`` must be set.
        end:   exclusive UTC end.  ``tzinfo`` must be set.
        interval: Binance kline interval — ``1h`` recommended for S11 DC.
        max_candles_per_req: Binance hard max is 1000.
        request_delay_s: polite delay between pages.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes")
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)

    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": max_candles_per_req,
        }
        page = _http_get_json(BINANCE_US_SPOT_KLINES, params=params)
        if not isinstance(page, list) or not page:
            break
        for c in page:
            rows.append({
                "time": _ms_to_iso_utc(c[0]),
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
                "quote_volume": c[7],
                "n_trades": c[8],
                "taker_buy_volume": c[9],
            })
        cursor = page[-1][0] + 1
        if len(page) < max_candles_per_req:
            break
        time.sleep(request_delay_s)

    prov = FetchProvenance(
        source_url=BINANCE_US_SPOT_KLINES,
        fetched_at=datetime.now(timezone.utc),
        row_count=len(rows),
    )
    return rows, prov


def load_btc_usd_hourly(path: str | Path) -> Optional[pd.DataFrame]:
    """Load a Binance kline CSV and return hourly OHLCV.

    Accepts either an already-hourly CSV or a 5-min CSV (the existing project
    layout stores BTC as ``data/binance/btcusdt_5m.csv``). When 5-min is
    detected, resamples to hourly.

    Returns a DataFrame indexed by UTC timestamp with columns
    ``[open, high, low, close, volume]``. Returns ``None`` if file missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.set_index("time").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    if len(df) < 2:
        return df
    median_gap = (df.index[1:] - df.index[:-1]).to_series().median()
    if median_gap < pd.Timedelta(minutes=50):
        hourly = df.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        return hourly
    return df


# ---------------------------------------------------------------------------
# Perpetual funding rates (Binance USD-M)
# ---------------------------------------------------------------------------

FUNDING_COLUMNS = ["time", "symbol", "funding_rate", "mark_price"]


def _to_mexc_symbol(binance_symbol: str) -> str:
    """Translate a Binance-style symbol (BTCUSDT) to MEXC's underscore form
    (BTC_USDT). Already-underscored input passes through."""
    if "_" in binance_symbol:
        return binance_symbol
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if binance_symbol.endswith(quote):
            return f"{binance_symbol[:-len(quote)]}_{quote}"
    return binance_symbol


def fetch_funding_rate_history_mexc(
    symbol: str,
    start: datetime,
    end: datetime,
    request_delay_s: float = 0.3,
) -> tuple[list[dict], FetchProvenance]:
    """Fetch perpetual funding rate history from MEXC contract API.

    Geo-block-resistant fallback for environments where fapi.binance.com
    returns 451 (e.g. some EU VPS providers). MEXC contract API accepts the
    underscored symbol form (``BTC_USDT``); we translate the Binance form
    automatically. Cadence is 8h, same as Binance USD-M. History depth is
    ~1.5 years (vs Binance's ~5+); for our 12-month default this is fine.

    Pagination: MEXC returns descending order with page_size up to 100; we
    walk pages forward until we exit the requested window. Output rows are
    sorted ascending to match the Binance shape.

    The output ``symbol`` field preserves the *input* symbol form so consumers
    don't have to handle two conventions.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes")
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)

    mexc_symbol = _to_mexc_symbol(symbol)
    rows: list[dict] = []
    page_num = 1
    while True:
        params = {"symbol": mexc_symbol, "page_size": 100, "page_num": page_num}
        resp = _http_get_json(MEXC_FUNDING_HISTORY, params=params)
        if not isinstance(resp, dict) or not resp.get("success"):
            break
        data = resp.get("data") or {}
        result_list = data.get("resultList") or []
        if not result_list:
            break
        # MEXC returns desc by settleTime. Walk this page; stop paging once
        # this page's oldest settleTime is past start_ms (we've covered the
        # window) OR we've hit totalPage.
        page_oldest_ms = None
        for r in result_list:
            settle_ms = int(r["settleTime"])
            page_oldest_ms = settle_ms if page_oldest_ms is None else min(page_oldest_ms, settle_ms)
            if start_ms <= settle_ms <= end_ms:
                rows.append({
                    "time": _ms_to_iso_utc(settle_ms),
                    "symbol": symbol,
                    "funding_rate": str(r["fundingRate"]),
                    "mark_price": None,
                })
        total_page = int(data.get("totalPage", page_num))
        if page_oldest_ms is not None and page_oldest_ms < start_ms:
            break
        if page_num >= total_page:
            break
        page_num += 1
        time.sleep(request_delay_s)

    rows.sort(key=lambda r: r["time"])

    prov = FetchProvenance(
        source_url=MEXC_FUNDING_HISTORY,
        fetched_at=datetime.now(timezone.utc),
        row_count=len(rows),
    )
    return rows, prov


def fetch_funding_rate_history(
    symbol: str,
    start: datetime,
    end: datetime,
    request_delay_s: float = 0.3,
    fallback_to_mexc: bool = True,
) -> tuple[list[dict], FetchProvenance]:
    """Fetch perpetual funding rate history for a symbol (e.g. ``BTCUSDT``).

    Tries Binance USD-M (``fapi.binance.com``) first. On HTTP 451
    (geo-block — common from EU VPS hosts like Contabo Germany), falls back
    to MEXC contract API via ``fetch_funding_rate_history_mexc``. Set
    ``fallback_to_mexc=False`` to suppress the fallback (Binance-only).

    Funding payments on Binance USD-M happen every 8h. Historical data
    available back to ~Sep 2019 for BTCUSDT. MEXC has ~1.5y of history
    (still adequate for the meta-agent's 12-month default backfill).
    Returns rows + provenance.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes")
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
    rows: list[dict] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            page = _http_get_json(BINANCE_FUTURES_FUNDING, params=params)
            if not isinstance(page, list) or not page:
                break
            for r in page:
                rows.append({
                    "time": _ms_to_iso_utc(r["fundingTime"]),
                    "symbol": r["symbol"],
                    "funding_rate": r["fundingRate"],
                    "mark_price": r.get("markPrice"),
                })
            cursor = page[-1]["fundingTime"] + 1
            if len(page) < 1000:
                break
            time.sleep(request_delay_s)
    except RuntimeError as e:
        if fallback_to_mexc and "451" in str(e):
            # Re-fetch the full window from MEXC. We discard any partial Binance
            # rows because the two sources may have slightly different funding
            # values per-event (different settle times, different mark prices),
            # and mixing would create discontinuities.
            return fetch_funding_rate_history_mexc(
                symbol=symbol, start=start, end=end, request_delay_s=request_delay_s,
            )
        raise

    prov = FetchProvenance(
        source_url=BINANCE_FUTURES_FUNDING,
        fetched_at=datetime.now(timezone.utc),
        row_count=len(rows),
    )
    return rows, prov


def load_funding_rates(path: str | Path) -> Optional[pd.DataFrame]:
    """Load a funding-rate CSV written by ``fetch_funding_rate_history``.

    Returns a DataFrame indexed by UTC timestamp with ``symbol, funding_rate,
    mark_price`` (mark_price may be absent for older files). Returns ``None``
    if file missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.set_index("time").sort_index()
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    if "mark_price" in df.columns:
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    return df.dropna(subset=["funding_rate"])


def persist_rows_csv(
    rows: list[dict],
    out_path: str | Path,
    columns: list[str],
    append: bool = False,
) -> FetchProvenance:
    """Write rows to CSV and return provenance including SHA-256."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out.exists() else "w"
    with open(out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    return FetchProvenance(
        source_url="file://" + str(out),
        fetched_at=datetime.now(timezone.utc),
        row_count=len(rows),
        file_path=str(out),
        file_sha256=_sha256_of_file(out),
    )


# ---------------------------------------------------------------------------
# BTC dominance (BTC.D)
# ---------------------------------------------------------------------------


def fetch_btc_dominance_snapshot(source: str = "coingecko") -> dict:
    """Fetch current BTC dominance from a free public endpoint.

    Returns a dict with keys: ``time`` (ISO UTC), ``source``,
    ``btc_dominance_pct`` (0-100), ``total_market_cap_usd``.

    Sources:
        ``coingecko``   — ``/api/v3/global`` (free, no key). Rate-limited; back
                          off on 429. No historical backfill on free tier.
        ``coinpaprika`` — ``/v1/global`` (free, no key). Independent source,
                          useful as cross-check or fallback.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")
    if source == "coingecko":
        payload = _http_get_json(COINGECKO_GLOBAL)
        d = payload.get("data", {}) if isinstance(payload, dict) else {}
        return {
            "time": now_iso,
            "source": "coingecko",
            "btc_dominance_pct": float(d.get("market_cap_percentage", {}).get("btc", float("nan"))),
            "total_market_cap_usd": float(d.get("total_market_cap", {}).get("usd", float("nan"))),
        }
    if source == "coinpaprika":
        payload = _http_get_json(COINPAPRIKA_GLOBAL)
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected coinpaprika response shape")
        return {
            "time": now_iso,
            "source": "coinpaprika",
            "btc_dominance_pct": float(payload.get("bitcoin_dominance_percentage", float("nan"))),
            "total_market_cap_usd": float(payload.get("market_cap_usd", float("nan"))),
        }
    raise ValueError(f"unknown source: {source!r}")


def append_btc_dominance_sample(
    out_path: str | Path,
    source: str = "coingecko",
) -> FetchProvenance:
    """Sample current BTC.D and append a row to the target CSV.

    Creates the file with a header if missing. Designed to be called on a
    daily cron / systemd timer — BTC.D moves on a weekly timescale per
    ``meta_agent_plan.md`` §4.8, so daily resolution is sufficient.
    """
    snap = fetch_btc_dominance_snapshot(source=source)
    columns = ["time", "source", "btc_dominance_pct", "total_market_cap_usd"]
    out = Path(out_path)
    is_new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            w.writeheader()
        w.writerow(snap)
    return FetchProvenance(
        source_url=COINGECKO_GLOBAL if source == "coingecko" else COINPAPRIKA_GLOBAL,
        fetched_at=datetime.now(timezone.utc),
        row_count=1,
        file_path=str(out),
        file_sha256=_sha256_of_file(out),
    )


def load_btc_dominance_series(path: str | Path) -> Optional[pd.DataFrame]:
    """Load the hand-accumulated BTC.D CSV.

    Expected schema: ``time, source, btc_dominance_pct, total_market_cap_usd``.
    Returns DataFrame indexed by UTC timestamp; ``None`` if missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.set_index("time").sort_index()
    df["btc_dominance_pct"] = pd.to_numeric(df["btc_dominance_pct"], errors="coerce")
    if "total_market_cap_usd" in df.columns:
        df["total_market_cap_usd"] = pd.to_numeric(df["total_market_cap_usd"], errors="coerce")
    return df.dropna(subset=["btc_dominance_pct"])


# ---------------------------------------------------------------------------
# Convenience: trailing-window funding average
# ---------------------------------------------------------------------------


def average_funding_rate(
    df: pd.DataFrame,
    as_of: datetime,
    window_hours: int = 24,
) -> float:
    """Average funding rate over a trailing window.

    Args:
        df: DataFrame from ``load_funding_rates`` (UTC index).
        as_of: endpoint of the window (UTC).
        window_hours: window size.

    Returns:
        Mean funding rate (fraction, not pct) in the window. NaN if empty.
    """
    if df is None or df.empty:
        return float("nan")
    end = as_of.astimezone(timezone.utc)
    start = end - timedelta(hours=window_hours)
    sub = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if sub.empty:
        return float("nan")
    return float(sub["funding_rate"].mean())


__all__ = [
    "FetchProvenance",
    "fetch_btc_usd_klines",
    "load_btc_usd_hourly",
    "fetch_funding_rate_history",
    "fetch_funding_rate_history_mexc",
    "load_funding_rates",
    "persist_rows_csv",
    "fetch_btc_dominance_snapshot",
    "append_btc_dominance_sample",
    "load_btc_dominance_series",
    "average_funding_rate",
    "KLINE_COLUMNS",
    "FUNDING_COLUMNS",
    "BINANCE_US_SPOT_KLINES",
    "BINANCE_FUTURES_FUNDING",
    "COINGECKO_GLOBAL",
    "COINPAPRIKA_GLOBAL",
]
