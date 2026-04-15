"""
Unified Data Loader — single source of truth for all taostats data.

Loads ALL columns from taostats data sources, handles regime
boundaries, lifecycle masking, and supports both hourly and 5-min resolution.

Usage:
    cfg = LoaderConfig(data_dir="/path/to/data/taostats")
    loader = UnifiedDataLoader(cfg)
    data = loader.load()
    # data.prices: (n_times, n_subnets)
    # data.pool_state: dict of (n_times, n_subnets) arrays
    # ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils.lifecycle import detect_lifecycle_boundaries


# TaoFlow activation date — different market regime before this
TAOFLOW_DATE = pd.Timestamp("2025-11-04", tz="UTC")


@dataclass
class LoaderConfig:
    """Configuration for data loading."""
    data_dir: str                          # required — no default
    freq: str = "5min"                     # "5min" (primary) or "1h" (fallback)
    subnets: Optional[list[int]] = None
    min_active_bars: int = 100
    post_taoflow_only: bool = True         # Filter to Nov 4 2025+ by default
    lifecycle_margin_hours: int = 168      # 7-day buffer after rebirth

    # Extra feature CSVs — list of paths to pre-built (time, netuid, ...) CSVs
    # Loaded generically into data.extra dict
    extra_feature_paths: list[str] = field(default_factory=list)

    # Validator history
    load_validators: bool = True

    # TAO/USD price
    load_tao_usd: bool = True

    # External market data (BTC, altcoins, Fear & Greed)
    binance_dir: Optional[str] = None      # directory with btcusdt_5m.csv etc.
    fear_greed_path: Optional[str] = None  # path to fear_greed.csv


@dataclass
class DataArrays:
    """All loaded data arrays, aligned on (n_times, n_subnets) grid."""
    # Core identifiers
    timestamps: np.ndarray              # (n_times,) datetime64
    subnet_ids: list[int]               # subnet netuid list
    n_times: int = 0
    n_subnets: int = 0

    # OHLCV — (n_times, n_subnets)
    prices: np.ndarray = None           # close price (alpha/tao)
    opens: np.ndarray = None            # open price
    highs: np.ndarray = None            # high price
    lows: np.ndarray = None             # low price
    volumes: np.ndarray = None          # TAO volume per bar
    n_trades: np.ndarray = None         # trade count per bar
    net_flows: np.ndarray = None        # signed TAO flow per bar

    # Derived pool state — (n_times, n_subnets)
    k_values: np.ndarray = None         # AMM invariant k = tao_in * alpha_in
    tao_pools: np.ndarray = None        # tao_in (derived from k and price)
    alpha_pools: np.ndarray = None      # alpha_in (derived from k and price)

    # Pool history columns — (n_times, n_subnets) daily, forward-filled
    alpha_staked: np.ndarray = None     # staked alpha tokens (÷1e9 from rao)
    root_prop: np.ndarray = None        # root proportion
    fee_global_tao: np.ndarray = None   # global tao fee
    protocol_provided_tao: np.ndarray = None
    protocol_provided_alpha: np.ndarray = None
    market_cap: np.ndarray = None
    pool_liquidity: np.ndarray = None   # liquidity field from pool_history

    # Subnet history columns — (n_times, n_subnets) daily, forward-filled
    emissions: np.ndarray = None        # daily emission TAO
    emission_raw: np.ndarray = None     # per-tempo emission in rao (same units as ema_tao_flow)
    projected_emission: np.ndarray = None
    ema_tao_flow: np.ndarray = None
    excess_tao: np.ndarray = None
    recycled_24_hours: np.ndarray = None
    active_miners: np.ndarray = None
    active_validators: np.ndarray = None
    registration_cost: np.ndarray = None

    # TAO/USD — (n_times,)
    tao_usd: Optional[np.ndarray] = None
    tao_usd_volume: Optional[np.ndarray] = None
    tao_usd_market_cap: Optional[np.ndarray] = None

    # Validator features — (n_times, n_subnets) daily, forward-filled
    top_validator_dominance: Optional[np.ndarray] = None
    nominator_return_per_day: Optional[np.ndarray] = None
    validator_count_change: Optional[np.ndarray] = None

    # External market data — (n_times,) arrays
    btc_usd: Optional[np.ndarray] = None    # BTC close price
    eth_usd: Optional[np.ndarray] = None    # ETH close price
    ai_basket: Optional[np.ndarray] = None  # equal-weight FET+NEAR+RENDER return
    fear_greed: Optional[np.ndarray] = None # Fear & Greed Index (0-100)

    # Extra features — loaded generically from CSVs via load_extra_features()
    # Keys are column names, values are (n_times, n_subnets) arrays
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    # Regime markers — (n_times,) bool
    is_post_taoflow: Optional[np.ndarray] = None

    # Lifecycle mask — (n_times, n_subnets) bool
    lifecycle_mask: np.ndarray = None


class UnifiedDataLoader:
    """
    Loads and aligns all data sources into a single DataArrays object.

    Replaces per-environment _load_data() methods. Call load() once,
    pass the result to environments and feature engines.
    """

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg

    def load(self) -> DataArrays:
        path = Path(self.cfg.data_dir)
        data = DataArrays(timestamps=np.array([]), subnet_ids=[])

        # 1. Load OHLCV
        ohlcv, pool_df, sh_df = self._load_raw_dataframes(path)

        # 2. Select subnets
        ohlcv = self._select_subnets(ohlcv)
        data.subnet_ids = sorted(ohlcv["netuid"].unique().tolist())
        if not data.subnet_ids:
            raise ValueError("No valid subnets after filtering")

        # 3. Build time index and pivot to arrays
        data.timestamps = np.sort(ohlcv["time"].unique())
        data.n_times = len(data.timestamps)
        data.n_subnets = len(data.subnet_ids)

        self._pivot_ohlcv(ohlcv, data)

        # 4. Pool history → k values + extra columns
        self._merge_pool_history(pool_df, data)

        # 5. Derive pool state from k and price
        self._derive_pool_state(data)

        # 6. Subnet history → emissions + extra columns
        self._merge_subnet_history(sh_df, data)

        # 7. Lifecycle masking
        self._apply_lifecycle_mask(pool_df, data)

        # 8. TAO/USD
        if self.cfg.load_tao_usd:
            self._load_tao_usd(path, data)

        # 9. Validator history
        if self.cfg.load_validators:
            self._load_validator_history(path, data)

        # 10. Extra feature CSVs (generic loader)
        for feat_path in self.cfg.extra_feature_paths:
            self._load_extra_features(feat_path, data)

        # 10b. External market data (BTC, altcoins, F&G)
        if self.cfg.binance_dir:
            self._load_external_market(data)

        # 11. Regime markers
        self._compute_regime_markers(data)

        # 12. Post-TaoFlow filter
        if self.cfg.post_taoflow_only:
            self._filter_post_taoflow(data)

        return data

    # ----------------------------------------------------------
    # Raw dataframe loading
    # ----------------------------------------------------------

    def _load_raw_dataframes(self, path: Path):
        """Load CSV files into dataframes."""
        # OHLCV
        freq = self.cfg.freq
        if freq == "5min":
            ohlcv_file = path / "delegation_ohlcv_5m.csv"
            if not ohlcv_file.exists():
                raise FileNotFoundError(
                    f"5-min OHLCV not found at {ohlcv_file}. "
                    "Run scripts/build_5min_ohlcv.py first."
                )
        else:
            ohlcv_file = path / "delegation_ohlcv_hourly.csv"

        ohlcv = pd.read_csv(ohlcv_file)
        ohlcv["time"] = pd.to_datetime(
            ohlcv["time"], format="ISO8601", utc=True
        ).dt.tz_localize(None)
        ohlcv["netuid"] = ohlcv["netuid"].astype(int)

        # Pool history
        pool = pd.read_csv(path / "pool_history.csv", low_memory=False)
        pool["timestamp"] = pd.to_datetime(
            pool["timestamp"], format="ISO8601", utc=True
        ).dt.tz_localize(None)
        pool["netuid"] = pool["netuid"].astype(int)
        # Derived columns (rao → TAO/tokens)
        pool["tao_in"] = pool["total_tao"].astype(float) / 1e9
        pool["alpha_in"] = pool["alpha_in_pool"].astype(float) / 1e9
        pool["k"] = pool["tao_in"] * pool["alpha_in"]
        pool["alpha_staked_tokens"] = pool["alpha_staked"].astype(float) / 1e9
        for col in ["protocol_provided_tao", "protocol_provided_alpha"]:
            if col in pool.columns:
                pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0) / 1e9

        # Filter startup-mode subnets from OHLCV
        if "startup_mode" in pool.columns:
            sm = pool["startup_mode"].fillna(False).astype(bool)
            startup_end = {}
            for nid in pool.loc[sm, "netuid"].unique():
                ok = pool[(pool["netuid"] == nid) & ~sm]
                if len(ok):
                    startup_end[nid] = ok["timestamp"].min()
            for nid, ts in startup_end.items():
                ohlcv = ohlcv[~((ohlcv["netuid"] == nid) & (ohlcv["time"] < ts))]
            always_startup = set(pool.loc[sm, "netuid"]) - set(startup_end)
            ohlcv = ohlcv[~ohlcv["netuid"].isin(always_startup)]

        # Subnet history
        sh = pd.read_csv(path / "subnet_history.csv")
        sh["timestamp"] = pd.to_datetime(
            sh["timestamp"], format="ISO8601", utc=True
        ).dt.tz_localize(None)
        sh["netuid"] = sh["netuid"].astype(int)
        sh["emission"] = sh["emission"].fillna(0).astype(float)
        sh["tempo"] = sh["tempo"].fillna(360).astype(float).clip(lower=1)
        sh["daily_emission_tao"] = sh["emission"] * (7200.0 / sh["tempo"]) / 1e9

        return ohlcv, pool, sh

    # ----------------------------------------------------------
    # Subnet selection
    # ----------------------------------------------------------

    def _select_subnets(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.subnets is not None:
            return ohlcv[ohlcv["netuid"].isin(self.cfg.subnets)].copy()

        counts = ohlcv.loc[ohlcv["n_trades"] > 0].groupby("netuid").size()
        valid = counts[counts >= self.cfg.min_active_bars].index
        return ohlcv[ohlcv["netuid"].isin(valid)].copy()

    # ----------------------------------------------------------
    # OHLCV pivoting
    # ----------------------------------------------------------

    def _pivot_ohlcv(self, ohlcv: pd.DataFrame, data: DataArrays):
        """Pivot long-format OHLCV into (n_times, n_subnets) arrays."""
        ts = data.timestamps
        sids = data.subnet_ids

        def _pivot(col, fill=0.0):
            pv = ohlcv.pivot_table(
                index="time", columns="netuid", values=col, aggfunc="last"
            )
            pv = pv.reindex(index=ts, columns=sids)
            return pv.fillna(fill) if fill is not None else pv

        prices_df = _pivot("close", fill=None).ffill().bfill()
        data.prices = prices_df.values.astype(np.float64)
        data.volumes = _pivot("volume").values.astype(np.float64)
        data.n_trades = _pivot("n_trades").values.astype(np.float64)
        data.net_flows = _pivot("net_flow_tao").values.astype(np.float64)

        # OHLC — new: load high/low/open for expanded price features
        if "open" in ohlcv.columns:
            data.opens = _pivot("open", fill=None).ffill().bfill().values.astype(np.float64)
        if "high" in ohlcv.columns:
            data.highs = _pivot("high", fill=None).ffill().bfill().values.astype(np.float64)
        if "low" in ohlcv.columns:
            data.lows = _pivot("low", fill=None).ffill().bfill().values.astype(np.float64)

    # ----------------------------------------------------------
    # Pool history merging
    # ----------------------------------------------------------

    def _merge_pool_history(self, pool: pd.DataFrame, data: DataArrays):
        """Merge daily pool history into aligned arrays via merge_asof."""
        ts_df = pd.DataFrame({"time": data.timestamps})

        # Columns to extract from pool_history (besides k)
        pool_cols = {
            "k": ("k", 0.0),
            "alpha_staked_tokens": ("alpha_staked", 0.0),
            "root_prop": ("root_prop", 0.0),
            "fee_global_tao": ("fee_global_tao", 0.0),
            "protocol_provided_tao": ("protocol_provided_tao", 0.0),
            "protocol_provided_alpha": ("protocol_provided_alpha", 0.0),
            "market_cap": ("market_cap", 0.0),
            "liquidity": ("pool_liquidity", 0.0),
        }

        # Initialize arrays
        result = {attr: np.zeros((data.n_times, data.n_subnets), dtype=np.float64)
                  for _, (attr, _) in pool_cols.items()}

        for si, nid in enumerate(data.subnet_ids):
            sub = pool[pool["netuid"] == nid].sort_values("timestamp")
            if len(sub) == 0:
                continue

            for src_col, (attr, fill) in pool_cols.items():
                if src_col not in sub.columns:
                    continue
                sub_col = sub[["timestamp", src_col]].dropna(subset=[src_col])
                if len(sub_col) == 0:
                    continue
                sub_col[src_col] = pd.to_numeric(sub_col[src_col], errors="coerce").fillna(fill)
                m = pd.merge_asof(
                    ts_df, sub_col,
                    left_on="time", right_on="timestamp",
                    direction="backward"
                )
                result[attr][:, si] = m[src_col].fillna(fill).values

        data.k_values = result["k"]
        data.alpha_staked = result["alpha_staked"]
        data.root_prop = result["root_prop"]
        data.fee_global_tao = result["fee_global_tao"]
        data.protocol_provided_tao = result["protocol_provided_tao"]
        data.protocol_provided_alpha = result["protocol_provided_alpha"]
        data.market_cap = result["market_cap"]
        data.pool_liquidity = result["pool_liquidity"]

    # ----------------------------------------------------------
    # Derived pool state
    # ----------------------------------------------------------

    def _derive_pool_state(self, data: DataArrays):
        """Compute tao_pools and alpha_pools from k and price."""
        ok = (data.prices > 0) & (data.k_values > 0)
        data.tao_pools = np.where(ok, np.sqrt(data.k_values * data.prices), 0.0)
        data.alpha_pools = np.where(ok, np.sqrt(data.k_values / np.maximum(data.prices, 1e-20)), 0.0)

    # ----------------------------------------------------------
    # Subnet history merging
    # ----------------------------------------------------------

    def _merge_subnet_history(self, sh: pd.DataFrame, data: DataArrays):
        """Merge daily subnet history into aligned arrays."""
        ts_df = pd.DataFrame({"time": data.timestamps})

        sh_cols = {
            "daily_emission_tao": ("emissions", 0.0),
            "emission": ("emission_raw", 0.0),
            "projected_emission": ("projected_emission", 0.0),
            "ema_tao_flow": ("ema_tao_flow", 0.0),
            "excess_tao": ("excess_tao", 0.0),
            "recycled_24_hours": ("recycled_24_hours", 0.0),
            "active_miners": ("active_miners", 0.0),
            "active_validators": ("active_validators", 0.0),
            "registration_cost": ("registration_cost", 0.0),
        }

        result = {attr: np.zeros((data.n_times, data.n_subnets), dtype=np.float64)
                  for _, (attr, _) in sh_cols.items()}

        for si, nid in enumerate(data.subnet_ids):
            sub = sh[sh["netuid"] == nid].sort_values("timestamp")
            if len(sub) == 0:
                continue

            for src_col, (attr, fill) in sh_cols.items():
                if src_col not in sub.columns:
                    continue
                sub_col = sub[["timestamp", src_col]].dropna(subset=[src_col])
                if len(sub_col) == 0:
                    continue
                sub_col[src_col] = pd.to_numeric(sub_col[src_col], errors="coerce").fillna(fill)
                m = pd.merge_asof(
                    ts_df, sub_col,
                    left_on="time", right_on="timestamp",
                    direction="backward"
                )
                result[attr][:, si] = m[src_col].fillna(fill).values

        data.emissions = result["emissions"]
        data.emission_raw = result["emission_raw"]
        data.projected_emission = result["projected_emission"]
        data.ema_tao_flow = result["ema_tao_flow"]
        data.excess_tao = result["excess_tao"]
        data.recycled_24_hours = result["recycled_24_hours"]
        data.active_miners = result["active_miners"]
        data.active_validators = result["active_validators"]
        data.registration_cost = result["registration_cost"]

    # ----------------------------------------------------------
    # Lifecycle masking
    # ----------------------------------------------------------

    def _apply_lifecycle_mask(self, pool_df: pd.DataFrame, data: DataArrays):
        """Detect lifecycle boundaries and mask invalid data."""
        data.lifecycle_mask = detect_lifecycle_boundaries(
            pool_df, data.subnet_ids, data.timestamps,
            margin_hours=self.cfg.lifecycle_margin_hours,
        )

        # Zero out invalid data
        mask = ~data.lifecycle_mask
        for arr in [data.prices, data.volumes, data.net_flows, data.n_trades,
                    data.tao_pools, data.alpha_pools, data.emissions, data.emission_raw]:
            if arr is not None:
                arr[mask] = 0.0

        # Forward-fill prices within valid segments
        for si in range(data.n_subnets):
            col = data.prices[:, si]
            vm = data.lifecycle_mask[:, si]
            last = 0.0
            for t in range(len(col)):
                if vm[t] and col[t] > 0:
                    last = col[t]
                elif vm[t] and col[t] == 0 and last > 0:
                    col[t] = last

    # ----------------------------------------------------------
    # TAO/USD price
    # ----------------------------------------------------------

    def _load_tao_usd(self, path: Path, data: DataArrays):
        """Load TAO/USD price, trying 15-min tao_price.csv first, hourly fallback."""
        ts_df = pd.DataFrame({"time": data.timestamps})

        # Primary: tao_price.csv (15-min resolution)
        tao_path = path / "tao_price.csv"
        if tao_path.exists():
            tao = pd.read_csv(tao_path)
            tao["timestamp"] = pd.to_datetime(
                tao["timestamp"], format="ISO8601", utc=True
            ).dt.tz_localize(None)
            tao = tao.sort_values("timestamp")

            m = pd.merge_asof(
                ts_df, tao[["timestamp", "price", "volume_24h", "market_cap"]],
                left_on="time", right_on="timestamp",
                direction="backward"
            )
            data.tao_usd = m["price"].ffill().fillna(0).values.astype(np.float64)
            data.tao_usd_volume = m["volume_24h"].ffill().fillna(0).values.astype(np.float64)
            data.tao_usd_market_cap = m["market_cap"].ffill().fillna(0).values.astype(np.float64)
            return

        # Fallback: tao_ohlc_hourly.csv
        hourly_path = path / "tao_ohlc_hourly.csv"
        if hourly_path.exists():
            tao = pd.read_csv(hourly_path)
            tao["timestamp"] = pd.to_datetime(
                tao["timestamp"], format="ISO8601", utc=True
            ).dt.tz_localize(None)
            tao = tao.sort_values("timestamp")

            m = pd.merge_asof(
                ts_df,
                tao[["timestamp", "close", "volume_24h"]].rename(
                    columns={"close": "price"}
                ),
                left_on="time", right_on="timestamp",
                direction="backward"
            )
            data.tao_usd = m["price"].ffill().fillna(0).values.astype(np.float64)
            data.tao_usd_volume = m["volume_24h"].ffill().fillna(0).values.astype(np.float64)

    # ----------------------------------------------------------
    # Validator history
    # ----------------------------------------------------------

    def _load_validator_history(self, path: Path, data: DataArrays):
        """Load validator history and compute per-subnet summary features."""
        val_path = path / "dtao_validator_history.csv"
        if not val_path.exists():
            return

        val = pd.read_csv(val_path, low_memory=False)
        val["timestamp"] = pd.to_datetime(
            val["timestamp"], format="ISO8601", utc=True
        ).dt.tz_localize(None)

        # Compute daily network-level validator stats
        daily_stats = val.groupby("timestamp").agg(
            top_dominance=("dominance", "max"),
            mean_nom_return=("nominator_return_per_day", "mean"),
            total_validators=("hotkey", "count"),
        ).reset_index()
        daily_stats = daily_stats.sort_values("timestamp")

        ts_df = pd.DataFrame({"time": data.timestamps})

        # Broadcast network-level stats across all subnets
        m = pd.merge_asof(
            ts_df, daily_stats,
            left_on="time", right_on="timestamp",
            direction="backward"
        )

        top_dom = m["top_dominance"].ffill().fillna(0).values.astype(np.float64)
        nom_ret = m["mean_nom_return"].ffill().fillna(0).values.astype(np.float64)
        val_count = m["total_validators"].ffill().fillna(0).values.astype(np.float64)

        # Broadcast to (n_times, n_subnets)
        data.top_validator_dominance = np.broadcast_to(
            top_dom[:, np.newaxis], (data.n_times, data.n_subnets)
        ).copy()
        data.nominator_return_per_day = np.broadcast_to(
            nom_ret[:, np.newaxis], (data.n_times, data.n_subnets)
        ).copy()

        # Validator count change (24h diff)
        val_diff = np.zeros_like(val_count)
        lag = 24 if self.cfg.freq == "1h" else 288  # 24h worth of bars
        val_diff[lag:] = val_count[lag:] - val_count[:-lag]
        data.validator_count_change = np.broadcast_to(
            val_diff[:, np.newaxis], (data.n_times, data.n_subnets)
        ).copy()

    # ----------------------------------------------------------
    # Extra features (generic CSV loader)
    # ----------------------------------------------------------

    def _load_extra_features(self, csv_path: str, data: DataArrays):
        """Load a pre-built feature CSV with (time, netuid, ...) columns.

        All non-time/netuid columns are pivoted to (n_times, n_subnets)
        arrays and stored in data.extra[column_name].
        """
        path = Path(csv_path)
        if not path.exists():
            return

        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True).dt.tz_localize(None)
        df["netuid"] = df["netuid"].astype(int)

        ts = data.timestamps
        sids = data.subnet_ids
        feature_cols = [c for c in df.columns if c not in ("time", "netuid")]

        for col in feature_cols:
            pv = df.pivot_table(index="time", columns="netuid", values=col, aggfunc="last")
            pv = pv.reindex(index=ts, columns=sids).fillna(0)
            data.extra[col] = pv.values.astype(np.float64)

    # ----------------------------------------------------------
    # External market data
    # ----------------------------------------------------------

    def _load_external_market(self, data: DataArrays):
        """Load BTC, ETH, AI altcoin, and Fear & Greed data."""
        binance_dir = Path(self.cfg.binance_dir)
        ts = pd.DatetimeIndex(data.timestamps)

        def _load_5min_close(filename):
            """Load a 5-min Binance/MEXC CSV and align to our time index."""
            path = binance_dir / filename
            if not path.exists():
                return None
            df = pd.read_csv(path, usecols=["time", "close"])
            df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True).dt.tz_localize(None)
            df = df.set_index("time").sort_index()
            # Reindex to our timestamps, forward-fill
            aligned = df.reindex(ts, method="ffill")
            return aligned["close"].values.astype(np.float64)

        # BTC and ETH (5-min)
        btc = _load_5min_close("btcusdt_5m.csv")
        if btc is not None:
            data.btc_usd = btc
            print(f"  BTC/USD: {np.sum(~np.isnan(btc)):,} valid bars")

        eth = _load_5min_close("ethusdt_5m.csv")
        if eth is not None:
            data.eth_usd = eth

        # AI altcoin basket (hourly CSVs — load and forward-fill to 5-min)
        ai_returns = []
        for filename in ["fetusdt_1h.csv", "nearusdt_1h.csv", "renderusdt_1h.csv"]:
            path = binance_dir / filename
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["time", "close"])
            df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True).dt.tz_localize(None)
            df = df.set_index("time").sort_index()
            aligned = df.reindex(ts, method="ffill")["close"].values.astype(np.float64)
            # Compute returns
            prev = np.roll(aligned, 12)  # 1h lag at 5-min
            prev[:12] = aligned[:12]
            safe = np.where(prev > 0, prev, 1.0)
            ret = (aligned - prev) / safe
            ret[:12] = 0.0
            ai_returns.append(ret)

        if ai_returns:
            data.ai_basket = np.mean(ai_returns, axis=0)

        # Fear & Greed (daily — forward-fill to 5-min)
        fg_path = Path(self.cfg.fear_greed_path) if self.cfg.fear_greed_path else None
        if fg_path and fg_path.exists():
            fg = pd.read_csv(fg_path, usecols=["time", "value"])
            fg["time"] = pd.to_datetime(fg["time"]).dt.tz_localize(None)
            fg = fg.set_index("time").sort_index()
            aligned = fg.reindex(ts, method="ffill")
            data.fear_greed = aligned["value"].values.astype(np.float64)
            print(f"  Fear & Greed: {np.sum(~np.isnan(data.fear_greed)):,} valid bars")

    # ----------------------------------------------------------
    # Regime markers
    # ----------------------------------------------------------

    def _compute_regime_markers(self, data: DataArrays):
        """Compute regime boundary markers."""
        taoflow_ts = TAOFLOW_DATE.tz_localize(None)
        data.is_post_taoflow = (data.timestamps >= np.datetime64(taoflow_ts)).astype(np.float64)

    # ----------------------------------------------------------
    # Post-TaoFlow filtering
    # ----------------------------------------------------------

    def _filter_post_taoflow(self, data: DataArrays):
        """Trim all arrays to post-TaoFlow period only."""
        taoflow_ts = TAOFLOW_DATE.tz_localize(None)
        mask = data.timestamps >= np.datetime64(taoflow_ts)

        if mask.sum() == 0:
            raise ValueError("No data after TaoFlow date (Nov 4, 2025)")

        idx = np.where(mask)[0]
        start = idx[0]

        # Slice all time-indexed arrays
        data.timestamps = data.timestamps[start:]
        data.n_times = len(data.timestamps)

        # (n_times, n_subnets) arrays
        for attr in [
            "prices", "opens", "highs", "lows", "volumes", "n_trades", "net_flows",
            "k_values", "tao_pools", "alpha_pools",
            "alpha_staked", "root_prop", "fee_global_tao",
            "protocol_provided_tao", "protocol_provided_alpha",
            "market_cap", "pool_liquidity",
            "emissions", "emission_raw", "projected_emission", "ema_tao_flow", "excess_tao",
            "recycled_24_hours", "active_miners", "active_validators",
            "registration_cost",
            "top_validator_dominance", "nominator_return_per_day",
            "validator_count_change",
            "lifecycle_mask",
        ]:
            arr = getattr(data, attr, None)
            if arr is not None and arr.ndim == 2 and arr.shape[0] > start:
                setattr(data, attr, arr[start:])

        # Slice extra features
        for key, arr in data.extra.items():
            if arr is not None and arr.ndim == 2 and arr.shape[0] > start:
                data.extra[key] = arr[start:]

        # (n_times,) arrays
        for attr in [
            "tao_usd", "tao_usd_volume", "tao_usd_market_cap",
            "btc_usd", "eth_usd", "ai_basket", "fear_greed",
            "is_post_taoflow",
        ]:
            arr = getattr(data, attr, None)
            if arr is not None and arr.ndim == 1 and arr.shape[0] > start:
                setattr(data, attr, arr[start:])
