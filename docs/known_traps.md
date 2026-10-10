# Known traps in backtests built on this package

Silent problems, each measured on real data in October 2026. Sections 1 to 4 were measured against a reference (many seeds, or contemporaneous SDK ticks), and sections 1, 2 and 4 each produced a plausible-looking but wrong return in at least one study. Sections 5 and 6 (added 2026-10-08) are exposures found in an audit, whose effect on published results has not been measured. Section 7 (added 2026-10-09, revised 2026-10-10) is a failure of the opt-in `reserves="rescale"` path that produced an absurd return in one rerun. Each is guarded, made visible, or documented in code; this page says what they are, how large they were, how to check for them, and what changed.

Terms: a **tick** is one timestamped snapshot of every subnet's price and pool reserves (`TickData` holding `SubnetTick`s). A **reserve** is the pool's TAO (`tao_pool`) or alpha (`alpha_pool`) balance; the pool-implied price is `tao_pool / alpha_pool`. **Realism noise** is the random part of the engine's execution model (random rejections, slippage noise, partial fills) drawn from a random number generator (RNG) that starts from `realism_rng_seed`.

## 1. A new engine per trade replays the same realism noise

**What happens.** `BacktestEngine` defaults to `realism_rng_seed=0` so reruns reproduce. The seed belongs to the engine instance, so every new engine built with seed 0 draws the same random numbers in the same order. A study that runs N independent one-trade windows (one engine each, as `panel_forward_returns` and `run_basket_window` do) and averages them is therefore averaging one draw, counted N times. If that draw is favorable, every window is favorable.

**Measured (2026-10-02).** One 1 TAO round trip on a deep pool (80,519 TAO) at unchanged price, 200 different seeds: mean -0.45% on cost, standard deviation 0.28%. Seed 0 gave -0.01%, about 0.44 percentage points better than the mean. The same gap appeared at 0.25 TAO (seed 0: -0.83%; 200-seed mean: -1.27%) and at 3 TAO (seed 0: +0.17%; 200-seed mean: -0.27%).
A synthetic check on this branch: 20 identical one-trade windows on one flat pool returned 20 identical net returns under the old code (mean -0.045%, one distinct value) and 20 distinct values under the new code (mean -0.555%, standard deviation 0.23%).

**Who is affected.** Any code that builds many engines, each executing few trades, under one seed. A single long run with many trades is not affected, because the draws advance from trade to trade. Paper bots default to `realism_rng_seed=None` (nondeterministic) and are not affected. At the time of writing the panel API had no callers in `bt-trading-tools`, `bt-strategy` or the `alpha-trading` bots and scripts other than the October 2026 emission-event and five-pillars studies; their affected tables are listed in those studies' results files.

**What changed.**
- `panel_forward_returns` and `run_basket_window` now give every window its own deterministic seed, `derive_window_seed(base_seed, entry_ts, netuids)`. `realism_rng_seed=<int>` sets the base seed; `None` is nondeterministic. If you pass your own `engine_factory` you own the seeding.
- `seeded_engine_factory(base_seed, **engine_kwargs)` builds engines with seeds `base_seed + 1, + 2, ...` for any code that makes one engine per trade or window.
- `BacktestEngine` warns (once per seed) after 25 runs with at most 4 fills each under one seed in a process. This is a heuristic and can miss cases.
- The default seed of a single engine is unchanged (0).

**What to do.** When you run many independent windows or trades, give each its own seed. If you must compare results from different code paths, compare them under the same seeding.

## 2. The engine fills from pool reserves, not from the tick price

**What happens.** `BacktestEngine` executes buys and sells against `SubnetTick.tao_pool` and `SubnetTick.alpha_pool` (constant-product math). `SubnetTick.price` is used for limit checks and bookkeeping but does not set the fill. If a tick pairs a fresh price with old reserves, fills are priced off the old reserves. `load_parquet_ticks` paired an hourly close with the latest DAILY pool snapshot, so the reserves lagged the price by up to a day.

**Measured (2026-10-02 to 2026-10-03).** On 160 volume-surge episodes between 2025-11-24 and 2026-02-10, at the entry tick the pool-implied price averaged 0.921 times the tick price (median 0.916; 10th percentile 0.803), and at the exit tick 24 hours later 1.007. Entries were priced about 8% below the tick price. The mean 24-hour net return on those same episodes was +11.2% with daily reserves and +7.8% with contemporaneous 15-minute SDK reserves (the SDK result is the valid one). A pre-written reading rule would have mechanically called the first number a confirmation.
On the SDK feed itself the pool-implied price is close to the tick price, but how close depends on the month (re-measured 2026-10-08 on the local `sdk_pool_state.csv`, all subnets, per calendar month): 2026-02 to 2026-06 median absolute gap 0.03% to 0.06%, 90th percentile about 0.6%, with rare large outliers (max 17% on 2026-04-15 to 2026-04-29, where 1.3% of subnet-ticks exceed 5%); 2026-07 onward median about 0.00001% and max under 0.0002%, i.e. the `price` column is effectively `tao_in / alpha_in`. The earlier figure on this page (median about 0.005%) did not match either period; its source is not known. Why the SDK gap changed around 2026-06/07 was not investigated.

**Who is affected.** Any strategy whose price changes faster than the reserve snapshot: hourly-signal studies, surge or breakout entries, and anything evaluated on `load_parquet_ticks`. At the time of writing that loader is used by roughly ten autobot research scripts, a bluechip-fade script, and `bt_strategy.research.baselines`. Their numbers were produced with daily reserves and have not been re-checked.

**What changed.**
- `load_parquet_ticks(..., reserves=...)` has two modes. `"raw"` (the default; the historical behavior) keeps the daily reserves. `"rescale"` keeps only the invariant `k = tao x alpha` and rescales the reserves to the tick price (`tao' = sqrt(k x price)`, `alpha' = sqrt(k / price)`). Correction (2026-10-07): an earlier version of this page and PR #1 made `"rescale"` the default; that was premature (see "Why rescale is not the default" above) and the default is back to `"raw"`.
- Each tick's `signals` carries `reserve_gap_raw` (raw implied price / close - 1) and `reserve_age_s` (age of the daily snapshot).
- `reserve_price_gap(ticks)` summarizes the mismatch for any tick source. `BacktestResults.orders_checked` and `.orders_on_inconsistent_reserves` count how many orders sat on reserves more than 5% off the tick price, and the engine warns when more than 10% of at least 10 orders do.
- `load_parquet_ticks` warns once per process that the source has known issues (this page), whatever the options.

**Measured age effect (2026-10-07, 376,113 subnet-hours with at least one trade, 2026-03-01 to 2026-07-30, against SDK pool prices).** Absolute log error of the daily-reserve implied price versus the true pool price at the tick label, by age of the daily snapshot: median 0.0% and p90 0.5% within 30 minutes of the snapshot; median 0.3% and p90 2.3% at 0.5 to 6 hours; median 0.9% and p90 4.5% at 6 to 12 hours; median 1.7% and p90 7.9% at 18 to 24 hours. Windows that begin and end at midnight sit within seconds of a snapshot, which is why buy-and-hold between midnight endpoints is barely affected.

**Why rescale is not the default.**
- It matches the price at constant k; it does not recover depth changes (a surge can double pool TAO within a day), and on active strategies it moved results by orders of magnitude (an autobot fade study: one variant +275% under raw reserves and +25,962% rescaled; a bluechip-fade variant -33% raw and +25% rescaled). Neither number is credible; the point is that rescaling is not a safe drop-in.
- Against SDK results on midnight-aligned HODL windows, with re-registered subnets removed, mean absolute error was 1.1 percentage points rescaled and 1.8 raw (11 window-rules): both close, rescale slightly closer. With re-registered subnets left in, both parquet modes were off by up to 40 points on the same universe because of problem 4.
- Where fills or entries depend on the price path, use SDK ticks.

**Custom tick builders (2026-10-08).** Bot code that builds its own ticks from an hourly close and a forward-filled daily `pool_history` snapshot has the same mismatch as `load_parquet_ticks(reserves="raw")`. `bt_trading_tools.data.rescale_tick_reserves(ticks)` applies the same opt-in transform as `reserves="rescale"` (it calls `rescale_reserves_to_price` per subnet) to any tick list, in place, records `reserve_gap_raw`, and applies the price-outlier guard of section 7 by default. Like `reserves="rescale"` it is an experiment, not a validated fix (see "Why rescale is not the default" above). `reserve_price_gap` now also reports the 90th percentile (`p90_abs_gap`). Measured on two 14-day windows (all subnet-ticks; gap = |tao_pool / alpha_pool / price - 1|; the builders were imported read-only from their repos at their current heads):

| Tick source | 2026-04-15 to 2026-04-29: median / p90 abs gap | 2026-08-01 to 2026-08-15: median / p90 abs gap |
|---|---|---|
| `load_parquet_ticks(reserves="raw")` | 0.53% / 3.8% | 1.05% / 6.7% |
| `load_parquet_ticks(reserves="rescale")` | 0 / 0 (by construction) | 0 / 0 |
| `load_sdk_ticks` (15-minute SDK) | 0.051% / 0.59% | 0.0000075% / 0.000024% |
| autobot `research/v3_backtest.py` `build_ticks`, daily pool | 0.53% / 3.8% | 1.05% / 6.7% |
| autobot `build_ticks`, hourly `pool_hour` pool | 0.18% / 2.4% | not run (`pool_hour` ends 2026-05) |
| bluechip-fade `research/bluechip_backtest.py` `build_ticks` | 0.53% / 3.8% | 1.05% / 6.7% |
| LIAI-momentum `src/data.py` `build_ticks` (April: the full function; August: its price and pool loaders with the same pairing, without the LIAI signal) | 0.53% / 3.8% | 1.03% / 6.3% |

Share of subnet-ticks with a gap above 5%: 7.3% (April) and 14.6% (August) for the raw-daily builders, 4.1% for the autobot hourly pool, 1.3% and 0% for SDK. The autobot daily and bluechip builders produce exactly the raw-loader numbers because they make the same merge. These are distributions over all subnet-ticks; the gap at the ticks where a strategy actually trades (for example surge entries, section above: 0.921 x) can be much larger, and was not measured per bot here. The whale-large-deep builder (`research/run_backtest.py` `build_ticks`) prices from a scan of the 7.3 GB `delegation_full.csv` and was not measured.

## 3. The bar label carries a one-hour lookahead (measured 2026-10-07)

The hourly OHLCV `time` is the bar START; the `close` is the pool price at the bar END. Against SDK pool prices (same 376,113 subnet-hours), the close matched the true pool price at the label with median absolute log error 0.27% (p90 2.0%) and one hour after the label with median 0.11% (p90 1.0%); in hours where the pool moved 5% or more the close matched the price one hour later with median error 0.7% and the price at the label with median 6.8%. `load_parquet_ticks` stamps ticks with the bar start, so every tick carries a price from one hour in the future. For buy-and-hold between midnight endpoints this shifts both ends together and does little. For any strategy that reads the price at the tick (hourly signals, spike fades) it is a lookahead. `load_parquet_ticks(..., bar_label="end")` stamps each tick with the bar end so the close is known at its timestamp; it is opt-in because it changes every existing result.

## 4. The parquet path misses some subnet re-registrations (measured 2026-10-07)

On the same universes and windows, the 15-minute SDK path closed positions on re-registration of netuids that the parquet path did not flag: SN82 in the 2026-04-22 to 2026-06-06 HODL window, SN69 in the cheapest-10 holdout (2026-04-16 to 2026-06-26), SN90 in the 2026-06-06 to 2026-07-21 window. A missed re-registration values the old position against the new subnet's pool and shows fabricated profit: +19.9 TAO (SN82) and +34.9 TAO (SN69) on 100 TAO of capital, against -0.1 and +0.2 TAO on SDK ticks. With `reserves="rescale"` fabricated gains also appeared where the guard did fire (SN116: +20.7 TAO rescaled vs -2.7 TAO on SDK). The 2K-5K HODL headline for the 2026-04-22 window was +14.5% (raw) and +34.3% (rescaled) against -6.0% on SDK ticks, almost entirely this effect. Parquet results over windows containing re-registrations are not reliable; use SDK ticks, or pass exact events (below).

Diagnosis (2026-10-09). The daily detector itself finds most events: against 51 events detected on SDK snapshots between 2025-09-23 and 2026-10-09, the daily `pool_history` rule found 49 (misses: SN69 on 2026-05-08, SN116 on 2026-10-04, the latter within a day of the data end) and reported none that SDK snapshots did not show. The fabricated profit comes from how the event is applied: the loader floors each event to midnight of its day, so a position opened earlier that day (for example the first tick of a window that starts on the event day, as in SN82 on 2026-04-22, true event 20:55 UTC) is stamped as the NEW subnet while sitting on the old pool, and the engine never closes it. A drop of the cumulative `recycled_since_registration` counter in `subnet_history` found 48 of the 51 and one event the SDK rule does not show; it did not beat the pool-history rule.

Fix: `load_parquet_ticks(..., reregistrations=...)` takes exact events, and `utils.lifecycle.reregistrations_from_csvs(paths)` builds them from SDK snapshot CSVs. Check on 11 strategy-window cases (HODL 2K-5K band and market in five windows, cheapest-10 holdout) against SDK ticks: mean absolute return difference 6.3 points with the daily detector, 2.5 points with SDK events (SN82 window 2K-5K: +14.5% daily detector, -5.4% SDK events, -6.0% SDK ticks). The cheapest-10 holdout went from +74.4% to +39.5% against +49.0% on SDK ticks (the remaining gap was not diagnosed). An event-day blackout (dropping the netuid's ticks on the event day) was tried and did not help (window 2026-04-22 2K-5K error 29.6 points), so it is not offered. Without SDK coverage (before 2025-09 here) the daily rule is still all there is, and its day-level dating limitation remains.

## 5. Duck-typed yield models skipped the rate plausibility rule

**What happens.** `AlphaYieldModel.rate` rejects a provider rate that is negative, non-finite, or above `MAX_PLAUSIBLE_RATE_PER_DAY` (0.02 per day, about 14 times the measured norm; added 2026-09-22 after a rebirth produced rates like 242 per day). `BacktestEngine` also accepts any object with `accrued_yield(netuid, alpha_qty, entry_time, now)` (duck typing). The two such models in this package did not apply the rule the same way: `HistoricalSubnetYieldModel` (per-subnet daily APY CSV) dropped only negative rows and credited any high row in full; `alpha_yield_history.HistoricalYieldModel` rejected high rates but silently. Both were keyed by bare netuid, so on unstamped ticks a held position could accrue the rates of the unrelated subnet that later took its netuid. Research-local copies of the CSV model in the yield-carry bot are outside this package and are not changed by this fix.

**Measured (2026-10-08).** Rows rejected by the rule in the APY CSVs used by research (daily rate = gross APY / 365): `research/taoflute-value-screen-pit/out/per_subnet_daily_yield_20260831.csv` (2025-11-01 to 2026-08-31, 38,792 rows) and `per_subnet_daily_yield_taoflow_20260804.csv` (35,224 rows): one each, netuid 49 on 2025-11-21 at 0.080 per day. `bots/yield-carry-bot/research/data/per_subnet_daily_yield_extended.csv` (2025-02-13 to 2026-04-28, 49,173 rows): two, netuid 73 on 2025-03-16 at 0.025 per day and netuid 49 on 2025-11-21 at 0.160 per day. One rejected day credits at most one day of that rate (for example 0.16 alpha per alpha held on netuid 49 that day under the extended CSV), so the size of the effect on any published result depends on whether it held netuid 49 across 2025-11-21; that was not checked.

**What changed.**
- Both models apply the provider rule per row (rate set to 0.0), expose `rejected_rate_rows` and `rejected_examples`, and emit one `UserWarning` at construction when any row was rejected. Shared helper: `bt_trading_tools.alpha_yield.implausible_rate`.
- Both accept an optional `reregistrations={netuid: [timestamps]}` (exact events from `bt_trading_tools.utils.lifecycle.reregistrations_from_csvs` on SDK snapshots where they cover the window, as recommended in section 4; otherwise `reregistrations_from_pool_history` on the UNFILTERED pool history, which dates events to the day). Accrual then stops at the first re-registration after entry, and the rebirth day's own rate row is not used. Without it the behavior is unchanged apart from the clamp. On ticks stamped with `stamp_generations` the engine already closes such positions at the deregistration refund, so this matters for unstamped ticks and custom loops.

**What to do.** Read `model.rejected_rate_rows` after building a CSV or history model. Pass `reregistrations` when the ticks are not stamped.

## 6. `yield_model=None` picks a rate source from the environment

**What happens.** `BacktestEngine(yield_model=None)` builds `build_default_yield_model()`, a cascade whose tiers are switched on by the environment: the validator-selection cache (`VALIDATOR_CACHE_PATH`, or the first existing file of `/root/.validator_selection/best_validators.json`, `~/.validator_selection/best_validators.json`, `/tmp/autobot_live_data/best_validators.json`), live taostats (`TAOSTATS_API_KEY`), live chain (`BT_NETWORK`), the empirical CSV estimate (`TAOSTATS_DATA_DIR`), then zero. The same script therefore credits different yield in different shells and on different machines. Every non-zero tier is a rate as of now (the cache file's last refresh, today's taostats or chain state, or a trailing window at the end of the CSV), so on a historical window the yield is not point-in-time, which is a form of lookahead. The tiers also disagree in size: on the emission-drought bot's 2026-06-30 to 2026-09-23 window, paper's logged rate (taostats or cache tier) was 2.4 to 9.5 times the backtest's empirical-tier rate on the same subnets (median 4.2; source: `bt_trading_tools/alpha_yield_history.py` module docstring and `alpha-trading/docs/edb_core_alignment_2026_10_05.md`).

**Measured (2026-10-08).** On this development machine, in a plain shell (no yield variables exported, no validator-cache file at the fallback paths), the default resolves to the zero tier: yield is not credited at all. With `TAOSTATS_DATA_DIR` set it resolves to the empirical tier; with `TAOSTATS_API_KEY` exported it would query today's live rates for whatever window is simulated. On the bot VPS the validator cache exists, so the same script resolves to the cache tier there.

**What changed.** The selection itself is unchanged (making the point-in-time `HistoricalYieldModel` the default is a pending decision). `BacktestEngine` now emits one `UserWarning` per distinct configuration per process naming the primary tier (`validator_cache`, `taostats_live`, `chain_live`, `empirical`, or `zero`), the configured tiers, and, for non-zero tiers, the lookahead note; the description is stored in `engine.default_yield_cascade` (None when a model is passed). `bt_trading_tools.alpha_yield.describe_default_yield_cascade()` returns the same description without building anything. The primary tier is what the configuration says will be tried first; a subnet the tier cannot answer falls through to the next one (`CascadingYieldProvider.instrumentation_snapshot()` reports what was actually used).

**What to do.** Pass `yield_model` explicitly in every backtest whose number will be relied on, and record which one. For point-in-time rates use `bt_trading_tools.alpha_yield_history.HistoricalYieldModel(build_rate_table(load_validator_yield_history(...)))`.

## 7. Rescaled reserves turn a close from a different pool into a fillable pool (measured 2026-10-09, revised 2026-10-10)

**What happens.** With the opt-in `reserves="rescale"` (section 2), or `rescale_tick_reserves` on a custom builder, the daily snapshot supplies only `k` and the reserves are moved to the hourly close. When the close was printed by a different pool than the snapshot describes, the engine gets a pool at that close with the old pool's depth, fills against it and marks positions at it. The ratio used below is `close / implied`, where `implied = tao_pool / alpha_pool` of the paired daily snapshot (at or before the hour).

**Where the bad closes come from (checked 2026-10-09 against `delegations` parquet and `pool_history`).** The three cases flagged by the AutoBot rerun are netuid re-registrations, not corrupt rows in `delegation_ohlcv_hourly`:
- netuid 97, 2026-03-13 09:00 UTC, close 40.09 against implied 0.00266. At block 7,735,450 (09:19:36 UTC) about 200 UNDELEGATE rows in one block mark the old subnet's deregistration; from 09:37 a new subnet trades in a tiny pool, its price rising from about 1.7 to above 7 TAO per alpha by 09:51 (DELEGATE rows of 0.01 to 0.1 TAO). The end-of-day snapshot shows the new pool: 0.255 TAO and 0.258 alpha (implied 0.99). The 40.09 close is the new pool's real price; the paired snapshot is the old pool (3,645 TAO, implied 0.00266). Some refund rows in the deregistration block carry meaningless prices (amount / alpha, for example 44,698), but the close itself came from the new subnet's trades.
- netuid 82, 2026-04-22 20:00 UTC (close 4.43, implied 0.00216): same pattern, deregistration block 8,026,517 at 20:29:12 UTC, new subnet's intra-day snapshot at 22:49 shows 0.81 TAO in the pool.
- netuid 76, 2026-02-19 to 2026-02-20 (a rebirth day): a new subnet bootstrapping from about 23:07 UTC on 2026-02-19 with DELEGATE rows priced up to about 96,000 TAO per alpha in a pool holding 0.04 alpha; the 2026-02-19 end-of-day snapshot implies 539, the next day's 0.0096.
So the right treatment is to drop these hours: raw reserves describe a pool that no longer exists, and clipping would invent a price.

**Measured distribution (2025-11-14 to 2026-06-26, `load_parquet_ticks` with `bar_label="start"` before the guard existed, parquet cache as of 2026-10-09 06:04 local).** 652,681 subnet-hours on 129 netuids. Root (netuid 0) has a fixed close of 1.0 and a snapshot implying about 4 (ratio about 0.25 on every hour from 2026-04-13, 1,383 hours); it is not a constant-product alpha pool and is exempt. Over the other 651,298 subnet-hours, |log(close / implied)| has median log(1.007), p90 log(1.049), p99 log(1.18), p99.9 log(1.83), p99.99 log(175). Counts beyond a factor (either direction), split by whether the hour is within 2 days of a re-registration of that netuid (`reregistrations_from_pool_history`), within 2 days of the end of the subnet's startup mode, or neither:

| Factor away from implied | Subnet-hours | Near re-registration | Near startup-mode end | Neither |
|---|---|---|---|---|
| 1.5x to 2x | 485 | 19 | 90 | 376 |
| 2x to 3x | 126 | 8 | 47 | 71 |
| 3x to 5x | 148 | 56 | 57 | 35 |
| 5x to 10x | 145 | 90 | 24 | 31 |
| over 10x | 139 | 108 | 27 | 4 |

The startup-mode-end hours are mostly real price discovery (for example netuid 109 on 2026-01-13: price fell from 0.033 to about 0.004 within the day, with the previous snapshot still at 0.033); they are dropped because their depth is unknown, not because the price is wrong.

**What changed.**
- Price-outlier guard. `load_parquet_ticks` drops a subnet-hour whose close is more than `outlier_max_ratio` times away from the paired snapshot's implied price, either direction (`DEFAULT_OUTLIER_MAX_RATIO = 3.0`; root exempt via `OUTLIER_EXEMPT_NETUIDS`). It is causal: it uses only the close and a snapshot at or before it. Default: ON with `reserves="rescale"`, OFF with `reserves="raw"` (the default path stays the historical one; pass a number to opt in). The reason for the split: rescaling is what turns such a close into a pool the engine fills, marks and refunds at; under raw reserves the engine fills and marks at the snapshot, so the bad close can mislead a strategy's decision but does not move fills or marks. `rescale_tick_reserves` applies the same test by default, removing the subnet from the tick instead of rescaling it.
- Exact events and the event bar. With exact `reregistrations=` (section 4) a position opened earlier on the event day is closed at the old pool (tested under raw/start, raw/end and rescale/end). One hole remains with `bar_label="start"`: the bar containing the event is labeled before the event, so it is stamped as the old subnet, but its close is the new pool's. On the rescale path the ratio guard removes that hour when the new price is more than 3x away (tested).
- Pre-rebirth reserves (opt-in). `drop_pre_rebirth_reserves=True` drops a subnet-hour at or after a re-registration whose snapshot predates the event (reserves of the dead subnet), in any mode. Events are the ones used for stamping: exact when `reregistrations=` is given (cut at the event time), else the daily detector's (cut at midnight, so on the event day the hours before the end-of-day snapshot go too: up to one day of lookahead). It is off by default because with the daily detector it is close to the event-day blackout that did not help on a HODL window (section 4). An earlier branch (`fix/trap-fixes-2026-10-08`) had it on by default when rescaling.
- Every drop warns with counts and examples; `outlier_report=[]` collects `(timestamp, netuid, close, implied, reason)` per dropped hour, reason `"ratio"` or `"pre_rebirth_reserves"`.

**The 3x default is a judgment call, for Aaron to confirm.** The argument: 1.5x to 3x is dominated by hours near neither event (447 of 611), which are mostly ordinary fast moves that surge and spike strategies trade, so a tighter cut would mainly remove real signal; beyond 3x, 362 of 432 hours are near a re-registration or a startup-mode end. A looser cut (5x or 10x) would keep 148 to 293 more hours in which the snapshot is at least 3x off, which means the fill depth is unknown by at least that much. The cut matches the stopgap `guard_outliers` in the AutoBot rerun branch.

**Drops on the measured window (2025-11-14 to 2026-06-26).** With the ratio guard alone, the hours beyond 3x in the table above: 432 non-root subnet-hours (148 + 145 + 139), about 0.07% of 651,298. With the old branch's default-on pre-rebirth drop applied first, 1,600 hours were dropped (1,230 on 34 netuids for pre-rebirth reserves, then 370 on 21 netuids for the ratio).

**Effect on the AutoBot cross-validation that exposed it (rerun 2026-10-10, parquet cache of that morning).** AutoBot V2-moderate, original `v2_moderate_depth_fit.py` protocol: train window 2025-11-14 to 2026-04-15, 3-fold `PurgedWalkForwardCV` (purge 7 days), config spike 7%, trade 0.5 TAO, inventory cap 0.5 TAO, 30 TAO capital, friction-true engine with the AutoBot fee and realism models, `realism_rng_seed=42`, zero yield, pool filter off. One seed only. Return per test fold (max drawdown):

| Loader | Fold 0 (2026-01-11 to 02-11) | Fold 1 (2026-02-11 to 03-14) | Fold 2 (2026-03-14 to 04-15) |
|---|---|---|---|
| raw, daily detector (main's default) | +17.42% (2.8%) | +12.95% (42.8%) | +8.69% (6.0%) |
| rescale, no guard | +33.45% (1.1%) | +64,312.55% (1.0%) | +10.04% (85.2%) |
| rescale, ratio guard (the rescale default) | +33.45% (1.1%) | +19.62% (41.0%) | +12.02% (4.9%) |
| rescale, ratio guard, `drop_pre_rebirth_reserves=True` | not run | +5.01% (3.1%) | not run |
| rescale, ratio guard, exact SDK events | not run (no local SDK snapshots) | +4.88% (3.2%) | not run |
| raw, exact SDK events | not run | +1.92% (43.5%) | not run |
| raw, `drop_pre_rebirth_reserves=True` | not run | -2.35% (10.7%) | not run |

Fold 1's remaining 41% drawdown under the rescale default comes from netuid 76's rebirth day (equity peak 64.35 TAO on 2026-02-20 01:00 UTC): the old subnet's hours on 2026-02-19 are stamped as the new subnet by the midnight-floored daily detector, and their closes are within 3x of the snapshot, so the ratio guard keeps them. Exact SDK events (event at 2026-02-19 03:00 UTC) or the opt-in pre-rebirth drop remove it. The raw rows show that the raw default is exposed too (43% drawdowns in fold 1), and the ratio guard is off there by default. Local SDK snapshots start 2026-02-11, so folds 0 and 2 were not run with exact events (fold 2 is covered and was simply not run). These are single-seed results on one configuration; they show which loader options remove absurd marks, not what the strategy earns.

**What to do.** For anything built on `load_parquet_ticks` or a custom builder: pass exact `reregistrations=` where SDK snapshots reach; on the rescale path keep the ratio guard on and read the drop counts in the warnings; outside SDK coverage, consider `drop_pre_rebirth_reserves=True` and report results with and without it. Results that held a subnet across a re-registration day should not be relied on until rerun.

## Sparse presence

The parquet loader emits a subnet only in hours where it traded (no forward fill), so a strategy that needs a subnet present every hour must handle gaps, and the set of subnets present at the first tick depends on the label convention.

## Quick checks before trusting a number

```python
from bt_trading_tools.data import reserve_price_gap
print(reserve_price_gap(ticks))            # SDK ticks: median under ~0.001; raw daily reserves: 0.005 to 0.01 (2026 windows above)
# custom builder pairing an intra-day price with daily reserves (opt-in, section 2):
# from bt_trading_tools.data import rescale_tick_reserves; rescale_tick_reserves(ticks)
# rep = []; load_parquet_ticks(..., outlier_report=rep)  # rep lists hours dropped as price outliers or pre-rebirth reserves (section 7)
print(engine.default_yield_cascade)        # None if you passed yield_model; else which env tier was used (section 6)
print(getattr(engine.yield_model, "rejected_rate_rows", None))  # CSV/history models: rows rejected as implausible (section 5)
res = BacktestEngine(...).run(ticks, strategy)
print(res.orders_checked, res.orders_on_inconsistent_reserves)
```

For many independent windows or trades: `panel_forward_returns(...)`, or `engine_factory=seeded_engine_factory(base_seed)`.

## Costs measured with the corrected setup (1 TAO round trip at unchanged price, distinct seeds, 2026-08-01 to 2026-09-20)

| Pool depth (TAO) | 0.25 TAO clip | 1 TAO clip | 3 TAO clip |
|---|---|---|---|
| under 500 | -1.38% | -0.90% | -1.60% |
| 500 to 2,000 | -1.28% | -0.59% | -0.76% |
| 2,000 to 10,000 | -1.28% | -0.49% | -0.39% |
| 10,000 to 50,000 | -1.29% | -0.47% | -0.31% |
| over 50,000 | -1.25% | -0.43% | -0.26% |

Net return on cost of a buy followed by a sell 15 minutes later on random subnet-times, so the negative of each cell is the cost of trading in and out plus a little price noise. Fixed gas and proxy fees dominate at 0.25 TAO. Samples were 146 to 393 per cell; standard errors were 0.01% to 0.03%.
