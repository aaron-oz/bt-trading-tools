# Known traps in backtests built on this package

Two silent problems, each measured on real data in October 2026, each of which produced a plausible-looking but wrong return in at least one study. Both are now guarded in code; this page says what they are, how large they were, how to check for them, and what changed.

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
On the SDK feed itself the pool-implied price matches the tick price (median absolute gap about 0.005%; about 0.8% of subnet-ticks exceed 5%, mostly odd or new pools).

**Who is affected.** Any strategy whose price changes faster than the reserve snapshot: hourly-signal studies, surge or breakout entries, and anything evaluated on `load_parquet_ticks`. At the time of writing that loader is used by roughly ten autobot research scripts, a bluechip-fade script, and `bt_strategy.research.baselines`. Their numbers were produced with daily reserves and have not been re-checked.

**What changed.**
- `load_parquet_ticks(..., reserves=...)` has two modes. `"raw"` (the default; the historical behavior) keeps the daily reserves. `"rescale"` keeps only the invariant `k = tao x alpha` and rescales the reserves to the tick price (`tao' = sqrt(k x price)`, `alpha' = sqrt(k / price)`). Correction (2026-10-07): an earlier version of this page and PR #1 made `"rescale"` the default; that was premature (see "Why rescale is not the default" below) and the default is back to `"raw"`.
- Each tick's `signals` carries `reserve_gap_raw` (raw implied price / close - 1) and `reserve_age_s` (age of the daily snapshot).
- `reserve_price_gap(ticks)` summarizes the mismatch for any tick source. `BacktestResults.orders_checked` and `.orders_on_inconsistent_reserves` count how many orders sat on reserves more than 5% off the tick price, and the engine warns when more than 10% of at least 10 orders do.
- `load_parquet_ticks` warns once per process that the source has known issues (this page), whatever the options.

**Measured age effect (2026-10-07, 376,113 subnet-hours with at least one trade, 2026-03-01 to 2026-07-30, against SDK pool prices).** Absolute log error of the daily-reserve implied price versus the true pool price at the tick label, by age of the daily snapshot: median 0.0% and p90 0.5% within 30 minutes of the snapshot; median 0.3% and p90 2.3% at 0.5 to 6 hours; median 0.9% and p90 4.5% at 6 to 12 hours; median 1.7% and p90 7.9% at 18 to 24 hours. Windows that begin and end at midnight sit within seconds of a snapshot, which is why buy-and-hold between midnight endpoints is barely affected.

**Why rescale is not the default.**
- It matches the price at constant k; it does not recover depth changes (a surge can double pool TAO within a day), and on active strategies it moved results by orders of magnitude (an autobot fade study: one variant +275% under raw reserves and +25,962% rescaled; a bluechip-fade variant -33% raw and +25% rescaled). Neither number is credible; the point is that rescaling is not a safe drop-in.
- Against SDK results on midnight-aligned HODL windows, with re-registered subnets removed, mean absolute error was 1.1 percentage points rescaled and 1.8 raw (11 window-rules): both close, rescale slightly closer. With re-registered subnets left in, both parquet modes were off by up to 40 points on the same universe because of problem 4.
- Where fills or entries depend on the price path, use SDK ticks.

## 3. The bar label carries a one-hour lookahead (measured 2026-10-07)

The hourly OHLCV `time` is the bar START; the `close` is the pool price at the bar END. Against SDK pool prices (same 376,113 subnet-hours), the close matched the true pool price at the label with median absolute log error 0.27% (p90 2.0%) and one hour after the label with median 0.11% (p90 1.0%); in hours where the pool moved 5% or more the close matched the price one hour later with median error 0.7% and the price at the label with median 6.8%. `load_parquet_ticks` stamps ticks with the bar start, so every tick carries a price from one hour in the future. For buy-and-hold between midnight endpoints this shifts both ends together and does little. For any strategy that reads the price at the tick (hourly signals, spike fades) it is a lookahead. `load_parquet_ticks(..., bar_label="end")` stamps each tick with the bar end so the close is known at its timestamp; it is opt-in because it changes every existing result.

## 4. The parquet path misses some subnet re-registrations (measured 2026-10-07)

On the same universes and windows, the 15-minute SDK path closed positions on re-registration of netuids that the parquet path did not flag: SN82 in the 2026-04-22 to 2026-06-06 HODL window, SN69 in the cheapest-10 holdout (2026-04-16 to 2026-06-26), SN90 in the 2026-06-06 to 2026-07-21 window. A missed re-registration values the old position against the new subnet's pool and shows fabricated profit: +19.9 TAO (SN82) and +34.9 TAO (SN69) on 100 TAO of capital, against -0.1 and +0.2 TAO on SDK ticks. With `reserves="rescale"` fabricated gains also appeared where the guard did fire (SN116: +20.7 TAO rescaled vs -2.7 TAO on SDK). The 2K-5K HODL headline for the 2026-04-22 window was +14.5% (raw) and +34.3% (rescaled) against -6.0% on SDK ticks, almost entirely this effect. Parquet results over windows containing re-registrations are not reliable; use SDK ticks, or pass exact events (below).

Diagnosis (2026-10-09). The daily detector itself finds most events: against 51 events detected on SDK snapshots between 2025-09-23 and 2026-10-09, the daily `pool_history` rule found 49 (misses: SN69 on 2026-05-08, SN116 on 2026-10-04, the latter within a day of the data end) and reported none that SDK snapshots did not show. The fabricated profit comes from how the event is applied: the loader floors each event to midnight of its day, so a position opened earlier that day (for example the first tick of a window that starts on the event day, as in SN82 on 2026-04-22, true event 20:55 UTC) is stamped as the NEW subnet while sitting on the old pool, and the engine never closes it. A drop of the cumulative `recycled_since_registration` counter in `subnet_history` found 48 of the 51 and one event the SDK rule does not show; it did not beat the pool-history rule.

Fix: `load_parquet_ticks(..., reregistrations=...)` takes exact events, and `utils.lifecycle.reregistrations_from_csvs(paths)` builds them from SDK snapshot CSVs. Check on 11 strategy-window cases (HODL 2K-5K band and market in five windows, cheapest-10 holdout) against SDK ticks: mean absolute return difference 6.3 points with the daily detector, 2.5 points with SDK events (SN82 window 2K-5K: +14.5% daily detector, -5.4% SDK events, -6.0% SDK ticks). The cheapest-10 holdout went from +74.4% to +39.5% against +49.0% on SDK ticks (the remaining gap was not diagnosed). An event-day blackout (dropping the netuid's ticks on the event day) was tried and did not help (window 2026-04-22 2K-5K error 29.6 points), so it is not offered. Without SDK coverage (before 2025-09 here) the daily rule is still all there is, and its day-level dating limitation remains.

## Sparse presence

The parquet loader emits a subnet only in hours where it traded (no forward fill), so a strategy that needs a subnet present every hour must handle gaps, and the set of subnets present at the first tick depends on the label convention.

## Quick checks before trusting a number

```python
from bt_trading_tools.data import reserve_price_gap
print(reserve_price_gap(ticks))            # median_abs_gap near 0.00005 on SDK ticks; above ~0.01 means stale reserves
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
