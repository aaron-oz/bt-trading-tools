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
- `load_parquet_ticks(..., reserves="rescale")` is now the default. The daily snapshot supplies only the invariant `k = tao x alpha`; reserves are rescaled to the hourly close (`tao' = sqrt(k x price)`, `alpha' = sqrt(k / price)`), the approach the xs-momentum-rotation study validated against true hourly reserves for 2026-03 to 2026-05. `reserves="raw"` restores the old behavior and warns. This changes the numbers of any caller relying on the old default; those numbers were mispriced.
- Each tick's `signals` now carries `reserve_gap_raw` (raw implied price / close - 1) and `reserve_age_s` (age of the daily snapshot).
- `reserve_price_gap(ticks)` summarizes the mismatch for any tick source. `BacktestResults.orders_checked` and `.orders_on_inconsistent_reserves` count how many orders sat on reserves more than 5% off the tick price, and the engine warns when more than 10% of at least 10 orders do.

**What rescaling does not fix.** Pool depth (`k`) can change during the day, for example when a surge draws in TAO. Rescaling matches the price, not the depth, so slippage on large clips can still be wrong around big flows. Where depth matters, use SDK ticks (contemporaneous reserves). SDK snapshots are local from 2026-02-11 and, on the bot VPS, a 5-minute file exists back to 2025-08-18 (`/root/sdk_backfill/sdk_snapshots/sdk_5min.csv`, ends 2026-02-14).
The parquet loader also emits a subnet only in hours where it traded (no forward fill), so a strategy that needs a subnet present every hour must handle gaps.

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
