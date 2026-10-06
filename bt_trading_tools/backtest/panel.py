"""Cross-sectional panel forward returns through the BacktestEngine.

Motivation: research panels (screens, IC studies, event studies) need
per-(subnet, entry-date, horizon) NET forward returns. Before this module
those were computed in ad-hoc loops, which re-introduced already-squashed
bugs (zero/partial friction, missing fees, hand-rolled yield, sub -100%
returns). Per the project hard rule, ALL forward-return computation goes
through the engine; this module is the engine-native way to do panels.

Public-safe: generic mechanics only — no universe logic, no signals, no
strategy IP. Callers supply the (netuid, clip) baskets.

Design: each (entry_ts, horizon) window is one ``BacktestEngine.run`` over
the tick slice [entry_ts, exit_ts]. A ``ScheduledBasketStrategy`` buys the
basket at the first available tick per subnet and never sells; the engine's
end-of-run force-close provides the exit with full sell-side friction and
yield accrual, and its missing-subnet fallback (AMM sell against the
last-seen pool state; ~= spot for clip-sized positions) approximates the
empirically observed dereg payout rule (T/alpha_staked ~= spot; SN103
observation 2026-08-03, see alpha-trading
docs/bittensor-mechanics-primer.md § Deregistration). Before 2026-09-10
the engine's fallback was ENTRY price (bug; see engine.py force-close),
which neutralized trades on subnets absent from the final tick.
Per-position outcomes are read from ``results.trades``.

Realism seeds (known trap, fixed 2026-10): each window is a NEW engine, and a
new engine with the default ``realism_rng_seed=0`` replays the same realism
draws. Averaging many one-trade windows then averages one draw N times and
overstates returns (measured about +0.4 percentage points per 1 TAO round
trip). This module therefore gives every window its own deterministic seed,
derived from a base seed and the window (``derive_window_seed``). Passing
``realism_rng_seed=<int>`` sets the BASE seed (reproducible, still distinct per
window); ``None`` is nondeterministic. If you pass your own ``engine_factory``
you own the seeding; ``seeded_engine_factory`` builds a safe one. See
docs/known_traps.md.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from itertools import count
from typing import Any, Callable, Iterable, Optional

from bt_trading_tools.backtest.engine import BacktestEngine
from bt_trading_tools.backtest.types import Order, Position, TickData


def derive_window_seed(base_seed: int, entry_ts: int, netuids: Iterable[int]) -> int:
    """Deterministic, process-independent realism seed for one window.

    A hash (not Python's salted ``hash``) of ``(base_seed, entry_ts, sorted
    netuids)``, so reruns reproduce exactly while different windows get
    different draws.
    """
    key = (f"{int(base_seed)}|{int(entry_ts)}|"
           f"{','.join(str(int(n)) for n in sorted(netuids))}").encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**31 - 1)


def seeded_engine_factory(base_seed: int = 0, **fixed_kwargs: Any) -> Callable[..., BacktestEngine]:
    """An ``engine_factory`` that gives every engine it builds a distinct seed.

    Use it whenever you run many independent windows or trades, each in its own
    engine (a new engine with a repeated seed repeats the same realism draws).
    Seeds are ``base_seed + 1, base_seed + 2, ...`` in call order, so a run is
    reproducible as long as the calls happen in the same order. Extra keyword
    arguments are passed to every engine; per-call keywords override them.
    """
    counter = count(int(base_seed) + 1)

    def factory(**kwargs: Any) -> BacktestEngine:
        merged = {**fixed_kwargs, **kwargs}
        merged["realism_rng_seed"] = next(counter)
        return BacktestEngine(**merged)

    return factory


class ScheduledBasketStrategy:
    """Buy a fixed (netuid -> tao clip) basket at/after ``entry_ts``; hold.

    Each subnet is attempted once, at the first tick at/after ``entry_ts``
    where it is present, up to ``entry_deadline_ts`` (no retry after a
    realism-layer rejection — one-shot semantics so failed buys surface as
    missing cells rather than silently retried fills). Exits are the
    engine's end-of-run force-close.
    """

    def __init__(
        self,
        clips: dict[int, float],
        entry_ts: int,
        entry_deadline_ts: Optional[int] = None,
    ):
        self.clips = {int(k): float(v) for k, v in clips.items()}
        self.entry_ts = int(entry_ts)
        self.entry_deadline_ts = (
            int(entry_deadline_ts) if entry_deadline_ts is not None else None
        )
        self._attempted: set[int] = set()

    def on_tick(
        self,
        tick: TickData,
        positions: dict[int, Position],
        capital: float,
        portfolio_value: float,
    ) -> list[Order]:
        if tick.timestamp < self.entry_ts:
            return []
        if (
            self.entry_deadline_ts is not None
            and tick.timestamp > self.entry_deadline_ts
        ):
            return []
        orders = []
        for netuid, tao_amt in self.clips.items():
            if netuid in self._attempted or netuid not in tick.subnets:
                continue
            self._attempted.add(netuid)
            orders.append(
                Order(
                    netuid=netuid,
                    side="buy",
                    tao_amount=tao_amt,
                    reason="basket_entry",
                )
            )
        return orders


def slice_ticks(ticks: list, start_ts: int, end_ts: int) -> list:
    """Slice a timestamp-ascending TickData list to [start_ts, end_ts]."""
    keys = [t.timestamp for t in ticks]
    lo = bisect_left(keys, int(start_ts))
    hi = bisect_right(keys, int(end_ts))
    return ticks[lo:hi]


def run_basket_window(
    ticks: list,
    clips: dict[int, float],
    entry_ts: int,
    exit_ts: int,
    entry_deadline_s: int = 5 * 86400,
    engine_factory: Optional[Callable[..., BacktestEngine]] = None,
    **engine_kwargs: Any,
) -> dict[int, dict]:
    """Run one basket window through the engine; return per-netuid outcomes.

    Returns {netuid: {"net_return", "tao_cost", "tao_received", "fees",
    "alpha_yield_accrued", "exit_reason", "status"}}. Subnets whose single
    buy attempt failed (realism rejection, pool-safety drop, missing at
    entry) appear with status "no_fill" and net_return None.

    ``engine_kwargs`` are forwarded to BacktestEngine (friction defaults
    stay ON per project policy). Capital defaults to the basket total.

    Seeding: without an ``engine_factory`` the window's realism seed is
    ``derive_window_seed(base, entry_ts, clips)`` where ``base`` is the
    ``realism_rng_seed`` you pass (default 0); ``realism_rng_seed=None`` stays
    nondeterministic. With an ``engine_factory`` you own the seeding (see
    ``seeded_engine_factory``).
    """
    window = slice_ticks(ticks, entry_ts, exit_ts)
    strategy = ScheduledBasketStrategy(
        clips, entry_ts, entry_deadline_ts=entry_ts + entry_deadline_s
    )
    engine_kwargs.setdefault("capital", sum(clips.values()) * 1.001)
    if engine_factory is None:
        base = engine_kwargs.pop("realism_rng_seed", 0)
        engine_kwargs["realism_rng_seed"] = (
            None if base is None else derive_window_seed(base, entry_ts, clips)
        )
    factory = engine_factory or BacktestEngine
    engine = factory(**engine_kwargs)
    results = engine.run(window, strategy)

    out: dict[int, dict] = {}
    for tr in results.trades:
        if tr.get("status") == "failed":
            out.setdefault(
                int(tr["netuid"]), {"status": "no_fill", "net_return": None}
            )
            continue
        if tr.get("tao_received", 0) == 0 and tr.get("tao_cost", 0) == 0:
            continue  # buy-side record; wait for the closing sell record
        if tr.get("tao_cost", 0) > 0 and tr.get("tao_received", 0) >= 0:
            netuid = int(tr["netuid"])
            out[netuid] = {
                "status": "closed",
                "net_return": tr["pnl"] / tr["tao_cost"],
                "tao_cost": tr["tao_cost"],
                "tao_received": tr["tao_received"],
                "fees": tr.get("fees", 0.0),
                "alpha_yield_accrued": tr.get("alpha_yield_accrued", 0.0),
                "exit_reason": tr.get("reason", ""),
                "exit_source": tr.get("exit_source", ""),
            }
    for netuid in clips:
        out.setdefault(
            int(netuid), {"status": "no_fill", "net_return": None}
        )
    return out


def panel_forward_returns(
    ticks: list,
    entries: Iterable[tuple[int, dict[int, float]]],
    horizon_s: int,
    engine_factory: Optional[Callable[..., BacktestEngine]] = None,
    **engine_kwargs: Any,
) -> list[dict]:
    """Panel API: for each (entry_ts, clips) run one basket window.

    Returns a flat list of row dicts: {"entry_ts", "netuid", plus the
    run_basket_window outcome fields}. Deterministic given engine_kwargs and
    the base ``realism_rng_seed`` (each window still gets its own derived
    seed, so windows do not share one set of realism draws).
    """
    rows: list[dict] = []
    for entry_ts, clips in entries:
        res = run_basket_window(
            ticks,
            clips,
            entry_ts,
            entry_ts + horizon_s,
            engine_factory=engine_factory,
            **dict(engine_kwargs),
        )
        for netuid, rec in res.items():
            rows.append({"entry_ts": int(entry_ts), "netuid": netuid, **rec})
    return rows
