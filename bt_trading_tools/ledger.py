"""Position ledger: the bookkeeping shared by backtest and paper execution.

Why this module exists
----------------------
Backtest (``BacktestEngine``) and paper (``bt_strategy.bot.PaperBotBase``)
each grew their own position bookkeeping. Three defects in five weeks lived in
that duplicated layer, each on one side only: paper double-sell (2026-08),
backtest force-close at entry price (2026-09-10), and netuid rebirth (paper
fixed 2026-09-22, backtest not). Every function here is pure arithmetic on a
``Position`` and numbers the caller supplies, so both execution paths can call
the same code and a fix lands once.

Conventions (decided 2026-10-01, see
``alpha-trading/docs/backtest_paper_duplication_audit_2026_10_01.md``)
---------------------------------------------------------------------
* **Valuation is liquidation value net of fees**: what an AMM sale of the
  whole position would return right now, minus the sell fee. Not mid price.
* **No implicit pool cap.** Any cap on order size relative to pool depth is a
  per-bot decision made by the caller.
* **Partial fills and partial sells keep the remainder.** Nothing is deleted
  unless the remaining alpha is effectively zero.
* **Yield is folded into the position** whenever its size changes (top-up or
  sell), and the yield clock restarts from that moment, so alpha added later
  never accrues yield it did not earn. Yield alpha carries zero cost basis.
* **Yield accrues along the rate path** (2026-10-10). The caller calls
  :func:`accrue_yield` on every open position once per tick; each call
  credits ``yield_fn`` over the interval since the previous call, so each
  interval is paid at the rate in effect for that interval and earlier
  accrual is never revalued. Before this, yield was
  ``alpha x rate(now) x (now - anchor)``: the CURRENT rate applied to the
  whole time since the anchor, so a falling rate marked down yield already
  earned (measured on yield-carry 2026-09-23..10-08: -0.22 TAO on a 0.31 TAO
  window figure, ``docs/yc_onchain_yield_2026_10_10.md`` in alpha-trading).
  A caller that never calls :func:`accrue_yield` gets exactly the old
  formula, which is how ``BacktestEngine(yield_accrual="anchor")`` and
  ``PaperBotBase.yield_accrual = "anchor"`` reproduce it for regression
  comparisons. Between calls, and from the last call to a sale or a mark,
  the uncredited tail is paid at the rate ``yield_fn`` returns for that tail.
  A position that has never been accrued (``yield_accrued_to is None``, for
  example one loaded from a state file written before this change) starts
  from the old formula's value at its first accrual, so a restart onto this
  code does not move any position's yield.
* **A deregistered subnet pays a refund**, approximated as spot price times
  alpha at the last snapshot before the re-registration. That reproduced the
  project's measured SN103 payout within -2.9% (2026-09-22). Refunds measured
  on other subnets ranged from 41% to 120% of spot (2026-07-28), so treat it
  as an approximation.
"""

from __future__ import annotations

from typing import Callable, Optional

from bt_trading_tools.amm import amm_sell
from bt_trading_tools.backtest.types import Position

# (netuid, alpha_qty, since_ts, now_ts) -> accrued alpha. Pure.
# A yield function MAY also accept a keyword ``origin``: the start of the
# position's current yield clock (its anchor). Interval accrual calls the
# function on short intervals, so a model that stops accrual at a netuid
# re-registration must search for the re-registration from ``origin``, not
# from the interval start, or an interval after the rebirth would earn the new
# subnet's rate. BacktestEngine._yield_fn and PaperBotBase._yield_fn forward it
# to the model as ``generation_origin`` when the model accepts it.
YieldFn = Callable[[int, float, float, float], float]

DUST_ALPHA = 1e-9

_ORIGIN_SUPPORT: dict = {}


class GenerationMismatch(ValueError):
    """A fill was booked against a position from a different subnet
    generation. The caller must close the old position first."""


def yield_anchor(pos: Position) -> float:
    return pos.entry_time if pos.yield_anchor_time is None else pos.yield_anchor_time


def accepts_kwarg(fn, name: str) -> bool:
    """True when callable ``fn`` takes keyword ``name`` (or ``**kwargs``).
    Cached per underlying function, so bound methods are cheap to test."""
    key = (getattr(fn, "__func__", fn), name)
    hit = _ORIGIN_SUPPORT.get(key)
    if hit is None:
        import inspect
        try:
            params = inspect.signature(fn).parameters
            hit = name in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            hit = False
        _ORIGIN_SUPPORT[key] = hit
    return hit


def _interval_yield(pos: Position, start: float, end: float,
                    yield_fn: YieldFn) -> float:
    """``yield_fn`` over ``[start, end]`` on the position's current alpha."""
    if end <= start:
        return 0.0
    if accepts_kwarg(yield_fn, "origin"):
        v = yield_fn(pos.netuid, pos.alpha_qty, start, end, origin=yield_anchor(pos))
    else:
        v = yield_fn(pos.netuid, pos.alpha_qty, start, end)
    return max(0.0, float(v))


def accrued_yield(pos: Position, now: float, yield_fn: Optional[YieldFn]) -> float:
    """Alpha accrued since the position's yield anchor, not yet folded.

    Equals ``yield_accrued`` (credited interval by interval by
    :func:`accrue_yield`) plus the uncredited tail from ``yield_accrued_to``
    to ``now``. For a position never accrued this is the pre-2026-10-10
    formula, ``yield_fn`` from the anchor to ``now``. When ``now`` is before
    ``yield_accrued_to`` (a deregistration close at the dead subnet's last
    snapshot, after ticks that already accrued past it), the overrun is
    removed at the rate ``yield_fn`` gives for it, floored at 0. With
    ``yield_fn`` None there is no yield at all. Does not mutate.
    """
    if yield_fn is None or pos.alpha_qty <= 0:
        return 0.0
    if pos.yield_accrued_to is None:
        return _interval_yield(pos, yield_anchor(pos), now, yield_fn)
    base = max(0.0, float(pos.yield_accrued))
    if now >= pos.yield_accrued_to:
        return base + _interval_yield(pos, pos.yield_accrued_to, now, yield_fn)
    overrun = _interval_yield(pos, max(now, yield_anchor(pos)),
                              pos.yield_accrued_to, yield_fn)
    return max(0.0, base - overrun)


def accrue_yield(pos: Position, now: float, yield_fn: Optional[YieldFn]) -> float:
    """Credit yield from ``yield_accrued_to`` (or the anchor) to ``now``.

    Call once per tick on every open position, before anything reads the
    position's value. Returns the alpha credited by this call. A call with
    ``now`` at or before ``yield_accrued_to`` changes nothing. With
    ``yield_fn`` None nothing is credited and the clock is not moved.
    """
    if yield_fn is None:
        return 0.0
    if pos.yield_accrued_to is not None and now <= pos.yield_accrued_to:
        return 0.0
    before = 0.0 if pos.yield_accrued_to is None else max(0.0, float(pos.yield_accrued))
    total = accrued_yield(pos, now, yield_fn)
    pos.yield_accrued = total
    pos.yield_accrued_to = now
    return total - before


def clear_accrual(pos: Position) -> None:
    """Forget the path accumulator, so :func:`accrued_yield` falls back to the
    pre-2026-10-10 formula (``yield_fn`` from the anchor to now). Used by the
    ``"anchor"`` accrual mode each tick, so that mode reproduces the old
    numbers exactly even on positions that were accrued in path mode before
    (for example a paper state file after a rollback)."""
    pos.yield_accrued = 0.0
    pos.yield_accrued_to = None


def fold_yield(pos: Position, now: float, yield_fn: Optional[YieldFn]) -> float:
    """Add accrued yield to ``alpha_qty`` and restart the yield clock.

    Yield alpha has zero cost basis, so ``tao_cost`` is unchanged and the
    per-alpha cost basis (``entry_price``) falls slightly. Returns the alpha
    folded in.
    """
    accrued = accrued_yield(pos, now, yield_fn)
    if accrued > 0:
        pos.alpha_qty += accrued
        if pos.alpha_qty > 0:
            pos.entry_price = pos.tao_cost / pos.alpha_qty
    pos.yield_anchor_time = int(now)
    pos.yield_accrued = 0.0
    pos.yield_accrued_to = int(now)
    return accrued


def book_buy(
    positions: dict[int, Position],
    netuid: int,
    alpha_received: float,
    tao_spent: float,
    fee: float,
    now: float,
    *,
    generation: Optional[int] = None,
    yield_fn: Optional[YieldFn] = None,
    metadata: Optional[dict] = None,
) -> Position:
    """Open a position or add to an existing one (cost-weighted).

    ``entry_time`` is kept from the first fill. Raises GenerationMismatch if
    the existing position belongs to a different subnet generation.
    """
    pos = positions.get(netuid)
    if pos is None:
        pos = Position(
            netuid=netuid,
            entry_price=tao_spent / alpha_received,
            alpha_qty=alpha_received,
            tao_cost=tao_spent,
            entry_time=int(now),
            entry_fees=fee,
            metadata=dict(metadata or {}),
            generation=generation,
            yield_anchor_time=int(now),
            yield_accrued=0.0,
            yield_accrued_to=int(now),
        )
        positions[netuid] = pos
        return pos

    if (generation is not None and pos.generation is not None
            and generation != pos.generation):
        raise GenerationMismatch(
            f"netuid {netuid}: buy in generation {generation} but the open "
            f"position is generation {pos.generation}")
    fold_yield(pos, now, yield_fn)
    pos.alpha_qty += alpha_received
    pos.tao_cost += tao_spent
    pos.entry_fees += fee
    pos.entry_price = pos.tao_cost / pos.alpha_qty
    if pos.generation is None:
        pos.generation = generation
    return pos


def book_sell(
    positions: dict[int, Position],
    netuid: int,
    alpha_sold: float,
    tao_received: float,
    fee: float,
    now: float,
    *,
    yield_fn: Optional[YieldFn] = None,
) -> dict:
    """Reduce a position by ``alpha_sold`` (after folding accrued yield).

    Cost basis and entry fees are released pro rata. The remainder stays
    open; the position is removed only when what is left is dust. Returns the
    accounting fields of the sell for the caller's trade record.
    """
    pos = positions[netuid]
    folded = fold_yield(pos, now, yield_fn)
    held = pos.alpha_qty
    alpha_sold = min(alpha_sold, held)
    fraction = alpha_sold / held if held > 0 else 1.0
    cost_of_sold = pos.tao_cost * fraction
    fees_of_sold = pos.entry_fees * fraction

    record = {
        "entry_time": pos.entry_time,
        "entry_price": pos.entry_price,
        "alpha_qty": alpha_sold,
        "tao_cost": cost_of_sold,
        "tao_received": tao_received,
        "pnl": tao_received - cost_of_sold,
        "fees": fees_of_sold + fee,
        "hold_seconds": now - pos.entry_time,
        "alpha_yield_accrued": folded,
        "generation": pos.generation,
    }

    pos.alpha_qty -= alpha_sold
    pos.tao_cost -= cost_of_sold
    pos.entry_fees -= fees_of_sold
    closed = pos.alpha_qty <= DUST_ALPHA
    if closed:
        del positions[netuid]
    record["position_closed"] = closed
    record["alpha_remaining"] = 0.0 if closed else pos.alpha_qty
    return record


def liquidation_value(
    alpha_qty: float, tao_pool: float, alpha_pool: float, sell_fee: float,
) -> float:
    """TAO an AMM sale of ``alpha_qty`` returns, minus ``sell_fee``, floored at 0."""
    if alpha_qty <= 0:
        return 0.0
    if tao_pool <= 0 or alpha_pool <= 0:
        return 0.0
    tao_out, _, _ = amm_sell(alpha_qty, tao_pool, alpha_pool)
    return max(0.0, tao_out - sell_fee)


def dereg_refund(alpha_qty: float, tao_pool: float, alpha_pool: float) -> float:
    """Approximate deregistration refund: spot price times alpha held.

    Use the pool state at the LAST snapshot of the old subnet, before the
    re-registration. See the module docstring for how accurate this is.
    """
    if alpha_qty <= 0 or tao_pool <= 0 or alpha_pool <= 0:
        return 0.0
    return alpha_qty * tao_pool / alpha_pool


def close_deregistered(
    positions: dict[int, Position],
    netuid: int,
    last_tao_pool: float,
    last_alpha_pool: float,
    last_seen_ts: float,
    *,
    yield_fn: Optional[YieldFn] = None,
) -> dict:
    """Close a position whose subnet was deregistered, at the refund value.

    Yield accrues up to the old subnet's last snapshot (``last_seen_ts``), not
    beyond: after deregistration the alpha no longer exists.
    """
    pos = positions[netuid]
    folded = fold_yield(pos, last_seen_ts, yield_fn)
    alpha = pos.alpha_qty
    refund = dereg_refund(alpha, last_tao_pool, last_alpha_pool)
    record = {
        "entry_time": pos.entry_time,
        "entry_price": pos.entry_price,
        "alpha_qty": alpha,
        "tao_cost": pos.tao_cost,
        "tao_received": refund,
        "pnl": refund - pos.tao_cost,
        "fees": pos.entry_fees,
        "hold_seconds": last_seen_ts - pos.entry_time,
        "alpha_yield_accrued": folded,
        "generation": pos.generation,
        "exit_price": refund / alpha if alpha > 0 else 0.0,
        "position_closed": True,
        "alpha_remaining": 0.0,
    }
    del positions[netuid]
    return record
