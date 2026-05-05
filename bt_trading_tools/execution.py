"""
Shared execution-realism layers for paper bots and backtest engines.

Both ``PaperBotBase.simulate_execution`` (paper) and
``BacktestEngine._execute_*`` (historical backtest) need the same realism
treatment: random order failures, latency draw, slippage noise, rate-tolerance
breach detection, partial fills. This module centralises the logic so the two
code paths can never drift.

Layer order (load-bearing — see docs/realistic_paper_execution_design.md §3
in the alpha-trading repo):

    1. Bernoulli failure (non-slippage causes — random_reject)
    2. CSV-only Gaussian slippage noise (skipped when live fetch already
       captured drift via ``live_spot_price``)
    3. Rate-tolerance breach (against post-noise exec_slippage_pct)
    4. Partial fill (default off — Bernoulli at ``partial_fill_rate``)
    5. Latency (always drawn, even for failed orders, for diagnostics)

Calibration:

    Defaults match the values in PaperBotBase as of 2026-04-21 (after
    DoubleDip live measurement of buy-failure rate). Per-bot subclasses
    override via class attributes; env-var overrides take precedence
    (``REALISM_<ATTR>``). Backtest defaults to the same values so a
    paper bot's expected performance is comparable to the backtest's
    expected performance — that's the whole point of unifying the layer.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, fields
from typing import Optional

from bt_trading_tools.amm import slippage_pct


logger = logging.getLogger(__name__)


# ── Calibration constants ────────────────────────────────────────────
#
# Observed chain-level buy-failure range: bagbot 0.3% (live, low-frequency,
# conservative thresholds) ↔ DoubleDip 8% (measured Mar 2026, high-frequency,
# reactive). Default 0.03 is a neutral midpoint; per-bot subclasses should
# override once they observe their own rate.
#
# Latency means/stds measured from DoubleDip live, Mar 2026.

DEFAULT_BUY_FAILURE_RATE: float = 0.03
DEFAULT_SELL_FAILURE_RATE: float = 0.0
DEFAULT_BUY_LATENCY_MEAN_S: float = 15.6
DEFAULT_BUY_LATENCY_STD_S: float = 5.4
DEFAULT_SELL_LATENCY_MEAN_S: float = 9.8
DEFAULT_SELL_LATENCY_STD_S: float = 4.3
DEFAULT_SETTLEMENT_DELAY_S: float = 12.0           # ≈ 1 Bittensor block

DEFAULT_RATE_TOLERANCE_BUY_PP: float = 2.0          # matches live default_slippage_buffer
DEFAULT_RATE_TOLERANCE_SELL_PCT: float = 50.0       # matches live default_sell_rate_tolerance

DEFAULT_SLIPPAGE_NOISE_MEAN_PCT: float = 0.05
DEFAULT_SLIPPAGE_NOISE_STD_PCT: float = 0.3
DEFAULT_SLIPPAGE_NOISE_FLOOR_PCT: float = -2.0

DEFAULT_PARTIAL_FILL_RATE: float = 0.0              # off until allow_partial_stake lands
DEFAULT_PENDING_ORDER_MAX_AGE_S: float = 3600.0     # orphan threshold on restart


@dataclass
class RealismConfig:
    """Calibrated realism parameters for execution simulation.

    Defaults are the paper-bot calibration as of 2026-04-21. Backtest +
    paper share these values so backtest Sharpe is comparable to live
    paper Sharpe (the whole point of unifying the layer).

    Construct directly or via ``RealismConfig.from_env()`` for env-var
    overrides (``REALISM_<UPPERCASE_ATTR>``).
    """
    enabled: bool = True

    buy_failure_rate: float = DEFAULT_BUY_FAILURE_RATE
    sell_failure_rate: float = DEFAULT_SELL_FAILURE_RATE

    buy_latency_mean_s: float = DEFAULT_BUY_LATENCY_MEAN_S
    buy_latency_std_s: float = DEFAULT_BUY_LATENCY_STD_S
    sell_latency_mean_s: float = DEFAULT_SELL_LATENCY_MEAN_S
    sell_latency_std_s: float = DEFAULT_SELL_LATENCY_STD_S
    settlement_delay_s: float = DEFAULT_SETTLEMENT_DELAY_S

    rate_tolerance_buy_pp: float = DEFAULT_RATE_TOLERANCE_BUY_PP
    rate_tolerance_sell_pct: float = DEFAULT_RATE_TOLERANCE_SELL_PCT

    slippage_noise_mean_pct: float = DEFAULT_SLIPPAGE_NOISE_MEAN_PCT
    slippage_noise_std_pct: float = DEFAULT_SLIPPAGE_NOISE_STD_PCT
    slippage_noise_floor_pct: float = DEFAULT_SLIPPAGE_NOISE_FLOOR_PCT

    partial_fill_rate: float = DEFAULT_PARTIAL_FILL_RATE
    pending_order_max_age_s: float = DEFAULT_PENDING_ORDER_MAX_AGE_S

    @classmethod
    def from_env(cls, prefix: str = "REALISM_") -> "RealismConfig":
        """Build a config with env-var overrides applied on top of defaults.

        Env keys are ``f"{prefix}{ATTR.upper()}"``. Boolean ``enabled`` accepts
        the strings ``1/true/yes/on`` (case-insensitive); other types are
        coerced via their type constructor.
        """
        cfg = cls()
        for f in fields(cls):
            env_key = f"{prefix}{f.name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if f.type == bool or f.name == "enabled":
                setattr(cfg, f.name, raw.strip().lower() in ("1", "true", "yes", "on"))
            else:
                cur = getattr(cfg, f.name)
                try:
                    setattr(cfg, f.name, type(cur)(raw))
                except (TypeError, ValueError):
                    logger.warning(
                        "RealismConfig: ignoring invalid env override %s=%r",
                        env_key, raw,
                    )
        return cfg


class RealismSimulator:
    """Apply the five realism layers to an action dict.

    Stateful only via the seeded RNG — every other input flows through the
    config + the action dict. Reusable across paper bots and backtest
    engines: both build an action dict (side, requested amounts, executed
    price, decision-time pool state) and call ``simulate_fill`` to get a
    realism-adjusted result.

    Action-dict contract:

        Required: ``"type"`` ∈ {"buy", "sell"} OR ``"side"``,
                  ``"netuid"``, plus side-specific amount fields below.
        Buys:     ``"tao_spent"`` (executed), ``"alpha_qty"`` (executed),
                  ``"price"`` (executed), ``"decision_pool_tao"`` (decision).
        Sells:    ``"alpha_qty"`` (executed), ``"tao_received"`` (executed),
                  optional ``"exit_price"``, optional ``"decision_pool_alpha"``.
        Optional: ``"live_spot_price"`` — when present, layer 2 (slippage
                  noise) is skipped because the live fetch already
                  captured real drift via ``"exec_slippage_pct"``.

    Mutates the action dict in place AND returns it (for ergonomic chaining).

    Sets / updates these fields:
        ``"status"`` ∈ {"executed", "failed", "partial"}
        ``"latency_ms"`` (always)
        ``"failure_reason"`` (only on status="failed")
        ``"intended_slippage_tolerance_pct"`` (always — for audit)
        ``"exec_slippage_pct"`` (when slippage noise applied)
        ``"requested_tao_amount"`` (buys, when not pre-set)
        ``"requested_alpha_amount"`` (sells, when not pre-set)
    """

    def __init__(self, config: Optional[RealismConfig] = None, rng_seed: Optional[int] = None):
        self.config = config or RealismConfig()
        self._rng = random.Random(rng_seed)

    # ── Public entry point ────────────────────────────────────────────

    def simulate_fill(self, action: dict) -> dict:
        """Run the five realism layers in order. Mutates + returns ``action``."""
        if not self.config.enabled:
            action.setdefault("status", "executed")
            return action
        side = action.get("type") or action.get("side")
        if side not in ("buy", "sell"):
            return action

        # Preserve requested amounts so partial/failed orders can compare
        # what we asked for vs what filled.
        if side == "buy" and "requested_tao_amount" not in action:
            action["requested_tao_amount"] = action.get("tao_spent", 0.0)
        if side == "sell" and "requested_alpha_amount" not in action:
            action["requested_alpha_amount"] = action.get("alpha_qty", 0.0)

        # Stamp intended tolerance regardless of outcome (audit trail).
        _, intended_tol = self.check_rate_tolerance(action)
        action["intended_slippage_tolerance_pct"] = round(intended_tol, 4)

        # Layer 5 — latency drawn even for failed orders.
        action["latency_ms"] = self.draw_latency_ms(side)

        # Layer 1 — Bernoulli non-slippage failure.
        failure_rate = (
            self.config.buy_failure_rate if side == "buy"
            else self.config.sell_failure_rate
        )
        if failure_rate > 0 and self._rng.random() < failure_rate:
            action["status"] = "failed"
            action["failure_reason"] = "random_reject"
            return action

        # Layer 2 — CSV-only slippage noise (skipped if live fetch succeeded).
        self.apply_slippage_noise(action)

        # Layer 3 — rate-tolerance check (sees post-noise slippage).
        breach, _ = self.check_rate_tolerance(action)
        if breach:
            action["status"] = "failed"
            action["failure_reason"] = "rate_tolerance"
            return action

        # Layer 4 — partial fill (default rate 0 = no-op).
        self.maybe_partial_fill(action)

        action.setdefault("status", "executed")
        return action

    # ── Individual layers (exposed for unit tests + advanced callers) ──

    def draw_latency_ms(self, side: str) -> int:
        """Draw Gaussian latency for a side, clamped ≥ 0, returned in ms."""
        if side == "buy":
            mean = self.config.buy_latency_mean_s
            std = self.config.buy_latency_std_s
        else:
            mean = self.config.sell_latency_mean_s
            std = self.config.sell_latency_std_s
        draw_s = self._rng.gauss(mean, std)
        return max(0, int(round(draw_s * 1000)))

    def apply_slippage_noise(self, action: dict) -> None:
        """Layer 2 — CSV-only Gaussian noise. No-op when ``live_spot_price``
        is set (the live-fetch path already captured real drift).
        Updates ``price``/``alpha_qty``/``tao_received``/``exit_price`` and
        sets ``exec_slippage_pct`` so layer 3 sees the realised slippage."""
        if "live_spot_price" in action:
            return
        mean = self.config.slippage_noise_mean_pct / 100.0
        std = self.config.slippage_noise_std_pct / 100.0
        floor = self.config.slippage_noise_floor_pct / 100.0
        noise = max(self._rng.gauss(mean, std), floor)
        if noise == 0:
            return
        side = action.get("type") or action.get("side")
        if side == "buy":
            old_price = action.get("price", 0.0)
            new_price = old_price * (1 + noise)
            if new_price > 0 and old_price > 0:
                action["alpha_qty"] = action.get("alpha_qty", 0.0) * (old_price / new_price)
            action["price"] = new_price
        else:  # sell — worse fill = less TAO for the same alpha
            scale = 1 / (1 + noise) if (1 + noise) > 0 else 1.0
            action["tao_received"] = action.get("tao_received", 0.0) * scale
            if action.get("exit_price"):
                action["exit_price"] = action["exit_price"] * scale
        action["exec_slippage_pct"] = round(noise * 100, 4)

    def check_rate_tolerance(self, action: dict) -> tuple[bool, float]:
        """Layer 3 — return ``(breach, intended_tolerance_pct)``.

        Buys breach when post-noise exec_slippage_pct exceeds
        ``base_slippage + buy_pp``, where base is the AMM's natural
        impact at the requested size against the decision pool.
        Sells breach when exec_slippage_pct exceeds the absolute sell
        tolerance percent (no AMM-impact base on sells in this impl)."""
        exec_slip = action.get("exec_slippage_pct", 0.0) or 0.0
        side = action.get("type") or action.get("side")
        if side == "buy":
            decision_pool_tao = action.get("decision_pool_tao") or 0.0
            decision_tao_spent = action.get(
                "requested_tao_amount",
                action.get("tao_spent", 0.0),
            )
            if decision_pool_tao > 0 and decision_tao_spent > 0:
                base = slippage_pct(decision_tao_spent, decision_pool_tao) * 100
            else:
                base = 0.0
            intended = base + self.config.rate_tolerance_buy_pp
        else:
            intended = self.config.rate_tolerance_sell_pct
        return exec_slip > intended, intended

    def maybe_partial_fill(self, action: dict) -> None:
        """Layer 4 — Bernoulli partial fill. Default rate=0 (no-op).
        Scales executed amounts by ``uniform(0.5, 1.0)``; partial_fill_ratio
        is derivable downstream from ``executed/requested``."""
        if self.config.partial_fill_rate <= 0:
            return
        if self._rng.random() >= self.config.partial_fill_rate:
            return
        frac = self._rng.uniform(0.5, 1.0)
        action["status"] = "partial"
        side = action.get("type") or action.get("side")
        if side == "buy":
            action.setdefault("requested_tao_amount", action.get("tao_spent", 0.0))
            action["tao_spent"] = action.get("tao_spent", 0.0) * frac
            action["alpha_qty"] = action.get("alpha_qty", 0.0) * frac
        else:
            action.setdefault("requested_alpha_amount", action.get("alpha_qty", 0.0))
            action["alpha_qty"] = action.get("alpha_qty", 0.0) * frac
            action["tao_received"] = action.get("tao_received", 0.0) * frac
