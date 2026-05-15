"""
Default backtest harness factory — yield, fees, and execution realism
wired in with sensible defaults so individual bot backtests don't have
to reinvent the configuration.

This module is the canonical entry point for **research backtests**.
Production paper bots construct their engines via ``PaperBotBase``,
which already wires in ``build_default_yield_model()`` + a calibrated
``FeeModel`` + a calibrated ``RealismConfig``. Backtests should use the
factories here to stay aligned with that paper-bot configuration so
backtest-vs-paper drift is minimized.

Quick start:

    from bt_trading_tools.harness import make_research_engine

    engine = make_research_engine(capital=10.0)
    result = engine.run(ticks, strategy)

Per-bot calibrated overrides:

    from bt_trading_tools.harness import make_research_engine
    from bt_trading_tools.fees import FeeModel
    from bt_trading_tools.execution import RealismConfig

    AUTOBOT_FEE_MODEL = FeeModel(
        fallback_gas_tao=2.58e-4,   # 4.5× scale-down for 0.17 TAO trades
        fallback_proxy_tao=4.2e-5,
    )
    AUTOBOT_REALISM = RealismConfig(
        buy_failure_rate=0.0397,
        sell_failure_rate=0.678,
        slippage_noise_mean_pct=0.32,
        slippage_noise_std_pct=4.0,
    )
    engine = make_research_engine(
        capital=10.0,
        fee_model=AUTOBOT_FEE_MODEL,
        realism_config=AUTOBOT_REALISM,
    )

What this fixes (the "ZeroYieldProvider" silent under-statement):

    Production paper bots have VALIDATOR_CACHE_PATH / TAOSTATS_API_KEY /
    BT_NETWORK / TAOSTATS_DATA_DIR set in their systemd units, so
    ``build_default_yield_model()`` returns a working cascade.

    Research backtests run on dev machines with none of those env vars
    set. The default cascade silently falls through to ZeroYieldProvider,
    so any strategy that holds positions across multiple ticks
    under-reports realized return by the yield component (~0.04-0.1 TAO
    per closed position for autobot-sized trades). Cross-regime
    comparisons get a regime-dependent bias: P1-heavy windows where
    positions cycle fast are barely affected; P2-heavy windows where
    positions hold for days are systematically under-stated.

    ``make_research_yield_model()`` below picks the validator-selection
    cache from a list of common dev / research paths, and disables the
    paper-bot-tuned 36-hour staleness gate (research snapshots are
    intentionally days/weeks/months stale relative to the backtest
    window). This gives backtests a working ValidatorCacheYieldProvider
    on any machine that has a snapshot at one of the standard paths.

    Cascade tier 2 (TaostatsYieldProvider) is omitted from the research
    cascade because backtests should not make live API calls.

See ``docs/realistic_backtesting_guide.md`` in the alpha-trading repo
for the full set of conventions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from bt_trading_tools.alpha_yield import (
    AlphaYieldModel,
    CascadingYieldProvider,
    EmpiricalYieldProvider,
    ValidatorCacheYieldProvider,
    _resolve_validator_cache_path,
)
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.execution import RealismConfig
from bt_trading_tools.fees import FeeModel

# Days/weeks old validator-cache snapshots are normal for research backtests
# (the backtest window is in the past). 365 days is "essentially unlimited"
# while still rejecting truly broken / never-generated files.
RESEARCH_VALIDATOR_CACHE_MAX_AGE_S: float = 365 * 86400


def make_research_yield_model(
    validator_cache_path: "str | Path | None" = None,
    validator_cache_max_age_s: float = RESEARCH_VALIDATOR_CACHE_MAX_AGE_S,
    taostats_data_dir: "str | Path | None" = None,
) -> AlphaYieldModel:
    """Construct a yield model suited to research backtests.

    Differs from ``build_default_yield_model`` in three ways:
      1. Validator-cache staleness gate is set to ~1 year (vs 36h prod),
         so historical snapshots used for backtest replay aren't rejected.
      2. Live providers (TaostatsYieldProvider, ChainYieldProvider) are
         excluded — backtests should not make network calls.
      3. ``EmpiricalYieldProvider`` is included if ``taostats_data_dir``
         is given OR ``TAOSTATS_DATA_DIR`` env is set; otherwise omitted
         (its docstring formula is known buggy per autobot v2_calibration
         — leave for explicit opt-in).

    The cascade is:
      ValidatorCacheYieldProvider (long-stale tolerance) → EmpiricalYieldProvider
      (if data_dir given) → ZeroYieldProvider built-in

    Args:
        validator_cache_path: explicit override for the cache file path.
            If None, searches the same fallback list as build_default
            (VALIDATOR_CACHE_PATH env → /root/.validator_selection/...
            → ~/.validator_selection/... → /tmp/autobot_live_data/...).
        validator_cache_max_age_s: staleness limit. Defaults to ~1 year
            (research snapshots are intentionally stale relative to the
            backtest window — staleness is a feature, not a bug, here).
        taostats_data_dir: opt-in EmpiricalYieldProvider data dir.
            Default None (provider omitted). Pass a path to enable.

    Returns:
        ``AlphaYieldModel`` whose cascade is research-appropriate. If no
        validator cache is found AND no taostats data dir given, the
        cascade is empty and falls through to ZeroYieldProvider (with
        explicit emit on first lookup so the caller can see it).
    """
    import os

    providers: list = []

    # Validator cache: resolve path, attach with long staleness budget
    if validator_cache_path is not None:
        cache_path = Path(validator_cache_path)
        if cache_path.exists():
            providers.append(ValidatorCacheYieldProvider(
                cache_path=str(cache_path),
                max_age_s=validator_cache_max_age_s,
            ))
    else:
        resolved = _resolve_validator_cache_path()
        if resolved is not None:
            providers.append(ValidatorCacheYieldProvider(
                cache_path=str(resolved),
                max_age_s=validator_cache_max_age_s,
            ))

    # Empirical CSV (opt-in only — known buggy formula)
    data_dir = (
        str(taostats_data_dir)
        if taostats_data_dir is not None
        else os.environ.get("TAOSTATS_DATA_DIR")
    )
    if data_dir and Path(data_dir).exists():
        providers.append(EmpiricalYieldProvider(data_dir=data_dir))

    return AlphaYieldModel(CascadingYieldProvider(providers))


def make_research_engine(
    capital: float,
    fee_model: Optional[FeeModel] = None,
    realism_config: Optional[RealismConfig] = None,
    yield_model: Optional[AlphaYieldModel] = None,
    realism_rng_seed: int = 0,
    **engine_kwargs,
) -> BacktestEngine:
    """Construct a ``BacktestEngine`` with sensible research-backtest defaults.

    Defaults (all overridable):
      - ``fee_model``: ``FeeModel()`` with calibrated fallback constants
        (gas 1.18 mTAO, proxy 0.19 mTAO; calibrated 2026-05-04 on
        yield-carry-sized trades). Bot-specific calibration (e.g.,
        autobot's 4.5× scale-down for 0.17 TAO trades) must be passed
        explicitly.
      - ``realism_config``: ``RealismConfig()`` engine defaults. Bot-
        specific overrides (e.g., autobot's 67.8% sell_failure_rate)
        must be passed explicitly.
      - ``yield_model``: ``make_research_yield_model()`` — picks up the
        validator-selection cache from standard local paths with long
        staleness tolerance. **No silent ZeroYieldProvider fall-through
        on machines that have a cache snapshot.**
      - ``realism_rng_seed``: 0 (reproducible by default).

    Any other ``BacktestEngine`` kwargs (e.g., ``ticks_per_year``,
    ``portfolio_log``) pass through.

    Args:
        capital: starting capital in TAO.
        fee_model: override the default FeeModel.
        realism_config: override the default RealismConfig.
        yield_model: override the default yield model.
        realism_rng_seed: random seed for realism noise (default 0).
        **engine_kwargs: forwarded to BacktestEngine.

    Returns:
        Configured ``BacktestEngine`` ready for ``.run(ticks, strategy)``.
    """
    return BacktestEngine(
        capital=capital,
        fee_model=fee_model or FeeModel(),
        realism_config=realism_config or RealismConfig(),
        yield_model=yield_model or make_research_yield_model(),
        realism_rng_seed=realism_rng_seed,
        **engine_kwargs,
    )
