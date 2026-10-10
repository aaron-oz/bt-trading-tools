"""yield_model=None resolves through an environment-dependent cascade; the
engine must say which tier it got (once per configuration per process)."""

import warnings

import pytest

import bt_trading_tools.backtest.engine as engine_mod
from bt_trading_tools.alpha_yield import (
    TIER_EMPIRICAL, TIER_TAOSTATS_LIVE, TIER_VALIDATOR_CACHE, TIER_ZERO,
    describe_default_yield_cascade)
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.alpha_yield import ZeroYieldProvider, AlphaYieldModel

ENV = ("VALIDATOR_CACHE_PATH", "TAOSTATS_API_KEY", "BT_NETWORK", "TAOSTATS_DATA_DIR")


@pytest.fixture
def bare_env(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    # pin the validator cache off so a developer's ~/.validator_selection is ignored
    monkeypatch.setenv("VALIDATOR_CACHE_PATH", str(tmp_path / "nonexistent.json"))
    monkeypatch.setattr(engine_mod, "_CASCADE_WARNED", set())
    return monkeypatch


def test_zero_tier_when_nothing_configured(bare_env):
    d = describe_default_yield_cascade()
    assert d["tiers"] == [TIER_ZERO] and d["primary"] == TIER_ZERO
    assert d["not_point_in_time"] is False


def test_tier_order_follows_cascade(bare_env, tmp_path):
    cache = tmp_path / "best_validators.json"
    cache.write_text("{}")
    bare_env.setenv("VALIDATOR_CACHE_PATH", str(cache))
    bare_env.setenv("TAOSTATS_API_KEY", "x")
    bare_env.setenv("TAOSTATS_DATA_DIR", str(tmp_path))
    d = describe_default_yield_cascade()
    assert d["tiers"] == [TIER_VALIDATOR_CACHE, TIER_TAOSTATS_LIVE, TIER_EMPIRICAL, TIER_ZERO]
    assert d["primary"] == TIER_VALIDATOR_CACHE
    assert d["validator_cache_path"] == str(cache)
    assert d["not_point_in_time"] is True


def test_engine_warns_once_naming_tier(bare_env, tmp_path):
    bare_env.setenv("TAOSTATS_DATA_DIR", str(tmp_path))
    with pytest.warns(UserWarning, match="primary tier 'empirical'.*lookahead"):
        e = BacktestEngine(capital=1.0)
    assert e.default_yield_cascade["primary"] == TIER_EMPIRICAL
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BacktestEngine(capital=1.0)          # same configuration: silent


def test_engine_warns_again_when_configuration_changes(bare_env, tmp_path):
    with pytest.warns(UserWarning, match="primary tier 'zero'.*ZERO"):
        BacktestEngine(capital=1.0)
    bare_env.setenv("TAOSTATS_API_KEY", "x")
    with pytest.warns(UserWarning, match="taostats_live"):
        BacktestEngine(capital=1.0)


def test_explicit_yield_model_is_silent(bare_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        e = BacktestEngine(capital=1.0, yield_model=AlphaYieldModel(ZeroYieldProvider()))
    assert e.default_yield_cascade is None
