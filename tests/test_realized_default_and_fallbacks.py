"""2026-10-14 rulings: no take cap by default; realized on-chain yield is the
engine default; every fallback is counted and announced once per process."""
import warnings

import pandas as pd
import pytest

import bt_trading_tools.backtest.engine as engine_mod
import bt_trading_tools.realized_yield as ry
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.realized_yield import RealizedYieldModel, ValidatorRule

from test_realized_yield import DAY, LAG, T0, _BuyOnce, _ticks, model, table


@pytest.fixture(autouse=True)
def _fresh_warn_state(monkeypatch):
    monkeypatch.setattr(ry, "_FALLBACK_WARNED", set())
    monkeypatch.setattr(ry, "_DEFAULT_MODELS", {})
    monkeypatch.setattr(engine_mod, "_CASCADE_WARNED", set())


# ------------------------------------------------------------ no take cap
def test_default_rule_has_no_take_filter():
    assert ValidatorRule().max_take is None
    df = table(takes={"B": 0.18})                 # B: best rate, highest take
    T = T0 + 5 * DAY + LAG
    assert model(df).choose_hotkey(1, T) == "B"
    capped = model(df, rule=ValidatorRule(window_days=3, min_valid_days=2, max_take=0.12))
    assert capped.choose_hotkey(1, T) == "C"      # opt-in cap still works


def test_unknown_take_passes_without_cap():
    df = table()
    df["take"] = float("nan")
    m = model(df, rule=ValidatorRule(window_days=3, min_valid_days=2, unknown_take_passes=False))
    assert m.choose_hotkey(1, T0 + 5 * DAY + LAG) == "B"


# ------------------------------------------------------------ fallbacks
def test_validator_row_credits_without_fallback():
    m = model(table())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m.accrued_yield(1, 10.0, T0 + DAY, T0 + 4 * DAY, hotkey="A", as_of=T0 + 6 * DAY)
    assert sum(m.fallback_counts.values()) == 0


def test_pooled_for_validator_warns_once_and_counts():
    m = model(table(drop={("A", 2), ("A", 3)}))
    with pytest.warns(UserWarning, match=r"FALLBACK \(pooled_for_validator\)"):
        m.accrued_yield(1, 10.0, T0 + DAY, T0 + 5 * DAY, hotkey="A", as_of=T0 + 7 * DAY)
    assert m.fallback_counts["pooled_for_validator"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # second time: silent
        m.accrued_yield(1, 10.0, T0 + DAY, T0 + 5 * DAY, hotkey="A", as_of=T0 + 7 * DAY)
    assert m.fallback_counts["pooled_for_validator"] == 4


def test_pooled_no_validator_and_pooled_mode():
    m = model(table())
    with pytest.warns(UserWarning, match="pooled_no_validator"):
        m.accrued_yield(1, 10.0, T0 + DAY, T0 + 3 * DAY, hotkey=None, as_of=T0 + 6 * DAY)
    assert m.fallback_counts["pooled_no_validator"] == 2
    p = model(table(), mode="pooled")              # deliberate, not a fallback
    p.accrued_yield(1, 10.0, T0 + DAY, T0 + 3 * DAY, hotkey="A", as_of=T0 + 6 * DAY)
    assert p.fallback_counts["pooled_no_validator"] == 0
    assert p.fallback_counts["pooled_for_validator"] == 0


def test_no_coverage_warns_and_credits_zero():
    m = model(table())
    with pytest.warns(UserWarning, match=r"no_coverage.*ZERO"):
        a = m.accrued_yield(99, 10.0, T0 + DAY, T0 + 3 * DAY, hotkey="A", as_of=T0 + 6 * DAY)
    assert a == 0.0 and m.fallback_counts["no_coverage"] == 2


def test_stale_table_warns():
    m = model(table(n_days=5))
    with pytest.warns(UserWarning, match="stale_table"):
        m.accrued_yield(1, 10.0, T0 + 8 * DAY, T0 + 12 * DAY, hotkey="A", as_of=T0 + 12 * DAY)
    assert m.fallback_counts["stale_table"] > 0


# ------------------------------------------------------------ engine default
def _write(tmp_path, df):
    p = tmp_path / "onchain_yield_daily.parquet"
    df.to_parquet(p)
    return p


def test_engine_default_is_realized_model(tmp_path, monkeypatch):
    p = _write(tmp_path, table())
    monkeypatch.setenv("BT_ONCHAIN_YIELD_TABLE", str(p))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        eng = BacktestEngine(capital=10.0, realism_rng_seed=1)
    assert isinstance(eng.yield_model, RealizedYieldModel)
    assert eng.default_yield_source["source"] == "onchain_realized"
    assert eng.default_yield_source["path"] == str(p)
    assert eng.yield_model.rule.max_take is None
    # one model per process: the second engine shares it
    assert BacktestEngine(capital=1.0).yield_model is eng.yield_model


def test_engine_default_credits_rule_validator(tmp_path, monkeypatch):
    # default rule: 7-day window, >= 5 valid days; B has the best rate
    p = _write(tmp_path, table(n_days=12))
    monkeypatch.setenv("BT_ONCHAIN_YIELD_TABLE", str(p))
    eng = BacktestEngine(capital=10.0, realism_rng_seed=1)
    strat = _BuyOnce(None)
    eng.run(_ticks(48, T0 + 8 * DAY), strat)
    assert strat.seen[1].validator_hotkey == "B"


def test_engine_falls_back_loudly_without_table(tmp_path, monkeypatch):
    monkeypatch.setenv("BT_ONCHAIN_YIELD_TABLE", str(tmp_path / "missing.parquet"))
    monkeypatch.setenv("VALIDATOR_CACHE_PATH", str(tmp_path / "nonexistent.json"))
    for k in ("TAOSTATS_API_KEY", "BT_NETWORK", "TAOSTATS_DATA_DIR"):
        monkeypatch.delenv(k, raising=False)
    with pytest.warns(UserWarning, match="FALLBACK: yield_model=None and no realized"):
        eng = BacktestEngine(capital=1.0)
    assert not isinstance(eng.yield_model, RealizedYieldModel)
    assert eng.default_yield_source["source"] == "cascade"


def test_default_table_search_order(tmp_path, monkeypatch):
    monkeypatch.delenv("BT_ONCHAIN_YIELD_TABLE", raising=False)
    monkeypatch.delenv("ONCHAIN_YIELD_TABLE", raising=False)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    b.write_bytes(b"x")
    monkeypatch.setattr(ry, "DEFAULT_TABLE_CANDIDATES", ("/root/no/such.parquet", str(a), str(b)))
    path, searched = ry.default_table_path()
    assert path == b and searched == ["/root/no/such.parquet", str(a), str(b)]
    monkeypatch.setenv("ONCHAIN_YIELD_TABLE", str(a))       # explicit wins, even if missing
    assert ry.default_table_path()[0] is None
    monkeypatch.setenv("BT_ONCHAIN_YIELD_TABLE", "off")
    assert ry.default_table_path() == (None, ["BT_ONCHAIN_YIELD_TABLE='off' (disabled)"])
