"""Suite-wide: keep tests hermetic with respect to the engine's default yield.

Since 2026-10-14 ``BacktestEngine(yield_model=None)`` uses the realized
on-chain table if one exists on the machine (``realized_yield.
default_table_path``). Tests must not depend on a developer's local data, so
the default is disabled here; tests of the default set
``BT_ONCHAIN_YIELD_TABLE`` themselves.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_local_realized_table(monkeypatch):
    monkeypatch.setenv("BT_ONCHAIN_YIELD_TABLE", "off")
    yield
