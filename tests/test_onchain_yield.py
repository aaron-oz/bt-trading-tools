"""On-chain daily yield table -> HistoricalYieldModel rate table."""

import unittest

import pandas as pd

from bt_trading_tools.alpha_yield_history import (
    DAY_S, HistoricalYieldModel, build_rate_table, select_validators)
from bt_trading_tools.onchain_yield import (
    build_onchain_rate_table, hotkey_rate_table, onchain_selection)

D = "2026-07-01"
D0 = int(pd.Timestamp(D, tz="UTC").timestamp())


def drow(hk, rate, alpha, netuid=1, date=D, valid=True, permit=True, share=0.3):
    return {"date": pd.Timestamp(date, tz="UTC"), "netuid": netuid, "hotkey": hk,
            "rate_per_day": rate, "total_hotkey_alpha": alpha, "valid": valid,
            "permit": permit, "stake_share": share}


class TestOnchainRateTable(unittest.TestCase):
    def test_stake_weighted_over_given_set(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("B", 0.001, 300), drow("C", 0.009, 50)])
        sets = pd.DataFrame([{"netuid": 1, "date": D, "hotkey": h} for h in ("A", "B")])
        t = build_onchain_rate_table(daily, sets)
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], (100 * 0.002 + 300 * 0.001) / 400, places=15)
        self.assertEqual(t.iloc[0]["n_validators"], 2)
        self.assertEqual(t.iloc[0]["selection"], "given")

    def test_invalid_rows_are_left_out(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("B", 5.0, 300, valid=False)])
        sets = pd.DataFrame([{"netuid": 1, "date": D, "hotkey": h} for h in ("A", "B")])
        t = build_onchain_rate_table(daily, sets)
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], 0.002, places=15)
        self.assertEqual(t.iloc[0]["n_selected"], 2)

    def test_onchain_rule_top_k_by_stake_permit_and_share(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("B", 0.001, 300),
                              drow("C", 0.009, 500, permit=False),
                              drow("E", 0.009, 1, share=0.001)])
        sel = onchain_selection(daily, top_k=1)
        self.assertEqual(list(sel["hotkey"]), ["B"])
        t = build_onchain_rate_table(daily, top_k=2)
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], (100 * 0.002 + 300 * 0.001) / 400, places=15)
        self.assertEqual(t.iloc[0]["selection"], "onchain")

    def test_fallback_only_where_no_set(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("B", 0.004, 100),
                              drow("A", 0.003, 100, netuid=2)])
        sets = pd.DataFrame([{"netuid": 1, "date": D, "hotkey": "A"}])
        t = build_onchain_rate_table(daily, sets).set_index("netuid")
        self.assertAlmostEqual(t.loc[1, "rate_per_day"], 0.002, places=15)   # given set, not pooled with B
        self.assertEqual(t.loc[2, "selection"], "onchain")
        t2 = build_onchain_rate_table(daily, sets, fallback_onchain=False)
        self.assertEqual(list(t2["netuid"]), [1])

    def test_plugs_into_historical_yield_model(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("A", 0.004, 100, date="2026-07-02")])
        t = build_onchain_rate_table(daily)
        m = HistoricalYieldModel(t, convention="integrated")
        got = m.accrued_yield(1, 10.0, D0, D0 + 2 * DAY_S)
        self.assertAlmostEqual(got, 10.0 * (0.002 + 0.004), places=12)

    def test_hotkey_rate_table(self):
        daily = pd.DataFrame([drow("A", 0.002, 100), drow("B", 0.004, 100),
                              drow("A", 0.007, 100, valid=False, date="2026-07-02")])
        t = hotkey_rate_table(daily, "A")
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], 0.002)


class TestSelectValidatorsAndRateColumn(unittest.TestCase):
    def hist(self):
        r = lambda hk, st, a30, a7: {"date": D, "netuid": 1, "hotkey": hk, "stake_rao": st,
                                    "thirty_day_apy": a30, "seven_day_apy": a7,
                                    "thirty_day_epoch_participation": 1.0, "timestamp": f"{D}T12:00:00Z"}
        return pd.DataFrame([r("A", 100, 0.5, 0.2), r("B", 300, 0.1, 0.3), r("C", 100, 0.05, 0.9)])

    def test_selection_matches_rate_table(self):
        s = select_validators(self.hist(), top_k=2)
        self.assertEqual(sorted(s["hotkey"]), ["A", "B"])

    def test_seven_day_rate_column_uses_thirty_day_selection(self):
        t = build_rate_table(self.hist(), top_k=2, rate_column="seven_day_apy")
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], (100 * 0.2 + 300 * 0.3) / 400 / 365.0, places=12)


if __name__ == "__main__":
    unittest.main()
