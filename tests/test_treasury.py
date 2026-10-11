import unittest

import numpy as np
import pandas as pd

from treasury import backtest as B
from treasury.calendar import business_days, settlement_index
from treasury.data import RECEIPT_RATE, generate
from treasury.models import DriverBased, SeasonalNaive, holt_winters


class Data:
    volume, actuals, plan, all_bdays = generate(start="2023-01-03", end="2025-06-27")


class TestCalendar(unittest.TestCase):
    def test_business_days_skip_weekends_and_holidays(self):
        days = business_days("2025-12-20", "2026-01-05")
        self.assertNotIn(pd.Timestamp("2025-12-25"), days)
        self.assertNotIn(pd.Timestamp("2026-01-01"), days)
        self.assertTrue((days.dayofweek < 5).all())

    def test_weekend_volume_settles_monday(self):
        bdays = business_days("2025-03-01", "2025-03-31")
        cal = pd.date_range("2025-03-07", "2025-03-09")  # Fri, Sat, Sun
        settled_on = bdays[settlement_index(cal, bdays)]
        self.assertTrue((settled_on == pd.Timestamp("2025-03-10")).all())


class TestData(unittest.TestCase):
    def test_receipts_are_settled_volume_net_of_interchange(self):
        a = Data.actuals
        np.testing.assert_allclose(a["receipts"], RECEIPT_RATE * a["settled_volume"])

    def test_balance_rolls_forward(self):
        a = Data.actuals
        np.testing.assert_allclose(a["balance"].diff().iloc[1:], a["net"].iloc[1:], rtol=1e-9)

    def test_mondays_carry_weekend_volume(self):
        a = Data.actuals
        by_dow = a.groupby(a["date"].dt.dayofweek)["receipts"].mean()
        self.assertGreater(by_dow[0], 2 * by_dow[2])


class TestModels(unittest.TestCase):
    def test_holt_winters_learns_weekday_season(self):
        dow = np.tile(np.arange(5), 60)
        y = 100 * np.array([1.5, 0.9, 0.8, 0.9, 0.9])[dow]
        (level, trend, season), ape = holt_winters(y, dow, 0.2, 0.0, 0.2)
        self.assertLess(ape[-50:].mean(), 1e-6)
        self.assertAlmostEqual(season[0] / season[2], 1.5 / 0.8, places=4)

    def test_seasonal_naive_repeats_last_week(self):
        hist = Data.actuals
        future = B.horizon_days(hist["date"].iloc[-1], Data.all_bdays, weeks=1)
        fc = SeasonalNaive().forecast(Data.volume, hist, Data.plan, Data.all_bdays, future)
        last = hist.tail(5).set_index(hist.tail(5)["date"].dt.dayofweek)
        for d, v in zip(future, fc["inflows"]):
            self.assertEqual(v, last.loc[d.dayofweek, "inflows"])

    def test_forecast_uses_only_history(self):
        """Changing anything after the origin must not change the forecast."""
        o = pd.Timestamp("2025-03-28")
        hist = Data.actuals[Data.actuals["date"] <= o]
        vol = Data.volume[Data.volume["date"] <= o]
        tampered = Data.volume.copy()
        tampered.loc[tampered["date"] > o, "volume"] *= 3
        future = B.horizon_days(o, Data.all_bdays, weeks=4)
        a = DriverBased().forecast(vol, hist, Data.plan, Data.all_bdays, future)
        b = DriverBased().forecast(tampered, hist, Data.plan, Data.all_bdays, future)
        pd.testing.assert_frame_equal(a, b)

    def test_driver_model_beats_naive_on_balance(self):
        origins = [pd.Timestamp("2025-01-24"), pd.Timestamp("2025-03-21")]
        bt = B.run([SeasonalNaive, DriverBased], Data.volume, Data.actuals, Data.plan, Data.all_bdays, origins)
        s = B.score(bt).set_index("model")
        self.assertLess(s.loc["Driver-based", "week13_balance_mae"], s.loc["Seasonal naive", "week13_balance_mae"])
        self.assertLess(s.loc["Driver-based", "daily_mape_inflows"], 0.10)


if __name__ == "__main__":
    unittest.main()
