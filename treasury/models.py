"""Four ways to forecast daily inflows and outflows.

Every model gets only what was known at the forecast origin: history up to that day plus the
treasury calendar (business days, payroll, vendor and tax runs, payout batch days).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .calendar import bday_features, cal_features, settlement_index
from .data import DAILY_SHARE

TARGETS = ("inflows", "outflows")


class SeasonalNaive:
    """Same weekday last week. The usual spreadsheet starting point."""
    name = "Seasonal naive"

    def forecast(self, volume, hist, plan, all_bdays, future):
        last = hist.tail(10)
        out = {}
        for col in TARGETS:
            by_dow = last.groupby(last["date"].dt.dayofweek)[col].last()
            out[col] = [by_dow.get(d.dayofweek, last[col].mean()) for d in future]
        return pd.DataFrame(out, index=future)


def holt_winters(y, dow, alpha, beta, gamma, phi=0.98):
    """Damped-trend exponential smoothing with a multiplicative weekday season (indexed by
    weekday, so a bank holiday does not knock the season out of phase).
    Returns the final state and the one-step-ahead absolute percentage errors."""
    warm = 20
    level, trend = y[:warm].mean(), 0.0
    season = np.array([y[:warm][dow[:warm] == d].mean() / level for d in range(5)])
    ape = np.empty(len(y) - warm)
    for i in range(warm, len(y)):
        d = dow[i]
        s = season[d]
        ape[i - warm] = abs(y[i] - (level + phi * trend) * s) / y[i]
        new_level = alpha * y[i] / s + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        season[d] = gamma * y[i] / new_level + (1 - gamma) * s
        level = new_level
    return (level, trend, season), ape


class HoltWinters:
    """Exponential smoothing with a weekday season, tuned on the last year of history."""
    name = "Holt-Winters"
    grid = list(itertools.product([0.05, 0.1, 0.2, 0.35], [0.0, 0.02, 0.08], [0.05, 0.15, 0.3]))

    def forecast(self, volume, hist, plan, all_bdays, future, phi=0.98):
        out = {}
        self.params = {}
        dow = hist["date"].dt.dayofweek.values
        for col in TARGETS:
            y = hist[col].values
            best = min(self.grid, key=lambda p: holt_winters(y, dow, *p, phi=phi)[1][-250:].mean())
            (level, trend, season), _ = holt_winters(y, dow, *best, phi=phi)
            self.params[col] = best
            h = np.arange(1, len(future) + 1)
            damp = np.cumsum(phi ** h)
            out[col] = (level + damp * trend) * season[future.dayofweek]
        return pd.DataFrame(out, index=future)


class DirectRatioGBM:
    """Gradient boosting that predicts y(target day) / level(origin) from calendar features.

    One model covers every horizon: training rows are (origin, target-day) pairs taken from
    history, with the level measured as the trailing mean at the origin. This handles growth
    and lets the model learn weekday, month-end, holiday and payroll effects directly.
    """

    def __init__(self, level_window, horizon, origin_step, seed=0):
        self.level_window, self.horizon, self.origin_step, self.seed = level_window, horizon, origin_step, seed

    def _rows(self, feats, origin_pos, target_pos, level, origin_doy):
        X = feats.iloc[target_pos].reset_index(drop=True).copy()
        X["horizon"] = target_pos - origin_pos
        X["origin_day_of_year"] = origin_doy
        X["origin_level_trend"] = level[1]
        return X, level[0]

    def _level(self, y, pos):
        w = self.level_window
        recent = y[pos - w + 1: pos + 1].mean()
        older = y[pos - 2 * w + 1: pos - w + 1].mean()
        return recent, recent / older - 1

    def fit(self, dates, y, feats):
        y = np.asarray(y, float)
        Xs, ts = [], []
        first = 2 * self.level_window
        for o in range(first, len(y) - 1, self.origin_step):
            tgt = np.arange(o + 1, min(o + 1 + self.horizon, len(y)))
            lvl = self._level(y, o)
            X, base = self._rows(feats, o, tgt, lvl, dates[o].dayofyear)
            Xs.append(X)
            ts.append(np.log(y[tgt] / base))
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=self.seed)
        self.model.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ts))
        self.y, self.dates = y, dates
        return self

    def predict(self, future_feats):
        o = len(self.y) - 1
        lvl = self._level(self.y, o)
        X = future_feats.reset_index(drop=True).copy()
        X["horizon"] = np.arange(1, len(X) + 1)
        X["origin_day_of_year"] = self.dates[o].dayofyear
        X["origin_level_trend"] = lvl[1]
        return lvl[0] * np.exp(self.model.predict(X))


class GradientBoosting:
    """Direct gradient boosting on total daily inflows and outflows."""
    name = "Gradient boosting"

    def forecast(self, volume, hist, plan, all_bdays, future):
        dates = pd.DatetimeIndex(hist["date"])
        feats = bday_features(dates, all_bdays, plan)
        ffeats = bday_features(future, all_bdays, plan)
        out = {}
        for i, col in enumerate(TARGETS):
            m = DirectRatioGBM(level_window=20, horizon=len(future), origin_step=5, seed=i)
            out[col] = m.fit(dates, hist[col].values, feats).predict(ffeats)
        return pd.DataFrame(out, index=future)


class DriverBased:
    """Forecast the driver (card volume by calendar day) with gradient boosting, then build every
    cash line from it using the settlement rules and the treasury calendar:

    receipts      = receipt rate x volume settling that business day (T+1, weekends pile up)
    daily payouts = paid the business day after settlement
    batch payouts = weekly merchants paid on batch day for everything settled since the last batch
    chargebacks   = rate x volume settled three business days earlier
    payroll, tax  = scheduled amounts; vendors = planned run; interest = last month's amount
    """
    name = "Driver-based"

    def fit_rates(self, hist):
        recent = hist.tail(250)
        s = recent["settled_volume"].values
        self.receipt_rate = recent["receipts"].sum() / s.sum()
        self.payout_rate = (recent["payout_daily"].values[1:].sum() / s[:-1].sum()) / DAILY_SHARE
        self.chargeback_rate = recent["chargebacks"].values[3:].sum() / s[:-3].sum()
        self.interest = hist.loc[hist["interest"] > 0, "interest"].iloc[-1]

    def forecast(self, volume, hist, plan, all_bdays, future, return_components=False, volume_shock=0.0):
        """volume_shock scales forecast card volume (e.g. -0.2 = 20% drop) for scenario analysis."""
        self.fit_rates(hist)
        origin = hist["date"].iloc[-1]
        vol = volume[volume["date"] <= origin]
        vdates = pd.DatetimeIndex(vol["date"])

        # 1. forecast calendar-day volume from the day after the origin to the last day needed
        fut_cal = pd.date_range(origin + pd.Timedelta(days=1), future[-1] - pd.Timedelta(days=1)).as_unit("ns")
        gbm = DirectRatioGBM(level_window=28, horizon=len(fut_cal), origin_step=7)
        gbm.fit(vdates, vol["volume"].values, cal_features(vdates))
        vol_fc = gbm.predict(cal_features(fut_cal)) * (1 + volume_shock)

        # 2. settle it onto business days
        idx = settlement_index(fut_cal, future)
        settled = np.bincount(idx, weights=vol_fc, minlength=len(future))[:len(future)]
        # the origin day's own volume is already known and settles on the first future day
        settled[0] += vol["volume"].values[-1]

        # 3. build each cash line
        p = plan.set_index("date").loc[future]
        prev_settled = np.concatenate([[hist["settled_volume"].iloc[-1]], settled[:-1]])
        payout_daily = DAILY_SHARE * self.payout_rate * prev_settled
        acc = self._unbatched(hist, plan)
        payout_batch = np.zeros(len(future))
        for i, is_batch in enumerate(p["is_batch_day"].values):
            if is_batch:
                payout_batch[i] = (1 - DAILY_SHARE) * self.payout_rate * acc
                acc = 0.0
            acc += settled[i]
        lag3 = np.concatenate([hist["settled_volume"].values[-3:], settled])[:len(future)]
        first_of_month = ~future.to_period("M").isin(pd.DatetimeIndex([origin]).to_period("M")) & \
            ~future.to_period("M").duplicated()
        comp = pd.DataFrame({
            "receipts": self.receipt_rate * settled,
            "interest": np.where(first_of_month, self.interest, 0.0),
            "payout_daily": payout_daily,
            "payout_batch": payout_batch,
            "chargebacks": self.chargeback_rate * lag3,
            "payroll": p["payroll_plan"].values,
            "vendors": p["vendor_plan"].values,
            "tax": p["tax_plan"].values,
        }, index=future)
        out = pd.DataFrame({
            "inflows": comp["receipts"] + comp["interest"],
            "outflows": comp[["payout_daily", "payout_batch", "chargebacks", "payroll", "vendors", "tax"]].sum(axis=1),
        }, index=future)
        return (out, comp) if return_components else out

    @staticmethod
    def _unbatched(hist, plan):
        """Volume settled since (and including) the last batch day, owed to weekly merchants."""
        p = plan.set_index("date").loc[hist["date"]]
        last_batch = np.flatnonzero(p["is_batch_day"].values)[-1]
        return hist["settled_volume"].values[last_batch:].sum()


MODELS = [SeasonalNaive, HoltWinters, GradientBoosting, DriverBased]
