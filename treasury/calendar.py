"""Banking calendar: business days, settlement mapping and calendar features."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def holidays(start, end):
    return pd.DatetimeIndex(USFederalHolidayCalendar().holidays(start, end)).as_unit("ns")


def business_days(start, end):
    """Weekdays that are not US federal (Fed banking) holidays."""
    days = pd.bdate_range(start, end).as_unit("ns")
    return days[~days.isin(holidays(start, end))]


def settlement_index(cal_days, bdays):
    """Card volume on calendar day t settles on the first business day after t (T+1).

    Returns, for each calendar day, the position in `bdays` it settles on.
    Weekend and holiday volume therefore piles onto the next business day.
    """
    return np.searchsorted(bdays.values, pd.DatetimeIndex(cal_days).as_unit("ns").values, side="right")


def last_bday_on_or_before(day, bdays):
    i = np.searchsorted(bdays.values, np.datetime64(pd.Timestamp(day).as_unit("ns")), side="right") - 1
    return bdays[i]


def first_bday_on_or_after(day, bdays):
    i = np.searchsorted(bdays.values, np.datetime64(pd.Timestamp(day).as_unit("ns")), side="left")
    return bdays[i]


def batch_days(bdays):
    """Weekly merchant payouts run on the last business day of each week (normally Friday)."""
    s = pd.Series(bdays, index=bdays)
    week = bdays.to_period("W-FRI")
    return pd.DatetimeIndex(s.groupby(week).max().values).as_unit("ns")


def bday_features(dates, all_bdays, plan):
    """Calendar features for business days. Everything here is known in advance."""
    dates = pd.DatetimeIndex(dates).as_unit("ns")
    pos = np.searchsorted(all_bdays.values, dates.values)
    prev = all_bdays[np.maximum(pos - 1, 0)]
    gap = (dates - prev).days.values
    month_key = all_bdays.to_period("M")
    bdom = pd.Series(1, index=all_bdays).groupby(month_key).cumsum().values
    n_in_month = pd.Series(1, index=all_bdays).groupby(month_key).transform("sum").values
    p = plan.set_index("date").reindex(dates)
    return pd.DataFrame({
        "dow": dates.dayofweek,
        "gap_days": gap,
        "bday_of_month": bdom[pos],
        "bdays_to_month_end": n_in_month[pos] - bdom[pos],
        "month": dates.month,
        "day_of_month": dates.day,
        "is_batch_day": p["is_batch_day"].values.astype(int),
        "is_payroll_day": (p["payroll_plan"].values > 0).astype(int),
        "is_vendor_day": (p["vendor_plan"].values > 0).astype(int),
        "is_tax_day": (p["tax_plan"].values > 0).astype(int),
    }, index=dates)


def cal_features(dates):
    """Calendar-day features for card volume."""
    dates = pd.DatetimeIndex(dates).as_unit("ns")
    hol = holidays(dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=1))
    return pd.DataFrame({
        "dow": dates.dayofweek,
        "day_of_month": dates.day,
        "month": dates.month,
        "day_of_year": dates.dayofyear,
        "is_holiday": dates.isin(hol).astype(int),
    }, index=dates)
