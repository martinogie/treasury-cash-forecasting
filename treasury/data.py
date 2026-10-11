"""Synthetic daily cash flows for a card-payments business.

Money comes in when card networks settle (T+1, net of interchange). Money goes out as merchant
payouts (60% of merchants paid daily, 40% in a weekly Friday batch), chargebacks, biweekly
payroll, vendor runs on the 15th and month end, quarterly estimated taxes and a few
unplanned payments a year. Weekends and bank holidays shift and pile up the flows.
"""
import numpy as np
import pandas as pd

from .calendar import (batch_days, business_days, first_bday_on_or_after, holidays,
                       last_bday_on_or_before, settlement_index)

START = "2023-01-03"
END = "2026-10-09"          # last day of actuals (a Friday)
PLAN_END = "2027-01-31"     # the treasury calendar is known further ahead
OPENING_BALANCE = 120e6
RECEIPT_RATE = 0.984        # settlement net of interchange
PAYOUT_RATE = 0.970         # merchants receive volume less fees
DAILY_SHARE = 0.60          # share of merchants on daily payout
CHARGEBACK_RATE = 0.002


def card_volume(cal, rng):
    t = (cal - cal[0]).days.values / 365.25
    dow = np.array([0.95, 0.97, 1.00, 1.03, 1.15, 1.10, 0.85])[cal.dayofweek]
    doy, m, d = cal.dayofyear.values, cal.month.values, cal.day.values
    season = 1 + 0.04 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    peak = ((m == 11) & (d >= 20)) | ((m == 12) & (d <= 24))
    season *= np.where(peak, 1.18, 1.0) * np.where(m <= 2, 0.95, 1.0)
    payday = np.where(np.isin(d, [1, 2, 3, 15, 16, 17]), 1.07, 1.0)
    hol = np.where(cal.isin(holidays(cal[0], cal[-1])), 0.80, 1.0)
    drift = np.zeros(len(cal))
    shocks = rng.normal(0, 0.012, len(cal))
    for i in range(1, len(cal)):
        drift[i] = 0.97 * drift[i - 1] + shocks[i]
    noise = rng.normal(0, 0.04, len(cal))
    return 14e6 * np.exp(0.14 * t) * dow * season * payday * hol * np.exp(drift + noise)


def treasury_plan(bdays):
    """Scheduled items the treasury team knows in advance (payroll, vendors, tax)."""
    plan = pd.DataFrame({"date": bdays, "payroll_plan": 0.0, "vendor_plan": 0.0, "tax_plan": 0.0})
    plan = plan.set_index("date")
    years = lambda day: day.year - 2023 + (day.month - 1) / 12

    for friday in pd.date_range("2023-01-06", bdays[-1], freq="14D"):
        day = last_bday_on_or_before(friday, bdays)
        plan.loc[day, "payroll_plan"] = round(1.10e6 * 1.08 ** (day.year - 2023), -3)

    for month in pd.period_range(bdays[0], bdays[-1], freq="M"):
        mid = last_bday_on_or_before(month.start_time + pd.Timedelta(days=14), bdays)
        end = last_bday_on_or_before(month.end_time.normalize(), bdays)
        for day in (mid, end):
            plan.loc[day, "vendor_plan"] += round(0.60e6 * 1.07 ** years(day), -3)

    for year in range(bdays[0].year, bdays[-1].year + 1):
        for md in ("01-15", "04-15", "06-15", "09-15"):
            due = pd.Timestamp(f"{year}-{md}")
            if bdays[0] <= due <= bdays[-1]:
                day = first_bday_on_or_after(due, bdays)
                plan.loc[day, "tax_plan"] = round(3.2e6 * 1.10 ** (year - 2023), -3)

    plan["is_batch_day"] = plan.index.isin(batch_days(bdays))
    return plan.reset_index()


def settle(volume, cal, bdays):
    """Gross card volume settling on each business day."""
    idx = settlement_index(cal, bdays)
    ok = idx < len(bdays)
    return np.bincount(idx[ok], weights=volume[ok], minlength=len(bdays))


def merchant_payouts(settled, is_batch):
    """Daily merchants are paid the business day after settlement; weekly merchants in a batch
    covering everything settled since the previous batch day."""
    daily = np.zeros_like(settled)
    daily[1:] = DAILY_SHARE * PAYOUT_RATE * settled[:-1]
    batch = np.zeros_like(settled)
    acc = 0.0
    for i in range(len(settled)):
        if is_batch[i]:
            batch[i] = (1 - DAILY_SHARE) * PAYOUT_RATE * acc
            acc = 0.0
        acc += settled[i]
    return daily, batch


def generate(seed=7, start=START, end=END, plan_end=PLAN_END):
    rng = np.random.default_rng(seed)
    cal = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=7), end).as_unit("ns")
    vol = card_volume(cal, rng)
    all_bdays = business_days(cal[0], plan_end)
    plan = treasury_plan(all_bdays)

    settled_all = settle(vol, cal, all_bdays)
    keep = (all_bdays >= pd.Timestamp(start)) & (all_bdays <= pd.Timestamp(end))
    bdays = all_bdays[keep]
    settled = settled_all[keep]
    p = plan.set_index("date").loc[bdays]
    n = len(bdays)

    daily, batch = merchant_payouts(settled, p["is_batch_day"].values)
    chargebacks = np.zeros(n)
    chargebacks[3:] = CHARGEBACK_RATE * settled[:-3] * rng.lognormal(-0.03, 0.25, n - 3)
    vendors = p["vendor_plan"].values * rng.normal(1.0, 0.12, n)
    first_of_month = ~bdays.to_period("M").duplicated()
    interest = np.where(first_of_month, 0.38e6 * rng.normal(1, 0.05, n), 0.0)
    adhoc = np.where(rng.random(n) < 4 / 252, rng.uniform(2e6, 7e6, n), 0.0)

    df = pd.DataFrame({
        "date": bdays,
        "settled_volume": settled,
        "receipts": RECEIPT_RATE * settled,
        "interest": interest,
        "payout_daily": daily,
        "payout_batch": batch,
        "chargebacks": chargebacks,
        "payroll": p["payroll_plan"].values,
        "vendors": vendors,
        "tax": p["tax_plan"].values,
        "unplanned": adhoc,
    })
    df["inflows"] = df["receipts"] + df["interest"]
    df["outflows"] = df[["payout_daily", "payout_batch", "chargebacks", "payroll", "vendors",
                         "tax", "unplanned"]].sum(axis=1)
    df = df.iloc[5:].copy()   # drop the warm-up week (payout cycle not yet running)
    df["net"] = df["inflows"] - df["outflows"]
    df["balance"] = OPENING_BALANCE + df["net"].cumsum()

    volume = pd.DataFrame({"date": cal, "volume": vol})
    volume = volume[volume["date"] <= pd.Timestamp(end)].reset_index(drop=True)
    return volume, df.reset_index(drop=True), plan, all_bdays
