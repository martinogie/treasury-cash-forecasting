"""Produce a 13-week daily cash forecast with the driver-based model.

    python -m treasury.forecast --out forecast.csv                       # synthetic data
    python -m treasury.forecast --volume volume.csv --actuals actuals.csv --plan plan.csv \
        --volume-shock -0.15 --out stress.csv

volume.csv:  date, volume                       (card volume by calendar day)
actuals.csv: date, settled_volume, receipts, interest, payout_daily, payout_batch,
             chargebacks, balance               (one row per business day)
plan.csv:    date, payroll_plan, vendor_plan, tax_plan, is_batch_day   (business days, incl. future)
"""
import argparse

import numpy as np
import pandas as pd

from .backtest import horizon_days, week_number
from .data import generate
from .models import DriverBased


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume"), ap.add_argument("--actuals"), ap.add_argument("--plan")
    ap.add_argument("--volume-shock", type=float, default=0.0, help="e.g. -0.15 for a 15%% volume drop")
    ap.add_argument("--weeks", type=int, default=13)
    ap.add_argument("--out", default="forecast_13_week_daily.csv")
    a = ap.parse_args(argv)

    if a.volume and a.actuals and a.plan:
        volume = pd.read_csv(a.volume, parse_dates=["date"])
        actuals = pd.read_csv(a.actuals, parse_dates=["date"])
        plan = pd.read_csv(a.plan, parse_dates=["date"])
        plan["is_batch_day"] = plan["is_batch_day"].astype(bool)
        all_bdays = pd.DatetimeIndex(plan["date"]).as_unit("ns")
    else:
        volume, actuals, plan, all_bdays = generate()

    origin = actuals["date"].iloc[-1]
    future = horizon_days(origin, all_bdays, a.weeks)
    fc, comp = DriverBased().forecast(volume, actuals, plan, all_bdays, future,
                                      return_components=True, volume_shock=a.volume_shock)
    comp["net"] = fc["inflows"] - fc["outflows"]
    comp["balance"] = actuals["balance"].iloc[-1] + comp["net"].cumsum()
    comp["week"] = week_number(future, origin)
    comp.index.name = "date"
    comp.round(0).to_csv(a.out)
    low = comp["balance"].idxmin()
    print(f"Forecast from {origin:%Y-%m-%d}: {len(future)} business days -> {a.out}")
    print(f"Lowest balance ${comp['balance'].min() / 1e6:.1f}M on {low:%Y-%m-%d}; "
          f"week-{a.weeks} balance ${comp['balance'].iloc[-1] / 1e6:.1f}M")


if __name__ == "__main__":
    main()
