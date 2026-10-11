"""Rolling-origin backtest: forecast 13 weeks from many past Fridays and score against actuals."""
import numpy as np
import pandas as pd

from .models import TARGETS

WEEKS = 13


def horizon_days(origin, all_bdays, weeks=WEEKS):
    """Business days from the day after `origin` through the end of the 13th following week."""
    end = (origin + pd.offsets.Week(weekday=4) * weeks) if origin.dayofweek != 4 else origin + pd.Timedelta(weeks=weeks)
    return all_bdays[(all_bdays > origin) & (all_bdays <= end)]


def week_number(dates, origin):
    """1 for the week ending the first Friday after the origin, 2 for the next, ..."""
    weeks = pd.DatetimeIndex(dates).to_period("W-FRI")
    return np.array([(w - origin.to_period("W-FRI")).n for w in weeks])


def origins(actuals, start, every_weeks=4, weeks=WEEKS):
    last = actuals["date"].iloc[-1] - pd.Timedelta(weeks=weeks)
    fridays = pd.date_range(start, last, freq=f"{every_weeks}W-FRI")
    days = set(actuals["date"])
    return [d if d in days else actuals.loc[actuals["date"] < d, "date"].iloc[-1] for d in fridays]


def mape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs(actual - forecast) / np.abs(actual)))


def run(models, volume, actuals, plan, all_bdays, origin_list):
    """Returns one row per (origin, model, business day) with forecast and actual flows."""
    rows = []
    for o in origin_list:
        hist = actuals[actuals["date"] <= o]
        future = horizon_days(o, all_bdays)
        truth = actuals.set_index("date").loc[future]
        for M in models:
            fc = M().forecast(volume, hist, plan, all_bdays, future)
            frame = pd.DataFrame({
                "origin": o, "model": M.name, "date": future,
                "horizon": np.arange(1, len(future) + 1),
                "week": week_number(future, o),
                "fc_inflows": fc["inflows"].values, "fc_outflows": fc["outflows"].values,
                "inflows": truth["inflows"].values, "outflows": truth["outflows"].values,
            })
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def score(bt):
    """Daily MAPE over the first two weeks, weekly MAPE over 13 weeks, and the error on
    cumulative net cash (what drives the balance) at week 13."""
    out = []
    for name, g in bt.groupby("model", sort=False):
        short = g[g["week"] <= 2]
        weekly = g.groupby(["origin", "week"])[["inflows", "outflows", "fc_inflows", "fc_outflows"]].sum()
        net = g.assign(net=g["inflows"] - g["outflows"], fc_net=g["fc_inflows"] - g["fc_outflows"])
        cum = net.groupby("origin")[["net", "fc_net"]].sum()
        out.append({
            "model": name,
            "daily_mape_inflows": mape(short["inflows"], short["fc_inflows"]),
            "daily_mape_outflows": mape(short["outflows"], short["fc_outflows"]),
            "weekly_mape_inflows": mape(weekly["inflows"], weekly["fc_inflows"]),
            "weekly_mape_outflows": mape(weekly["outflows"], weekly["fc_outflows"]),
            "week13_balance_mae": float(np.mean(np.abs(cum["net"] - cum["fc_net"]))),
        })
    return pd.DataFrame(out)


def balance_band(bt, model, z=1.2816):
    """Spread of cumulative net-cash error by horizon, scaled as sigma * sqrt(h).
    Returns sigma and how often actuals fell inside the resulting P10-P90 band."""
    g = bt[bt["model"] == model].sort_values(["origin", "horizon"]).copy()
    g["err"] = (g["inflows"] - g["outflows"]) - (g["fc_inflows"] - g["fc_outflows"])
    g["cum_err"] = g.groupby("origin")["err"].cumsum()
    sigma = float(np.sqrt(np.mean(g["cum_err"] ** 2 / g["horizon"])))
    inside = np.abs(g["cum_err"]) <= z * sigma * np.sqrt(g["horizon"])
    return sigma, float(inside.mean())
