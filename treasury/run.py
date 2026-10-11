"""End to end: generate data, backtest four models, build the live 13-week forecast, write reports
and the project page.

    python -m treasury.run
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest as B
from .data import CHARGEBACK_RATE, generate
from .models import MODELS, DriverBased

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
Z90 = 1.2816
SENSITIVE = ["receipts", "payout_daily", "payout_batch", "chargebacks"]
COLORS = {"Seasonal naive": "#9AA3AD", "Holt-Winters": "#C0892B", "Gradient boosting": "#4A78A8", "Driver-based": "#1F7A5A"}


def live_forecast(volume, actuals, plan, all_bdays, sigma):
    origin = actuals["date"].iloc[-1]
    future = B.horizon_days(origin, all_bdays)
    fc, comp = DriverBased().forecast(volume, actuals, plan, all_bdays, future, return_components=True)
    # every volume-driven line is linear in forecast volume, so a +100% run gives the sensitivity
    _, shocked = DriverBased().forecast(volume, actuals, plan, all_bdays, future, return_components=True, volume_shock=1.0)
    for c in SENSITIVE:
        comp[c + "_sens"] = shocked[c] - comp[c]
    comp["net"] = fc["inflows"] - fc["outflows"]
    comp["balance"] = actuals["balance"].iloc[-1] + comp["net"].cumsum()
    h = np.arange(1, len(future) + 1)
    comp["p10"] = comp["balance"] - Z90 * sigma * np.sqrt(h)
    comp["p90"] = comp["balance"] + Z90 * sigma * np.sqrt(h)
    comp["week"] = B.week_number(future, origin)
    comp.index.name = "date"
    return comp


def plot(scores, bt, actuals, live, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1.2, 1.5]})
    names = scores["model"].tolist()
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, scores["weekly_mape_inflows"] * 100, 0.4, label="Inflows", color="#4A78A8")
    ax[0].bar(x + 0.2, scores["weekly_mape_outflows"] * 100, 0.4, label="Outflows", color="#C0892B")
    ax[0].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=9)
    ax[0].set_ylabel("Weekly MAPE, %")
    ax[0].set_title("13-week forecast error (14 backtests)")
    ax[0].legend(frameon=False)

    o = pd.Timestamp("2025-10-31")
    g = bt[bt["origin"] == o]
    for name, m in g.groupby("model", sort=False):
        wk = m.groupby("week").agg(date=("date", "max"), fc=("fc_inflows", "sum"), fo=("fc_outflows", "sum"))
        ax[1].plot(wk["date"], (wk["fc"] - wk["fo"]).cumsum() / 1e6, color=COLORS[name], label=name, lw=1.6, marker="o", ms=3)
    a = g[g["model"] == names[0]].groupby("week").agg(date=("date", "max"), i=("inflows", "sum"), o=("outflows", "sum"))
    ax[1].plot(a["date"], (a["i"] - a["o"]).cumsum() / 1e6, color="black", lw=2.4, label="Actual", marker="o", ms=3)
    ax[1].axhline(0, color="#ccc", lw=0.8)
    ax[1].set_title(f"Cumulative net cash by week from {o:%d %b %Y}, $M")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].tick_params(axis="x", labelrotation=30, labelsize=8)

    hist = actuals.tail(60)
    ax[2].plot(hist["date"], hist["balance"] / 1e6, color="black", lw=1.6, label="Actual")
    ax[2].fill_between(live.index, live["p10"] / 1e6, live["p90"] / 1e6, color=COLORS["Driver-based"], alpha=0.18, label="P10-P90")
    ax[2].plot(live.index, live["balance"] / 1e6, color=COLORS["Driver-based"], lw=1.6, label="Forecast")
    ax[2].set_title("13-week cash balance forecast, $M")
    ax[2].legend(frameon=False, fontsize=8, loc="upper left")
    ax[2].tick_params(axis="x", labelrotation=30, labelsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def page_data(scores, actuals, live, sigma, coverage, n_origins):
    m = lambda v: round(float(v) / 1e6, 3)
    cols = ["receipts", "interest", "payout_daily", "payout_batch", "chargebacks", "payroll", "vendors", "tax"]
    return {
        "origin": actuals["date"].iloc[-1].strftime("%Y-%m-%d"),
        "opening_balance": m(actuals["balance"].iloc[-1]),
        "sigma": m(sigma),
        "chargeback_rate": CHARGEBACK_RATE,
        "band_coverage": round(coverage, 3),
        "backtests": n_origins,
        "models": [{"model": r["model"],
                    "daily_in": round(r["daily_mape_inflows"], 4), "daily_out": round(r["daily_mape_outflows"], 4),
                    "weekly_in": round(r["weekly_mape_inflows"], 4), "weekly_out": round(r["weekly_mape_outflows"], 4),
                    "balance_mae": m(r["week13_balance_mae"])} for r in scores.to_dict("records")],
        "history": [{"date": d.strftime("%Y-%m-%d"), "balance": m(b)}
                    for d, b in zip(actuals["date"].tail(40), actuals["balance"].tail(40))],
        "days": [{"date": d.strftime("%Y-%m-%d"), "week": int(r["week"]), **{c: m(r[c]) for c in cols},
                  "sens": [m(r[c + "_sens"]) for c in SENSITIVE]}
                 for d, r in live.iterrows()],
    }


def build_page(data):
    template = (ROOT / "web" / "template.html").read_text()
    html = template.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (ROOT / "index.html").write_text(html)


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    volume, actuals, plan, all_bdays = generate()
    origin_list = B.origins(actuals, "2025-07-11")
    bt = B.run(MODELS, volume, actuals, plan, all_bdays, origin_list)
    scores = B.score(bt)
    sigma, coverage = B.balance_band(bt, DriverBased.name)
    live = live_forecast(volume, actuals, plan, all_bdays, sigma)

    bt.to_csv(REPORTS / "backtest_daily.csv.gz", index=False)
    live.drop(columns=[c + "_sens" for c in SENSITIVE]).round(0).to_csv(REPORTS / "forecast_13_week_daily.csv")
    weekly = live.groupby("week").agg(
        week_ending=("balance", lambda s: s.index.max().strftime("%Y-%m-%d")),
        receipts=("receipts", "sum"), merchant_payouts=("payout_daily", "sum"),
        batch_payouts=("payout_batch", "sum"), payroll=("payroll", "sum"), vendors=("vendors", "sum"),
        tax=("tax", "sum"), chargebacks=("chargebacks", "sum"), interest=("interest", "sum"),
        net=("net", "sum"), ending_balance=("balance", "last"), low_p10=("p10", "min"))
    weekly.round(0).to_csv(REPORTS / "forecast_13_week.csv")
    plot(scores, bt, actuals, live, REPORTS / "results.png")

    metrics = {"scores": scores.to_dict("records"), "balance_band_sigma_per_sqrt_day": sigma,
               "p10_p90_coverage": coverage, "origins": [str(o.date()) for o in origin_list],
               "transactions_days": int(len(actuals)), "runtime_seconds": round(time.time() - t0, 1)}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_results_md(scores, sigma, coverage, len(origin_list), weekly, actuals)
    build_page(page_data(scores, actuals, live, sigma, coverage, len(origin_list)))
    print(scores.round(4).to_string(index=False))
    print(f"P10-P90 coverage {coverage:.1%}; runtime {time.time() - t0:.0f}s")


def write_results_md(scores, sigma, coverage, n, weekly, actuals):
    lines = [f"# Results ({n} rolling 13-week backtests, Jul 2025 - Oct 2026)", "",
             "| Model | Daily MAPE in | Daily MAPE out | Weekly MAPE in | Weekly MAPE out | Week-13 balance error |",
             "|---|---|---|---|---|---|"]
    for r in scores.itertuples():
        lines.append(f"| {r.model} | {r.daily_mape_inflows:.1%} | {r.daily_mape_outflows:.1%} | "
                     f"{r.weekly_mape_inflows:.1%} | {r.weekly_mape_outflows:.1%} | ${r.week13_balance_mae / 1e6:.1f}M |")
    lines += ["", f"Driver-based P10-P90 balance band: sigma ${sigma / 1e6:.2f}M x sqrt(business days ahead); "
              f"{coverage:.1%} of backtest days fell inside (target 80%).", "",
              f"## Live 13-week forecast from {actuals['date'].iloc[-1]:%d %b %Y} (opening balance ${actuals['balance'].iloc[-1] / 1e6:.1f}M)", "",
              "| Week | Week ending | Receipts | Payouts | Payroll | Other out | Net | Ending balance | Lowest P10 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for w, r in weekly.iterrows():
        other = r.vendors + r.tax + r.chargebacks - r.interest
        lines.append(f"| {w} | {r.week_ending} | ${r.receipts / 1e6:.1f}M | ${(r.merchant_payouts + r.batch_payouts) / 1e6:.1f}M | "
                     f"${r.payroll / 1e6:.1f}M | ${other / 1e6:.1f}M | ${r.net / 1e6:+.1f}M | ${r.ending_balance / 1e6:.1f}M | ${r.low_p10 / 1e6:.1f}M |")
    (REPORTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
