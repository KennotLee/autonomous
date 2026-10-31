"""Pick the worst missed-weather day in the test set and lay out its flights.

Selection (h = 6, test flights with a label): the hub-day with the most severe
delays (> 45 min) that the model scored below the alert threshold. The narrative
(why / how a user notices) is hand-written in failure_notes.md and appended.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import airports

H = 6
ROOT = Path(__file__).resolve().parents[3]
TS = {23, 24, 25, 26, 27}


def _hub_local(df):
    """Local time at the hub of the flight's hub event (arrival if inbound)."""
    t = pd.Series(np.where(df.arr_to_hub == 1, df.sched_arr_utc, df.sched_time_utc), index=df.index)
    t = pd.to_datetime(t, utc=True)
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for icao, a in airports.hubs().items():
        m = df.hub == icao
        out[m] = t[m].dt.tz_convert(a.tz).dt.tz_localize(None)
    return out


def pick_day(te, thr):
    d = te[(te.h == H)].copy()
    d["hub_local"] = _hub_local(d)
    d["date"] = d.hub_local.dt.normalize()
    d["missed"] = (d.label_45 == 1) & (d.p_gbm < thr)
    days = d.groupby(["hub", "date"]).agg(flights=("label_45", "size"), severe=("label_45", "sum"),
                                         missed=("missed", "sum"), mean_risk=("p_gbm", "mean"),
                                         mean_oracle=("p_gbm_oracle", "mean"))
    days = days.sort_values(["missed", "severe"], ascending=False)
    hub, date = days.index[0]
    return hub, date, days, d


def write_failure_case(fl, thr, out_dir):
    te = fl[(fl.split == "test") & fl.label_45.notna()]
    hub, date, days, d6 = pick_day(te, thr)
    day = d6[(d6.hub == hub) & (d6.date == date)].sort_values("hub_local")
    by_h = te[te.flight_id.isin(day.flight_id)].pivot_table(index="flight_id", columns="h", values="p_gbm")
    by_h.columns = [f"risk_h{h}" for h in by_h.columns]
    end = np.where(day.arr_to_hub == 1, "dest", "origin")
    table = pd.DataFrame({
        "flight_id": day.flight_id.values,
        "hub_time_local": day.hub_local.dt.strftime("%H:%M").values,
        "delay_min": day.delay_min.values,
        "oracle_h6": day.p_gbm_oracle.round(2).values,
        "hub_coco_at_issue": [day[f"{e}_coco"].iat[i] for i, e in enumerate(end)],
        "hub_coco_at_flight_hour": [day[f"oracle_{e}_coco"].iat[i] for i, e in enumerate(end)],
        "top_drivers_h6": day.top_drivers.values,
    }).merge(by_h.round(2).reset_index(), on="flight_id")

    # hub weather timeline for the day (Meteostat file from pipeline.download, if present)
    a = airports.hubs()[hub]
    wx_path = ROOT / "data" / "raw" / "weather" / f"{hub}.parquet"
    fig, ax = plt.subplots(figsize=(8, 4))
    if wx_path.exists():
        wx = pd.read_parquet(wx_path)
        wx.index = wx.index.tz_convert(a.tz).tz_localize(None)
        w = wx.loc[date: date + pd.Timedelta(hours=23)]
        for t in w.index[w.coco.isin(TS)]:
            ax.axvspan(t.hour - 0.5, t.hour + 0.5, color="#ffd54f", alpha=0.4, lw=0)
        ax2 = ax.twinx()
        ax2.bar(w.index.hour, w.prcp.fillna(0), color="#9ecae1", alpha=0.5, width=0.8)
        ax2.set_ylabel("precip mm/h (hub)")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    x = day.hub_local.dt.hour + day.hub_local.dt.minute / 60
    sev = day.label_45 == 1
    ax.scatter(x[sev], day.p_gbm[sev], marker="X", s=80, color="#d62728", label="model risk h=6, delayed > 45", zorder=3)
    ax.scatter(x[~sev], day.p_gbm[~sev], marker="o", s=50, color="#2ca02c", label="model risk h=6, on time-ish", zorder=3)
    ax.scatter(x, day.p_gbm_oracle, marker="s", s=25, facecolors="none", edgecolors="#ff7f0e", label="oracle", zorder=3)
    ax.axhline(thr, color="k", lw=0.8, ls=":", label=f"alert threshold {thr:.2f}")
    ax.set(xlim=(-0.5, 23.5), ylim=(0, 1), xlabel=f"hub time (local), {hub} {date:%Y-%m-%d}",
           ylabel="P(delay > 45 min)", title="Failure case: shaded = thunderstorm coded at the hub")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_failure_case.png", dpi=130)
    plt.close(fig)

    with open(out_dir / "delay_failure_case.md", "w") as f:
        f.write(f"# Failure case: {hub} on {date:%Y-%m-%d} (h = {H} h)\n\n")
        f.write("Auto-selected: the test hub-day with the most > 45 min delays scored below the alert "
                f"threshold ({thr:.2f}) at h = 6.\n\n![timeline](delay_failure_case.png)\n\n")
        f.write("## Flights that day\n\n" + table.to_markdown(index=False) + "\n\n")
        f.write("## Worst-missed hub-days\n\n" + days.head(5).round(3).to_markdown() + "\n\n")
        notes = Path(__file__).with_name("failure_notes.md")
        f.write(notes.read_text() if notes.exists() else "_No notes written yet._\n")
    print(f"failure case: {hub} {date:%Y-%m-%d}\n{table.drop(columns='top_drivers_h6').to_string(index=False)}")
