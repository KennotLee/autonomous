"""Pick the worst missed-storm day in the test set and lay out its timeline.

Selection (h = 6, flights arriving at a hub, days with >= 80 sampled flights):
the day with the largest gap between the observed severe-delay share and the
model's mean predicted risk. The narrative (why / how a user notices) is
written by hand in delay_failure_case.md below the generated tables.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H = 6


def pick_day(te: pd.DataFrame) -> tuple[str, pd.Timestamp, pd.DataFrame]:
    d = te[(te.h == H) & (te.arr_to_hub == 1)].copy()
    d["date"] = d.sched_arr.dt.normalize()
    days = d.groupby(["hub", "date"]).agg(n=("label", "size"), actual=("label", "mean"),
                                         pred=("p_gbm", "mean"), oracle=("p_oracle", "mean"))
    days = days[days.n >= 80]
    days["miss"] = days.actual - days.pred
    hub, date = days.miss.idxmax()
    return hub, date, days.sort_values("miss", ascending=False)


def timeline(te: pd.DataFrame, hub: str, date: pd.Timestamp, thr: float) -> pd.DataFrame:
    d = te[(te.h == H) & (te.arr_to_hub == 1) & (te.hub == hub) & (te.sched_arr.dt.normalize() == date)]
    return d.groupby(d.sched_arr.dt.hour).agg(
        flights=("label", "size"),
        actual_gt45=("label", "mean"),
        model_risk_h6=("p_gbm", "mean"),
        flagged=("p_gbm", lambda p: (p >= thr).mean()),
        oracle_risk=("p_oracle", "mean"),
        ts_seen_at_issue=("dest_ts_3h", "max"),
        ts_at_arrival_hour=("now_dest_ts", "max"),
        prcp_at_arrival_mm=("now_dest_prcp", "mean"),
        wind_at_arrival_kt=("now_dest_wspd_kt", "mean"),
    ).rename_axis("local_hour")


def write_failure_case(te: pd.DataFrame, thr: float, out_dir) -> None:
    hub, date, days = pick_day(te)
    tl = timeline(te, hub, date, thr)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tl.index, tl.actual_gt45, color="#d9d9d9", label="observed share > 45 min")
    ax.plot(tl.index, tl.model_risk_h6, "o-", color="#1f77b4", label="model risk (issued 6 h ahead)")
    ax.plot(tl.index, tl.oracle_risk, "s--", color="#ff7f0e", label="oracle (same-hour weather)")
    ax.axhline(thr, color="k", lw=0.8, ls=":", label=f"alert threshold {thr:.2f}")
    ts = tl.index[tl.ts_at_arrival_hour.fillna(0) > 0]
    for x in ts:
        ax.axvspan(x - 0.5, x + 0.5, color="#ffd54f", alpha=0.35, lw=0)
    ax.set(xlabel=f"scheduled arrival hour (local), {hub} {date:%Y-%m-%d}", ylabel="share / probability",
           ylim=(0, 1), title="Failure case: shaded = thunderstorm coded at the arrival hour")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_failure_case.png", dpi=130)
    plt.close(fig)

    path = out_dir / "delay_failure_case.md"
    with open(path, "w") as f:
        f.write(f"# Failure case: {hub} arrivals on {date:%Y-%m-%d} (h = {H} h)\n\n")
        f.write("Auto-selected as the test day with the largest gap between observed and predicted "
                "severe-delay share (h = 6, hub arrivals, >= 80 sampled flights).\n\n")
        f.write("![timeline](delay_failure_case.png)\n\n## Hourly timeline\n\n")
        f.write(tl.round(3).to_markdown() + "\n\n## Five worst-missed days\n\n")
        f.write(days.head(5).round(3).to_markdown() + "\n\n")
        notes = Path(__file__).with_name("failure_notes.md")
        f.write(notes.read_text() if notes.exists() else "_No notes written yet._\n")
    print(f"failure case: {hub} {date:%Y-%m-%d}\n{tl.round(2).to_string()}")
