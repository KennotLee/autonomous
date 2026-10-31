"""P4 Task 3: failure cases on 2025 test events at h=3, per airport x local day.

  missed storm  = day with many >45 min delays the predictor did not expect (max observed - expected, >= 4 late)
  false alarm   = day the predictor expected many delays that did not happen (max expected - observed, <= 1 late,
                  >= 3 events in the global top-15% flags)

For each case, per event: predicted risk, outcome, weather at t-3h (the row the predictor saw) vs weather at
the scheduled time, and the BTS delay-cause minutes (CARRIER/WEATHER/NAS/LATE_AIRCRAFT). Cause columns and
flight-time weather are post-event: used here only to EXPLAIN a failure, never as features.
Flight-time weather: Meteostat hourly (the organiser file's source) at floor(t) + 1 h, which holds the
METAR issued at ~t:52 (organiser row T == Meteostat row T+1 on 66% of hours, checked in the leakage review).

Usage: python src/eval/failure_cases.py [--preds <csv> --name <tag>] [--events <parquet>]
    no --preds: scores B3 (best baseline).  -> outputs/metrics/failure_cases_<name>.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import OUT, PROCESSED, ROOT, load_airports, meteostat_hourly
from baselines_events import ADVERSE_CODES, CROSSWIND_KMH, TAILWIND_KMH, fit_predict, load, top_flag

CAUSES = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY"]
COCO = {1: "clear", 2: "fair", 3: "cloudy", 4: "overcast", 5: "fog", 6: "freezing fog", 7: "light rain", 8: "rain",
        9: "heavy rain", 10: "freezing rain", 14: "light snow", 15: "snow", 16: "heavy snow", 17: "rain shower",
        18: "heavy rain shower", 23: "lightning", 24: "hail", 25: "thunderstorm", 26: "heavy thunderstorm", 27: "storm"}


def flight_time_wx(te, airports):
    """Meteostat at floor(t)+1h per event airport, with runway components and the organiser adverse rule."""
    out = []
    for icao, g in te.groupby("airport"):
        m = meteostat_hourly(icao)
        m = m.reset_index(level=0, drop=True) if isinstance(m.index, pd.MultiIndex) else m
        m.index = pd.DatetimeIndex(m.index).tz_localize(None) if m.index.tz is None else m.index.tz_convert(None)
        key = (g["sched_time_utc"].dt.floor("h") + pd.Timedelta(hours=1)).dt.tz_convert(None)
        w = m.reindex(key)[["coco", "wspd", "wdir", "prcp"]].set_axis(g.index)
        th = np.deg2rad(w["wdir"] - airports[icao]["runway_heading_deg"])
        head = w["wspd"] * np.cos(th)
        w["cross"], w["tail"] = (w["wspd"] * np.sin(th)).abs(), (-head).clip(lower=0)
        w["adverse"] = w["coco"].isin(ADVERSE_CODES) | (w["cross"] >= CROSSWIND_KMH) | (w["tail"] >= TAILWIND_KMH)
        out.append(w.add_prefix("ft_"))
    return pd.concat(out)


def causes(te):
    raw = pd.read_csv(ROOT / "data/raw/Flight_Data_With_Weather_Final.csv", low_memory=False,
                      usecols=["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST"] + CAUSES)
    raw["flight_key"] = (raw.FL_DATE + "_" + raw.OP_UNIQUE_CARRIER + raw.OP_CARRIER_FL_NUM.astype("Int64").astype(str)
                         + "_" + raw.ORIGIN + "_" + raw.DEST)
    raw = raw.drop_duplicates("flight_key", keep=False).set_index("flight_key")[CAUSES]
    return te[["flight_key"]].join(raw, on="flight_key")[CAUSES]


def pick(days):
    missed = days[days["late"] >= 4].assign(gap=lambda d: d["late"] - d["expected"]).nlargest(1, "gap")
    alarm = days[(days["late"] <= 1) & (days["flagged"] >= 3)].assign(gap=lambda d: d["expected"] - d["late"]).nlargest(1, "gap")
    return [("missed storm", missed.iloc[0]), ("false alarm", alarm.iloc[0])]


def wx_text(code, cross, tail, adverse):
    c = COCO.get(int(code), f"code {int(code)}") if pd.notna(code) else "n/a"
    return f"{'**ADV** ' if adverse else ''}{c}, x{cross:.0f}/t{tail:.0f}"


def case_md(kind, d, te):
    ev = te[(te["apt"] == d["apt"]) & (te["date"] == d["date"])].sort_values("sched_time_local")
    late = ev[ev["y"] == 1]
    cause_tot = late[CAUSES].sum()
    lines = [f"### {kind}: {d['apt']} {d['date']} (local)",
             f"- {int(d['n'])} events, **{int(d['late'])} late >45 min** ({d['late'] / d['n']:.0%}); predictor expected "
             f"{d['expected']:.1f} (mean risk {d['mean_risk']:.3f}); {int(d['flagged'])} in global top-15% flags",
             f"- adverse wx at t-3h on {ev['adverse'].mean():.0%} of events vs at flight time on {ev['ft_adverse'].mean():.0%}",
             f"- BTS cause minutes over late events: " + ", ".join(f"{c.replace('_DELAY', '').lower()} {cause_tot[c]:.0f}"
                                                                 for c in CAUSES),
             "", "| local time | type | other end | risk | delay min | wx at t-3h (obs) | wx at flight time | main cause |",
             "|---|---|---|---|---|---|---|---|"]
    for r in ev.itertuples():
        cs = pd.Series({c: getattr(r, c) for c in CAUSES}).dropna()
        main = cs.idxmax().replace("_DELAY", "").lower() if len(cs) and cs.max() > 0 else ""
        lines.append(f"| {r.sched_time_local:%H:%M} | {r.event_type} | {r.other_airport} | {r.pred:.3f} | {r.delay_min:.0f} "
                     f"| {wx_text(r.coco, abs(r.crosswind), r.tailwind, r.adverse)} ({r.feature_obs_time:%H:%MZ}) "
                     f"| {wx_text(r.ft_coco, r.ft_cross, r.ft_tail, r.ft_adverse)} | {main} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--name", default="B3")
    ap.add_argument("--events", type=Path, default=PROCESSED / "event_features.parquet")
    args = ap.parse_args()
    ev = fit_predict(load(args.events))
    te = ev[(ev["split"] == "test") & (ev["h"] == 3)].copy()
    if args.preds:
        p = pd.read_csv(args.preds, usecols=["event_id", "h", "risk_prob"])
        te = te.merge(p[p["h"] == 3].drop(columns="h"), on="event_id", how="inner").rename(columns={"risk_prob": "pred"})
    else:
        te["pred"] = te["B3_B2_x_adverse_wx"]
    te = te.reset_index(drop=True)
    te["flag"] = top_flag(te["pred"].to_numpy())
    te["date"] = te["sched_time_local"].dt.date.astype(str)
    te = te.join(flight_time_wx(te, load_airports())).join(causes(te))
    days = te.groupby(["apt", "date"]).agg(n=("y", "size"), late=("y", "sum"), expected=("pred", "sum"),
                                           mean_risk=("pred", "mean"), flagged=("flag", "sum")).reset_index()
    md = [f"# Failure cases: {args.name}, h=3, 2025 test (generated by src/eval/failure_cases.py)\n",
          "Risk = predicted P(delay > 45 min). Weather: x = crosswind, t = tailwind (km/h); **ADV** = organiser adverse "
          "rule. t-3h = the row the predictor saw (obs stamp); flight time = Meteostat at the scheduled hour. "
          "Main cause = largest BTS cause column (post-hoc, explanation only).\n"]
    md += [case_md(kind, d, te) + "\n" for kind, d in pick(days)]
    path = OUT / f"failure_cases_{args.name}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("wrote", path)


if __name__ == "__main__":
    main()
