"""P4: check P2's full-BTS AAR-proxy model against an independently rebuilt target and P4 baselines.

Target (independent of P2): bts_arrivals.py, landings per UTC hour from BTS WheelsOn clock time.
Baselines, all fit on target hours before 2025-01-01 UTC (P2 trains on the same period):
  sched_x_ratio   scheduled arrivals in hour t x train completion ratio (per airport)
  hour_of_week    train mean arrivals per airport x local hour-of-week
  sched_spread    published schedule spread over the train landing-offset distribution:
                  pred(t) = sum_k sched(t-k) * P(land k hours after the scheduled hour | airport, local sched hour),
                  k in -3..+8; P includes the chance of not landing (cancel/divert). Uses only the schedule.
All three are h-independent (no information at t-h is used). P2's own baselines from its CSV are shown too.
Test = 2025 hours, P2 rows with split == test. 90% CI = day-block bootstrap of RMSE(model) - RMSE(best P4 baseline).

Usage: python src/eval/evaluate_aar_bts.py --preds data/interim/p2_aar_preds_bts.csv
    -> outputs/metrics/aar_bts_model_vs_baseline.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import OUT, load_airports
from bts_arrivals import FLIGHTS, HOURLY

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
OFFSETS = range(-3, 9)
P4_BASELINES = ["sched_x_ratio", "hour_of_week", "sched_spread"]
N_BOOT = 500


def baselines(hourly, flights, airports):
    out = []
    for icao, g in hourly.groupby("icao"):
        g = g.sort_values("target_hour_utc").set_index("target_hour_utc").copy()
        local = g.index.tz_convert(airports[icao]["timezone"])
        g["how"], g["hod"] = local.dayofweek * 24 + local.hour, local.hour
        tr = g[g.index < TEST_START]
        g["sched_x_ratio"] = g["sched_arr_count"] * tr["arr_count"].sum() / tr["sched_arr_count"].sum()
        g["hour_of_week"] = g["how"].map(tr.groupby("how")["arr_count"].mean())
        f = flights[(flights["icao"] == icao) & (flights["sched_hour_utc"] < TEST_START)].dropna(subset=["sched_hour_utc"])
        f = f.assign(hod=f["sched_hour_utc"].dt.tz_convert(airports[icao]["timezone"]).dt.hour,
                     k=((f["wheels_hour_utc"] - f["sched_hour_utc"]) / pd.Timedelta(hours=1)))
        p = (f[f["k"].isin(OFFSETS)].groupby(["hod", "k"]).size() / f.groupby("hod").size()).unstack(fill_value=0)
        pred = np.zeros(len(g))
        for k in OFFSETS:
            s_k = g["sched_arr_count"].shift(k).to_numpy()  # schedule of hour t-k (grid is complete hourly)
            hod_k = g["hod"].shift(k)
            pk = hod_k.map(p[k] if k in p else pd.Series(dtype=float)).fillna(0).to_numpy()
            pred += np.nan_to_num(s_k) * pk
        g["sched_spread"] = pred
        print(f"{icao}: train completion ratio {tr['arr_count'].sum() / tr['sched_arr_count'].sum():.4f}; "
              f"landing offset share k=0 {f['k'].eq(0).mean():.3f}, k=1 {f['k'].eq(1).mean():.3f}, "
              f"k<0 {f['k'].lt(0).mean():.3f}, not landed {f['k'].isna().mean():.3f}")
        out.append(g.reset_index()[["icao", "target_hour_utc"] + P4_BASELINES])
    return pd.concat(out, ignore_index=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def boot(g, best, rng):
    day = g["target_hour_utc"].dt.floor("D").to_numpy()
    days, inv = np.unique(day, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    y, m, b = g["y"].to_numpy(float), g["model"].to_numpy(float), g[best].to_numpy(float)
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        d.append(rmse(y[idx], m[idx]) - rmse(y[idx], b[idx]))
    return np.percentile(d, [5, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    args = ap.parse_args()
    airports = load_airports()
    hourly, flights = pd.read_parquet(HOURLY), pd.read_parquet(FLIGHTS)
    base = baselines(hourly, flights, airports)
    p2 = pd.read_csv(args.preds, parse_dates=["target_hour_utc"])
    p2 = p2[p2["split"] == "test"].rename(columns={"pred_arr_count": "model"})
    df = (p2.merge(hourly[["icao", "target_hour_utc", "arr_count", "sched_arr_count"]], on=["icao", "target_hour_utc"])
          .merge(base, on=["icao", "target_hour_utc"]).rename(columns={"arr_count": "y"}))
    one = df[df["h"] == 3]
    diff = one["y"] - one["actual_arr_count"]
    print(f"P2 test rows {len(p2)}, joined {len(df)}. Target check (h=3 rows, {len(one)} hours): P4 == P2 on "
          f"{(diff == 0).mean():.1%} of hours, |diff| <= 1 on {(diff.abs() <= 1).mean():.1%}, mean diff {diff.mean():+.3f}, "
          f"P4 mean {one['y'].mean():.2f} vs P2 mean {one['actual_arr_count'].mean():.2f} arrivals/h")

    slices = {"all hours": lambda d: np.ones(len(d), bool)}
    if "eval_hub_weather_actual" in df:
        for v in sorted(df["eval_hub_weather_actual"].dropna().unique()):
            slices[f"hub wx at t: {v}"] = lambda d, v=v: (d["eval_hub_weather_actual"] == v).to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for sl, fn in slices.items():
        for (icao, h), g in df[fn(df)].groupby(["icao", "h"]):
            r = {"slice": sl, "icao": icao, "h": h, "n_hours": len(g), "mean_actual": g["y"].mean(),
                 "rmse_model": rmse(g["y"], g["model"]), "mae_model": float(np.mean(np.abs(g["y"] - g["model"])))}
            for b in P4_BASELINES + ["pred_schedule", "pred_hist_avg", "pred_persistence"]:
                if b in g:
                    r[f"rmse_{b}"] = rmse(g["y"], g[b])
            best = min(P4_BASELINES, key=lambda b: r[f"rmse_{b}"])
            r["best_p4_baseline"], r["impr_vs_best_pct"] = best, 100 * (r["rmse_model"] / r[f"rmse_{best}"] - 1)
            r["diff_ci90_lo"], r["diff_ci90_hi"] = boot(g, best, rng)
            if {"p10", "p90"} <= set(g):
                r["band_coverage"] = float(((g["y"] >= g["p10"]) & (g["y"] <= g["p90"])).mean())
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "aar_bts_model_vs_baseline.csv", index=False)
    cols = ["slice", "icao", "h", "n_hours", "mean_actual", "rmse_model", "rmse_sched_spread", "rmse_sched_x_ratio",
            "rmse_hour_of_week", "rmse_pred_persistence", "best_p4_baseline", "impr_vs_best_pct", "diff_ci90_lo",
            "diff_ci90_hi", "band_coverage"]
    pd.set_option("display.width", 250)
    print(res[[c for c in cols if c in res]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
