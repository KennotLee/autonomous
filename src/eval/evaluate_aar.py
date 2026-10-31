"""P4: score P2's AAR-proxy predictions against P4 baselines on identical 2025 test hours.

Usage:
    python src/eval/evaluate_aar.py --preds outputs/predictions/aar_preds.csv [--model-col pred_arr_count]

Reads P2's prediction CSV (icao, target_hour_utc, h, <model column>), P4 baseline preds
(outputs/metrics/aar_baseline_preds.csv, from baselines.py) and P1's hourly_features.parquet
(arr_count target, sched_arr_count, oracle_coco for slicing only).

Writes outputs/metrics/aar_model_vs_baseline.csv: RMSE per model x icao x h x slice, improvement % of the
model over each baseline, and a 90% day-block bootstrap CI on the RMSE difference (model - baseline).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import OUT, PROCESSED, SEVERE_CODES, TEST_START

KEYS = ["icao", "target_hour_utc", "h"]
N_BOOT = 500


def load(preds_path, model_col):
    p2 = pd.read_csv(preds_path, parse_dates=["target_hour_utc"])
    if "split" in p2:
        p2 = p2[p2["split"] == "test"]
    band = [c for c in ("p10", "p90") if c in p2]
    p2 = p2[KEYS + [model_col] + band].rename(columns={model_col: "model"})
    base_long = pd.read_csv(OUT / "aar_baseline_preds.csv", parse_dates=["target_hour_utc"])
    base = base_long.pivot_table(index=KEYS, columns="model", values="pred_arr_count").reset_index()
    # band of the strongest baseline (residual quantiles from baselines.py), to compare calibration
    sched_band = (base_long[base_long["model"] == "scheduled_x_ratio"][KEYS + ["p10", "p90"]]
                  .rename(columns={"p10": "sched_p10", "p90": "sched_p90"}))
    base = base.merge(sched_band, on=KEYS, how="left")
    hourly = pd.read_parquet(PROCESSED / "hourly_features.parquet",
                             columns=KEYS + ["arr_count", "sched_arr_count", "oracle_coco"])
    hourly = hourly[hourly["target_hour_utc"] >= TEST_START]
    df = hourly.merge(p2, on=KEYS, how="inner").merge(base, on=KEYS, how="inner")
    print(f"test rows: hourly {len(hourly)}, P2 preds {len(p2)}, joined {len(df)}")
    return df


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def band_stats(y, lo, hi, prefix):
    """Calibration of a p10-p90 band: coverage (nominal 0.80), share below p10 / above p90 (nominal 0.10 each)."""
    return {f"{prefix}_coverage": float(np.mean((y >= lo) & (y <= hi))), f"{prefix}_below_p10": float(np.mean(y < lo)),
            f"{prefix}_above_p90": float(np.mean(y > hi)), f"{prefix}_width": float(np.mean(hi - lo))}


def boot_diff(g, base_col, rng):
    """90% CI of RMSE(model) - RMSE(baseline), resampling whole UTC days (hours within a day are correlated)."""
    day = g["target_hour_utc"].dt.floor("D").to_numpy()
    days = np.unique(day)
    groups = {d: np.flatnonzero(day == d) for d in days}
    y, m, b = g["arr_count"].to_numpy(float), g["model"].to_numpy(float), g[base_col].to_numpy(float)
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[d] for d in rng.choice(days, len(days))])
        diffs.append(rmse(y[idx], m[idx]) - rmse(y[idx], b[idx]))
    return np.percentile(diffs, [5, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("outputs/predictions/aar_preds.csv"))
    ap.add_argument("--model-col", default="pred_arr_count")  # P2 schema since 6ac3dcf (was pred_lgbm)
    args = ap.parse_args()
    df = load(args.preds, args.model_col)
    baselines = [c for c in ["persistence", "hour_of_day_mean", "hour_of_week_mean", "scheduled_x_ratio"] if c in df]
    slices = {"all hours": np.ones(len(df), bool),
              "hours with scheduled arrivals": (df["sched_arr_count"] > 0).to_numpy(),
              "severe wx at hour t": df["oracle_coco"].isin(SEVERE_CODES).to_numpy()}
    rng = np.random.default_rng(0)
    rows = []
    for sl, mask in slices.items():
        for (icao, h), g in df[mask].groupby(["icao", "h"]):
            row = {"slice": sl, "icao": icao, "h": h, "n_hours": len(g), "mean_actual": g["arr_count"].mean(),
                   "rmse_model": rmse(g["arr_count"], g["model"])}
            for b in baselines:
                row[f"rmse_{b}"] = rmse(g["arr_count"], g[b])
                row[f"impr_vs_{b}_pct"] = 100 * (1 - row["rmse_model"] / row[f"rmse_{b}"])
            lo, hi = boot_diff(g, "scheduled_x_ratio", rng)
            row["diff_vs_sched_ci90_lo"], row["diff_vs_sched_ci90_hi"] = lo, hi
            y = g["arr_count"].to_numpy(float)
            if "p10" in g:
                row.update(band_stats(y, g["p10"].to_numpy(float), g["p90"].to_numpy(float), "band_model"))
            row.update(band_stats(y, g["sched_p10"].to_numpy(float), g["sched_p90"].to_numpy(float), "band_sched"))
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "aar_model_vs_baseline.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["slice", "icao", "h", "n_hours", "rmse_model", "rmse_scheduled_x_ratio", "rmse_hour_of_week_mean",
            "rmse_persistence", "impr_vs_scheduled_x_ratio_pct", "impr_vs_hour_of_week_mean_pct",
            "diff_vs_sched_ci90_lo", "diff_vs_sched_ci90_hi"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    band_cols = [c for c in res.columns if c.startswith("band_")]
    print("\np10-p90 band calibration (nominal coverage 0.80, 0.10 below / 0.10 above):")
    print(res[["slice", "icao", "h", "n_hours"] + band_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
