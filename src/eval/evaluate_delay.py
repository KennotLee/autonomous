"""P4: score a delay model's predictions against P4 baselines on identical 2025 test flights.

Usage:
    python src/eval/evaluate_delay.py --preds outputs/predictions/delay_preds.csv [--name gbm]

Reads the model CSV (flight_id, h, p_delay45[, split]), P4 baseline preds
(outputs/metrics/delay_baseline_preds.csv, from baselines.py) and P1's flight_features.parquet
(label_45, hub ends, oracle_*_coco for slicing only).

Writes outputs/metrics/delay_model_vs_baseline_<name>.csv: PR-AUC / Brier per h x slice for the model and
each baseline, and a 90% bootstrap CI (resampling flights) on PR-AUC(model) - PR-AUC(best baseline).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from baselines import HORIZONS, OUT, PROCESSED, TEST_START, delay_slices, load_airports

BEST_BASELINE = "base_rate_origin_dephour_month"
N_BOOT = 500


def load(preds_path):
    m = pd.read_csv(preds_path, usecols=lambda c: c in {"flight_id", "h", "p_delay45", "split"})
    if "split" in m:
        m = m[m["split"] == "test"]
    m = m.rename(columns={"p_delay45": "model"}).drop(columns="split", errors="ignore")
    base = pd.read_csv(OUT / "delay_baseline_preds.csv", usecols=["flight_id", "h", "p_delay45", "model"])
    base = base.pivot_table(index=["flight_id", "h"], columns="model", values="p_delay45").reset_index()
    fl = pd.read_parquet(PROCESSED / "flight_features.parquet",
                         columns=["flight_id", "h", "sched_time_utc", "origin_icao", "dest_icao", "label_45",
                                  "oracle_origin_coco", "oracle_dest_coco"])
    fl = fl[(fl["sched_time_utc"] >= TEST_START) & fl["label_45"].notna()]
    df = fl.merge(m, on=["flight_id", "h"], how="inner").merge(base, on=["flight_id", "h"], how="inner")
    print(f"test flight x h rows: P1 {len(fl)}, model {len(m)}, joined {len(df)}")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("outputs/predictions/delay_preds.csv"))
    ap.add_argument("--name", default="model")
    args = ap.parse_args()
    airports = load_airports()
    df = load(args.preds)
    baselines = [c for c in ["base_rate", "base_rate_origin_dephour_month", "base_rate_dest_arrhour_month"] if c in df]
    rng = np.random.default_rng(0)
    rows = []
    for h in HORIZONS:
        d = df[df["h"] == h].reset_index(drop=True)
        for sl, mask in delay_slices(d, airports):
            g = d[mask]
            y = g["label_45"].to_numpy(int)
            if y.sum() == 0:
                continue
            row = {"h": h, "slice": sl, "n": len(g), "positives": int(y.sum()), "pos_rate": y.mean(),
                   "pr_auc_model": average_precision_score(y, g["model"]), "brier_model": brier_score_loss(y, g["model"])}
            for b in baselines:
                row[f"pr_auc_{b}"] = average_precision_score(y, g[b])
                row[f"brier_{b}"] = brier_score_loss(y, g[b])
            row["pr_auc_impr_vs_best_pct"] = 100 * (row["pr_auc_model"] / row[f"pr_auc_{BEST_BASELINE}"] - 1)
            diffs = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(g), len(g))
                if y[i].sum():
                    diffs.append(average_precision_score(y[i], g["model"].to_numpy()[i])
                                 - average_precision_score(y[i], g[BEST_BASELINE].to_numpy()[i]))
            row["diff_vs_best_ci90_lo"], row["diff_vs_best_ci90_hi"] = np.percentile(diffs, [5, 95])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"delay_model_vs_baseline_{args.name}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["h", "slice", "n", "positives", "pr_auc_model", f"pr_auc_{BEST_BASELINE}", "pr_auc_base_rate",
            "pr_auc_impr_vs_best_pct", "diff_vs_best_ci90_lo", "diff_vs_best_ci90_hi", "brier_model",
            f"brier_{BEST_BASELINE}"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
