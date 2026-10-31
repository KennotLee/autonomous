"""P4 Task 2: score a delay model's event predictions against the best P4 baseline (slide table).

Input: a CSV with event_id, h, risk_prob[, top_drivers] (team schema on P1 event_features v2).
Scored on the same test events as baselines_events.py (FL_DATE 2025, cancelled/diverted dropped);
model rows outside that set are ignored, test events the model is missing are reported.

Best baseline = the baseline with the highest PR-AUC on ALL test events at that h. Choosing it on test
makes the bar harder for the model, never easier.
Recall @ precision >= 0.30: largest recall reachable at a threshold whose precision is at least 0.30.
90% CI: bootstrap over whole FL_DATE days (events on the same day share weather, so they are not independent).
Calibration: 5 equal-count bins of predicted risk, mean predicted vs observed rate.

Usage: python src/eval/evaluate_events.py --preds data/interim/p3_delay_preds_v1.csv --name p3_v1
           [--events data/processed/event_features.parquet]
    -> outputs/metrics/event_model_vs_baseline_<name>.csv, outputs/metrics/event_calibration_<name>.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from baselines import OUT, PROCESSED, ROOT, recall_at_precision
from baselines_events import MODELS, fit_predict, load

MIN_PRECISION = 0.30
N_BOOT = 300
N_BINS = 5


def test_frame(events_path, preds_path):
    ev = fit_predict(load(events_path))
    te = ev[ev["split"] == "test"][["event_id", "h", "fl_date", "apt", "event_type", "adverse_t3", "y"] + MODELS]
    p = pd.read_csv(preds_path, usecols=["event_id", "h", "risk_prob"])
    df = te.merge(p, on=["event_id", "h"], how="left")
    miss = df["risk_prob"].isna()
    print(f"model rows {len(p)}; test event x h rows {len(te)}; missing model preds {miss.sum()} "
          f"({miss.mean():.2%}); model rows outside test set {len(p) - (len(te) - miss.sum())}")
    return df[~miss].rename(columns={"risk_prob": "model"})


def day_boot_diff(g, base, rng):
    """90% CI of PR-AUC(model) - PR-AUC(baseline), resampling whole days."""
    day = g["fl_date"].to_numpy()
    days, inv = np.unique(day, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    y, m, b = g["y"].to_numpy(), g["model"].to_numpy(), g[base].to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        if y[idx].any():
            diffs.append(average_precision_score(y[idx], m[idx]) - average_precision_score(y[idx], b[idx]))
    return np.percentile(diffs, [5, 95])


def slices(g):
    yield "ALL", np.ones(len(g), bool)
    for a in sorted(g["apt"].unique()):
        yield a, (g["apt"] == a).to_numpy()
    for d in ("arr", "dep"):
        yield d, (g["event_type"] == d).to_numpy()
    yield "adverse wx t-3h", g["adverse_t3"].to_numpy()


def compare(df):
    rng = np.random.default_rng(0)
    rows = []
    for h, g in df.groupby("h"):
        best = max(MODELS, key=lambda m: average_precision_score(g["y"], g[m]))
        for sl, mask in slices(g):
            s = g[mask]
            y = s["y"].to_numpy()
            lo, hi = day_boot_diff(s, best, rng)
            r = {"h": h, "slice": sl, "n": len(s), "positives": int(y.sum()), "best_baseline": best}
            for col, name in [("model", "model"), (best, "base")]:
                r[f"pr_auc_{name}"] = average_precision_score(y, s[col])
                r[f"recall_at_p{MIN_PRECISION:.2f}_{name}"] = recall_at_precision(y, s[col], MIN_PRECISION)
                r[f"brier_{name}"] = brier_score_loss(y, s[col])
            r["pr_auc_impr_pct"] = 100 * (r["pr_auc_model"] / r["pr_auc_base"] - 1)
            r["pr_auc_diff_ci90_lo"], r["pr_auc_diff_ci90_hi"] = lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def calibration(df):
    rows = []
    for h, g in df.groupby("h"):
        best = max(MODELS, key=lambda m: average_precision_score(g["y"], g[m]))
        for name, col in [("model", "model"), (best, best)]:
            bins = pd.qcut(g[col].rank(method="first"), N_BINS, labels=False)
            for b, s in g.groupby(bins):
                rows.append({"h": h, "predictor": name, "bin": int(b) + 1, "n": len(s), "positives": int(s["y"].sum()),
                             "mean_pred": s[col].mean(), "observed_rate": s["y"].mean(),
                             "pred_min": s[col].min(), "pred_max": s[col].max()})
    return pd.DataFrame(rows)


def evaluate(preds_path, name, events_path=PROCESSED / "event_features.parquet"):
    df = test_frame(events_path, preds_path)
    res, cal = compare(df), calibration(df)
    res.to_csv(OUT / f"event_model_vs_baseline_{name}.csv", index=False)
    cal.to_csv(OUT / f"event_calibration_{name}.csv", index=False)

    rc = f"recall_at_p{MIN_PRECISION:.2f}"
    t = res[res["slice"].isin(["ALL", "ATL", "DFW"])].copy()
    t["PR-AUC model"] = t["pr_auc_model"].map("{:.3f}".format)
    t["PR-AUC best base"] = t["best_baseline"].str[:2] + " " + t["pr_auc_base"].map("{:.3f}".format)
    t["vs base"] = t["pr_auc_impr_pct"].map("{:+.0f}%".format)
    t["90% CI diff"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(t["pr_auc_diff_ci90_lo"], t["pr_auc_diff_ci90_hi"])]
    t[f"recall@P>={MIN_PRECISION:.2f} model / base"] = [f"{a:.3f} / {b:.3f}" for a, b in zip(t[f"{rc}_model"], t[f"{rc}_base"])]
    t["Brier model / base"] = [f"{a:.3f} / {b:.3f}" for a, b in zip(t["brier_model"], t["brier_base"])]
    cols = ["h", "slice", "n", "positives", "PR-AUC model", "PR-AUC best base", "vs base", "90% CI diff",
            f"recall@P>={MIN_PRECISION:.2f} model / base", "Brier model / base"]
    print(f"\n== {name} vs best P4 baseline (2025 test events)")
    print(t[cols].to_string(index=False))
    other = res[~res["slice"].isin(["ALL", "ATL", "DFW"])]
    print("\n== other slices (PR-AUC model vs best baseline, 90% CI of diff)")
    print(other[["h", "slice", "n", "positives", "pr_auc_model", "pr_auc_base", "pr_auc_impr_pct",
                 "pr_auc_diff_ci90_lo", "pr_auc_diff_ci90_hi"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== calibration, h=3 (5 equal-count bins: mean predicted vs observed rate)")
    c3 = cal[cal["h"] == 3]
    print(c3.pivot(index="bin", columns="predictor", values=["n", "mean_pred", "observed_rate"])
          .to_string(float_format=lambda x: f"{x:.3f}"))
    return res, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--events", type=Path, default=PROCESSED / "event_features.parquet")
    args = ap.parse_args()
    evaluate(args.preds, args.name, args.events)


if __name__ == "__main__":
    main()
