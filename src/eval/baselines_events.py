"""P4 Task 1 (organiser update): delay >45 min baselines on P1's event_features.parquet (v2).

One event = a flight leaving ATL/DFW (DEP_DELAY > 45) or landing at ATL/DFW (ARR_DELAY > 45).
Split: train FL_DATE 2023-2024, test FL_DATE 2025-01..2025-07 (FL_DATE = first 10 chars of event_id).
Cancelled / diverted / unlabelled events are dropped (counts printed).

Baselines (all fit on train only):
  B0  constant train rate
  B1  rate per airport x direction x scheduled local hour, smoothed toward B0 (m pseudo-events)
  B2  rate per airport x direction x hour x month, smoothed toward B1
  B3  B2 x k, k = train observed/expected for events with adverse vs normal weather at t-h
      (the weather row P1 attached at as-of <= t-h). Adverse = coco fog/rain/thunderstorm (5-9, 25-27)
      or crosswind >= 35 km/h or tailwind >= 15 km/h (organiser definition).
Top-15% threshold: flag the 15% highest-risk test events per model x h (ties broken at random, seeded),
then precision / recall per slice. Slice "adverse/normal wx at t-3h" always uses the h=3 weather row.

Usage: python src/eval/baselines_events.py [--events data/processed/event_features.parquet]
    -> outputs/metrics/event_baseline_metrics.csv, outputs/predictions/event_baseline_preds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from baselines import OUT, PROCESSED, ROOT, SMOOTHING_M

ADVERSE_CODES = {5, 6, 7, 8, 9, 25, 26, 27}
CROSSWIND_KMH, TAILWIND_KMH = 35, 15
TOP_SHARE = 0.15
MODELS = ["B0_const", "B1_apt_dir_hour", "B2_apt_dir_hour_month", "B3_B2_x_adverse_wx"]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def adverse(df):
    code = df["coco"].isin(ADVERSE_CODES)
    cross = df["crosswind"].abs() >= CROSSWIND_KMH
    tail = df["tailwind"] >= TAILWIND_KMH
    reason = np.select([code, cross, tail], ["wx code " + df["coco"].astype("Int64").astype(str),
                                             "crosswind " + df["crosswind"].round().astype("Int64").astype(str) + " km/h",
                                             "tailwind " + df["tailwind"].round().astype("Int64").astype(str) + " km/h"], "")
    return (code | cross | tail).to_numpy(), reason


def load(path):
    ev = pd.read_parquet(path)
    ev["fl_date"] = ev["event_id"].str[:10]
    ev["split"] = np.where(ev["fl_date"] < "2025-01-01", "train", "test")
    one = ev[ev["h"] == 3]
    drop = (one["cancelled"] == 1) | (one["diverted"] == 1) | one["label_45"].isna()
    print(f"events: {len(one)}; dropped {drop.sum()} ({drop.mean():.1%}): cancelled {int((one.cancelled == 1).sum())}, "
          f"diverted {int(((one.diverted == 1) & (one.cancelled != 1)).sum())}, other unlabelled "
          f"{int((one.label_45.isna() & (one.cancelled != 1) & (one.diverted != 1)).sum())}")
    ev = ev[ev["event_id"].isin(one.loc[~drop, "event_id"])].copy()
    ev["y"] = ev["label_45"].astype(int)
    ev["apt"] = ev["airport"].str[1:]
    ev["adverse"], ev["adverse_reason"] = adverse(ev)
    adv3 = ev.loc[ev["h"] == 3].set_index("event_id")["adverse"]
    ev["adverse_t3"] = ev["event_id"].map(adv3).fillna(False).astype(bool)
    return ev


def smoothed(train, keys, prior_col, m=SMOOTHING_M):
    """Cell rate (pos + m * prior) / (n + m), prior = the cell's coarser-baseline prediction."""
    agg = train.groupby(keys).agg(pos=("y", "sum"), n=("y", "size"), prior=(prior_col, "first"))
    return ((agg["pos"] + m * agg["prior"]) / (agg["n"] + m)).rename("rate")


def fit_predict(ev):
    tr = ev[(ev["split"] == "train") & (ev["h"] == 3)]
    g = tr["y"].mean()
    ev["B0_const"] = g
    tr = tr.assign(B0_const=g)
    k1 = ["airport", "event_type", "hour"]
    r1 = smoothed(tr, k1, "B0_const")
    ev["B1_apt_dir_hour"] = ev.join(r1, on=k1)["rate"].fillna(g)
    tr = tr.join(r1, on=k1).rename(columns={"rate": "B1_apt_dir_hour"})
    k2 = k1 + ["month"]
    r2 = smoothed(tr, k2, "B1_apt_dir_hour")
    ev["B2_apt_dir_hour_month"] = ev.join(r2, on=k2)["rate"].fillna(ev["B1_apt_dir_hour"])
    # B3 multipliers per h (the weather row moves with h), fit on train only
    trh = ev[ev["split"] == "train"]
    k = trh.groupby(["h", "adverse"]).apply(lambda d: d["y"].sum() / d["B2_apt_dir_hour_month"].sum(),
                                             include_groups=False).rename("k")
    ev["k_wx"] = ev.join(k, on=["h", "adverse"])["k"]
    ev["B3_B2_x_adverse_wx"] = (ev["B2_apt_dir_hour_month"] * ev["k_wx"]).clip(0, 1)
    print(f"train rate B0 = {g:.4f}; B3 multipliers (h, adverse) -> k:\n{k.unstack().round(3).to_string()}")
    return ev


def drivers(ev, model):
    base = ev["apt"] + " " + ev["event_type"] + " " + ev["hour"].astype(str).str.zfill(2) + ":00"
    if model == "B0_const":
        return "train base rate " + ev["B0_const"].round(3).astype(str)
    if model == "B1_apt_dir_hour":
        return base + " hist rate " + ev[model].round(3).astype(str)
    mon = base + " " + ev["month"].map(lambda m: MONTHS[m - 1])
    if model == "B2_apt_dir_hour_month":
        return mon + " hist rate " + ev[model].round(3).astype(str)
    wx = np.where(ev["adverse"], "ADVERSE wx at t-" + ev["h"].astype(str) + "h (" + ev["adverse_reason"] + ")",
                  "normal wx at t-" + ev["h"].astype(str) + "h")
    return (mon + " hist rate " + ev["B2_apt_dir_hour_month"].round(3).astype(str) + "; " + wx
            + " x" + ev["k_wx"].round(2).astype(str) + " (obs " + ev["feature_obs_time"].dt.strftime("%H:%MZ") + ")")


def top_flag(p, share=TOP_SHARE, seed=0):
    """Flag exactly the top `share` of rows; random tie-break so a constant model flags a random 15%."""
    jitter = np.random.default_rng(seed).random(len(p)) * 1e-9
    order = np.argsort(-(p + jitter), kind="stable")
    flag = np.zeros(len(p), bool)
    flag[order[: int(round(share * len(p)))]] = True
    return flag


def slices(te):
    out = {"ALL": np.ones(len(te), bool)}
    for a in sorted(te["apt"].unique()):
        out[a] = (te["apt"] == a).to_numpy()
    for d, name in [("arr", "arrivals"), ("dep", "departures")]:
        out[name] = (te["event_type"] == d).to_numpy()
    for a in sorted(te["apt"].unique()):
        for d in ("arr", "dep"):
            out[f"{a} {d}"] = ((te["apt"] == a) & (te["event_type"] == d)).to_numpy()
    out["adverse wx at t-3h"] = te["adverse_t3"].to_numpy()
    out["normal wx at t-3h"] = ~te["adverse_t3"].to_numpy()
    return out


def score(ev):
    rows = []
    for h, te in ev[ev["split"] == "test"].groupby("h"):
        y = te["y"].to_numpy()
        for model in MODELS:
            p = te[model].to_numpy(float)
            flag = top_flag(p)
            for sl, m in slices(te).items():
                ym, pm, fm = y[m], p[m], flag[m]
                rows.append({"h": h, "model": model, "slice": sl, "n": int(m.sum()), "positives": int(ym.sum()),
                             "pos_rate": ym.mean(), "pr_auc": average_precision_score(ym, pm) if ym.any() else np.nan,
                             "brier": brier_score_loss(ym, pm), "flagged": int(fm.sum()),
                             "precision_top15": ym[fm].mean() if fm.any() else np.nan,
                             "recall_top15": ym[fm].sum() / ym.sum() if ym.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=PROCESSED / "event_features.parquet")
    args = ap.parse_args()
    ev = fit_predict(load(args.events))
    te3 = ev[(ev["split"] == "test") & (ev["h"] == 3)]
    tr3 = ev[(ev["split"] == "train") & (ev["h"] == 3)]
    print(f"train events {len(tr3)} (pos {tr3.y.mean():.3f}); test events {len(te3)} (pos {te3.y.mean():.3f}); "
          f"adverse at t-3h: train {tr3.adverse.mean():.1%}, test {te3.adverse.mean():.1%}")
    res = score(ev)
    res.to_csv(OUT / "event_baseline_metrics.csv", index=False)

    test = ev[ev["split"] == "test"]
    preds = pd.concat([test[["event_id", "h"]].assign(risk_prob=test[m].round(4), top_drivers=drivers(test, m), model=m)
                       for m in MODELS])
    pred_dir = ROOT / "outputs/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_dir / "event_baseline_preds.csv", index=False)
    print(f"wrote {OUT / 'event_baseline_metrics.csv'} and {pred_dir / 'event_baseline_preds.csv'} ({len(preds)} rows)")

    f = lambda x: f"{x:.3f}"  # noqa: E731
    cols = ["model", "n", "positives", "pr_auc", "brier", "flagged", "precision_top15", "recall_top15"]
    r3 = res[res["h"] == 3]
    for sl in r3["slice"].unique():
        print(f"\n== h=3 | {sl}")
        print(r3[r3["slice"] == sl][cols].to_string(index=False, float_format=f))
    print("\n== PR-AUC, all test events, by h")
    print(res[res["slice"] == "ALL"].pivot(index="model", columns="h", values="pr_auc").to_string(float_format=f))


if __name__ == "__main__":
    main()
