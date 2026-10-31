"""P3 delay-risk model: train, evaluate against baselines, write deliverables.

    python -m src.models.delay.train

Splits by scheduled arrival (local): train 2023-01..2024-06, validation
2024-07..2024-12 (early stopping + threshold choice), test 2025-01..2025-07.
Training/eval flights are a seeded random sample of all BTS ATL/DFW flights;
predictions are also written for every ATL/DFW flight in the team CSV.

Outputs
  outputs/predictions/delay_preds.csv       flight_id, h, risk_prob, high_risk, top_drivers (+ context)
  outputs/metrics/delay_metrics.md          tables
  outputs/metrics/delay_calibration.png, delay_pr_curve.png
  outputs/metrics/delay_failure_case.md     (+ .png) auto-selected worst miss
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from pipeline.panel import load_flights

from . import features as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PRED_DIR, MET_DIR = ROOT / "outputs" / "predictions", ROOT / "outputs" / "metrics"
TEAM_CSV = [ROOT / "data" / "raw" / "Flight_Data_With_Weather_Final.csv",
            ROOT / "data" / "original_datasets" / "Flight_Data_With_Weather_Final.csv"]
SAMPLE = {"train": 300_000, "val": 100_000, "test": 200_000}
SEED = 7
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
NON_FEATURES = {"flight_id", "ORIGIN", "DEST", "sched_dep_utc", "sched_arr_utc", "sched_arr",
                "ARR_DELAY", "label", "issue_time_utc", "feature_obs_time", "hub_hour_utc",
                "split", "team"}
CATEGORICAL = ["hub", "OP_UNIQUE_CARRIER"]


def split_of(t: pd.Series) -> pd.Series:
    return pd.Series(np.select([t < "2024-07-01", t < "2025-01-01"], ["train", "val"], "test"), index=t.index)


def team_flight_ids() -> set[str]:
    path = next((p for p in TEAM_CSV if p.exists()), None)
    if path is None:
        print("team CSV not found; delay_preds.csv will be empty")
        return set()
    return set(F.flight_id(load_flights(path)))


def population(flights: pd.DataFrame, team_ids: set[str]) -> pd.DataFrame:
    f = flights[flights.label.notna()].copy()
    f["split"] = split_of(f.sched_arr)
    f["team"] = f.flight_id.isin(team_ids)
    rng = np.random.default_rng(SEED)
    keep = f.team.to_numpy().copy()
    for s, n in SAMPLE.items():
        idx = np.flatnonzero((f.split == s).to_numpy())
        keep[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    return f[keep]


def feature_cols(ds, oracle):
    cols = [c for c in ds.columns if c not in NON_FEATURES]
    return cols if oracle else [c for c in cols if not c.startswith("now_")]


def X(ds, cols):
    x = ds[cols].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype(pd.CategoricalDtype(sorted(ds[c].dropna().unique())))
    return x


def fit(train, val, cols):
    m = lgb.LGBMClassifier(n_estimators=2000, **PARAMS)
    m.fit(X(train, cols), train.label, eval_set=[(X(val, cols), val.label)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


# ------------------------------------------------------------------ baselines
def rule_score(ds: pd.DataFrame) -> pd.Series:
    """'Bad weather at either end in the 3 h before issue': thunderstorm or
    heavy-precip code, gust > 25 kt, or sustained wind > 20 kt. Gusts are
    missing ~97% of hours in Meteostat, so the wind clause uses sustained speed too."""
    hit = pd.Series(False, index=ds.index)
    for end in ("origin", "dest"):
        hit |= ds[f"{end}_ts_3h"].fillna(0).gt(0) | ds[f"{end}_heavy_precip"].fillna(0).gt(0)
        hit |= ds[f"{end}_gust_max_3h"].fillna(0).gt(25) | ds[f"{end}_wspd_kt"].fillna(0).gt(20)
    return hit.astype(float)


def climatology(train, ds):
    keys = ["hub", "arr_to_hub", "hour_local", "month"]
    rate = train.drop_duplicates("flight_id").groupby(keys).label.mean().rename("clim").reset_index()
    return ds[keys].merge(rate, on=keys, how="left").clim.fillna(train.label.mean()).to_numpy()


# -------------------------------------------------------------------- metrics
def scores(y, p, thr=None):
    out = {"pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, np.clip(p, 0, 1))}
    if thr is not None:
        flag = p >= thr
        tp = (flag & (y == 1)).sum()
        out |= {"precision": tp / max(flag.sum(), 1), "recall": tp / max((y == 1).sum(), 1),
                "alert_rate": flag.mean()}
    return out


def choose_threshold(y, p):
    """Max F1 on validation. F1 weights a missed severe delay and a false alert
    equally, which suits an advisory list a coordinator scans each hour."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = np.nanargmax(f1[:-1])
    return float(thr[i]), float(prec[i]), float(rec[i])


def top_drivers(model, Xd: pd.DataFrame, k=3) -> list[str]:
    """Top-k features pushing risk UP (TreeSHAP contributions, log-odds)."""
    contrib = model.booster_.predict(Xd, pred_contrib=True)[:, :-1]
    names = np.array(Xd.columns)
    order = np.argsort(-contrib, axis=1)[:, :k]
    out = []
    for r in range(len(Xd)):
        parts = []
        for j in order[r]:
            if contrib[r, j] <= 0:
                break
            v = Xd.iat[r, j]
            v = f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
            parts.append(f"{names[j]}={v} (+{contrib[r, j]:.2f})")
        out.append("; ".join(parts))
    return out


def calibration_plot(y, preds: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    for name, p in preds.items():
        bins = np.minimum((p * 10).astype(int), 9)
        df = pd.DataFrame({"b": bins, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        df = df[df.n >= 50]
        ax.plot(df.p, df.y, marker="o", label=name)
    ax.set(xlabel="predicted P(delay > 45 min)", ylabel="observed share", title="Calibration, test 2025",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_plot(y, preds: dict, thr_point, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.2f})")
    ax.axhline(y.mean(), color="#999", ls="--", lw=1, label=f"base rate {y.mean():.2f}")
    ax.plot(thr_point[1], thr_point[0], "k*", ms=12, label="chosen threshold")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, test 2025")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    return df.round(3).to_markdown()


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)
    print("loading flights ...")
    flights = F.load_hub_flights()
    pop = population(flights, team_flight_ids())
    print(f"population: {pop.split.value_counts().to_dict()}, team flights {pop.team.sum():,}")
    ds = F.build(pop, flights)
    ds = ds.merge(pop[["flight_id", "split", "team"]], on="flight_id")
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)

    tr, va = ds[ds.split == "train"], ds[ds.split == "val"]
    te = ds[ds.split == "test"]

    models = {}
    for name, oracle in (("gbm", False), ("gbm_oracle", True)):
        cols = feature_cols(ds, oracle)
        models[name] = (fit(tr, va, cols), cols)
        print(f"{name}: {models[name][0].best_iteration_} trees, {len(cols)} features")

    def predict(df):
        p = {"base_rate": np.full(len(df), tr.label.mean()),
             "rule": rule_score(df).to_numpy(),
             "climatology": climatology(tr, df)}
        for name, (m, cols) in models.items():
            p[name] = m.predict_proba(X(df, cols))[:, 1]
        return p

    thr, vp, vr = choose_threshold(va.label.to_numpy(), predict(va)["gbm"])
    print(f"threshold {thr:.3f} (val precision {vp:.2f}, recall {vr:.2f})")

    pt = predict(te)
    y = te.label.to_numpy()
    rows = {}
    for name, p in pt.items():
        t = 0.5 if name == "rule" else (None if name == "base_rate" else thr)
        rows[name] = scores(y, p, t)
    overall = pd.DataFrame(rows).T
    by_h = pd.DataFrame({h: {f"{n}_pr_auc": average_precision_score(y[te.h == h], pt[n][te.h == h])
                             for n in ("climatology", "rule", "gbm", "gbm_oracle")}
                         | {"base_rate": y[te.h == h].mean()} for h in F.HORIZONS}).T
    by_hub = pd.DataFrame({hb: scores(y[te.hub == hb], pt["gbm"][te.hub == hb], thr)
                           | {"base_rate": y[te.hub == hb].mean(),
                              "clim_pr_auc": average_precision_score(y[te.hub == hb], pt["climatology"][te.hub == hb])}
                           for hb in sorted(te.hub.unique())}).T
    team_te = te.team.to_numpy()
    team = pd.DataFrame({n: scores(y[team_te], pt[n][team_te], thr if n.startswith("gbm") or n == "climatology" else None)
                         for n in ("climatology", "gbm")}).T
    # precision/recall trade-off table for the user to pick their own alert budget
    tradeoff = pd.DataFrame({t: scores(y, pt["gbm"], t) for t in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, round(thr, 3))}).T

    calibration_plot(y, {k: pt[k] for k in ("climatology", "gbm", "gbm_oracle")}, MET_DIR / "delay_calibration.png")
    h6 = (te.h == 6).to_numpy()
    s6 = scores(y[h6], pt["gbm"][h6], thr)
    pr_plot(y[h6], {k: pt[k][h6] for k in ("climatology", "rule", "gbm", "gbm_oracle")},
            (s6["precision"], s6["recall"]), MET_DIR / "delay_pr_curve.png")

    imp = pd.Series(models["gbm"][0].booster_.feature_importance("gain"), index=models["gbm"][1])
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15).to_frame("gain_share")

    with open(MET_DIR / "delay_metrics.md", "w") as f:
        f.write("# P3 delay risk: P(arrival delay > 45 min)\n\n")
        f.write(f"Test = sampled BTS ATL/DFW flights with scheduled arrival 2025-01..07 "
                f"({te.flight_id.nunique():,} flights x {len(F.HORIZONS)} horizons). "
                f"Cancelled/diverted flights excluded (no arrival delay). "
                f"Decision threshold {thr:.3f} = max F1 on validation (2024 H2; "
                f"precision {vp:.2f}, recall {vr:.2f}). Rule baseline is binary (scored at 0.5).\n\n")
        f.write("## Overall (all horizons)\n\n" + md(overall) + "\n\n")
        f.write("## PR-AUC by horizon\n\n" + md(by_h) + "\n\n")
        f.write("## By hub (gbm)\n\n" + md(by_hub) + "\n\n")
        f.write("## Team CSV flights in the test period\n\n" + md(team) + "\n\n")
        f.write("## Threshold trade-off (gbm, test)\n\n" + md(tradeoff) + "\n\n")
        f.write("## Top features (gain share)\n\n" + md(imp) + "\n")
    print(open(MET_DIR / "delay_metrics.md").read())

    # deliverable: predictions for the team CSV flights
    out = ds[ds.team].copy()
    m, cols = models["gbm"]
    Xo = X(out, cols)
    out["risk_prob"] = m.predict_proba(Xo)[:, 1].round(4)
    out["high_risk"] = (out.risk_prob >= thr).astype(int)
    out["top_drivers"] = top_drivers(m, Xo)
    out = out.rename(columns={"ORIGIN": "origin", "DEST": "dest", "label": "actual_delay_gt45"})
    out[["flight_id", "h", "risk_prob", "high_risk", "top_drivers", "origin", "dest",
         "sched_arr_utc", "issue_time_utc", "actual_delay_gt45", "split"]].to_csv(
        PRED_DIR / "delay_preds.csv", index=False)
    print(f"wrote {len(out):,} rows to {PRED_DIR / 'delay_preds.csv'}")

    te = te.assign(p_gbm=pt["gbm"], p_oracle=pt["gbm_oracle"])
    te.to_parquet(ROOT / "data" / "interim" / "delay_test_preds.parquet")
    from .failure import write_failure_case
    write_failure_case(te, thr, MET_DIR)


if __name__ == "__main__":
    main()
