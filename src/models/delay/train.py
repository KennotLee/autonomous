"""P3 delay-risk model on P1's flight_features.parquet (label_45 = ARR_DELAY > 45).

    python -m src.models.delay.train

One LightGBM per horizon h (P1 rule: one h per model). Split on scheduled
departure: train < 2024-07, validation 2024-07..12 (early stopping, threshold),
test 2025-01..07. Cancelled/diverted flights (label NaN) are not trained or
scored but still get a prediction.

Outputs
  outputs/predictions/delay_preds.csv   flight_id, sched_time_utc, h, p_delay45 (contract)
                                         + high_risk, top_drivers, label_45, split
  outputs/metrics/delay_metrics.md, delay_calibration.png, delay_pr_curve.png
  outputs/metrics/delay_failure_case.md (+ .png)
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from . import features as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PRED_DIR, MET_DIR = ROOT / "outputs" / "predictions", ROOT / "outputs" / "metrics"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
CATEGORICAL = ["hub", "carrier"]
MODELS = {"gbm": {}, "gbm_p1_only": {"p1_only": True}, "gbm_oracle": {"oracle": True}}


def split_of(t: pd.Series) -> pd.Series:
    return pd.Series(np.select([t < "2024-07-01", t < "2025-01-01"], ["train", "val"], "test"), index=t.index)


def X(df, cols, cats):
    x = df[cols].copy()
    for c in CATEGORICAL:
        if c in x:
            x[c] = x[c].astype(cats[c])
    return x


def fit(tr, va, cols, cats):
    m = lgb.LGBMClassifier(n_estimators=3000, **PARAMS)
    m.fit(X(tr, cols, cats), tr.label_45, eval_set=[(X(va, cols, cats), va.label_45)],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


# ------------------------------------------------------------------ baselines
def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Bad weather at either end as of t - h: thunderstorm/heavy-precip code,
    gust > 25 kt (3 h max), or sustained wind > 20 kt (3 h max). Gusts are
    missing ~97% of hours, so the sustained-wind clause does most of the work."""
    hit = np.zeros(len(df), bool)
    for end in ("origin", "dest"):
        hit |= df[f"{end}_ts"].fillna(0).gt(0).to_numpy() | df[f"{end}_heavy_precip"].fillna(0).gt(0).to_numpy()
        hit |= df[f"{end}_wpgt_max_3h_kt"].fillna(0).gt(25).to_numpy()
        hit |= df[f"{end}_wspd_max_3h_kt"].fillna(0).gt(20).to_numpy()
    return hit.astype(float)


def climatology(tr: pd.DataFrame, df: pd.DataFrame, m=30) -> np.ndarray:
    """Train rate by hub x direction x local arrival hour, shrunk to the
    overall rate (cells are small in the sample)."""
    keys = ["hub", "arr_to_hub", "arr_hour_local"]
    t = tr.drop_duplicates("flight_id")
    g = t.groupby(keys).label_45.agg(["sum", "count"])
    base = t.label_45.mean()
    rate = ((g["sum"] + m * base) / (g["count"] + m)).rename("clim").reset_index()
    return df[keys].merge(rate, on=keys, how="left").clim.fillna(base).to_numpy()


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
    """Max F1 on validation: a missed severe delay and a false alert weigh the
    same, which suits an advisory list the coordinator scans each hour."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
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


def calibration_plot(y, preds, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    for name, p in preds.items():
        df = pd.DataFrame({"b": np.minimum((p * 10).astype(int), 9), "p": p, "y": y})
        df = df.groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        df = df[df.n >= 30]
        ax.plot(df.p, df.y, marker="o", label=name)
    ax.set(xlabel="predicted P(delay > 45 min)", ylabel="observed share", xlim=(0, 1), ylim=(0, 1),
           title="Calibration, test 2025 (bins with >= 30 flights)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_plot(y, preds, point, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.2f})")
    ax.axhline(y.mean(), color="#999", ls="--", lw=1, label=f"base rate {y.mean():.2f}")
    ax.plot(point[1], point[0], "k*", ms=12, label="chosen threshold")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, test 2025, h = 6")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md(df):
    return df.round(3).to_markdown()


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)
    fl = F.add_hub_features(F.load_p1())
    fl["split"] = split_of(fl.sched_time_utc)
    cats = {c: pd.CategoricalDtype(sorted(fl[c].dropna().unique())) for c in CATEGORICAL}
    lab = fl[fl.label_45.notna()]
    print(lab.drop_duplicates("flight_id").split.value_counts().to_dict())

    # one model per (variant, h)
    fitted, preds = {}, {}
    for name, kw in MODELS.items():
        cols = F.feature_cols(fl, **kw)
        p = np.full(len(fl), np.nan)
        for h in F.HORIZONS:
            d = lab[lab.h == h]
            m = fit(d[d.split == "train"], d[d.split == "val"], cols, cats)
            fitted[name, h] = (m, cols)
            rows = (fl.h == h).to_numpy()
            p[rows] = m.predict_proba(X(fl[rows], cols, cats))[:, 1]
        preds[name] = p
        print(f"{name}: {len(cols)} features, trees per h "
              f"{[fitted[name, h][0].best_iteration_ for h in F.HORIZONS]}")
    tr_rows = lab[lab.split == "train"]
    preds["climatology"] = climatology(tr_rows, fl)
    preds["rule"] = rule_score(fl)
    preds["base_rate"] = np.full(len(fl), tr_rows.drop_duplicates("flight_id").label_45.mean())
    fl = fl.assign(**{f"p_{k}": v for k, v in preds.items()})

    va = fl[(fl.split == "val") & fl.label_45.notna()]
    thr, vp, vr = choose_threshold(va.label_45.to_numpy(), va.p_gbm.to_numpy())
    print(f"threshold {thr:.3f} (val precision {vp:.2f}, recall {vr:.2f})")

    te = fl[(fl.split == "test") & fl.label_45.notna()]
    y = te.label_45.to_numpy()
    order = ["base_rate", "rule", "climatology", "gbm_p1_only", "gbm", "gbm_oracle"]
    overall = pd.DataFrame({n: scores(y, te[f"p_{n}"].to_numpy(),
                                      None if n == "base_rate" else (0.5 if n == "rule" else thr))
                            for n in order}).T
    by_h = pd.DataFrame({h: {n: average_precision_score(y[te.h == h], te[f"p_{n}"][te.h == h])
                             for n in order if n != "base_rate"} | {"base_rate": y[te.h == h].mean()}
                         for h in F.HORIZONS}).T
    by_hub = pd.DataFrame({hb: scores(y[te.hub == hb], te.p_gbm[te.hub == hb].to_numpy(), thr)
                           | {"base_rate": y[te.hub == hb].mean(),
                              "climatology_pr_auc": average_precision_score(y[te.hub == hb], te.p_climatology[te.hub == hb])}
                           for hb in sorted(te.hub.unique())}).T
    tradeoff = pd.DataFrame({t: scores(y, te.p_gbm.to_numpy(), t)
                             for t in sorted({0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, round(thr, 3)})}).T

    calibration_plot(y, {n: te[f"p_{n}"].to_numpy() for n in ("climatology", "gbm", "gbm_oracle")},
                     MET_DIR / "delay_calibration.png")
    t6 = te[te.h == 6]
    s6 = scores(t6.label_45.to_numpy(), t6.p_gbm.to_numpy(), thr)
    pr_plot(t6.label_45.to_numpy(), {n: t6[f"p_{n}"].to_numpy() for n in ("climatology", "rule", "gbm_p1_only", "gbm", "gbm_oracle")},
            (s6["precision"], s6["recall"]), MET_DIR / "delay_pr_curve.png")

    m6, cols6 = fitted["gbm", 6]
    imp = pd.Series(m6.booster_.feature_importance("gain"), index=cols6)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15).to_frame("gain_share (h=6)")

    n_flights = te.flight_id.nunique()
    with open(MET_DIR / "delay_metrics.md", "w") as f:
        f.write("# P3 delay risk: P(arrival delay > 45 min) = P1 `label_45`\n\n")
        f.write(f"Data: P1 `flight_features.parquet` (team CSV flights touching ATL/DFW). "
                f"Test = scheduled departure 2025-01..07: {n_flights:,} flights x {len(F.HORIZONS)} horizons, "
                f"base rate {y.mean():.3f}. Cancelled/diverted (label NaN) excluded from scoring. "
                f"One model per h. Threshold {thr:.3f} = max F1 on validation (2024 H2; precision "
                f"{vp:.2f}, recall {vr:.2f}). Rule baseline is binary (scored at 0.5).\n\n"
                f"`gbm_p1_only` = P1 features only; `gbm` adds hub schedule congestion + hub state "
                f"(full BTS, as of t - h); `gbm_oracle` adds P1 same-hour `oracle_*` weather.\n\n")
        f.write("## Overall (all horizons)\n\n" + md(overall) + "\n\n")
        f.write("## PR-AUC by horizon\n\n" + md(by_h) + "\n\n")
        f.write("## By hub (gbm)\n\n" + md(by_hub) + "\n\n")
        f.write("## Threshold trade-off (gbm, test)\n\n" + md(tradeoff) + "\n\n")
        f.write("## Top features (gain share)\n\n" + md(imp) + "\n")
    print(open(MET_DIR / "delay_metrics.md").read())

    # deliverable: every P1 flight row (contract columns first)
    drivers = pd.Series("", index=fl.index)
    for h in F.HORIZONS:
        m, cols = fitted["gbm", h]
        rows = fl.index[fl.h == h]
        drivers.loc[rows] = top_drivers(m, X(fl.loc[rows], cols, cats))
    out = fl.assign(p_delay45=fl.p_gbm.round(4), high_risk=(fl.p_gbm >= thr).astype(int), top_drivers=drivers)
    out[["flight_id", "sched_time_utc", "h", "p_delay45", "high_risk", "top_drivers", "label_45", "split"]].to_csv(
        PRED_DIR / "delay_preds.csv", index=False)
    print(f"wrote {len(out):,} rows to {PRED_DIR / 'delay_preds.csv'}")

    from .failure import write_failure_case
    write_failure_case(fl.assign(top_drivers=drivers), thr, MET_DIR)


if __name__ == "__main__":
    main()
