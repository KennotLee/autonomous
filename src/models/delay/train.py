"""P3 delay-risk model on P1's event_features.parquet: P(label_45) per (event_id, h).

    python -m src.models.delay.train

One LightGBM for all horizons (h, airport, event_type are features). Split on
sched_time_utc: train 2023-01..2024-06, validation 2024-07..12 (early stopping,
threshold, max F1), test 2025. Events with label_45 NaN (cancelled, diverted
arrivals) are not trained or scored. No class reweighting, so probabilities stay
calibrated; the imbalance is handled by the threshold.

Output: outputs/predictions/delay_preds.csv = event_id, airport, event_type, sched_time_utc, h,
risk_prob, high_risk, top_drivers (schema: outputs/predictions/SCHEMA.md)
for every 2025 row of event_features (join coverage asserted 100%).
"""
import hashlib
import io
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "event_features.parquet"
PRED_DIR, MET_DIR = ROOT / "outputs" / "predictions", ROOT / "outputs" / "metrics"
MODEL_DIR = Path(__file__).resolve().parent / "artifacts"

KEYS = ["event_id", "flight_key", "sched_time_local", "sched_time_utc", "feature_obs_time", "other_feature_obs_time"]
TARGETS = ["delay_min", "label_45", "cancelled", "diverted"]
CATEGORICAL = ["event_type", "airport", "other_airport", "carrier"]
EVENT_WX = ["temp", "prcp", "wspd", "pres", "coco", "wdir", "headwind", "crosswind", "tailwind", "ts_flag",
            "snow_flag", "pres_tend_3h", "prcp_sum_6h"]  # + <v>_lag{1,3,6}h
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              random_state=0, verbose=-1)
TRAIN_END, VAL_END = pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")


def _p1_blob(path: str) -> bytes:
    return subprocess.run(["git", "show", f"origin/p1:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def _data_bytes() -> bytes:
    """P1's file; if it isn't checked out locally, read it from origin/p1."""
    return DATA.read_bytes() if DATA.exists() else _p1_blob("data/processed/event_features.parquet")


def load() -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_data_bytes()))


def data_md5() -> str:
    """Provenance: md5 of the event_features.parquet bytes the model was trained on."""
    return hashlib.md5(_data_bytes()).hexdigest()


def config() -> dict:
    """config/airports.yaml (P1 owns it). Until p1 is merged here, the local copy may predate
    the `weather` section; then read P1's version from origin/p1."""
    cfg = yaml.safe_load((ROOT / "config" / "airports.yaml").read_text())
    if "weather" not in cfg:
        cfg = yaml.safe_load(_p1_blob("config/airports.yaml"))
    return cfg


def iata_to_icao(cfg: dict) -> dict[str, str]:
    return {a["iata"]: icao for icao, a in cfg["airports"].items()}


def split_of(t: pd.Series) -> pd.Series:
    return pd.Series(np.select([t < TRAIN_END, t < VAL_END], ["train", "val"], "test"), index=t.index)


def feature_cols(ev: pd.DataFrame, oracle=False) -> list[str]:
    cols = [c for c in ev.columns if c not in KEYS + TARGETS + ["split"] and not c.startswith("oracle_")]
    if oracle:  # oracle model: same-time weather replaces all lagged weather at the event airport
        cols = [c for c in cols if c not in EVENT_WX and "_lag" not in c or c.startswith("other_")]
        cols += [c for c in ev.columns if c.startswith("oracle_")]
    bad = [c for c in cols if c in KEYS + TARGETS or (c.startswith("oracle_") and not oracle)]
    assert not bad, f"leaky columns in features: {bad}"
    return cols


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


def choose_threshold(y, p):
    """Max F1 on validation: a missed severe delay and a false alert weigh the same."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(prec[i]), float(rec[i])


def top_drivers(model, Xd: pd.DataFrame, k=3) -> list[str]:
    """Top-k features pushing risk up (LightGBM pred_contrib, log-odds) as 'feature=value'."""
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
            parts.append(f"{names[j]}={v}")
        out.append("; ".join(parts))
    return out


def prepare():
    ev = load()
    ev["split"] = split_of(ev.sched_time_utc)
    cats = {c: pd.CategoricalDtype(sorted(ev[c].dropna().unique())) for c in CATEGORICAL}
    lab = ev[ev.label_45.notna()]
    return ev, lab, cats


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    ev, lab, cats = prepare()
    print("labelled rows by split:", lab.split.value_counts().to_dict())
    cols = feature_cols(ev)
    tr, va = lab[lab.split == "train"], lab[lab.split == "val"]
    m = fit(tr, va, cols, cats)
    print(f"gbm: {len(cols)} features, {m.best_iteration_} trees")
    pv = m.predict_proba(X(va, cols, cats))[:, 1]
    thr, vp, vr = choose_threshold(va.label_45.to_numpy(), pv)
    print(f"global threshold {thr:.3f} (max F1 on val: precision {vp:.3f}, recall {vr:.3f})")
    # per-airport thresholds, also max F1 on validation only (base rates differ by airport)
    by_ap = {}
    for ap in sorted(va.airport.unique()):
        k = (va.airport == ap).to_numpy()
        t, p_, r_ = choose_threshold(va.label_45.to_numpy()[k], pv[k])
        by_ap[ap] = {"threshold": t, "val_precision": p_, "val_recall": r_, "val_alert_rate": float((pv[k] >= t).mean())}
        print(f"  {ap}: threshold {t:.3f} (val precision {p_:.3f}, recall {r_:.3f})")

    te = ev[ev.split == "test"].reset_index(drop=True)
    xt = X(te, cols, cats)
    risk = m.predict_proba(xt)[:, 1]
    thr_row = te.airport.map({ap: d["threshold"] for ap, d in by_ap.items()}).fillna(thr).to_numpy()
    out = pd.DataFrame({"event_id": te.event_id, "airport": te.airport, "event_type": te.event_type,
                        "sched_time_utc": te.sched_time_utc, "h": te.h, "risk_prob": risk.round(4),
                        "high_risk": (risk >= thr_row).astype(int), "top_drivers": top_drivers(m, xt)})
    # join coverage with P1's file must be exact, both ways
    p1_test = ev.loc[ev.split == "test", ["event_id", "h"]]
    j = p1_test.merge(out[["event_id", "h"]], how="outer", indicator=True)
    assert (j._merge == "both").all() and len(out) == len(p1_test) and not out.duplicated(["event_id", "h"]).any()
    out.to_csv(PRED_DIR / "delay_preds.csv", index=False)
    print(f"wrote {len(out):,} rows ({out.event_id.nunique():,} events), join coverage 100%")

    m.booster_.save_model(str(MODEL_DIR / "gbm.txt"))
    (MODEL_DIR / "meta.json").write_text(json.dumps(
        {"event_features_md5": data_md5(), "threshold": thr, "val_precision": vp, "val_recall": vr,
         "thresholds_by_airport": by_ap, "features": cols,
         "best_iteration": m.best_iteration_, "train_end": str(TRAIN_END), "val_end": str(VAL_END)}, indent=1))


if __name__ == "__main__":
    main()
