"""Baselines vs model on a time-based split (train 2023-24, test 2025).

AAR proxy (regression, RMSE in arrivals/hour)
  schedule       pred = scheduled arrivals at T (the plan, no weather)
  persistence    pred = scheduled at T x capacity ratio of the 3 h before issue
  climatology    pred = train mean arrivals by airport x local hour x weekday
  gbm            LightGBM on leakage-safe features
  gbm_oracle     same + weather AT hour T (not available at issue time)

Severe delay (>45 min / cancelled / diverted, scored per scheduled flight)
  Hour rows are expanded into flight-level labels with weights (n_sev
  positives, sched - n_sev negatives), so the metrics are per flight and
  automatically normalised by volume.
  climatology    train rate by airport x local hour x month
  persistence    severe rate of the 3 h before issue
  gbm / gbm_oracle as above
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import TEST_YEARS, TRAIN_YEARS

ID_COLS = ["target_hour_utc", "airport", "local_time", "issue_time_utc"]
TARGETS = ["y_act_arr", "y_n_sev", "y_cap_ratio"]
GBM_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def split(ds: pd.DataFrame):
    year = ds.local_time.dt.year
    return ds[year.isin(TRAIN_YEARS)].copy(), ds[year.isin(TEST_YEARS)].copy()


def feature_cols(ds: pd.DataFrame, oracle: bool) -> list[str]:
    cols = [c for c in ds.columns if c not in ID_COLS + TARGETS]
    return cols if oracle else [c for c in cols if not c.startswith("now_")]


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["airport"] = X["airport"].astype("category")
    return X


def _xy(df, cols):
    return _prep(df[cols + ["airport"]])


# ---------------------------------------------------------------- AAR proxy
def aar_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test[ID_COLS + ["h", "sched_arr_T", "y_act_arr"]].copy()
    out["schedule"] = test.sched_arr_T
    ratio = test.obs_cap_ratio_3h.clip(0, 1.2).fillna(1.0)
    out["persistence"] = test.sched_arr_T * ratio
    keys = ["airport", "hour_local", "dow"]
    clim = train.drop_duplicates(["target_hour_utc", "airport"]).groupby(keys).y_act_arr.mean()
    out["climatology"] = test[keys].merge(clim.rename("c").reset_index(), on=keys, how="left").c.values
    for name, oracle in (("gbm", False), ("gbm_oracle", True)):
        cols = feature_cols(train, oracle)
        m = lgb.LGBMRegressor(n_estimators=600, **GBM_PARAMS)
        m.fit(_xy(train, cols), train.y_act_arr)
        out[name] = np.clip(m.predict(_xy(test, cols)), 0, None)
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ------------------------------------------------------------- severe delay
def _expand(df: pd.DataFrame):
    """Hour rows -> (rows, labels, weights) at flight level."""
    n = df.sched_arr_T.to_numpy()
    pos = df.y_n_sev.to_numpy()
    keep = n > 0
    rows = np.concatenate([np.flatnonzero(keep), np.flatnonzero(keep)])
    y = np.r_[np.ones(keep.sum()), np.zeros(keep.sum())]
    w = np.r_[pos[keep], (n - pos)[keep]]
    nz = w > 0
    return rows[nz], y[nz], w[nz]


def sev_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test[ID_COLS + ["h", "sched_arr_T", "y_n_sev"]].copy()
    keys = ["airport", "hour_local", "month"]
    hours = train.drop_duplicates(["target_hour_utc", "airport"])
    g = hours.groupby(keys)[["y_n_sev", "sched_arr_T"]].sum()
    clim = (g.y_n_sev / g.sched_arr_T.where(g.sched_arr_T > 0)).rename("c").reset_index()
    base = hours.y_n_sev.sum() / hours.sched_arr_T.sum()
    out["climatology"] = test[keys].merge(clim, on=keys, how="left").c.fillna(base).values
    out["persistence"] = test.obs_sev_rate_3h.fillna(base).clip(0, 1).values
    rows, y, w = _expand(train)
    for name, oracle in (("gbm", False), ("gbm_oracle", True)):
        cols = feature_cols(train, oracle)
        m = lgb.LGBMClassifier(n_estimators=600, **GBM_PARAMS)
        m.fit(_xy(train.iloc[rows], cols), y, sample_weight=w)
        out[name] = m.predict_proba(_xy(test, cols))[:, 1]
    return out


def sev_metrics(pred: pd.DataFrame, models) -> dict:
    rows, y, w = _expand(pred)
    res = {"base_rate": float(np.average(y, weights=w)), "n_flights": int(w.sum())}
    for m in models:
        p = pred[m].to_numpy()[rows]
        res[f"{m}_pr_auc"] = average_precision_score(y, p, sample_weight=w)
        res[f"{m}_brier"] = brier_score_loss(y, p, sample_weight=w)
    return res


def calibration_table(pred: pd.DataFrame, model: str, bins=10) -> pd.DataFrame:
    rows, y, w = _expand(pred)
    p = pred[model].to_numpy()[rows]
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p * w, "y": y * w, "w": w}).groupby("bin").sum()
    return pd.DataFrame({"mean_pred": df.p / df.w, "observed": df.y / df.w, "n_flights": df.w})
