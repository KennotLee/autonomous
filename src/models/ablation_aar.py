"""Which inputs actually help? Trains each variant on train and scores it on VALIDATION only
(2025 test is untouched), then writes outputs/predictions/aar_ablation.csv.

    python -m src.models.ablation_aar --data data/interim/hourly_features.parquet \
                                      --flights data/interim/flight_features.parquet

The "oracle" variant swaps in same-hour weather (oracle_* columns). It is a diagnostic upper
bound on what perfect weather knowledge could add and is never used for the production model.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import aar_config as cfg
from .aar_baselines import HistoricalAverage
from .aar_context import HUB_EXPOSURE_COLUMNS, ORIGIN_COLUMNS, SCHED_OFFSETS, add_context, load_flights, sched_offset_name, weather_flags
from .aar_data import load_airport_timezones, load_hourly, split_chronological
from .aar_features import AARPreprocessor

DEFAULT_FLIGHTS = cfg.HOURLY_FEATURES.with_name("flight_features.parquet")


def hub_weather_columns(df: pd.DataFrame) -> list[str]:
    oracle_bases = {c.removeprefix("oracle_") for c in df.columns if c.startswith("oracle_")}
    rolling = [c for c in df.columns if c.startswith(("wspd_max_", "wpgt_max_", "prcp_sum_", "pres_chg_"))]
    return sorted((oracle_bases & set(df.columns)) | set(rolling))


def score(train, val, timezones, params, wx: set[str], bad: pd.Series, severe: pd.Series) -> dict:
    pre = AARPreprocessor(timezones).fit(train)
    dtrain = lgb.Dataset(pre.transform(train), train[cfg.TARGET], categorical_feature=cfg.CATEGORICAL)
    dval = lgb.Dataset(pre.transform(val), val[cfg.TARGET], reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=cfg.NUM_BOOST_ROUND, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False)])
    err = pd.Series(booster.predict(pre.transform(val)).clip(min=0), index=val.index) - val[cfg.TARGET]
    gain = dict(zip(pre.feature_columns, booster.feature_importance("gain")))
    total = sum(gain.values()) or 1.0
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_bad_weather_hours": float(np.sqrt(np.mean(err[bad] ** 2))),
        "rmse_severe_weather_hours": float(np.sqrt(np.mean(err[severe] ** 2))),
        "best_iter": int(booster.best_iteration),
        "n_features": len(pre.feature_columns),
        "weather_gain_share": round(sum(v for k, v in gain.items() if k in wx) / total, 3),
    }


def run(data_path: Path, flights_path: Path, out_dir: Path = cfg.PRED_DIR) -> pd.DataFrame:
    timezones = load_airport_timezones()
    panel = load_hourly(data_path)
    flights = load_flights(flights_path)
    if flights is None:
        raise FileNotFoundError(f"{flights_path} not found; origin-weather variants need it")
    full = add_context(panel.dropna(subset=[cfg.TARGET]), panel, flights)
    s = split_chronological(full)
    train, val = s["train"], s["val"]

    hub_wx = hub_weather_columns(full)
    sched_ctx = [sched_offset_name(k) for k in SCHED_OFFSETS]
    old_origin = ORIGIN_COLUMNS[:7]  # origin aggregates as first tested (variants E, F)
    new_wx = HUB_EXPOSURE_COLUMNS + ORIGIN_COLUMNS
    oracle = {f"oracle_{c}": c for c in hub_wx if f"oracle_{c}" in full}
    l2 = {**cfg.LGBM_PARAMS, "objective": "regression"}
    poisson = {**cfg.LGBM_PARAMS, "objective": "poisson"}
    wx = set(hub_wx) | set(new_wx)
    # Hours with actually bad weather at the hub (same-hour oracle values; used only to pick rows to score).
    bad = ((val.get("oracle_prcp_mm", 0) >= 1) | (val.get("oracle_coco", 0) >= 7)
           | (val.get("oracle_wspd_kt", 0) >= 20) | (val.get("oracle_crosswind_kt", 0) >= 15))
    severe = weather_flags(val["oracle_coco"], val["oracle_prcp_mm"], val["oracle_wspd_kt"])[1].astype(bool)

    def drop(cols):
        return lambda d: d.drop(columns=cols)

    def with_oracle(d):
        d = d.drop(columns=[c for c in hub_wx if c not in oracle.values()] + new_wx)
        return d.drop(columns=list(oracle.values())).rename(columns=oracle)

    variants = [
        ("A v1 (hub weather + schedule t)", drop(sched_ctx + new_wx), l2),
        ("B no weather at all", drop(hub_wx + sched_ctx + new_wx), l2),
        ("C + schedule t-2..t+2", drop(new_wx), l2),
        ("D C without weather", drop(hub_wx + new_wx), l2),
        ("D2 C with poisson objective (v2)", drop(new_wx), poisson),
        ("E C + origin-airport weather maxima", drop(HUB_EXPOSURE_COLUMNS + ORIGIN_COLUMNS[7:]), l2),
        ("F E with poisson objective", drop(HUB_EXPOSURE_COLUMNS + ORIGIN_COLUMNS[7:]), poisson),
        ("G oracle: C with same-hour hub weather (diagnostic)", with_oracle, l2),
        ("H D2 + hub weather exposure", drop(ORIGIN_COLUMNS), poisson),
        ("I D2 + hub + origin weather exposure (all weather)", lambda d: d, poisson),
        ("J I without raw hub weather (exposure features only)", drop(hub_wx), poisson),
    ]
    hist = HistoricalAverage(timezones).fit(train)
    hist_err = hist.predict(val) - val[cfg.TARGET]
    rows = [{"variant": "baseline: historical average", "rmse": float(np.sqrt((hist_err**2).mean())),
             "mae": float(hist_err.abs().mean()), "rmse_bad_weather_hours": float(np.sqrt((hist_err[bad]**2).mean())),
             "rmse_severe_weather_hours": float(np.sqrt((hist_err[severe]**2).mean()))}]
    for name, transform, params in variants:
        print(f"running {name} ...")
        rows.append({"variant": name, **score(transform(train), transform(val), timezones, params, wx, bad, severe)})

    table = pd.DataFrame(rows)
    print(f"bad-weather validation rows: {int(bad.sum())}, severe: {int(severe.sum())}, of {len(val)}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "aar_ablation.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=cfg.HOURLY_FEATURES)
    ap.add_argument("--flights", type=Path, default=DEFAULT_FLIGHTS)
    ap.add_argument("--out-dir", type=Path, default=cfg.PRED_DIR)
    args = ap.parse_args()
    run(args.data, args.flights, args.out_dir)


if __name__ == "__main__":
    main()
