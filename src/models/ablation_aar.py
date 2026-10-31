"""Which inputs actually help? Trains each variant on train and scores it on VALIDATION only
(2025 test is untouched), then writes outputs/predictions/aar_ablation.csv.

    python -m src.models.ablation_aar

The "oracle" variant swaps in the weather of the target hour itself (oracle_* columns). It is a
diagnostic upper bound on what perfect weather knowledge could add and is never used for the
production model.
"""
import argparse
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import aar_config as cfg
from .aar_baselines import HistoricalAverage, ScheduleBaseline
from .aar_context import BAD_CROSSWIND_KMH, HUB_EXPOSURE_COLUMNS, add_context, weather_flags
from .aar_data import load_airport_timezones, load_hourly, split_chronological
from .aar_features import AARPreprocessor


def hub_weather_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in cfg.WEATHER_OBS_COLUMNS or re.fullmatch(cfg.WEATHER_OBS_PATTERN, c)]


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


def baseline_row(name, pred, val, bad, severe) -> dict:
    err = pred - val[cfg.TARGET]
    return {"variant": name, "rmse": float(np.sqrt((err**2).mean())), "mae": float(err.abs().mean()),
            "rmse_bad_weather_hours": float(np.sqrt((err[bad] ** 2).mean())),
            "rmse_severe_weather_hours": float(np.sqrt((err[severe] ** 2).mean()))}


def run(data_path: Path = cfg.PANEL, out_dir: Path = cfg.PRED_DIR) -> pd.DataFrame:
    timezones = load_airport_timezones()
    panel = load_hourly(data_path)
    full = add_context(panel.dropna(subset=[cfg.TARGET]), panel)
    s = split_chronological(full)
    train, val = s["train"], s["val"]

    hub_wx = hub_weather_columns(full)
    ops = [c for c in full.columns if c.startswith("p2_ops_arr_")]
    oracle = {f"oracle_{c}": c for c in cfg.WEATHER_COLUMNS if f"oracle_{c}" in full}
    wx = set(hub_wx) | set(HUB_EXPOSURE_COLUMNS)
    l2 = {**cfg.LGBM_PARAMS, "objective": "regression"}
    poisson = {**cfg.LGBM_PARAMS, "objective": "poisson"}
    # Hours with actually bad weather at the hub (target-hour oracle values; used only to pick rows to score).
    bad_f, severe_f, _ = weather_flags(val["oracle_condition_code"], val["oracle_precip_mm"], val["oracle_wind_speed_kmh"])
    bad = (bad_f == 1) | (val["oracle_crosswind_kmh"] >= BAD_CROSSWIND_KMH)
    severe = severe_f == 1

    def drop(cols):
        return lambda d: d.drop(columns=cols)

    def with_oracle(d):
        # Weather of hour t in place of t - h weather; rolling windows and exposure flags removed.
        d = d.drop(columns=[c for c in hub_wx if c not in oracle.values()] + HUB_EXPOSURE_COLUMNS)
        return d.drop(columns=list(oracle.values())).rename(columns=oracle)

    variants = [
        ("A schedule + calendar only", drop(hub_wx + HUB_EXPOSURE_COLUMNS + ops), poisson),
        ("B A + hub weather at t-h", drop(HUB_EXPOSURE_COLUMNS + ops), poisson),
        ("C B + weather exposure", drop(ops), poisson),
        ("D C + recent arrivals (all inputs, production)", lambda d: d, poisson),
        ("E D without any weather", drop(hub_wx + HUB_EXPOSURE_COLUMNS), poisson),
        ("F D with L2 objective", lambda d: d, l2),
        ("G oracle: D with weather of hour t itself (diagnostic)", with_oracle, poisson),
    ]
    rows = [baseline_row("baseline: schedule x completion ratio", ScheduleBaseline().fit(train).predict(val), val, bad, severe),
            baseline_row("baseline: hour x weekday mean", HistoricalAverage(timezones).fit(train).predict(val), val, bad, severe)]
    for name, transform, params in variants:
        print(f"running {name} ...", flush=True)
        rows.append({"variant": name, **score(transform(train), transform(val), timezones, params, wx, bad, severe)})

    table = pd.DataFrame(rows)
    print(f"validation rows: {len(val)}; bad-weather {int(bad.sum())}, severe {int(severe.sum())}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "aar_ablation.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=cfg.PANEL)
    ap.add_argument("--out-dir", type=Path, default=cfg.PRED_DIR)
    args = ap.parse_args()
    run(args.data, args.out_dir)


if __name__ == "__main__":
    main()
