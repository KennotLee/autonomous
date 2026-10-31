"""P4: cross-airport generalisation test for the AAR-proxy model (brief: "test cross-airport generalisation").

Recipe ~ P2 v3 (Poisson LightGBM, lr 0.05, 31 leaves, 458 rounds, seed 42) on P1's hourly_features.parquet,
with inputs = P1 hub weather + wind components (runway heading from config, so they transfer) + calendar
+ sched_arr_count at t-2..t+2. Differences from P2 v3: no `icao` input (it cannot transfer) and no
p2_hub_* weather-exposure flags. Train = target_hour_utc < 2025 (UTC), test = 2025.

For each target airport B:  within = trained on B;  cross = trained on the other airport A only.
The schedule baseline uses the completion ratio of the airport the model was trained on, so the
cross comparison never touches B's history.

Usage: python src/eval/cross_airport.py   ->  outputs/metrics/aar_cross_airport.csv
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import OUT, PROCESSED, SEVERE_CODES, TEST_START, load_airports

PARAMS = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42,
          "deterministic": True, "force_col_wise": True, "verbose": -1}
NUM_ROUNDS = 458  # P2 v3 point model best iteration (outputs/predictions/aar_model/metadata.json)
OFFSETS = (-2, -1, 1, 2)
WEATHER = ["temp_c", "dwpt_c", "rhum_pct", "prcp_mm", "wdir_deg", "wspd_kt", "wpgt_kt", "pres_hpa", "coco",
           "wspd_max_3h_kt", "wpgt_max_3h_kt", "prcp_sum_3h_mm", "pres_chg_3h_hpa", "wspd_max_6h_kt",
           "wpgt_max_6h_kt", "prcp_sum_6h_mm", "pres_chg_6h_hpa", "headwind_kt", "crosswind_kt", "tailwind_kt",
           "gust_headwind_kt", "gust_crosswind_kt", "gust_tailwind_kt"]
FEATURES = ["h"] + WEATHER + ["hour_local", "dow_local", "month_local", "hour_utc", "sched_arr_count"] + \
           [f"sched_{'m' if k < 0 else 'p'}{abs(k)}" for k in OFFSETS]


def load():
    hr = pd.read_parquet(PROCESSED / "hourly_features.parquet")
    sched = hr.drop_duplicates(["icao", "target_hour_utc"]).set_index(["icao", "target_hour_utc"])["sched_arr_count"]
    for k in OFFSETS:  # published schedule of neighbouring hours: known in advance
        idx = pd.MultiIndex.from_arrays([hr["icao"], hr["target_hour_utc"] + np.timedelta64(k, "h")])
        hr[f"sched_{'m' if k < 0 else 'p'}{abs(k)}"] = sched.reindex(idx).to_numpy(dtype=float)
    hr[FEATURES] = hr[FEATURES].astype(float)
    return hr


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    hr = load()
    icaos = list(load_airports())
    train, test = hr[hr["target_hour_utc"] < TEST_START], hr[hr["target_hour_utc"] >= TEST_START]
    models, ratios = {}, {}
    for a in icaos:
        tr = train[train["icao"] == a]
        models[a] = lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr["arr_count"]), num_boost_round=NUM_ROUNDS)
        one = tr[tr["h"] == tr["h"].min()]
        ratios[a] = one["arr_count"].sum() / one["sched_arr_count"].sum()
    rows = []
    for b in icaos:
        te = test[test["icao"] == b]
        y = te["arr_count"].to_numpy(float)
        sev = te["oracle_coco"].isin(SEVERE_CODES).to_numpy()
        for a in icaos:
            pred = models[a].predict(te[FEATURES])
            sched = te["sched_arr_count"].to_numpy(float) * ratios[a]
            for sl, m in [("all hours", np.ones(len(te), bool)), ("severe wx at hour t", sev)]:
                rows.append({"test_airport": b, "trained_on": a, "setting": "within" if a == b else "cross",
                             "slice": sl, "n_rows": int(m.sum()), "rmse_model": rmse(y[m], pred[m]),
                             "rmse_schedule_baseline": rmse(y[m], sched[m]),
                             "mean_pred": float(pred[m].mean()), "mean_actual": float(y[m].mean())})
    res = pd.DataFrame(rows)
    res["impr_vs_schedule_pct"] = 100 * (1 - res["rmse_model"] / res["rmse_schedule_baseline"])
    res.to_csv(OUT / "aar_cross_airport.csv", index=False)
    print(f"train ratios (arrivals/scheduled): { {k: round(v, 4) for k, v in ratios.items()} }")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
