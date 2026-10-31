"""P4 baselines: delay >45 min (Task 1a) and hourly arrivals / AAR proxy (Task 1b).

Usage (Windows: .venv/Scripts/python, macOS/Linux: .venv/bin/python):
    python src/eval/baselines.py delay        # flight_features.parquet -> delay baselines
    python src/eval/baselines.py aar          # hourly_features.parquet -> AAR baselines
    python src/eval/baselines.py alignment    # raw CSV weather vs Meteostat (evidence for the 4-6 h misalignment)
    python src/eval/baselines.py all

Inputs are P1's files (schemas in CLAUDE.md). Outputs go to outputs/metrics/.

Split (same as P2's final refit): train = everything before 2025-01-01 00:00 UTC, test = 2025-01-01 UTC
onward (data ends 2025-07-31 local). Flights split on sched_time_utc, hours on target_hour_utc. No random splits.

Delay label: P1's label_45 (ARR_DELAY > 45). Cancelled and diverted flights have label_45 = NaN and are
dropped (team decision 5 in CLAUDE.md); counts are printed.

Severe-weather slices use P1's oracle_* coco (Meteostat at the flight's own scheduled hour). Oracle columns
are used ONLY to choose which rows to score, never as an input to any baseline.
Meteostat coco (https://dev.meteostat.net/formats.html):
    snow = 14 Light Snowfall, 15 Snowfall, 16 Heavy Snowfall, 21 Snow Shower, 22 Heavy Snow Shower
    thunderstorm/storm = 23 Lightning, 24 Hail, 25 Thunderstorm, 26 Heavy Thunderstorm, 27 Storm
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
OUT = ROOT / "outputs/metrics"
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
HORIZONS = (3, 6, 9, 12)
SNOW_CODES = {14, 15, 16, 21, 22}
STORM_CODES = {23, 24, 25, 26, 27}
SEVERE_CODES = SNOW_CODES | STORM_CODES
SMOOTHING_M = 20  # pseudo-flights pulling sparse cells toward the global rate (fixed, not tuned on test)

pd.set_option("display.width", 220)
fmt = lambda x: f"{x:.3f}"  # noqa: E731


def load_airports(path=ROOT / "config/airports.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["airports"]


def is_severe(coco):
    return coco.isin(SEVERE_CODES).to_numpy()


# ================================================================ Task 1a: delay > 45 min

def fit_cell_rate(train, keys, g, m=SMOOTHING_M):
    """Smoothed positive rate per cell: (pos + m * global) / (n + m). Unseen cells fall back to global."""
    agg = train.groupby(list(keys))["y"].agg(["sum", "count"])
    agg["rate"] = (agg["sum"] + m * g) / (agg["count"] + m)
    return agg


def predict_cell_rate(df, agg, g, keys):
    out = df[list(keys)].merge(agg, left_on=list(keys), right_index=True, how="left")
    return out["rate"].fillna(g).to_numpy(), out["count"].fillna(0).astype(int).to_numpy()


def recall_at_precision(y, p, min_precision):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def delay_metrics(y, p, n_boot=200, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    res = {"n": len(y), "positives": int(y.sum()), "pos_rate": y.mean() if len(y) else np.nan}
    if len(y) == 0 or y.sum() == 0:
        return res
    res["pr_auc"] = average_precision_score(y, p)
    res["brier"] = brier_score_loss(y, p)
    res["recall@p0.3"] = recall_at_precision(y, p, 0.3)
    res["recall@p0.5"] = recall_at_precision(y, p, 0.5)
    # recall@p is trivially 1.0 when the slice's own positive rate already exceeds p
    res["recall_trivial@p0.3"] = res["pos_rate"] >= 0.3
    res["recall_trivial@p0.5"] = res["pos_rate"] >= 0.5
    rng = np.random.default_rng(seed)  # bootstrap 90% CI for PR-AUC so small slices are not over-read
    boots = [average_precision_score(y[i], p[i])
             for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if y[i].sum() > 0]
    res["pr_auc_lo"], res["pr_auc_hi"] = np.percentile(boots, [5, 95])
    return res


def delay_slices(df, airports):
    yield "ALL (hub flights)", np.ones(len(df), bool)
    hub_sev = np.zeros(len(df), bool)
    for icao in airports:
        arr, dep = (df["dest_icao"] == icao).to_numpy(), (df["origin_icao"] == icao).to_numpy()
        yield f"{icao} arrivals", arr
        yield f"{icao} departures", dep
        sev = (arr & is_severe(df["oracle_dest_coco"])) | (dep & is_severe(df["oracle_origin_coco"]))
        hub_sev |= sev
        yield f"severe wx at {icao} at sched hour", sev
    yield "severe wx at hub end", hub_sev
    yield "severe wx at either end", is_severe(df["oracle_origin_coco"]) | is_severe(df["oracle_dest_coco"])


def run_delay(airports):
    fl = pd.read_parquet(PROCESSED / "flight_features.parquet")
    one = fl[fl["h"] == HORIZONS[0]]  # baselines do not depend on h: score each flight once
    n_all = len(one)
    one = one[one["label_45"].notna()].copy()
    print(f"[delay] {n_all} hub flights; dropped {n_all - len(one)} with label_45 NaN "
          f"(cancelled {int(fl.loc[fl.h == HORIZONS[0], 'cancelled'].sum())}, "
          f"diverted {int(fl.loc[fl.h == HORIZONS[0], 'diverted'].sum())})")
    one["y"] = one["label_45"].astype(int)
    one["dep_hour_local"] = one["dep_hour_local"].astype(int)
    train, test = one[one["sched_time_utc"] < TEST_START], one[one["sched_time_utc"] >= TEST_START].copy()
    g = float(train["y"].mean())
    print(f"[delay] train {len(train)} flights ({g:.4f} positive), "
          f"test {len(test)} flights ({test['y'].mean():.4f} positive)")

    specs = {
        "base_rate_origin_dephour_month": ("origin_icao", "dep_hour_local", "month_local"),
        # extra: keyed on the arrival end, closer to the AAR question
        "base_rate_dest_arrhour_month": ("dest_icao", "arr_hour_local", "month_local"),
    }
    models = {"base_rate": (np.full(len(test), g), np.full(len(test), f"global train rate (n={len(train)})"))}
    for name, keys in specs.items():
        agg = fit_cell_rate(train, keys, g)
        p, n = predict_cell_rate(test, agg, g, keys)
        drivers = ("historical rate for " + test[keys[0]] + " | hour " + test[keys[1]].astype(str)
                   + " | month " + test[keys[2]].astype(str) + " (train n="
                   + pd.Series(n, index=test.index).astype(str) + ")").to_numpy()
        models[name] = (p, drivers)
        print(f"[delay] {name}: {len(agg)} cells, test coverage (cell seen in train) = {(n > 0).mean():.3f}, "
              f"median train n per test cell = {np.median(n):.0f}")

    rows = [{"model": name, "slice": sl, **delay_metrics(test["y"].to_numpy()[mask], p[mask])}
            for name, (p, _) in models.items() for sl, mask in delay_slices(test, airports)]
    metrics = pd.DataFrame(rows)

    # volume-normalised: share of flights >45 min late per hub and direction, train vs test
    one["direction"] = np.where(one["dest_icao"].isin(list(airports)), "arrivals", "departures")
    one["hub"] = np.where(one["direction"] == "arrivals", one["dest_icao"], one["origin_icao"])
    one["period"] = np.where(one["sched_time_utc"] < TEST_START, "train_2023_2024", "test_2025")
    share = (one.groupby(["hub", "direction", "period"])["y"].agg(share_gt45="mean", flights="count")
             .unstack("period"))
    share.columns = [f"{a}_{b}" for a, b in share.columns]

    # predictions: CLAUDE.md schema (flight_id, sched_time_utc, h, p_delay45) + model, top_drivers
    preds = pd.concat([
        pd.DataFrame({"flight_id": test["flight_id"].to_numpy(), "sched_time_utc": test["sched_time_utc"].to_numpy(),
                      "h": h, "p_delay45": p.round(5), "model": name, "top_drivers": drivers})
        for name, (p, drivers) in models.items() for h in HORIZONS])

    metrics.to_csv(OUT / "delay_baseline_metrics.csv", index=False)
    share.to_csv(OUT / "delay_share_by_hub.csv")
    preds.to_csv(OUT / "delay_baseline_preds.csv", index=False)

    show = metrics[["model", "slice", "n", "positives", "pos_rate", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                    "brier", "recall@p0.3", "recall@p0.5"]]
    print("\n" + show.to_string(index=False, float_format=fmt))
    print("\nShare of flights >45 min late, by hub and direction:")
    print(share.to_string(float_format=fmt))
    print(f"\n[delay] wrote {len(preds)} prediction rows ({len(test)} test flights x {len(HORIZONS)} h x {len(models)} models)")


# ================================================================ Task 1b: hourly arrivals (AAR proxy)

def persistence(df, series):
    """Arrivals in the last COMPLETED hour at issue time t - h, i.e. the hour [t-h-1, t-h).
    (arrivals(t-h) itself would leak: that hour is not over at issue time.)"""
    h = df["h"].astype("int64")
    issue = df["target_hour_utc"] - pd.to_timedelta(h, unit="h")
    lag_hour = issue - np.timedelta64(1, "h")
    assert (lag_hour + np.timedelta64(1, "h") <= issue).all(), "persistence would read the future"
    idx = pd.MultiIndex.from_arrays([df["icao"], lag_hour])
    return series.reindex(idx).to_numpy(dtype=float)


def fit_mean(train_hours, keys):
    return train_hours.groupby(keys)["arr_count"].mean()


def predict_mean(df, table, keys, fallback):
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
    return pd.Series(table.reindex(idx).to_numpy(), index=df.index).fillna(fallback).to_numpy(dtype=float)


def aar_baseline_preds(hourly):
    """Return {model: predictions aligned to hourly rows}. All fit on train rows only."""
    series = hourly.drop_duplicates(["icao", "target_hour_utc"]).set_index(["icao", "target_hour_utc"])["arr_count"]
    train_hours = hourly[(hourly["target_hour_utc"] < TEST_START)].drop_duplicates(["icao", "target_hour_utc"])
    icao_mean = train_hours.groupby("icao")["arr_count"].mean()
    fallback = hourly["icao"].map(icao_mean)

    preds = {"persistence": persistence(hourly, series)}
    k1 = ["icao", "hour_local"]
    preds["hour_of_day_mean"] = predict_mean(hourly, fit_mean(train_hours, k1), k1, fallback)
    k2 = ["icao", "hour_local", "dow_local"]
    preds["hour_of_week_mean"] = predict_mean(hourly, fit_mean(train_hours, k2), k2, fallback)
    # schedule is known in advance: scale scheduled arrivals by the train-period completion ratio per icao
    ratio = train_hours.groupby("icao")["arr_count"].sum() / train_hours.groupby("icao")["sched_arr_count"].sum()
    preds["scheduled_x_ratio"] = (hourly["sched_arr_count"] * hourly["icao"].map(ratio)).to_numpy(dtype=float)
    return preds


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan


def run_aar(airports):
    hourly = pd.read_parquet(PROCESSED / "hourly_features.parquet")
    hourly = hourly[hourly["icao"].isin(list(airports))].reset_index(drop=True)
    print(f"[aar] {len(hourly)} rows ({hourly.icao.nunique()} icao x hours x {hourly.h.nunique()} h); "
          f"arr_count == 0 in {(hourly.arr_count == 0).mean():.1%} of rows")
    preds = aar_baseline_preds(hourly)
    is_test = (hourly["target_hour_utc"] >= TEST_START).to_numpy()
    y = hourly["arr_count"].to_numpy(dtype=float)
    sev = is_severe(hourly["oracle_coco"])
    busy = (hourly["sched_arr_count"] > 0).to_numpy()

    rows, out = [], []
    for name, p in preds.items():
        # p10/p90 from empirical TRAIN residuals per icao x h, added to the point prediction, floored at 0
        res = pd.DataFrame({"icao": hourly["icao"], "h": hourly["h"], "r": y - p})[~is_test & ~np.isnan(p)]
        q = res.groupby(["icao", "h"])["r"].quantile([0.1, 0.9]).unstack()
        key = pd.MultiIndex.from_frame(hourly[["icao", "h"]])
        p10 = np.clip(p + q[0.1].reindex(key).to_numpy(), 0, None)
        p90 = np.clip(p + q[0.9].reindex(key).to_numpy(), 0, None)
        for (icao, h), grp in hourly[is_test].groupby(["icao", "h"]):
            for sl_name, sl in [("all hours", np.ones(len(hourly), bool)), ("hours with scheduled arrivals", busy),
                                ("severe wx at hour t", sev)]:
                m = is_test & sl & (hourly["icao"] == icao).to_numpy() & (hourly["h"] == h).to_numpy() & ~np.isnan(p)
                rows.append({"model": name, "icao": icao, "h": h, "slice": sl_name, "n_hours": int(m.sum()),
                             "mean_actual": y[m].mean() if m.any() else np.nan, "rmse": rmse(y[m], p[m]),
                             "mae": float(np.mean(np.abs(y[m] - p[m]))) if m.any() else np.nan,
                             "p10_p90_coverage": float(np.mean((y[m] >= p10[m]) & (y[m] <= p90[m]))) if m.any() else np.nan})
        out.append(pd.DataFrame({"icao": hourly["icao"], "target_hour_utc": hourly["target_hour_utc"], "h": hourly["h"],
                                 "pred_arr_count": p.round(4), "p10": p10.round(4), "p90": p90.round(4),
                                 "model": name})[is_test])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "aar_baseline_metrics.csv", index=False)
    pd.concat(out).to_csv(OUT / "aar_baseline_preds.csv", index=False)

    print("\n[aar] Test RMSE by icao x h (all hours / hours with scheduled arrivals / severe-wx hours):")
    wide = metrics.pivot_table(index=["icao", "h"], columns=["slice", "model"], values="rmse")
    for sl in ["all hours", "hours with scheduled arrivals", "severe wx at hour t"]:
        n = metrics[metrics.slice == sl].groupby(["icao", "h"])["n_hours"].first()
        print(f"\n-- {sl} (n_hours per row: {n.min()}-{n.max()})")
        print(wide[sl].to_string(float_format=fmt))
    cov = metrics[metrics.slice == "all hours"].groupby("model")["p10_p90_coverage"].mean()
    print("\n[aar] mean p10-p90 coverage on test, all hours (nominal 0.80):", cov.round(3).to_dict())
    print("\n*** CAVEAT: arr_count comes from a ~0.5% BTS sample (0 in ~82% of hours). These numbers prove the code\n"
          "*** runs and give P2 a bar on THIS target; they are NOT runway-capacity (AAR) accuracy.")


# ================================================================ evidence: CSV weather alignment

def local_hhmm_to_utc(dates, hhmm, tz):
    hhmm = hhmm.astype(int)
    local = dates + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    return local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)


def meteostat_hourly(icao, cache_dir=ROOT / "data/interim"):
    cache = cache_dir / f"meteostat_hourly_{icao}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from meteostat import Hourly, Stations
    st = Stations().fetch()
    wx = Hourly(st.index[st["icao"] == icao][0], datetime(2023, 1, 1), datetime(2025, 8, 1, 23), model=True).fetch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(cache)
    return wx


def run_alignment(airports, csv_path=ROOT / "data/raw/Flight_Data_With_Weather_Final.csv"):
    """Share of rows where the CSV's temp equals Meteostat temp under the naive (local-as-UTC) key vs the correct UTC key."""
    df = pd.read_csv(csv_path, usecols=["FL_DATE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
                                        "temp_ORIGIN", "temp_DEST"])
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    rows = []
    for icao, cfg in airports.items():
        wx = meteostat_hourly(icao)
        for end, tcol in [("ORIGIN", "CRS_DEP_TIME"), ("DEST", "CRS_ARR_TIME")]:
            sub = df[df[end] == cfg["iata"]]
            dates = sub["FL_DATE"]
            if end == "DEST":  # overnight arrivals land the day after FL_DATE
                dates = dates + pd.to_timedelta((sub["CRS_ARR_TIME"] < sub["CRS_DEP_TIME"]).astype(int), unit="D")
            keys = {"naive local-as-UTC": (dates + pd.to_timedelta(sub[tcol].astype(int) // 100, unit="h")).dt.floor("h"),
                    "correct UTC": local_hhmm_to_utc(dates, sub[tcol], cfg["timezone"]).dt.floor("h")}
            for key_name, key in keys.items():
                ms = wx["temp"].reindex(key).to_numpy(dtype=float)
                ok = ~np.isnan(ms)
                rows.append({"icao": icao, "end": end, "key": key_name, "n": int(ok.sum()),
                             "temp_match_share": float(np.isclose(sub[f"temp_{end}"].to_numpy()[ok], ms[ok], atol=0.05).mean())})
    align = pd.DataFrame(rows)
    align.to_csv(OUT / "weather_alignment_check.csv", index=False)
    print("[alignment] CSV temp == Meteostat temp:\n" + align.to_string(index=False, float_format=fmt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["delay", "aar", "alignment", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    airports = load_airports()
    if args.task in ("alignment", "all"):
        run_alignment(airports)
    if args.task in ("delay", "all"):
        run_delay(airports)
    if args.task in ("aar", "all"):
        run_aar(airports)


if __name__ == "__main__":
    main()
