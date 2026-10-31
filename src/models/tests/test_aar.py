"""AAR tests on SYNTHETIC data shaped like the P2 panel (challenge-5 weather + BTS arrivals).

Synthetic numbers only exercise the code paths; they say nothing about real model quality.
Run from the repo root:  python -m pytest src/models/tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src.models import aar_config as cfg
from src.models.aar_baselines import HistoricalAverage, ScheduleBaseline, persistence
from src.models.aar_bts import arrival_times, hourly_arrivals
from src.models.aar_context import add_hub_exposure, add_schedule_context
from src.models.aar_data import SchemaError, check, load_airport_timezones, split_chronological, validate_hourly
from src.models.aar_dataset import build_panel
from src.models.aar_features import AARPreprocessor, select_feature_columns
from src.models.aar_metrics import evaluate

TZ = load_airport_timezones()
IATA = {"ATL": "KATL", "DFW": "KDFW"}


def make_panel(start="2024-05-01", end="2025-02-15", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    frames = []
    for icao in ("KATL", "KDFW"):
        sched = rng.poisson(20 + 30 * (hours.hour % 24 > 10), len(hours)).astype(float)
        wind = rng.gamma(2.0, 8.0, len(hours))
        arr = rng.poisson(np.clip(sched - 0.2 * wind, 0, None))
        for h in cfg.HORIZONS:
            frames.append(pd.DataFrame({
                "icao": icao,
                cfg.TIME: hours,
                "h": h,
                cfg.OBS_TIME: hours - pd.Timedelta(hours=h),
                "wind_speed_kmh": wind + rng.normal(0, 2, len(hours)),
                "crosswind_kmh": rng.normal(0, 5, len(hours)).clip(0),
                "sched_arr_count": sched,
                "p2_ops_arr_last1h": rng.poisson(30, len(hours)).astype(float),
                cfg.TARGET: arr,
                # Columns that must never become features:
                "capacity_ratio": arr / 3,
                "mean_arr_delay": rng.normal(10, 5, len(hours)),
                "share_delayed_45": rng.random(len(hours)),
                "cancellations": rng.poisson(0.1, len(hours)),
                "diversions": rng.poisson(0.05, len(hours)),
                "oracle_wind_speed_kmh": wind,
                "oracle_arr_hint": arr,
            }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


# ---------- data: BTS arrivals and the weather join ----------

def test_bts_arrival_times_local_to_utc_and_next_day():
    f = pd.DataFrame({
        "FlightDate": ["2024-03-01", "2024-03-01", "2024-03-01"],
        "Dest": ["ATL", "ATL", "DFW"],
        # LAX 22:00 -> ATL 05:30 next day (red-eye), ATL-bound short hop, cancelled DFW arrival
        "CRSDepTime": [2200, 900, 1000], "CRSArrTime": [530, 1030, 1200], "CRSElapsedTime": [270, 90, 150],
        "ArrDelay": [10.0, -5.0, np.nan], "TaxiIn": [8.0, 5.0, np.nan], "WheelsOn": [532, 1020, np.nan],
        "Cancelled": [0.0, 0.0, 1.0], "Diverted": [0.0, 0.0, 0.0],
    })
    t = arrival_times(f, TZ, IATA)
    # 2024-03-02 05:30 EST = 10:30 UTC; wheels-on = 05:30 + 10 - 8 = 05:32 local = 10:32 UTC
    assert t.loc[0, "sched_arr_utc"] == pd.Timestamp("2024-03-02 10:30", tz="UTC")
    assert t.loc[0, "wheels_on_utc"] == pd.Timestamp("2024-03-02 10:32", tz="UTC")
    assert t.loc[1, "wheels_on_utc"] == pd.Timestamp("2024-03-01 15:20", tz="UTC")
    assert pd.isna(t.loc[2, "wheels_on_utc"]) and t.loc[2, "cancelled"] == 1
    assert t.attrs["wheels_on_match"] == 1.0

    grid = pd.date_range("2024-03-01 15:00", "2024-03-02 11:00", freq="h", tz="UTC")
    hourly = hourly_arrivals(t, grid).set_index(["icao", cfg.TIME])
    assert hourly.loc[("KATL", pd.Timestamp("2024-03-02 10:00", tz="UTC")), cfg.TARGET] == 1
    assert hourly.loc[("KDFW", pd.Timestamp("2024-03-01 18:00", tz="UTC")), "sched_arr_count"] == 1
    assert hourly.loc[("KDFW", pd.Timestamp("2024-03-01 18:00", tz="UTC")), cfg.TARGET] == 0


def _toy_sources(hours=48):
    grid = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    wx = pd.DataFrame({"icao": "KATL", "time_utc": grid, **{c: 0.0 for c in cfg.WEATHER_COLUMNS}})
    wx["temp_c"] = np.arange(hours, dtype=float)  # temp == hour index, so shifts are visible
    arrivals = pd.DataFrame({"icao": "KATL", cfg.TIME: grid, cfg.TARGET: np.arange(hours) * 10,
                             "sched_arr_count": 100.0, "cancellations": 0, "diversions": 0})
    return wx, arrivals


def test_panel_weather_is_shifted_by_h_and_oracle_is_hour_t():
    from src.models.aar_dataset import weather_availability_lag_hours
    wx, arrivals = _toy_sources()
    lag = weather_availability_lag_hours()
    assert lag == 1  # P1's config: a row stamped HH:00 is available at HH:00 + 1 h
    p = build_panel(wx, arrivals).set_index([cfg.TIME, "h"])
    t = pd.Timestamp("2024-01-01 20:00", tz="UTC")  # hour index 20
    for h in cfg.HORIZONS:
        row = p.loc[(t, h)]
        assert row["temp_c"] == 20 - h - lag                 # row stamped t - h - lag, available by t - h
        assert row[cfg.OBS_TIME] == t - pd.Timedelta(hours=h)
        assert row["oracle_temp_c"] == 20                    # target hour itself (evaluation only)
    validate_hourly_ok = validate_hourly(p.reset_index(), TZ)
    assert validate_hourly_ok["errors"] == []


def test_every_weather_stamp_is_available_by_forecast_time():
    """No weather row is used unless its stamp + availability lag <= t - h, for every panel row."""
    from src.models.aar_dataset import weather_availability_lag_hours
    wx, arrivals = _toy_sources()
    lag = pd.Timedelta(hours=weather_availability_lag_hours())
    p = build_panel(wx, arrivals)
    cutoff = p[cfg.TIME] - pd.to_timedelta(p["h"], unit="h")
    assert (p["weather_stamp_utc"] + lag <= cutoff).all()
    assert (p["weather_stamp_utc"] <= cutoff - lag).all()


def test_recent_arrivals_count_only_hours_ended_before_issue():
    wx, arrivals = _toy_sources()
    p = build_panel(wx, arrivals).set_index([cfg.TIME, "h"])
    row = p.loc[(pd.Timestamp("2024-01-01 20:00", tz="UTC"), 6)]  # issued at hour 14
    assert row["p2_ops_arr_last1h"] == 130                    # hour 13 only
    assert row["p2_ops_arr_last3h"] == 110 + 120 + 130        # hours 11-13, never 14+
    assert row["p2_ops_sched_last3h"] == 300


# ---------- validation and leakage guards ----------

def test_valid_panel_passes(panel):
    report = validate_hourly(panel, TZ)
    assert report["errors"] == []
    check(report)


def test_feature_obs_time_after_cutoff_is_rejected(panel):
    bad = panel.copy()
    bad.loc[bad.index[:5], cfg.OBS_TIME] += pd.Timedelta(minutes=1)
    report = validate_hourly(bad, TZ)
    assert any("leakage" in e for e in report["errors"])
    with pytest.raises(SchemaError):
        check(report)


def test_duplicate_keys_and_bad_horizon_rejected(panel):
    bad = pd.concat([panel.head(3), panel.head(1)])
    bad.loc[bad.index[1], "h"] = 5
    errors = validate_hourly(bad, TZ)["errors"]
    assert any("duplicate" in e for e in errors)
    assert any("unexpected horizons" in e for e in errors)


def test_leaky_columns_never_features(panel):
    features, dropped = select_feature_columns(panel)
    for col in cfg.EXCLUDED_COLUMNS | {"oracle_wind_speed_kmh", "oracle_arr_hint", cfg.TIME, cfg.OBS_TIME}:
        assert col not in features
        assert col in dropped
    assert {"icao", "h", "wind_speed_kmh", "crosswind_kmh", "sched_arr_count", "p2_ops_arr_last1h"} <= set(features)


def test_unknown_input_column_fails_closed(panel):
    with pytest.raises(ValueError, match="no known availability"):
        select_feature_columns(panel.assign(mystery_feature=1.0))


def test_preprocessor_roundtrip(panel, tmp_path):
    pre = AARPreprocessor(TZ).fit(panel)
    pre.save(tmp_path / "pre.json")
    loaded = AARPreprocessor.load(tmp_path / "pre.json")
    X = loaded.transform(panel.head(10))
    assert list(X.columns) == pre.feature_columns
    assert not any(c.startswith("oracle_") or c in cfg.EXCLUDED_COLUMNS for c in X.columns)
    with pytest.raises(KeyError):
        loaded.transform(panel.head(2).drop(columns=["wind_speed_kmh"]))


def test_split_is_chronological(panel):
    s = split_chronological(panel)
    assert s["train"][cfg.TIME].max() < s["val"][cfg.TIME].min()
    assert s["val"][cfg.TIME].max() < s["test"][cfg.TIME].min()
    assert (s["test"][cfg.TIME] >= cfg.TEST_START).all()
    assert sum(map(len, s.values())) == len(panel)


# ---------- baselines ----------

def test_persistence_reads_last_completed_hour_only():
    hours = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    rows = pd.DataFrame({"icao": "KATL", cfg.TIME: hours, "h": 3,
                         cfg.OBS_TIME: hours - pd.Timedelta(hours=3), cfg.TARGET: np.arange(48.0)})
    pred, _ = persistence(rows, rows)
    # Target hour index i, issued at i-3: last completed hour is i-4.
    expected = pd.Series(np.arange(48.0) - 4).where(lambda s: s >= 0)
    pd.testing.assert_series_equal(pred.reset_index(drop=True), expected, check_names=False)
    pred12, _ = persistence(rows.assign(h=12), rows)
    assert pred12.iloc[20] == 7.0


def test_hist_avg_matches_training_group_means(panel):
    s = split_chronological(panel)
    train = s["train"].drop_duplicates(["icao", cfg.TIME])
    local = train[cfg.TIME].dt.tz_convert(TZ["KATL"])
    mask = (train.icao == "KATL") & (local.dt.hour == 14) & (local.dt.dayofweek == 2)
    expected = train.loc[mask, cfg.TARGET].mean()
    pred = HistoricalAverage(TZ).fit(s["train"]).predict(s["test"])
    tl = s["test"][cfg.TIME].dt.tz_convert(TZ["KATL"])
    tmask = (s["test"].icao == "KATL") & (tl.dt.hour == 14) & (tl.dt.dayofweek == 2)
    assert tmask.any() and np.allclose(pred[tmask], expected)
    tampered = s["test"].assign(**{cfg.TARGET: 999})
    pd.testing.assert_series_equal(pred, HistoricalAverage(TZ).fit(s["train"]).predict(tampered))


def test_schedule_baseline_ratio_from_training_only(panel):
    s = split_chronological(panel)
    b = ScheduleBaseline().fit(s["train"])
    hourly = s["train"].drop_duplicates(["icao", cfg.TIME])
    katl = hourly[hourly.icao == "KATL"]
    assert np.isclose(b.ratio["KATL"], katl[cfg.TARGET].sum() / katl["sched_arr_count"].sum())
    tampered = pd.concat([s["train"], s["test"].assign(**{cfg.TARGET: 999})])
    assert not np.allclose(ScheduleBaseline().fit(tampered).ratio, b.ratio)  # would change if test leaked in


# ---------- derived inputs ----------

def test_schedule_context_reads_neighbouring_schedule_only(panel):
    ctx = add_schedule_context(panel, panel)
    katl = ctx[(ctx.icao == "KATL") & (ctx.h == 3)].sort_values(cfg.TIME).reset_index(drop=True)
    assert katl.loc[10, "p2_sched_p1"] == katl.loc[11, "sched_arr_count"]
    assert katl.loc[10, "p2_sched_m2"] == katl.loc[8, "sched_arr_count"]
    assert np.isnan(katl.loc[0, "p2_sched_m1"])
    scrambled = add_schedule_context(panel.assign(**{cfg.TARGET: -1}), panel).assign(**{cfg.TARGET: ctx[cfg.TARGET]})
    pd.testing.assert_frame_equal(ctx, scrambled)


def test_hub_exposure_uses_observed_weather_not_oracle():
    rows = pd.DataFrame({
        "icao": "KATL", cfg.TIME: pd.Timestamp("2024-01-01 12:00", tz="UTC"), "h": 3,
        "sched_arr_count": [20.0, 20.0, 30.0], "condition_code": [2.0, 25.0, 15.0], "precip_mm": [0.0, 5.0, 0.0],
        "wind_speed_kmh": [9.0, 18.0, 9.0], "crosswind_kmh": [30.0, 0.0, 0.0],
        "oracle_condition_code": [27.0, 1.0, 1.0],  # must be ignored
    })
    out = add_hub_exposure(rows)
    assert list(out.p2_hub_bad) == [1.0, 1.0, 1.0]       # crosswind / thunderstorm / snow
    assert list(out.p2_hub_severe) == [0.0, 1.0, 1.0]
    assert list(out.p2_hub_winter) == [0.0, 0.0, 1.0]
    assert list(out.p2_sched_x_hub_severe) == [0.0, 20.0, 30.0]
    pd.testing.assert_frame_equal(out.drop(columns="oracle_condition_code"),
                                  add_hub_exposure(rows.assign(oracle_condition_code=1.0)).drop(columns="oracle_condition_code"))


# ---------- metrics ----------

def test_metrics_cover_airports_and_horizons():
    preds = pd.DataFrame({"split": "test", "icao": ["KATL", "KDFW"] * 4, "h": [3, 3, 6, 6, 9, 9, 12, 12],
                          "y_true": [1.0] * 8, "p": [0.0] * 8})
    m = evaluate(preds, {"m": "p"})
    assert {("KATL", 3), ("KDFW", 12), ("ALL", "ALL"), ("KATL", "ALL"), ("ALL", 6)} <= set(zip(m.icao, m.h))
    assert np.allclose(m.rmse, 1.0) and np.allclose(m.mae, 1.0)


def test_metrics_weather_breakdown():
    preds = pd.DataFrame({"split": "test", "icao": "KATL", "h": 3, "y_true": [1.0, 0.0, 2.0],
                          "p": [1.0, 0.0, 0.0], "wx": ["normal", "normal", "severe"]})
    m = evaluate(preds, {"m": "p"}, weather_col="wx")
    sev = m[(m.weather == "severe") & (m.icao == "ALL")]
    assert sev.n.item() == 1 and sev.rmse.item() == 2.0


# ---------- end to end ----------

def test_end_to_end_train_and_predict(panel, tmp_path):
    from src.models.predict_aar import predict_aar
    from src.models.train_aar import run

    data = tmp_path / "panel.parquet"
    panel.to_parquet(data)
    metrics = run(data, tmp_path)
    for f in ("aar_preds.csv", "aar_metrics.csv", "aar_metrics_table.md", "aar_model/model.txt",
              "aar_model/model_p10.txt", "aar_model/model_p90.txt",
              "aar_model/preprocessor.json", "aar_model/metadata.json"):
        assert (tmp_path / f).exists()
    test_rows = metrics[(metrics.split == "test") & (metrics.icao != "ALL")]
    assert set(test_rows.icao) == {"KATL", "KDFW"}
    assert {"lightgbm", "schedule", "persistence", "hist_avg"} <= set(test_rows.model)
    assert test_rows[test_rows.model == "lightgbm"].coverage_p10_p90.notna().all()

    csv = pd.read_csv(tmp_path / "aar_preds.csv")
    assert list(csv.columns[:6]) == cfg.PREDS_COLUMNS
    assert ((csv.p10 <= csv.pred_arr_count + 1e-9) & (csv.pred_arr_count <= csv.p90 + 1e-9) & (csv.p10 >= 0)).all()

    out = predict_aar(icao="KDFW", h=[3, 12], start="2025-01-01", model_dir=tmp_path / "aar_model", data_path=data)
    assert list(out.columns) == cfg.PREDS_COLUMNS
    assert set(out.icao) == {"KDFW"} and set(out.h) == {3, 12}
    assert (out.pred_arr_count >= 0).all()
    # Purely predictive: scrambling everything not known at t - h (target, outcomes, oracle weather)
    # must leave the point forecast and the interval unchanged.
    future = [cfg.TARGET, "capacity_ratio", "mean_arr_delay", "share_delayed_45", "cancellations",
              "diversions"] + [c for c in panel.columns if c.startswith("oracle_")]
    scrambled = panel.assign(**{c: -7 for c in future})
    out2 = predict_aar(scrambled, icao="KDFW", h=[3, 12], start="2025-01-01", model_dir=tmp_path / "aar_model")
    pd.testing.assert_frame_equal(out, out2)


def test_final_model_never_fits_on_test_period(panel, tmp_path, monkeypatch):
    """Every row handed to LightGBM for fitting must be before 2025."""
    import lightgbm as lgb
    from src.models import train_aar

    seen = []
    real = lgb.Dataset

    def spy(data, label=None, **kw):
        if kw.get("reference") is None:  # training sets only, not validation sets
            seen.append(len(data))
        return real(data, label, **kw)

    monkeypatch.setattr(train_aar.lgb, "Dataset", spy)
    data = tmp_path / "panel.parquet"
    panel.to_parquet(data)
    train_aar.run(data, tmp_path)
    s = split_chronological(panel)
    assert max(seen) == len(s["train"]) + len(s["val"]) and len(s["test"]) > 0
