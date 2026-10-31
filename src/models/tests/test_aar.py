"""AAR regression tests on a SYNTHETIC panel shaped like the team contract.

Synthetic numbers only exercise the code paths; they say nothing about real model quality.
Run from the repo root:  python -m pytest src/models/tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src.models import aar_config as cfg
from src.models.aar_baselines import HistoricalAverage, persistence
from src.models.aar_context import add_schedule_context, origin_weather
from src.models.aar_data import SchemaError, check, load_airport_timezones, split_chronological, validate_hourly
from src.models.aar_features import AARPreprocessor, select_feature_columns
from src.models.aar_metrics import evaluate

TZ = load_airport_timezones()


def make_panel(start="2024-05-01", end="2025-02-15", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    frames = []
    for icao in ("KATL", "KDFW"):
        sched = rng.poisson(1.0 + (hours.hour % 24 > 10), len(hours)).astype(float)
        wind = rng.gamma(2.0, 8.0, len(hours))
        arr = rng.poisson(np.clip(sched - 0.02 * wind, 0, None))
        for h in cfg.HORIZONS:
            frames.append(pd.DataFrame({
                "icao": icao,
                cfg.TIME: hours,
                "h": h,
                cfg.OBS_TIME: hours - pd.Timedelta(hours=h),
                "wspd_kt": wind + rng.normal(0, 2, len(hours)),
                "crosswind_kt": rng.normal(0, 5, len(hours)),
                "sched_arr_count": sched,
                cfg.TARGET: arr,
                # Leaky columns that must never become features:
                "capacity_ratio": arr / 3,
                "mean_arr_delay": rng.normal(10, 5, len(hours)),
                "share_delayed_45": rng.random(len(hours)),
                "cancellations": rng.poisson(0.1, len(hours)),
                "diversions": rng.poisson(0.05, len(hours)),
                "oracle_wspd": wind,
                "oracle_arr_hint": arr,
            }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


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
    for col in cfg.EXCLUDED_COLUMNS | {"oracle_wspd", "oracle_arr_hint", cfg.TIME, cfg.OBS_TIME}:
        assert col not in features
        assert col in dropped
    assert {"icao", "h", "wspd_kt", "crosswind_kt", "sched_arr_count"} <= set(features)


def test_preprocessor_roundtrip_and_unseen_airport(panel, tmp_path):
    pre = AARPreprocessor(TZ).fit(panel)
    pre.save(tmp_path / "pre.json")
    loaded = AARPreprocessor.load(tmp_path / "pre.json")
    X = loaded.transform(panel.head(10))
    assert list(X.columns) == pre.feature_columns
    assert not any(c.startswith("oracle_") or c in cfg.EXCLUDED_COLUMNS for c in X.columns)
    with pytest.raises(KeyError):
        loaded.transform(panel.head(2).drop(columns=["wspd_kt"]))


def test_split_is_chronological(panel):
    s = split_chronological(panel)
    assert s["train"][cfg.TIME].max() < s["val"][cfg.TIME].min()
    assert s["val"][cfg.TIME].max() < s["test"][cfg.TIME].min()
    assert (s["test"][cfg.TIME] >= cfg.TEST_START).all()
    assert sum(map(len, s.values())) == len(panel)


def test_persistence_reads_last_completed_hour_only():
    hours = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    rows = pd.DataFrame({"icao": "KATL", cfg.TIME: hours, "h": 3,
                         cfg.OBS_TIME: hours - pd.Timedelta(hours=3), cfg.TARGET: np.arange(48.0)})
    pred, _ = persistence(rows, rows)
    # Target hour index i, issued at i-3: last completed hour is i-4.
    expected = pd.Series(np.arange(48.0) - 4).where(lambda s: s >= 0)
    pd.testing.assert_series_equal(pred.reset_index(drop=True), expected, check_names=False)

    # A lagging weather stamp must not move the arrivals reference point (t - h).
    rows2 = rows.copy()
    rows2[cfg.OBS_TIME] = rows2[cfg.OBS_TIME] - pd.Timedelta(hours=2)
    pred2, _ = persistence(rows2, rows)
    pd.testing.assert_series_equal(pred, pred2)

    # h = 12: issued at i-12, last completed hour is i-13.
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
    # Test labels play no part: wildly different test targets leave predictions unchanged.
    tampered = s["test"].assign(**{cfg.TARGET: 999})
    pd.testing.assert_series_equal(pred, HistoricalAverage(TZ).fit(s["train"]).predict(tampered))


def test_metrics_cover_airports_and_horizons():
    preds = pd.DataFrame({"split": "test", "icao": ["KATL", "KDFW"] * 4, "h": [3, 3, 6, 6, 9, 9, 12, 12],
                          "y_true": [1.0] * 8, "p": [0.0] * 8})
    m = evaluate(preds, {"m": "p"})
    assert {("KATL", 3), ("KDFW", 12), ("ALL", "ALL"), ("KATL", "ALL"), ("ALL", 6)} <= set(zip(m.icao, m.h))
    assert np.allclose(m.rmse, 1.0) and np.allclose(m.mae, 1.0)


def test_end_to_end_train_and_predict(panel, tmp_path):
    from src.models.predict_aar import predict_aar
    from src.models.train_aar import run

    data = tmp_path / "hourly_features.parquet"
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

    # Contract columns come first, in the agreed order.
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


def test_unknown_input_column_fails_closed(panel):
    with pytest.raises(ValueError, match="no known availability"):
        select_feature_columns(panel.assign(mystery_feature=1.0))


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
    data = tmp_path / "hourly_features.parquet"
    panel.to_parquet(data)
    train_aar.run(data, tmp_path)
    s = split_chronological(panel)
    assert max(seen) == len(s["train"]) + len(s["val"]) and len(s["test"]) > 0


def test_schedule_baseline_ratio_from_training_only(panel):
    from src.models.aar_baselines import ScheduleBaseline
    s = split_chronological(panel)
    b = ScheduleBaseline().fit(s["train"])
    before = b.ratio.copy()
    ScheduleBaseline().fit(s["train"])  # refit identical
    tampered = pd.concat([s["train"], s["test"].assign(**{cfg.TARGET: 99})])
    assert not np.allclose(ScheduleBaseline().fit(tampered).ratio, before)  # would change if test leaked in
    pd.testing.assert_series_equal(b.ratio, before)


def test_schedule_context_reads_neighbouring_schedule_only(panel):
    ctx = add_schedule_context(panel, panel)
    katl = ctx[(ctx.icao == "KATL") & (ctx.h == 3)].sort_values(cfg.TIME).reset_index(drop=True)
    assert katl.loc[10, "p2_sched_p1"] == katl.loc[11, "sched_arr_count"]
    assert katl.loc[10, "p2_sched_m2"] == katl.loc[8, "sched_arr_count"]
    assert np.isnan(katl.loc[0, "p2_sched_m1"])
    # Built from the schedule, never from realised arrivals.
    pd.testing.assert_frame_equal(ctx, add_schedule_context(panel.assign(**{cfg.TARGET: -1}), panel).assign(**{cfg.TARGET: ctx[cfg.TARGET]}))


def test_origin_weather_drops_observations_after_cutoff():
    t = pd.Timestamp("2024-03-01 12:00", tz="UTC")
    flights = pd.DataFrame({
        "dest_icao": "KATL", "sched_arr_utc": t + pd.Timedelta(minutes=50), "h": 3,
        "origin_feature_obs_time": [t - pd.Timedelta(hours=3), t - pd.Timedelta(hours=2)],
        "origin_prcp_mm": [0.0, 9.0], "origin_prcp_sum_3h_mm": 0.0, "origin_wspd_kt": 5.0,
        "origin_wspd_max_3h_kt": 5.0, "origin_coco": 2.0,
    })
    out = origin_weather(flights, ["KATL"])
    assert out.loc[0, "p2_org_n_wx"] == 1 and out.loc[0, "p2_org_prcp_max"] == 0.0


def test_hub_exposure_uses_observed_weather_not_oracle():
    from src.models.aar_context import add_hub_exposure
    rows = pd.DataFrame({
        "icao": "KATL", cfg.TIME: pd.Timestamp("2024-01-01 12:00", tz="UTC"), "h": 3,
        "sched_arr_count": [2.0, 2.0, 3.0], "coco": [2.0, 25.0, 15.0], "prcp_mm": [0.0, 5.0, 0.0],
        "wspd_kt": [5.0, 10.0, 5.0], "crosswind_kt": [16.0, 0.0, 0.0],
        "oracle_coco": [27.0, 1.0, 1.0],  # must be ignored
    })
    out = add_hub_exposure(rows)
    assert list(out.p2_hub_bad) == [1.0, 1.0, 1.0]       # crosswind / thunderstorm / snow
    assert list(out.p2_hub_severe) == [0.0, 1.0, 1.0]
    assert list(out.p2_hub_winter) == [0.0, 0.0, 1.0]
    assert list(out.p2_sched_x_hub_severe) == [0.0, 2.0, 3.0]
    pd.testing.assert_frame_equal(out.drop(columns="oracle_coco"),
                                  add_hub_exposure(rows.assign(oracle_coco=1.0)).drop(columns="oracle_coco"))


def test_origin_counts_for_previous_hour_respect_cutoff():
    t = pd.Timestamp("2024-03-01 12:00", tz="UTC")
    flights = pd.DataFrame({
        "dest_icao": "KATL", "sched_arr_utc": [t - pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=10)], "h": 3,
        "origin_feature_obs_time": [t - pd.Timedelta(hours=4), t - pd.Timedelta(hours=3)],
        "origin_prcp_mm": 0.0, "origin_prcp_sum_3h_mm": 0.0, "origin_wspd_kt": 5.0,
        "origin_wspd_max_3h_kt": 5.0, "origin_coco": [25.0, 2.0],
    })
    out = origin_weather(flights, ["KATL"]).set_index(cfg.TIME)
    assert out.loc[t, "p2_org_severe_n_m1"] == 1 and out.loc[t, "p2_org_severe_n"] == 0


def test_metrics_weather_breakdown():
    preds = pd.DataFrame({"split": "test", "icao": "KATL", "h": 3, "y_true": [1.0, 0.0, 2.0],
                          "p": [1.0, 0.0, 0.0], "wx": ["normal", "normal", "severe"]})
    m = evaluate(preds, {"m": "p"}, weather_col="wx")
    sev = m[(m.weather == "severe") & (m.icao == "ALL")]
    assert sev.n.item() == 1 and sev.rmse.item() == 2.0
