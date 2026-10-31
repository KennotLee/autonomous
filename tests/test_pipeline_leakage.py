"""Leakage guard: obs_* features for (T, h) must not change when anything at or
after the issue time T-h is altered. Run: python -m pytest tests"""
import numpy as np
import pandas as pd

from pipeline.config import get_airport
from pipeline.features import build_dataset, ops_state, wind_components
from pipeline.panel import build_panel

AP = get_airport("KATL")


def _synthetic(seed=0, days=10):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01 05:00", periods=24 * days, freq="h", tz="UTC", name="hour_utc")
    n = len(idx)
    wx = pd.DataFrame({
        "temp": rng.normal(20, 5, n), "dwpt": rng.normal(10, 5, n), "rhum": rng.uniform(30, 100, n),
        "prcp": rng.exponential(0.3, n), "wdir": rng.uniform(0, 360, n), "wspd": rng.uniform(0, 40, n),
        "wpgt": rng.uniform(0, 60, n), "pres": rng.normal(1015, 5, n), "vsby": rng.uniform(1, 16, n),
        "cldc": rng.integers(0, 9, n), "coco": rng.integers(1, 28, n)}, index=idx)
    sched = rng.integers(20, 80, n)
    act = np.minimum(sched, rng.integers(10, 80, n))
    panel = pd.DataFrame({
        "airport": AP.icao, "sched_arr": sched, "act_arr": act, "n_sev": rng.integers(0, 10, n),
        "n_cancel_arr": rng.integers(0, 3, n), "arr_delay_sum": rng.uniform(0, 900, n),
        "n_arr_completed": act, "sched_dep": sched, "act_dep": act,
        "dep_delay_sum": rng.uniform(0, 900, n), "n_dep_completed": act,
        "n_sev_act": rng.integers(0, 10, n), "arr_delay_sum_act": rng.uniform(0, 900, n),
        "dep_delay_sum_act": rng.uniform(0, 900, n)}, index=idx)
    return panel, wx


def test_obs_features_ignore_data_at_or_after_issue_time():
    panel, wx = _synthetic()
    T = panel.index[150]
    for h in (3, 7, 12):
        issue = T - pd.Timedelta(hours=h)
        p2, w2 = panel.copy(), wx.copy()
        late = p2.index >= issue
        num_p = p2.columns.drop("airport")
        p2.loc[late, num_p] = p2.loc[late, num_p] * 3 + 7
        w2.loc[late] = w2.loc[late] * 2 + 1
        a = build_dataset(panel, wx, AP, horizons=[h]).set_index("target_hour_utc").loc[T]
        b = build_dataset(p2, w2, AP, horizons=[h]).set_index("target_hour_utc").loc[T]
        obs = [c for c in a.index if c.startswith("obs_")]
        pd.testing.assert_series_equal(a[obs].astype(float), b[obs].astype(float), check_names=False)
        # sanity: the oracle columns DO see the change, so the test can detect leakage
        assert not np.allclose(a.filter(like="now_").astype(float), b.filter(like="now_").astype(float),
                               equal_nan=True)


def _synthetic_flights(seed=0, n=400):
    """Hub arrivals and departures on one April day, local hhmm like BTS."""
    rng = np.random.default_rng(seed)
    minute = rng.integers(6 * 60, 22 * 60, n)
    inbound = rng.random(n) < 0.5
    hhmm = (minute // 60) * 100 + minute % 60
    return pd.DataFrame({
        "FL_DATE": pd.Timestamp("2024-04-10"),
        "ORIGIN": np.where(inbound, "XYZ", AP.iata), "DEST": np.where(inbound, AP.iata, "XYZ"),
        "CRS_DEP_TIME": np.where(inbound, 500, hhmm), "CRS_ARR_TIME": np.where(inbound, hhmm, 2330),
        "CRS_ELAPSED_TIME": np.where(inbound, minute - 300, 23 * 60 + 30 - minute).astype(float),
        "DEP_DELAY": rng.integers(-10, 200, n).astype(float),
        "ARR_DELAY": rng.integers(-10, 200, n).astype(float),
        "CANCELLED": 0.0, "DIVERTED": 0.0})


def test_hub_state_ignores_flights_not_completed_by_issue_time():
    """A flight scheduled before the issue time but landing / leaving after it
    must not move the hub state of any hour that has closed by then."""
    fl = _synthetic_flights()
    issue_local = pd.Timestamp("2024-04-10 15:00")
    issue = issue_local.tz_localize(AP.tz).tz_convert("UTC")
    inbound = fl.DEST == AP.iata
    sched = fl.FL_DATE + pd.to_timedelta(np.where(inbound, fl.CRS_ARR_TIME, fl.CRS_DEP_TIME) // 100 * 60
                                         + np.where(inbound, fl.CRS_ARR_TIME, fl.CRS_DEP_TIME) % 100, unit="min")
    delay = np.where(inbound, fl.ARR_DELAY, fl.DEP_DELAY)
    pending = sched + pd.to_timedelta(delay, unit="min") >= issue_local
    assert (pending & (sched < issue_local)).sum() > 10  # the case that leaked
    fl2 = fl.copy()  # pending flights end up even later: unknowable at issue time
    fl2.loc[pending & inbound, "ARR_DELAY"] += 120
    fl2.loc[pending & ~inbound, "DEP_DELAY"] += 120

    p1, p2 = build_panel(fl, AP), build_panel(fl2, AP)
    closed = p1.index + pd.Timedelta(hours=1) <= issue
    cols = ["cap_ratio_3h", "sev_rate_3h", "cancel_rate_3h", "arr_delay_mean", "dep_delay_mean"]
    pd.testing.assert_frame_equal(ops_state(p1)[cols][closed], ops_state(p2)[cols][closed])
    # sanity: scheduled-hour buckets (targets only) DO change, so the test can see this leak
    assert not p1.arr_delay_sum[closed].equals(p2.arr_delay_sum[closed])
    assert not p1.dep_delay_sum[closed].equals(p2.dep_delay_sum[closed])


def test_wind_components():
    v = pd.Series([10.0, 10.0, 10.0, 0.0])
    d = pd.Series([80.0, 170.0, 260.0, np.nan])  # head-on, pure cross, pure tail, calm
    cross, head, tail = wind_components(v, d, 80.0)
    np.testing.assert_allclose(cross, [0, 10, 0, 0], atol=1e-9)
    np.testing.assert_allclose(head, [10, 0, -10, 0], atol=1e-9)
    np.testing.assert_allclose(tail, [0, 0, 10, 0], atol=1e-9)
