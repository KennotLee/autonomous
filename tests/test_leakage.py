"""Leakage test for data/processed/event_features.parquet. Fails loudly (AssertionError, exit 1).

Organiser rule: only weather observed at least 3 h before the scheduled time being predicted.
Checks, for every row and every h:
 1. feature_obs_time and other_feature_obs_time <= sched_time_utc - h.
 2. Availability lag: in the hourly weather table, avail_time == time_utc + availability_lag_hours (>= 1 h),
    because the file's time_utc can precede the METAR it holds by up to ~52 min.
 3. Provenance: if an obs time is missing, every weather feature from that source is NaN.
 4. Recompute: for a random sample of rows, every weather feature equals the value recomputed
    independently from data/interim/hourly_weather_features.parquet using only rows available <= t - h.
 5. No wpgt_* / snow depth columns; oracle_* columns are only the flights-CSV comparison columns.
 6. Self-test: an injected violation must be caught.
Run: .venv/bin/python tests/test_leakage.py   (or pytest tests/test_leakage.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(REPO / "config" / "airports.yaml"))
EVENTS = REPO / "data" / "processed" / "event_features.parquet"
WX = REPO / "data" / "interim" / "hourly_weather_features.parquet"
TOL = pd.Timedelta(hours=CFG["weather"]["asof_tolerance_hours"])
LAG = pd.Timedelta(hours=CFG["weather"]["availability_lag_hours"])
N_SAMPLE = 500
NON_FEATURE = {"airport_iata", "time_utc", "avail_time", "has_obs", "icao"}


def _load():
    ev = pd.read_parquet(EVENTS)
    wx = pd.read_parquet(WX)
    wx["icao"] = wx.airport_iata.map({v["iata"]: k for k, v in CFG["airports"].items()})
    return ev, wx


def _feat_cols(wx):
    return [c for c in wx.columns if c not in NON_FEATURE]


def check_obs_times(ev, cols, name):
    cutoff = ev["sched_time_utc"] - pd.to_timedelta(ev["h"], unit="h")
    for c in cols:
        bad = ev[c].notna() & (ev[c] > cutoff)
        if bad.any():
            raise AssertionError(f"[{name}] {bad.sum()} rows with {c} > sched_time_utc - h:\n"
                                 f"{ev.loc[bad, ['event_id', 'sched_time_utc', 'h', c]].head(5).to_string()}")


def test_obs_times():
    ev, _ = _load()
    check_obs_times(ev, ["feature_obs_time", "other_feature_obs_time"], "events")
    assert (ev["h"] >= 3).all(), "horizon below the organiser's 3 h minimum"


def test_availability_lag():
    _, wx = _load()
    assert LAG >= pd.Timedelta(hours=1), "availability lag must be >= 1 h (time_utc can precede the METAR)"
    d = wx["avail_time"] - wx["time_utc"]
    assert (d == LAG).all(), "avail_time != time_utc + availability_lag_hours"


def test_provenance():
    ev, wx = _load()
    for obs, prefix in (("feature_obs_time", ""), ("other_feature_obs_time", "other_")):
        cols = [prefix + c for c in _feat_cols(wx)]
        orphan = ev[obs].isna() & ev[cols].notna().any(axis=1)
        assert not orphan.any(), f"{orphan.sum()} rows have {prefix}features but no {obs}"


def test_recompute_sample(seed=0):
    ev, wx = _load()
    cols = _feat_cols(wx)
    by_icao = {k: g.sort_values("avail_time") for k, g in wx[wx.has_obs].groupby("icao")}
    s = ev.sample(min(N_SAMPLE, len(ev)), random_state=seed)
    for _, r in s.iterrows():
        cutoff = r["sched_time_utc"] - pd.Timedelta(hours=int(r["h"]))
        for ap_col, prefix in (("airport", ""), ("other_airport", "other_")):
            g = by_icao.get(r[ap_col])
            w = None if g is None else g[(g.avail_time <= cutoff) & (g.avail_time >= cutoff - TOL)]
            exp = w.iloc[-1] if w is not None and len(w) else None
            for c in cols:
                got, want = r[prefix + c], (np.nan if exp is None else exp[c])
                if not ((pd.isna(got) and pd.isna(want)) or np.isclose(got, want)):
                    raise AssertionError(f"{prefix}{c} mismatch for {r['event_id']} h={r['h']}: "
                                         f"stored {got}, recomputed (avail <= t-h) {want}")


def test_forbidden_columns():
    ev, _ = _load()
    bad = [c for c in ev.columns if "wpgt" in c or c.endswith("snow") or "snow_ORIGIN" in c or "snow_DEST" in c]
    assert not bad, f"placeholder columns present: {bad}"
    oracle = [c for c in ev.columns if c.startswith("oracle_")]
    assert oracle and all(c.split("oracle_")[1] in {"temp", "prcp", "wspd", "pres", "coco"} for c in oracle)


def test_self_check_catches_violation():
    ev, _ = _load()
    bad = ev.head(10).copy()
    bad.loc[bad.index[0], "feature_obs_time"] = bad.loc[bad.index[0], "sched_time_utc"]
    try:
        check_obs_times(bad, ["feature_obs_time"], "selftest")
    except AssertionError:
        return
    raise AssertionError("self-test: injected leakage was NOT detected")


if __name__ == "__main__":
    tests = [test_obs_times, test_availability_lag, test_provenance, test_recompute_sample,
             test_forbidden_columns, test_self_check_catches_violation]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            sys.exit(1)
    print("ALL LEAKAGE CHECKS PASSED")
