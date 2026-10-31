"""Leakage test for data/processed/*.parquet. Fails loudly (AssertionError, exit 1).

Checks, for every row and every h:
 1. feature_obs_time (and origin_/dest_feature_obs_time) <= t − h, where t = target_hour_utc (hourly)
    or sched_time_utc (flight).
 2. Provenance: if an obs time is missing, every non-oracle weather feature from that source is NaN.
 3. Recompute: for a random sample of rows, every non-oracle weather feature equals the value
    recomputed independently from data/interim/weather_features.parquet using only hours <= t − h.
 4. Self-test: an injected violation must be caught (guards against a vacuous test).
`oracle_*` columns are excluded by design (same-hour weather, oracle comparison only).
Run: .venv/bin/python tests/test_leakage.py   (or pytest tests/test_leakage.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
PROC, INTERIM = REPO / "data" / "processed", REPO / "data" / "interim"
CFG = yaml.safe_load(open(REPO / "config" / "airports.yaml"))
TOL = pd.Timedelta(hours=CFG["meteostat"]["asof_tolerance_hours"])
N_SAMPLE = 400


def _wx_cols(wx):
    return [c for c in wx.columns if c not in ("icao", "obs_time", "has_obs")]


def check_obs_times(df, t_col, obs_cols, name):
    cutoff = df[t_col] - pd.to_timedelta(df["h"], unit="h")
    for c in obs_cols:
        bad = df[c].notna() & (df[c] > cutoff)
        if bad.any():
            ex = df.loc[bad, [t_col, "h", c]].head(5).to_string()
            raise AssertionError(f"[{name}] {bad.sum()} rows with {c} > {t_col} − h:\n{ex}")


def check_provenance(df, obs_col, feat_cols, name):
    orphan = df[obs_col].isna() & df[feat_cols].notna().any(axis=1)
    if orphan.any():
        raise AssertionError(f"[{name}] {orphan.sum()} rows have features but no {obs_col}")


def recompute(df, wx, t_col, icao_col, prefix, name, seed=0):
    cols = _wx_cols(wx)
    obs = wx[wx.has_obs].set_index("icao")
    s = df.sample(min(N_SAMPLE, len(df)), random_state=seed)
    for _, r in s.iterrows():
        cutoff = r[t_col] - pd.Timedelta(hours=int(r["h"]))
        if r[icao_col] in obs.index:
            w = obs.loc[[r[icao_col]]]
            w = w[(w.obs_time <= cutoff) & (w.obs_time >= cutoff - TOL)]
        else:
            w = obs.iloc[0:0]
        exp = w.sort_values("obs_time").iloc[-1] if len(w) else None
        for c in cols:
            got = r[f"{prefix}{c}"]
            want = np.nan if exp is None else exp[c]
            if not ((pd.isna(got) and pd.isna(want)) or np.isclose(got, want)):
                raise AssertionError(f"[{name}] {prefix}{c} mismatch for {r[icao_col]} t={r[t_col]} "
                                     f"h={r['h']}: stored {got}, recomputed (≤ t−h) {want}")


def _load():
    return (pd.read_parquet(PROC / "hourly_features.parquet"),
            pd.read_parquet(PROC / "flight_features.parquet"),
            pd.read_parquet(INTERIM / "weather_features.parquet"))


def test_hourly():
    hr, _, wx = _load()
    check_obs_times(hr, "target_hour_utc", ["feature_obs_time"], "hourly")
    check_provenance(hr, "feature_obs_time", _wx_cols(wx), "hourly")
    recompute(hr, wx, "target_hour_utc", "icao", "", "hourly")


def test_flight():
    _, fl, wx = _load()
    obs_cols = ["feature_obs_time", "origin_feature_obs_time", "dest_feature_obs_time"]
    check_obs_times(fl, "sched_time_utc", obs_cols, "flight")
    for side in ("origin", "dest"):
        check_provenance(fl, f"{side}_feature_obs_time", [f"{side}_{c}" for c in _wx_cols(wx)], "flight")
        recompute(fl, wx, "sched_time_utc", f"{side}_icao", f"{side}_", f"flight/{side}")


def test_no_oracle_in_features():
    hr, fl, _ = _load()
    for df in (hr, fl):
        assert all(c.startswith("oracle_") for c in df.columns if "oracle" in c)


def test_self_check_catches_violation():
    hr, _, _ = _load()
    bad = hr.head(10).copy()
    bad.loc[bad.index[0], "feature_obs_time"] = bad.loc[bad.index[0], "target_hour_utc"]
    try:
        check_obs_times(bad, "target_hour_utc", ["feature_obs_time"], "selftest")
    except AssertionError:
        return
    raise AssertionError("self-test: injected leakage was NOT detected")


if __name__ == "__main__":
    tests = [test_hourly, test_flight, test_no_oracle_in_features, test_self_check_catches_violation]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            sys.exit(1)
    print("ALL LEAKAGE CHECKS PASSED")
