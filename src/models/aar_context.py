"""Extra leakage-safe inputs built from P1's files.

* Neighbouring-hour schedule: ~30% of scheduled flights land outside their scheduled hour, so
  sched_arr_count at t-2..t+2 helps. The schedule is published in advance, so t+k is not leakage.
* Weather exposure: weather acts on arrivals through delays, cancellations and diversions (snow/ice
  at the hub cancels ~2/3 of flights; thunderstorms push ~half of departures past 45 min late). These
  features say how many scheduled flights are exposed to bad weather, rather than leaving the model
  to discover the interaction from a handful of storm hours.
  - Hub: flags from the hub's weather observed at feature_obs_time (<= t - h), times the schedule.
  - Origin: weather where inbound flights depart, aggregated per (hub, arrival hour, h) from
    flight_features.parquet, for flights due in hour t and in t-1 (delayed t-1 flights spill into t).
    Only observations at or before t - h of the row's arrival hour are kept.
"""
from pathlib import Path

import pandas as pd

from . import aar_config as cfg

SCHED = "sched_arr_count"
SCHED_OFFSETS = (-2, -1, 1, 2)

# Meteostat condition codes: 7-8 rain, 9 heavy rain, 10-16 freezing rain/sleet/snow, 17-22 showers
# (19-22 sleet/snow showers), 23-27 lightning/hail/thunderstorm/storm.
BAD_COCO, BAD_PRCP_MM, BAD_WSPD_KT, BAD_CROSSWIND_KT = 7, 1.0, 20.0, 15.0
SEVERE_COCO, SEVERE_PRCP_MM, SEVERE_WSPD_KT = 23, 4.0, 25.0
WINTER_COCO = {10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}

HUB_EXPOSURE_COLUMNS = [
    "p2_hub_bad", "p2_hub_severe", "p2_hub_winter",
    "p2_sched_x_hub_bad", "p2_sched_x_hub_severe", "p2_sched_m1_x_hub_bad",
]
ORIGIN_COLUMNS = [
    "p2_org_n_wx", "p2_org_prcp_max", "p2_org_prcp3h_max", "p2_org_wspd_max", "p2_org_wspd3h_max",
    "p2_org_coco_max", "p2_org_bad_share", "p2_org_bad_n", "p2_org_severe_n", "p2_org_winter_n",
    "p2_org_bad_n_m1", "p2_org_severe_n_m1",
]


def sched_offset_name(k: int) -> str:
    return f"p2_sched_{'m' if k < 0 else 'p'}{abs(k)}"


def weather_flags(coco: pd.Series, prcp: pd.Series, wspd: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(bad, severe, winter) as 0/1 floats. Missing values count as not bad."""
    winter = coco.isin(WINTER_COCO)
    bad = (coco >= BAD_COCO) | (prcp >= BAD_PRCP_MM) | (wspd >= BAD_WSPD_KT)
    severe = winter | (coco >= SEVERE_COCO) | (prcp >= SEVERE_PRCP_MM) | (wspd >= SEVERE_WSPD_KT)
    return bad.astype(float), severe.astype(float), winter.astype(float)


def add_schedule_context(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """sched_arr_count of the same airport k hours before/after the target hour (NaN off-grid)."""
    if SCHED not in df:
        return df
    df = df.copy()
    sched = panel.drop_duplicates(["icao", cfg.TIME]).set_index(["icao", cfg.TIME])[SCHED]
    for k in SCHED_OFFSETS:
        idx = pd.MultiIndex.from_arrays([df["icao"], df[cfg.TIME] + pd.Timedelta(hours=k)])
        df[sched_offset_name(k)] = sched.reindex(idx).to_numpy(dtype="float64")
    return df


def add_hub_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Hub weather flags from P1's observed (t - h) weather, and scheduled flights exposed to it."""
    needed = {"coco", "prcp_mm", "wspd_kt", SCHED}
    if not needed <= set(df.columns):
        return df
    df = df.copy()
    wspd = df[["wspd_kt", "wspd_max_3h_kt"]].max(axis=1) if "wspd_max_3h_kt" in df else df["wspd_kt"]
    bad, severe, winter = weather_flags(df["coco"], df["prcp_mm"], wspd)
    if "crosswind_kt" in df:
        bad = bad.where(~(df["crosswind_kt"] >= BAD_CROSSWIND_KT), 1.0)
    df["p2_hub_bad"], df["p2_hub_severe"], df["p2_hub_winter"] = bad, severe, winter
    df["p2_sched_x_hub_bad"] = df[SCHED] * bad
    df["p2_sched_x_hub_severe"] = df[SCHED] * severe
    if sched_offset_name(-1) in df:
        df["p2_sched_m1_x_hub_bad"] = df[sched_offset_name(-1)] * bad
    return df


def load_flights(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    cols = ["dest_icao", "sched_arr_utc", "h", "origin_feature_obs_time", "origin_prcp_mm",
            "origin_prcp_sum_3h_mm", "origin_wspd_kt", "origin_wspd_max_3h_kt", "origin_coco"]
    f = pd.read_parquet(path, columns=cols)
    for c in ("sched_arr_utc", "origin_feature_obs_time"):
        f[c] = pd.to_datetime(f[c], utc=True)
    return f


def origin_weather(flights: pd.DataFrame, hubs: list[str]) -> pd.DataFrame:
    """Per (icao, target_hour_utc, h): weather at the origins of flights scheduled to land in hour t,
    plus bad/severe counts for flights due in t - 1."""
    f = flights[flights.dest_icao.isin(hubs)].copy()
    f[cfg.TIME] = f["sched_arr_utc"].dt.floor("h")
    cutoff = f[cfg.TIME] - pd.to_timedelta(f["h"], unit="h")
    f = f[f["origin_feature_obs_time"].notna() & (f["origin_feature_obs_time"] <= cutoff)]
    f["bad"], f["severe"], f["winter"] = weather_flags(f["origin_coco"], f["origin_prcp_mm"], f["origin_wspd_kt"])
    g = f.rename(columns={"dest_icao": "icao"}).groupby(["icao", cfg.TIME, "h"])
    out = pd.DataFrame({
        "p2_org_n_wx": g.size(),
        "p2_org_prcp_max": g["origin_prcp_mm"].max(),
        "p2_org_prcp3h_max": g["origin_prcp_sum_3h_mm"].max(),
        "p2_org_wspd_max": g["origin_wspd_kt"].max(),
        "p2_org_wspd3h_max": g["origin_wspd_max_3h_kt"].max(),
        "p2_org_coco_max": g["origin_coco"].max(),
        "p2_org_bad_share": g["bad"].mean(),
        "p2_org_bad_n": g["bad"].sum(),
        "p2_org_severe_n": g["severe"].sum(),
        "p2_org_winter_n": g["winter"].sum(),
    }).reset_index()
    # Flights due in t-1 (their obs are <= (t-1) - h, so also <= t - h) shifted onto hour t.
    prev = out[["icao", cfg.TIME, "h", "p2_org_bad_n", "p2_org_severe_n"]].copy()
    prev[cfg.TIME] = prev[cfg.TIME] + pd.Timedelta(hours=1)
    prev = prev.rename(columns={"p2_org_bad_n": "p2_org_bad_n_m1", "p2_org_severe_n": "p2_org_severe_n_m1"})
    return out.merge(prev, on=["icao", cfg.TIME, "h"], how="outer")


def add_context(df: pd.DataFrame, panel: pd.DataFrame, flights: pd.DataFrame | None) -> pd.DataFrame:
    """Schedule context and hub weather exposure; origin weather when the flight file is given.

    Hours with no inbound flights get zero counts and NaN weather maxima (LightGBM handles NaN).
    """
    df = add_hub_exposure(add_schedule_context(df, panel))
    if flights is None:
        return df
    hubs = sorted(df["icao"].unique())
    org = origin_weather(flights, hubs)
    org[cfg.TIME] = org[cfg.TIME].astype(df[cfg.TIME].dtype)
    df = df.merge(org, on=["icao", cfg.TIME, "h"], how="left")
    counts = ["p2_org_n_wx", "p2_org_bad_n", "p2_org_severe_n", "p2_org_winter_n", "p2_org_bad_n_m1", "p2_org_severe_n_m1"]
    df[counts] = df[counts].fillna(0)
    return df
