"""Extra leakage-safe inputs built from P1's files.

* Neighbouring-hour schedule: ~30% of scheduled flights land outside their scheduled hour, so
  sched_arr_count at t-2..t+2 helps. The schedule is published in advance, so t+k is not leakage.
* Origin-airport weather: inbound flights are delayed by weather where they depart, not only at
  the hub. Aggregated per (hub, arrival hour t, h) from flight_features.parquet, keeping only
  observations at or before t - h.
"""
from pathlib import Path

import pandas as pd

from . import aar_config as cfg

SCHED = "sched_arr_count"
SCHED_OFFSETS = (-2, -1, 1, 2)
ORIGIN_COLUMNS = [
    "p2_org_n_wx", "p2_org_prcp_max", "p2_org_prcp3h_max", "p2_org_wspd_max",
    "p2_org_wspd3h_max", "p2_org_coco_max", "p2_org_bad_share",
]
# Meteostat coco >= 5 is fog, rain, snow, sleet or thunderstorm.
BAD_COCO, BAD_PRCP_MM, BAD_WSPD_KT = 5, 1.0, 20.0


def sched_offset_name(k: int) -> str:
    return f"p2_sched_{'m' if k < 0 else 'p'}{abs(k)}"


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
    """Per (icao, target_hour_utc, h): weather at the origins of flights scheduled to land in hour t."""
    f = flights[flights.dest_icao.isin(hubs)].copy()
    f[cfg.TIME] = f["sched_arr_utc"].dt.floor("h")
    cutoff = f[cfg.TIME] - pd.to_timedelta(f["h"], unit="h")
    f = f[f["origin_feature_obs_time"].notna() & (f["origin_feature_obs_time"] <= cutoff)]
    f["bad"] = (
        (f["origin_coco"] >= BAD_COCO) | (f["origin_prcp_mm"] >= BAD_PRCP_MM) | (f["origin_wspd_kt"] >= BAD_WSPD_KT)
    ).astype(float)
    g = f.rename(columns={"dest_icao": "icao"}).groupby(["icao", cfg.TIME, "h"])
    out = pd.DataFrame({
        "p2_org_n_wx": g.size(),
        "p2_org_prcp_max": g["origin_prcp_mm"].max(),
        "p2_org_prcp3h_max": g["origin_prcp_sum_3h_mm"].max(),
        "p2_org_wspd_max": g["origin_wspd_kt"].max(),
        "p2_org_wspd3h_max": g["origin_wspd_max_3h_kt"].max(),
        "p2_org_coco_max": g["origin_coco"].max(),
        "p2_org_bad_share": g["bad"].mean(),
    })
    return out.reset_index()


def add_context(df: pd.DataFrame, panel: pd.DataFrame, flights: pd.DataFrame | None) -> pd.DataFrame:
    """Add schedule context, and origin weather when the flight file is available.

    Hours with no inbound flights get p2_org_n_wx = 0 and NaN weather (LightGBM handles NaN).
    """
    df = add_schedule_context(df, panel)
    if flights is None:
        return df
    hubs = sorted(df["icao"].unique())
    df = df.merge(origin_weather(flights, hubs), on=["icao", cfg.TIME, "h"], how="left")
    df["p2_org_n_wx"] = df["p2_org_n_wx"].fillna(0)
    return df
