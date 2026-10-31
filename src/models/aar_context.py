"""Extra leakage-safe inputs derived inside the panel.

* Neighbouring-hour schedule: scheduled flights often land outside their scheduled hour, so
  sched_arr_count at t-2..t+2 helps. The schedule is published in advance, so t+k is not leakage.
* Weather exposure: weather acts on arrivals through delays, cancellations, diversions and reduced
  acceptance rates. These inputs say how many scheduled flights are exposed to bad weather, rather
  than leaving the model to discover the weather x traffic interaction alone. They use only the hub
  weather observed at t - h (the panel's shifted weather columns), never oracle_*.
"""
import pandas as pd

from . import aar_config as cfg

SCHED = "sched_arr_count"
SCHED_OFFSETS = (-2, -1, 1, 2)

# Thresholds in the weather file's units (km/h). Condition codes (data guide, Meteostat): 1-2 clear,
# 3-4 cloudy, 5-6 fog, 7-9 rain, 10-13 freezing rain/sleet, 14-16 snow, 17-22 showers,
# 23-27 lightning/hail/thunderstorm.
BAD_COCO, BAD_PRCP_MM, BAD_WSPD_KMH, BAD_CROSSWIND_KMH = 7, 1.0, 37.0, 28.0  # 37 km/h = 20 kt, 28 = 15 kt
SEVERE_COCO, SEVERE_PRCP_MM, SEVERE_WSPD_KMH = 23, 4.0, 46.0                 # 46 km/h = 25 kt
WINTER_COCO = {10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}

HUB_EXPOSURE_COLUMNS = [
    "p2_hub_bad", "p2_hub_severe", "p2_hub_winter",
    "p2_sched_x_hub_bad", "p2_sched_x_hub_severe", "p2_sched_m1_x_hub_bad",
]


def sched_offset_name(k: int) -> str:
    return f"p2_sched_{'m' if k < 0 else 'p'}{abs(k)}"


def weather_flags(coco: pd.Series, prcp: pd.Series, wspd: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(bad, severe, winter) as 0/1 floats. Missing values count as not bad."""
    winter = coco.isin(WINTER_COCO)
    bad = (coco >= BAD_COCO) | (prcp >= BAD_PRCP_MM) | (wspd >= BAD_WSPD_KMH)
    severe = winter | (coco >= SEVERE_COCO) | (prcp >= SEVERE_PRCP_MM) | (wspd >= SEVERE_WSPD_KMH)
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
    """Hub weather flags from the weather observed at t - h, and scheduled flights exposed to it."""
    needed = {"condition_code", "precip_mm", "wind_speed_kmh", SCHED}
    if not needed <= set(df.columns):
        return df
    df = df.copy()
    wspd = df[["wind_speed_kmh", "wind_max_3h_kmh"]].max(axis=1) if "wind_max_3h_kmh" in df else df["wind_speed_kmh"]
    bad, severe, winter = weather_flags(df["condition_code"], df["precip_mm"], wspd)
    if "crosswind_kmh" in df:
        bad = bad.where(~(df["crosswind_kmh"] >= BAD_CROSSWIND_KMH), 1.0)
    df["p2_hub_bad"], df["p2_hub_severe"], df["p2_hub_winter"] = bad, severe, winter
    df["p2_sched_x_hub_bad"] = df[SCHED] * bad
    df["p2_sched_x_hub_severe"] = df[SCHED] * severe
    if sched_offset_name(-1) in df:
        df["p2_sched_m1_x_hub_bad"] = df[sched_offset_name(-1)] * bad
    return df


def add_context(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Schedule context and hub weather exposure."""
    return add_hub_exposure(add_schedule_context(df, panel))
