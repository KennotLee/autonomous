"""Organiser hourly weather (data/raw/challenge5_hourly_weather_ATL_DFW.csv) -> per-airport feature table.

Source (data guide): Meteostat hourly (NOAA ISD), stations 72219 (ATL) and 72259 (DFW). Units as provided:
°C, mm, km/h, hPa, degrees (wind FROM, true); headwind/crosswind/tailwind already computed (km/h).
wind_gust_kmh is 100% empty in the file and is not used.

Availability: the file's time_utc can precede the METAR it holds by up to ~52 min, so each row is
treated as available at avail_time = time_utc + availability_lag_hours (config, 1 h). All lags/rolling
windows below look backwards from the row's own hour, so they are also known at avail_time.
"""
import pandas as pd

RENAME = {"temp_c": "temp", "precip_mm": "prcp", "wind_speed_kmh": "wspd", "pressure_hpa": "pres",
          "condition_code": "coco", "wind_dir_deg": "wdir", "headwind_kmh": "headwind",
          "crosswind_kmh": "crosswind", "tailwind_kmh": "tailwind"}
BASE = list(RENAME.values())
LAG_VARS = ["wspd", "crosswind", "tailwind", "prcp", "temp"]


def feature_names(cfg: dict) -> list[str]:
    w = cfg["weather"]
    lags = [f"{v}_lag{k}h" for k in w["lags_hours"] for v in LAG_VARS]
    return BASE + ["ts_flag", "snow_flag"] + lags + [f"pres_tend_{w['pressure_tendency_hours']}h",
                                                     f"prcp_sum_{w['precip_sum_hours']}h"]


def load_hourly(path, cfg: dict) -> pd.DataFrame:
    """Return long table: airport_iata, time_utc (tz-aware), avail_time, features. Complete hourly grid."""
    w = cfg["weather"]
    raw = pd.read_csv(path)
    raw["time_utc"] = pd.to_datetime(raw["time_utc"]).dt.tz_localize("UTC")
    parts = []
    for iata, g in raw.groupby("airport"):
        g = g.set_index("time_utc").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="h", name="time_utc")
        g = g[~g.index.duplicated()].reindex(grid)
        f = g[list(RENAME)].rename(columns=RENAME).astype("float64")
        f["ts_flag"] = f["coco"].isin(w["thunderstorm_codes"]).astype(float).where(f["coco"].notna())
        f["snow_flag"] = f["coco"].isin(w["snow_codes"]).astype(float).where(f["coco"].notna())
        for k in w["lags_hours"]:  # grid is complete -> shift(k) = exactly k hours earlier
            for v in LAG_VARS:
                f[f"{v}_lag{k}h"] = f[v].shift(k)
        p = w["pressure_tendency_hours"]
        f[f"pres_tend_{p}h"] = f["pres"] - f["pres"].shift(p)
        s = w["precip_sum_hours"]
        f[f"prcp_sum_{s}h"] = f["prcp"].rolling(f"{s}h", min_periods=1).sum()  # (T - s h, T]
        f["has_obs"] = g[["temp_c", "wind_speed_kmh", "pressure_hpa"]].notna().any(axis=1)
        f = f.reset_index()
        f.insert(0, "airport_iata", iata)
        f["avail_time"] = f["time_utc"] + pd.Timedelta(hours=w["availability_lag_hours"])
        parts.append(f)
    return pd.concat(parts, ignore_index=True)
