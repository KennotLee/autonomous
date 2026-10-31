"""Meteostat hourly weather: station lookup by ICAO, cached fetch, hourly feature table per airport.

Meteostat timestamps are UTC (dev.meteostat.net). Units from Meteostat: °C, mm, km/h, hPa, degrees.
We convert wind speeds to knots (÷1.852). Wind direction = meteorological "from" convention.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from .config import INTERIM, KMH_PER_KT

RAW_COLS = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres"]
# Base (non-rolling, non-component) feature columns, in output order.
BASE_FEATURES = ["temp_c", "dwpt_c", "rhum_pct", "prcp_mm", "wdir_deg", "wspd_kt", "wpgt_kt", "pres_hpa", "coco"]
WIND_COMPONENTS = ["headwind_kt", "crosswind_kt", "tailwind_kt",
                   "gust_headwind_kt", "gust_crosswind_kt", "gust_tailwind_kt"]


def rolling_feature_names(windows: list[int]) -> list[str]:
    out = []
    for w in windows:
        out += [f"wspd_max_{w}h_kt", f"wpgt_max_{w}h_kt", f"prcp_sum_{w}h_mm", f"pres_chg_{w}h_hpa"]
    return out


def resolve_airports(iatas, cfg: dict) -> pd.DataFrame:
    """Map BTS IATA codes -> ICAO, Meteostat station id, timezone, runway heading.

    Hubs come from cfg['airports']; others: 'K'+IATA then 'P'+IATA if present in Meteostat,
    else cfg['iata_icao_overrides']. Raises if any IATA stays unresolved or has no timezone.
    """
    from meteostat import Stations

    st = Stations().fetch()
    st = st[st.icao.notna()]
    by_icao = {r.icao: (idx, r.timezone) for idx, r in st.iterrows()}
    hubs = {v["iata"]: (k, v) for k, v in cfg["airports"].items()}
    overrides = cfg.get("iata_icao_overrides", {}) or {}
    rows = []
    for iata in sorted(set(iatas)):
        heading, use_ms, tz_cfg = np.nan, True, None
        if iata in hubs:
            icao, hcfg = hubs[iata]
            heading, tz_cfg = float(hcfg["runway_heading_deg"]), hcfg.get("timezone")
        elif iata in overrides:
            o = overrides[iata]
            icao, tz_cfg, use_ms = o["icao"], o.get("timezone"), o.get("meteostat", True)
        else:
            icao = next((c for c in ("K" + iata, "P" + iata) if c in by_icao), None)
        if icao is None:
            raise KeyError(f"Cannot resolve IATA {iata}; add it to iata_icao_overrides in config")
        sid, tz_ms = by_icao.get(icao, (None, None)) if use_ms else (None, None)
        tz = tz_cfg or tz_ms
        if tz is None:
            raise KeyError(f"No timezone for {iata}/{icao}; add timezone to config")
        if tz_cfg and tz_ms and tz_cfg != tz_ms:
            raise ValueError(f"Timezone mismatch for {icao}: config {tz_cfg} vs Meteostat {tz_ms}")
        rows.append({"iata": iata, "icao": icao, "station_id": sid, "timezone": tz,
                     "runway_heading_deg": heading})
    return pd.DataFrame(rows)


def fetch_station_hourly(station_id: str, start: str, end: str) -> pd.DataFrame:
    """Observations (model=False) for numeric vars + coco from model=True. Index: naive UTC hour."""
    from meteostat import Hourly

    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    obs = Hourly(station_id, s, e, model=False).fetch()
    mod = Hourly(station_id, s, e, model=True).fetch()
    out = obs.reindex(columns=RAW_COLS).astype("float64")
    out["coco"] = mod["coco"].astype("float64")  # aligned on time index
    out.index.name = "obs_time"
    return out


def fetch_all(airports: pd.DataFrame, start: str, end: str, workers: int = 8, refresh: bool = False) -> dict:
    """Fetch + cache raw hourly weather per ICAO in data/interim/weather_raw/{icao}.parquet."""
    cache = INTERIM / "weather_raw"
    cache.mkdir(parents=True, exist_ok=True)

    def one(row):
        p = cache / f"{row.icao}.parquet"
        if p.exists() and not refresh:
            return row.icao, pd.read_parquet(p)
        if row.station_id is None or pd.isna(row.station_id):
            return row.icao, None
        df = fetch_station_hourly(row.station_id, start, end)
        df.to_parquet(p)
        return row.icao, df

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, airports.itertuples()))


def wind_components(speed_kt, wdir_deg, heading_deg):
    """θ = wdir − heading; headwind = s·cosθ, crosswind = |s·sinθ|, tailwind = max(0, −headwind).
    Calm (speed 0) -> all components 0 even if direction missing."""
    th = np.deg2rad(wdir_deg - heading_deg)
    hw = speed_kt * np.cos(th)
    xw = np.abs(speed_kt * np.sin(th))
    calm = speed_kt == 0
    hw = np.where(calm, 0.0, hw)
    xw = np.where(calm, 0.0, xw)
    tw = np.where(np.isnan(hw), np.nan, np.maximum(0.0, -hw))
    return hw, xw, tw


def build_weather_features(raw: pd.DataFrame | None, heading_deg: float, windows: list[int],
                           start: str, end: str) -> pd.DataFrame:
    """Complete hourly grid (naive UTC) with base, rolling and wind-component features.

    Rolling windows are trailing: value at T uses obs in (T − w h, T]. `has_obs` marks hours with
    at least one of temp/wspd/pres observed; only those hours are eligible for as-of joins.
    """
    grid = pd.date_range(start, end, freq="h", name="obs_time", tz="UTC")
    if raw is None:
        raw = pd.DataFrame(index=grid, columns=RAW_COLS + ["coco"], dtype="float64")
    elif raw.index.tz is None:
        raw = raw.copy()
        raw.index = raw.index.tz_localize("UTC")  # Meteostat returns naive UTC
    r = raw.reindex(grid)
    f = pd.DataFrame(index=grid)
    f["temp_c"], f["dwpt_c"], f["rhum_pct"] = r.temp, r.dwpt, r.rhum
    f["prcp_mm"], f["wdir_deg"] = r.prcp, r.wdir
    f["wspd_kt"], f["wpgt_kt"] = r.wspd / KMH_PER_KT, r.wpgt / KMH_PER_KT
    f["pres_hpa"], f["coco"] = r.pres, r.coco
    f["has_obs"] = r[["temp", "wspd", "pres"]].notna().any(axis=1)
    for w in windows:
        roll = f.rolling(f"{w}h", min_periods=1)
        f[f"wspd_max_{w}h_kt"] = roll["wspd_kt"].max()
        f[f"wpgt_max_{w}h_kt"] = roll["wpgt_kt"].max()
        f[f"prcp_sum_{w}h_mm"] = roll["prcp_mm"].sum()  # all-NaN window -> NaN (min_periods=1)
        f[f"pres_chg_{w}h_hpa"] = f["pres_hpa"] - f["pres_hpa"].shift(w)  # grid is complete -> exact w h
    if np.isnan(heading_deg):
        for c in WIND_COMPONENTS:
            f[c] = np.nan
    else:
        f["headwind_kt"], f["crosswind_kt"], f["tailwind_kt"] = wind_components(f.wspd_kt, f.wdir_deg, heading_deg)
        f["gust_headwind_kt"], f["gust_crosswind_kt"], f["gust_tailwind_kt"] = wind_components(
            f.wpgt_kt, f.wdir_deg, heading_deg)
    return f.reset_index()
