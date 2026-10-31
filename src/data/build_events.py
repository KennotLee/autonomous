"""Join events with leakage-safe weather from the organiser hourly file -> event_features table.

For each event and h in config horizons: cutoff = sched_time_utc - h. Weather comes from the latest row
of the event airport's hourly table with avail_time (= time_utc + 1 h) <= cutoff, at most
asof_tolerance_hours older (else NaN). feature_obs_time = that row's avail_time, so
feature_obs_time <= sched_time_utc - h always. other_* = same for other_airport (only when the hourly
file covers it, i.e. ATL<->DFW). oracle_* = flights-CSV weather at event time (comparison only).
"""
import numpy as np
import pandas as pd

from .hourly_weather import feature_names

ID_COLS = ["event_id", "flight_key", "event_type", "airport", "other_airport", "sched_time_local",
           "sched_time_utc", "h", "feature_obs_time"]
CAL_COLS = ["hour", "dow", "month", "carrier"]
TARGET_COLS = ["delay_min", "label_45", "cancelled", "diverted"]


def _asof(keys: pd.DataFrame, wx: pd.DataFrame, cols: list, tol_h: float) -> pd.DataFrame:
    """keys: row_id, icao, cutoff -> row_id, avail_time, cols (last row with avail_time <= cutoff)."""
    right = wx.loc[wx.has_obs, ["icao", "avail_time"] + cols].sort_values("avail_time")
    left = keys.sort_values("cutoff")
    out = pd.merge_asof(left, right, left_on="cutoff", right_on="avail_time", by="icao",
                        direction="backward", tolerance=pd.Timedelta(hours=tol_h))
    return keys[["row_id"]].merge(out.drop(columns=["icao", "cutoff"]), on="row_id", how="left")


def build_event_features(ev: pd.DataFrame, wx: pd.DataFrame, hubs_all: dict, cfg: dict) -> pd.DataFrame:
    iata_to_icao = {v["iata"]: k for k, v in hubs_all.items()}
    wx = wx.assign(icao=wx.airport_iata.map(iata_to_icao)).dropna(subset=["icao"])
    fcols = feature_names(cfg)
    tol = cfg["weather"]["asof_tolerance_hours"]

    p = ev.merge(pd.DataFrame({"h": cfg["horizons"]}), how="cross")
    p["row_id"] = np.arange(len(p))
    cutoff = p.sched_time_utc - pd.to_timedelta(p.h, unit="h")

    a = _asof(pd.DataFrame({"row_id": p.row_id, "icao": p.airport, "cutoff": cutoff}), wx, fcols, tol)
    a = a.rename(columns={"avail_time": "feature_obs_time"})
    o = _asof(pd.DataFrame({"row_id": p.row_id, "icao": p.other_airport, "cutoff": cutoff}), wx, fcols, tol)
    o = o.rename(columns={"avail_time": "other_feature_obs_time", **{c: f"other_{c}" for c in fcols}})
    p = p.merge(a, on="row_id").merge(o, on="row_id")

    other = ["other_feature_obs_time"] + [f"other_{c}" for c in fcols]
    oracle = [c for c in p.columns if c.startswith("oracle_")]
    order = ID_COLS + fcols + other + CAL_COLS + TARGET_COLS + oracle
    return p[order].sort_values(["sched_time_utc", "event_id", "h"]).reset_index(drop=True)
