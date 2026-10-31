"""Build leakage-safe hourly and flight feature tables.

Rule: every non-oracle weather feature for target time t and horizon h comes from the last Meteostat
hour with observations at or before t − h (as-of backward join, tolerance from config), so
feature_obs_time ≤ t − h. Rolling features at obs time T use only (T − w, T].
`oracle_*` columns = weather at the target hour itself; for oracle comparison only (leaky by design).
"""
import numpy as np
import pandas as pd

from .weather import BASE_FEATURES, WIND_COMPONENTS, build_weather_features, rolling_feature_names


def feature_cols(windows):
    return BASE_FEATURES + rolling_feature_names(windows) + WIND_COMPONENTS


ORACLE_SRC = BASE_FEATURES + WIND_COMPONENTS


def weather_table(airports: pd.DataFrame, raw_by_icao: dict, cfg: dict) -> pd.DataFrame:
    m = cfg["meteostat"]
    parts = []
    for r in airports.itertuples():
        w = build_weather_features(raw_by_icao.get(r.icao), r.runway_heading_deg,
                                   m["rolling_windows_hours"], m["weather_start"], m["weather_end"])
        w.insert(0, "icao", r.icao)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def asof_join(keys: pd.DataFrame, wx: pd.DataFrame, cols: list, tol_h: float) -> pd.DataFrame:
    """keys: [row_id, icao, cutoff]. Returns [row_id, obs_time, cols] using last obs ≤ cutoff."""
    right = wx.loc[wx.has_obs, ["icao", "obs_time"] + cols].sort_values("obs_time")
    left = keys.dropna(subset=["cutoff"]).sort_values("cutoff")
    out = pd.merge_asof(left, right, left_on="cutoff", right_on="obs_time", by="icao",
                        direction="backward", tolerance=pd.Timedelta(hours=tol_h))
    return keys[["row_id"]].merge(out.drop(columns=["icao", "cutoff"]), on="row_id", how="left")


def exact_join(keys: pd.DataFrame, wx: pd.DataFrame, cols: list) -> pd.DataFrame:
    """keys: [row_id, icao, hour]. Weather stamped exactly at `hour` (oracle)."""
    right = wx[["icao", "obs_time"] + cols].rename(columns={"obs_time": "hour"})
    return keys.merge(right, on=["icao", "hour"], how="left")[["row_id"] + cols]


def _local_parts(ts_utc: pd.Series, tz: pd.Series, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=ts_utc.index, columns=[f"{prefix}hour_local", "dow_local", "month_local"])
    for z, idx in ts_utc.groupby(tz).groups.items():
        loc = ts_utc.loc[idx].dt.tz_convert(z)
        out.loc[idx, f"{prefix}hour_local"] = loc.dt.hour.values
        out.loc[idx, "dow_local"] = loc.dt.dayofweek.values  # 0 = Monday
        out.loc[idx, "month_local"] = loc.dt.month.values
    return out.astype("Int64")


def build_hourly(flights: pd.DataFrame, hubs: dict, wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["meteostat"]
    fcols = feature_cols(m["rolling_windows_hours"])
    d0, d1 = pd.Timestamp(flights.fl_date.min()), pd.Timestamp(flights.fl_date.max()) + pd.Timedelta(days=1)
    panels = []
    for icao, acfg in hubs.items():
        tz = acfg["timezone"]
        hours = pd.date_range(d0.tz_localize(tz).tz_convert("UTC"), d1.tz_localize(tz).tz_convert("UTC"),
                              freq="h", inclusive="left", name="target_hour_utc")
        arr = flights[flights.dest_icao == icao]
        landed = arr[(arr.cancelled == 0) & (arr.diverted == 0) & arr.wheels_on_utc.notna()]
        g_s = arr.groupby(arr.sched_arr_utc.dt.floor("h"))
        t = pd.DataFrame(index=hours)
        t["arr_count"] = landed.groupby(landed.wheels_on_utc.dt.floor("h")).size()
        t["sched_arr_count"] = g_s.size()
        t["cancellations"] = g_s.cancelled.sum()
        t["diversions"] = g_s.diverted.sum()  # by scheduled arrival hour, like cancellations
        cnt = ["arr_count", "sched_arr_count", "cancellations", "diversions"]
        t[cnt] = t[cnt].fillna(0).astype(int)
        t["mean_arr_delay"] = g_s.delay_min.mean()
        t["share_delayed_45"] = g_s.label_45.mean()
        t["capacity_ratio"] = t.arr_count / t.sched_arr_count.replace(0, np.nan)
        t = t.reset_index()
        t.insert(0, "icao", icao)
        cal = _local_parts(t.target_hour_utc, pd.Series(tz, index=t.index), "")
        t = pd.concat([t, cal], axis=1)
        t["hour_utc"] = t.target_hour_utc.dt.hour
        panels.append(t)
    base = pd.concat(panels, ignore_index=True)

    p = base.merge(pd.DataFrame({"h": m["horizons"]}), how="cross")
    p["row_id"] = np.arange(len(p))
    keys = p[["row_id", "icao"]].assign(cutoff=p.target_hour_utc - pd.to_timedelta(p.h, unit="h"))
    feats = asof_join(keys, wx, fcols, m["asof_tolerance_hours"]).rename(columns={"obs_time": "feature_obs_time"})
    orc = exact_join(p[["row_id", "icao"]].assign(hour=p.target_hour_utc), wx, ORACLE_SRC)
    orc = orc.rename(columns={c: f"oracle_{c}" for c in ORACLE_SRC})
    p = p.merge(feats, on="row_id").merge(orc, on="row_id")
    cal_cols = ["hour_local", "dow_local", "month_local", "hour_utc"]
    tgt = ["sched_arr_count", "arr_count", "mean_arr_delay", "share_delayed_45", "cancellations", "diversions",
           "capacity_ratio"]
    order = (["icao", "target_hour_utc", "h", "feature_obs_time"] + fcols + cal_cols + tgt
             + [f"oracle_{c}" for c in ORACLE_SRC])
    return p[order].sort_values(["icao", "target_hour_utc", "h"]).reset_index(drop=True)


def build_flight(flights: pd.DataFrame, wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["meteostat"]
    fcols = feature_cols(m["rolling_windows_hours"])
    f = flights.copy()
    cal = _local_parts(f.sched_time_utc, f.origin_tz, "dep_")
    arr_cal = _local_parts(f.sched_arr_utc, f.dest_tz, "arr_")[["arr_hour_local"]]
    f = pd.concat([f, cal, arr_cal], axis=1)
    f["dep_hour_utc"] = f.sched_time_utc.dt.hour

    p = f.merge(pd.DataFrame({"h": m["horizons"]}), how="cross")
    p["row_id"] = np.arange(len(p))
    cutoff = p.sched_time_utc - pd.to_timedelta(p.h, unit="h")
    parts = [p]
    for side in ("origin", "dest"):
        keys = pd.DataFrame({"row_id": p.row_id, "icao": p[f"{side}_icao"], "cutoff": cutoff})
        a = asof_join(keys, wx, fcols, m["asof_tolerance_hours"])
        a = a.rename(columns={"obs_time": f"{side}_feature_obs_time", **{c: f"{side}_{c}" for c in fcols}})
        hour = (p.sched_time_utc if side == "origin" else p.sched_arr_utc).dt.floor("h")
        o = exact_join(pd.DataFrame({"row_id": p.row_id, "icao": p[f"{side}_icao"], "hour": hour}), wx, ORACLE_SRC)
        o = o.rename(columns={c: f"oracle_{side}_{c}" for c in ORACLE_SRC})
        parts += [a.set_index("row_id"), o.set_index("row_id")]
    p = p.set_index("row_id").join(parts[1:]).reset_index(drop=True)
    p["feature_obs_time"] = p[["origin_feature_obs_time", "dest_feature_obs_time"]].max(axis=1)

    side_cols = [f"{s}_{c}" for s in ("origin", "dest") for c in fcols]
    oracle = [f"oracle_{s}_{c}" for s in ("origin", "dest") for c in ORACLE_SRC]
    order = (["flight_id", "origin_icao", "dest_icao", "sched_time_utc", "sched_arr_utc", "h",
              "feature_obs_time", "origin_feature_obs_time", "dest_feature_obs_time",
              "carrier", "distance_mi", "crs_elapsed_min",
              "dep_hour_local", "dow_local", "month_local", "arr_hour_local", "dep_hour_utc"]
             + side_cols + ["delay_min", "label_45", "cancelled", "diverted"] + oracle)
    return p[order].sort_values(["sched_time_utc", "flight_id", "h"]).reset_index(drop=True)
