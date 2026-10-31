"""Per-flight, leakage-safe features for P(arrival delay > 45 min).

Reference time: the flight's scheduled ARRIVAL (UTC); the label is ARR_DELAY > 45.
For horizon h the forecast is issued at t0 = sched_arr_utc - h, and every
weather/ops input comes from state row k = floor(t0) - 1 h: weather observed
at k and hub flight counts for [k, k+1), all complete by t0.
Known-in-advance inputs: the schedule (hub congestion at the flight's hub hour),
carrier, distance, block time, calendar.
`now_*` columns are weather at the flight's own scheduled hour at each end:
oracle only, never used by the production model.
"""
import numpy as np
import pandas as pd

from pipeline.features import add_history, ops_state, weather_state
from pipeline.panel import add_local_times, build_panel, load_flights

from . import airports
from .weather import fetch

HORIZONS = [3, 6, 9, 12]           # team contract (README)
SEVERE_MIN = 45
WX_FEATS = ["temp", "dewpt_spread", "prcp", "pres", "vsby", "cldc", "wspd_kt", "gust_kt",
            "cross_kt", "tail_kt", "head_kt", "ts", "winter", "fog", "heavy_precip",
            "dp_3h", "dp_6h", "prcp_3h", "prcp_6h", "ts_3h", "ts_6h", "ts_24h",
            "gust_max_3h", "cross_max_3h", "vsby_min_3h"]
ORACLE_FEATS = ["prcp", "wspd_kt", "gust_kt", "cross_kt", "vsby", "ts", "heavy_precip", "winter"]
OPS_FEATS = ["cap_ratio_3h", "sev_rate_3h", "cancel_rate_3h", "arr_delay_mean", "dep_delay_mean"]


def flight_id(df: pd.DataFrame) -> pd.Series:
    """date_carrier_flightnumber_origin_dest; identical for BTS and the team CSV."""
    return (df.FL_DATE.dt.strftime("%Y-%m-%d") + "_" + df.OP_UNIQUE_CARRIER.astype(str) + "_"
            + df.OP_CARRIER_FL_NUM.astype("Int64").astype(str) + "_" + df.ORIGIN + "_" + df.DEST)


def _localize(local: pd.Series, codes: pd.Series) -> pd.Series:
    """Naive local timestamps -> UTC, using each row's airport timezone."""
    tz = codes.map(lambda c: airports.get(c).tz)
    out = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns, UTC]")
    for z, idx in tz.groupby(tz).groups.items():
        out.loc[idx] = local.loc[idx].dt.tz_localize(z, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return out


def load_hub_flights() -> pd.DataFrame:
    hub_codes = list(airports.hubs())
    f = add_local_times(load_flights("bts"))
    f = f[f.ORIGIN.isin(hub_codes) | f.DEST.isin(hub_codes)].copy()
    f["flight_id"] = flight_id(f)
    f = f.drop_duplicates("flight_id")
    f["sched_dep_utc"] = _localize(f.sched_dep, f.ORIGIN)
    f["sched_arr_utc"] = _localize(f.sched_arr, f.DEST)
    f["hub"] = np.where(f.DEST.isin(hub_codes), f.DEST, f.ORIGIN)
    f["arr_to_hub"] = f.DEST.isin(hub_codes).astype(int)
    f["label"] = (f.ARR_DELAY > SEVERE_MIN).astype(float).where(f.ARR_DELAY.notna())
    return f


def _wx_table(codes) -> pd.DataFrame:
    """Stacked hourly weather state for many airports, keyed (iata, hour)."""
    frames = []
    for c in sorted(codes):
        try:
            wx = fetch(airports.get(c), "2023-01", "2025-07")
        except Exception:
            continue  # airport without weather: its features stay NaN
        a = airports.get(c)
        heading = a.runway_heading if a.runway_heading is not None else np.nan
        idx = pd.date_range(wx.index.min(), wx.index.max(), freq="h", name="hour_utc")
        s = add_history(weather_state(wx.reindex(idx), type("A", (), {"runway_heading": heading})))
        s = s[WX_FEATS].astype("float32")
        s["iata"] = c
        frames.append(s.reset_index())
    return pd.concat(frames, ignore_index=True)


def _hub_tables(flights_all: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for code, a in airports.hubs().items():
        panel = build_panel(flights_all, a)
        idx = pd.date_range(panel.index.min(), panel.index.max(), freq="h", name="hour_utc")
        o = ops_state(panel.reindex(idx))[OPS_FEATS].astype("float32")
        o["sched_arr_hr"] = panel.sched_arr.reindex(idx)
        o["sched_dep_hr"] = panel.sched_dep.reindex(idx)
        o["hub"] = code
        frames.append(o.reset_index())
    return pd.concat(frames, ignore_index=True)


def build(flights: pd.DataFrame, flights_all: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """One row per (flight, h). `flights_all` (every hub flight) feeds the hub
    congestion/ops panel; `flights` is the subset to featurise."""
    wx = _wx_table(set(flights.ORIGIN) | set(flights.DEST))
    hub = _hub_tables(flights_all)

    base = flights[["flight_id", "ORIGIN", "DEST", "hub", "arr_to_hub", "OP_UNIQUE_CARRIER",
                    "DISTANCE", "CRS_ELAPSED_TIME", "sched_dep_utc", "sched_arr_utc",
                    "sched_arr", "ARR_DELAY", "label"]].copy()
    base["hour_local"] = base.sched_arr.dt.hour
    base["dow"] = base.sched_arr.dt.dayofweek
    base["month"] = base.sched_arr.dt.month
    # schedule congestion at the hub in the hour the flight uses it (known in advance)
    base["hub_hour_utc"] = np.where(base.arr_to_hub == 1, base.sched_arr_utc.dt.floor("h"),
                                    base.sched_dep_utc.dt.floor("h"))
    base["hub_hour_utc"] = pd.to_datetime(base.hub_hour_utc, utc=True)
    sched = hub[["hub", "hour_utc", "sched_arr_hr", "sched_dep_hr"]].rename(
        columns={"hour_utc": "hub_hour_utc", "sched_arr_hr": "hub_sched_arr", "sched_dep_hr": "hub_sched_dep"})
    base = base.merge(sched, on=["hub", "hub_hour_utc"], how="left")

    # oracle: weather at each end in the flight's own scheduled hour
    for end, code, col in (("dest", "DEST", "sched_arr_utc"), ("origin", "ORIGIN", "sched_dep_utc")):
        w = wx[["iata", "hour_utc"] + ORACLE_FEATS].rename(
            columns={c: f"now_{end}_{c}" for c in ORACLE_FEATS})
        base["_k"] = base[col].dt.floor("h")
        base = base.merge(w, left_on=[code, "_k"], right_on=["iata", "hour_utc"],
                          how="left").drop(columns=["iata", "hour_utc", "_k"])

    out = []
    for h in horizons:
        d = base.copy()
        d["h"] = h
        d["issue_time_utc"] = d.sched_arr_utc - pd.Timedelta(hours=h)
        d["feature_obs_time"] = d.issue_time_utc.dt.floor("h") - pd.Timedelta(hours=1)
        for end, code in (("origin", "ORIGIN"), ("dest", "DEST")):
            w = wx.rename(columns={c: f"{end}_{c}" for c in WX_FEATS})
            d = d.merge(w, left_on=[code, "feature_obs_time"], right_on=["iata", "hour_utc"],
                        how="left").drop(columns=["iata", "hour_utc"])
        o = hub[["hub", "hour_utc"] + OPS_FEATS].rename(columns={c: f"hub_{c}" for c in OPS_FEATS})
        d = d.merge(o, left_on=["hub", "feature_obs_time"], right_on=["hub", "hour_utc"],
                    how="left").drop(columns=["hour_utc"])
        out.append(d)
    return pd.concat(out, ignore_index=True)
