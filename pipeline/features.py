"""Leakage-safe features for forecasting airport-hour T at horizon h.

Timing convention (the core leakage guard):
  issue time      t0 = T - h            (forecast made at the start of hour T-h)
  usable inputs   anything time-stamped strictly before t0
                  -> state row k = T-h-1: weather observed at T-h-1 and flight
                     counts for [T-h-1, T-h), complete exactly at t0
  known schedule  scheduled arrivals/departures at T (published in advance)

So every `obs_*` column is the hourly state table shifted forward by h+1 hours.
`now_*` columns hold weather AT hour T and are used only by the oracle model
to show what same-hour weather (the leakage trap / a perfect forecast) is worth.
"""
import numpy as np
import pandas as pd

from .config import HORIZONS, KMH_PER_KT, Airport

# Meteostat condition codes
COCO_TS = {23, 24, 25, 26, 27}                  # lightning, hail, thunderstorm, storm
COCO_WINTER = {10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}
COCO_FOG = {5, 6}
COCO_HEAVY = {9, 11, 13, 16, 18, 20, 22, 26}


def wind_components(wspd_kt: pd.Series, wdir_deg: pd.Series, runway_heading: float):
    """Crosswind, headwind and tailwind (kt) relative to a runway heading.

    wdir is the direction the wind blows FROM. Calm/variable (NaN wdir) gives
    zero components when speed is ~0, NaN otherwise.
    """
    theta = np.deg2rad(wdir_deg - runway_heading)
    cross = wspd_kt * np.abs(np.sin(theta))
    head = wspd_kt * np.cos(theta)
    calm = wspd_kt.fillna(0) < 1
    cross, head = cross.mask(calm, 0.0), head.mask(calm, 0.0)
    return cross, head, (-head).clip(lower=0)


def weather_state(wx: pd.DataFrame, airport: Airport) -> pd.DataFrame:
    """Per-hour weather features from a Meteostat hourly frame (UTC index)."""
    wx = wx.copy()
    # Meteostat fills unknown gust/snow with 0 in some exports; a 0 kt gust is not an observation.
    wx["wpgt"] = wx["wpgt"].mask(wx["wpgt"] <= 0)
    s = pd.DataFrame(index=wx.index)
    s["temp"] = wx.temp
    s["dewpt_spread"] = wx.temp - wx.dwpt
    s["rhum"] = wx.rhum
    s["prcp"] = wx.prcp
    s["pres"] = wx.pres
    s["vsby"] = wx.vsby
    s["cldc"] = wx.cldc
    s["wspd_kt"] = wx.wspd / KMH_PER_KT
    s["gust_kt"] = wx.wpgt / KMH_PER_KT
    s["gust_factor"] = (s.gust_kt / s.wspd_kt.replace(0, np.nan))
    s["cross_kt"], s["head_kt"], s["tail_kt"] = wind_components(s.wspd_kt, wx.wdir, airport.runway_heading)
    s["wdir_sin"], s["wdir_cos"] = np.sin(np.deg2rad(wx.wdir)), np.cos(np.deg2rad(wx.wdir))
    coco = wx.coco
    s["ts"] = coco.isin(COCO_TS).astype(float).where(coco.notna())
    s["winter"] = coco.isin(COCO_WINTER).astype(float).where(coco.notna())
    s["fog"] = coco.isin(COCO_FOG).astype(float).where(coco.notna())
    s["heavy_precip"] = coco.isin(COCO_HEAVY).astype(float).where(coco.notna())
    s["coco"] = coco
    return s


def ops_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-hour operational state observable once the hour has closed."""
    p = panel
    o = pd.DataFrame(index=p.index)
    o["act_arr"] = p.act_arr
    o["sched_arr"] = p.sched_arr
    o["cap_ratio"] = p.act_arr / p.sched_arr.where(p.sched_arr > 0)
    o["sev_rate"] = p.n_sev / p.sched_arr.where(p.sched_arr > 0)
    o["arr_delay_mean"] = p.arr_delay_sum / p.n_arr_completed.where(p.n_arr_completed > 0)
    o["dep_delay_mean"] = p.dep_delay_sum / p.n_dep_completed.where(p.n_dep_completed > 0)
    # 3-hour pooled rates are far less noisy than single hours
    r = p[["act_arr", "sched_arr", "n_sev", "n_cancel_arr"]].rolling(3, min_periods=1).sum()
    o["cap_ratio_3h"] = r.act_arr / r.sched_arr.where(r.sched_arr > 0)
    o["sev_rate_3h"] = r.n_sev / r.sched_arr.where(r.sched_arr > 0)
    o["cancel_rate_3h"] = r.n_cancel_arr / r.sched_arr.where(r.sched_arr > 0)
    return o


def add_history(s: pd.DataFrame) -> pd.DataFrame:
    """Trailing windows and lags. Every window ends at the row's own hour, so
    these only ever look backwards from the state row."""
    s = s.copy()
    s["dp_3h"] = s.pres - s.pres.shift(3)
    s["dp_6h"] = s.pres - s.pres.shift(6)
    for w in (3, 6):
        s[f"prcp_{w}h"] = s.prcp.rolling(w, min_periods=1).sum()
        s[f"ts_{w}h"] = s.ts.rolling(w, min_periods=1).max()
        s[f"cross_max_{w}h"] = s.cross_kt.rolling(w, min_periods=1).max()
        s[f"gust_max_{w}h"] = s.gust_kt.rolling(w, min_periods=1).max()
    s["vsby_min_3h"] = s.vsby.rolling(3, min_periods=1).min()
    s["ts_24h"] = s.ts.rolling(24, min_periods=1).max()
    for lag in (1, 3, 6):
        for c in ("cross_kt", "tail_kt", "wspd_kt", "prcp"):
            s[f"{c}_lag{lag}"] = s[c].shift(lag)
    return s


def build_state(panel: pd.DataFrame, wx: pd.DataFrame, airport: Airport) -> pd.DataFrame:
    """Hourly state table on a gap-free UTC index: weather + ops + history."""
    idx = pd.date_range(panel.index.min(), panel.index.max(), freq="h", name="hour_utc")
    w = weather_state(wx.reindex(idx), airport)
    o = ops_state(panel.reindex(idx))
    return add_history(pd.concat([w, o], axis=1))


ORACLE_COLS = ["prcp", "wspd_kt", "gust_kt", "cross_kt", "tail_kt", "vsby", "ts",
               "winter", "fog", "heavy_precip", "dewpt_spread", "cldc"]


def build_dataset(panel: pd.DataFrame, wx: pd.DataFrame, airport: Airport,
                  horizons=HORIZONS) -> pd.DataFrame:
    """Long table: one row per (airport, target hour T, horizon h)."""
    state = build_state(panel, wx, airport)
    idx = state.index
    p = panel.reindex(idx)
    local = idx.tz_convert(airport.tz)

    target = pd.DataFrame(index=idx)
    target["airport"] = airport.icao
    target["local_time"] = local.tz_localize(None)
    # known in advance: schedule at T and its neighbours
    target["sched_arr_T"] = p.sched_arr
    target["sched_dep_T"] = p.sched_dep
    target["sched_arr_Tm1"] = p.sched_arr.shift(1)
    target["sched_arr_Tp1"] = p.sched_arr.shift(-1)
    target["hour_local"] = local.hour
    target["dow"] = local.dayofweek
    target["month"] = local.month
    target["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    target["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    target["runway_heading"] = airport.runway_heading
    # targets and evaluation-only columns
    target["y_act_arr"] = p.act_arr
    target["y_n_sev"] = p.n_sev
    target["y_cap_ratio"] = p.act_arr / p.sched_arr.where(p.sched_arr > 0)
    oracle = state[ORACLE_COLS].add_prefix("now_")

    frames = []
    for h in horizons:
        obs = state.shift(h + 1).add_prefix("obs_")
        f = pd.concat([target, obs, oracle], axis=1)
        f.insert(1, "h", h)
        f["issue_time_utc"] = idx - pd.Timedelta(hours=h)
        frames.append(f)
    ds = pd.concat(frames)
    ds = ds[ds.index.isin(panel.index) & ds.sched_arr_T.notna()]
    ds.index.name = "target_hour_utc"
    return ds.reset_index()
