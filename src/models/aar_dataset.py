"""Build the P2 hourly panel: one row per (icao, target_hour_utc, h).

    python -m src.models.aar_dataset            # needs the weather CSV and the BTS download

Sources (see aar_config): challenge-5 hourly weather (ATL, DFW) and full BTS arrivals (aar_bts.py).

Leakage rule (data guide): weather is joined on airport + hour, then shifted by h, so every weather
input for target hour t comes from the observation hour t - h (feature_obs_time). Rolling windows
end at t - h. Recent-operations inputs count only flights that had LANDED before t - h.
`oracle_*` columns hold the weather of hour t itself, for evaluation and diagnostics only.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import aar_config as cfg
from .aar_bts import arrival_times, hourly_arrivals, load_flights
from .aar_data import load_airports

OPS_WINDOWS_H = (1, 3)  # realised arrivals in the last 1 and 3 completed hours before t - h


def load_weather(path: Path = cfg.WEATHER_CSV) -> pd.DataFrame:
    """Weather file -> icao, time_utc (tz-aware) + cfg.WEATHER_COLUMNS, one row per airport hour."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; put challenge5_hourly_weather_ATL_DFW.csv in {cfg.RAW_DIR}")
    wx = pd.read_csv(path)
    iata_to_icao = {a["iata"]: icao for icao, a in load_airports().items()}
    unknown = sorted(set(wx["airport"]) - set(iata_to_icao))
    if unknown:
        raise ValueError(f"weather airports not in config/airports.yaml: {unknown}")
    wx["icao"] = wx["airport"].map(iata_to_icao)
    wx["time_utc"] = pd.to_datetime(wx["time_utc"], utc=True)
    if wx.duplicated(["icao", "time_utc"]).any():
        raise ValueError("duplicate (airport, time_utc) rows in the weather file")
    return wx[["icao", "time_utc", *cfg.WEATHER_COLUMNS]].sort_values(["icao", "time_utc"])


def weather_features(wx: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (icao, observation hour T): base weather plus rolling windows (T - w, T]."""
    frames = []
    for icao, g in wx.groupby("icao"):
        g = g.set_index("time_utc").reindex(grid)  # missing hours stay NaN
        out = g[cfg.WEATHER_COLUMNS].copy()
        for w in cfg.ROLLING_WINDOWS_H:
            out[f"precip_sum_{w}h_mm"] = g["precip_mm"].rolling(w, min_periods=1).sum()
            out[f"wind_max_{w}h_kmh"] = g["wind_speed_kmh"].rolling(w, min_periods=1).max()
            out[f"crosswind_max_{w}h_kmh"] = g["crosswind_kmh"].rolling(w, min_periods=1).max()
        out["pressure_chg_3h_hpa"] = g["pressure_hpa"] - g["pressure_hpa"].shift(3)
        out.insert(0, "icao", icao)
        frames.append(out.rename_axis("obs_hour").reset_index())
    return pd.concat(frames, ignore_index=True)


def weather_availability_lag_hours(path: Path = cfg.AIRPORTS_YAML) -> int:
    """P1's config: a weather row stamped time_utc holds a METAR observed up to ~52 min later, so it
    counts as available at time_utc + lag (config/airports.yaml weather.availability_lag_hours)."""
    with open(path, encoding="utf-8") as f:
        return int(yaml.safe_load(f).get("weather", {}).get("availability_lag_hours", 1))


def build_panel(wx: pd.DataFrame, arrivals: pd.DataFrame, lag_hours: int | None = None) -> pd.DataFrame:
    """Join arrivals (target hour t) with weather available by t - h, recent operations before t - h,
    and oracle weather at t. arrivals: icao, target_hour_utc, arr_count, sched_arr_count, outcomes.

    feature_obs_time = t - h is the availability time; the weather row joined is the one stamped
    t - h - lag (weather_stamp_utc), whose observation is complete by t - h."""
    lag = weather_availability_lag_hours() if lag_hours is None else lag_hours
    grid = pd.DatetimeIndex(sorted(arrivals[cfg.TIME].unique()))
    feats = weather_features(wx, grid)
    base = set(cfg.WEATHER_COLUMNS)
    oracle = feats[["icao", "obs_hour", *cfg.WEATHER_COLUMNS]].rename(
        columns={"obs_hour": cfg.TIME, **{c: f"oracle_{c}" for c in base}})

    hourly = arrivals.set_index(["icao", cfg.TIME])
    rows = []
    for h in cfg.HORIZONS:
        p = arrivals.copy()
        p["h"] = h
        p[cfg.OBS_TIME] = p[cfg.TIME] - pd.Timedelta(hours=h)
        p["weather_stamp_utc"] = p[cfg.OBS_TIME] - pd.Timedelta(hours=lag)
        p = p.merge(feats, left_on=["icao", "weather_stamp_utc"], right_on=["icao", "obs_hour"], how="left").drop(columns="obs_hour")
        for w in OPS_WINDOWS_H:  # hours [t-h-w, t-h): all ended by the forecast time
            landed = sum(
                hourly[cfg.TARGET].reindex(pd.MultiIndex.from_arrays([p["icao"], p[cfg.OBS_TIME] - pd.Timedelta(hours=k)])).to_numpy()
                for k in range(1, w + 1)
            )
            sched = sum(
                hourly["sched_arr_count"].reindex(pd.MultiIndex.from_arrays([p["icao"], p[cfg.OBS_TIME] - pd.Timedelta(hours=k)])).to_numpy()
                for k in range(1, w + 1)
            )
            p[f"p2_ops_arr_last{w}h"] = landed
            p[f"p2_ops_sched_last{w}h"] = sched
        rows.append(p)
    panel = pd.concat(rows, ignore_index=True).merge(oracle, on=["icao", cfg.TIME], how="left")
    return panel.sort_values(["icao", cfg.TIME, "h"]).reset_index(drop=True)


def build(weather_csv: Path = cfg.WEATHER_CSV, out: Path = cfg.PANEL) -> pd.DataFrame:
    airports = load_airports()
    timezones = {icao: a["timezone"] for icao, a in airports.items()}
    iata_to_icao = {a["iata"]: icao for icao, a in airports.items()}
    wx = load_weather(weather_csv)

    flights = load_flights(cfg.BTS_START, cfg.BTS_END)
    times = arrival_times(flights, timezones, iata_to_icao)
    print(f"{len(flights):,} BTS flights to {sorted(iata_to_icao)}; wheels-on clock check vs BTS "
          f"WheelsOn: {times.attrs['wheels_on_match']:.4f}")

    # UTC hours with complete data: from the first full local day (2023-01-02, so red-eyes that
    # left before 2023 are not missing) to the last weather hour.
    start = max(pd.Timestamp("2023-01-02", tz=tz).tz_convert("UTC") for tz in timezones.values())
    end = wx["time_utc"].max()
    grid = pd.date_range(start.ceil("h"), end, freq="h", tz="UTC")
    arrivals = hourly_arrivals(times, grid)
    arrivals = arrivals[arrivals.icao.isin(wx.icao.unique())]
    panel = build_panel(wx, arrivals)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    print(f"panel {panel.shape} -> {out}")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weather", type=Path, default=cfg.WEATHER_CSV)
    ap.add_argument("--out", type=Path, default=cfg.PANEL)
    args = ap.parse_args()
    build(args.weather, args.out)


if __name__ == "__main__":
    main()
