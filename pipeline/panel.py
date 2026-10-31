"""Flight records -> hourly airport panel (one row per airport x UTC hour).

BTS times are local HHMM at the origin (departure) or destination (arrival).
Flights are bucketed by the airport's own clock, converted to UTC so that
weather (UTC) and flights line up and DST changes don't duplicate hours.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW, SEVERE_DELAY_MIN, Airport
from .download import FLIGHT_COLS, normalise_flight_columns


def load_flights(source: str | Path = "bts") -> pd.DataFrame:
    """'bts' = monthly parquet files from download.py; otherwise a CSV path
    (e.g. Flight_Data_With_Weather_Final.csv; its weather columns are ignored
    because they are joined at each flight's own hour, which leaks)."""
    if str(source) == "bts":
        files = sorted((RAW / "bts").glob("*.parquet"))
        if not files:
            raise FileNotFoundError("no BTS files; run python -m pipeline.download first")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        names = set(FLIGHT_COLS) | set(FLIGHT_COLS.values())
        df = pd.read_csv(source, usecols=lambda c: c in names, low_memory=False)
        df = normalise_flight_columns(df)
    df["FL_DATE"] = pd.to_datetime(df.FL_DATE, format="mixed")
    return df


def _hhmm_to_td(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return pd.to_timedelta((x // 100) * 60 + x % 100, unit="min")


def add_local_times(df: pd.DataFrame) -> pd.DataFrame:
    """Scheduled/actual departure and arrival as naive local timestamps.

    FL_DATE is the local departure date. The arrival date is whichever of
    day -1/0/+1 makes (arrival - departure) closest to the scheduled block
    time; the residual is the time-zone difference (< 7 h in the US), while a
    wrong day is off by >= 17 h, so the choice is unambiguous.
    """
    df = df.copy()
    df["sched_dep"] = df.FL_DATE + _hhmm_to_td(df.CRS_DEP_TIME)
    arr0 = df.FL_DATE + _hhmm_to_td(df.CRS_ARR_TIME)
    block = pd.to_timedelta(df.CRS_ELAPSED_TIME.fillna(120), unit="min")
    cands = [arr0 + pd.Timedelta(days=d) for d in (-1, 0, 1)]
    resid = np.column_stack([((c - df.sched_dep - block).abs()).to_numpy() for c in cands])
    pick = resid.argmin(axis=1)
    df["sched_arr"] = arr0 + pd.to_timedelta(pick - 1, unit="D")
    # Actual = scheduled + delay. Avoids ARR_TIME's 2400 / past-midnight wrap.
    df["act_dep"] = df.sched_dep + pd.to_timedelta(df.DEP_DELAY, unit="min")
    df["act_arr"] = df.sched_arr + pd.to_timedelta(df.ARR_DELAY, unit="min")
    return df


def _to_utc_hour(local: pd.Series, tz: str) -> pd.Series:
    utc = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return utc.dt.floor("h")


def build_panel(flights: pd.DataFrame, airport: Airport) -> pd.DataFrame:
    """Hourly counts and rates for one airport, on a gap-free UTC hour index.

    Columns
      sched_arr / sched_dep   flights scheduled to arrive / depart that hour (incl. cancelled)
      act_arr / act_dep       flights that actually arrived / departed that hour
      n_sev                   scheduled arrivals that hour that were >45 min late,
                              cancelled or diverted (all three are "severe" for the user)
      n_cancel_arr, arr_delay_sum, n_arr_completed, dep_delay_sum, n_dep_completed
    act_arr is the AAR proxy: true FAA AAR (ASPM) isn't public. It equals
    min(demand, capacity), so it only measures capacity when demand is high.
    """
    if "sched_arr" not in flights:
        flights = add_local_times(flights)
    tz, code = airport.tz, airport.iata
    arr = flights[flights.DEST == code]
    dep = flights[flights.ORIGIN == code]

    cancelled = arr.CANCELLED.fillna(0).astype(bool)
    diverted = arr.DIVERTED.fillna(0).astype(bool)
    completed = ~cancelled & ~diverted & arr.ARR_DELAY.notna()
    arr = arr.assign(
        h_sched=_to_utc_hour(arr.sched_arr, tz),
        h_act=_to_utc_hour(arr.act_arr, tz),
        sev=((arr.ARR_DELAY > SEVERE_DELAY_MIN) | cancelled | diverted).astype(int),
        cxl=cancelled.astype(int),
        delay=arr.ARR_DELAY.where(completed),
        done=completed.astype(int),
    )
    dep_done = ~dep.CANCELLED.fillna(0).astype(bool) & dep.DEP_DELAY.notna()
    dep = dep.assign(
        h_sched=_to_utc_hour(dep.sched_dep, tz),
        h_act=_to_utc_hour(dep.act_dep, tz),
        delay=dep.DEP_DELAY.where(dep_done),
        done=dep_done.astype(int),
    )

    by_sched = arr.groupby("h_sched").agg(
        sched_arr=("sev", "size"), n_sev=("sev", "sum"), n_cancel_arr=("cxl", "sum"),
        arr_delay_sum=("delay", "sum"), n_arr_completed=("done", "sum"))
    act_arr = arr[completed].groupby("h_act").size().rename("act_arr")
    dep_sched = dep.groupby("h_sched").agg(
        sched_dep=("done", "size"), dep_delay_sum=("delay", "sum"), n_dep_completed=("done", "sum"))
    act_dep = dep[dep_done].groupby("h_act").size().rename("act_dep")

    # Gap-free hourly index over the months that have data. Hours with no
    # flights are genuine zeros; months with no file are excluded entirely.
    months = flights.FL_DATE.dt.to_period("M").unique()
    lo = pd.Timestamp(min(months).start_time).tz_localize(tz).tz_convert("UTC")
    hi = pd.Timestamp(max(months).end_time).tz_localize(tz).tz_convert("UTC").floor("h")
    idx = pd.date_range(lo, hi, freq="h", name="hour_utc")
    panel = pd.concat([by_sched, act_arr, dep_sched, act_dep], axis=1).reindex(idx).fillna(0)
    panel = panel.astype({c: int for c in panel.columns if not c.endswith("_sum")})

    local = idx.tz_convert(tz)
    covered = pd.PeriodIndex(local.tz_localize(None), freq="M").isin(months)
    panel = panel[covered]
    panel.insert(0, "airport", airport.icao)
    panel["local_time"] = local[covered].tz_localize(None)
    return panel
