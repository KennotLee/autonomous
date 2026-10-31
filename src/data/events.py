"""Flights CSV -> event table: one departure event per flight leaving a configured hub, one arrival event
per flight landing at one (ATL<->DFW flights give two events).

Organiser target (data guide, rules): severe delay = DEP_DELAY > 45 for departures from the hub,
ARR_DELAY > 45 for arrivals at the hub. Cancelled events get label_45 = NaN (diverted arrivals have no
ARR_DELAY, so their label is NaN too).

Times: BTS hhmm are local. Departure event: FL_DATE + CRS_DEP_TIME in the hub's tz. Arrival event:
FL_DATE + CRS_ARR_TIME in the hub's tz, with the day (-1/0/+1) chosen so that arrival - departure is
closest to CRS_ELAPSED_TIME (residual = time-zone difference, < 7 h in the US; a wrong day is >= 17 h off).
Local -> UTC with the hub's IANA tz; DST-ambiguous/nonexistent local times -> NaT and the event is dropped.
"""
import numpy as np
import pandas as pd

from .config import other_icao

ORACLE_VARS = {"temp": "temp", "prcp": "prcp", "wspd": "wspd", "pres": "pres", "coco": "coco"}  # CSV prefix -> out
USECOLS = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "ORIGIN_STATE_ABR",
           "DEST_STATE_ABR", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DEP_DELAY", "ARR_DELAY",
           "CANCELLED", "DIVERTED"] + [f"{v}_{s}" for v in ORACLE_VARS for s in ("ORIGIN", "DEST")]


def _hhmm(x: pd.Series) -> pd.Series:
    x = x.astype("float64")
    return pd.to_timedelta(x // 100, unit="h") + pd.to_timedelta(x % 100, unit="m")


def build_events(raw_csv, hubs: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    hub_iata = {v["iata"]: k for k, v in hubs.items()}
    df = pd.read_csv(raw_csv, usecols=USECOLS, low_memory=False)
    df = df[df.ORIGIN.isin(hub_iata) | df.DEST.isin(hub_iata)].reset_index(drop=True)

    key = (df.FL_DATE + "_" + df.OP_UNIQUE_CARRIER + df.OP_CARRIER_FL_NUM.astype("Int64").astype(str)
           + "_" + df.ORIGIN + "_" + df.DEST)
    dup = key.duplicated(keep=False)
    key[dup] = key[dup] + "_" + df.loc[dup, "CRS_DEP_TIME"].astype(str).str.zfill(4)
    assert key.is_unique, "flight_key not unique"

    date = pd.to_datetime(df.FL_DATE)
    dep_local = date + _hhmm(df.CRS_DEP_TIME)
    arr0 = date + _hhmm(df.CRS_ARR_TIME)
    block = pd.to_timedelta(df.CRS_ELAPSED_TIME, unit="m")
    cands = np.column_stack([((arr0 + pd.Timedelta(days=d)) - dep_local - block).abs().to_numpy()
                             for d in (-1, 0, 1)])
    arr_local = arr0 + pd.to_timedelta(cands.argmin(axis=1) - 1, unit="D")

    events = []
    for etype, side, other, local, delay in (("dep", "ORIGIN", "DEST", dep_local, df.DEP_DELAY),
                                             ("arr", "DEST", "ORIGIN", arr_local, df.ARR_DELAY)):
        m = df[side].isin(hub_iata)
        e = pd.DataFrame({
            "flight_key": key[m],
            "event_type": etype,
            "airport": df.loc[m, side].map(hub_iata),
            "other_airport": [other_icao(i, s, cfg) for i, s in zip(df.loc[m, other], df.loc[m, f"{other}_STATE_ABR"])],
            "sched_time_local": local[m],
            "carrier": df.loc[m, "OP_UNIQUE_CARRIER"],
            "cancelled": df.loc[m, "CANCELLED"].astype(int),
            "diverted": df.loc[m, "DIVERTED"].astype(int),
            "delay_min": delay[m],
        })
        for v, out in ORACLE_VARS.items():  # flights-CSV weather at event time: oracle only
            e[f"oracle_{out}"] = df.loc[m, f"{v}_{side}"]
        events.append(e)
    ev = pd.concat(events, ignore_index=True)
    ev["event_id"] = ev.flight_key + "_" + ev.event_type

    tz = ev.airport.map({k: v["timezone"] for k, v in hubs.items()})
    ev["sched_time_utc"] = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns, UTC]")
    for z, idx in ev.groupby(tz).groups.items():
        ev.loc[idx, "sched_time_utc"] = (ev.loc[idx, "sched_time_local"]
                                         .dt.tz_localize(z, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    n_dst = int(ev.sched_time_utc.isna().sum())
    ev = ev[ev.sched_time_utc.notna()].reset_index(drop=True)

    thr = cfg["delay_threshold_min"]
    ev.loc[ev.cancelled == 1, "delay_min"] = np.nan
    ev["label_45"] = np.where(ev.delay_min.isna(), np.nan, (ev.delay_min > thr).astype(float))
    ev["hour"] = ev.sched_time_local.dt.hour
    ev["dow"] = ev.sched_time_local.dt.dayofweek  # 0 = Monday
    ev["month"] = ev.sched_time_local.dt.month
    stats = {"flights_touching_hubs": int(len(df)), "events": int(len(ev)),
             "events_dropped_dst": n_dst, "dep_events": int((ev.event_type == "dep").sum()),
             "arr_events": int((ev.event_type == "arr").sum()),
             "label_45_rate": round(float(ev.label_45.mean()), 4),
             "label_nan": int(ev.label_45.isna().sum())}
    return ev, stats
