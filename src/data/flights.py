"""Load BTS flights touching the configured hubs, convert times to UTC, build flight-level targets.

BTS hhmm times are local: departure in origin tz, arrival in destination tz.
sched_dep_utc  = FL_DATE + CRS_DEP_TIME localised in origin tz
sched_arr_utc  = sched_dep_utc + CRS_ELAPSED_TIME     (avoids midnight/date ambiguity)
gate_arr_utc   = sched_arr_utc + ARR_DELAY            (BTS: ARR_DELAY = ARR_TIME − CRS_ARR_TIME)
wheels_on_utc  = gate_arr_utc − TAXI_IN
"""
import numpy as np
import pandas as pd

USECOLS = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM", "ORIGIN", "DEST",
           "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_TIME", "WHEELS_ON", "TAXI_IN", "ARR_DELAY",
           "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME", "DISTANCE"]


def _hhmm_to_td(x: pd.Series) -> pd.Series:
    x = x.astype("float64")
    return pd.to_timedelta(x // 100, unit="h") + pd.to_timedelta(x % 100, unit="m")


def _localize(naive: pd.Series, tz: pd.Series) -> pd.Series:
    """Localise naive local times with a per-row tz; DST-ambiguous/nonexistent -> NaT. Returns UTC."""
    out = pd.Series(pd.NaT, index=naive.index, dtype="datetime64[ns, UTC]")
    for z, idx in naive.groupby(tz).groups.items():
        out.loc[idx] = (naive.loc[idx].dt.tz_localize(z, ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert("UTC"))
    return out


def _local_hhmm(ts_utc: pd.Series, tz: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=ts_utc.index)
    for z, idx in ts_utc.groupby(tz).groups.items():
        loc = ts_utc.loc[idx].dt.tz_convert(z)
        out.loc[idx] = loc.dt.hour * 100 + loc.dt.minute
    return out


def load_hub_flights(raw_csv, hub_iatas: list[str], airports: pd.DataFrame):
    """Return (flights DataFrame, validation dict). `airports` = resolve_airports() output."""
    df = pd.read_csv(raw_csv, usecols=USECOLS, low_memory=False)
    df = df[df.ORIGIN.isin(hub_iatas) | df.DEST.isin(hub_iatas)].reset_index(drop=True)
    a = airports.set_index("iata")
    f = pd.DataFrame({
        "fl_date": df.FL_DATE,
        "carrier": df.OP_UNIQUE_CARRIER,
        "fl_num": df.OP_CARRIER_FL_NUM.astype("Int64").astype(str),
        "tail_num": df.TAIL_NUM,
        "origin_icao": df.ORIGIN.map(a.icao), "dest_icao": df.DEST.map(a.icao),
        "origin_tz": df.ORIGIN.map(a.timezone), "dest_tz": df.DEST.map(a.timezone),
    })
    f["flight_id"] = (f.fl_date + "_" + f.carrier + f.fl_num + "_" + df.ORIGIN + "_" + df.DEST)
    dup = f.flight_id.duplicated(keep=False)
    if dup.any():  # same flight number/route twice a day: disambiguate by scheduled dep time
        f.loc[dup, "flight_id"] += "_" + df.loc[dup, "CRS_DEP_TIME"].astype(str).str.zfill(4)
    assert f.flight_id.is_unique, "flight_id not unique"

    naive_dep = pd.to_datetime(df.FL_DATE) + _hhmm_to_td(df.CRS_DEP_TIME)
    f["sched_time_utc"] = _localize(naive_dep, f.origin_tz)          # = scheduled departure UTC
    f["sched_arr_utc"] = f.sched_time_utc + pd.to_timedelta(df.CRS_ELAPSED_TIME, unit="m")
    f["gate_arr_utc"] = f.sched_arr_utc + pd.to_timedelta(df.ARR_DELAY, unit="m")
    f["wheels_on_utc"] = f.gate_arr_utc - pd.to_timedelta(df.TAXI_IN, unit="m")
    f["crs_elapsed_min"] = df.CRS_ELAPSED_TIME
    f["distance_mi"] = df.DISTANCE
    f["cancelled"] = df.CANCELLED.astype(int)
    f["diverted"] = df.DIVERTED.astype(int)
    f["delay_min"] = df.ARR_DELAY  # NaN for cancelled/diverted
    f["label_45"] = np.where(f.delay_min.isna(), np.nan, (f.delay_min > 45).astype(float))

    # Validation against the local hhmm columns (2400 == 0000).
    def share(pred, col):
        ok = pred.notna() & col.notna()
        return round(float((pred[ok] == (col[ok] % 2400)).mean()), 4), int(ok.sum())
    v = {
        "rows_hub": len(f),
        "dst_unresolved_sched_dep": int(f.sched_time_utc.isna().sum()),
        "unmapped_airports": int(f.origin_icao.isna().sum() + f.dest_icao.isna().sum()),
        "sched_arr_vs_CRS_ARR_TIME": share(_local_hhmm(f.sched_arr_utc, f.dest_tz), df.CRS_ARR_TIME),
        "gate_arr_vs_ARR_TIME": share(_local_hhmm(f.gate_arr_utc, f.dest_tz), df.ARR_TIME),
        "wheels_on_vs_WHEELS_ON": share(_local_hhmm(f.wheels_on_utc, f.dest_tz), df.WHEELS_ON),
    }
    return f, v
