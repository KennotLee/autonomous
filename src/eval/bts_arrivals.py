"""P4: independent hourly-arrivals target from full BTS on-time data, to check P2's AAR model.

    python src/eval/bts_arrivals.py download          # monthly BTS zips -> data/interim/bts_p4/*.parquet (hub arrivals)
    python src/eval/bts_arrivals.py hourly            # -> data/interim/bts_p4_hourly.parquet

Independent of P2's aar_bts.py on purpose: landing hour comes from BTS's own WheelsOn clock time (local hhmm at
the destination), with the date chosen closest to scheduled arrival + ArrDelay. P2 derives it from scheduled
arrival + ArrDelay - TaxiIn. arr_count = flights landed (not cancelled/diverted) by floor(wheels_on_utc, hour);
sched_arr_count = flights scheduled to arrive in the hour (incl. later cancelled), by floor(sched_arr_utc).
"""
import io
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from baselines import ROOT, load_airports

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
COLS = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime",
        "CRSArrTime", "CRSElapsedTime", "ArrDelay", "WheelsOn", "Cancelled", "Diverted"]
CACHE = ROOT / "data/interim/bts_p4"
HOURLY = ROOT / "data/interim/bts_p4_hourly.parquet"
FLIGHTS = ROOT / "data/interim/bts_p4_flights.parquet"  # per flight: icao, sched_hour_utc, wheels_hour_utc (NaT = not landed)
START, END = "2023-01", "2025-07"


def month(ym, iatas):
    out = CACHE / f"{ym.year}_{ym.month:02d}.parquet"
    if out.exists():
        return f"{ym} cached"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL.format(y=ym.year, m=ym.month), timeout=900) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(z.open(name), usecols=COLS, low_memory=False)
            df[df["Dest"].isin(iatas)].to_parquet(out, index=False)
            return f"{ym} {int(df['Dest'].isin(iatas).sum())} hub arrivals of {len(df)}"
        except Exception as e:  # noqa: BLE001  network: retry then report
            if attempt == 2:
                return f"{ym} FAILED {e}"
            time.sleep(10 * (attempt + 1))


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    iatas = [a["iata"] for a in load_airports().values()]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(lambda ym: month(ym, iatas), pd.period_range(START, END, freq="M")):
            print(msg, flush=True)


def hhmm(x):
    x = pd.to_numeric(x, errors="coerce")
    return pd.to_timedelta((x // 100) * 60 + x % 100, unit="min")


def closest_day(base_date, clock, target):
    """date + clock on day -1/0/+1 closest to target (all naive local)."""
    c = np.column_stack([((base_date + pd.Timedelta(days=d) + clock) - target).abs().to_numpy() for d in (-1, 0, 1)])
    return base_date + clock + pd.to_timedelta(np.nanargmin(np.where(np.isnan(c.astype(float)), np.inf, c.astype(float)),
                                                            axis=1) - 1, unit="D")


def hourly():
    aps = load_airports()
    f = pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))], ignore_index=True)
    date = pd.to_datetime(f["FlightDate"])
    dep = date + hhmm(f["CRSDepTime"])
    sched = closest_day(date, hhmm(f["CRSArrTime"]), dep + pd.to_timedelta(f["CRSElapsedTime"].fillna(120), unit="min"))
    landed = (f["Cancelled"] == 0) & (f["Diverted"] == 0) & f["WheelsOn"].notna()
    wheels = closest_day(date, hhmm(f["WheelsOn"]), sched + pd.to_timedelta(f["ArrDelay"].fillna(0), unit="min"))
    frames, flights = [], []
    for icao, a in aps.items():
        m = (f["Dest"] == a["iata"]).to_numpy()
        kw = dict(ambiguous="NaT", nonexistent="NaT")
        s_utc = sched[m].dt.tz_localize(a["timezone"], **kw).dt.tz_convert("UTC").dt.floor("h")
        w_utc = wheels[m & landed.to_numpy()].dt.tz_localize(a["timezone"], **kw).dt.tz_convert("UTC").dt.floor("h")
        lo = pd.Timestamp(START, tz=a["timezone"]).tz_convert("UTC")
        hi = (pd.Timestamp(END, tz=a["timezone"]) + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)).tz_convert("UTC")
        grid = pd.date_range(lo.floor("h"), hi.floor("h"), freq="h")
        flights.append(pd.DataFrame({"icao": icao, "sched_hour_utc": s_utc,
                                     "wheels_hour_utc": w_utc.reindex(s_utc.index)}))
        frames.append(pd.DataFrame({"icao": icao, "target_hour_utc": grid,
                                    "arr_count": w_utc.value_counts().reindex(grid, fill_value=0).to_numpy(),
                                    "sched_arr_count": s_utc.value_counts().reindex(grid, fill_value=0).to_numpy()}))
        print(f"{icao}: {int(m.sum())} scheduled, {int((m & landed).sum())} landed; DST-dropped "
              f"{int(s_utc.isna().sum())} sched / {int(w_utc.isna().sum())} landed")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(HOURLY, index=False)
    pd.concat(flights, ignore_index=True).to_parquet(FLIGHTS, index=False)
    print(f"wrote {HOURLY} ({len(out)} rows) and {FLIGHTS}")


if __name__ == "__main__":
    {"download": download, "hourly": hourly}[sys.argv[1]]()
