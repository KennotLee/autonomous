"""Download public data: BTS on-time performance and Meteostat hourly weather.

Both are cached under data/raw, so re-running only fetches what's missing.

    python -m pipeline.download --airports KATL KDFW --start 2023-01 --end 2025-12
"""
import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests

from .config import RAW, get_airport

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")

# Canonical (BTS download-tool) names -> PREZIP names. Both spellings are accepted.
FLIGHT_COLS = {
    "FL_DATE": "FlightDate", "OP_UNIQUE_CARRIER": "Reporting_Airline",
    "ORIGIN": "Origin", "DEST": "Dest",
    "CRS_DEP_TIME": "CRSDepTime", "DEP_TIME": "DepTime", "DEP_DELAY": "DepDelay",
    "CRS_ARR_TIME": "CRSArrTime", "ARR_TIME": "ArrTime", "ARR_DELAY": "ArrDelay",
    "CANCELLED": "Cancelled", "CANCELLATION_CODE": "CancellationCode", "DIVERTED": "Diverted",
    "CRS_ELAPSED_TIME": "CRSElapsedTime",
    "WEATHER_DELAY": "WeatherDelay", "NAS_DELAY": "NASDelay",
}

WX_PARAMS = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres", "vsby", "cldc", "coco"]


def normalise_flight_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {v: k for k, v in FLIGHT_COLS.items() if v in df.columns}
    df = df.rename(columns=rename)
    return df[[c for c in FLIGHT_COLS if c in df.columns]]


def _bts_month(y: int, m: int, iatas: list[str], retries: int = 3) -> str:
    out = RAW / "bts" / f"{y}_{m:02d}.parquet"
    if out.exists():
        return f"{y}-{m:02d} cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    cached_zip = out.parent / f"bts_{y}_{m:02d}.zip"
    for attempt in range(retries):
        try:
            if cached_zip.exists():
                z = zipfile.ZipFile(cached_zip)
            else:
                r = requests.get(BTS_URL.format(y=y, m=m), timeout=900)
                if r.status_code == 404:
                    return f"{y}-{m:02d} not published"
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(z.open(name), usecols=lambda c: c in FLIGHT_COLS.values(),
                             low_memory=False)
            df = normalise_flight_columns(df)
            df = df[df.ORIGIN.isin(iatas) | df.DEST.isin(iatas)]
            df.to_parquet(out, index=False)
            return f"{y}-{m:02d} {len(df):,} rows"
        except Exception as e:  # network hiccups are common on transtats
            if attempt == retries - 1:
                return f"{y}-{m:02d} FAILED: {e}"
            time.sleep(5 * (attempt + 1))


def download_bts(iatas: list[str], start: str, end: str, workers: int = 4) -> None:
    """Fetch monthly BTS files and keep only flights touching `iatas`.

    The filter is applied at download time, so adding an airport later means
    deleting data/raw/bts and re-running.
    """
    months = pd.period_range(start, end, freq="M")
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda p: _bts_month(p.year, p.month, iatas), months):
            print("  BTS", msg, flush=True)


def download_weather(icao: str, start: str, end: str) -> pd.DataFrame:
    """Hourly Meteostat observations for one airport, indexed by UTC hour."""
    import meteostat as ms

    a = get_airport(icao)
    out = RAW / "weather" / f"{a.icao}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    station = a.wx_station
    if station is None:
        station = ms.stations.nearby(ms.Point(a.lat, a.lon), radius=50_000, limit=1).index[0]
    t0 = datetime.strptime(start, "%Y-%m")
    t1 = (pd.Period(end, "M").end_time).to_pydatetime()
    params = [ms.Parameter(p) for p in WX_PARAMS]
    chunks = []
    for y in range(t0.year, t1.year + 1):  # yearly chunks keep requests small
        s, e = max(t0, datetime(y, 1, 1)), min(t1, datetime(y, 12, 31, 23))
        df = ms.hourly(station, s, e, parameters=params).fetch()
        if df is not None and len(df):
            chunks.append(df)
    wx = pd.concat(chunks)
    wx.columns = [getattr(c, "value", c) for c in wx.columns]
    wx = wx.reindex(columns=WX_PARAMS).apply(pd.to_numeric, errors="coerce")
    wx.index = pd.DatetimeIndex(wx.index).tz_localize("UTC")
    wx.index.name = "hour_utc"
    wx["station"] = station
    out.parent.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(out)
    print(f"  weather {a.icao} station {station}: {len(wx):,} hours, "
          f"wdir missing {wx.wdir.isna().mean():.1%}")
    return wx


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=["KATL", "KDFW"])
    p.add_argument("--start", default="2023-01")
    p.add_argument("--end", default="2025-07")
    p.add_argument("--skip-bts", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    for icao in args.airports:
        download_weather(icao, args.start, args.end)
    if not args.skip_bts:
        download_bts([get_airport(c).iata for c in args.airports], args.start, args.end, args.workers)


if __name__ == "__main__":
    main()
