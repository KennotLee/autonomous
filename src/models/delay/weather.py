"""Hourly Meteostat weather for any airport, cached in data/raw/weather/{ICAO}.parquet.

    python -m src.models.delay.weather --start 2023-01 --end 2025-07   # all airports in BTS files
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import airports

ROOT = Path(__file__).resolve().parents[3]
WX_DIR = ROOT / "data" / "raw" / "weather"
PARAMS = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres", "vsby", "cldc", "coco"]
# Stations whose Meteostat metadata sits far from the airport; pinned by hand.
KNOWN_STATIONS = {"KATL": "72219", "KDFW": "72259"}


def _station(a: airports.Airport) -> str:
    import meteostat as ms
    if a.icao in KNOWN_STATIONS:
        return KNOWN_STATIONS[a.icao]
    near = ms.stations.nearby(ms.Point(a.lat, a.lon), radius=40_000, limit=1)
    if near.empty:
        raise LookupError(f"no Meteostat station within 40 km of {a.icao}")
    return near.index[0]


def fetch(a: airports.Airport, start: str, end: str) -> pd.DataFrame:
    """Requires meteostat>=2 (`ms.hourly`); output is indexed by UTC hour."""
    import meteostat as ms
    out = WX_DIR / f"{a.icao}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    station = _station(a)
    t0 = datetime.strptime(start, "%Y-%m")
    t1 = pd.Period(end, "M").end_time.to_pydatetime()
    chunks = []
    for y in range(t0.year, t1.year + 1):
        s, e = max(t0, datetime(y, 1, 1)), min(t1, datetime(y, 12, 31, 23))
        df = ms.hourly(station, s, e, parameters=[ms.Parameter(p) for p in PARAMS]).fetch()
        if df is not None and len(df):
            chunks.append(df)
    if not chunks:
        raise LookupError(f"no hourly data for {a.icao} station {station}")
    wx = pd.concat(chunks)
    wx.columns = [getattr(c, "value", c) for c in wx.columns]
    wx = wx.reindex(columns=PARAMS).apply(pd.to_numeric, errors="coerce")
    wx.index = pd.DatetimeIndex(wx.index).tz_localize("UTC")
    wx.index.name = "hour_utc"
    wx["station"] = station
    WX_DIR.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(out)
    return wx


def fetch_all(iatas, start, end, workers=6):
    def one(code):
        try:
            a = airports.get(code)
            wx = fetch(a, start, end)
            return f"{code}: {len(wx):,} h, wdir missing {wx.wdir.isna().mean():.0%}"
        except Exception as e:
            return f"{code}: FAILED {e}"
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(one, sorted(iatas)):
            print(" ", msg, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01")
    p.add_argument("--end", default="2025-07")
    args = p.parse_args(argv)
    files = sorted((ROOT / "data" / "raw" / "bts").glob("*.parquet"))
    codes = set()
    for f in files:
        df = pd.read_parquet(f, columns=["ORIGIN", "DEST"])
        codes |= set(df.ORIGIN) | set(df.DEST)
    fetch_all(codes, args.start, args.end)


if __name__ == "__main__":
    main()
