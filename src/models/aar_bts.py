"""Full BTS on-time data -> hourly arrivals per airport (the AAR-proxy target).

    python -m src.models.aar_bts --start 2023-01 --end 2025-07     # download + filter, resumable

Downloads the monthly BTS "Reporting Carrier On-Time Performance" files, keeps only flights whose
destination is an airport in config/airports.yaml, and caches one small parquet per month in
data/interim/bts/. The monthly zips (~32 MB each) are not kept.

Times: BTS hhmm times are local (departure = origin clock, arrival = destination clock) and
FlightDate is the local departure date. The scheduled arrival date is whichever of day -1/0/+1
makes (arrival - departure) closest to the scheduled block time; the residual is the time-zone
difference (< 7 h in the US), while a wrong day is off by >= 17 h, so the choice is unambiguous.
Landing (wheels-on) = scheduled arrival + ArrDelay - TaxiIn (ArrDelay is gate-to-gate), converted
from the destination's time zone to UTC.
"""
import argparse
import io
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import aar_config as cfg

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
COLUMNS = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
           "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "ArrDelay", "TaxiIn", "WheelsOn",
           "Cancelled", "Diverted"]


def month_path(y: int, m: int, out_dir: Path = cfg.BTS_DIR) -> Path:
    return Path(out_dir) / f"bts_arrivals_{y}_{m:02d}.parquet"


def download_month(y: int, m: int, iatas: list[str], out_dir: Path = cfg.BTS_DIR, retries: int = 3) -> str:
    out = month_path(y, m, out_dir)
    if out.exists():
        return f"{y}-{m:02d} cached"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(BTS_URL.format(y=y, m=m), timeout=900) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(z.open(name), usecols=lambda c: c in COLUMNS, low_memory=False)
            df = df[df["Dest"].isin(iatas)]
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            return f"{y}-{m:02d} {len(df):,} arrivals"
        except Exception as e:  # network hiccups: retry, then report
            if attempt == retries:
                return f"{y}-{m:02d} FAILED: {e}"
            time.sleep(10 * attempt)


def months(start: str, end: str) -> list[tuple[int, int]]:
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def download(start: str, end: str, iatas: list[str], out_dir: Path = cfg.BTS_DIR, workers: int = 4) -> None:
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda ym: download_month(*ym, iatas, out_dir), months(start, end)):
            print(msg, flush=True)


def load_flights(start: str, end: str, out_dir: Path = cfg.BTS_DIR) -> pd.DataFrame:
    paths = [month_path(y, m, out_dir) for y, m in months(start, end)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"BTS months not downloaded: {missing}; run python -m src.models.aar_bts")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _hhmm(x: pd.Series) -> pd.Series:
    """hhmm (2400 allowed) -> timedelta since midnight."""
    x = pd.to_numeric(x, errors="coerce")
    return pd.to_timedelta((x // 100) * 60 + x % 100, unit="min")


def arrival_times(f: pd.DataFrame, timezones: dict[str, str], iata_to_icao: dict[str, str]) -> pd.DataFrame:
    """Per flight: icao, sched_arr_utc, wheels_on_utc (NaT if cancelled/diverted), cancelled, diverted."""
    date = pd.to_datetime(f["FlightDate"])
    dep = date + _hhmm(f["CRSDepTime"])
    arr0 = date + _hhmm(f["CRSArrTime"])
    block = pd.to_timedelta(f["CRSElapsedTime"].fillna(120), unit="min")
    cands = [arr0 + pd.Timedelta(days=d) for d in (-1, 0, 1)]
    resid = np.column_stack([(c - dep - block).abs().to_numpy() for c in cands])
    sched_local = arr0 + pd.to_timedelta(resid.argmin(axis=1) - 1, unit="D")
    landed = (f["Cancelled"] == 0) & (f["Diverted"] == 0) & f["ArrDelay"].notna() & f["TaxiIn"].notna()
    wheels_local = sched_local + pd.to_timedelta(f["ArrDelay"], unit="min") - pd.to_timedelta(f["TaxiIn"], unit="min")

    out = pd.DataFrame({"icao": f["Dest"].map(iata_to_icao), "cancelled": f["Cancelled"].astype(int),
                        "diverted": f["Diverted"].astype(int)}, index=f.index)
    sched_utc, wheels_utc = [], []
    for icao, idx in out.groupby("icao").groups.items():
        tz = timezones[icao]
        # Repeated fall-back hour: read as standard time (1 h/year); skipped spring-forward hour: shift on.
        kw = {"ambiguous": np.zeros(len(idx), dtype=bool), "nonexistent": "shift_forward"}
        sched_utc.append(sched_local[idx].dt.tz_localize(tz, **kw).dt.tz_convert("UTC"))
        wheels_utc.append(wheels_local[idx].where(landed[idx]).dt.tz_localize(tz, **kw).dt.tz_convert("UTC"))
    out["sched_arr_utc"] = pd.concat(sched_utc).reindex(out.index)
    out["wheels_on_utc"] = pd.concat(wheels_utc).reindex(out.index)
    # Check against BTS's own WheelsOn clock time (local hhmm) where present.
    ok = out["wheels_on_utc"].notna() & f["WheelsOn"].notna()
    clock = (wheels_local.dt.hour * 100 + wheels_local.dt.minute)[ok]
    out.attrs["wheels_on_match"] = float((clock == f.loc[ok, "WheelsOn"].astype(int) % 2400).mean())
    return out


def hourly_arrivals(times: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (icao, UTC hour on `grid`): arr_count (landed by floor(wheels_on)), sched_arr_count
    (incl. later-cancelled, by floor(scheduled arrival)), cancellations, diversions (by scheduled hour)."""
    frames = []
    for icao, g in times.groupby("icao"):
        landed = g["wheels_on_utc"].dropna().dt.floor("h").value_counts()
        sched = g.groupby(g["sched_arr_utc"].dt.floor("h"))
        frames.append(pd.DataFrame({
            "icao": icao, cfg.TIME: grid,
            cfg.TARGET: landed.reindex(grid, fill_value=0).to_numpy(),
            "sched_arr_count": sched.size().reindex(grid, fill_value=0).to_numpy(),
            "cancellations": sched["cancelled"].sum().reindex(grid, fill_value=0).to_numpy(),
            "diversions": sched["diverted"].sum().reindex(grid, fill_value=0).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    from .aar_data import load_airports

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=cfg.BTS_START)
    ap.add_argument("--end", default=cfg.BTS_END)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    iatas = [a["iata"] for a in load_airports().values()]
    download(args.start, args.end, iatas, workers=args.workers)


if __name__ == "__main__":
    main()
