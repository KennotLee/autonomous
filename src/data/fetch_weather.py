"""Download METAR/ASOS observations for any ICAO code(s) and date range from the Iowa Environmental
Mesonet (IEM) ASOS archive. Standalone utility: not used by run_pipeline.py.

Source (verified 2026-10-31 from IEM docs):
  endpoint  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py   (HTTP GET)
  help      https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
  variables https://mesonet.agron.iastate.edu/request/download.phtml
  Station = "three or four character site identifier" (tested: KATL and ATL both work).
  Limits: 1 request/s per IP (HTTP 429 if exceeded), 503 under load, 422 if > 1000 station-years.
  Archive is "as-is ... very little quality control".

Output columns (IEM units, per IEM docs):
  icao (as requested), station (IEM id), valid_utc (tz-aware UTC obs time),
  tmpf °F, dwpf °F, relh %, drct deg FROM TRUE NORTH, sknt kt, gust kt, mslp mb (sea level),
  alti inHg (altimeter), p01i inches (1-h precip since last reset; varies by site), vsby miles,
  wxcodes (present-weather codes), peak_wind_gust kt, peak_wind_drct deg, metar (raw text).
  Missing = NaN. report_type default: 3 (routine) + 4 (specials).

Usage:
  .venv/bin/python src/data/fetch_weather.py --icao KATL KDFW --start 2024-07-01 --end 2024-07-31
  (--end is inclusive; files -> data/interim/metar/{ICAO}_{start}_{end}.parquet)
"""
import argparse
import io
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
VARS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "mslp", "alti", "p01i", "vsby",
        "wxcodes", "peak_wind_gust", "peak_wind_drct", "metar"]
NUMERIC = [v for v in VARS if v not in ("wxcodes", "metar")]
MIN_INTERVAL_S = 1.5  # IEM throttle is 1 req/s per IP; stay above it
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "interim" / "metar"
_last_call = [0.0]


def _get(params: dict, retries: int = 5) -> str:
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aar-hackathon-fetch_weather/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"IEM HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"IEM request failed for {url}: {e}") from e
    raise RuntimeError("unreachable")


def _year_chunks(start: date, end_incl: date):
    """Split [start, end_incl] into calendar-year chunks; yields (sts, ets_exclusive)."""
    s = start
    while s <= end_incl:
        e = min(date(s.year + 1, 1, 1), end_incl + timedelta(days=1))
        yield s, e
        s = e


def fetch_metar(icao: str, start: str, end: str, report_types=(3, 4)) -> pd.DataFrame:
    """All METAR/ASOS reports for one station between start and end (inclusive, UTC dates)."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if e < s:
        raise ValueError("end before start")
    parts = []
    for cs, ce in _year_chunks(s, e):
        text = _get({
            "station": icao, "data": ",".join(VARS),
            "sts": f"{cs.isoformat()}T00:00Z", "ets": f"{ce.isoformat()}T00:00Z",
            "tz": "Etc/UTC", "format": "onlycomma", "missing": "empty", "trace": "empty",
            "report_type": ",".join(map(str, report_types)),
        })
        if not text.startswith("station,valid"):
            raise RuntimeError(f"Unexpected IEM response for {icao}: {text[:200]!r}")
        parts.append(pd.read_csv(io.StringIO(text), dtype={"wxcodes": "string", "metar": "string"}))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "icao", icao.upper())
    df["valid_utc"] = pd.to_datetime(df.pop("valid"), utc=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cols = ["icao", "station", "valid_utc"] + VARS
    return df[cols].drop_duplicates().sort_values("valid_utc").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icao", nargs="+", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report-types", default="3,4", help="IEM report_type: 1=HFMETAR, 3=routine, 4=specials")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rts = tuple(int(x) for x in a.report_types.split(","))
    for icao in a.icao:
        df = fetch_metar(icao, a.start, a.end, rts)
        if df.empty:
            print(f"WARNING {icao}: 0 reports — unknown station id or no data in range; nothing written",
                  file=sys.stderr)
            continue
        p = out / f"{icao.upper()}_{a.start}_{a.end}.parquet"
        df.to_parquet(p, index=False)
        print(f"{icao}: {len(df)} reports, {df.valid_utc.min()} .. {df.valid_utc.max()}, "
              f"gust non-null {df.gust.notna().mean():.1%} -> {p}")


if __name__ == "__main__":
    sys.exit(main())
