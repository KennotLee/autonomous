# 📚 Data sources

Every external dataset used by this repo, how it is fetched, and its terms. All external data is fetched
by a script in this repo (no manual downloads). **Access date for all entries: 2026-10-31.**
Licence/terms are quoted from each source's own website; where that could not be verified it says "not checked".

| # | Source | Used by | Fetch script | Licence / terms |
|---|---|---|---|---|
| 0 | Organiser flight + weather CSV | P1 (all features and targets) | provided by organisers (not fetched) | organiser-provided |
| 1 | Meteostat hourly weather + station metadata | P1 features (both models) | `src/data/weather.py` via `scripts/run_pipeline.py` | CC BY 4.0 (attribution required) |
| 2 | BTS Reporting Carrier On-Time Performance (full months) | P3 delay model hub features | `pipeline/download.py` | not checked |
| 3 | Iowa Environmental Mesonet (IEM) ASOS/METAR archive | standalone tool, not used by any model | `src/data/fetch_weather.py` | public domain |

---

## 0. Organiser dataset: `Flight_Data_With_Weather_Final.csv`
- **What:** BTS on-time flight records joined with Meteostat-style weather columns (148,273 rows × 123 columns,
  2023-01-01 to 2025-07-31). Facts: CLAUDE.md "Verified facts about the raw data".
- **Provided by:** hackathon organisers. Placed in `data/raw/` locally; **never committed** (`.gitignore`).
- **Fields used:** `src/data/flights.py:12-14` (`USECOLS`: FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, TAIL_NUM,
  ORIGIN, DEST, CRS_DEP_TIME, CRS_ARR_TIME, ARR_TIME, WHEELS_ON, TAXI_IN, ARR_DELAY, CANCELLED, DIVERTED,
  CRS_ELAPSED_TIME, DISTANCE).
- **Not used:** the CSV's own weather columns (`*_ORIGIN`, `*_DEST`). They are joined 4–6 h off the true hour
  (verified against Meteostat; CLAUDE.md "Weather: what the file really contains").

## 1. Meteostat: hourly weather and station metadata
- **Website:** https://meteostat.net
- **Docs:** API hourly format https://dev.meteostat.net/api/point/hourly.html ·
  data formats/units https://dev.meteostat.net/formats.html · Python library https://dev.meteostat.net/python/
- **Endpoint actually called:** the `meteostat` Python package (v1.7.6, `requirements.txt`) downloads from
  `https://bulk.meteostat.net/v2/` (installed package, `meteostat/interface/base.py:21`).
- **Fetch script:** `src/data/weather.py:34-71` (`Stations().fetch()`, `Hourly(..., model=False/True).fetch()`),
  run by `scripts/run_pipeline.py`; cached in `data/interim/weather_raw/` (not committed).
- **Fields used:** `temp, dwpt, rhum, prcp, wdir, wspd, wpgt, pres` from observations (`src/data/weather.py:14`);
  `coco` from model-filled data (`src/data/weather.py:71-73`); station `icao` and `timezone` (`src/data/weather.py:36-38`).
  Units per Meteostat docs: °C, mm (1 h total), km/h, sea-level hPa, degrees; timestamps UTC.
- **Stations:** looked up by ICAO at runtime (e.g. KATL → 72219, KDFW → 72259); see `config/airports.yaml`.
- **Licence:** "Creative Commons Attribution 4.0 International (CC BY 4.0)" https://creativecommons.org/licenses/by/4.0
  (https://dev.meteostat.net/license.html). "You must give credit to Meteostat and its meteorological data providers."
  Terms: https://dev.meteostat.net/terms.html ("We cannot guarantee the accuracy or completeness…").
- **Attribution line (use in the demo):** Source: [Meteostat](https://meteostat.net) and its
  [data providers](https://dev.meteostat.net/providers), CC BY 4.0.
- **Also:** `pipeline/download.py:85-118` (P3's older pipeline) can fetch Meteostat with meteostat ≥ 2
  (`WX_PARAMS` at `pipeline/download.py:35`). The delay model does **not** need it (runs with `--skip-weather`).

## 2. BTS: Reporting Carrier On-Time Performance (1987–present)
- **Publisher:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS).
- **Website / table page:** https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=FGJ
- **Endpoint actually called:** `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip`
  (`pipeline/download.py:19-20`), one zip per month, 2023-01 to 2025-07.
- **Fetch script:** `python -m pipeline.download --skip-weather` (`pipeline/download.py`); saved to `data/raw/bts/` (not committed).
- **Fields used:** `pipeline/download.py:23-33` (`FLIGHT_COLS`: FlightDate, Reporting_Airline,
  Flight_Number_Reporting_Airline, Tail_Number, Distance, Origin, Dest, CRSDepTime, DepTime, DepDelay, CRSArrTime,
  ArrTime, ArrDelay, Cancelled, CancellationCode, Diverted, CRSElapsedTime, WeatherDelay, NASDelay).
  Only ATL/DFW flights are kept, to build hub congestion/state features (`src/models/delay/features.py`).
- **Why:** the organiser CSV is a ~0.5% sample, too thin to measure hub congestion.
- **Licence:** not checked (BTS policy page https://www.bts.gov/website-policies returned HTTP 403; the table page
  states no terms).

## 3. Iowa Environmental Mesonet (IEM): ASOS/METAR archive
- **Publisher:** Iowa Environmental Mesonet, Iowa State University.
- **Website / download page:** https://mesonet.agron.iastate.edu/request/download.phtml
- **API help:** https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
- **Endpoint actually called:** `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (`src/data/fetch_weather.py:35`).
- **Fetch script:** `src/data/fetch_weather.py` (standalone; **not used by the pipeline or any model**).
- **Fields used:** `tmpf, dwpf, relh, drct, sknt, gust, mslp, alti, p01i, vsby, wxcodes, peak_wind_gust,
  peak_wind_drct, metar` (`src/data/fetch_weather.py:36-37`). Units per IEM docs: °F, %, degrees from true north,
  knots, mb, inches, miles.
- **Licence / terms:** https://mesonet.agron.iastate.edu/disclaimer.php: "The materials found on this website are in
  the public domain and may be used freely by anyone for any lawful purpose." "Attributing the Iowa Environmental
  Mesonet of Iowa State University would be appreciated." Provided "without any warranty of accuracy."
- **Access limits:** 1 request/s per IP (the script waits 1.5 s between requests).

## Not external data
Python packages (pandas, LightGBM, scikit-learn, …) are listed in `requirements.txt`. No API keys or credentials are used.
