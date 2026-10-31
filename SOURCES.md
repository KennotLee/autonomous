# 📚 Data sources

Every external dataset used by this repo, how it is fetched, and its terms. All external data is fetched
by a script in this repo (no manual downloads). **Access date for all entries: 2026-10-31.**
Licence/terms are quoted from each source's own website; where that could not be verified it says "not checked".

| # | Source | Used by | Fetch script | Licence / terms |
|---|---|---|---|---|
| 0 | Organiser flight + weather CSV | P1 events and targets (its weather only as `oracle_*`) | provided by organisers (not fetched) | organiser-provided |
| 0b | Organiser hourly weather ATL/DFW + data guide | P1 weather features (the only weather source in v2) | provided by organisers (not fetched) | organiser-provided; derived from Meteostat (see #1) |
| 1 | Meteostat hourly weather (via the organiser file) | v2: indirectly — the organiser hourly file is Meteostat/NOAA ISD | none in v2 (the v1 Meteostat fetch code was removed) | CC BY 4.0 (attribution required) |
| 2 | BTS Reporting Carrier On-Time Performance (full months) | P2 optional AAR target (arrivals per hour) | `src/models/aar_bts.py` | not verified (site blocks automated checks) |
| 3 | Iowa Environmental Mesonet (IEM) ASOS/METAR archive | standalone tool, not used by any model | `src/data/fetch_weather.py` | public domain |

---

## 0. Organiser dataset: `Flight_Data_With_Weather_Final.csv`
- **What:** BTS on-time flight records joined with Meteostat-style weather columns (148,273 rows × 123 columns,
  2023-01-01 to 2025-07-31). Facts: CLAUDE.md "Verified facts about the raw data".
- **Provided by:** hackathon organisers. Placed in `data/raw/` locally; **never committed** (`.gitignore`).
- **Fields used:** `src/data/events.py:19-21` (`USECOLS`: FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST,
  ORIGIN/DEST_STATE_ABR, CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME, DEP_DELAY, ARR_DELAY, CANCELLED, DIVERTED,
  plus temp/prcp/wspd/pres/coco _ORIGIN/_DEST as `oracle_*` only).
- **Weather columns:** attached at flight time (and 4–6 h off the true hour, CLAUDE.md), so used only as `oracle_*`;
  `wpgt_*`/`snow_*` ignored (organiser placeholders).

## 0b. Organiser hourly weather: `challenge5_hourly_weather_ATL_DFW.csv` (+ `challenge5_data_guide.xlsx`)
- **What:** hourly weather for ATL and DFW, 2023-01-01 to 2025-07-31, UTC and local time, wind direction and
  headwind/crosswind/tailwind (km/h) for runways 080°/180°. Data guide: "Meteostat hourly (NOAA ISD), stations 72219 (ATL) and 72259 (DFW)".
- **Provided by:** hackathon organisers; placed in `data/raw/`; never committed. Path in `config/airports.yaml:31`.
- **Fields used:** `src/data/hourly_weather.py:13-15` (temp_c, precip_mm, wind_speed_kmh, pressure_hpa, condition_code,
  wind_dir_deg, headwind_kmh, crosswind_kmh, tailwind_kmh) + `time_utc`. Not used: `wind_gust_kmh` (100% empty), `time_local` (DST duplicates).
- **Attribution:** as Meteostat-derived data, credit Meteostat (#1).

## 1. Meteostat: hourly weather and station metadata (v1 fetch removed in v2; still the upstream of #0b)
- **Website:** https://meteostat.net
- **Docs:** API hourly format https://dev.meteostat.net/api/point/hourly.html ·
  data formats/units https://dev.meteostat.net/formats.html · Python library https://dev.meteostat.net/python/
- **Endpoint actually called:** the `meteostat` Python package (v1.7.6, `requirements.txt`) downloads from
  `https://bulk.meteostat.net/v2/` (installed package, `meteostat/interface/base.py:21`).
- **v2 status:** the P1 pipeline no longer fetches Meteostat (organiser update: use only the provided hourly file).
  v1 fetched it via `src/data/weather.py` (deleted in v2; see git history before commit `p1: switch to provided hourly weather`).
- **Units (Meteostat docs):** °C, mm (1 h total), km/h, sea-level hPa, degrees; timestamps UTC.
- **Stations:** looked up by ICAO at runtime (e.g. KATL → 72219, KDFW → 72259); see `config/airports.yaml`.
- **Licence:** "Creative Commons Attribution 4.0 International (CC BY 4.0)" https://creativecommons.org/licenses/by/4.0
  (https://dev.meteostat.net/license.html). "You must give credit to Meteostat and its meteorological data providers."
  Terms: https://dev.meteostat.net/terms.html ("We cannot guarantee the accuracy or completeness…").
- **Attribution line (use in the demo):** Source: [Meteostat](https://meteostat.net) and its
  [data providers](https://dev.meteostat.net/providers), CC BY 4.0.

## 2. BTS: Reporting Carrier On-Time Performance (1987–present)
- **Publisher:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS).
- **Website / table page:** https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=FGJ
- **Endpoint actually called:** `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip`
  (`src/models/aar_bts.py:29-30`), one zip per month, 2023-01 to 2025-07 (`src/models/aar_config.py:24`).
- **Fetch script:** `python -m src.models.aar_bts --start 2023-01 --end 2025-07` (resumable); filtered monthly parquet cached in
  `data/interim/bts/` (`src/models/aar_config.py:22`, not committed). The monthly zips (~32 MB each) are not kept.
- **Fields used:** `src/models/aar_bts.py:31-33` (FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest,
  CRSDepTime, CRSArrTime, CRSElapsedTime, ArrDelay, TaxiIn, WheelsOn, Cancelled, Diverted). Only flights to configured airports.
- **Used by:** P2's optional arrivals (AAR) model only, as the target and recent-arrivals inputs. **Not used by the delay model (P3).**
- **Why:** the organiser flights file is a ~0.5% sample, so hourly arrivals can't be measured from it; the data guide allows teams
  to "pull the full BTS on-time data automatically".
- **Licence:** not verified. Checked 2026-10-31 15:20 SGT: https://www.bts.gov/website-policies and
  https://www.bts.gov/policies-and-notices return HTTP 403 to automated requests; the TranStats table page states no
  terms. BTS is a U.S. federal agency, but public-domain status is **not confirmed here**: check in a browser before claiming it.

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
