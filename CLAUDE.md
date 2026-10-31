# Delay-Risk Hackathon (Challenge #5) — Data & Features (P1)

**Organiser update (2026-10-31, overrides earlier goals):** main target = severe delay per ATL/DFW event:
`ARR_DELAY > 45` for arrivals at the hub, `DEP_DELAY > 45` for departures from it (~10% positive).
Hourly AAR is optional and **dropped** (flights file is a ~0.5% sample). Weather features come ONLY from the
organiser hourly file; the flights CSV's weather is allowed only as `oracle_*`. Leakage rule: weather observed
at least 3 h before the scheduled time. Horizons h ∈ {3,6,9,12}. Pipeline accepts any configured airport (IATA or ICAO).
Organiser assumptions: KATL runway heading 080°, KDFW runway heading 180° (data guide xlsx).
Out of scope for P1: splits, models, baselines, metrics, dashboard.

## Layout
- `data/raw/Flight_Data_With_Weather_Final.csv`, `data/raw/challenge5_hourly_weather_ATL_DFW.csv`,
  `data/raw/challenge5_data_guide.xlsx` — organiser files. Never edit, never committed.
- `data/sample/sample_1000.csv` — first 1,000 raw flight rows. Committed.
- `config/airports.yaml` — per-ICAO iata, timezone, runway heading; weather settings; horizons; delay threshold.
- `src/data/` — P1 code: `events.py` (flights → dep/arr events, UTC, labels), `hourly_weather.py` (organiser
  hourly file → features), `build_events.py` (as-of join → event_features), `config.py`.
- `scripts/run_pipeline.py` — single entry. `tests/test_leakage.py` — leakage test (~2 s).
- `src/data/fetch_weather.py` — optional standalone METAR/ASOS downloader (IEM); not used by the pipeline.
- Repo: https://github.com/KennotLee/autonomous. P1 work on branch `p1`; integrator merges into `main`.

## Verified facts about the raw data (full file, 2026-10-31)
- 148,273 rows × 123 cols, BTS On-Time Performance + 14 Meteostat-style weather cols.
- FL_DATE is ISO `YYYY-MM-DD` (100% of rows), not M/D/YYYY. Range 2023-01-01..2025-07-31.
- Only 25,299 rows (17%) touch ATL or DFW; 206 other airports appear on those flights.
- **Sparse sample:** ~5.7 ATL and ~4.2 DFW non-cancelled arrivals/day (real traffic is ~1,000+/day).
  Per (date, hour) with any arrival: mean 1.1–1.2, max 3–4. arr_count is mostly 0/1.
- BTS hhmm times are local (dep = origin tz, arr = dest tz); values of 2400 occur (DEP 15, ARR 104, WHEELS_ON 74).
- ARR_DELAY NaN for 2,278 rows = 1,901 cancelled + 377 diverted.
- `*_DELAY` cause columns are post-hoc attributions → never use as features.
- No wind direction column in the file.

## Flights CSV weather (now oracle_* only; verified against Meteostat)
- File weather == Meteostat hourly (model=True) for the airport's station, joined on
  **FL_DATE + scheduled LOCAL clock hour treated as a UTC hour**. Test on all ATL/DFW departures:
  match 100% (temp/wspd/pres) with naive key, ≤12% when the local time is properly converted to UTC.
  ⇒ upstream join is misaligned by the UTC offset (4–6 h). Do not use the file's weather columns as-is.
- Meteostat docs (dev.meteostat.net API hourly + formats): timestamps UTC; temp °C, prcp = 1-h total mm,
  snow = depth mm, wdir °, wspd km/h (avg), wpgt km/h (peak gust), pres = sea-level hPa, coco 1–27.
  Docs do not say whether wdir is "from" or "to" (underlying METAR/ISD uses "from").
- Meteostat via package: stations looked up by ICAO from `Stations().fetch()` → KATL=72219
  (America/New_York), KDFW=72259 (America/Chicago). Hourly 2023-01..2025-07, model=False null shares:
  KATL wdir 0.5%, wpgt 98.7%, snow 100%, coco 96.9%; KDFW wdir 0.6%, wpgt 97.3%, snow 100%, coco 97.7%.
  coco is only populated with model=True (model-filled). Meteostat does NOT give better gusts.
- In the file, wpgt 0 in ~97.8% rows and snow 0 in 100% rows → NaNs filled with 0 upstream; treat as missing.


## Organiser hourly weather file (checked 2026-10-31)
- 45,264 rows = ATL, DFW × 22,632 hours, 2023-01-01 00:00 to 2025-07-31 23:00 UTC; columns: airport (IATA),
  runway_heading_deg, time_utc, time_local, temp_c, dewpoint_c, rel_humidity_pct, precip_mm, wind_dir_deg,
  wind_speed_kmh, wind_gust_kmh, pressure_hpa, condition_code, headwind_kmh, crosswind_kmh, tailwind_kmh.
- `wind_gust_kmh` 100% empty (not used). precip_mm 0.12% and condition_code 0.03% missing. Calm = dir 0, speed 0.
- `time_local` is wall clock with DST (2 duplicate and 3 missing local hours per airport) → we join on `time_utc`.
- **Stamp ≠ observation time:** vs IEM METAR (KATL 2024-06-30/07-01) a row stamped HH:00 holds either the
  HH−1:52 or the HH:52 report (e.g. 00:00 = 00:52 METAR). So a row is treated as available at `time_utc + 1 h`.

## How to load (teammates)
```python
import pandas as pd
ev = pd.read_parquet("data/processed/event_features.parquet")  # 102,448 rows = 25,612 events × 4 horizons
X = ev[ev.h == 6]                    # one h per model, or keep all h and use h as a feature
y = X.label_45                       # NaN for cancelled (and diverted arrivals): drop or set a policy
```
Rebuild: `.venv/bin/python scripts/run_pipeline.py --airports ATL DFW` then `.venv/bin/python tests/test_leakage.py`.
All `*_utc` timestamps are tz-aware UTC; `sched_time_local` is naive local wall clock of the event airport.

## Output schema: `data/processed/event_features.parquet` (v2, 2026-10-31)
One row per (event_id, h). Events: 16,151 dep + 9,461 arr (ATL↔DFW flights give one of each); label_45 rate
10.7% (dep 11.1%, arr 10.0%; KATL 9.1%, KDFW 12.6%); 381 events with label NaN; 0 events dropped for DST.

| column | meaning |
|---|---|
| event_id | `flight_key + "_" + event_type` (unique per event) |
| flight_key | `FL_DATE_<carrier><flnum>_<ORIGIN>_<DEST>` (+ `_<CRS_DEP_TIME>` if duplicated) |
| event_type | `dep` (flight leaves the hub) or `arr` (flight lands at the hub) |
| airport, other_airport | event hub ICAO; the other end's ICAO (overrides, else P+IATA for AK/HI/GU, else K+IATA) |
| sched_time_local | scheduled time at the event airport, local wall clock (naive) |
| sched_time_utc | same in UTC (t). Arrival date resolved via CRS_ELAPSED_TIME; matches previous method 100% |
| h | horizon, hours ∈ {3,6,9,12} |
| feature_obs_time | availability time (= time_utc + 1 h) of the weather row used; always ≤ sched_time_utc − h |
| temp, prcp, wspd, pres, coco, wdir | event-airport weather (°C, mm, km/h, hPa, Meteostat code, deg FROM true) |
| headwind, crosswind, tailwind | provided components, km/h (runway 080° ATL / 180° DFW) |
| ts_flag, snow_flag | coco ∈ {25,26,27} thunderstorm; coco ∈ {14,15,16} snow (data guide groups); NaN if coco NaN |
| `<v>_lag{1,3,6}h` | v ∈ wspd, crosswind, tailwind, prcp, temp, k hours before the used row |
| pres_tend_3h, prcp_sum_6h | pressure change over 3 h; precip sum over the 6 h ending at the used row |
| other_feature_obs_time, other_* | same weather set at other_airport; only non-null for ATL↔DFW (2.4% of rows) |
| hour, dow, month, carrier | calendar of sched_time_local (dow 0 = Monday); operating carrier |
| delay_min | **target**: DEP_DELAY (dep) or ARR_DELAY (arr), minutes; NaN if cancelled or no delay recorded |
| label_45 | **target**: delay_min > 45 (1/0); NaN if delay_min NaN |
| cancelled, diverted | 0/1 outcomes (not features) |
| oracle_temp, oracle_prcp, oracle_wspd, oracle_pres, oracle_coco | flights-CSV weather at event time (comparison only, never features) |

Never features: ids, `sched_time_*`, `feature_obs_time`, `other_feature_obs_time`, targets, `cancelled`, `diverted`, `oracle_*`.
Dropped in v2 (organiser update): `hourly_features.parquet`, `flight_features.parquet`, arr_count, capacity_ratio,
sched counts, all Meteostat-fetched weather.

## Raw schema
Weather meanings/units are from Meteostat docs; the "@ (date, sched local hr)" join key is the misaligned upstream join described above.

| column | dtype | units | % missing | meaning |
|---|---|---|---|---|
| YEAR | int64 |  | 0.0 | Year of flight date |
| QUARTER | int64 |  | 0.0 | Quarter 1-4 |
| MONTH | int64 |  | 0.0 | Month 1-12 |
| DAY_OF_MONTH | int64 |  | 0.0 | Day of month |
| DAY_OF_WEEK | int64 |  | 0.0 | 1=Mon..7=Sun (BTS) |
| FL_DATE | object | local date | 0.0 | Flight date (scheduled departure, local) |
| OP_UNIQUE_CARRIER | object |  | 0.0 | Operating carrier code |
| OP_CARRIER_AIRLINE_ID | int64 |  | 0.0 | DOT carrier id |
| OP_CARRIER | object |  | 0.0 | Carrier code |
| TAIL_NUM | object |  | 0.22 | Aircraft tail number |
| OP_CARRIER_FL_NUM | float64 |  | 0.0 | Flight number |
| ORIGIN_AIRPORT_ID | int64 |  | 0.0 | DOT airport id |
| ORIGIN_AIRPORT_SEQ_ID | int64 |  | 0.0 | DOT airport seq id |
| ORIGIN_CITY_MARKET_ID | int64 |  | 0.0 | DOT city market id |
| ORIGIN | object | IATA | 0.0 | Origin airport (IATA, e.g. ATL) |
| ORIGIN_CITY_NAME | object |  | 0.0 | Origin city |
| ORIGIN_STATE_ABR | object |  | 0.0 | Origin state |
| ORIGIN_STATE_FIPS | int64 |  | 0.0 | State FIPS |
| ORIGIN_STATE_NM | object |  | 0.0 | State name |
| ORIGIN_WAC | int64 |  | 0.0 | World area code |
| DEST_AIRPORT_ID | int64 |  | 0.0 | DOT airport id |
| DEST_AIRPORT_SEQ_ID | int64 |  | 0.0 | DOT airport seq id |
| DEST_CITY_MARKET_ID | int64 |  | 0.0 | DOT city market id |
| DEST | object | IATA | 0.0 | Destination airport (IATA) |
| DEST_CITY_NAME | object |  | 0.0 | Dest city |
| DEST_STATE_ABR | object |  | 0.0 | Dest state |
| DEST_STATE_FIPS | int64 |  | 0.0 | State FIPS |
| DEST_STATE_NM | object |  | 0.0 | State name |
| DEST_WAC | int64 |  | 0.0 | World area code |
| CRS_DEP_TIME | int64 | hhmm local | 0.0 | Scheduled departure time |
| DEP_TIME | float64 | hhmm local | 1.23 | Actual gate departure |
| DEP_DELAY | float64 | min | 1.23 | Dep delay, signed (early<0) |
| DEP_DELAY_NEW | float64 | min | 1.23 | Dep delay, early clipped to 0 |
| DEP_DEL15 | float64 | 0/1 | 1.23 | Dep delay >=15 |
| DEP_DELAY_GROUP | float64 | 15-min bins | 1.23 | Dep delay group |
| DEP_TIME_BLK | object | hhmm-hhmm local | 0.0 | Scheduled dep hour block |
| TAXI_OUT | float64 | min | 1.27 | Taxi-out time |
| WHEELS_OFF | float64 | hhmm local | 1.27 | Wheels-off (runway takeoff) |
| WHEELS_ON | float64 | hhmm local | 1.31 | Wheels-on (runway touchdown) |
| TAXI_IN | float64 | min | 1.31 | Taxi-in time |
| CRS_ARR_TIME | int64 | hhmm local | 0.0 | Scheduled arrival time |
| ARR_TIME | float64 | hhmm local | 1.31 | Actual gate arrival |
| ARR_DELAY | float64 | min | 1.54 | Arr delay, signed |
| ARR_DELAY_NEW | float64 | min | 1.54 | Arr delay, early clipped to 0 |
| ARR_DEL15 | float64 | 0/1 | 1.54 | Arr delay >=15 |
| ARR_DELAY_GROUP | float64 | 15-min bins | 1.54 | Arr delay group |
| ARR_TIME_BLK | object | hhmm-hhmm local | 0.0 | Scheduled arr hour block |
| CANCELLED | float64 | 0/1 | 0.0 | Cancelled |
| CANCELLATION_CODE | object | A-D | 98.72 | A carrier, B weather, C NAS, D security |
| DIVERTED | float64 | 0/1 | 0.0 | Diverted |
| CRS_ELAPSED_TIME | float64 | min | 0.0 | Scheduled elapsed |
| ACTUAL_ELAPSED_TIME | float64 | min | 1.54 | Actual elapsed |
| AIR_TIME | float64 | min | 1.54 | Airborne time |
| FLIGHTS | float64 | count | 0.0 | Always 1 |
| DISTANCE | float64 | miles | 0.0 | Route distance |
| DISTANCE_GROUP | int64 | 250-mi bins | 0.0 | Distance group |
| CARRIER_DELAY | float64 | min | 78.78 | Cause attribution (only when ARR_DELAY>=15) |
| WEATHER_DELAY | float64 | min | 78.78 | Cause attribution (post-hoc; LEAKY) |
| NAS_DELAY | float64 | min | 78.78 | Cause attribution (post-hoc; LEAKY) |
| SECURITY_DELAY | float64 | min | 78.78 | Cause attribution (post-hoc; LEAKY) |
| LATE_AIRCRAFT_DELAY | float64 | min | 78.78 | Cause attribution (post-hoc; LEAKY) |
| FIRST_DEP_TIME | float64 | hhmm local | 99.27 | First gate dep (gate return cases) |
| TOTAL_ADD_GTIME | float64 | min | 99.27 | Added ground time (gate return) |
| LONGEST_ADD_GTIME | float64 | min | 99.27 | Longest added ground time |
| DIV_AIRPORT_LANDINGS | float64 | count | 0.0 | Diversion landings (9 = BTS code) |
| DIV_REACHED_DEST | float64 | 0/1 | 99.75 | Diverted flight reached dest |
| DIV_ACTUAL_ELAPSED_TIME | float64 | min | 99.77 | Elapsed, diverted |
| DIV_ARR_DELAY | float64 | min | 99.77 | Arr delay, diverted |
| DIV_DISTANCE | float64 | miles | 99.75 | Diversion distance |
| DIV1_AIRPORT | object |  | 99.74 | 1st diversion airport fields |
| DIV1_AIRPORT_ID | float64 |  | 99.74 | 1st diversion airport fields |
| DIV1_AIRPORT_SEQ_ID | float64 |  | 99.74 | 1st diversion airport fields |
| DIV1_WHEELS_ON | float64 |  | 99.74 | 1st diversion airport fields |
| DIV1_TOTAL_GTIME | float64 |  | 99.74 | 1st diversion airport fields |
| DIV1_LONGEST_GTIME | float64 |  | 99.74 | 1st diversion airport fields |
| DIV1_WHEELS_OFF | float64 |  | 99.77 | 1st diversion airport fields |
| DIV1_TAIL_NUM | object |  | 99.77 | 1st diversion airport fields |
| DIV2_* .. DIV5_* (32 cols) | mixed | | 100.0 | All-null; drop |
| temp_ORIGIN | float64 | °C | 0.0 | Air temperature; range -35..48 @ origin airport, @ (date, sched local hr) misaligned |
| prcp_ORIGIN | float64 | mm | 0.0 | 1-h precipitation total @ origin airport, @ (date, sched local hr) misaligned |
| wspd_ORIGIN | float64 | km/h | 0.0 | Sustained wind speed; 11% zeros (calm or filled?) @ origin airport, @ (date, sched local hr) misaligned |
| wpgt_ORIGIN | float64 | km/h | 0.0 | Peak gust; 97.8% zeros -> almost surely NaN filled with 0 @ origin airport, @ (date, sched local hr) misaligned |
| pres_ORIGIN | float64 | hPa | 0.0 | Sea-level pressure @ origin airport, @ (date, sched local hr) misaligned |
| snow_ORIGIN | float64 | mm | 0.0 | Snow depth; 100% zeros -> no information @ origin airport, @ (date, sched local hr) misaligned |
| coco_ORIGIN | float64 | code | 0.0 | Meteostat weather condition code 1-27; 0.8% zeros = likely filled NaN @ origin airport, @ (date, sched local hr) misaligned |
| temp_DEST | float64 | °C | 0.0 | Air temperature; range -35..48 @ dest airport, @ (date, sched local hr) misaligned |
| prcp_DEST | float64 | mm | 0.0 | 1-h precipitation total @ dest airport, @ (date, sched local hr) misaligned |
| wspd_DEST | float64 | km/h | 0.0 | Sustained wind speed; 11% zeros (calm or filled?) @ dest airport, @ (date, sched local hr) misaligned |
| wpgt_DEST | float64 | km/h | 0.0 | Peak gust; 97.8% zeros -> almost surely NaN filled with 0 @ dest airport, @ (date, sched local hr) misaligned |
| pres_DEST | float64 | hPa | 0.0 | Sea-level pressure @ dest airport, @ (date, sched local hr) misaligned |
| snow_DEST | float64 | mm | 0.0 | Snow depth; 100% zeros -> no information @ dest airport, @ (date, sched local hr) misaligned |
| coco_DEST | float64 | code | 0.0 | Meteostat weather condition code 1-27; 0.8% zeros = likely filled NaN @ dest airport, @ (date, sched local hr) misaligned |

## Decisions log (approved by team lead 2026-10-31)
1. Weather source = Meteostat hourly (UTC) per ICAO, NOT the CSV weather columns (misaligned 4–6 h).
2. Numeric weather from model=False (observations only); `coco` only from model=True (model-filled).
3. UTC: sched_dep_utc = FL_DATE+CRS_DEP_TIME localised in origin tz; sched_arr_utc = sched_dep_utc +
   CRS_ELAPSED_TIME; actual gate arr = sched_arr_utc + ARR_DELAY; wheels_on_utc = gate arr − TAXI_IN.
   Validated against local hhmm columns. Time zones from Meteostat station metadata.
4. arr_count = non-cancelled, non-diverted arrivals by floor(wheels_on_utc, hour), full hourly grid, 0-filled.
   sched_arr_count by floor(sched_arr_utc) incl. cancelled. capacity_ratio NaN when sched_arr_count = 0.
5. delay_min = ARR_DELAY (signed); label_45 = delay_min > 45; cancelled/diverted kept with NaN labels +
   `cancelled`, `diverted` columns.
6. Flight t = sched_time_utc = scheduled departure UTC; origin_* and dest_* both as-of t − h;
   feature_obs_time = max(origin, dest obs time).
7. Weather fetched for every airport on ATL/DFW flights; wind components only where a runway heading
   is configured (KATL, KDFW), NaN elsewhere.
8. snow dropped; missing gust left NaN (no imputation).
9. Meteostat stamp HH:00 = obs time; feature allowed iff stamp ≤ t − h; as-of tolerance 2 h (else NaN).
10. Rolling windows 3 h and 6 h ending at the obs time: max gust, max wspd, prcp sum, pressure change.
11. wdir = meteorological "from"; runway headings 080/180 used as given (no magnetic correction).
A. Duplicate sample removed; schema_profile.csv → data/interim/.
B. Prediction CSVs: aar_preds.csv (icao, target_hour_utc, h, pred_arr_count);
   delay_preds.csv (flight_id, sched_time_utc, h, p_delay45).

(Decisions 1, 2, 4, 7, 8, 9, 10 above are superseded by C1–C4 for the v2 event pipeline.)
C1. (organiser update) Target per event: dep → DEP_DELAY > 45, arr → ARR_DELAY > 45; ATL↔DFW = 2 events;
    cancelled → label NaN. Hourly panel / AAR dropped. Weather only from the organiser hourly file.
C2. Join weather on UTC (time_local has DST duplicates/gaps); flight local → UTC with hub tz; DST-ambiguous → dropped (0).
C3. Hourly row available at time_utc + 1 h (stamp can precede the METAR by ~52 min); as-of ≤ t − h, tolerance 2 h.
C4. Units kept as provided (km/h); coco flags use the data-guide groups; gust/snow placeholders ignored.
