# AAR / Delay-Risk Hackathon — Data & Features (P1)

Goal: features for (a) hourly arrival capacity at KATL/KDFW (proxy: realised arrivals/hr) and
(b) per-flight P(arr delay > 45 min), horizons h ∈ {3,6,9,12} h, leakage-safe.
Organiser assumptions: KATL runway heading 080°, KDFW runway heading 180°.
Out of scope for P1: splits, models, baselines, metrics, dashboard.

## Layout
- `data/raw/Flight_Data_With_Weather_Final.csv` — raw copy (from ~/Downloads). Never edit.
- `data/sample/sample_1000.csv` — first 1,000 raw rows (all airports, unfiltered). Committed.
- `data/interim/schema_profile.csv` — per-column profile (dtype, % missing, min/max, examples). Not committed.
- `config/airports.yaml` — per-ICAO runway heading, timezone, meteostat lookup. `src/data/` — P1 pipeline code.
- `scripts/` — profiling scripts + `run_pipeline.py` entry. Env: `.venv/bin/python`; `pip install -r requirements.txt`.
- Repo: https://github.com/KennotLee/autonomous (main). Team workflow rules in README.md.

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

## Weather: what the file really contains (verified against Meteostat)
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

## How to load (teammates)
```python
import pandas as pd
hr = pd.read_parquet("data/processed/hourly_features.parquet")   # 181,048 rows = 2 icao × 22,631 h × 4 horizons
fl = pd.read_parquet("data/processed/flight_features.parquet")   # 101,196 rows = 25,299 flights × 4 horizons
X = hr[hr.h == 6]   # pick ONE horizon per model; never train on oracle_* columns or target columns
```
Rebuild: `.venv/bin/python scripts/run_pipeline.py --icao KATL KDFW --start 2023-01-01 --end 2025-07-31 [--weather-scope all]`
then `.venv/bin/python tests/test_leakage.py` (must print ALL LEAKAGE CHECKS PASSED). All timestamps are tz-aware UTC.

## Output schemas (v1, 2026-10-31; branch p1)
Weather feature block `<F>` (from Meteostat, last obs hour ≤ t − h, tolerance 2 h; units in name):
- base: temp_c, dwpt_c, rhum_pct, prcp_mm (1-h total), wdir_deg (from, true), wspd_kt, wpgt_kt (98% NaN), pres_hpa (sea level), coco (1–27, model-filled)
- rolling, window (T−w, T] at obs time T, w ∈ {3,6}: wspd_max_{w}h_kt, wpgt_max_{w}h_kt, prcp_sum_{w}h_mm, pres_chg_{w}h_hpa (= pres(T) − pres(T−w))
- runway components (only airports with runway_heading_deg in config, else NaN): headwind_kt, crosswind_kt, tailwind_kt,
  gust_headwind_kt, gust_crosswind_kt, gust_tailwind_kt. θ = wdir − heading; head = s·cosθ; cross = |s·sinθ|; tail = max(0, −head); calm → 0.

**hourly_features.parquet** (one row per icao × target_hour_utc × h)
| column | meaning |
|---|---|
| icao | KATL / KDFW |
| target_hour_utc | start of UTC hour t (grid covers full local days 2023-01-01..2025-07-31) |
| h | horizon, hours ∈ {3,6,9,12} |
| feature_obs_time | UTC stamp of the weather hour used (≤ t − h) |
| `<F>` | weather features at icao |
| hour_local, dow_local (0=Mon), month_local, hour_utc | calendar of t (allowed) |
| sched_arr_count | flights scheduled to arrive in hour t (incl. cancelled) — schedule, known in advance |
| arr_count | **target**: landed (not cancelled/diverted) by floor(wheels_on_utc) |
| mean_arr_delay, share_delayed_45, cancellations | outcomes by scheduled-arrival hour (targets/eval only, NOT features) |
| capacity_ratio | arr_count / sched_arr_count, NaN if 0 (target/eval only) |
| oracle_`<base + components>` | weather stamped exactly at t — oracle comparison only |

**flight_features.parquet** (one row per flight × h; flights with ORIGIN or DEST in hubs)
| column | meaning |
|---|---|
| flight_id | `FL_DATE_<carrier><flnum>_<ORIGIN>_<DEST>` (unique) |
| origin_icao, dest_icao | ICAO codes (K/P+IATA or config overrides) |
| sched_time_utc | t = scheduled departure UTC |
| sched_arr_utc | scheduled arrival UTC (= t + CRS_ELAPSED_TIME) |
| h | horizon ∈ {3,6,9,12} |
| feature_obs_time | max(origin_, dest_feature_obs_time), all ≤ t − h |
| carrier, distance_mi, crs_elapsed_min | schedule info (allowed) |
| dep_hour_local, dow_local, month_local (origin tz), arr_hour_local (dest tz), dep_hour_utc | calendar (allowed) |
| origin_`<F>`, dest_`<F>` | weather features; **v1: only hub ends have weather** (non-hub side NaN until `--weather-scope all`) |
| delay_min | **target**: ARR_DELAY minutes (signed); NaN if cancelled/diverted |
| label_45 | **target**: delay_min > 45 (1/0), NaN if cancelled/diverted; base rate 10.7% |
| cancelled, diverted | 0/1 outcomes (not features) |
| oracle_origin_*, oracle_dest_* | weather at floor(sched dep hour) / floor(sched arr hour) — oracle only |

Known limitations: hourly arr_count is mostly 0/1 (sparse BTS sample: 0 in 82% of hours);
gust 97–99% missing; LAW (KLAW) and YUM (KYUM, Meteostat ends 2007) have no weather.

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
