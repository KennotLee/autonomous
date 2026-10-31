# 🧩 Assumptions (v2: event-level pipeline after the organiser update)

**Given by** = *organiser* (updated problem statement and `data/raw/challenge5_data_guide.xlsx`, recorded in
CLAUDE.md lines 3–9) or *ours* (team decision). "Impact if wrong" is our estimate unless a number is cited.

## ✈️ Problem framing

1. **Target = severe delay per event:** `DEP_DELAY > 45` for departures from ATL/DFW, `ARR_DELAY > 45` for arrivals there. An ATL↔DFW flight gives one departure and one arrival event.
   - Where: `src/data/events.py:49-50,78-79`; `config/airports.yaml:44`.
   - Given by: organiser (threshold, delay column per type). **Ours:** "strictly greater"; two events for ATL↔DFW flights.
   - Result: label rate 10.7% (`data/processed/build_meta.json`), matching the organisers' "about 10%".
2. **Cancelled events get `label_45` = NaN**; diverted arrivals have no `ARR_DELAY`, so they are NaN too.
   - Where: `src/data/events.py:78-79`.
   - Given by: organiser (cancelled); ours (diverted follows from missing `ARR_DELAY`).
   - Impact: 381 unlabelled events (`build_meta.json`).
3. **Hourly AAR is not built.**
   - Why: the organisers say it is optional and can't be measured from the ~0.5% sample.
   - Given by: organiser.
4. **Runway headings: ATL 080°, DFW 180°.**
   - Where: `config/airports.yaml:10,14`; data guide sheet `weather_columns`.
   - Given by: organiser. The provided headwind/crosswind/tailwind already use these headings.
5. **Horizons h ∈ {3, 6, 9, 12} h, minimum 3 h.**
   - Where: `config/airports.yaml:43`; enforced in `tests/test_leakage.py` (`test_obs_times`).
   - Given by: organiser.

## 🕒 Time handling

6. **Flight times are local hhmm at the event airport; weather is joined on UTC.**
   - Why: `time_local` in the hourly file has DST duplicates and gaps (CLAUDE.md "Organiser hourly weather file").
   - Where: `src/data/events.py:69-73`.
   - Given by: organiser (local times); ours (join on UTC).
7. **Arrival date** = FL_DATE −1/0/+1 day, whichever makes arrival − departure closest to `CRS_ELAPSED_TIME`.
   - Why: avoids needing the other airport's time zone; a wrong day is ≥ 17 h off.
   - Where: `src/data/events.py:42-46`.
   - Given by: ours. Matches the previous, validated tz-based method on 100% of events (integration check 2026-10-31).
8. **Local times in the DST-ambiguous or nonexistent hour → NaT, and the event is dropped.**
   - Where: `src/data/events.py:73-75`.
   - Given by: ours. Events dropped: 0 (`build_meta.json`).
9. **A weather row stamped `time_utc` is available at `time_utc + 1 h`.** A feature is allowed only if its availability time ≤ `sched_time_utc − h`; we use the latest such row within 2 h.
   - Why: the stamp can precede the METAR it holds by up to ~52 min (checked against IEM METAR for KATL 2024-06-30/07-01; CLAUDE.md).
   - Where: `config/airports.yaml:34-35`; `src/data/hourly_weather.py:50`; `src/data/build_events.py:24-25`; `tests/test_leakage.py` (`test_availability_lag`).
   - Given by: ours.
   - Impact: some features are about 1 h older than strictly necessary (a small accuracy cost). Without it, some features could break the 3 h rule by up to ~52 min.

## 🌦️ Weather

10. **Weather features come only from the organiser hourly file** (Meteostat / NOAA ISD, stations 72219 and 72259).
    - Where: `config/airports.yaml:31`; `src/data/hourly_weather.py`.
    - Given by: organiser.
11. **The flights CSV's weather is used only as `oracle_*`.**
    - Why: the organisers say it is attached at flight time; in fact that join reads the local clock hour as UTC, so it is ~4–6 h before the event (CLAUDE.md). It is never a feature.
    - Where: `src/data/events.py:18,63-64`; `tests/test_leakage.py` (`test_forbidden_columns`).
    - Given by: organiser.
12. **Units are kept as provided:** °C, mm, km/h, hPa; wind direction FROM, degrees true; headwind/crosswind/tailwind as computed by the organisers.
    - Where: `src/data/hourly_weather.py:13-15`.
    - Given by: organiser (data guide).
13. **Thunderstorm flag = coco ∈ {25, 26, 27}; snow flag = coco ∈ {14, 15, 16}.**
    - Where: `config/airports.yaml:40-41`; `src/data/hourly_weather.py:38-39`.
    - Given by: organiser groups (data guide `condition_code`).
    - Impact: other Meteostat winter codes (10–13, 19–22) are not flagged as snow.
14. **`wpgt_*`, `snow_*` (flights CSV) and `wind_gust_kmh` (hourly file, 100% empty) are not used.**
    - Given by: organiser (placeholders); ours (empty gust column).
15. **Lags (1/3/6 h), 3 h pressure tendency and 6 h precipitation sum are computed backwards from the used row**, so they are known at the same availability time.
    - Where: `src/data/hourly_weather.py:40-46`.
    - Given by: ours (lag set from the organiser spec).
16. **`other_*` weather** is filled only when the hourly file covers `other_airport` (ATL↔DFW, 2.4% of rows).
    - Given by: organiser (optional field); ours (no other source).
17. **`other_airport` ICAO:** config overrides, else P+IATA for AK/HI/GU, else K+IATA (overrides verified against the Meteostat station list).
    - Where: `src/data/config.py:34-42`; `config/airports.yaml:19-27`.
    - Given by: ours. Label only; no weather depends on it except ATL/DFW.

## 🤖 Models

18. **P2 arrivals model (optional AAR) uses full BTS data for its target**: realised arrivals per UTC hour by wheels-on time.
    - Where: `src/models/aar_bts.py:29-33` (download), `src/models/aar_dataset.py` (panel); SOURCES.md #2.
    - Given by: organiser allows it ("may pull the full BTS on-time data automatically"); the choice is ours (P2).
    - Weather timing: same rule as P1/P3. P2 joins the row stamped `t − h − 1 h` (lag from `config/airports.yaml`
      `weather.availability_lag_hours`), so no weather feature is observed after t − h (`src/models/aar_dataset.py`
      `build_panel`; test `test_every_weather_stamp_is_available_by_forecast_time`). Test RMSE KATL 6.86, KDFW 8.96.
19. **P3 delay model split:** train before 2024-07-01, validation 2024-07..12, test 2025-01..07. One model for all horizons (`h` is a feature).
    - Where: `src/models/delay/train.py` (`split_of`, `TRAIN_END`, `VAL_END`).
    - Given by: ours.
20. **P3 alert thresholds are per airport, maximum F1 on validation:** KATL 0.098, KDFW 0.123.
    - Where: `src/models/delay/artifacts/meta.json` (`thresholds_by_airport`); `outputs/metrics/threshold.md`.
    - Given by: ours.
    - Impact: treats a missed delay and a false alert as equally costly.
21. **P3 uses no BTS or hub-state features.** It uses only P1's `event_features` (organiser files).
    - Given by: ours (P3).
    - Impact: the earlier BTS hub features gave about +0.04 PR-AUC at 3 h; a possible next step.
22. **P3 weather-rule baseline:** `ts_flag` OR crosswind ≥ c OR tailwind ≥ t, with c and t = max F1 on train (currently 7.5 and 10.0 km/h). No climatology baseline.
    - Where: `src/models/delay/evaluate.py` (`rule_fit`); `outputs/metrics/threshold.md`.
    - Given by: ours.
23. **Year-to-year shift:** the positive rate is 10.1% in train, 9.6% in validation and 13.2% in test (2025); cancellations rise from 1.1% to 2.1%.
    - Given by: data (P1 check).
    - Impact: probabilities may under-predict in 2025 (see `outputs/metrics/calibration.png`).

## 🧠 AI use

24. **No LLM at runtime.** Risk scores come from LightGBM, recommendations from explicit rules, and per-flight reasons from LightGBM
    feature contributions (`src/models/delay/train.py` `top_drivers`; dashboard rules in `app/app.py` `recommendations`).
    - **Development:** code and docs were written with AI coding assistance (Claude Code), visible as `Co-Authored-By` lines
      in the git history. Explanatory text (README, failure-case write-ups, dashboard notes) was AI-drafted and reviewed by the team.
    - Given by: ours. Rubric: "Any LLM use is explanation-only and labelled."
