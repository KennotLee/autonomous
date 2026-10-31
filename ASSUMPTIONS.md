# 🧩 Assumptions

Every assumption the pipeline and models rely on. **Given by** = *organiser* (from the challenge brief, recorded
in CLAUDE.md lines 3–5) or *ours* (team decision). "Impact if wrong" is our estimate, not a measured result,
unless a number is cited.

## ✈️ Problem framing

1. **Runway headings: KATL 080°, KDFW 180°; all arrivals use that heading.**
   - Why: needed to turn wind into headwind, crosswind and tailwind.
   - Where: `config/airports.yaml:13,19`; CLAUDE.md:5.
   - Given by: organiser. Used as given, with no magnetic-to-true correction (`config/airports.yaml:2`; CLAUDE.md decision 11).
   - Impact if wrong: wind components are off. A ~5° magnetic/true difference changes crosswind by at most ~9% of wind speed.
2. **Arrival capacity (AAR) proxy = realised arrivals per UTC hour.**
   - Why: true runway capacity is not in the data.
   - Where: CLAUDE.md:3; `src/data/build.py:67-70`; `src/models/aar_config.py:82`.
   - Given by: organiser (proxy). **Ours:** count by touchdown time (`wheels_on_utc`), excluding cancelled/diverted flights (CLAUDE.md decision 4).
   - Impact if wrong: the target measures demand, not capacity. With ~5 sampled arrivals per hub per day the runway is never full, so weather effects on capacity can't be learned (README "Known data limitations" #1).
3. **Forecast horizons h ∈ {3, 6, 9, 12} h.**
   - Where: CLAUDE.md:4; `config/airports.yaml:40`.
   - Given by: organiser.
4. **Severe delay = arrival delay strictly greater than 45 min.**
   - Where: CLAUDE.md:4; `src/data/flights.py:68`.
   - Given by: organiser (threshold). **Ours:** "strictly greater", and `ARR_DELAY` (gate arrival) as the delay measure (CLAUDE.md decision 5).
   - Impact if wrong: labels flip only for flights exactly 45 min late.

## 🕒 Time handling

5. **BTS times are local** (departure in the origin's time zone, arrival in the destination's), **and Meteostat times are UTC.**
   - Why: BTS reports local clock times; Meteostat documents UTC (SOURCES.md #1).
   - Where: `src/data/flights.py:1-8,59`; `src/data/weather.py:1-4`.
   - Given by: ours, from source docs. Verified: converted times match the original local columns on 99.99–100% of rows (`data/processed/build_meta.json`).
   - Impact if wrong: features misaligned by hours (the organiser CSV's own weather has exactly this error).
6. **Times made ambiguous by daylight-saving changes are left blank (NaT), not guessed.**
   - Where: `src/data/flights.py:26`.
   - Given by: ours. Rows affected: 0 (`build_meta.json`, `dst_unresolved_sched_dep`).
7. **Scheduled arrival = scheduled departure + `CRS_ELAPSED_TIME`; touchdown = scheduled arrival + `ARR_DELAY` − `TAXI_IN`.**
   - Why: avoids guessing the date when a flight crosses midnight.
   - Where: `src/data/flights.py:60-62`.
   - Given by: ours. Matches the original hhmm columns on 99.99–100% of rows (`build_meta.json`).
8. **For flights, forecast time t = scheduled departure (UTC).**
   - Where: `src/data/flights.py:59`; CLAUDE.md decision 6.
   - Given by: ours.
   - Impact if wrong: using arrival time would allow up to one flight-duration more recent weather for the destination.
9. **A Meteostat value stamped HH:00 counts as observed at HH:00. A feature is allowed only if its stamp ≤ t − h, using the latest one within a 2 h tolerance.**
   - Why: leakage safety. IEM shows the routine METAR at HH−1:52 equals Meteostat's HH:00 value, so the stamp is at or after the true observation (CLAUDE.md "Layout", `fetch_weather.py` note).
   - Where: `config/airports.yaml:39`; `src/data/build.py:90,117`; enforced by `tests/test_leakage.py`.
   - Given by: ours.

## 🌦️ Weather

10. **Use Meteostat hourly data, not the organiser CSV's weather columns.**
    - Why: the CSV's weather is joined 4–6 h off the true hour (100% match to Meteostat with the wrong join; CLAUDE.md "Weather").
    - Where: `src/data/weather.py`; CLAUDE.md decision 1.
    - Given by: ours.
11. **Wind units: Meteostat km/h → knots (÷ 1.852); wind direction = meteorological "from", true north.**
    - Where: `src/data/config.py:11`; `src/data/weather.py:4`.
    - Given by: ours. Units are from Meteostat docs; the "from" convention is from the underlying METAR source (Meteostat docs do not say).
    - Impact if wrong: if direction meant "to", headwind and tailwind would swap.
12. **Calm wind (speed 0) means all wind components are 0, even with no direction.**
    - Where: `src/data/weather.py:103`.
    - Given by: ours.
13. **Zeros in the CSV's gust (`wpgt`) and snow columns mean "missing".** We drop snow and leave missing gusts blank, with no filling-in.
    - Why: `wpgt` is 0 in 97.8% of rows and snow in 100% (CLAUDE.md).
    - Where: `src/data/weather.py:14` (snow not fetched); CLAUDE.md decision 8.
    - Given by: ours.
    - Impact: gust features are about 98% empty (README limitation #3).
14. **Weather condition code (`coco`) comes from Meteostat's model-filled data; other variables are observations only.**
    - Where: `src/data/weather.py:70-73`; CLAUDE.md decision 2.
    - Given by: ours.
    - Impact: `coco` is a model estimate, not an observation.
15. **Airport codes: ICAO = "K"+IATA, else "P"+IATA, else the overrides in config. LAW has no Meteostat station, so no weather.**
    - Where: `src/data/weather.py:51`; `config/airports.yaml:26-34`.
    - Given by: ours.
16. **Suspect wind readings in Meteostat are kept as-is** (e.g. KATL 2025-03-21 02:00 UTC, 46.7 kt in fair weather).
    - Where: README limitation #9.
    - Given by: ours. Only 9 of ~45k hub-hours have wind ≥ 30 kt.

## 🎯 Targets and outcomes

17. **Cancelled and diverted flights are kept, with blank `delay_min`/`label_45`.** They count in `sched_arr_count`/`cancellations`/`diversions` but not in `arr_count`.
    - Where: `src/data/flights.py:67-68`; `src/data/build.py:67`; CLAUDE.md decisions 4–5.
    - Given by: ours.
    - Impact: 345 cancelled and 78 diverted hub flights are not scored by the delay model.
18. **The schedule is known in advance:** scheduled counts at t (and t−2…t+2) are allowed as features.
    - Where: `src/models/aar_context.py:21`; `src/models/delay/features.py:76` (hub schedule congestion).
    - Given by: ours.
    - Impact if wrong: if the schedule changed late, these features would include some future information.

## 🤖 Models (P2 arrivals model, P3 delay model)

19. **Split by date: train before 2024-07-01, validation 2024-07..12, test 2025-01..07.**
    - Where: `src/models/aar_config.py:28-29`; `src/models/delay/train.py:37-38`.
    - Given by: ours.
    - Impact: test = Jan–Jul 2025 only (no autumn); validation = H2 only.
20. **Arrivals model: Poisson objective, with p10/p90 from quantile models; p10 is capped at the point forecast and all values at ≥ 0.**
    - Where: `src/models/aar_config.py:62,79`; `src/models/train_aar.py:51-52`.
    - Given by: ours (Poisson beat L2 on validation, per the comment at `aar_config.py:62`).
21. **Arrivals model "bad/severe weather" exposure flags:**
    - bad = coco ≥ 7, or precip ≥ 1 mm, or wind ≥ 20 kt, or crosswind ≥ 15 kt
    - severe = coco ≥ 23, or precip ≥ 4 mm, or wind ≥ 25 kt, or a winter code
    - Where: `src/models/aar_context.py:25-27`.
    - Given by: ours.
22. **Delay model: one LightGBM per horizon; alert threshold = maximum F1 on validation** (0.148 in the committed run).
    - Where: `src/models/delay/train.py:91-97`; `outputs/metrics/delay_metrics.md`.
    - Given by: ours.
    - Impact: treats a missed severe delay and a false alert as equally costly.
23. **Delay model hub state uses only flights that had landed or left by t − h** (grouped by actual hour). Cancellation rate is excluded because BTS has no cancellation time.
    - Where: `pipeline/panel.py:112-116`; `src/models/delay/features.py:15-17,35,82`.
    - Given by: ours (the leak fix; regression test `tests/test_pipeline_leakage.py`).
24. **Delay model weather-rule baseline:** high risk if thunderstorm or heavy-precip code, gust > 25 kt, or wind > 20 kt (3 h max) at either end.
    - Where: `src/models/delay/train.py:57-66`; condition codes at `src/models/delay/features.py:36-37`.
    - Given by: ours.
25. **Delay model climatology baseline:** training rate by hub × direction × local arrival hour, shrunk toward the overall rate with weight m = 30.
    - Where: `src/models/delay/train.py:69`.
    - Given by: ours.
26. **Full BTS data (all ATL/DFW flights) is used only for hub-level features, never as labels.**
    - Where: `src/models/delay/features.py`; SOURCES.md #2.
    - Given by: ours.
    - Note: the full model's score (PR-AUC 0.299) requires the BTS download and was not reproduced in integration review (UNVERIFIED). The P1-only variant (0.277) was reproduced.
27. **P3's older pipeline (`pipeline/config.py:35-39`) hardcodes airports, with "approximate" runway headings for KDEN/KORD.**
    - Used by `pipeline/download.py` only, not by either model's features.
    - Given by: ours (P3). Should move to `config/airports.yaml`.
