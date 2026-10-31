# ✈️ AAR & Delay-Risk Hackathon (KATL, KDFW)

Goal: build features and models that predict, for KATL and KDFW over 2023-2025:
Main target (organiser update): per ATL/DFW event, will it be severely delayed (> 45 min)? Departures from the hub use DEP_DELAY, arrivals at the hub use ARR_DELAY (~10% positive). Hourly arrival rate (AAR) is optional per the organisers and is no longer built: the flights file is a ~0.5% sample, so hourly counts can't be measured.
Horizons h are in {3, 6, 9, 12} h. Only features observed at or before target time minus h may be used (leakage-safe: weather observed at least 3 h before the scheduled time).
Schemas and the team contract live in CLAUDE.md.

## 🛠️ What P1 (Data & Features) does

**In one line:** P1 turns the raw flight CSV into one clean, leakage-safe feature table that the model trains on.

What `scripts/run_pipeline.py` does, step by step:

1. 🔎 **Events:** every flight leaving ATL/DFW becomes a `dep` event, every flight landing there an `arr` event (ATL↔DFW flights give both). Airports can be passed as IATA or ICAO (`config/airports.yaml`).
2. 🕒 **Clocks:** local hhmm → UTC with each hub's time zone; flights in the DST-ambiguous hour are dropped (none in our data).
3. 🌦️ **Weather:** ONLY the organiser hourly file (Meteostat/NOAA ISD, stations 72219/72259). A row stamped `time_utc` is treated as available at `time_utc + 1 h` because the stamp can precede the METAR it holds by up to ~52 min.
4. ⏪ **No peeking:** for each event and h ∈ {3,6,9,12}, features use only rows available at or before `sched_time_utc − h`. Features: temp, prcp, wspd, pres, coco, wdir, headwind, crosswind, tailwind (km/h, provided), thunderstorm/snow flags, 1/3/6 h lags, 3 h pressure tendency, 6 h precipitation sum; `other_*` = same at the other airport (only for ATL↔DFW).
5. 🎯 **Target:** `delay_min` (DEP_DELAY or ARR_DELAY by event type), `label_45` = delay_min > 45; cancelled → NaN.
6. ✅ **Leakage test:** `tests/test_leakage.py`.

| 📦 P1 output | One row per | Used for |
|---|---|---|
| `data/processed/event_features.parquet` | event × h | delay > 45 min model |

A 5-row sample is at `data/processed/event_features_sample5.csv`.

Also: `src/data/fetch_weather.py` is an optional standalone tool, not used by the pipeline, that downloads raw airport weather reports (METAR/ASOS) for any ICAO code and date range.

## ⚙️ Setup

```
git clone https://github.com/KennotLee/autonomous.git
cd autonomous
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The raw files are NOT in git; put `Flight_Data_With_Weather_Final.csv`, `challenge5_hourly_weather_ATL_DFW.csv` and `challenge5_data_guide.xlsx` (organiser-provided) into `data/raw/`.
`data/sample/sample_1000.csv` is a 1,000-row sample for quick starts.

## ▶️ Run

```
.venv/bin/python scripts/run_pipeline.py --airports ATL DFW
.venv/bin/python tests/test_leakage.py      # must print ALL LEAKAGE CHECKS PASSED
```

Outputs: `data/processed/event_features.parquet`, schema in CLAUDE.md.

## ⚠️ Known data limitations (read before modelling) — P2/P4

1. **Flights file is a ~0.5% sample** (~6 ATL arrivals/day) → no hourly AAR.
2. **The flights CSV's own weather is attached at flight time** → used only as `oracle_*` columns (never features).
3. **`wpgt_*` and `snow_*` are placeholders** → ignored. The hourly file's `wind_gust_kmh` is empty → no gust features.
4. **Weather exists only for ATL and DFW** → `other_*` features are NaN except for ATL↔DFW flights.
5. **Cancelled events have `label_45` = NaN** (diverted arrivals too, no ARR_DELAY); drop them or set a policy.
6. **Never use as features:** `oracle_*`, `delay_min`, `label_45`, `cancelled`, `diverted`, `feature_obs_time`, `sched_time_*`, ids.
7. **Data window:** 2023-01-01 to 2025-07-31.
8. **Compare airports by delay rates** (share of events), not raw minutes (organiser rule).

## 📚 Sources & assumptions

See SOURCES.md (every data source, links, licences) and ASSUMPTIONS.md (every assumption with file:line).

## 🗂️ Layout

```
aar_hackathon/
├── CLAUDE.md                  team contract: schemas and rules
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── airports.yaml          airport list (P1)
├── data/
│   ├── raw/                   raw CSV, never committed
│   ├── sample/                sample_1000.csv, committed
│   ├── interim/               intermediate files, not committed
│   └── processed/             parquet outputs (P1)
├── src/
│   ├── data/                  P1: data and features (+ fetch_weather.py)
│   ├── models/                P2: models
│   └── eval/                  P4: evaluation
├── app/                       P3: app/dashboard
├── outputs/
│   ├── predictions/           P2: prediction CSVs
│   └── metrics/               P4: metrics
├── scripts/
│   └── run_pipeline.py        pipeline entry point (P1)
└── tests/
    └── test_leakage.py        leakage checks
```

## 🤝 Team workflow

- Each person pushes to their own branch (p1, p2, …); the integrator merges into `main`. Work only inside your own folder:
  - P1: `src/data`, `config`, `data/processed`
  - P2: `src/models`, `outputs/predictions`
  - P3: `app/`
  - P4: `src/eval`, `outputs/metrics`
- Shared files (`CLAUDE.md`, `requirements.txt`, `README.md`): P1 edits; others ask P1.
- Always `git pull --rebase` before `git push`. Small, frequent commits with clear messages (e.g. "p1: add wind components").
- Never commit files > 50 MB, `data/raw/`, or secrets/API keys.
- Checkpoints (SGT):
  - 13:00 features v1 pushed
  - ~14:00 v1.1 with crosswind
  - 14:30 predictions pushed
  - 15:10 freeze
