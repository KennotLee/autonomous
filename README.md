# AAR & Delay-Risk Hackathon (KATL, KDFW)

Goal: build features and models that predict, for KATL and KDFW over 2023-2025:
(a) hourly runway arrival capacity (proxy: realised arrivals per UTC hour), and
(b) per-flight risk of a >45 min arrival delay, 3-12 h ahead.
Horizons h are in {3, 6, 9, 12} h. Only features observed at or before target time minus h may be used (leakage-safe).
Schemas and the team contract live in CLAUDE.md.

## Setup

```
git clone https://github.com/KennotLee/autonomous.git
cd autonomous && git checkout p1
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The raw CSV is NOT in git. Put `Flight_Data_With_Weather_Final.csv` into `data/raw/` yourself (ask P1).
`data/sample/sample_1000.csv` is a 1,000-row sample for quick starts.

## Run

```
.venv/bin/python scripts/run_pipeline.py --icao KATL KDFW --start 2023-01-01 --end 2025-07-31 --weather-scope all
.venv/bin/python tests/test_leakage.py      # must print ALL LEAKAGE CHECKS PASSED
```
(`--weather-scope all` fetches Meteostat for ~208 airports, ~9 min first time; `hubs` is fast but leaves non-hub weather NaN.)

Outputs (already committed on branch `p1`, no need to rebuild): column-by-column schemas in CLAUDE.md.
- `data/processed/hourly_features.parquet` — one row per icao × UTC hour × h
- `data/processed/flight_features.parquet` — one row per flight × h

## Known data limitations (read before modelling) — P2/P4

1. **Hourly arrivals are very sparse.** The CSV is a thin sample of BTS flights: ~5.7 ATL and ~4.2 DFW
   arrivals/day (real traffic is ~1,000+/day). `arr_count` is 0 in 82% of hours, max 4. It is NOT a
   realistic runway capacity (AAR); treat the hourly task as count/rare-event modelling and say so in the demo.
2. **The CSV's own weather columns are misaligned by 4–6 h** (upstream joined UTC weather to local clock
   hours). We do NOT use them; all weather comes from Meteostat hourly (UTC), re-fetched per airport.
3. **Gusts are 97–99% missing** (`wpgt_kt`, `gust_*` components, `wpgt_max_*`). Snow depth is 100% missing (dropped).
   `coco` (weather condition) is model-filled by Meteostat, not observed.
4. **Wind components exist only at KATL/KDFW** (organiser runway headings 080°/180°, used as given; no
   magnetic correction). Other airports have weather but NaN components.
5. **No weather for 2 airports:** LAW (no Meteostat station) and YUM (station data ends 2007).
   Weather coverage in the flight file: origin 99.95%, dest 99.7%.
6. **Cancelled (345) and diverted (78) hub flights are kept** in the flight file with `delay_min`/`label_45`
   = NaN. Drop them (`fl[fl.label_45.notna()]`) or decide a policy. In the hourly file they count in
   `sched_arr_count`, `cancellations`/`diversions`, but not in `arr_count`.
7. **Class imbalance:** `label_45` positive rate is 10.7%.
8. **Never use as features:** `oracle_*` (same-hour weather), `arr_count`, `mean_arr_delay`, `share_delayed_45`,
   `cancellations`, `diversions`, `capacity_ratio`, `delay_min`, `label_45`, `cancelled`, `diverted`.
   `sched_arr_count` (schedule) is known in advance and may be used. Use one `h` per model.
9. **Data window:** 2023-01-01 to 2025-07-31 (2025 is partial). Weather at t − h uses the last obs hour
   ≤ t − h, tolerance 2 h.

## Layout

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
│   ├── data/                  P1: data and features
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

## Team workflow

- Everyone commits to `main`, but only inside their own folder:
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
