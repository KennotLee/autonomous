# AAR & Delay-Risk Hackathon (KATL, KDFW)

Goal: build features and models that predict, for KATL and KDFW over 2023-2025:
(a) hourly runway arrival capacity (proxy: realised arrivals per UTC hour), and
(b) per-flight risk of a >45 min arrival delay, 3-12 h ahead.
Horizons h are in {3, 6, 9, 12} h. Only features observed at or before target time minus h may be used (leakage-safe).
Schemas and the team contract live in CLAUDE.md.

## Setup

```
git clone <REPO_URL>
cd aar_hackathon
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The raw CSV is NOT in git. Put `Flight_Data_With_Weather_Final.csv` into `data/raw/` yourself (ask P1).
`data/sample/sample_1000.csv` is a 1,000-row sample for quick starts.

## Run (being built by P1)

```
python scripts/run_pipeline.py --icao KATL KDFW --start 2023-01-01 --end 2025-07-31
```

Outputs:
- `data/processed/hourly_features.parquet`
- `data/processed/flight_features.parquet`

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
