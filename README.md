# ✈️ Delay-Risk Hackathon: Challenge #5 (KATL, KDFW)

**Goal:** for every flight leaving or landing at ATL/DFW, predict 3, 6, 9 or 12 hours ahead whether it will be
more than 45 min late (departures: `DEP_DELAY`, arrivals: `ARR_DELAY`; ~10% are). Optional extra: arrivals per hour (AAR).
**Rule:** only weather observed at least 3 h before the scheduled time. Schemas and team contract: `CLAUDE.md`.

## 🧩 What's inside (ELI5)

| Part | In one line | Run | Output |
|---|---|---|---|
| 🛠️ **Features** (P1) | Turns every ATL/DFW flight into a row with the weather we *could* have known h hours before. | `scripts/run_pipeline.py` | `data/processed/event_features.parquet` |
| ✅ **Leakage test** (P1) | Checks every row and fails loudly if any feature peeks into the future. | `tests/test_leakage.py` | pass/fail |
| 🎯 **Delay model** (P3) | Learns from 2023–24 which situations lead to big delays and gives each 2025 flight a risk score. | `src.models.delay.train` / `.evaluate` | `outputs/predictions/delay_preds.csv`, `outputs/metrics/` |
| 🛬 **Arrivals model** (P2, optional) | Predicts how many planes land per hour, using the full public BTS flight data. | `src.models.aar_bts` → `.aar_dataset` → `.train_aar` | `outputs/predictions/aar_preds.csv` |
| 📊 **Dashboard** (P4) | Shows the risk scores and why (top drivers) per airport and hour. | `app/` (in progress) | — |

## ⚙️ Setup

```
git clone https://github.com/KennotLee/autonomous.git && cd autonomous
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Put the organiser files (`Flight_Data_With_Weather_Final.csv`, `challenge5_hourly_weather_ATL_DFW.csv`,
`challenge5_data_guide.xlsx`) in `data/raw/` (not in git).

## ▶️ Run

```
.venv/bin/python scripts/run_pipeline.py --airports ATL DFW     # features (~2 s)
.venv/bin/python tests/test_leakage.py                          # must print ALL LEAKAGE CHECKS PASSED
.venv/bin/python -m src.models.delay.train                      # delay model + predictions
.venv/bin/python -m src.models.delay.evaluate                   # metrics, calibration, oracle gap, failure cases
# optional AAR: downloads ~31 monthly BTS files first
.venv/bin/python -m src.models.aar_bts && .venv/bin/python -m src.models.aar_dataset && .venv/bin/python -m src.models.train_aar
```

## 📈 Results (2025 test)

- **Delay model:** PR-AUC 0.219 vs base rate 0.132 (~1.7× better than guessing); alert thresholds per airport (KATL 0.098, KDFW 0.123).
- **Arrivals model (P2-reported):** RMSE KATL 6.82, KDFW 8.89 arrivals/hour, 21% and 17% better than the best baseline.

## ⚠️ Limitations

1. **Weather explains only a small part of big delays.** Most misses happen in calm weather (late aircraft, airline operations), which we can't see.
2. **2025 had more delays** than 2023–24 (13.2% vs ~10%), so risk scores may run low.
3. **Weather is only available for ATL and DFW**, so arrivals can't see the weather where the flight came from.
4. **The flights file is a ~0.5% sample:** fine for per-flight labels, but hourly arrival counts need the full BTS data (P2).
5. **No gust or snow-depth data** (empty or placeholder in the organiser files). The flights file's own weather is ~4–6 h misaligned, so it's only used as `oracle_*` for comparison.
6. **Cancelled flights aren't scored** (organiser rule: no label). Compare airports by delay *rates*, not minutes.

## 📚 Sources & assumptions

`SOURCES.md`: every data source with links and licences (credit **Meteostat**, CC BY 4.0, in the demo).
`ASSUMPTIONS.md`: every assumption with file:line.

## 🗂️ Layout

```
config/airports.yaml        airports (IATA/ICAO, timezone, runway heading), weather settings
data/raw/                   organiser files (never committed)
data/processed/             event_features.parquet + 5-row sample (P1)
src/data/                   P1 feature pipeline (+ optional fetch_weather.py)
src/models/delay/           P3 delay model (+ saved model in artifacts/)
src/models/aar_*.py         P2 optional arrivals model
outputs/predictions/        delay_preds.csv (P3), aar_preds.csv (P2), SCHEMA.md
outputs/metrics/            P3 metrics, plots, failure cases
app/                        P4 dashboard
scripts/run_pipeline.py     feature pipeline entry point
tests/test_leakage.py       leakage checks
```

## 🤝 Team workflow

- Each person pushes to their own branch (`p1`…`p4`); the integrator merges into `main`.
- Shared files (`CLAUDE.md`, `README.md`, `requirements.txt`, `config/`): P1 edits, others ask.
- `git pull --rebase` before pushing; no files > 50 MB, no `data/raw/`, no secrets.
