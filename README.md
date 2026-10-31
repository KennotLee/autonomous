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
| 🔍 **Evaluation** (P4) | Independent baselines and checks of both models (incl. bad-weather hours). | `src/eval/` (see `src/eval/README.md`) | `outputs/metrics/event_*`, `aar_bts_model_vs_baseline.csv` |
| 🖥️ **Dashboard** (P4) | Shows each flight's risk, why, how sure we are, "Consider …" recommendations, and logs your decisions. | `streamlit run app/app.py` | browser at `localhost:8501` |

## 🖥️ Dashboard: quick start (no data download or pipeline run needed)

All inputs the dashboard reads are committed, so it runs straight after install.

**1. Install (once, from the repo root)**
```
git clone https://github.com/KennotLee/autonomous.git && cd autonomous
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt                 # macOS / Linux
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt                 # Windows (PowerShell)
```
**2. Start it**
```
.venv/bin/streamlit run app/app.py            # macOS / Linux
.venv\Scripts\streamlit run app/app.py        # Windows
```
**3. Open http://localhost:8501** (first load ~10 s). Stop with **Ctrl+C**.

**4. Use it**
1. **Pick in the sidebar:** hub (from `config/airports.yaml`), how far ahead to warn (3/6/9/12 h), and a day in Jan–Jul 2025 (test period). Or click a **demo day**: a good day, a missed storm, or a false alarm.
2. **Read the 🛫 Risk board:** the trust line (model skill at this horizon), alerts vs normal late rate, yellow **"Consider …" recommendations** (each with the rule that fired, the weather and its observation time), then the flight table, riskiest first, with risk, alert 🔴, the uncertainty range, the reasons ("why") and the weather used.
3. **Decide:** type your name, pick **Act / Watch / Override** in the first column, and click **✅ Log my decisions**. Decisions are saved to `app/decision_log.csv` (not in git). Nothing happens automatically.
4. **Check (optional):** turn on **Replay** to see what actually happened; **Settings → Alert at risk ≥** changes the threshold (default tuned per airport on 2024 validation data).

| Tab | What it's for |
|---|---|
| 🛫 Risk board | the forecast, recommendations and decisions |
| 📈 Performance & failures | model vs baselines, precision/recall at the threshold, calibration, failure days and how you'd notice |
| 🛬 Arrivals/hour (bonus) | P2's hourly arrivals forecast with a P10–P90 range |
| 🔒 How it works | live no-leakage check, inputs, assumptions, AI use, `SOURCES.md` / `ASSUMPTIONS.md`, rebuild commands |

Full walkthrough: `app/README.md`.

## ⚙️ Rebuild everything from the raw files (optional)

Put the organiser files (`Flight_Data_With_Weather_Final.csv`, `challenge5_hourly_weather_ATL_DFW.csv`,
`challenge5_data_guide.xlsx`) in `data/raw/` (not in git). On Windows, replace `.venv/bin/` with `.venv\Scripts\`.
```
.venv/bin/python scripts/run_pipeline.py --airports ATL DFW     # features (~2 s)
.venv/bin/python tests/test_leakage.py                          # must print ALL LEAKAGE CHECKS PASSED
.venv/bin/python -m src.models.delay.train                      # delay model + predictions
.venv/bin/python -m src.models.delay.evaluate                   # metrics, calibration, oracle gap, failure cases
# optional AAR: downloads ~31 monthly BTS files first (~1 GB)
.venv/bin/python -m src.models.aar_bts && .venv/bin/python -m src.models.aar_dataset && .venv/bin/python -m src.models.train_aar
```
All code uses `pathlib` and reads text as UTF-8, so it runs on macOS, Linux and Windows.

## 📈 Results (2025 test)

- **Delay model:** PR-AUC 0.219 (95% CI 0.19–0.25) vs base rate 0.132 (~1.7× better than guessing); alert thresholds per airport (KATL 0.098, KDFW 0.123).
- **Arrivals model (optional):** RMSE KATL 6.86, KDFW 8.96 arrivals/hour. Only **3.5% / 7.0% better than the strongest
  (schedule-spread) baseline** at h = 3 (1–7% across horizons), but **4–13% in bad-weather and 8–27% in severe-weather hours**,
  where it matters (`outputs/metrics/aar_bts_model_vs_baseline.csv`). Vs simpler baselines: 20.9% / 16.1%.

## ⚠️ Limitations

1. **Weather explains only a small part of big delays.** Most misses happen in calm weather (late aircraft, airline operations), which we can't see.
2. **2025 had more delays** than 2023–24 (13.2% vs ~10%), so risk scores may run low.
3. **Weather is only available for ATL and DFW**, so arrivals can't see the weather where the flight came from.
4. **The flights file is a ~0.5% sample:** fine for per-flight labels, but hourly arrival counts need the full BTS data (P2).
5. **No gust or snow-depth data** (empty or placeholder in the organiser files). The flights file's own weather is ~4–6 h misaligned, so it's only used as `oracle_*` for comparison.
6. **Cancelled flights aren't scored** (organiser rule: no label). Compare airports by delay *rates*, not minutes.
7. **Dashboard demo days are hand-picked ATL/DFW dates;** everything else (hubs, defaults, thresholds) comes from config.

## 📚 Sources & assumptions

`SOURCES.md`: every data source with links and licences (credit **Meteostat**, CC BY 4.0, in the demo).
`ASSUMPTIONS.md`: every assumption with file:line. Both are also shown in the dashboard's 🔒 How it works tab.

**AI use:** no LLM at runtime (risk = LightGBM, recommendations = explicit rules, reasons = LightGBM feature
contributions). Built with AI coding assistance (Claude Code); explanatory text was AI-drafted and reviewed by the team.

## 🗂️ Layout

```
config/airports.yaml        airports (IATA/ICAO, timezone, runway heading), weather settings
data/raw/                   organiser files (never committed)
data/processed/             event_features.parquet + 5-row sample (P1)
src/data/                   P1 feature pipeline (+ optional fetch_weather.py)
src/models/delay/           P3 delay model (+ saved model in artifacts/)
src/models/aar_*.py         P2 optional arrivals model
src/eval/                   P4 baselines and independent checks
outputs/predictions/        delay_preds.csv (P3), aar_preds.csv (P2), event_baseline_preds.csv (P4), SCHEMA.md
outputs/metrics/            metrics, plots, failure cases (P3, P4), pitch.md
app/                        P4 dashboard (app.py, README.md; decision_log.csv is local only)
scripts/run_pipeline.py     feature pipeline entry point
tests/test_leakage.py       leakage checks
```

## 🤝 Team workflow

- Each person pushes to their own branch (`p1`…`p4`); the integrator merges into `main`.
- Shared files (`CLAUDE.md`, `README.md`, `requirements.txt`, `config/`): P1 edits, others ask.
- `git pull --rebase` before pushing; no files > 50 MB, no `data/raw/`, no secrets.
