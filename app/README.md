# Weather delay-risk dashboard (P4)

For each departure from and arrival at ATL / DFW, the board shows the model's risk of a **delay over 45 minutes**, forecast 3–12 h ahead from weather already observed. It also shows the evidence behind each risk, and lets the user (an ATC traffic management coordinator, with airline ops) **decide and log** what to do.

## Run it

**1. One-time setup** (from the repo root):

```bash
# macOS / Linux
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

**2. Start the dashboard:**

```bash
.venv/bin/streamlit run app/app.py          # macOS / Linux
.venv\Scripts\streamlit run app/app.py      # Windows
```

**3. Open http://localhost:8501.** The first load takes about 10 s while it reads the data. Stop it with **Ctrl+C**.

Everything it reads is committed in the repo, so no download or pipeline run is needed:
- `data/processed/event_features.parquet` (P1)
- `outputs/predictions/delay_preds.csv` (P3), `aar_preds.csv` (P2) and `event_baseline_preds.csv` (P4)
- `outputs/metrics/event_*.csv`
- `src/models/delay/artifacts/meta.json`
- `config/airports.yaml`, `SOURCES.md`, `ASSUMPTIONS.md`

To rebuild everything from the raw files, see the commands in the app's **How it works** tab or the root `README.md`.

## Use it (step by step)

1. **Choose what to look at** in the left sidebar:
   - **Hub:** any airport in `config/airports.yaml` that has data. To check another code, use **Settings → Other airport code**; an unconfigured code such as ORD shows a message saying what to add.
   - **Warn how far ahead:** 3, 6, 9 or 12 h. The model only uses weather observed at least that long before each flight.
   - **Day:** any day from January to July 2025 (the test period, never used for training).
   - **Demo days:** one click to a hand-picked good day, a missed storm or a false alarm.
2. **Read the 🛫 Risk board,** top to bottom:
   - **How much to trust this:** the model's tested skill at this horizon vs a simple weather rule. At 3 h it is a tie, so use the risk to *rank* flights, not as a certainty.
   - **Three numbers:** flights in the sample, alerts, and expected late rate vs normal.
   - **Recommendation lines** (yellow) for risky 3-hour windows. Each shows the rule that fired, the weather behind it and its observation time, and a *Consider …* action.
   - **Flight table,** riskiest first. Columns:

     | column | shows |
     |---|---|
     | **decision** | first column, where you choose |
     | time, flight | when, and dep/arr, carrier and route |
     | **risk** | the model's probability of a delay over 45 min |
     | alert 🔴 | risk ≥ threshold |
     | **similar flights late** | how often 2025 flights with this risk were really late (90% range): the uncertainty |
     | **why** | the model's top factors in plain words |
     | **weather used** | the observation and its timestamp |
     | caution | only shown when something is off: ⛔ no forecast, or rare weather |
     | note | your free-text note |

3. **Decide:**
   - Type **your name** in the sidebar.
   - In the **decision** column (first) pick **Act**, **Watch** or **Override: no action** for any flights, and optionally add a note.
   - Click **✅ Log my decisions** (above the table).
   - Each decision is saved to `app/decision_log.csv` (gitignored) with your name, risk, threshold, weather time and model version. See **Decision log** under the table. Nothing happens automatically.
4. **Check it (optional):**
   - Turn on **Replay: show what actually happened** to add the real outcomes and the actual late rate.
   - **Settings → Alert at risk ≥** changes the alert threshold. Its default is tuned per airport on 2024 validation data.

## Tabs

| tab | what it's for |
|---|---|
| 🛫 Risk board | the forecast, recommendations, decisions (above) |
| 📈 Performance & failures | model vs baselines (PR-AUC, significance), precision/recall at the alert threshold, calibration, monthly late-rate history, and two failure days with why and how you'd notice |
| 🛬 Arrivals/hour (bonus) | P2's hourly arrivals forecast with a P10–P90 range (full BTS data) |
| 🔒 How it works | live no-leakage check, inputs and assumptions, AI use, `SOURCES.md` / `ASSUMPTIONS.md`, rebuild commands |

**AI use:** no LLM at runtime. Risk comes from LightGBM, recommendations from explicit rules, and reasons from LightGBM feature contributions. Built with AI coding assistance (Claude Code); explanatory text was AI-drafted and reviewed by the team.

**Data:** organiser files (BTS on-time sample; Meteostat / NOAA ISD hourly weather, CC BY 4.0) and US DOT BTS On-Time Performance (full, for arrivals per hour).
