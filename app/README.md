# P4 dashboard: delay-risk board

```
.venv/bin/streamlit run app/app.py          # Windows: .venv\Scripts\streamlit run app/app.py  → http://localhost:8501
```

**User:** the airline duty manager at the ATL/DFW hub. The model ranks flights; the human decides and logs the decision.

| tab | what it shows | judging criterion |
|---|---|---|
| 🛫 Risk board | Flights for the chosen hub, day and horizon h. Each row shows: risk P(delay > 45 min); alert vs the per-airport threshold (tuned on 2024-H2 validation); how often flights scored like this were late in 2025; the weather-rule (B3) risk alongside; the model's top drivers; the exact weather observation used, with UTC and local timestamp; a decision and note. "Log my decisions" appends to `app/decision_log.csv`. Replay mode reveals outcomes. | problem realism, trust & traceability, human decides |
| 📈 How good is it? | PR-AUC of B0–B3 vs the model for this h; significance from the day-block CI; table by airport, direction and adverse weather; calibration plot | honest evaluation |
| ⚠️ Failure cases | Missed storm DFW 2025-04-23 and false alarm DFW 2025-05-30, with one-click jumps. In replay mode, today's missed flights | honest evaluation |
| 🔒 Data & leakage | Live check that every weather obs is ≤ t − h; the no-leakage rules; sources and Meteostat credit | technical soundness, trust |
| 🛬 Arrivals/hour | P2's optional AAR-proxy forecast with P10–P90 band and the schedule baseline | impact (bonus) |
| 📝 Decision log | Every logged decision, with user, time, risk, threshold, weather obs time and model version | trust & traceability |

**Airport codes:** any code typed as IATA or ICAO is checked against `config/airports.yaml`. If the code isn't there, or has no events yet, the app says what's missing instead of failing.

**Inputs** (all committed team outputs):
- `data/processed/event_features.parquet` (P1)
- `outputs/predictions/delay_preds.csv` (P3) and `aar_preds.csv` (P2)
- `outputs/predictions/event_baseline_preds.csv` and `outputs/metrics/event_*.csv`, `failure_cases_p3_final.md` (P4)
- `src/models/delay/artifacts/meta.json` (P3 thresholds)
