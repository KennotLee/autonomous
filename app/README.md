# P4 dashboard: delay-risk board

```
.venv/bin/streamlit run app/app.py          # Windows: .venv\Scripts\streamlit run app/app.py  → http://localhost:8501
```

**Users:** the ATC traffic management coordinator at ATL/DFW (default view) and the airline duty manager (second view). The model ranks flights; explicit rules make recommendations; the human decides and logs it. No LLM.

| tab | what it shows | rubric criterion / brief item |
|---|---|---|
| 🛫 Risk board & recommendations | **Recommendations:** 3-hour windows flagged by explicit rules (mean risk ≥ 1.5× the 2023–24 rate, or ≥ 2 alerts). Each shows the rule that fired, the weather context and a role-specific action. **Flights table**, one row per flight: risk; alert; "flights scored like this were late X% (90% CI)"; weather-rule risk; cautions (rare condition, model/rule disagreement, ⛔ no forecast when no weather ≤ t − h); drivers; the weather obs used with its timestamp and head/cross/tailwind; decision and note, logged with name and role. **Replay mode** reveals outcomes | brief: risk scores + recommendations; trust & traceability; human decides |
| 📊 Historical trends | Monthly late *rates* for both hubs with the train/test line; late rate by hour (adverse vs normal at t − 3h); late rate by weather trigger; 2025 weekly predicted vs actual | brief: historical trends; un-normalised delays avoided |
| 📈 How good is it? | PR-AUC of B0–B3 vs the model, with a significance verdict; precision and recall at the alert threshold vs the rule on the same alert budget; all slices; calibration; cross-airport transfer; impact box separating measured from reasoned | honest evaluation, impact |
| ⚠️ Failure cases | Missed storm and false alarm with one-click jumps: why each fails, the fix, and how the user would know | honest evaluation |
| 🔒 Data, leakage & pipeline | Live check that obs ≤ t − h; never-used columns; runway directionality; LLM/non-LLM split; the commands that download and process everything | technical soundness, automatic processing |
| 🛬 Arrivals/hour (bonus) | P2's AAR-proxy forecast with P10–P90 band, RMSE table, and the P4 independent check when available | optional AAR |
| 📝 Decision log | Every logged decision | trust & traceability |

**Airport codes:** any code typed as IATA or ICAO is checked against `config/airports.yaml`. If the code isn't there, or has no events yet, the app says what's missing instead of failing.

**Inputs** (all committed team outputs):
- `data/processed/event_features.parquet` (P1)
- `outputs/predictions/delay_preds.csv` (P3) and `aar_preds.csv` (P2)
- `outputs/predictions/event_baseline_preds.csv` and `outputs/metrics/event_*.csv`, `failure_cases_p3_final.md` (P4)
- `src/models/delay/artifacts/meta.json` (P3 thresholds)
