"""P4 dashboard: delay-risk board for the airline duty manager at an ATL/DFW hub.

    .venv/bin/streamlit run app/app.py        (Windows: .venv\\Scripts\\streamlit run app/app.py)

Reads only committed team outputs:
  data/processed/event_features.parquet       P1 events + weather observed at t - h (and outcomes, shown only in replay)
  outputs/predictions/delay_preds.csv         P3 model risk per (event_id, h)
  outputs/predictions/event_baseline_preds.csv P4 baselines (B3 weather rule shown next to the model)
  outputs/metrics/event_*.csv, failure_cases_*.md, src/models/delay/artifacts/meta.json
  outputs/predictions/aar_preds.csv           P2 optional hourly arrivals (p10-p90)
Writes app/decision_log.csv (human decisions). Nothing in the app acts on its own.
"""
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRED, MET = ROOT / "outputs/predictions", ROOT / "outputs/metrics"
LOG = ROOT / "app/decision_log.csv"
HORIZONS = [3, 6, 9, 12]
ADVERSE_CODES, CROSS_KMH, TAIL_KMH = {5, 6, 7, 8, 9, 25, 26, 27}, 35, 15  # organiser "adverse weather" rule
COCO = {1: "clear", 2: "fair", 3: "cloudy", 4: "overcast", 5: "fog", 6: "freezing fog", 7: "light rain", 8: "rain",
        9: "heavy rain", 10: "freezing rain", 11: "heavy freezing rain", 12: "sleet", 13: "heavy sleet",
        14: "light snow", 15: "snow", 16: "heavy snow", 17: "rain shower", 18: "heavy rain shower",
        19: "sleet shower", 20: "heavy sleet shower", 21: "snow shower", 22: "heavy snow shower", 23: "lightning",
        24: "hail", 25: "thunderstorm", 26: "heavy thunderstorm", 27: "storm"}
DEMO_DAYS = {"Good day (hand-picked): ATL 2025-04-06": ("KATL", date(2025, 4, 6), 3),
             "Failure: missed storm, DFW 2025-04-23": ("KDFW", date(2025, 4, 23), 3),
             "Failure: false alarm, DFW 2025-05-30": ("KDFW", date(2025, 5, 30), 3)}
DECISIONS = ["—", "Act (protect / swap / re-crew)", "Watch", "Override: no action"]

st.set_page_config(page_title="Delay-risk board", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_config():
    with open(ROOT / "config/airports.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_events():
    cols = ["event_id", "event_type", "airport", "other_airport", "sched_time_local", "sched_time_utc", "h",
            "feature_obs_time", "coco", "crosswind", "tailwind", "wspd", "prcp", "temp", "carrier", "delay_min",
            "label_45", "cancelled", "diverted"]
    return pd.read_parquet(ROOT / "data/processed/event_features.parquet", columns=cols)


@st.cache_data
def load_preds():
    p = pd.read_csv(PRED / "delay_preds.csv", usecols=["event_id", "h", "risk_prob", "top_drivers"])
    b = pd.read_csv(PRED / "event_baseline_preds.csv")
    b = b[b["model"] == "B3_B2_x_adverse_wx"][["event_id", "h", "risk_prob", "top_drivers"]]
    return p.merge(b.rename(columns={"risk_prob": "rule_risk", "top_drivers": "rule_why"}), on=["event_id", "h"], how="left")


@st.cache_data
def load_meta():
    meta = json.loads((ROOT / "src/models/delay/artifacts/meta.json").read_text())
    try:
        sha = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=format:%Y-%m-%d %H:%M", "--",
                              "outputs/predictions/delay_preds.csv"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    return meta, sha or "unknown"


@st.cache_data
def load_csv(name):
    path = MET / name
    return pd.read_csv(path) if path.exists() else None


def board(ev, preds, icao, day, h, tz):
    d = ev[(ev["airport"] == icao) & (ev["h"] == h) & (ev["sched_time_local"].dt.date == day)]
    d = d.merge(preds, on=["event_id", "h"], how="left").sort_values("sched_time_local")
    code = d["coco"].isin(ADVERSE_CODES)
    cross, tail = d["crosswind"].abs() >= CROSS_KMH, d["tailwind"] >= TAIL_KMH
    d["adverse"] = code | cross | tail
    obs_local = d["feature_obs_time"].dt.tz_convert(tz)
    reason = np.select([code, cross, tail], ["condition code", f"crosswind ≥ {CROSS_KMH}", f"tailwind ≥ {TAIL_KMH}"], "")
    d["weather_used"] = [
        (f"{'⚠ ADVERSE (' + r + ') · ' if a else ''}{COCO.get(int(c), '?') if pd.notna(c) else 'n/a'}, "
         f"cross {abs(x):.0f} / tail {t:.0f} km/h · obs {o:%H:%M}Z ({ol:%H:%M} local)") if pd.notna(o) else "no weather ≤ t−h"
        for a, r, c, x, t, o, ol in zip(d["adverse"], reason, d["coco"], d["crosswind"].fillna(0), d["tailwind"].fillna(0),
                                        d["feature_obs_time"], obs_local)]
    return d


def calib_text(risk, cal):
    if cal is None or pd.isna(risk):
        return ""
    row = cal[cal["pred_max"] >= risk].head(1)
    row = cal.tail(1) if row.empty else row
    r = row.iloc[0]
    return f"{r.observed_rate:.0%} late (n={int(r.n)})"


# ------------------------------------------------------------------ sidebar
cfg, ev, preds = load_config(), load_events(), load_preds()
meta, model_version = load_meta()
airports = cfg["airports"]
thr_by_apt = {k: v["threshold"] for k, v in meta.get("thresholds_by_airport", {}).items()}

for key, val in [("icao", "KDFW"), ("day", date(2025, 5, 2)), ("h", 3)]:
    st.session_state.setdefault(key, val)


def jump(icao, day, h):
    st.session_state.update(icao=icao, day=day, h=h)


def pick_code():
    """Airport typed as IATA or ICAO -> hub selection, or a message saying what is missing."""
    c = st.session_state["code"].strip().upper()
    match = c if c in airports else next((k for k, v in airports.items() if v["iata"] == c), None)
    if not c:
        msg = None
    elif match is None:
        msg = ("error", f"{c} is not configured. Add it to config/airports.yaml (iata, timezone, runway_heading_deg) "
                        f"and run `scripts/run_pipeline.py --airports {c}`.")
    elif match not in set(ev["airport"]):
        msg = ("warning", f"{match} is configured but has no events yet: run the pipeline and model for it.")
    else:
        msg = None
        st.session_state["icao"] = match
    st.session_state["code_msg"] = msg


with st.sidebar:
    st.markdown("### ✈️ Delay-risk board")
    st.caption("User: **airline duty manager** at the hub. The model ranks flights; **you decide**.")
    st.text_input("Airport code (IATA or ICAO)", key="code", on_change=pick_code, placeholder="e.g. DFW, KATL, ORD")
    if st.session_state.get("code_msg"):
        kind, text = st.session_state["code_msg"]
        getattr(st, kind)(text)
    icao = st.selectbox("Hub", sorted(set(ev["airport"])), key="icao",
                        format_func=lambda k: f"{airports.get(k, {}).get('iata', k)} ({k})")
    h = st.radio("Forecast horizon h (hours before scheduled time)", HORIZONS, key="h", horizontal=True)
    tz = airports[icao]["timezone"]
    test_days = sorted(ev.loc[ev["event_id"].isin(preds["event_id"]), "sched_time_local"].dt.date.unique())
    day = st.date_input("Day (local, 2025 test period)", key="day", min_value=test_days[0], max_value=test_days[-1])
    default_thr = thr_by_apt.get(icao, meta["threshold"])
    thr = st.slider("Alert threshold (risk ≥)", 0.02, 0.50, float(round(default_thr, 3)), 0.005,
                    help=f"Default {default_thr:.3f}: max F1 on {icao}'s 2024-H2 validation slice (never test).")
    reveal = st.toggle("Replay mode: reveal what actually happened", value=False,
                       help="Off = what the duty manager sees at forecast time. On = compare with outcomes.")
    user = st.text_input("Your name (for the decision log)", value="duty manager")
    st.markdown("**Demo days**")
    for label, (a, dd, hh) in DEMO_DAYS.items():
        st.button(label, on_click=jump, args=(a, dd, hh), width="stretch")
    st.caption(f"Model: P3 LightGBM · preds commit {model_version} · features md5 "
               f"{meta.get('event_features_md5', '?')[:8]}\n\nWeather: organiser hourly file (Meteostat, CC BY 4.0).")

cal_all = load_csv("event_calibration_p3_final.csv")
cal = None if cal_all is None else cal_all[(cal_all["h"] == h) & (cal_all["predictor"] == "model")].sort_values("bin")

tabs = st.tabs(["🛫 Risk board", "📈 How good is it?", "⚠️ Failure cases", "🔒 Data & leakage", "🛬 Arrivals/hour (optional)",
                "📝 Decision log"])

# ------------------------------------------------------------------ 1 risk board
with tabs[0]:
    d = board(ev, preds, icao, day, h, tz)
    st.subheader(f"{airports[icao]['iata']} · {day:%a %d %b %Y} · forecast issued {h} h before each flight")
    if d.empty:
        st.info("No flights for this airport and day in the (0.5%) flight sample. Pick another day.")
    else:
        d["alert"] = d["risk_prob"] >= thr
        k = st.columns(5 if reveal else 4)
        k[0].metric("Flights (dep + arr)", len(d))
        k[1].metric("Alerts", int(d["alert"].sum()), help=f"risk ≥ {thr:.3f}")
        k[2].metric("Expected >45 min late", f"{d['risk_prob'].sum():.1f}", help="Sum of risks. Weather rule (B3): "
                    f"{d['rule_risk'].sum():.1f}")
        k[3].metric("Adverse weather at t−h", f"{d['adverse'].mean():.0%}")
        if reveal:
            lab = d.dropna(subset=["label_45"])
            tp = int((lab["alert"] & (lab["label_45"] == 1)).sum())
            k[4].metric("Actually late", int(lab["label_45"].sum()),
                        help=f"alerts correct: {tp}/{int(lab['alert'].sum())}; late flights caught: {tp}/{int(lab['label_45'].sum())}")

        d["hour"] = d["sched_time_local"].dt.hour
        hourly = d.groupby("hour").agg(model=("risk_prob", "sum"), rule=("rule_risk", "sum"),
                                       actual=("label_45", "sum")).reset_index()
        long = hourly.melt("hour", ["model", "rule"] + (["actual"] if reveal else []), "series", "late")
        long["series"] = long["series"].map({"model": "Model: expected late", "rule": "Weather rule (B3): expected late",
                                             "actual": "Actual late (replay)"})
        chart = alt.Chart(long).mark_bar(opacity=0.85).encode(
            x=alt.X("hour:O", title="Scheduled local hour"), xOffset="series:N",
            y=alt.Y("late:Q", title="Flights > 45 min late"), color=alt.Color("series:N", legend=alt.Legend(orient="top")),
            tooltip=["hour", "series", alt.Tooltip("late:Q", format=".2f")]).properties(height=220)
        st.altair_chart(chart, width="stretch")

        view = pd.DataFrame({
            "event_id": d["event_id"],
            "time (local)": d["sched_time_local"].dt.strftime("%H:%M"),
            "type": d["event_type"].map({"dep": "🛫 dep", "arr": "🛬 arr"}),
            "other end": d["other_airport"], "carrier": d["carrier"],
            "risk": d["risk_prob"], "alert": np.where(d["alert"], "🔴 ALERT", ""),
            "similar flights (2025)": [calib_text(r, cal) for r in d["risk_prob"]],
            "weather rule risk": d["rule_risk"],
            "why (model drivers)": d["top_drivers"].fillna(""),
            "weather used (observed ≤ t−h)": d["weather_used"],
            "decision": "—", "note": ""})
        if reveal:
            view["outcome"] = np.select(
                [d["cancelled"] == 1, d["label_45"].isna(), d["label_45"] == 1],
                ["cancelled (not scored)", "no label", "LATE " + d["delay_min"].round().astype("Int64").astype(str) + " min"],
                "on time-ish " + d["delay_min"].round().astype("Int64").astype(str) + " min")
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key=f"ed_{icao}_{day}_{h}",
            disabled=[c for c in view.columns if c not in ("decision", "note")],
            column_config={
                "event_id": None,
                "risk": st.column_config.ProgressColumn("risk P(>45 min)", min_value=0, max_value=1, format="%.2f"),
                "weather rule risk": st.column_config.NumberColumn(format="%.2f"),
                "decision": st.column_config.SelectboxColumn(options=DECISIONS, required=True),
                "similar flights (2025)": st.column_config.TextColumn(
                    help="Calibration: of 2025 test flights the model scored in this risk band, this share was late."),
            })
        c1, c2 = st.columns([1, 3])
        if c1.button("✅ Log my decisions", type="primary"):
            chosen = edited[edited["decision"] != "—"]
            if chosen.empty:
                st.warning("No decisions selected.")
            else:
                rows = chosen.merge(d[["event_id", "risk_prob", "feature_obs_time"]], on="event_id").assign(
                    logged_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), user=user, airport=icao,
                    h=h, threshold=thr, model_version=model_version)
                rows = rows[["logged_at_utc", "user", "airport", "h", "event_id", "risk_prob", "threshold",
                             "feature_obs_time", "decision", "note", "model_version"]]
                rows.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
                st.success(f"Logged {len(rows)} decision(s) to app/decision_log.csv")
        c2.caption("Risk = model probability of a delay > 45 min (dep: DEP_DELAY, arr: ARR_DELAY). The alert is advice "
                   "only; nothing is sent or changed until you decide. Every row shows the exact weather observation "
                   "the forecast used and its timestamp.")

# ------------------------------------------------------------------ 2 evaluation
with tabs[1]:
    st.subheader("Model vs baselines on unseen 2025 flights")
    res, basem = load_csv("event_model_vs_baseline_p3_final.csv"), load_csv("event_baseline_metrics.csv")
    if res is None or basem is None:
        st.warning("Run src/eval/baselines_events.py and evaluate_events.py first.")
    else:
        st.caption("Train 2023–24, test 2025-01..07 (5,704 events, 13.2% late). PR-AUC: higher is better; a random "
                   "guess scores the late rate (0.132). 90% CI resamples whole days. Best baseline = the transparent "
                   "weather rule a controller could apply today (B3).")
        b = basem[(basem["slice"] == "ALL") & (basem["h"] == h)][["model", "pr_auc"]]
        b["model"] = b["model"].map({"B0_const": "B0 constant", "B1_apt_dir_hour": "B1 history (airport×dir×hour)",
                                     "B2_apt_dir_hour_month": "B2 + month", "B3_B2_x_adverse_wx": "B3 + weather rule"})
        m = res[(res["h"] == h) & (res["slice"] == "ALL")].iloc[0]
        b = pd.concat([b, pd.DataFrame({"model": ["Model (P3 LightGBM)"], "pr_auc": [m["pr_auc_model"]]})])
        bar = alt.Chart(b).mark_bar().encode(
            x=alt.X("pr_auc:Q", title=f"PR-AUC at h = {h}", scale=alt.Scale(domain=[0, 0.3])),
            y=alt.Y("model:N", sort=None, title=None),
            color=alt.condition(alt.datum.model == "Model (P3 LightGBM)", alt.value("#d62728"), alt.value("#7f7f7f")),
            tooltip=["model", alt.Tooltip("pr_auc:Q", format=".3f")]).properties(height=200)
        st.altair_chart(bar + bar.mark_text(align="left", dx=3).encode(text=alt.Text("pr_auc:Q", format=".3f")),
                        width="stretch")
        sig = m["pr_auc_diff_ci90_lo"] > 0
        (st.success if sig else st.warning)(
            f"h = {h}: model {m['pr_auc_model']:.3f} vs weather rule {m['pr_auc_base']:.3f} "
            f"({m['pr_auc_impr_pct']:+.0f}%, 90% CI of difference [{m['pr_auc_diff_ci90_lo']:+.3f}, "
            f"{m['pr_auc_diff_ci90_hi']:+.3f}]) — " + ("significant." if sig else "**not significant: treat as a tie.**"))
        t = res[res["slice"].isin(["ALL", "ATL", "DFW", "arr", "dep", "adverse wx t-3h"])].copy()
        t["CI"] = [f"[{a:+.3f}, {b_:+.3f}]" for a, b_ in zip(t["pr_auc_diff_ci90_lo"], t["pr_auc_diff_ci90_hi"])]
        t = t.rename(columns={"pr_auc_model": "PR-AUC model", "pr_auc_base": "PR-AUC rule", "pr_auc_impr_pct": "change %",
                              "recall_at_p0.30_model": "recall@P≥0.30 model", "recall_at_p0.30_base": "recall@P≥0.30 rule"})
        st.dataframe(t[["h", "slice", "n", "positives", "PR-AUC model", "PR-AUC rule", "change %", "CI",
                        "recall@P≥0.30 model", "recall@P≥0.30 rule"]].style.format(precision=3),
                     hide_index=True, width="stretch")
        if cal_all is not None:
            st.markdown("**Calibration** (5 equal-size bins): predicted vs observed late rate. On the diagonal = "
                        "the risk number can be taken at face value.")
            c = cal_all[cal_all["h"] == h].replace({"predictor": {"model": "Model", "B3_B2_x_adverse_wx": "Weather rule"}})
            pts = alt.Chart(c).mark_line(point=True).encode(
                x=alt.X("mean_pred:Q", title="mean predicted risk"), y=alt.Y("observed_rate:Q", title="observed late rate"),
                color="predictor:N", tooltip=["predictor", "bin", "n", alt.Tooltip("mean_pred:Q", format=".3f"),
                                              alt.Tooltip("observed_rate:Q", format=".3f")])
            diag = alt.Chart(pd.DataFrame({"x": [0, 0.3], "y": [0, 0.3]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            st.altair_chart((diag + pts).properties(height=260), width="stretch")
            st.caption("Both run 2–4 points low in the lower bins: 2025 had more delays (13.2%) than 2023–24 (10.0%).")

# ------------------------------------------------------------------ 3 failure cases
with tabs[2]:
    st.subheader("Where it fails (2025 test, h = 3)")
    c1, c2 = st.columns(2)
    c1.button("Open missed storm: DFW 2025-04-23", on_click=jump, args=("KDFW", date(2025, 4, 23), 3))
    c2.button("Open false alarm: DFW 2025-05-30", on_click=jump, args=("KDFW", date(2025, 5, 30), 3))
    st.markdown("""
- **Missed storm (DFW 2025-04-23):** 9 of 11 flights > 45 min late; model expected about 2. At t−3h the sky was
  overcast; a thunderstorm appears only at 12:00, i.e. it formed **< 3 h ahead**. BTS cause minutes: late aircraft 574,
  weather 243, NAS 139 → delay **propagated from other airports**. *Fix:* forecasts (TAF / convective outlook),
  inbound-aircraft delay, recent observed hub delays. *How you'd notice:* risk board says low while the morning
  flights are already late → always compare with live delays.
- **False alarm (DFW 2025-05-30):** 0 of 22 late. A 17–30 km/h north wind counts as a *tailwind* on the assumed
  runway heading 180°, but DFW simply operates north flow. Across 2025, "tailwind-only" adverse hours have a late
  rate of 0.145 vs 0.117 normal and 0.202 for rain/fog codes. *Fix:* take the smaller tailwind over both runway ends.
- **Biggest gap overall:** no gain over the weather rule at **ATL** at any horizon.
""")
    if reveal:
        d = board(ev, preds, icao, day, h, tz)
        miss = d[(d["label_45"] == 1) & (d["risk_prob"] < thr)]
        st.markdown(f"**Missed today ({airports[icao]['iata']} {day}, h={h}):** {len(miss)} late flights below the "
                    f"alert threshold.")
        if len(miss):
            st.dataframe(miss[["sched_time_local", "event_type", "other_airport", "risk_prob", "delay_min", "weather_used"]],
                         hide_index=True, width="stretch")
    fc = MET / "failure_cases_p3_final.md"
    fc = fc if fc.exists() else MET / "failure_cases_p3_v1.md"
    if fc.exists():
        with st.expander(f"Per-flight detail ({fc.name})"):
            st.markdown(fc.read_text(encoding="utf-8"))

# ------------------------------------------------------------------ 4 data & leakage
with tabs[3]:
    st.subheader("What the model is allowed to see")
    sub = ev.dropna(subset=["feature_obs_time"])
    margin = (sub["sched_time_utc"] - pd.to_timedelta(sub["h"], unit="h") - sub["feature_obs_time"]) / pd.Timedelta(hours=1)
    k = st.columns(3)
    k[0].metric("Rows with weather obs ≤ t − h", f"{(margin >= 0).mean():.2%}", help=f"{len(sub):,} rows checked live")
    k[1].metric("Smallest margin (obs → t − h)", f"{margin.min():.2f} h")
    k[2].metric("Rows without weather (start of data)", int(ev["feature_obs_time"].isna().sum()))
    st.markdown(f"""
- **Weather:** only the organiser hourly file (Meteostat), joined on airport + UTC hour. A row stamped T holds the
  METAR from ~T:52, so it is treated as available at **T + 1 h**; the latest usable row is ≤ scheduled time − h.
  Verified: the file is true UTC (matches Meteostat T+1 on 66% of hours; every other offset ≤ 1.3%), DST included.
- **Never used as inputs:** the flights file's own weather (`*_ORIGIN/*_DEST`, misaligned 4–6 h), any actual time
  (DEP_TIME, ARR_TIME, WHEELS_*, TAXI_*), any `*_DELAY` / cause column, CANCELLED, DIVERTED, DIV*, AIR_TIME.
- **Split:** train 2023–24, test 2025 (no random split); thresholds tuned on 2024-H2 validation only.
- **Runway:** ATL 080°, DFW 180° (organiser assumption) → headwind / crosswind / tailwind.
- Full file:line leakage review: `src/eval/README.md`.
""")
    st.caption("Data: BTS On-Time Performance (sample) + organiser hourly weather from Meteostat (CC BY 4.0).")

# ------------------------------------------------------------------ 5 arrivals
with tabs[4]:
    st.subheader("Optional: arrivals per hour (AAR proxy, P2, full BTS data)")
    aar_path = PRED / "aar_preds.csv"
    if not aar_path.exists():
        st.info("No aar_preds.csv.")
    else:
        a = pd.read_csv(aar_path, parse_dates=["target_hour_utc"])
        a = a[(a["icao"] == icao) & (a["h"] == h)]
        a["local"] = a["target_hour_utc"].dt.tz_convert(tz)
        a = a[a["local"].dt.date == day]
        if a.empty:
            st.info("No arrival forecasts for this airport/day.")
        else:
            base = alt.Chart(a).encode(x=alt.X("local:T", title="local hour"))
            band = base.mark_area(opacity=0.25).encode(y=alt.Y("p10:Q", title="arrivals per hour"), y2="p90:Q")
            line = base.mark_line(point=True, color="#d62728").encode(y="pred_arr_count:Q",
                                                                       tooltip=["local:T", "pred_arr_count", "p10", "p90"])
            layers = band + line + base.mark_line(strokeDash=[4, 4], color="gray").encode(y="pred_schedule:Q")
            if reveal:
                layers += base.mark_point(color="black", filled=True).encode(y="actual_arr_count:Q")
            st.altair_chart(layers.properties(height=300), width="stretch")
            st.caption("Red = forecast, shaded = P10–P90 band, dashed = schedule baseline"
                       + (", black dots = actual" if reveal else "") + ". Realised arrivals are an AAR *proxy* "
                       "(min of demand and capacity), not the FAA rate. P2-reported test RMSE: KATL 6.82, KDFW 8.89 "
                       "(−21% / −17% vs best baseline); independent P4 check in progress.")

# ------------------------------------------------------------------ 6 decision log
with tabs[5]:
    st.subheader("Decision log (human in the loop)")
    if LOG.exists():
        st.dataframe(pd.read_csv(LOG).iloc[::-1], hide_index=True, width="stretch")
    else:
        st.info("No decisions logged yet. Use the decision column on the risk board.")
