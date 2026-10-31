"""P4 dashboard: weather delay-risk board for ATC (traffic management coordinator) and airline ops at ATL/DFW.

    .venv/bin/streamlit run app/app.py        (Windows: .venv\\Scripts\\streamlit run app/app.py)

Reads only committed team outputs:
  data/processed/event_features.parquet        P1 events + weather observed at t - h (outcomes shown only in replay/trends)
  outputs/predictions/delay_preds.csv          P3 model risk per (event_id, h)
  outputs/predictions/event_baseline_preds.csv P4 baselines (B3 weather rule shown next to the model)
  outputs/metrics/event_*.csv, failure_cases_*.md, generalisation.csv, src/models/delay/artifacts/meta.json
  outputs/predictions/aar_preds.csv, aar_metrics_table.md   P2 optional hourly arrivals
Writes app/decision_log.csv (human decisions). No LLM: risk = LightGBM, recommendations = explicit rules shown on screen.
Nothing in the app acts on its own.
"""
import json
import math
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
TEST_START = "2025-01-01"  # FL_DATE split: train 2023-24, test 2025
ADVERSE_CODES, CROSS_KMH, TAIL_KMH = {5, 6, 7, 8, 9, 25, 26, 27}, 35, 15  # organiser "adverse weather" rule
RARE_CODES = {"thunderstorm": {23, 24, 25, 26, 27}, "snow/ice": {10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}}
DISAGREE = 0.10       # |model - rule| at which the board says "model and weather rule disagree"
ELEVATED_X = 1.5      # a 3-h window is elevated if its mean risk >= 1.5 x the airport's 2023-24 late rate ...
ELEVATED_ALERTS = 2   # ... or it holds >= 2 alerts
COCO = {1: "clear", 2: "fair", 3: "cloudy", 4: "overcast", 5: "fog", 6: "freezing fog", 7: "light rain", 8: "rain",
        9: "heavy rain", 10: "freezing rain", 11: "heavy freezing rain", 12: "sleet", 13: "heavy sleet",
        14: "light snow", 15: "snow", 16: "heavy snow", 17: "rain shower", 18: "heavy rain shower",
        19: "sleet shower", 20: "heavy sleet shower", 21: "snow shower", 22: "heavy snow shower", 23: "lightning",
        24: "hail", 25: "thunderstorm", 26: "heavy thunderstorm", 27: "storm"}
DEMO_DAYS = {"Good day (hand-picked): ATL 2025-04-06": ("KATL", date(2025, 4, 6), 3),
             "Failure: missed storm, DFW 2025-04-23": ("KDFW", date(2025, 4, 23), 3),
             "Failure: false alarm, DFW 2025-05-30": ("KDFW", date(2025, 5, 30), 3)}
ROLES = ["ATC traffic management coordinator (TMC)", "Airline ops duty manager"]
DECISIONS = ["—", "Act", "Watch", "Override: no action"]

st.set_page_config(page_title="Weather delay-risk board", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_config():
    with open(ROOT / "config/airports.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_events():
    cols = ["event_id", "event_type", "airport", "other_airport", "sched_time_local", "sched_time_utc", "h",
            "feature_obs_time", "coco", "crosswind", "tailwind", "headwind", "carrier", "delay_min", "label_45",
            "cancelled", "diverted"]
    ev = pd.read_parquet(ROOT / "data/processed/event_features.parquet", columns=cols)
    ev["test"] = ev["event_id"].str[:10] >= TEST_START
    ev["scored"] = ev["label_45"].notna() & (ev["cancelled"] == 0) & (ev["diverted"] == 0)
    code = ev["coco"].isin(ADVERSE_CODES)
    cross, tail = ev["crosswind"].abs() >= CROSS_KMH, ev["tailwind"] >= TAIL_KMH
    ev["adverse"] = code | cross | tail
    ev["adverse_reason"] = np.select([code, cross, tail], ["condition code", f"crosswind ≥ {CROSS_KMH} km/h",
                                                           f"tailwind ≥ {TAIL_KMH} km/h"], "")
    ev["rare"] = ""
    for name, codes in RARE_CODES.items():
        ev.loc[ev["coco"].isin(codes), "rare"] = name
    return ev


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
def load_csv(name, folder="metrics"):
    path = (MET if folder == "metrics" else PRED) / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def train_stats():
    """2023-24 late rate per airport and number of training events per rare weather condition (h=3 rows)."""
    ev = load_events()
    tr = ev[(ev["h"] == 3) & ~ev["test"] & ev["scored"]]
    return tr.groupby("airport")["label_45"].mean().to_dict(), tr[tr["rare"] != ""].groupby("rare").size().to_dict()


@st.cache_data
def operating_point(thr_by_apt, default_thr):
    """Measured on 2025 test: alerts at P3's per-airport threshold vs the weather rule given the same alert budget."""
    ev, preds = load_events(), load_preds()
    t = ev[ev["test"] & ev["scored"]].merge(preds, on=["event_id", "h"])
    rows = []
    for (apt, h), g in t.groupby(["airport", "h"]):
        y = g["label_45"].to_numpy() == 1
        flag_m = g["risk_prob"].to_numpy() >= thr_by_apt.get(apt, default_thr)
        k = int(flag_m.sum())
        flag_r = np.zeros(len(g), bool)
        flag_r[np.argsort(-g["rule_risk"].to_numpy(), kind="stable")[:k]] = True
        days = g["event_id"].str[:10].nunique()
        row = {"airport": apt, "h": h, "flights": len(g), "late": int(y.sum()), "alerts": k,
               "alerts/day (sample)": k / days}
        for name, f in [("model", flag_m), ("rule", flag_r)]:
            tp = int((f & y).sum())
            row |= {f"{name} caught": tp, f"{name} false alarms": int((f & ~y).sum()),
                    f"{name} precision": tp / max(k, 1), f"{name} recall": tp / max(int(y.sum()), 1)}
        rows.append(row)
    return pd.DataFrame(rows)


def wilson(p, n, z=1.645):
    if n == 0:
        return np.nan, np.nan
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def calib_text(risk, cal):
    if cal is None or pd.isna(risk):
        return ""
    row = cal[cal["pred_max"] >= risk].head(1)
    r = (cal.tail(1) if row.empty else row).iloc[0]
    lo, hi = wilson(r.observed_rate, int(r.n))
    return f"{r.observed_rate:.0%} late (90% CI {lo:.0%}–{hi:.0%}, n={int(r.n)})"


def board(icao, day, h, tz):
    ev, preds = load_events(), load_preds()
    d = ev[(ev["airport"] == icao) & (ev["h"] == h) & (ev["sched_time_local"].dt.date == day)]
    d = d.merge(preds, on=["event_id", "h"], how="left").sort_values("sched_time_local")
    obs_local = d["feature_obs_time"].dt.tz_convert(tz)
    d["weather_used"] = [
        (f"{'⚠ ADVERSE (' + r + ') · ' if a else ''}{COCO.get(int(c), '?') if pd.notna(c) else 'n/a'}, "
         f"head {hw:+.0f} / cross {abs(x):.0f} / tail {t:.0f} km/h · obs {o:%H:%M}Z ({ol:%H:%M} local)")
        if pd.notna(o) else "—"
        for a, r, c, hw, x, t, o, ol in zip(d["adverse"], d["adverse_reason"], d["coco"], d["headwind"].fillna(0),
                                            d["crosswind"].fillna(0), d["tailwind"].fillna(0), d["feature_obs_time"],
                                            obs_local)]
    return d


def cautions(d, rare_counts):
    out = []
    for r in d.itertuples():
        c = []
        if pd.isna(r.feature_obs_time) or pd.isna(r.risk_prob):
            c.append("⛔ NO FORECAST: no weather observed ≤ t−h")
        if r.rare:
            c.append(f"rare condition ({r.rare} at t−h): only {rare_counts.get(r.rare, 0)} training cases")
        if pd.notna(r.risk_prob) and pd.notna(r.rule_risk) and abs(r.risk_prob - r.rule_risk) >= DISAGREE:
            c.append("model and weather rule disagree")
        out.append("; ".join(c))
    return out


def recommendations(d, role, base_rate, thr):
    """Explicit rules (no ML): 3-hour windows whose mean risk or alert count is elevated."""
    d = d[d["risk_prob"].notna()].copy()
    d["win"] = (d["sched_time_local"].dt.hour // 3) * 3
    recs = []
    for w, g in d.groupby("win"):
        alerts, mean_risk = g[g["risk_prob"] >= thr], g["risk_prob"].mean()
        fired = []
        if mean_risk >= ELEVATED_X * base_rate:
            fired.append(f"mean risk {mean_risk:.0%} ≥ {ELEVATED_X}× normal {base_rate:.0%}")
        if len(alerts) >= ELEVATED_ALERTS:
            fired.append(f"{len(alerts)} alerts ≥ {ELEVATED_ALERTS}")
        if not fired:
            continue
        adv = g[g["adverse"]]
        wx = (f"adverse weather at t−h on {len(adv)}/{len(g)} flights ({', '.join(sorted(set(adv['adverse_reason'])))})"
              if len(adv) else "no adverse weather at t−h: risk comes from history (hour/month/carrier), not weather")
        if role == ROLES[0]:
            action = ("Review the arrival/departure rate for this window with the tower and ATCSCC now, and plan a "
                      "Ground Delay Program early rather than a Ground Stop at onset.")
        else:
            ids = ", ".join(f"{r.carrier} {r.event_type} {r.sched_time_local:%H:%M} ({r.other_airport})"
                            for r in alerts.itertuples()) or "none above threshold"
            action = f"Check connections, spare aircraft and crew for: {ids}."
        recs.append({"window": f"{w:02d}:00–{w + 3:02d}:00", "level": "ELEVATED", "rule fired": "; ".join(fired),
                     "weather context": wx, "suggested action (you decide)": action,
                     "flights in sample": len(g), "alerts": len(alerts)})
    return pd.DataFrame(recs)


# ------------------------------------------------------------------ sidebar
cfg, ev, preds = load_config(), load_events(), load_preds()
meta, model_version = load_meta()
airports = cfg["airports"]
thr_by_apt = {k: v["threshold"] for k, v in meta.get("thresholds_by_airport", {}).items()}
base_rates, rare_counts = train_stats()

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
    st.markdown("### ✈️ Weather delay-risk board")
    role = st.radio("I am the…", ROLES, index=0)
    st.text_input("Airport code (IATA or ICAO)", key="code", on_change=pick_code, placeholder="e.g. DFW, KATL, ORD")
    if st.session_state.get("code_msg"):
        kind, text = st.session_state["code_msg"]
        getattr(st, kind)(text)
    icao = st.selectbox("Hub", sorted(set(ev["airport"])), key="icao",
                        format_func=lambda k: f"{airports.get(k, {}).get('iata', k)} ({k})")
    st.caption(f"Runway heading {airports[icao]['runway_heading_deg']}° (organiser assumption): wind is split into "
               "head-, cross- and tailwind on this heading.")
    h = st.radio("Warning horizon h (hours before scheduled time)", HORIZONS, key="h", horizontal=True)
    tz = airports[icao]["timezone"]
    test_days = sorted(ev.loc[ev["event_id"].isin(preds["event_id"]), "sched_time_local"].dt.date.unique())
    day = st.date_input("Day (local, 2025 test period)", key="day", min_value=test_days[0], max_value=test_days[-1])
    default_thr = thr_by_apt.get(icao, meta["threshold"])
    thr = st.slider("Alert threshold (risk ≥)", 0.02, 0.50, float(round(default_thr, 3)), 0.005,
                    help=f"Default {default_thr:.3f}: max F1 on {icao}'s 2024-H2 validation slice (never test).")
    reveal = st.toggle("Replay mode: reveal what actually happened", value=False,
                       help="Off = what you would see at forecast time. On = compare with outcomes.")
    user = st.text_input("Your name (for the decision log)", value="")
    st.markdown("**Demo days**")
    for label, (a, dd, hh) in DEMO_DAYS.items():
        st.button(label, on_click=jump, args=(a, dd, hh), width="stretch")
    st.caption(f"Risk: P3 LightGBM (preds commit {model_version}; features md5 "
               f"{meta.get('event_features_md5', '?')[:8]}). Recommendations: explicit rules. No LLM.\n\n"
               "Weather: organiser hourly file (Meteostat, CC BY 4.0). Flights: BTS On-Time sample.")

cal_all = load_csv("event_calibration_p3_final.csv")
cal = None if cal_all is None else cal_all[(cal_all["h"] == h) & (cal_all["predictor"] == "model")].sort_values("bin")
base_rate = base_rates.get(icao, 0.10)

st.markdown(f"## {airports[icao]['iata']} weather delay-risk board · {role}")
st.caption("**Today:** arrival/departure rates are cut and Ground Stops issued once the weather arrives — FAA training "
           "material: ground stops are \"issued at the time of the constraint\". A block minute costs a US airline "
           "$98.41 (Airlines for America, 2025); airborne arrival management burns ~35 kg fuel/min vs ~0 at the gate "
           "(EUROCONTROL). **This board:** each flight's risk of a > 45 min delay, 3–12 h ahead, from weather observed "
           "≥ h before the scheduled time. It advises; **you decide**.")

tabs = st.tabs(["🛫 Risk board & recommendations", "📊 Historical trends", "📈 How good is it?", "⚠️ Failure cases",
                "🔒 Data, leakage & pipeline", "🛬 Arrivals/hour (bonus)", "📝 Decision log"])

# ------------------------------------------------------------------ 1 risk board
with tabs[0]:
    d = board(icao, day, h, tz)
    st.subheader(f"{day:%a %d %b %Y} · warning issued {h} h before each flight")
    if d.empty:
        st.info("No flights for this airport and day in the flight sample (~0.5% of traffic). Pick another day.")
    else:
        d["alert"] = d["risk_prob"] >= thr
        k = st.columns(5 if reveal else 4)
        k[0].metric("Flights in sample (dep + arr)", len(d))
        k[1].metric("Alerts", int(d["alert"].sum()), help=f"risk ≥ {thr:.3f}")
        k[2].metric("Expected late rate", f"{d['risk_prob'].mean():.0%}", delta=f"normal {base_rate:.0%}",
                    delta_color="off", help=f"Mean risk. Weather rule (B3): {d['rule_risk'].mean():.0%}. "
                                            "Rates, not minutes, so airports compare fairly.")
        k[3].metric("Adverse weather at t−h", f"{d['adverse'].mean():.0%}")
        if reveal:
            lab = d[d["scored"]]
            tp = int((lab["alert"] & (lab["label_45"] == 1)).sum())
            k[4].metric("Actual late rate", f"{lab['label_45'].mean():.0%}",
                        help=f"alerts correct {tp}/{int(lab['alert'].sum())}; late flights caught "
                             f"{tp}/{int(lab['label_45'].sum())}")

        st.markdown("#### Recommendations")
        recs = recommendations(d, role, base_rate, thr)
        if recs.empty:
            st.success(f"No elevated windows (rule: mean risk ≥ {ELEVATED_X}× normal or ≥ {ELEVATED_ALERTS} alerts in a "
                       "3-hour window). Normal operations; keep watching live delays — the model misses delays that "
                       "propagate from other airports.")
        else:
            st.dataframe(recs, hide_index=True, width="stretch")
            st.caption("Every recommendation shows the rule that fired and the weather it used. Counts are from a ~0.5% "
                       "flight sample; real traffic is ~200× larger.")

        d["hour"] = d["sched_time_local"].dt.hour
        hourly = d.groupby("hour").agg(model=("risk_prob", "sum"), rule=("rule_risk", "sum"),
                                       actual=("label_45", "sum")).reset_index()
        long = hourly.melt("hour", ["model", "rule"] + (["actual"] if reveal else []), "series", "late")
        long["series"] = long["series"].map({"model": "Model: expected late", "rule": "Weather rule (B3): expected late",
                                             "actual": "Actual late (replay)"})
        chart = alt.Chart(long).mark_bar(opacity=0.85).encode(
            x=alt.X("hour:O", title="Scheduled local hour"), xOffset="series:N",
            y=alt.Y("late:Q", title="Flights > 45 min late"), color=alt.Color("series:N", legend=alt.Legend(orient="top")),
            tooltip=["hour", "series", alt.Tooltip("late:Q", format=".2f")]).properties(height=200)
        st.altair_chart(chart, width="stretch")

        st.markdown("#### Flights")
        view = pd.DataFrame({
            "event_id": d["event_id"],
            "time (local)": d["sched_time_local"].dt.strftime("%H:%M"),
            "type": d["event_type"].map({"dep": "🛫 dep", "arr": "🛬 arr"}),
            "other end": d["other_airport"], "carrier": d["carrier"],
            "risk": d["risk_prob"], "alert": np.where(d["alert"], "🔴 ALERT", ""),
            "flights scored like this (2025)": [calib_text(r, cal) for r in d["risk_prob"]],
            "weather rule risk": d["rule_risk"],
            "caution": cautions(d, rare_counts),
            "why (model drivers)": d["top_drivers"].fillna(""),
            "weather used (observed ≤ t−h)": d["weather_used"],
            "decision": "—", "note": ""})
        if reveal:
            view["outcome"] = np.select(
                [d["cancelled"] == 1, d["diverted"] == 1, d["label_45"].isna(), d["label_45"] == 1],
                ["cancelled (not scored)", "diverted (not scored)", "no label",
                 "LATE " + d["delay_min"].round().astype("Int64").astype(str) + " min"],
                "not late (" + d["delay_min"].round().astype("Int64").astype(str) + " min)")
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key=f"ed_{icao}_{day}_{h}",
            disabled=[c for c in view.columns if c not in ("decision", "note")],
            column_config={
                "event_id": None,
                "risk": st.column_config.ProgressColumn("risk P(>45 min)", min_value=0, max_value=1, format="%.2f"),
                "weather rule risk": st.column_config.NumberColumn(format="%.2f"),
                "decision": st.column_config.SelectboxColumn(options=DECISIONS, required=True),
                "flights scored like this (2025)": st.column_config.TextColumn(
                    help="Uncertainty: of 2025 test flights the model scored in this risk band, this share was late "
                         "(Wilson 90% interval)."),
            })
        c1, c2 = st.columns([1, 3])
        if c1.button("✅ Log my decisions", type="primary"):
            chosen = edited[edited["decision"] != "—"]
            if not user.strip():
                st.warning("Enter your name in the sidebar first: every decision is attributed.")
            elif chosen.empty:
                st.warning("No decisions selected.")
            else:
                rows = chosen.merge(d[["event_id", "risk_prob", "rule_risk", "feature_obs_time"]], on="event_id").assign(
                    logged_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), user=user.strip(),
                    role=role, airport=icao, h=h, threshold=thr, model_version=model_version)
                rows = rows[["logged_at_utc", "user", "role", "airport", "h", "event_id", "risk_prob", "rule_risk",
                             "threshold", "feature_obs_time", "decision", "note", "model_version"]]
                rows.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
                st.success(f"Logged {len(rows)} decision(s) to app/decision_log.csv")
        c2.caption("Risk = model probability of a delay > 45 min (dep: DEP_DELAY, arr: ARR_DELAY). Alerts and "
                   "recommendations are advice only; nothing is sent or changed until you decide. Every row shows "
                   "the exact weather observation and timestamp it used; ⛔ rows have no forecast.")

# ------------------------------------------------------------------ 2 trends
with tabs[1]:
    st.subheader("Historical trends (delay *rates*, so hubs compare fairly)")
    hist = ev[(ev["h"] == 3) & ev["scored"]].copy()
    hist["month"] = hist["sched_time_local"].dt.to_period("M").dt.to_timestamp()
    m = hist.groupby(["month", "airport"]).agg(rate=("label_45", "mean"), n=("label_45", "size")).reset_index()
    line = alt.Chart(m).mark_line(point=True).encode(
        x=alt.X("month:T", title=None), y=alt.Y("rate:Q", title="share > 45 min late", axis=alt.Axis(format="%")),
        color="airport:N", tooltip=["airport", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("rate:Q", format=".1%"), "n"])
    split = alt.Chart(pd.DataFrame({"x": [pd.Timestamp(TEST_START)]})).mark_rule(strokeDash=[4, 4]).encode(x="x:T")
    st.altair_chart((line + split).properties(height=260), width="stretch")
    st.caption("Dashed line: train (2023–24) | test (2025). 2025 runs hotter (13.2% vs 10.0%), so risks read low.")

    c1, c2 = st.columns(2)
    hh = hist[(hist["airport"] == icao) & ~hist["test"]].copy()
    hh["wx at t−3h"] = np.where(hh["adverse"], "adverse", "normal")
    hh["hour"] = hh["sched_time_local"].dt.hour
    byh = hh.groupby(["hour", "wx at t−3h"]).agg(rate=("label_45", "mean"), n=("label_45", "size")).reset_index()
    c1.markdown(f"**{airports[icao]['iata']} 2023–24: late rate by scheduled hour**")
    c1.altair_chart(alt.Chart(byh[byh["n"] >= 20]).mark_line(point=True).encode(
        x=alt.X("hour:O", title="local hour"), y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="late rate"),
        color="wx at t−3h:N", tooltip=["hour", "wx at t−3h", alt.Tooltip("rate:Q", format=".1%"), "n"]).properties(height=240),
        width="stretch")
    c1.caption("Points with ≥ 20 flights. Evening delays build through the day; adverse weather 3 h before adds on top.")
    trig = hist[(hist["airport"] == icao)].copy()
    trig["trigger at t−3h"] = np.where(trig["adverse"], trig["adverse_reason"], "normal")
    bt = trig.groupby(["trigger at t−3h", "test"]).agg(rate=("label_45", "mean"), n=("label_45", "size")).reset_index()
    bt["period"] = np.where(bt["test"], "2025", "2023–24")
    c2.markdown(f"**{airports[icao]['iata']}: late rate by what made the weather adverse**")
    c2.altair_chart(alt.Chart(bt).mark_bar().encode(
        y=alt.Y("trigger at t−3h:N", title=None), x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="late rate"),
        yOffset="period:N", color="period:N", tooltip=["trigger at t−3h", "period", alt.Tooltip("rate:Q", format=".1%"), "n"])
        .properties(height=240), width="stretch")
    c2.caption("Tailwind on the single assumed runway heading adds little: the airport can turn around (see Failure cases).")

    wk = ev[(ev["airport"] == icao) & (ev["h"] == h) & ev["test"] & ev["scored"]].merge(preds, on=["event_id", "h"])
    wk["week"] = wk["sched_time_local"].dt.to_period("W").dt.start_time
    wk = wk.groupby("week").agg(model=("risk_prob", "mean"), rule=("rule_risk", "mean"), actual=("label_45", "mean"),
                                n=("label_45", "size")).reset_index().melt(["week", "n"], var_name="series", value_name="rate")
    st.markdown(f"**2025, weekly: predicted vs actual late rate ({airports[icao]['iata']}, h = {h})**")
    st.altair_chart(alt.Chart(wk).mark_line(point=True).encode(
        x=alt.X("week:T", title=None), y=alt.Y("rate:Q", axis=alt.Axis(format="%")), color="series:N",
        tooltip=["series", alt.Tooltip("week:T"), alt.Tooltip("rate:Q", format=".1%"), "n"]).properties(height=240),
        width="stretch")

# ------------------------------------------------------------------ 3 evaluation
with tabs[2]:
    st.subheader("Tested on unseen 2025 flights, against baselines")
    res, basem = load_csv("event_model_vs_baseline_p3_final.csv"), load_csv("event_baseline_metrics.csv")
    if res is None or basem is None:
        st.warning("Run src/eval/baselines_events.py and evaluate_events.py first.")
    else:
        st.caption("Train 2023–24, test 2025-01..07: 5,704 departures/arrivals at ATL/DFW, 13.2% late > 45 min. "
                   "PR-AUC: higher is better; guessing scores 0.132. CIs resample whole days. Strongest baseline = the "
                   "transparent weather rule a controller could apply today (B3).")
        b = basem[(basem["slice"] == "ALL") & (basem["h"] == h)][["model", "pr_auc"]]
        b["model"] = b["model"].map({"B0_const": "B0 constant", "B1_apt_dir_hour": "B1 history (airport×dir×hour)",
                                     "B2_apt_dir_hour_month": "B2 + month", "B3_B2_x_adverse_wx": "B3 + weather rule"})
        mrow = res[(res["h"] == h) & (res["slice"] == "ALL")].iloc[0]
        b = pd.concat([b, pd.DataFrame({"model": ["Model (P3 LightGBM)"], "pr_auc": [mrow["pr_auc_model"]]})])
        bar = alt.Chart(b).mark_bar().encode(
            x=alt.X("pr_auc:Q", title=f"PR-AUC at h = {h}", scale=alt.Scale(domain=[0, 0.3])),
            y=alt.Y("model:N", sort=None, title=None),
            color=alt.condition(alt.datum.model == "Model (P3 LightGBM)", alt.value("#d62728"), alt.value("#7f7f7f")),
            tooltip=["model", alt.Tooltip("pr_auc:Q", format=".3f")]).properties(height=190)
        st.altair_chart(bar + bar.mark_text(align="left", dx=3).encode(text=alt.Text("pr_auc:Q", format=".3f")),
                        width="stretch")
        sig = mrow["pr_auc_diff_ci90_lo"] > 0
        (st.success if sig else st.warning)(
            f"h = {h}: model {mrow['pr_auc_model']:.3f} vs weather rule {mrow['pr_auc_base']:.3f} "
            f"({mrow['pr_auc_impr_pct']:+.0f}%, 90% CI of difference [{mrow['pr_auc_diff_ci90_lo']:+.3f}, "
            f"{mrow['pr_auc_diff_ci90_hi']:+.3f}]): " + ("significant." if sig else "**not significant: treat as a tie.**"))

        st.markdown("**Precision and recall at the alert threshold** (per-airport threshold from 2024-H2 validation; "
                    "the weather rule gets the same number of alerts)")
        op = operating_point(thr_by_apt, meta["threshold"])
        opv = op[op["h"] == h].copy()
        st.dataframe(opv.style.format({c: "{:.2f}" for c in opv.columns if "precision" in c or "recall" in c or "/day" in c}),
                     hide_index=True, width="stretch")

        t = res[res["slice"].isin(["ALL", "ATL", "DFW", "arr", "dep", "adverse wx t-3h"])].copy()
        t["CI"] = [f"[{a:+.3f}, {b_:+.3f}]" for a, b_ in zip(t["pr_auc_diff_ci90_lo"], t["pr_auc_diff_ci90_hi"])]
        t = t.rename(columns={"pr_auc_model": "PR-AUC model", "pr_auc_base": "PR-AUC rule", "pr_auc_impr_pct": "change %",
                              "recall_at_p0.30_model": "recall@P≥0.30 model", "recall_at_p0.30_base": "recall@P≥0.30 rule"})
        with st.expander("All horizons and slices (PR-AUC, 90% CI)"):
            st.dataframe(t[["h", "slice", "n", "positives", "PR-AUC model", "PR-AUC rule", "change %", "CI",
                            "recall@P≥0.30 model", "recall@P≥0.30 rule"]].style.format(precision=3),
                         hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        if cal_all is not None:
            c1.markdown("**Calibration** (5 equal-size bins): on the diagonal = the risk can be read at face value")
            c = cal_all[cal_all["h"] == h].replace({"predictor": {"model": "Model", "B3_B2_x_adverse_wx": "Weather rule"}})
            pts = alt.Chart(c).mark_line(point=True).encode(
                x=alt.X("mean_pred:Q", title="mean predicted risk"), y=alt.Y("observed_rate:Q", title="observed late rate"),
                color="predictor:N", tooltip=["predictor", "bin", "n", alt.Tooltip("mean_pred:Q", format=".3f"),
                                              alt.Tooltip("observed_rate:Q", format=".3f")])
            diag = alt.Chart(pd.DataFrame({"x": [0, 0.3], "y": [0, 0.3]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            c1.altair_chart((diag + pts).properties(height=240), width="stretch")
            c1.caption("Risks run 2–4 points low: 2025 had more delays than 2023–24.")
        gen = load_csv("generalisation.csv")
        if gen is not None:
            c2.markdown("**Any airport: train on one hub, test on the other** (P3, PR-AUC)")
            c2.dataframe(gen, hide_index=True, width="stretch")
            c2.caption("A model that never saw the test hub keeps most of its skill (vs late rates 0.149 / 0.117), "
                       "so the pipeline is not fitted to one airport.")

        st.markdown("#### Impact: what changes vs today")
        all_h = op.groupby("h")[["late", "alerts", "model caught", "model false alarms", "rule caught"]].sum()
        r = all_h.loc[h]
        st.info(f"**Measured (2025 test, both hubs, h = {h}):** of {int(r['late'])} flights later than 45 min, "
                f"{int(r['model caught'])} ({r['model caught'] / r['late']:.0%}) were flagged **{h} h ahead**, at a cost of "
                f"{int(r['model false alarms'])} false alarms ({int(r['alerts'])} alerts, precision "
                f"{r['model caught'] / max(r['alerts'], 1):.0%}). The weather rule with the same number of alerts catches "
                f"{int(r['rule caught'])}, so most of today's gain comes from flagging early at all, not from the model "
                f"beating the rule.  \n**Reasoned, not measured:** today the warning comes "
                "at the time of the constraint (0 h). Each delay known hours ahead can be absorbed on the ground or "
                "re-planned instead of in airborne holding (~35 kg fuel/min).")

# ------------------------------------------------------------------ 4 failure cases
with tabs[3]:
    st.subheader("Where it fails (2025 test, h = 3), and how you would know")
    c1, c2 = st.columns(2)
    c1.button("Open missed storm: DFW 2025-04-23", on_click=jump, args=("KDFW", date(2025, 4, 23), 3))
    c2.button("Open false alarm: DFW 2025-05-30", on_click=jump, args=("KDFW", date(2025, 5, 30), 3))
    st.markdown("""
- **Missed storm (DFW 2025-04-23):** 9 of 11 flights > 45 min late; the model expected about 2. At t−3h the sky
  was overcast; a thunderstorm appears only at 12:00, so it formed **< 3 h ahead**. BTS cause minutes: late aircraft
  574, weather 243, NAS 139, so the delay **propagated from other airports**. *Fix:* forecasts (TAF / convective
  outlook), inbound-aircraft delay, recent observed hub delays. *How you'd know:* the board shows "no elevated
  windows" while morning flights are already late, so the board always reminds you to watch live delays.
- **False alarm (DFW 2025-05-30):** 0 of 22 late. A 17–30 km/h north wind counts as a *tailwind* on the assumed
  runway heading 180°, but DFW simply operates north flow. In 2025, "tailwind-only" adverse hours are late 14.5%
  of the time vs 11.7% normal and 20.2% for rain/fog codes (see Historical trends). *Fix:* take the smaller
  tailwind over both runway ends. *How you'd know:* the recommendation says *which* rule fired (tailwind), so you
  can discount it.
- **No gain at ATL** over the weather rule at any horizon; overall the model only ties the rule at h = 3.
- **Rare weather:** thunderstorm at t−h occurs in only a few dozen training flights; rows in that condition carry a
  "rare condition" caution.
""")
    if reveal:
        d = board(icao, day, h, tz)
        miss = d[(d["label_45"] == 1) & (d["risk_prob"] < thr)]
        st.markdown(f"**Missed on {airports[icao]['iata']} {day} (h = {h}):** {len(miss)} late flights below the threshold.")
        if len(miss):
            st.dataframe(miss[["sched_time_local", "event_type", "other_airport", "risk_prob", "delay_min", "weather_used"]],
                         hide_index=True, width="stretch")
    fc = MET / "failure_cases_p3_final.md"
    if fc.exists():
        with st.expander(f"Per-flight detail ({fc.name})"):
            st.markdown(fc.read_text(encoding="utf-8"))

# ------------------------------------------------------------------ 5 data & leakage
with tabs[4]:
    st.subheader("What the model may see, and how the data is built")
    sub = ev.dropna(subset=["feature_obs_time"])
    margin = (sub["sched_time_utc"] - pd.to_timedelta(sub["h"], unit="h") - sub["feature_obs_time"]) / pd.Timedelta(hours=1)
    k = st.columns(3)
    k[0].metric("Rows with weather obs ≤ t − h", f"{(margin >= 0).mean():.2%}", help=f"{len(sub):,} rows checked live")
    k[1].metric("Smallest margin (obs → t − h)", f"{margin.min():.2f} h")
    k[2].metric("Rows with no forecast (no earlier weather)", int(ev["feature_obs_time"].isna().sum()))
    st.markdown(f"""
**No leakage**
- Weather comes only from the organiser hourly file, joined on airport + UTC hour. A row stamped T holds the METAR from
  about T:52, so it counts as available at **T + 1 h**; the latest usable row is ≤ scheduled time − h. Verified: the
  file is true UTC (it matches Meteostat T+1 on 66% of hours, every other offset ≤ 1.3%), including DST.
- **Never used as inputs:** the flights file's own weather (`*_ORIGIN/*_DEST`, misaligned 4–6 h), actual times
  (DEP_TIME, ARR_TIME, WHEELS_*, TAXI_*), any `*_DELAY` / cause column, CANCELLED, DIVERTED, DIV*, AIR_TIME.
- **Split:** train 2023–24, test 2025, no random split. Thresholds are tuned on 2024-H2 validation only.

**Runway directionality:** wind is split on the organiser runway headings (ATL 080°, DFW 180°): headwind = speed·cos(θ),
crosswind = speed·|sin(θ)|, tailwind = −headwind. Assumption: one heading per airport (see the false-alarm case).

**Right tool for each job (no LLM):** LightGBM gives the risk; explicit thresholds and rules give the alerts and
recommendations; statistics (bootstrap CIs, Wilson intervals) give the uncertainty.

**Automatic processing (no manual steps)**
```
python scripts/run_pipeline.py --airports ATL DFW      # P1: flights + hourly weather -> event_features (any airport code)
python tests/test_leakage.py                           # must print ALL LEAKAGE CHECKS PASSED
python -m src.models.delay.train                       # P3: delay model + predictions
python src/eval/baselines_events.py                    # P4: baselines B0-B3
python src/eval/evaluate_events.py --preds outputs/predictions/delay_preds.csv --name p3_final
python -m src.models.aar_bts && python -m src.models.aar_dataset && python -m src.models.train_aar   # P2: downloads full BTS
streamlit run app/app.py
```
Full file:line leakage review: `src/eval/README.md`. Sources: `SOURCES.md`; assumptions: `ASSUMPTIONS.md`.
""")
    st.caption("Data: BTS On-Time Performance (sample, plus full BTS for arrivals/hour) and organiser hourly weather "
               "from Meteostat (CC BY 4.0).")

# ------------------------------------------------------------------ 6 arrivals
with tabs[5]:
    st.subheader("Bonus: arrivals per hour (AAR proxy, P2, full BTS data)")
    aar_path = PRED / "aar_preds.csv"
    if not aar_path.exists():
        st.info("No aar_preds.csv.")
    else:
        a = pd.read_csv(aar_path, parse_dates=["target_hour_utc"])
        a = a[(a["icao"] == icao) & (a["h"] == h)]
        a["local"] = a["target_hour_utc"].dt.tz_convert(tz).dt.tz_localize(None)
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
            st.altair_chart(layers.properties(height=280), width="stretch")
            st.caption("Red = forecast, shaded = P10–P90 band, dashed = schedule baseline"
                       + (", black dots = actual" if reveal else "") + ". Realised arrivals are an AAR *proxy* "
                       "(the smaller of demand and capacity), not the FAA rate.")
        tbl = PRED / "aar_metrics_table.md"
        if tbl.exists():
            with st.expander("RMSE vs baselines, 2025 test (P2-reported)"):
                st.markdown(tbl.read_text(encoding="utf-8"))
        chk = load_csv("aar_bts_model_vs_baseline.csv")
        if chk is not None:
            st.markdown("**Independent P4 check** (target rebuilt from BTS WheelsOn; P4 baselines)")
            st.dataframe(chk[chk["slice"] == "all hours"], hide_index=True, width="stretch")
        else:
            st.caption("Independent P4 re-check of these RMSEs: in progress (src/eval/evaluate_aar_bts.py).")

# ------------------------------------------------------------------ 7 decision log
with tabs[6]:
    st.subheader("Decision log (the human decides)")
    if LOG.exists():
        st.dataframe(pd.read_csv(LOG).iloc[::-1], hide_index=True, width="stretch")
    else:
        st.info("No decisions logged yet. Use the decision column on the risk board.")
