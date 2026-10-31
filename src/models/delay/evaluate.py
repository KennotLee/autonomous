"""P3 evaluation on the 2025 test set (labelled rows only).

    python -m src.models.delay.evaluate      (after train.py)

Writes outputs/metrics/: metrics.csv (model vs base rate vs weather rule, per
slice), threshold.md, calibration.png, oracle_gap.csv, generalisation.csv,
failure_cases.md. Thresholds and rule cut-offs never look at test data.
"""
import json
from itertools import product

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .train import MET_DIR, MODEL_DIR, X, choose_threshold, feature_cols, fit, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rule_fit(tr: pd.DataFrame):
    """Weather rule ts_flag OR crosswind >= c OR tailwind >= t; (c, t) = max F1 on TRAIN."""
    y = tr.label_45.to_numpy()
    ts = tr.ts_flag.fillna(0).to_numpy() > 0
    cw, tw = tr.crosswind.to_numpy(), tr.tailwind.to_numpy()
    grid_c = np.unique(np.nanquantile(cw, np.linspace(0.5, 0.995, 40)))
    grid_t = np.unique(np.nanquantile(tw, np.linspace(0.5, 0.995, 40)))
    best = (-1, None, None)
    for c, t in product(grid_c, grid_t):
        f = ts | (cw >= c) | (tw >= t)
        tp = (f & (y == 1)).sum()
        f1 = 2 * tp / max(f.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(c), float(t))
    return best


def rule_score(df, c, t):
    return (df.ts_flag.fillna(0).gt(0) | df.crosswind.ge(c) | df.tailwind.ge(t)).to_numpy().astype(float)


def scores(y, p, thr):
    out = {"n": len(y), "pos_rate": y.mean(), "pr_auc": average_precision_score(y, p),
           "brier": brier_score_loss(y, np.clip(p, 0, 1))}
    if thr is None:  # constant predictor: no alert decision
        return out | {"precision": np.nan, "recall": np.nan, "alert_rate": np.nan}
    flag = p >= thr
    tp = (flag & (y == 1)).sum()
    return out | {"precision": tp / max(flag.sum(), 1), "recall": tp / max((y == 1).sum(), 1),
                  "alert_rate": flag.mean()}


def main():
    ev, lab, cats = prepare()
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    thr, cols = meta["threshold"], meta["features"]
    model = lgb.Booster(model_file=str(MODEL_DIR / "gbm.txt"))
    tr, va = lab[lab.split == "train"], lab[lab.split == "val"]
    te = lab[lab.split == "test"].copy()
    te["p_gbm"] = model.predict(X(te, cols, cats))
    te["p_base_rate"] = tr.label_45.mean()
    rf1, rc, rt = rule_fit(tr)
    te["p_rule"] = rule_score(te, rc, rt)
    thresholds = {"gbm": thr, "rule": 0.5, "base_rate": None}

    # ---- metrics.csv: overall and per h / airport / event_type
    rows = []
    slices = [("all", "all", te)] + [(k, v, g) for k in ("h", "airport", "event_type") for v, g in te.groupby(k)]
    for kind, val, g in slices:
        for name, t in thresholds.items():
            rows.append({"slice": kind, "value": val, "model": name} | scores(g.label_45.to_numpy(), g[f"p_{name}"].to_numpy(), t))
    met = pd.DataFrame(rows)
    met.to_csv(MET_DIR / "metrics.csv", index=False, float_format="%.4f")

    # ---- threshold.md
    (MET_DIR / "threshold.md").write_text(
        f"# Decision threshold: {thr:.3f}\n\n"
        f"Max F1 on the validation slice (2024-07..12, never test): a missed severe delay and a false alert "
        f"weigh the same. Validation precision {meta['val_precision']:.3f}, recall {meta['val_recall']:.3f}.\n\n"
        f"Weather rule baseline: ts_flag OR crosswind >= {rc:.1f} km/h OR tailwind >= {rt:.1f} km/h "
        f"(cut-offs = max F1 on train 2023-01..2024-06, F1 {rf1:.3f}).\n")

    # ---- calibration.png
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    b = pd.DataFrame({"b": np.minimum((te.p_gbm * 10).astype(int), 9), "p": te.p_gbm, "y": te.label_45})
    b = b.groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    b = b[b.n >= 30]
    ax.plot(b.p, b.y, marker="o", label="LightGBM (test 2025)")
    ax.axhline(te.label_45.mean(), color="#d62728", lw=0.8, ls=":", label=f"test base rate {te.label_45.mean():.3f}")
    ax.set(xlabel="predicted P(delay > 45 min)", ylabel="observed share", xlim=(0, 0.6), ylim=(0, 0.6),
           title="Reliability, test 2025 (bins with >= 30 rows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MET_DIR / "calibration.png", dpi=130)
    plt.close(fig)
    print("calibration bins:\n" + b.round(3).to_string())

    # ---- oracle gap: same model, same-time weather instead of the lagged weather
    ocols = feature_cols(ev, oracle=True)
    om = fit(tr, va, ocols, cats)
    te["p_oracle"] = om.predict_proba(X(te, ocols, cats))[:, 1]
    gap = pd.DataFrame([{"h": h, "pr_auc_lagged": average_precision_score(g.label_45, g.p_gbm),
                         "pr_auc_oracle": average_precision_score(g.label_45, g.p_oracle)}
                        for h, g in [("all", te)] + list(te.groupby("h"))])
    gap["gap"] = gap.pr_auc_oracle - gap.pr_auc_lagged
    gap.to_csv(MET_DIR / "oracle_gap.csv", index=False, float_format="%.4f")

    # ---- generalisation: train on one airport, test on the other (airport is constant -> dropped)
    gcols = [c for c in cols if c != "airport"]
    gen = []
    for src, dst in (("KATL", "KDFW"), ("KDFW", "KATL")):
        gm = fit(tr[tr.airport == src], va[va.airport == src], gcols, cats)
        d = te[te.airport == dst]
        gen.append({"train_airport": src, "test_airport": dst,
                    "pr_auc_transfer": average_precision_score(d.label_45, gm.predict_proba(X(d, gcols, cats))[:, 1]),
                    "pr_auc_in_airport_model": average_precision_score(d.label_45, d.p_gbm),
                    "test_pos_rate": d.label_45.mean()})
    pd.DataFrame(gen).to_csv(MET_DIR / "generalisation.csv", index=False, float_format="%.4f")

    # ---- failure cases: worst high-confidence misses (false negatives, one row per event)
    fn = te[(te.label_45 == 1)].sort_values(["p_gbm", "delay_min"], ascending=[True, False])
    fn = fn.drop_duplicates("event_id").head(3)
    wx = ["temp", "prcp", "wspd", "pres", "coco", "crosswind", "tailwind", "ts_flag"]
    owx = ["oracle_temp", "oracle_prcp", "oracle_wspd", "oracle_pres", "oracle_coco"]
    lines = ["# Failure cases: 3 worst high-confidence misses (test 2025)\n",
             f"False negatives with the lowest predicted risk (threshold {thr:.3f}); one row per event, "
             "ties broken by larger delay. Weather 'seen' = event-airport row available at feature_obs_time; "
             "'oracle' = P1's oracle_* (flights-CSV weather; its join puts it ~4-6 h before the event, see "
             "notes.md). Never a feature.\n"]
    for r in fn.itertuples():
        seen = {c: getattr(r, c) for c in wx}
        orac = {c: getattr(r, c) for c in owx}
        benign = lambda coco, prcp: coco <= 4 and prcp == 0  # noqa: E731  clear/cloudy, dry
        if benign(r.coco, r.prcp) and benign(r.oracle_coco, r.oracle_prcp) and not r.ts_flag:
            reason = (f"weather was benign in both rows (coco {r.coco:.0f}/{r.oracle_coco:.0f}, no precipitation, "
                      f"wind <= {max(r.wspd, r.oracle_wspd):.0f} km/h): nothing in our data explains the "
                      f"{r.delay_min:.0f}-min delay, so its cause is outside the weather features")
        elif benign(r.coco, r.prcp) and not benign(r.oracle_coco, r.oracle_prcp):
            reason = f"weather worsened after feature_obs_time (coco {r.coco:.0f} -> {r.oracle_coco:.0f})"
        else:
            reason = "adverse weather was already visible at feature_obs_time but the model scored it low"
        lines += [f"## {r.event_id} (h = {r.h})",
                  f"- scheduled {r.sched_time_local} local ({r.sched_time_utc} UTC), {r.event_type} at {r.airport}, "
                  f"carrier {r.carrier}; delay {r.delay_min:.0f} min; predicted risk {r.p_gbm:.3f}",
                  f"- feature_obs_time {r.feature_obs_time}: " + ", ".join(f"{k}={v:.3g}" for k, v in seen.items()),
                  "- oracle: " + ", ".join(f"{k}={v:.3g}" for k, v in orac.items()),
                  f"- likely reason: {reason}\n"]
    (MET_DIR / "failure_cases.md").write_text("\n".join(lines))

    (MET_DIR / "notes.md").write_text(
        "# Notes on the metrics\n\n"
        "- **Oracle gap is not a leakage ceiling.** P1's `oracle_*` = flights-CSV weather. Checked against the "
        "organiser hourly file: it matches the true event hour (UTC) in ~5% of rows but the *local hour read as "
        "UTC* in ~44% (the misaligned upstream join in CLAUDE.md), i.e. it is weather from ~4-6 h before the event. "
        "So `oracle_gap.csv` compares lagged weather with other lagged weather; the small gap does not show what "
        "true same-time weather would fake.\n"
        f"- **Weather rule** cut-offs were chosen by max F1 on train; with a weak signal that lands on loose "
        f"cut-offs (crosswind >= {rc:.1f}, tailwind >= {rt:.1f} km/h) and flags about half of all events.\n"
        "- **Generalisation** models drop `airport` (constant in training).\n")

    with pd.option_context("display.width", 200):
        print(met[met.slice == "all"].round(3).to_string(index=False))
        print(gap.round(3).to_string(index=False)); print(pd.DataFrame(gen).round(3).to_string(index=False))
    print(f"rule: c={rc:.1f} t={rt:.1f} (train F1 {rf1:.3f}); threshold {thr:.3f}")


if __name__ == "__main__":
    main()
