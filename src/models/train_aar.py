"""Train and evaluate the AAR regression model (P2).

    python -m src.models.train_aar                  # validate, train, evaluate, save
    python -m src.models.train_aar --validate-only  # schema/leakage checks only

Flow: validate -> chronological split -> baselines -> LightGBM with early stopping on val ->
refit on train+val with the best iteration count -> score 2025 test -> save artifacts.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from . import aar_config as cfg
from .aar_baselines import HistoricalAverage, persistence
from .aar_data import check, load_airport_timezones, load_hourly, split_chronological, validate_hourly
from .aar_features import AARPreprocessor
from .aar_metrics import evaluate

MODEL_COLS = {"lightgbm": "pred_lgbm", "persistence": "pred_persistence", "hist_avg": "pred_hist_avg"}


def _fit_lgbm(train: pd.DataFrame, timezones, val: pd.DataFrame | None = None, num_rounds: int | None = None):
    pre = AARPreprocessor(timezones).fit(train)
    dtrain = lgb.Dataset(pre.transform(train), train[cfg.TARGET], categorical_feature=cfg.CATEGORICAL)
    if val is None:
        return lgb.train(cfg.LGBM_PARAMS, dtrain, num_boost_round=num_rounds), pre
    dval = lgb.Dataset(pre.transform(val), val[cfg.TARGET], reference=dtrain)
    booster = lgb.train(
        cfg.LGBM_PARAMS,
        dtrain,
        num_boost_round=cfg.NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(100)],
    )
    return booster, pre


def _predict_frame(split_name, df, booster, pre, hist, panel) -> pd.DataFrame:
    out = df[cfg.KEYS].copy()
    out.insert(0, "split", split_name)
    out["y_true"] = df[cfg.TARGET]
    out["pred_lgbm"] = booster.predict(pre.transform(df)).clip(min=0)  # counts cannot be negative
    out["pred_persistence"], _ = persistence(df, panel)
    out["pred_hist_avg"] = hist.predict(df)
    return out


def run(data_path: Path = cfg.HOURLY_FEATURES, out_dir: Path = cfg.PRED_DIR, refit: bool = True) -> pd.DataFrame:
    timezones = load_airport_timezones()
    panel = load_hourly(data_path)
    report = validate_hourly(panel, timezones)
    check(report)

    labelled = panel.dropna(subset=[cfg.TARGET])
    splits = split_chronological(labelled)
    sizes = {k: int(len(v)) for k, v in splits.items()}
    print(f"rows per split: {sizes}")
    if not sizes["train"] or not sizes["val"]:
        raise ValueError(f"train and val must be non-empty, got {sizes}")

    # Stage 1: fit on train, early-stop on val, score val.
    booster, pre = _fit_lgbm(splits["train"], timezones, val=splits["val"])
    best_iter = booster.best_iteration or cfg.NUM_BOOST_ROUND
    hist = HistoricalAverage(timezones).fit(splits["train"])
    frames = [_predict_frame("val", splits["val"], booster, pre, hist, panel)]

    # Stage 2: refit on everything before 2025 with the chosen iteration count, score test.
    fit_df = pd.concat([splits["train"], splits["val"]]) if refit else splits["train"]
    if refit:
        booster, pre = _fit_lgbm(fit_df, timezones, num_rounds=best_iter)
        hist = HistoricalAverage(timezones).fit(fit_df)
    if sizes["test"]:
        frames.append(_predict_frame("test", splits["test"], booster, pre, hist, panel))
    else:
        print("WARNING: no 2025 rows; test metrics not produced")

    preds = pd.concat(frames, ignore_index=True)
    metrics = evaluate(preds, MODEL_COLS)

    out_dir = Path(out_dir)
    model_dir = out_dir / cfg.MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model.txt"))
    pre.save(model_dir / "preprocessor.json")
    _, persistence_source = persistence(splits["val"].head(1), panel)
    gain = booster.feature_importance("gain")
    metadata = {
        "lgbm_params": cfg.LGBM_PARAMS,
        "best_iteration": int(best_iter),
        "refit_on_train_val": refit,
        "final_model_fit_rows": int(len(fit_df)),
        "final_model_fit_range": [fit_df[cfg.TIME].min().isoformat(), fit_df[cfg.TIME].max().isoformat()],
        "split": {"val_start": cfg.VAL_START.isoformat(), "test_start": cfg.TEST_START.isoformat(), "rows": sizes},
        "feature_columns": pre.feature_columns,
        "dropped_columns": pre.dropped,
        "feature_importance_gain": dict(sorted(zip(pre.feature_columns, map(float, gain)), key=lambda kv: -kv[1])),
        "persistence_source": persistence_source,
        "validation_report": report,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    preds.to_csv(out_dir / cfg.PREDS_FILE, index=False)
    metrics.to_csv(out_dir / cfg.METRICS_FILE, index=False)

    summary = metrics[(metrics.icao != "ALL") & (metrics.h == "ALL")]
    print(summary.pivot_table(index=["split", "icao"], columns="model", values=["rmse", "mae"]).round(3))
    print(f"saved model to {model_dir}, predictions and metrics to {out_dir}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=cfg.HOURLY_FEATURES)
    ap.add_argument("--out-dir", type=Path, default=cfg.PRED_DIR)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-refit", action="store_true", help="score test with the train-only model")
    args = ap.parse_args()
    if args.validate_only:
        report = validate_hourly(load_hourly(args.data), load_airport_timezones())
        print(json.dumps(report, indent=2, default=str))
        check(report)
        print("validation passed")
        return
    run(args.data, args.out_dir, refit=not args.no_refit)


if __name__ == "__main__":
    main()
