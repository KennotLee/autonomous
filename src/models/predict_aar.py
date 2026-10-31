"""Inference for the dashboard.

    from src.models.predict_aar import predict_aar
    preds = predict_aar(icao="KATL", h=6, start="2025-03-01", end="2025-03-08")

pred_arr_count is an AAR PROXY (expected realised arrivals in the UTC hour), with a p10-p90 band.
"""
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from . import aar_config as cfg
from .aar_context import add_context
from .aar_data import load_hourly, normalise
from .aar_features import AARPreprocessor
from .train_aar import predict_with_interval

OUTPUT_COLUMNS = cfg.PREDS_COLUMNS  # icao, target_hour_utc, h, pred_arr_count, p10, p90


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@lru_cache(maxsize=4)
def load_aar_model(model_dir: str = str(cfg.PRED_DIR / cfg.MODEL_SUBDIR)) -> tuple[dict[str, lgb.Booster], AARPreprocessor]:
    model_dir = Path(model_dir)
    files = {"point": "model.txt", **{k: f"model_{k}.txt" for k in cfg.QUANTILES}}
    missing = [f for f in files.values() if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {model_dir}; run python -m src.models.train_aar")
    models = {k: lgb.Booster(model_file=str(model_dir / f)) for k, f in files.items()}
    return models, AARPreprocessor.load(model_dir / "preprocessor.json")


def predict_aar(
    features: pd.DataFrame | None = None,
    *,
    icao: str | list[str] | None = None,
    h: int | list[int] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    model_dir: Path | str = cfg.PRED_DIR / cfg.MODEL_SUBDIR,
    data_path: Path | str = cfg.PANEL,
) -> pd.DataFrame:
    """Predicted arrivals per hour (AAR proxy) with p10/p90, for rows of the hourly feature panel.

    features: rows shaped like the P2 panel (aar_dataset.py). If None, the panel is loaded.
    icao / h / start / end: optional filters; start and end are inclusive UTC times on target_hour_utc.
    Returns icao, target_hour_utc, h, pred_arr_count, p10, p90 (arrivals/hour, >= 0), sorted by time.

    Only inputs known at t - h are used (see preprocessor.json "provenance"); realised arrivals,
    outcome and oracle_* columns in `features` are ignored. Neighbouring-hour schedule inputs are
    built from all rows before filtering, so pass `features` covering at least 2 hours either side
    of the hours you want; missing ones become NaN.
    """
    df = load_hourly(data_path) if features is None else normalise(features)
    df = add_context(df, df)
    if icao is not None:
        df = df[df["icao"].isin([icao] if isinstance(icao, str) else icao)]
    if h is not None:
        df = df[df["h"].isin([h] if isinstance(h, int) else h)]
    if start is not None:
        df = df[df[cfg.TIME] >= _utc(start)]
    if end is not None:
        df = df[df[cfg.TIME] <= _utc(end)]
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    models, pre = load_aar_model(str(model_dir))
    out = df[["icao", cfg.TIME, "h"]].join(predict_with_interval(models, pre.transform(df)))
    return out.sort_values([cfg.TIME, "icao", "h"]).reset_index(drop=True)[OUTPUT_COLUMNS]
