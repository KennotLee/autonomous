"""RMSE / MAE overall, per airport, per horizon and per airport x horizon."""
import numpy as np
import pandas as pd

ALL = "ALL"


def _scores(y: pd.Series, p: pd.Series) -> dict:
    ok = y.notna() & p.notna()
    err = (p[ok] - y[ok]).to_numpy()
    if not len(err):
        return {"n": 0, "rmse": np.nan, "mae": np.nan}
    return {"n": int(len(err)), "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def evaluate(preds: pd.DataFrame, model_cols: dict[str, str], y_col: str = "y_true",
             weather_col: str | None = None) -> pd.DataFrame:
    """Long table: split, model, icao, h, weather, n, rmse, mae. "ALL" marks an aggregated level.

    If weather_col is given, extra rows score each weather category per airport (h = ALL).
    Rows where a model has no prediction (e.g. persistence for an hour absent from the panel)
    are skipped for that model only, so compare n across models.
    """
    rows = []
    for split, s in preds.groupby("split", sort=False):
        groupings = [
            ((ALL, ALL, ALL), s),
            *(((icao, ALL, ALL), g) for icao, g in s.groupby("icao")),
            *(((ALL, int(h), ALL), g) for h, g in s.groupby("h")),
            *(((icao, int(h), ALL), g) for (icao, h), g in s.groupby(["icao", "h"])),
        ]
        if weather_col:
            groupings += [((ALL, ALL, w), g) for w, g in s.groupby(weather_col)]
            groupings += [((icao, ALL, w), g) for (icao, w), g in s.groupby(["icao", weather_col])]
        for (icao, h, w), g in groupings:
            for model, col in model_cols.items():
                rows.append({"split": split, "model": model, "icao": icao, "h": h, "weather": w,
                             **_scores(g[y_col], g[col])})
    return pd.DataFrame(rows)
