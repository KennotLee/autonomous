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


def evaluate(preds: pd.DataFrame, model_cols: dict[str, str], y_col: str = "y_true") -> pd.DataFrame:
    """Long table: split, model, icao, h, n, rmse, mae. "ALL" marks an aggregated level.

    Rows where a model has no prediction (e.g. persistence for an hour absent from the panel)
    are skipped for that model only, so compare n across models.
    """
    rows = []
    for split, s in preds.groupby("split", sort=False):
        groupings = [
            ((ALL, ALL), s),
            *(((icao, ALL), g) for icao, g in s.groupby("icao")),
            *(((ALL, int(h)), g) for h, g in s.groupby("h")),
            *(((icao, int(h)), g) for (icao, h), g in s.groupby(["icao", "h"])),
        ]
        for (icao, h), g in groupings:
            for model, col in model_cols.items():
                rows.append({"split": split, "model": model, "icao": icao, "h": h, **_scores(g[y_col], g[col])})
    return pd.DataFrame(rows)
