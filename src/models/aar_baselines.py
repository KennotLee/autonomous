"""Baselines for AAR regression: persistence and historical average."""
import pandas as pd

from . import aar_config as cfg
from .aar_features import add_calendar


def issue_time(df: pd.DataFrame) -> pd.Series:
    """Prediction time t - h (the forecast issue time)."""
    return df[cfg.TIME] - pd.to_timedelta(df["h"], unit="h")


def persistence(df: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.Series, str]:
    """Arrivals in the last hour that had fully ended by the prediction time t - h.

    Uses cfg.PERSISTENCE_COLUMN if set. Otherwise looks up the realised
    arr_count of hour [floor(t - h) - 1h, floor(t - h)) in `panel`. That hour ends at or
    before the prediction time, so only past observations are used; this is asserted below.
    Hours absent from the panel give NaN (never filled with a guess).
    """
    if cfg.PERSISTENCE_COLUMN:
        if cfg.PERSISTENCE_COLUMN not in df:
            raise KeyError(f"PERSISTENCE_COLUMN {cfg.PERSISTENCE_COLUMN!r} not in data")
        return df[cfg.PERSISTENCE_COLUMN].astype("float64"), f"column {cfg.PERSISTENCE_COLUMN}"

    lag_hour = issue_time(df).dt.floor("h") - pd.Timedelta(hours=1)
    assert (lag_hour + pd.Timedelta(hours=1) <= issue_time(df)).all(), "persistence would read the future"
    observed = (
        panel.dropna(subset=[cfg.TARGET])
        .drop_duplicates(["icao", cfg.TIME])
        .set_index(["icao", cfg.TIME])[cfg.TARGET]
    )
    lookup = pd.MultiIndex.from_arrays([df["icao"], lag_hour])
    values = observed.reindex(lookup).to_numpy(dtype="float64")
    return pd.Series(values, index=df.index), "derived: panel arr_count of last completed hour before t - h"


class HistoricalAverage:
    """Mean arr_count by (icao, local hour, local day of week), fit on training rows only.

    Each (icao, hour) is counted once even though it appears for every horizon. Groups unseen
    in training fall back to (icao, local hour), then icao.
    """

    LEVELS = (["icao", "p2_local_hour", "p2_local_dow"], ["icao", "p2_local_hour"], ["icao"])

    def __init__(self, timezones: dict[str, str]):
        self.timezones = timezones
        self.tables: list[pd.Series] = []

    def fit(self, train: pd.DataFrame) -> "HistoricalAverage":
        hourly = add_calendar(
            train.dropna(subset=[cfg.TARGET]).drop_duplicates(["icao", cfg.TIME]), self.timezones
        )
        self.tables = [hourly.groupby(keys)[cfg.TARGET].mean() for keys in self.LEVELS]
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df = add_calendar(df, self.timezones)
        pred = pd.Series(float("nan"), index=df.index)
        for keys, table in zip(self.LEVELS, self.tables):
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
            pred = pred.fillna(pd.Series(table.reindex(idx).to_numpy(), index=df.index))
        return pred


class ScheduleBaseline:
    """sched_arr_count x (arrivals / scheduled) per airport, ratio fit on training rows only.

    The strongest simple baseline: "the schedule, minus the usual share of cancelled/diverted flights".
    """

    def __init__(self):
        self.ratio: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "ScheduleBaseline":
        hourly = train.dropna(subset=[cfg.TARGET]).drop_duplicates(["icao", cfg.TIME])
        g = hourly.groupby("icao")
        self.ratio = g[cfg.TARGET].sum() / g["sched_arr_count"].sum()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df["sched_arr_count"] * df["icao"].map(self.ratio)
