"""Baselines for AAR regression: persistence and historical average."""
import pandas as pd

from . import aar_config as cfg
from .aar_features import add_calendar


def issue_time(df: pd.DataFrame) -> pd.Series:
    """Latest instant whose information may be used for each row."""
    if cfg.OBS_TIME in df:
        return df[cfg.OBS_TIME]
    return df[cfg.TIME] - pd.to_timedelta(df["h"], unit="h")


def persistence(df: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.Series, str]:
    """Arrivals in the last hour that had fully ended by each row's issue time.

    Uses P1's column if cfg.PERSISTENCE_COLUMN is set. Otherwise looks up the realised
    arr_count of hour [floor(issue) - 1h, floor(issue)) in `panel`. That hour ends at or
    before the issue time, so only past observations are used; this is asserted below.
    Hours absent from the panel give NaN (never filled with a guess).
    """
    if cfg.PERSISTENCE_COLUMN:
        if cfg.PERSISTENCE_COLUMN not in df:
            raise KeyError(f"PERSISTENCE_COLUMN {cfg.PERSISTENCE_COLUMN!r} not in data")
        return df[cfg.PERSISTENCE_COLUMN].astype("float64"), f"P1 column {cfg.PERSISTENCE_COLUMN}"

    lag_hour = issue_time(df).dt.floor("h") - pd.Timedelta(hours=1)
    assert (lag_hour + pd.Timedelta(hours=1) <= issue_time(df)).all(), "persistence would read the future"
    observed = (
        panel.dropna(subset=[cfg.TARGET])
        .drop_duplicates(["icao", cfg.TIME])
        .set_index(["icao", cfg.TIME])[cfg.TARGET]
    )
    lookup = pd.MultiIndex.from_arrays([df["icao"], lag_hour])
    values = observed.reindex(lookup).to_numpy(dtype="float64")
    return pd.Series(values, index=df.index), "derived: panel arr_count of last completed hour before feature_obs_time"


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
