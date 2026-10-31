"""Load config/airports.yaml. All airport-specific values live there (no hardcoded airports in src/)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "airports.yaml"
RAW_CSV = REPO / "data" / "raw" / "Flight_Data_With_Weather_Final.csv"
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
KMH_PER_KT = 1.852


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hub_airports(cfg: dict, icaos: list[str] | None = None) -> dict:
    """Return {icao: airport_cfg} for the requested ICAOs (default: all configured airports)."""
    aps = cfg["airports"]
    icaos = icaos or list(aps)
    missing = [i for i in icaos if i not in aps]
    if missing:
        raise KeyError(f"ICAO(s) not in config/airports.yaml: {missing}")
    return {i: aps[i] for i in icaos}
