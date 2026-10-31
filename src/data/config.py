"""Load config/airports.yaml. All airport-specific values live there (no hardcoded airports in src/)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "airports.yaml"
RAW_CSV = REPO / "data" / "raw" / "Flight_Data_With_Weather_Final.csv"
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hub_airports(cfg: dict, codes: list[str] | None = None) -> dict:
    """Return {icao: airport_cfg} for the requested codes (IATA or ICAO; default: all configured)."""
    aps = cfg["airports"]
    if not codes:
        return dict(aps)
    by_iata = {v["iata"]: k for k, v in aps.items()}
    out = {}
    for c in codes:
        c = c.upper()
        icao = c if c in aps else by_iata.get(c)
        if icao is None:
            raise KeyError(f"Airport {c!r} not in config/airports.yaml (add iata, timezone, runway_heading_deg)")
        out[icao] = aps[icao]
    return out


def other_icao(iata: str, state: str | None, cfg: dict) -> str:
    """Best-effort ICAO for a non-configured airport: overrides, else P+IATA (AK/HI/GU), else K+IATA."""
    by_iata = {v["iata"]: k for k, v in cfg["airports"].items()}
    if iata in by_iata:
        return by_iata[iata]
    ov = cfg.get("iata_icao_overrides", {}) or {}
    if iata in ov:
        return ov[iata]
    return ("P" if state in ("AK", "HI", "GU") else "K") + iata
