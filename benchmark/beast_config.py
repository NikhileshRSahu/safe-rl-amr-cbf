from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path

from benchmark.beast_classical import BeastORCAConfig


DEV_SEEDS_BY_N = {
    2: tuple(range(2200, 2210)),
    4: tuple(range(2400, 2410)),
    6: tuple(range(2600, 2610)),
}

VALIDATION_SEEDS_BY_N = {
    2: tuple(range(3200, 3215)),
    4: tuple(range(3400, 3415)),
    6: tuple(range(3600, 3615)),
}

TEST_SEEDS_BY_N = {
    2: tuple(range(5200, 5230)),
    4: tuple(range(5400, 5430)),
    6: tuple(range(5600, 5630)),
}


def save_beast_config(config: BeastORCAConfig, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(config), indent=2, sort_keys=True))


def load_beast_config(path) -> BeastORCAConfig:
    data = json.loads(Path(path).read_text())
    required = {f.name for f in fields(BeastORCAConfig)}
    given = set(data)
    unknown = given - required
    missing = required - given
    if unknown:
        raise ValueError(f"unknown BeastORCAConfig fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing BeastORCAConfig fields: {sorted(missing)}")
    return BeastORCAConfig(**data)
