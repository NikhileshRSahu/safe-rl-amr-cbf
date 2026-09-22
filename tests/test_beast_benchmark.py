import json
from dataclasses import asdict
from pathlib import Path

from benchmark.beast_classical import BeastORCAConfig
from benchmark.beast_config import (
    DEV_SEEDS_BY_N,
    VALIDATION_SEEDS_BY_N,
    TEST_SEEDS_BY_N,
    load_beast_config,
    save_beast_config,
)


def test_seed_sets_are_pairwise_disjoint():
    for n in (2, 4, 6):
        d = set(DEV_SEEDS_BY_N[n])
        v = set(VALIDATION_SEEDS_BY_N[n])
        t = set(TEST_SEEDS_BY_N[n])
        assert not d & v
        assert not d & t
        assert not v & t


def test_test_seeds_preserve_existing_protocol():
    assert TEST_SEEDS_BY_N[2] == tuple(range(5200, 5230))
    assert TEST_SEEDS_BY_N[4] == tuple(range(5400, 5430))
    assert TEST_SEEDS_BY_N[6] == tuple(range(5600, 5630))


def test_config_roundtrip(tmp_path: Path):
    cfg = BeastORCAConfig(
        time_horizon=3.0,
        command_speed_samples=9,
        arc_horizon=1.0,
    )
    p = tmp_path / "cfg.json"
    save_beast_config(cfg, p)
    loaded = load_beast_config(p)
    assert asdict(loaded) == asdict(cfg)


def test_config_rejects_unknown_field(tmp_path: Path):
    cfg = asdict(BeastORCAConfig())
    cfg["surprise"] = 123
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg))
    try:
        load_beast_config(p)
    except ValueError as exc:
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("unknown field should fail")
