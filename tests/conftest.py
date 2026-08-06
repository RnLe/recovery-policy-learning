"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_YAML = REPO_ROOT / "configs" / "pilot.yaml"

TINY_SPLIT_COUNTS = {
    "base": 3,
    "collection": 3,
    "validation": 2,
    "operator_preflight": 4,
    "test_candidate": 3,
    "difficulty_shift": 2,
    "expert_diagnostic": 2,
    "visualization": 2,
}


@pytest.fixture(scope="session")
def tiny_config_factory(tmp_path_factory):
    """Factory writing structurally valid configs with miniature scenario counts."""

    def _make(**overrides: object) -> Path:
        with open(PILOT_YAML, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["data"]["split_counts"] = dict(TINY_SPLIT_COUNTS)
        raw["perturbation"]["preflight_episodes_per_family"] = TINY_SPLIT_COUNTS[
            "operator_preflight"
        ]
        for dotted, value in overrides.items():
            section = raw
            *parents, leaf = dotted.split(".")
            for parent in parents:
                section = section[parent]
            section[leaf] = value
        path = tmp_path_factory.mktemp("tiny-config") / "config.yaml"
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        return path

    return _make


@pytest.fixture()
def make_tiny_config(tiny_config_factory):
    return tiny_config_factory
