"""Configuration loading, validation, and contract hashing.

These tests protect the contract layer: every scientific promise maps to a
resolved config field, so silently ignored keys, accepted placeholders under
FROZEN status, or non-derangement operators would invalidate downstream
claims before any data is collected.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from grounded_recovery import config as config_module
from grounded_recovery.config import (
    ConfigError,
    contract_hash,
    load_and_validate,
    load_config,
    unresolved_fields,
    validate_config,
)

PILOT_YAML = Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml"

# Fields that may carry the PILOT_TO_FREEZE placeholder during the pilot.
PLACEHOLDER_FIELDS = (
    ("data", "n0"),
    ("data", "b"),
    ("data", "k"),
    ("data", "h"),
    ("perturbation", "collection_time_set"),
    ("perturbation", "unseen_time_set"),
    ("training", "new_targets_per_update"),
    ("training", "updates_per_round"),
    ("evaluation", "desired_interval_half_width"),
    ("evaluation", "r_target"),
    ("evaluation", "r_max"),
)


def pilot_dict() -> dict:
    with open(PILOT_YAML, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle)
    return path


def test_pilot_yaml_loads_and_validates() -> None:
    # After the pilot phase all values are resolved in place.
    cfg = load_and_validate(PILOT_YAML)
    assert cfg.study.status == "PILOT"
    assert cfg.environment.action_ids == (0, 1, 2)
    assert unresolved_fields(cfg) == ()


def test_unresolved_fields_reports_every_placeholder(tmp_path: Path) -> None:
    raw = pilot_dict()
    for section, key in PLACEHOLDER_FIELDS:
        raw[section][key] = config_module.PILOT_TO_FREEZE
    cfg = load_config(write_config(tmp_path, raw))
    assert unresolved_fields(cfg) == tuple(
        f"{section}.{key}" for section, key in PLACEHOLDER_FIELDS
    )


def test_contract_hash_is_stable_across_loads() -> None:
    first = contract_hash(load_config(PILOT_YAML))
    second = contract_hash(load_config(PILOT_YAML))
    assert first == second
    assert len(first) == 64


def test_contract_hash_changes_on_field_change() -> None:
    cfg = load_config(PILOT_YAML)
    changed = dataclasses.replace(
        cfg, seeds=dataclasses.replace(cfg.seeds, root_seed=cfg.seeds.root_seed + 1)
    )
    assert contract_hash(cfg) != contract_hash(changed)


def test_unknown_key_rejected_with_path(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["environment"]["surprise"] = 1
    with pytest.raises(ConfigError, match=r"environment.*surprise"):
        load_config(write_config(tmp_path, raw))


def test_missing_key_rejected_with_path(tmp_path: Path) -> None:
    raw = pilot_dict()
    del raw["training"]["learning_rate"]
    with pytest.raises(ConfigError, match=r"training.*learning_rate"):
        load_config(write_config(tmp_path, raw))


def test_frozen_status_rejects_placeholders(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["study"]["status"] = "FROZEN"
    raw["data"]["n0"] = config_module.PILOT_TO_FREEZE
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="FROZEN"):
        validate_config(cfg)


def test_placeholder_in_non_optional_field_rejected(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["training"]["learning_rate"] = config_module.PILOT_TO_FREEZE
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(write_config(tmp_path, raw))


def test_non_derangement_rejected(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["perturbation"]["collection_operator"]["mapping"] = [0, 1, 2]
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="fixed point"):
        validate_config(cfg)


def test_partial_fixed_point_rejected(tmp_path: Path) -> None:
    # A transposition of a three-element set necessarily leaves a fixed point.
    raw = pilot_dict()
    raw["perturbation"]["collection_operator"]["mapping"] = [1, 0, 2]
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="fixed point"):
        validate_config(cfg)


def test_operator_overlap_rejected(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["perturbation"]["unseen_operator"]["mapping"] = [1, 2, 0]
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="distinct"):
        validate_config(cfg)


def test_action_set_mismatch_rejected(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["perturbation"]["collection_operator"]["mapping"] = [1, 2, 3]
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="permutation"):
        validate_config(cfg)


def test_b_mod_k_rule(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["data"]["b"] = 10
    raw["data"]["k"] = 3
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="divisible"):
        validate_config(cfg)
    raw["data"]["k"] = 5
    validate_config(load_config(write_config(tmp_path, raw)))


def test_sequence_cap_must_cover_prefix_plus_target(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["training"]["max_sequence_length"] = raw["training"]["max_context_prefix"]
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError, match="max_sequence_length"):
        validate_config(cfg)


def test_validation_reports_all_errors_at_once(tmp_path: Path) -> None:
    raw = pilot_dict()
    raw["perturbation"]["collection_operator"]["mapping"] = [0, 1, 2]
    raw["training"]["learning_rate"] = -1.0
    cfg = load_config(write_config(tmp_path, raw))
    with pytest.raises(ConfigError) as excinfo:
        validate_config(cfg)
    message = str(excinfo.value)
    assert "fixed point" in message
    assert "learning_rate" in message
