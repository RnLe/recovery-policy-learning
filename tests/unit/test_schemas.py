"""Record schema serialization and scenario-hash semantics (no environment)."""

from __future__ import annotations

import json

from grounded_recovery.schemas import (
    ManifestEntry,
    canonical_scenario_hash,
    manifest_entry_from_json,
    manifest_entry_to_json,
)
from grounded_recovery.world import ScenarioState


def tiny_state(mission: str = "go to the red ball") -> ScenarioState:
    grid = tuple(
        tuple(tuple((x + y + c) % 4 for c in range(3)) for y in range(3)) for x in range(3)
    )
    return ScenarioState(
        env_id="BabyAI-GoToObjMazeS4-v0",
        grid_encoding=grid,
        agent_pos=(1, 2),
        agent_dir=3,
        mission=mission,
    )


def test_scenario_hash_deterministic_and_value_based() -> None:
    # The hash must be a function of the world content, not object identity.
    assert canonical_scenario_hash(tiny_state()) == canonical_scenario_hash(tiny_state())


def test_scenario_hash_sensitive_to_every_field() -> None:
    import dataclasses

    base = tiny_state()
    base_hash = canonical_scenario_hash(base)
    variants = [
        dataclasses.replace(base, mission="go to the blue key"),
        dataclasses.replace(base, agent_dir=0),
        dataclasses.replace(base, agent_pos=(2, 2)),
        dataclasses.replace(base, env_id="other-env"),
    ]
    for variant in variants:
        assert canonical_scenario_hash(variant) != base_hash


def test_manifest_entry_json_roundtrip() -> None:
    entry = ManifestEntry(
        split_name="operator_preflight",
        ordinal=7,
        environment_id="BabyAI-GoToObjMazeS4-v0",
        environment_seed=12345,
        canonical_scenario_hash="ab" * 32,
        mission="go to the red ball",
        nominal_oracle_path_length=13,
        perturbation_family=None,
        scheduled_intervention_times=(4,),
        manifest_version=1,
    )
    row = manifest_entry_to_json(entry)
    json.dumps(row)  # must be JSON-serializable as-is
    assert manifest_entry_from_json(row) == entry
