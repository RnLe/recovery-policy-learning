"""Regenerate the golden environment/oracle fixtures.

Run explicitly with ``uv run python tests/fixtures/generate_fixtures.py``.
Overwriting these fixtures changes the pinned environment identity that the
golden tests protect, so regenerate them only deliberately (for example after
an intentional, reviewed environment or dependency version change).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from grounded_recovery.config import load_and_validate
from grounded_recovery.oracle import run_synchronized_episode
from grounded_recovery.world import WorldSession, trace_digest

FIXTURE_DIR = Path(__file__).resolve().parent
PILOT_YAML = FIXTURE_DIR.parents[1] / "configs" / "pilot.yaml"

WORLD_TRACE_SEED = 101
ORACLE_SEEDS = tuple(range(200, 220))


def actions_digest(actions: list[int]) -> str:
    return hashlib.sha256(bytes(actions)).hexdigest()


def main() -> None:
    cfg = load_and_validate(PILOT_YAML)
    session = WorldSession(cfg.environment)

    nominal = run_synchronized_episode(session, WORLD_TRACE_SEED, lambda t, rec: rec)
    executed = [transition.executed for transition in nominal.transitions]
    world_trace = {
        "environment_id": cfg.environment.env_id,
        "doors_open": cfg.environment.doors_open,
        "minigrid_version": importlib.metadata.version("minigrid"),
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "seed": WORLD_TRACE_SEED,
        "mission": nominal.mission,
        "actions": executed,
        "scenario_hash": nominal.scenario_hash,
        "trace_digest": trace_digest(nominal.observations),
        "success": nominal.success,
    }
    with open(FIXTURE_DIR / "golden_world_trace.json", "w", encoding="utf-8") as handle:
        json.dump(world_trace, handle, indent=2, sort_keys=True)
        handle.write("\n")

    episodes = []
    for seed in ORACLE_SEEDS:
        trace = run_synchronized_episode(session, seed, lambda t, rec: rec)
        episodes.append(
            {
                "seed": seed,
                "mission": trace.mission,
                "success": trace.success,
                "transitions": len(trace.transitions),
                "oracle_calls": trace.oracle_calls,
                "scenario_hash": trace.scenario_hash,
                "actions_digest": actions_digest(
                    [transition.executed for transition in trace.transitions]
                ),
            }
        )
    oracle_fixture = {
        "environment_id": cfg.environment.env_id,
        "doors_open": cfg.environment.doors_open,
        "minigrid_version": importlib.metadata.version("minigrid"),
        "episodes": episodes,
    }
    with open(FIXTURE_DIR / "golden_oracle_episodes.json", "w", encoding="utf-8") as handle:
        json.dump(oracle_fixture, handle, indent=2, sort_keys=True)
        handle.write("\n")

    session.close()
    print(f"wrote fixtures for {len(episodes)} oracle episodes and one world trace")


if __name__ == "__main__":
    main()
