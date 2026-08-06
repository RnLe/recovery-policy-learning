"""Environment adapter against the live pinned MiniGrid environment.

These tests are gate G0's executable evidence: the frozen observation/action
schema holds, resets are deterministic, and the golden replay digest pins the
exact world-generation behavior of the resolved dependency versions.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from grounded_recovery.config import load_and_validate
from grounded_recovery.world import (
    WorldError,
    WorldSession,
    replay_episode,
    trace_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_TRACE = REPO_ROOT / "tests" / "fixtures" / "golden_world_trace.json"


@pytest.fixture(scope="module")
def env_cfg():
    return load_and_validate(REPO_ROOT / "configs" / "pilot.yaml").environment


@pytest.fixture(scope="module")
def session(env_cfg):
    session = WorldSession(env_cfg)
    yield session
    session.close()


def test_registration_and_schema(session) -> None:
    result = session.reset(seed=7)
    assert result.image.shape == (7, 7, 3)
    assert result.image.dtype == np.uint8
    assert 0 <= result.direction <= 3
    assert result.mission.startswith("go to ")
    assert int(session.env.action_space.n) == 7
    assert int(session.env.unwrapped.max_steps) == 144


def test_reset_determinism_same_seed(env_cfg, session) -> None:
    first = session.reset(seed=42)
    fresh = WorldSession(env_cfg)
    second = fresh.reset(seed=42)
    third = session.reset(seed=42)  # same session, explicit reseed
    fresh.close()
    for other in (second, third):
        assert np.array_equal(first.image, other.image)
        assert first.direction == other.direction
        assert first.mission == other.mission


def test_scripted_replay_golden_trace(env_cfg) -> None:
    golden = json.loads(GOLDEN_TRACE.read_text())
    results = replay_episode(env_cfg, golden["seed"], golden["actions"])
    assert trace_digest(results) == golden["trace_digest"]
    assert results[-1].terminated == golden["success"]
    assert results[0].mission == golden["mission"]


def test_step_after_termination_rejected(env_cfg) -> None:
    golden = json.loads(GOLDEN_TRACE.read_text())
    with pytest.raises(WorldError, match="past termination"):
        replay_episode(env_cfg, golden["seed"], golden["actions"] + [2])


def test_out_of_set_action_rejected(session) -> None:
    session.reset(seed=3)
    with pytest.raises(WorldError, match="frozen action set"):
        session.step(5)


def test_step_before_reset_rejected(env_cfg) -> None:
    fresh = WorldSession(env_cfg)
    with pytest.raises(WorldError, match="before reset"):
        fresh.step(2)
    fresh.close()


def test_scenario_state_only_at_t0(session) -> None:
    session.reset(seed=11)
    state = session.scenario_state()
    assert state.mission == session.last_observation.mission
    assert len(state.grid_encoding) > 0
    session.step(2)
    with pytest.raises(WorldError, match="t=0"):
        session.scenario_state()


def test_max_steps_contract_enforced(env_cfg) -> None:
    wrong = dataclasses.replace(env_cfg, max_steps=99)
    fresh = WorldSession(wrong)
    with pytest.raises(WorldError, match="max_steps"):
        fresh.reset(seed=1)
    fresh.close()


def test_fingerprint_contents(session) -> None:
    session.reset(seed=1)
    fingerprint = session.fingerprint()
    assert fingerprint["env_id"] == session.cfg.env_id
    assert fingerprint["doors_open"] is True
    assert fingerprint["max_steps"] == 144
    assert fingerprint["packages"]["minigrid"] is not None
    assert fingerprint["babyai_done_actions_env_var"] is None
