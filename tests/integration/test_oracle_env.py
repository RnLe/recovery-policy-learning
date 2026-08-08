"""Synchronized oracle against the live environment.

These tests are gate G0/G1 evidence: the pinned bot solves the task through
the synchronization wrapper, stays within the frozen action set, resynchronizes
after forced non-recommended actions, and receives exactly the executed action
sequence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from grounded_recovery import oracle as oracle_module
from grounded_recovery.config import load_and_validate
from grounded_recovery.oracle import load_bot_class, run_synchronized_episode
from grounded_recovery.world import WorldSession

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_EPISODES = REPO_ROOT / "tests" / "fixtures" / "golden_oracle_episodes.json"


@pytest.fixture(scope="module")
def cfg():
    return load_and_validate(REPO_ROOT / "configs" / "pilot.yaml")


@pytest.fixture(scope="module")
def session(cfg):
    session = WorldSession(cfg.environment)
    yield session
    session.close()


def test_nominal_oracle_matches_golden_episodes(session) -> None:
    golden = json.loads(GOLDEN_EPISODES.read_text())
    for episode in golden["episodes"]:
        trace = run_synchronized_episode(session, episode["seed"], lambda t, rec: rec)
        executed = bytes(transition.executed for transition in trace.transitions)
        assert trace.success, f"seed {episode['seed']} no longer solved by the oracle"
        assert trace.success == episode["success"]
        assert len(trace.transitions) == episode["transitions"]
        assert trace.oracle_calls == episode["oracle_calls"]
        assert trace.scenario_hash == episode["scenario_hash"]
        assert hashlib.sha256(executed).hexdigest() == episode["actions_digest"]


def test_one_oracle_call_per_active_step(session) -> None:
    trace = run_synchronized_episode(session, 205, lambda t, rec: rec)
    assert trace.oracle_calls == len(trace.transitions)


def test_resync_after_forced_action(cfg, session) -> None:
    # Force one corrupted transition at t=2 with the collection operator and
    # follow the oracle everywhere else: the episode must still succeed. This
    # is the mechanism recovery collection depends on.
    golden = json.loads(GOLDEN_EPISODES.read_text())
    mapping = cfg.perturbation.collection_operator.mapping
    checked = 0
    for episode in golden["episodes"]:
        if episode["transitions"] <= 3:
            continue
        trace = run_synchronized_episode(
            session,
            episode["seed"],
            lambda t, rec: mapping[rec] if t == 2 else rec,
        )
        assert trace.success, f"oracle failed to recover on seed {episode['seed']}"
        forced = trace.transitions[2]
        assert forced.executed == mapping[forced.recommended]
        assert forced.executed != forced.recommended
        checked += 1
    assert checked >= 10


class _RecordingBot:
    """Wraps the real bot and records every ``replan`` argument."""

    instances: list[_RecordingBot] = []

    def __init__(self, env: object) -> None:
        inner_class = load_bot_class("minigrid.utils.baby_ai_bot:BabyAIBot")
        self._inner = inner_class(env)
        self.replan_args: list[int | None] = []
        _RecordingBot.instances.append(self)

    def replan(self, action_taken: int | None = None) -> int:
        self.replan_args.append(action_taken)
        return int(self._inner.replan(action_taken))


def test_executed_not_proposed_threaded(cfg, session, monkeypatch) -> None:
    # After a forced action the next replan call must receive the action that
    # was actually executed, never the oracle's overridden recommendation.
    monkeypatch.setattr(oracle_module, "load_bot_class", lambda _: _RecordingBot)
    _RecordingBot.instances.clear()
    mapping = cfg.perturbation.collection_operator.mapping
    trace = run_synchronized_episode(
        session, 216, lambda t, rec: mapping[rec] if t == 2 else rec
    )
    bot = _RecordingBot.instances[-1]
    executed = [transition.executed for transition in trace.transitions]
    assert bot.replan_args == [None] + executed[:-1]
