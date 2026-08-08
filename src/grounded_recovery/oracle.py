"""Synchronized scripted oracle and the single authoritative transition loop.

``SynchronizedOracle`` wraps one ``BabyAIBot`` instance. Its only query method
is ``recommend(last_executed, step_index)``, which must be called exactly once
for every active simulator step and must receive the action actually executed
on the preceding transition, never the policy's proposal. Passing anything
else would silently desynchronize the bot's plan from the true world state and
corrupt every label derived from it.

``run_synchronized_episode`` is the one transition helper shared by all
collectors and evaluators, so proposed, recommended, and executed actions
cannot be confused between call sites.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

from grounded_recovery.schemas import canonical_scenario_hash
from grounded_recovery.world import StepResult, WorldSession


class OracleSyncError(RuntimeError):
    """The oracle was queried out of lockstep with the environment."""


class OracleSupportError(RuntimeError):
    """An action outside the frozen policy action set reached the oracle contract."""


def load_bot_class(bot_import: str) -> type:
    module_name, _, class_name = bot_import.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"bot_import must be 'module.path:ClassName', got {bot_import!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class SynchronizedOracle:
    """Exactly-once-per-active-step adapter around one BabyAIBot."""

    def __init__(self, session: WorldSession) -> None:
        if session.time != 0 or session.done:
            raise OracleSyncError("the oracle must be constructed on a freshly reset session")
        bot_class = load_bot_class(session.cfg.bot_import)
        self._bot = bot_class(session.env)
        self._session = session
        self._action_ids = frozenset(session.cfg.action_ids)
        self.next_step = 0
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def recommend(self, last_executed: int | None, step_index: int) -> int:
        """The oracle's recommendation for the current pre-action context."""
        if self._session.done:
            raise OracleSyncError("oracle queried after episode termination")
        if step_index != self.next_step:
            raise OracleSyncError(
                f"oracle called for step {step_index}, expected {self.next_step} "
                "(double or skipped call)"
            )
        if (step_index == 0) != (last_executed is None):
            raise OracleSyncError(
                "last_executed must be None exactly at step 0 "
                f"(got {last_executed!r} at step {step_index})"
            )
        if last_executed is not None and int(last_executed) not in self._action_ids:
            raise OracleSupportError(
                f"executed action {last_executed!r} is outside the frozen action set"
            )
        recommendation = int(self._bot.replan(last_executed))
        if recommendation not in self._action_ids:
            raise OracleSupportError(
                f"oracle recommended action {recommendation} outside the frozen action set "
                f"{sorted(self._action_ids)}"
            )
        self.next_step += 1
        self._calls += 1
        return recommendation


@dataclass(frozen=True)
class Transition:
    """One executed simulator transition and the actions surrounding it."""

    t: int
    recommended: int
    proposed: int | None
    executed: int
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeTrace:
    """Complete synchronized episode record for collectors and audits."""

    seed: int
    scenario_hash: str
    mission: str
    observations: tuple[StepResult, ...]
    transitions: tuple[Transition, ...]
    success: bool
    truncated: bool
    stopped_early: bool
    oracle_calls: int


def run_synchronized_episode(
    session: WorldSession,
    seed: int,
    choose_executed: Callable[[int, int], int],
    stop_after_step: Callable[[int], bool] | None = None,
) -> EpisodeTrace:
    """Reset, then run oracle-synchronized transitions until the episode ends.

    ``choose_executed(t, recommended)`` selects the executed action for step
    ``t``; the identity function reproduces the nominal oracle trajectory.
    ``stop_after_step(completed_transitions)`` may stop collection early (for
    exact label budgets); the trace then carries ``stopped_early=True``.
    The next oracle query always receives the executed action, so the bot
    stays synchronized even when the executed action differs from its
    recommendation.
    """
    reset_result = session.reset(seed)
    scenario_hash = canonical_scenario_hash(session.scenario_state())
    oracle = SynchronizedOracle(session)

    observations: list[StepResult] = [reset_result]
    transitions: list[Transition] = []
    last_executed: int | None = None
    stopped_early = False

    while not session.done:
        t = session.time
        recommended = oracle.recommend(last_executed, t)
        executed = int(choose_executed(t, recommended))
        result = session.step(executed)
        observations.append(result)
        transitions.append(
            Transition(
                t=t,
                recommended=recommended,
                proposed=None,
                executed=executed,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
            )
        )
        last_executed = executed
        if stop_after_step is not None and stop_after_step(len(transitions)):
            stopped_early = not session.done
            break

    final = observations[-1]
    return EpisodeTrace(
        seed=int(seed),
        scenario_hash=scenario_hash,
        mission=reset_result.mission,
        observations=tuple(observations),
        transitions=tuple(transitions),
        success=final.terminated,
        truncated=final.truncated,
        stopped_early=stopped_early,
        oracle_calls=oracle.calls,
    )
