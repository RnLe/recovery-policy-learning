"""Lab 3: policies, oracles, and where labels come from.

A policy is any rule from observations to actions. This lab measures a
spectrum of them on identical scenarios (uniformly random, a hand-written
wall follower, and the scripted BabyAI oracle) to establish how hard the task
is without learning and how good the teacher is. It then measures the fact the
study rests on (the oracle recovers after a forced off-plan action) and runs a
falsification attempt against its bookkeeping: lying about the executed
action, never informing it, and double-calling replan. On this movement-only
task none of that degrades the bot, because it replans from the live world state, so
the synchronization contract earns its keep as *accounting* rigor (a replan
call is the budgeted unit of supervision), not as fragility protection. The
lab closes by materializing what an "expert label" literally is: the oracle's
recommendation at a visited state.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX

from gr_foundations.common import (
    COLOR_BASE,
    COLOR_CAUTION,
    COLOR_NEUTRAL,
    COLOR_RECOVERY,
    LabPaths,
    derive_seed,
    export_typst_table,
    export_typst_values,
    prepare,
    save_figure,
    write_metrics,
    write_mini_report,
    write_table_csv,
)
from gr_foundations.gridart import WorldState, draw_state, state_snapshot
from gr_foundations.lab01_world import contract_environment
from grounded_recovery.config import EnvironmentConfig
from grounded_recovery.oracle import OracleSupportError, SynchronizedOracle
from grounded_recovery.world import StepResult, WorldSession

SPECTRUM_EPISODES = 300
SYNC_PAIRS = 200
TRAJECTORY_PANELS = 6

LEFT, RIGHT, FORWARD = 0, 1, 2

# Cells the agent can walk onto (open doors are handled separately).
_WALKABLE = {OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"], OBJECT_TO_IDX["goal"]}


class RandomPolicy:
    """Uniform over the frozen action set; one named RNG per episode."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, result: StepResult, time: int) -> int:
        return int(self._rng.integers(0, 3))


class WallFollowerPolicy:
    """Right-hand-rule maze walker reading only the observation.

    Note the one bit of internal state (``_pending_forward``): even this
    hand-written heuristic cannot be expressed as a pure function of single
    observations, a small echo of Lab 2's aliasing result.
    """

    def __init__(self) -> None:
        self._pending_forward = False

    @staticmethod
    def _passable(image: np.ndarray, x: int, y: int) -> bool:
        obj = int(image[x, y, 0])
        if obj in _WALKABLE:
            return True
        if obj == OBJECT_TO_IDX["door"]:
            return int(image[x, y, 2]) == STATE_TO_IDX["open"]
        return False

    def act(self, result: StepResult, time: int) -> int:
        image = result.image
        ahead = self._passable(image, 3, 5)  # agent sits at (3, 6) facing up
        right = self._passable(image, 4, 6)
        if self._pending_forward:
            self._pending_forward = False
            if ahead:
                return FORWARD
        if right:
            self._pending_forward = True
            return RIGHT
        if ahead:
            return FORWARD
        return LEFT


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    steps: int


def run_policy_episode(env_cfg: EnvironmentConfig, seed: int, policy) -> EpisodeOutcome:
    session = WorldSession(env_cfg)
    try:
        result = session.reset(seed)
        while not session.done:
            result = session.step(policy.act(result, session.time))
        return EpisodeOutcome(
            success=bool(result.terminated and result.reward > 0.0), steps=session.time
        )
    finally:
        session.close()


def run_oracle_episode(
    env_cfg: EnvironmentConfig, seed: int
) -> tuple[EpisodeOutcome, list[int]]:
    """Nominal oracle episode; returns the outcome and the executed actions."""
    session = WorldSession(env_cfg)
    try:
        result = session.reset(seed)
        oracle = SynchronizedOracle(session)
        last: int | None = None
        actions: list[int] = []
        while not session.done:
            recommendation = oracle.recommend(last, session.time)
            result = session.step(recommendation)
            actions.append(recommendation)
            last = recommendation
        outcome = EpisodeOutcome(
            success=bool(result.terminated and result.reward > 0.0), steps=session.time
        )
        return outcome, actions
    finally:
        session.close()


def measure_spectrum(
    env_cfg: EnvironmentConfig, n_episodes: int
) -> dict[str, dict[str, object]]:
    """Random, wall-follower, and oracle success on identical scenarios."""
    results: dict[str, dict[str, object]] = {}
    oracle_lengths: list[int] = []
    oracle_unsupported = 0
    rows: dict[str, list[EpisodeOutcome]] = {"random": [], "wall_follower": [], "oracle": []}
    for index in range(n_episodes):
        seed = derive_seed("lab03.oracle_eval", index)
        rows["random"].append(
            run_policy_episode(
                env_cfg, seed, RandomPolicy(derive_seed("lab03.random_policy", index))
            )
        )
        rows["wall_follower"].append(
            run_policy_episode(env_cfg, seed, WallFollowerPolicy())
        )
        try:
            outcome, _actions = run_oracle_episode(env_cfg, seed)
        except OracleSupportError:
            oracle_unsupported += 1
            continue
        rows["oracle"].append(outcome)
        oracle_lengths.append(outcome.steps)
    for name, outcomes in rows.items():
        successes = sum(o.success for o in outcomes)
        results[name] = {
            "episodes": len(outcomes),
            "successes": successes,
            "success_rate": successes / len(outcomes) if outcomes else 0.0,
            "mean_steps": float(np.mean([o.steps for o in outcomes])) if outcomes else 0.0,
        }
    results["oracle"]["unsupported_episodes"] = oracle_unsupported
    results["oracle"]["path_lengths"] = oracle_lengths
    return results


SYNC_PROTOCOLS = ("honest", "lied_to", "never_informed", "double_replan")


def run_protocol_episode(
    env_cfg: EnvironmentConfig, seed: int, forced_time: int, protocol: str
) -> dict[str, object]:
    """One forced off-proposal action at ``forced_time`` under one query protocol.

    ``honest`` and ``lied_to`` go through the study's ``SynchronizedOracle``
    (the lie: reporting the proposal instead of the executed action).
    ``never_informed`` and ``double_replan`` deliberately bypass the wrapper
    and drive the raw ``BabyAIBot`` in ways the wrapper structurally forbids,
    a falsification attempt against the synchronization contract.
    """
    from grounded_recovery.oracle import load_bot_class

    session = WorldSession(env_cfg)
    delivered = False
    try:
        result = session.reset(seed)
        if protocol in ("honest", "lied_to"):
            oracle = SynchronizedOracle(session)
            query = oracle.recommend
        else:
            bot = load_bot_class(env_cfg.bot_import)(session.env)

            def query(reported: int | None, _time: int) -> int:
                if protocol == "never_informed":
                    proposal = bot.replan(None)
                else:  # double_replan: two bookkeeping calls per step
                    proposal = bot.replan(reported)
                    proposal = bot.replan(int(proposal))
                if int(proposal) not in env_cfg.action_ids:
                    raise OracleSupportError(f"bot proposed {int(proposal)}")
                return int(proposal)

        reported: int | None = None
        while not session.done:
            time = session.time
            try:
                recommendation = query(reported, time)
            except OracleSupportError:
                return {"outcome": "support_error", "delivered": delivered}
            except Exception as error:  # a desynchronized bot may fail arbitrarily
                return {
                    "outcome": "bot_error",
                    "delivered": delivered,
                    "error": type(error).__name__,
                }
            if time == forced_time:
                executed = (recommendation + 1) % 3
                delivered = True
            else:
                executed = recommendation
            result = session.step(executed)
            reported = recommendation if protocol == "lied_to" else executed
        success = bool(result.terminated and result.reward > 0.0)
        return {"outcome": "success" if success else "failure", "delivered": delivered}
    finally:
        session.close()


def run_sync_experiment(env_cfg: EnvironmentConfig, n_pairs: int) -> dict[str, object]:
    """All four query protocols on identical (seed, forced-time) scenarios."""
    rng = np.random.default_rng(derive_seed("lab03.sync_experiment"))
    counts = {
        protocol: {"success": 0, "failure": 0, "support_error": 0, "bot_error": 0}
        for protocol in SYNC_PROTOCOLS
    }
    delivered_pairs = 0
    undelivered_pairs = 0
    for index in range(n_pairs):
        seed = derive_seed("lab03.sync_experiment", index + 1)
        forced_time = int(rng.integers(1, 9))
        outcomes = {
            protocol: run_protocol_episode(env_cfg, seed, forced_time, protocol)
            for protocol in SYNC_PROTOCOLS
        }
        if not outcomes["honest"]["delivered"]:
            undelivered_pairs += 1
            continue
        delivered_pairs += 1
        for protocol in SYNC_PROTOCOLS:
            counts[protocol][outcomes[protocol]["outcome"]] += 1
    rates = {
        protocol: (counts[protocol]["success"] / delivered_pairs if delivered_pairs else 0.0)
        for protocol in SYNC_PROTOCOLS
    }
    return {
        "pairs_planned": n_pairs,
        "pairs_delivered": delivered_pairs,
        "pairs_undelivered": undelivered_pairs,
        "counts": counts,
        "success_rates": rates,
    }


def _run_lengths(actions: list[int], names: tuple[str, ...] | list[str]) -> str:
    """Compact ``forward ×6, right ×1`` summary of an action stretch."""
    parts: list[str] = []
    index = 0
    while index < len(actions):
        run = 1
        while index + run < len(actions) and actions[index + run] == actions[index]:
            run += 1
        parts.append(f"{names[actions[index]]} ×{run}")
        index += run
    return ", ".join(parts)


def _render_trajectory(env_cfg: EnvironmentConfig, paths: LabPaths) -> list[list[object]]:
    """A labelled expert trajectory: states plus the oracle action at each one.

    The first panels are consecutive on purpose: the bot often turns in place
    to scan before committing to a move, and skipping steps here once made a
    single net turn look three steps late. Gaps between later panels state
    exactly which actions they skip.
    """
    seed = derive_seed("lab03.trajectory", 0)
    outcome, actions = run_oracle_episode(env_cfg, seed)
    total = len(actions)
    mid = (3 + total) // 2
    picks = sorted({t for t in (0, 1, 2, 3, mid, total) if 0 <= t <= total})
    session = WorldSession(env_cfg)
    label_rows: list[list[object]] = []
    try:
        session.reset(seed)
        states: dict[int, WorldState] = {}
        if 0 in picks:
            grid, pose, visible = state_snapshot(session)
            states[0] = WorldState(grid, pose, visible)
        for t, action in enumerate(actions):
            session.step(action)
            if t + 1 in picks:
                grid, pose, visible = state_snapshot(session)
                states[t + 1] = WorldState(grid, pose, visible)
    finally:
        session.close()
    for t, action in enumerate(actions):
        label_rows.append([t, env_cfg.action_names[action]])
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.4))
    for index, (axis, t) in enumerate(zip(axes.ravel(), picks, strict=False)):
        draw_state(axis, states[t])
        if t < total:
            caption = f"t={t} · oracle's label: `{env_cfg.action_names[actions[t]]}`"
        else:
            caption = f"t={t} · {'goal reached' if outcome.success else 'step limit'}"
        axis.set_title(caption, fontsize=11)
        # A gap to the next panel gets its skipped actions spelled out.
        if index + 1 < len(picks) and picks[index + 1] - t > 1:
            skipped = _run_lengths(actions[t : picks[index + 1]], env_cfg.action_names)
            note = "\n".join(
                textwrap.wrap(f"then t={t}…{picks[index + 1] - 1}: {skipped}", 34)
            )
            axis.text(
                0.5, -0.08, note,
                transform=axis.transAxes, ha="center", va="top", fontsize=11,
            )
    for axis in axes.ravel()[len(picks):]:
        axis.axis("off")
    fig.suptitle(
        "one expert demonstration = a sequence of (state, oracle action) labels\n"
        f"consecutive first steps ({_run_lengths(actions[:3], env_cfg.action_names)}): "
        "the oracle scans before it commits",
        fontsize=13,
    )
    save_figure(paths, fig, "labelled_trajectory.svg")
    return label_rows


def _render_spectrum(spectrum: dict[str, dict[str, object]], paths: LabPaths) -> None:
    fig, (left_axis, right_axis) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.0, 1.3]}
    )
    names = ["random", "wall_follower", "oracle"]
    colors = [COLOR_BASE, COLOR_CAUTION, COLOR_RECOVERY]
    rates = [spectrum[name]["success_rate"] for name in names]
    left_axis.bar(["random", "wall\nfollower", "oracle"], rates, color=colors)
    for index, rate in enumerate(rates):
        left_axis.text(index, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=11)
    left_axis.set_ylim(0, 1.12)
    left_axis.set_ylabel("success rate")
    left_axis.set_title(
        f"the policy spectrum (n={spectrum['oracle']['episodes']} scenarios)",
        fontsize=13,
    )
    right_axis.hist(
        spectrum["oracle"]["path_lengths"], bins=24, color=COLOR_NEUTRAL
    )
    right_axis.set_xlabel("oracle path length (steps)")
    right_axis.set_ylabel("episodes")
    right_axis.set_title("how long expert solutions are", fontsize=13)
    fig.tight_layout()
    save_figure(paths, fig, "policy_spectrum.svg")


def _render_sync(sync: dict[str, object], paths: LabPaths) -> None:
    fig, axis = plt.subplots(figsize=(8.6, 4.0))
    labels = [
        "synchronized\n(honest)",
        "lied to about\nthe executed action",
        "never informed\n(replan(None))",
        "double replan\nper step",
    ]
    rates = [sync["success_rates"][p] for p in SYNC_PROTOCOLS]
    colors = [COLOR_RECOVERY, COLOR_CAUTION, COLOR_CAUTION, COLOR_CAUTION]
    axis.bar(labels, rates, color=colors)
    for index, rate in enumerate(rates):
        axis.text(index, rate + 0.02, f"{rate:.1%}", ha="center", fontsize=11)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("success after one forced action")
    axis.tick_params(axis="x", labelsize=11)
    axis.set_title(
        "falsification attempt: recovery competence and bookkeeping misuse "
        f"(n={sync['pairs_delivered']} delivered scenarios each)",
        fontsize=13,
    )
    save_figure(paths, fig, "synchronization_experiment.svg")


def run(
    paths: LabPaths,
    *,
    force: bool,
    spectrum_episodes: int = SPECTRUM_EPISODES,
    sync_pairs: int = SYNC_PAIRS,
) -> dict[str, object]:
    prepare(paths, force=force)
    env_cfg = contract_environment(paths)

    spectrum = measure_spectrum(env_cfg, spectrum_episodes)
    sync = run_sync_experiment(env_cfg, sync_pairs)
    label_rows = _render_trajectory(env_cfg, paths)
    _render_spectrum(spectrum, paths)
    _render_sync(sync, paths)

    spectrum_rows = [
        [
            name,
            spectrum[name]["episodes"],
            f"{spectrum[name]['success_rate']:.1%}",
            f"{spectrum[name]['mean_steps']:.1f}",
        ]
        for name in ("random", "wall_follower", "oracle")
    ]
    export_typst_table(
        paths,
        "policy_spectrum",
        ["policy", "episodes", "success", "mean steps"],
        spectrum_rows,
    )
    write_table_csv(
        paths,
        "policy_spectrum.csv",
        ["policy", "episodes", "success_rate", "mean_steps"],
        spectrum_rows,
    )
    sync_rows = [
        [
            protocol,
            sync["counts"][protocol]["success"],
            sync["counts"][protocol]["failure"],
            sync["counts"][protocol]["support_error"],
            sync["counts"][protocol]["bot_error"],
        ]
        for protocol in SYNC_PROTOCOLS
    ]
    export_typst_table(
        paths,
        "synchronization_outcomes",
        ["protocol", "success", "failure", "support error", "bot error"],
        sync_rows,
    )
    write_table_csv(
        paths,
        "synchronization_outcomes.csv",
        ["protocol", "success", "failure", "support_error", "bot_error"],
        sync_rows,
    )
    write_table_csv(
        paths, "labelled_trajectory.csv", ["t", "oracle_action"], label_rows
    )
    export_typst_values(
        paths,
        "policy_facts",
        {
            "random-success": f"{spectrum['random']['success_rate']:.1%}",
            "wall-success": f"{spectrum['wall_follower']['success_rate']:.1%}",
            "oracle-success": f"{spectrum['oracle']['success_rate']:.1%}",
            "oracle-mean-steps": f"{spectrum['oracle']['mean_steps']:.1f}",
            "sync-honest-success": f"{sync['success_rates']['honest']:.1%}",
            "sync-lied-success": f"{sync['success_rates']['lied_to']:.1%}",
            "sync-never-success": f"{sync['success_rates']['never_informed']:.1%}",
            "sync-double-success": f"{sync['success_rates']['double_replan']:.1%}",
            "sync-pairs": str(sync["pairs_delivered"]),
        },
    )

    metrics = {
        "spectrum": {
            name: {k: v for k, v in stats.items() if k != "path_lengths"}
            for name, stats in spectrum.items()
        },
        "oracle_path_length_percentiles": {
            str(q): float(np.percentile(spectrum["oracle"]["path_lengths"], q))
            for q in (10, 50, 90)
        },
        "synchronization": sync,
    }
    metrics_hash = write_metrics(paths, metrics)

    write_mini_report(
        paths,
        question="What is a policy, what is the oracle, and what exactly is an expert label?",
        sections=[
            (
                "The policy spectrum",
                f"On {spectrum_episodes} identical scenarios: a uniformly random "
                f"policy succeeds in {spectrum['random']['success_rate']:.1%} of "
                "episodes, a genuinely measured surprise worth being honest "
                "about: the maze is small and 144 steps is a long time, so blind "
                "wandering does stumble onto the target occasionally (in "
                f"{spectrum['random']['mean_steps']:.0f} steps on average, versus "
                f"{spectrum['oracle']['mean_steps']:.1f} for the oracle). A "
                "hand-written right-hand wall follower reaches "
                f"{spectrum['wall_follower']['success_rate']:.1%}, and the "
                f"scripted oracle {spectrum['oracle']['success_rate']:.1%}. The "
                "reward signal is terminal-only, so what makes learning from "
                "reward hard here is not that success is unreachable but that a "
                "single end-of-episode bit must be attributed across up to 144 "
                "decisions, the credit-assignment framing Lab 4 makes precise. "
                "See `figures/policy_spectrum.svg`.",
            ),
            (
                "What the oracle is",
                "`BabyAIBot` (shipped with MiniGrid) is a scripted planner with "
                "privileged access: it reads the full grid, the true POMDP state "
                "Lab 2 showed the agent never observes, and maintains a subgoal "
                "stack it replans from at every step. It is not learned and not "
                "part of the policy; it exists to answer one question at any "
                "visited state: *what should be done here?* That answer is an "
                "expert label.",
            ),
            (
                "Recovery competence, and a falsification attempt",
                "Two separate questions were measured on identical scenarios "
                f"({sync['pairs_delivered']} episodes each, one forced "
                "off-proposal action at a random early step). First, the fact "
                "the whole study rests on: an honestly informed oracle *recovers* "
                f"{sync['success_rates']['honest']:.1%} success after the "
                "deviation, which is what makes it able to label learner-visited "
                "off-path states (Lab 6). Second, we tried to break the bot's "
                "bookkeeping three ways: lying about the executed action "
                f"({sync['success_rates']['lied_to']:.1%}), never informing it "
                f"({sync['success_rates']['never_informed']:.1%}), and calling "
                f"`replan` twice per step "
                f"({sync['success_rates']['double_replan']:.1%}). None of it "
                "degrades this task: `BabyAIBot` holds a live reference to the "
                "environment and replans from the *true* world state, and pure "
                "navigation barely uses the `action_taken` bookkeeping (it "
                "matters for pickup/drop/toggle subgoals, which the frozen "
                "action set excludes). The honest conclusion: on this task the "
                "synchronization contract is not fragility protection, it is "
                "*accounting* protection. An oracle query is the unit of "
                "supervision the study budgets and ledgers, so \"exactly one "
                "replan per executed step\" is what makes \"N labels\" a "
                "well-defined, auditable quantity.",
            ),
            (
                "Where labels come from",
                "`figures/labelled_trajectory.svg` shows one expert episode as "
                "the learner will consume it: a sequence of (observation, oracle "
                "action) pairs. A *demonstration* is nothing more than this "
                "sequence collected along the oracle's own path; a *recovery "
                "label* (Lab 6) is the same query issued at a state the learner "
                "reached instead.",
            ),
            (
                "Bridge to the study",
                "The study's collectors and evaluators all drive episodes through "
                "one shared loop (`run_synchronized_episode`) that threads the "
                "executed action back into the oracle, the honest protocol "
                "above. Labels only ever enter datasets through explicit budget "
                "accounting (`revealed_targets`), which is what makes the later "
                "arm comparison fair (Lab 7).",
            ),
        ],
    )

    return {
        "random_success": f"{spectrum['random']['success_rate']:.1%}",
        "wall_follower_success": f"{spectrum['wall_follower']['success_rate']:.1%}",
        "oracle_success": f"{spectrum['oracle']['success_rate']:.1%}",
        "forced_action_recovery": f"{sync['success_rates']['honest']:.1%}",
        "misuse_protocols_degraded": any(
            sync["success_rates"][p] < sync["success_rates"]["honest"]
            for p in ("lied_to", "never_informed", "double_replan")
        ),
        "metrics_hash": metrics_hash,
    }
