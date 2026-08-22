"""Lab 2: decision processes, from MDP to POMDP.

The task is formalized as a partially observable Markov decision process, and
this lab shows *why* the "partially" matters by measurement: it collects states
from oracle rollouts, finds pairs of genuinely different world states that emit
byte-identical observations (perceptual aliasing), and checks whether the
oracle's optimal action differs between them. Whenever it does, no memoryless
policy, however good, can act optimally from single observations, which is
the empirical case for the recurrent memory built in Labs 4 and 5. A fully
observable alternative (MiniGrid's ``FullyObsWrapper``) is rendered for
contrast and deliberately rejected.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT

from gr_foundations.common import (
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
from gr_foundations.gridart import (
    WorldState,
    draw_grid,
    draw_plane,
    draw_state,
    state_snapshot,
)
from gr_foundations.lab01_world import OBJECT_GLYPHS, contract_environment
from grounded_recovery.config import EnvironmentConfig
from grounded_recovery.oracle import OracleSupportError, OracleSyncError, SynchronizedOracle
from grounded_recovery.world import WorldSession

ROLLOUT_EPISODES = 300

POMDP_MAPPING_ROWS: list[list[str]] = [
    ["S (state)", "full grid, agent position and direction, mission",
     "environment internals; ScenarioState at t=0"],
    ["A (actions)", "frozen set {left, right, forward}", "EnvironmentConfig.action_ids"],
    ["T (transition)", "deterministic grid dynamics", "WorldSession.step"],
    ["R (reward)", "sparse: nonzero only on mission success", "StepResult.reward"],
    ["Omega (observations)", "egocentric occluded 7x7x3 crop + direction + mission",
     "StepResult.image / .direction / .mission"],
    ["O (observation fn)", "deterministic crop with occlusion", "MiniGrid gen_obs"],
    ["H (horizon)", "hard cap of max_steps", "EnvironmentConfig.max_steps"],
]


@dataclass(frozen=True)
class StateRecord:
    """One visited state: what the agent saw, where it truly was, and the label."""

    observation_key: str
    episode_index: int
    t: int
    position: tuple[int, int]
    agent_dir: int
    recommendation: int
    image_bytes: bytes


def observation_key(image: np.ndarray, direction: int, mission: str) -> str:
    digest = hashlib.sha256()
    digest.update(image.tobytes())
    digest.update(bytes([direction]))
    digest.update(mission.encode("utf-8"))
    return digest.hexdigest()


def collect_states(
    env_cfg: EnvironmentConfig, n_episodes: int
) -> tuple[list[StateRecord], dict[int, list[int]], dict[str, int]]:
    """Nominal oracle rollouts; every visited pre-action state becomes a record.

    This is the same explicit loop the study's ``run_synchronized_episode``
    implements, written out so the synchronization contract is visible:
    exactly one ``recommend`` per active step, always fed the executed action.
    """
    records: list[StateRecord] = []
    actions_by_episode: dict[int, list[int]] = {}
    counters = {"episodes": 0, "successes": 0, "truncated": 0, "oracle_unsupported": 0}

    for episode_index in range(n_episodes):
        seed = derive_seed("lab02.rollouts", episode_index)
        session = WorldSession(env_cfg)
        episode_records: list[StateRecord] = []
        actions: list[int] = []
        try:
            result = session.reset(seed)
            oracle = SynchronizedOracle(session)
            last_executed: int | None = None
            while not session.done:
                unwrapped = session.env.unwrapped
                position = (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]))
                agent_dir = int(unwrapped.agent_dir)
                recommendation = oracle.recommend(last_executed, session.time)
                episode_records.append(
                    StateRecord(
                        observation_key=observation_key(
                            result.image, result.direction, result.mission
                        ),
                        episode_index=episode_index,
                        t=session.time,
                        position=position,
                        agent_dir=agent_dir,
                        recommendation=recommendation,
                        image_bytes=result.image.tobytes(),
                    )
                )
                result = session.step(recommendation)
                actions.append(recommendation)
                last_executed = recommendation
        except (OracleSupportError, OracleSyncError):
            counters["oracle_unsupported"] += 1
            continue
        finally:
            session.close()
        counters["episodes"] += 1
        counters["successes"] += int(result.terminated and result.reward > 0.0)
        counters["truncated"] += int(result.truncated)
        records.extend(episode_records)
        actions_by_episode[episode_index] = actions
    return records, actions_by_episode, counters


def analyze_aliasing(records: list[StateRecord]) -> dict[str, object]:
    """Group visited states by observation; measure aliasing and label conflict."""
    classes: dict[str, list[StateRecord]] = defaultdict(list)
    for record in records:
        classes[record.observation_key].append(record)

    total = len(records)
    aliased_classes = 0
    cross_world_classes = 0
    heterogeneous_classes = 0
    minority_mass = 0
    showcase: dict[str, object] | None = None

    for entries in classes.values():
        states = {(e.episode_index, e.position, e.agent_dir) for e in entries}
        labels = {e.recommendation for e in entries}
        counts: dict[int, int] = defaultdict(int)
        for entry in entries:
            counts[entry.recommendation] += 1
        minority_mass += len(entries) - max(counts.values())
        if len(states) < 2:
            continue
        aliased_classes += 1
        worlds = {e.episode_index for e in entries}
        if len(worlds) >= 2:
            cross_world_classes += 1
        if len(labels) >= 2:
            heterogeneous_classes += 1
            if len(worlds) >= 2:
                ordered = sorted(entries, key=lambda e: (e.episode_index, e.t))
                first = ordered[0]
                partner = next(
                    (
                        e
                        for e in ordered[1:]
                        if e.episode_index != first.episode_index
                        and e.recommendation != first.recommendation
                    ),
                    None,
                )
                if partner is not None:
                    candidate = {
                        "first": first,
                        "second": partner,
                        "rank": (first.episode_index, first.t),
                    }
                    if showcase is None or candidate["rank"] < showcase["rank"]:
                        showcase = candidate

    return {
        "total_states": total,
        "observation_classes": len(classes),
        "aliased_classes": aliased_classes,
        "cross_world_aliased_classes": cross_world_classes,
        "label_heterogeneous_classes": heterogeneous_classes,
        "memoryless_error_lower_bound": (minority_mass / total) if total else 0.0,
        "showcase": showcase,
    }


def _replay_state(
    env_cfg: EnvironmentConfig, episode_index: int, actions: list[int], t: int
) -> tuple[WorldState, np.ndarray, str]:
    """Deterministic replay to step ``t``; returns (world state, observation, mission)."""
    session = WorldSession(env_cfg)
    try:
        result = session.reset(derive_seed("lab02.rollouts", episode_index))
        for action in actions[:t]:
            result = session.step(action)
        grid, pose, visible = state_snapshot(session)
        return WorldState(grid, pose, visible), result.image.copy(), result.mission
    finally:
        session.close()


def _annotate_object_plane(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    plane = image[:, :, 0].T
    glyphs = [
        [OBJECT_GLYPHS[IDX_TO_OBJECT[int(value)]] for value in row] for row in plane
    ]
    draw_plane(axis, plane, vmax=10, glyphs=glyphs)
    axis.set_title(title, fontsize=12)


def _render_showcase(
    env_cfg: EnvironmentConfig,
    showcase: dict[str, object],
    actions_by_episode: dict[int, list[int]],
    paths: LabPaths,
) -> dict[str, object]:
    first: StateRecord = showcase["first"]
    second: StateRecord = showcase["second"]
    world_a, obs_a, mission = _replay_state(
        env_cfg, first.episode_index, actions_by_episode[first.episode_index], first.t
    )
    world_b, obs_b, _ = _replay_state(
        env_cfg, second.episode_index, actions_by_episode[second.episode_index], second.t
    )
    if obs_a.tobytes() != obs_b.tobytes():
        raise RuntimeError("showcase replay produced diverging observations")

    names = env_cfg.action_names
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    draw_state(axes[0], world_a)
    axes[0].set_title(
        f"world A (episode {first.episode_index}, t={first.t})\n"
        f"oracle: `{names[first.recommendation]}`",
        fontsize=12,
    )
    _annotate_object_plane(axes[1], obs_a, "identical observation\n(object plane, byte-equal)")
    draw_state(axes[2], world_b)
    axes[2].set_title(
        f"world B (episode {second.episode_index}, t={second.t})\n"
        f"oracle: `{names[second.recommendation]}`",
        fontsize=12,
    )
    fig.suptitle(
        f'perceptual aliasing under mission "{mission}": one observation, '
        "two worlds, two different optimal actions",
        fontsize=13,
    )
    save_figure(paths, fig, "aliasing_showcase.svg")
    return {
        "mission": mission,
        "episode_a": first.episode_index,
        "t_a": first.t,
        "action_a": names[first.recommendation],
        "episode_b": second.episode_index,
        "t_b": second.t,
        "action_b": names[second.recommendation],
    }


def _render_full_obs_contrast(env_cfg: EnvironmentConfig, paths: LabPaths) -> None:
    """The rejected alternative: hand the policy the whole grid."""
    import gymnasium as gym
    from minigrid.wrappers import FullyObsWrapper

    seed = derive_seed("lab02.rollouts", 0)
    session = WorldSession(env_cfg)
    try:
        partial = session.reset(seed)
        grid, pose, visible = state_snapshot(session)
    finally:
        session.close()
    wrapped = FullyObsWrapper(
        gym.make(env_cfg.env_id, disable_env_checker=True, doors_open=env_cfg.doors_open)
    )
    try:
        full_obs, _ = wrapped.reset(seed=seed)
    finally:
        wrapped.close()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    draw_grid(axes[0], grid, agent=pose, visible=visible)
    axes[0].set_title("the world", fontsize=12)
    _annotate_object_plane(axes[1], partial.image, "what our policy sees\n(7x7 partial)")
    draw_plane(axes[2], full_obs["image"][:, :, 0].T, vmax=10)
    axes[2].set_title(
        f"FullyObsWrapper alternative\n({full_obs['image'].shape[0]}x"
        f"{full_obs['image'].shape[1]} full state, rejected)",
        fontsize=12,
    )
    fig.suptitle(
        "partial observability is a choice: the wrapper would make the task an MDP, "
        "but no physical agent observes the world state",
        fontsize=13,
    )
    save_figure(paths, fig, "full_observability_contrast.svg")


def run(
    paths: LabPaths,
    *,
    force: bool,
    rollout_episodes: int = ROLLOUT_EPISODES,
) -> dict[str, object]:
    prepare(paths, force=force)
    env_cfg = contract_environment(paths)

    records, actions_by_episode, counters = collect_states(env_cfg, rollout_episodes)
    analysis = analyze_aliasing(records)
    showcase = analysis.pop("showcase")
    showcase_info: dict[str, object] | None = None
    if showcase is not None:
        showcase_info = _render_showcase(env_cfg, showcase, actions_by_episode, paths)
    _render_full_obs_contrast(env_cfg, paths)

    export_typst_table(
        paths,
        "pomdp_mapping",
        ["symbol", "meaning here", "code entity"],
        POMDP_MAPPING_ROWS,
    )
    write_table_csv(
        paths, "pomdp_mapping.csv", ["symbol", "meaning", "code"], POMDP_MAPPING_ROWS
    )
    export_typst_values(
        paths,
        "aliasing_facts",
        {
            "n-episodes": str(counters["episodes"]),
            "n-states": str(analysis["total_states"]),
            "n-observation-classes": str(analysis["observation_classes"]),
            "n-aliased-classes": str(analysis["aliased_classes"]),
            "n-cross-world-classes": str(analysis["cross_world_aliased_classes"]),
            "n-heterogeneous-classes": str(analysis["label_heterogeneous_classes"]),
            "memoryless-error-lower-bound": f"{analysis['memoryless_error_lower_bound']:.4f}",
            "memoryless-error-lower-bound-pct": (
                f"{100.0 * analysis['memoryless_error_lower_bound']:.1f}%"
            ),
        },
    )

    metrics = {
        "rollouts": counters,
        "aliasing": analysis,
        "showcase": showcase_info,
    }
    metrics_hash = write_metrics(paths, metrics)

    bound_pct = 100.0 * analysis["memoryless_error_lower_bound"]
    write_mini_report(
        paths,
        question="What is a POMDP, why is this task one, and what would the alternatives cost?",
        sections=[
            (
                "The formal object",
                "A Markov decision process (MDP) is (S, A, T, R): states, actions, "
                "transitions, reward. A *partially observable* MDP adds an "
                "observation space and an observation function, because the agent never "
                "receives the state s, only an observation o = O(s). The mapping "
                "from each symbol to this repository's code is exported in "
                "`pomdp_mapping.typ`; here the dynamics are deterministic and all "
                "randomness sits in world generation (the reset seed).",
            ),
            (
                "Aliasing, measured",
                f"Across {counters['episodes']} nominal oracle episodes "
                f"({analysis['total_states']} visited states), the states collapse "
                f"into {analysis['observation_classes']} distinct observations; "
                f"{analysis['aliased_classes']} observation classes are *aliased* "
                "(the same bytes arise from provably different world states), "
                f"{analysis['cross_world_aliased_classes']} of them across entirely "
                f"different mazes, and {analysis['label_heterogeneous_classes']} "
                "carry *conflicting oracle actions*. Consequence: any memoryless "
                "policy, meaning any function from single observations to actions, must "
                f"disagree with the oracle on at least {bound_pct:.1f}% of visited "
                "states on this distribution. Memory is not a nicety; it is "
                "required for optimality.",
            ),
            (
                "The showcase pair",
                "`figures/aliasing_showcase.svg` shows two different mazes whose "
                "agents receive byte-identical observations while the oracle "
                "recommends different actions "
                + (
                    f"(`{showcase_info['action_a']}` vs `{showcase_info['action_b']}`). "
                    if showcase_info
                    else "(no cross-world conflicting pair found at this scale). "
                )
                + "Selection rule (disclosed): the conflicting cross-world class "
                "whose first member appears earliest in the rollout order, so no "
                "cherry-picking by appearance.",
            ),
            (
                "Alternatives and why we reject them",
                "Full observability (`figures/full_observability_contrast.svg`): "
                "MiniGrid can hand the policy the entire grid, turning the task "
                "into an MDP, but no physical agent observes the world state, and "
                "the study is about acting under realistic perception. Frame "
                "stacking approximates short memory with a fixed window; belief "
                "states are exact but require a known world model. The study's "
                "choice, learned memory in a recurrent network, is built and "
                "ablated in Labs 4 and 5.",
            ),
            (
                "Bridge to the study",
                "The study never constructs anything beyond this POMDP interface: "
                "policies consume exactly `StepResult` fields (image, direction, "
                "mission) plus their own previous executed action. The oracle, in "
                "contrast, *does* read the full state, and that asymmetry (privileged "
                "teacher, partially observing student) is what makes expert labels "
                "informative, and is the subject of Lab 3.",
            ),
        ],
    )

    return {
        "episodes": counters["episodes"],
        "aliased_classes": analysis["aliased_classes"],
        "heterogeneous_classes": analysis["label_heterogeneous_classes"],
        "memoryless_error_lower_bound": round(analysis["memoryless_error_lower_bound"], 4),
        "metrics_hash": metrics_hash,
    }
