"""Lab 1: the world, BabyAI/MiniGrid.

What is BabyAI, and what does the agent actually get to see and do? This lab
answers by measurement: a census over freshly generated worlds (missions,
objects, doors, geometry), a decomposition of the symbolic observation tensor
into its object/color/state planes, and rendered galleries showing what each
action does. Everything uses the exact environment configuration of the frozen
study contract, loaded read-only.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from minigrid.core.constants import (
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)

from gr_foundations.common import (
    COLOR_NEUTRAL,
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
from gr_foundations.gridart import draw_plane, draw_world
from grounded_recovery.config import EnvironmentConfig, load_and_validate
from grounded_recovery.world import OBSERVATION_IMAGE_SHAPE, WorldSession

IDX_TO_STATE = {index: name for name, index in STATE_TO_IDX.items()}

# Compact glyphs for annotating the 7x7 object plane in figures.
OBJECT_GLYPHS = {
    "unseen": "?",
    "empty": "·",
    "wall": "W",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "O",
    "box": "B",
    "goal": "G",
    "lava": "L",
    "agent": "A",
}

MISSION_PATTERN = re.compile(
    r"^go to (?:a|the) (?P<color>" + "|".join(COLOR_TO_IDX) + r") "
    r"(?P<kind>ball|box|key)$"
)

CENSUS_SEEDS = 500
CONTRAST_SEEDS = 100
GALLERY_SEEDS = 6


def contract_environment(paths: LabPaths) -> EnvironmentConfig:
    """The frozen study's environment configuration, loaded read-only."""
    contract = load_and_validate(paths.repo_root / "configs" / "experiment_contract.yaml")
    return contract.environment


def decompose_observation(image: np.ndarray) -> dict[str, np.ndarray]:
    """Split the (7, 7, 3) symbolic tensor into named integer planes."""
    if image.shape != OBSERVATION_IMAGE_SHAPE:
        raise ValueError(f"expected image shape {OBSERVATION_IMAGE_SHAPE}, got {image.shape}")
    return {
        "object": image[:, :, 0].copy(),
        "color": image[:, :, 1].copy(),
        "state": image[:, :, 2].copy(),
    }


def recompose_observation(planes: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`decompose_observation`."""
    return np.stack([planes["object"], planes["color"], planes["state"]], axis=-1).astype(
        np.uint8
    )


def describe_cell(cell: np.ndarray) -> str:
    """Human name for one (object, color, state) triple."""
    obj = IDX_TO_OBJECT[int(cell[0])]
    if obj in ("unseen", "empty", "wall", "floor"):
        return obj
    color = IDX_TO_COLOR[int(cell[1])]
    if obj == "door":
        return f"{color} door ({IDX_TO_STATE[int(cell[2])]})"
    return f"{color} {obj}"


def parse_mission(mission: str) -> tuple[str, str] | None:
    """Extract (color, kind) from a GoTo mission string, or None."""
    match = MISSION_PATTERN.match(mission)
    if match is None:
        return None
    return match.group("color"), match.group("kind")


def run_census(env_cfg: EnvironmentConfig, n_seeds: int, *, seed_offset: int = 0) -> dict:
    """Measured facts about ``n_seeds`` freshly generated worlds."""
    door_idx = OBJECT_TO_IDX["door"]
    countable = ("door", "key", "ball", "box")
    mission_counter: Counter[str] = Counter()
    color_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    direction_counter: Counter[int] = Counter()
    object_totals: Counter[str] = Counter()
    door_states: Counter[str] = Counter()
    grid_shapes = set()
    unparsed = 0

    session = WorldSession(env_cfg)
    try:
        for index in range(n_seeds):
            seed = derive_seed("lab01.census", seed_offset + index)
            result = session.reset(seed)
            state = session.scenario_state()
            grid = np.asarray(state.grid_encoding)
            grid_shapes.add(grid.shape[:2])
            mission_counter[result.mission] += 1
            direction_counter[state.agent_dir] += 1
            parsed = parse_mission(result.mission)
            if parsed is None:
                unparsed += 1
            else:
                color_counter[parsed[0]] += 1
                kind_counter[parsed[1]] += 1
            objects = grid[:, :, 0]
            for name in countable:
                object_totals[name] += int(np.sum(objects == OBJECT_TO_IDX[name]))
            for state_value in grid[:, :, 2][objects == door_idx].ravel():
                door_states[IDX_TO_STATE[int(state_value)]] += 1
    finally:
        session.close()

    return {
        "n_seeds": n_seeds,
        "doors_open": env_cfg.doors_open,
        "grid_shapes": sorted(f"{shape[0]}x{shape[1]}" for shape in grid_shapes),
        "unique_missions": len(mission_counter),
        "unparsed_missions": unparsed,
        "mission_color_counts": dict(sorted(color_counter.items())),
        "mission_kind_counts": dict(sorted(kind_counter.items())),
        "agent_start_direction_counts": {
            str(k): v for k, v in sorted(direction_counter.items())
        },
        "object_totals": dict(sorted(object_totals.items())),
        "door_state_counts": dict(sorted(door_states.items())),
        "top_missions": mission_counter.most_common(5),
    }


def _render_gallery(env_cfg: EnvironmentConfig, paths: LabPaths, n_panels: int) -> None:
    session = WorldSession(env_cfg)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.0))
        for panel, axis in enumerate(axes.ravel()):
            if panel >= n_panels:
                axis.axis("off")
                continue
            result = session.reset(derive_seed("lab01.gallery", panel))
            draw_world(axis, session)
            axis.set_title(f'"{result.mission}"', fontsize=11)
        fig.suptitle(
            f"{env_cfg.env_id}: six freshly generated worlds "
            "(bright cone = the agent's 7×7 field of view)",
            fontsize=13,
        )
        save_figure(paths, fig, "world_gallery.svg")
    finally:
        session.close()


def _render_action_effects(env_cfg: EnvironmentConfig, paths: LabPaths) -> None:
    seed = derive_seed("lab01.gallery", 100)
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.2))
    labels = ["start"] + [f"after `{name}`" for name in env_cfg.action_names]
    for axis, action, label in zip(
        axes, (None, *env_cfg.action_ids), labels, strict=True
    ):
        session = WorldSession(env_cfg)
        try:
            session.reset(seed)
            if action is not None:
                session.step(action)
            draw_world(axis, session)
        finally:
            session.close()
        axis.set_title(label, fontsize=12)
    fig.suptitle(
        "The frozen action set: turning changes only the view direction; "
        "`forward` moves the agent one cell",
        fontsize=13,
    )
    save_figure(paths, fig, "action_effects.svg")


def _render_observation_anatomy(env_cfg: EnvironmentConfig, paths: LabPaths) -> None:
    session = WorldSession(env_cfg)
    try:
        result = session.reset(derive_seed("lab01.gallery", 200))
        planes = decompose_observation(result.image)
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.2))
        draw_world(axes[0], session)
    finally:
        session.close()
    axes[0].set_title("world (render only; never observed)", fontsize=12)
    display = {
        "object": "object plane (what)",
        "color": "color plane",
        "state": "state plane (doors)",
    }
    for axis, (key, title) in zip(axes[1:], display.items(), strict=True):
        plane = planes[key].T  # row = depth into the view, agent at bottom row
        glyphs = None
        if key == "object":
            glyphs = [
                [OBJECT_GLYPHS[IDX_TO_OBJECT[int(value)]] for value in row]
                for row in plane
            ]
        draw_plane(axis, plane, vmax=max(10, int(plane.max())), glyphs=glyphs)
        axis.set_title(title, fontsize=12)
    fig.suptitle(
        f'mission: "{result.mission}", and the policy observes only these '
        "7×7 integer planes, the view direction, and the mission text",
        fontsize=13,
    )
    save_figure(paths, fig, "observation_anatomy.svg")


def _vocabulary_rows() -> dict[str, list[list[object]]]:
    vocabularies = {"objects": OBJECT_TO_IDX, "colors": COLOR_TO_IDX, "states": STATE_TO_IDX}
    return {
        name: [[index, entry] for entry, index in sorted(mapping.items(), key=lambda x: x[1])]
        for name, mapping in vocabularies.items()
    }


def _distribution_figure(census: dict, paths: LabPaths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for axis, key, title in (
        (axes[0], "mission_color_counts", "mission target color"),
        (axes[1], "mission_kind_counts", "mission target kind"),
    ):
        items = sorted(census[key].items())
        axis.bar(
            [name for name, _ in items],
            [count for _, count in items],
            color=COLOR_NEUTRAL,
        )
        axis.set_title(f"{title} (n={census['n_seeds']})", fontsize=13)
        axis.tick_params(axis="x", labelrotation=30, labelsize=12)
    fig.suptitle("What the task generator asks for", fontsize=13)
    save_figure(paths, fig, "mission_distribution.svg")


def run(
    paths: LabPaths,
    *,
    force: bool,
    census_seeds: int = CENSUS_SEEDS,
    contrast_seeds: int = CONTRAST_SEEDS,
) -> dict[str, object]:
    prepare(paths, force=force)
    env_cfg = contract_environment(paths)

    census = run_census(env_cfg, census_seeds)
    closed_cfg = dataclasses.replace(env_cfg, doors_open=False)
    contrast = run_census(closed_cfg, contrast_seeds, seed_offset=10_000)

    _render_gallery(env_cfg, paths, GALLERY_SEEDS)
    _render_action_effects(env_cfg, paths)
    _render_observation_anatomy(env_cfg, paths)
    _distribution_figure(census, paths)

    vocab = _vocabulary_rows()
    for name, rows in vocab.items():
        export_typst_table(paths, f"vocabulary_{name}", ["index", "name"], rows)
        write_table_csv(paths, f"vocabulary_{name}.csv", ["index", "name"], rows)
    export_typst_table(
        paths,
        "doors_contrast",
        ["parameterization", "doors seen", "open", "closed", "locked"],
        [
            [
                "contract (doors_open: true)",
                sum(census["door_state_counts"].values()),
                census["door_state_counts"].get("open", 0),
                census["door_state_counts"].get("closed", 0),
                census["door_state_counts"].get("locked", 0),
            ],
            [
                "default (doors_open: false)",
                sum(contrast["door_state_counts"].values()),
                contrast["door_state_counts"].get("open", 0),
                contrast["door_state_counts"].get("closed", 0),
                contrast["door_state_counts"].get("locked", 0),
            ],
        ],
    )
    export_typst_values(
        paths,
        "world_facts",
        {
            "env-id": env_cfg.env_id,
            "max-steps": str(env_cfg.max_steps),
            "grid-shape": census["grid_shapes"][0],
            "census-seeds": str(census_seeds),
            "unique-missions": str(census["unique_missions"]),
            "frozen-actions": ", ".join(env_cfg.action_names),
        },
    )

    metrics = {
        "environment": {
            "env_id": env_cfg.env_id,
            "observation_image_shape": list(OBSERVATION_IMAGE_SHAPE),
            "max_steps": env_cfg.max_steps,
            "frozen_action_ids": list(env_cfg.action_ids),
            "frozen_action_names": list(env_cfg.action_names),
            "doors_open": env_cfg.doors_open,
        },
        "census": census,
        "census_doors_closed_contrast": contrast,
    }
    metrics_hash = write_metrics(paths, metrics)

    write_mini_report(
        paths,
        question="What is BabyAI, and what does the agent actually see and do?",
        sections=[
            (
                "The environment",
                f"`{env_cfg.env_id}` is a maze of connected rooms on a "
                f"{census['grid_shapes'][0]} cell grid (measured over "
                f"{census_seeds} generated worlds). Every episode places the agent "
                "somewhere in the maze, scatters objects, and issues a natural-"
                "language mission. The episode ends in success when the agent "
                f"stands next to the requested object, or after {env_cfg.max_steps} "
                "steps (the environment's own limit).",
            ),
            (
                "What the agent observes",
                "Not the maze. The observation is a 7×7×3 integer tensor: an "
                "egocentric, occlusion-aware crop of the world in front of the "
                "agent, plus the view direction (0–3) and the mission string. "
                "The three channels are symbolic lookup indices (object kind, "
                "color, door state), not pixels; see "
                "`figures/observation_anatomy.svg` and the vocabulary tables. "
                "This partial view is what makes the task a POMDP (Lab 2).",
            ),
            (
                "What the agent can do",
                f"The study freezes three actions: {', '.join(env_cfg.action_names)} "
                "(ids 0/1/2 of MiniGrid's seven). Turning rotates the view in "
                "place; `forward` advances one cell if nothing blocks it, see "
                "`figures/action_effects.svg`. Why the other four actions are "
                "excluded, and why the set must stay frozen, is the subject of "
                "Lab 6.",
            ),
            (
                "Mission grammar",
                f"All {census_seeds} sampled missions follow one template "
                f"(`go to a/the <color> <kind>`; {census['unparsed_missions']} "
                f"unmatched), spanning {census['unique_missions']} distinct "
                "strings, with colors "
                f"{', '.join(census['mission_color_counts'])} and kinds "
                f"{', '.join(census['mission_kind_counts'])}. The language input "
                "is tiny but real: the policy must read it to know which object "
                "counts as success.",
            ),
            (
                "Doors",
                "With the study's contract setting `doors_open: true`, all "
                f"{sum(census['door_state_counts'].values())} doors seen across "
                f"{census_seeds} worlds are open; with the environment default, "
                f"{contrast['door_state_counts'].get('closed', 0)} of "
                f"{sum(contrast['door_state_counts'].values())} doors across "
                f"{contrast_seeds} worlds are closed. The study's choice keeps "
                "the frozen 3-action set sufficient. The full story is told "
                "with the corruption operators in Lab 6.",
            ),
            (
                "Bridge to the study",
                "The study wraps exactly this environment in "
                "`grounded_recovery.world.WorldSession`, which adds contract "
                "checks: resets demand an explicit seed, only frozen actions "
                "pass, stepping after termination is an error, and the reset "
                "world can be hashed into a scenario identity. Those checks are "
                "bookkeeping, not learning; the world itself is what this lab "
                "measured.",
            ),
        ],
    )

    return {
        "census_seeds": census_seeds,
        "unique_missions": census["unique_missions"],
        "metrics_hash": metrics_hash,
    }
