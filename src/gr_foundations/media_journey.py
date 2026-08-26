"""Rollout videos and step-through trajectories for the journey website.

Every item is generated deterministically from named seeds or stored
checkpoints, selected by a disclosed rule (never by appearance), and, where
an outcome is claimed, asserted against the stored evaluation rows before
anything is written. Videos are MP4 with a poster frame (the site never
autoplays GIFs); two episodes are additionally exported as JSON trajectories
for the in-page scrubber, carrying the full grid, the agent's observation,
and the executed action at every step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
from PIL import Image, ImageDraw, ImageFont

from gr_foundations.common import FoundationsError, derive_seed
from gr_foundations.gridart import state_snapshot as _state_snapshot
from gr_foundations.lab03_oracle import RandomPolicy
from gr_foundations.models import LabPolicy
from gr_foundations.training import (
    START_ACTION_TOKEN,
    build_bc_dataset,
    contract_config,
    load_checkpoint,
)
from grounded_recovery.artifacts import atomic_write_json, file_sha256
from grounded_recovery.config import EnvironmentConfig, ExperimentConfig
from grounded_recovery.data import Vocabulary
from grounded_recovery.media import write_mp4
from grounded_recovery.model import RecoveryPolicy
from grounded_recovery.oracle import SynchronizedOracle
from grounded_recovery.perturbations import ActionDerangement
from grounded_recovery.world import WorldSession

FPS = 5
HOLD_FRAMES = 10
# The raw render is 320 px per world; annotation happens after a nearest-
# neighbour 2x upscale (the study's convention), so text stays sharp.
SCALE = 2
HEADER = 40 * SCALE
FOOTER = 30 * SCALE
MARGIN = 8 * SCALE

# The site's arm palette: sage for recovery, sky for extra demonstrations,
# brown for the base policy, steel for neutral panes, gold for corruptions.
SAGE = (107, 143, 113)
SKY = (78, 154, 225)
BROWN = (133, 114, 85)
STEEL = (77, 123, 158)
GOLD = (235, 165, 56)
RED = (192, 57, 43)
INK = (79, 95, 107)
PAPER = (250, 249, 245)

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    if size not in _FONTS:
        for path in _FONT_PATHS:
            try:
                _FONTS[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _fit_text(draw: ImageDraw.ImageDraw, text: str, size: int, max_width: float):
    """The font (stepped down to 14 px) and text that fit inside ``max_width``.

    Overflow is structurally impossible: if even the smallest size cannot hold
    the string, it is ellipsized. Every drawn string goes through here.
    """
    while size > 14:
        font = _font(size)
        if draw.textlength(text, font=font) <= max_width:
            return text, font
        size -= 2
    font = _font(14)
    if draw.textlength(text, font=font) <= max_width:
        return text, font
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text + "…", font


# --------------------------------------------------------------------------
# Capture: one generic driver for oracle, random, and checkpointed policies.
# --------------------------------------------------------------------------


@dataclass
class Capture:
    """One rendered episode plus everything the annotator and scrubber need."""

    mission: str
    frames: list[np.ndarray]  # T+1 renders; frames[t] shows the world before step t
    actions: list[int]
    labels: list[int | None]  # oracle recommendation at each state, when queried
    success: bool
    steps: int
    corruption_time: int | None = None
    delivered: bool = False
    grids: list[np.ndarray] = field(default_factory=list)
    poses: list[tuple[int, int, int]] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    visible: list[list[tuple[int, int]]] = field(default_factory=list)


def capture_episode(
    env_cfg: EnvironmentConfig,
    seed: int,
    choose_action,
    *,
    with_oracle: bool = False,
    corruption: tuple[ActionDerangement, int] | None = None,
    record_state: bool = False,
) -> Capture:
    """Roll one episode with rendering.

    ``choose_action(result, t, label)`` picks the proposed action; ``label``
    is the oracle's recommendation when ``with_oracle`` is set (the oracle is
    kept synchronized with whatever was actually executed).
    """
    session = WorldSession(env_cfg, render_mode="rgb_array")
    try:
        result = session.reset(seed)
        oracle = SynchronizedOracle(session) if with_oracle else None
        last: int | None = None
        capture = Capture(
            mission=result.mission, frames=[], actions=[], labels=[],
            success=False, steps=0,
            corruption_time=corruption[1] if corruption else None,
        )
        while not session.done:
            time = session.time
            capture.frames.append(session.render_frame())
            if record_state:
                grid, pose, visible = _state_snapshot(session)
                capture.grids.append(grid)
                capture.poses.append(pose)
                capture.observations.append(result.image.copy())
                capture.visible.append(visible)
            label = oracle.recommend(last, time) if oracle is not None else None
            action = int(choose_action(result, time, label))
            if corruption is not None and time == corruption[1]:
                action = corruption[0].apply(action)
                capture.delivered = True
            capture.labels.append(label)
            capture.actions.append(action)
            result = session.step(action)
            last = action
        capture.frames.append(session.render_frame())
        if record_state:
            grid, pose, visible = _state_snapshot(session)
            capture.grids.append(grid)
            capture.poses.append(pose)
            capture.observations.append(result.image.copy())
            capture.visible.append(visible)
        capture.success = bool(result.terminated and result.reward > 0.0)
        capture.steps = session.time
        return capture
    finally:
        session.close()


class _PolicyDriver:
    """Greedy closed-loop driver around a checkpointed model."""

    def __init__(self, model: torch.nn.Module, vocab: Vocabulary, device: torch.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self._mission_feature: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._prev = torch.tensor([START_ACTION_TOKEN], dtype=torch.long, device=device)

    @torch.no_grad()
    def __call__(self, result, time: int, _label) -> int:
        if time == 0:
            tokens = self.vocab.encode(result.mission)
            mission_tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long, device=self.device)
            self._mission_feature = self.model.encode_mission(mission_tokens, lengths)
            self._hidden = None
            self._prev = torch.tensor(
                [START_ACTION_TOKEN], dtype=torch.long, device=self.device
            )
        image = torch.from_numpy(result.image.astype(np.int64)).to(self.device)
        direction = torch.tensor([result.direction], dtype=torch.long, device=self.device)
        logits, self._hidden = self.model.step(
            image.unsqueeze(0), direction, self._prev, self._mission_feature, self._hidden
        )
        action = int(torch.argmax(logits, dim=-1).item())
        self._prev = torch.tensor([action], dtype=torch.long, device=self.device)
        return action

    def note_executed(self, action: int) -> None:
        """Correct the prev-action input when the executed action differed."""
        self._prev = torch.tensor([action], dtype=torch.long, device=self.device)


def capture_policy_episode(
    env_cfg: EnvironmentConfig,
    seed: int,
    driver: _PolicyDriver,
    *,
    corruption: tuple[ActionDerangement, int] | None = None,
    record_state: bool = False,
) -> Capture:
    """Greedy policy rollout with rendering. A corrupted action is also fed
    back into the policy's previous-action input, the corruption channel."""
    session = WorldSession(env_cfg, render_mode="rgb_array")
    try:
        result = session.reset(seed)
        capture = Capture(
            mission=result.mission, frames=[], actions=[], labels=[],
            success=False, steps=0,
            corruption_time=corruption[1] if corruption else None,
        )
        while not session.done:
            time = session.time
            capture.frames.append(session.render_frame())
            if record_state:
                grid, pose, visible = _state_snapshot(session)
                capture.grids.append(grid)
                capture.poses.append(pose)
                capture.observations.append(result.image.copy())
                capture.visible.append(visible)
            action = driver(result, time, None)
            if corruption is not None and time == corruption[1]:
                action = corruption[0].apply(action)
                driver.note_executed(action)
                capture.delivered = True
            capture.labels.append(None)
            capture.actions.append(action)
            result = session.step(action)
        capture.frames.append(session.render_frame())
        if record_state:
            grid, pose, visible = _state_snapshot(session)
            capture.grids.append(grid)
            capture.poses.append(pose)
            capture.observations.append(result.image.copy())
            capture.visible.append(visible)
        capture.success = bool(result.terminated and result.reward > 0.0)
        capture.steps = session.time
        return capture
    finally:
        session.close()


# --------------------------------------------------------------------------
# Annotation.
# --------------------------------------------------------------------------


def _annotate(
    capture: Capture,
    *,
    title: str,
    color: tuple[int, int, int],
    env_cfg: EnvironmentConfig,
    show_labels: bool = False,
    final_caption: str | None = None,
    hold: int = HOLD_FRAMES,
) -> list[Image.Image]:
    """Header, per-step footer, corruption flash, and an outcome banner.

    Frames are upscaled 2x (nearest neighbour) before any text lands on them.
    The flash sits on the frame *after* the scheduled step, where the damage
    is visible, matching the study's videos; every string passes through
    ``_fit_text``. ``final_caption`` replaces the SUCCESS/FAILURE banner for
    clips that end mid-episode (the aliasing pair freezes at the aliased
    state, which is not an outcome).
    """
    names = env_cfg.action_names
    images: list[Image.Image] = []
    total = len(capture.frames)
    for index, frame in enumerate(capture.frames):
        base = Image.fromarray(frame)
        base = base.resize((base.width * SCALE, base.height * SCALE), Image.NEAREST)
        width, height = base.size
        text_width = width - 2 * MARGIN
        canvas = Image.new("RGB", (width, height + HEADER + FOOTER), PAPER)
        canvas.paste(base, (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, width, HEADER], fill=color)
        head, head_font = _fit_text(draw, title, 30, text_width)
        draw.text((MARGIN, 22), head, fill=(255, 255, 255), font=head_font)
        flashed = (
            capture.corruption_time is not None
            and capture.delivered
            and index == capture.corruption_time + 1
        )
        if flashed:
            draw.rectangle(
                [0, HEADER, width - 1, HEADER + height - 1], outline=GOLD, width=10
            )
        if index < total - 1:
            caption = f"t={index}"
            if show_labels and capture.labels[index] is not None:
                caption += f" · oracle: {names[capture.labels[index]]}"
            elif index < len(capture.actions):
                caption += f" · {names[capture.actions[index]]}"
            if flashed:
                caption += f"  ⚡ corrupted at t={capture.corruption_time}"
            line, font = _fit_text(draw, caption, 26, text_width)
            draw.text((MARGIN, HEADER + height + 12), line, fill=INK, font=font)
        elif final_caption is not None:
            draw.rectangle([0, HEADER + height, width, HEADER + height + FOOTER], fill=GOLD)
            line, font = _fit_text(draw, final_caption, 26, text_width)
            draw.text((MARGIN, HEADER + height + 12), line, fill=(255, 255, 255), font=font)
        else:
            outcome = "SUCCESS" if capture.success else "FAILURE"
            banner = SAGE if capture.success else RED
            draw.rectangle([0, HEADER + height, width, HEADER + height + FOOTER], fill=banner)
            line, font = _fit_text(draw, f"{outcome} · {capture.steps} steps", 28, text_width)
            draw.text((MARGIN, HEADER + height + 12), line, fill=(255, 255, 255), font=font)
        images.append(canvas)
    images.extend([images[-1]] * hold)
    return images


def _side_by_side(left: list[Image.Image], right: list[Image.Image], gap: int = 6 * SCALE):
    length = max(len(left), len(right))
    left = left + [left[-1]] * (length - len(left))
    right = right + [right[-1]] * (length - len(right))
    combined = []
    for a, b in zip(left, right, strict=True):
        height = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + gap + b.width, height), PAPER)
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + gap, 0))
        combined.append(canvas)
    return combined


def _write_video(images: list[Image.Image], out_dir: Path, name: str) -> dict[str, str]:
    video = out_dir / f"{name}.mp4"
    poster = out_dir / "posters" / f"{name}.webp"
    poster.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(images, video, fps=FPS)
    images[0].save(poster, "WEBP", quality=90, method=6)
    return {
        "href": f"media/{name}.mp4",
        "poster": f"media/posters/{name}.webp",
        "sha256": file_sha256(video),
    }


# --------------------------------------------------------------------------
# Trajectory export for the scrubber.
# --------------------------------------------------------------------------


def trajectory_document(
    capture: Capture, env_cfg: EnvironmentConfig, *, source: dict[str, object]
) -> dict[str, object]:
    if not capture.grids:
        raise FoundationsError("trajectory export requires record_state captures")
    steps = []
    # One entry per recorded state, terminal state included: the last step
    # carries no action, so the scrubber can land on the finished episode.
    for t in range(len(capture.grids)):
        x, y, direction = capture.poses[t]
        acting = t < len(capture.actions)
        steps.append(
            {
                "t": t,
                "grid": capture.grids[t].tolist(),
                "agent": {"x": x, "y": y, "dir": direction},
                "observation": capture.observations[t].tolist(),
                "visible": [list(cell) for cell in capture.visible[t]],
                "action": capture.actions[t] if acting else None,
                "label": capture.labels[t] if acting else None,
                "corrupted": acting and capture.corruption_time == t,
            }
        )
    return {
        "schema_version": "1.1.0",
        "mission": capture.mission,
        "outcome": "success" if capture.success else "failure",
        "steps_taken": capture.steps,
        "corruption_time": capture.corruption_time,
        "action_names": list(env_cfg.action_names),
        "legend": {
            "objects": {str(i): n for i, n in IDX_TO_OBJECT.items()},
            "colors": {str(i): n for i, n in IDX_TO_COLOR.items()},
        },
        "source": source,
        "steps": steps,
    }


# --------------------------------------------------------------------------
# Item builders.
# --------------------------------------------------------------------------


def _load_lab_model(repo_root: Path, contract: ExperimentConfig, relative: str):
    path = repo_root / "data" / "foundations" / relative
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FoundationsError(
            f"{path} is missing; run the lab that trains it (grf run lab04/lab06)"
        )
    meta = json.loads(meta_path.read_text())
    vocab = Vocabulary(tokens=tuple(meta["vocabulary"]))

    def factory():
        if meta.get("kind") == "memoryless":
            return LabPolicy(contract.model, vocab.size, 3, use_memory=False)
        return RecoveryPolicy(contract.model, vocab.size, 3)

    model, _meta = load_checkpoint(path, factory)
    return model, vocab


def build_media(repo_root: Path, out_dir: Path) -> dict[str, object]:
    contract = contract_config(repo_root)
    env_cfg = contract.environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest: list[dict[str, object]] = []
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    # -- expert labels + random wander: the same world, two drivers ---------
    seed = derive_seed("lab03.trajectory", 0)
    expert = capture_episode(
        env_cfg, seed, lambda _r, _t, label: label, with_oracle=True, record_state=True
    )
    if not expert.success:
        raise FoundationsError("expert episode unexpectedly failed")
    entry = _write_video(
        _annotate(expert, title="the oracle, labelling as it goes", color=STEEL,
                  env_cfg=env_cfg, show_labels=True),
        out_dir, "expert_labels",
    )
    trajectory_path = trajectories_dir / "expert_labels.json"
    atomic_write_json(
        trajectory_path,
        trajectory_document(
            expert, env_cfg,
            source={"kind": "oracle", "seed_component": "lab03.trajectory", "index": 0},
        ),
        overwrite=True,
    )
    manifest.append(
        {
            "id": "expert_labels", **entry, "empirical": True,
            "outcome": "success", "steps": expert.steps, "mission": expert.mission,
            "selection_rule": "The first expert trajectory of lab 03, in seed order.",
            "trace": "media/trajectories/expert_labels.json",
        }
    )

    random_policy = RandomPolicy(derive_seed("lab03.random_policy", 0))
    wander = capture_episode(
        env_cfg, seed, lambda result, time, _label: random_policy.act(result, time)
    )
    entry = _write_video(
        _annotate(wander, title="a random policy, same world", color=BROWN, env_cfg=env_cfg),
        out_dir, "random_wander",
    )
    manifest.append(
        {
            "id": "random_wander", **entry, "empirical": True,
            "outcome": "success" if wander.success else "failure",
            "steps": wander.steps, "mission": wander.mission,
            "selection_rule": "The same world as the expert clip, with the first "
                              "random policy of lab 03.",
        }
    )

    # -- the aliasing pair, animated ----------------------------------------
    from gr_foundations.lab02_decision import analyze_aliasing, collect_states

    records, actions_by_episode, _counters = collect_states(env_cfg, 300)
    analysis = analyze_aliasing(records)
    showcase = analysis["showcase"]
    if showcase is None:
        raise FoundationsError("no aliasing showcase found at 300 episodes")
    sides = []
    for role, record in (("A", showcase["first"]), ("B", showcase["second"])):
        replay = capture_episode(
            env_cfg,
            derive_seed("lab02.rollouts", record.episode_index),
            lambda _r, t, _l, actions=actions_by_episode[record.episode_index]:
                actions[t],
        )
        # Trim to the aliased moment and freeze there.
        partial = Capture(
            mission=replay.mission,
            frames=replay.frames[: record.t + 1],
            actions=replay.actions[: record.t],
            labels=[None] * record.t,
            success=False,
            steps=record.t,
        )
        title = (
            f"world {role} · oracle here: "
            f"{env_cfg.action_names[record.recommendation]}"
        )
        sides.append(
            _annotate(
                partial, title=title, color=STEEL, env_cfg=env_cfg,
                final_caption="→ byte-identical observation", hold=1,
            )
        )
    combined = _side_by_side(sides[0], sides[1])
    combined.extend([combined[-1]] * (HOLD_FRAMES * 2))
    entry = _write_video(combined, out_dir, "aliasing_pair")
    manifest.append(
        {
            "id": "aliasing_pair", **entry, "empirical": True,
            "outcome": "identical observations, different oracle actions",
            "selection_rule": "The earliest observation that repeats across two "
                              "worlds while the oracle asks for different "
                              "actions.",
        }
    )

    # -- imitation contrast: memoryless vs recurrent ------------------------
    recurrent, vocab04 = _load_lab_model(
        repo_root, contract, "lab04/checkpoints/recurrent_s0.pt"
    )
    memoryless, _ = _load_lab_model(
        repo_root, contract, "lab04/checkpoints/memoryless_s0.pt"
    )
    holdout, _ = build_bc_dataset(env_cfg, 100, "lab04.holdout")
    chosen = None
    for ordinal, episode in enumerate(holdout):
        good = capture_policy_episode(
            env_cfg, episode.seed, _PolicyDriver(recurrent, vocab04, device)
        )
        bad = capture_policy_episode(
            env_cfg, episode.seed, _PolicyDriver(memoryless, vocab04, device)
        )
        if good.success and not bad.success:
            chosen = (ordinal, episode.seed, good, bad)
            break
    if chosen is None:
        raise FoundationsError("no imitation-contrast scenario found in the holdout")
    ordinal, seed, good, bad = chosen
    combined = _side_by_side(
        _annotate(bad, title="memoryless policy", color=BROWN, env_cfg=env_cfg),
        _annotate(good, title="recurrent policy", color=SAGE, env_cfg=env_cfg),
    )
    entry = _write_video(combined, out_dir, "imitation_contrast")
    manifest.append(
        {
            "id": "imitation_contrast", **entry, "empirical": True,
            "outcome": "the memoryless policy stalls, the recurrent one reaches the goal",
            "scenario_ordinal": ordinal, "mission": good.mission,
            "selection_rule": "The first held-out world of lab 04 that separates "
                              "the two policies.",
        }
    )

    # -- recovery contrast and failure, asserted against stored rows --------
    rows_path = repo_root / "data" / "foundations" / "lab06" / "evaluation_rows.json"
    if not rows_path.exists():
        raise FoundationsError("lab06 evaluation rows missing; run grf run lab06")
    rows = json.loads(rows_path.read_text())
    unseen = {
        (row["arm"], row["scenario"]): row
        for row in rows
        if row["rep"] == 0 and row["slice"] == "unseen"
    }
    op_unseen = ActionDerangement(
        name=contract.perturbation.unseen_operator.name,
        action_ids=tuple(env_cfg.action_ids),
        mapping=tuple(contract.perturbation.unseen_operator.mapping),
    )
    base_model, vocab06 = _load_lab_model(repo_root, contract, "lab06/checkpoints/base_r0.pt")
    recovery_model, _ = _load_lab_model(
        repo_root, contract, "lab06/checkpoints/recovery_r0.pt"
    )

    def replay_arm(model, scenario: int, *, record_state: bool = False) -> Capture:
        row = unseen[("base", scenario)]
        corruption = (op_unseen, int(row["scheduled_time"]))
        capture = capture_policy_episode(
            env_cfg, int(row["seed"]), _PolicyDriver(model, vocab06, device),
            corruption=corruption, record_state=record_state,
        )
        return capture

    def assert_matches(capture: Capture, arm: str, scenario: int) -> None:
        row = unseen[(arm, scenario)]
        if capture.success != bool(row["success"]) or capture.delivered != bool(
            row["delivered"]
        ):
            raise FoundationsError(
                f"replay of {arm} on scenario {scenario} does not reproduce the "
                "stored evaluation row"
            )

    contrast_scenario = next(
        (
            scenario
            for scenario in sorted({s for (_a, s) in unseen})
            if unseen[("base", scenario)]["delivered"]
            and not unseen[("base", scenario)]["success"]
            and unseen[("recovery", scenario)]["success"]
        ),
        None,
    )
    if contrast_scenario is None:
        raise FoundationsError("no recovery-contrast scenario in lab06 rows")
    base_capture = replay_arm(base_model, contrast_scenario)
    recovery_capture = replay_arm(recovery_model, contrast_scenario, record_state=True)
    assert_matches(base_capture, "base", contrast_scenario)
    assert_matches(recovery_capture, "recovery", contrast_scenario)
    combined = _side_by_side(
        _annotate(base_capture, title="base policy", color=BROWN, env_cfg=env_cfg),
        _annotate(recovery_capture, title="recovery-trained policy", color=SAGE,
                  env_cfg=env_cfg),
    )
    entry = _write_video(combined, out_dir, "recovery_contrast")
    atomic_write_json(
        trajectories_dir / "recovery_contrast.json",
        trajectory_document(
            recovery_capture, env_cfg,
            source={
                "kind": "lab06 recovery arm, replicate 0, unseen operator",
                "scenario_ordinal": contrast_scenario,
            },
        ),
        overwrite=True,
    )
    manifest.append(
        {
            "id": "recovery_contrast", **entry, "empirical": True,
            "outcome": "the base policy stalls, the recovery policy reaches the goal",
            "scenario_ordinal": contrast_scenario,
            "mission": recovery_capture.mission,
            "corruption_time": recovery_capture.corruption_time,
            "selection_rule": "The first unseen scenario where the corruption "
                              "lands and the two policies part ways.",
            "trace": "media/trajectories/recovery_contrast.json",
        }
    )

    failure_scenario = next(
        (
            scenario
            for scenario in sorted({s for (_a, s) in unseen})
            if unseen[("recovery", scenario)]["delivered"]
            and not unseen[("recovery", scenario)]["success"]
        ),
        None,
    )
    if failure_scenario is None:
        raise FoundationsError("no recovery-failure scenario in lab06 rows")
    failure_capture = replay_arm(recovery_model, failure_scenario)
    assert_matches(failure_capture, "recovery", failure_scenario)
    entry = _write_video(
        _annotate(failure_capture, title="recovery-trained policy", color=SAGE,
                  env_cfg=env_cfg),
        out_dir, "recovery_failure",
    )
    manifest.append(
        {
            "id": "recovery_failure", **entry, "empirical": True,
            "outcome": "the recovery policy runs out of steps",
            "scenario_ordinal": failure_scenario,
            "mission": failure_capture.mission,
            "corruption_time": failure_capture.corruption_time,
            "selection_rule": "The first unseen scenario where the corruption "
                              "lands and the recovery policy misses it too.",
        }
    )

    document = {
        "schema_version": "1.0.0",
        "fps": FPS,
        "items": manifest,
        "trajectories": [
            {
                "id": path.stem,
                "href": f"media/trajectories/{path.name}",
                "sha256": file_sha256(path),
            }
            for path in sorted(trajectories_dir.glob("*.json"))
        ],
    }
    atomic_write_json(out_dir / "media_manifest.json", document, overwrite=True)
    return document


def run(repo_root: Path, *, force: bool) -> dict[str, object]:
    out_dir = repo_root / "foundations" / "media"
    if (out_dir / "media_manifest.json").exists() and not force:
        raise FoundationsError(
            f"{out_dir} already holds a media set; pass --force to regenerate"
        )
    if out_dir.exists():
        import shutil

        # Replace only this command's outputs; network/ belongs to
        # `grf network-trace` and survives a media regeneration.
        for child in out_dir.iterdir():
            if child.name == "network":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    document = build_media(repo_root, out_dir)
    return {
        "items": len(document["items"]),
        "trajectories": len(document["trajectories"]),
        "out_dir": str(out_dir),
    }
