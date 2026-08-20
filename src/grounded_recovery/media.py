"""Rollout media: annotated comparison animations from real evaluated episodes.

Every empirical animation deterministically replays an episode from the
confirmatory evaluation, with the same checkpoint, seed, schedule, and greedy rule,
with rendering enabled, and the replayed outcome is asserted to match the
stored evaluation row. Selection follows disclosed deterministic rules (never
visual cherry-picking), failures are shown alongside successes, and
illustrative material (the scripted oracle) is labelled as such on every
frame. A failing arm that stops changing is shown two seconds past the point
it froze, with the freeze step and the true step limit printed on the frame
and recorded in the manifest; an arm that keeps moving is never cut, and long
episodes are shown as a labelled time-lapse instead. Repeated oscillation is
not treated as idle: only a genuinely unchanging world qualifies.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grounded_recovery.config import ExperimentConfig
from grounded_recovery.data import ManifestError, Vocabulary, start_action_token
from grounded_recovery.oracle import SynchronizedOracle
from grounded_recovery.perturbations import ActionDerangement
from grounded_recovery.schemas import ManifestEntry, canonical_scenario_hash
from grounded_recovery.world import WorldSession

SCALE = 2  # 320 px grid frames are upscaled (nearest) to 640 px
HEADER = 44
FOOTER = 34
HOLD_FRAMES = 8  # hold the final annotated frame so the outcome is readable
FPS = 5
IDLE_HOLD_SECONDS = 2.0  # how long a frozen failing arm is shown past its freeze

TEAL = (42, 157, 143)
BLUE = (72, 120, 168)
RED = (186, 60, 50)
AMBER = (192, 124, 0)
INK = (24, 24, 28)
PAPER = (247, 245, 240)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class RolloutCapture:
    """One rendered episode: frames[t] is the world before transition t."""

    frames: list[np.ndarray]
    executed: list[int]
    proposal_at_scheduled: int | None
    scheduled_time: int | None
    delivered: bool
    success: bool
    truncated: bool
    mission: str


@dataclass(frozen=True)
class IdleCut:
    """A failing arm that stopped changing, and where its animation stops."""

    idle_from: int  # first frame index from which every later frame is identical
    last_shown: int  # last frame index rendered
    total: int  # the arm's true final step


def first_idle_frame(frames: Sequence[np.ndarray]) -> int | None:
    """Smallest index from which every later frame is identical, else None."""
    index = len(frames) - 1
    while index > 0 and np.array_equal(frames[index - 1], frames[index]):
        index -= 1
    return None if index == len(frames) - 1 else index


def idle_cut(
    capture: RolloutCapture,
    *,
    hold_seconds: float = IDLE_HOLD_SECONDS,
    fps: int = FPS,
) -> IdleCut | None:
    """Where to stop a failing arm that froze; successes are never cut.

    A policy that pushes into a wall renders the same frame forever, and the
    remaining steps carry no information. Showing the freeze plus a readable
    hold keeps the pair watchable without hiding the outcome: the true step
    limit stays on the frame and in the manifest.
    """
    if capture.success:
        return None
    idle_from = first_idle_frame(capture.frames)
    if idle_from is None:
        return None
    total = len(capture.frames) - 1
    last_shown = min(idle_from + round(hold_seconds * fps), total)
    return None if last_shown >= total else IdleCut(idle_from, last_shown, total)


def _outcome_text(capture: RolloutCapture, cut: IdleCut | None) -> str:
    """The banner wording; a cut arm states where it froze and the true limit."""
    if capture.success:
        return "SUCCESS"
    if cut is not None:
        return f"FAILURE (stuck from step {cut.idle_from}, step limit {cut.total})"
    return "FAILURE (step limit)" if capture.truncated else "FAILURE"


def capture_policy_rollout(
    cfg: ExperimentConfig,
    policy,
    vocab: Vocabulary,
    entry: ManifestEntry,
    *,
    scheduled_time: int | None,
    operator: ActionDerangement | None,
) -> RolloutCapture:
    """Deterministic re-run of one greedy evaluation episode with rendering."""
    import torch

    session = WorldSession(cfg.environment, render_mode="rgb_array")
    try:
        observation = session.reset(entry.environment_seed)
        if canonical_scenario_hash(session.scenario_state()) != entry.canonical_scenario_hash:
            raise ManifestError("media replay does not reproduce the manifested world")
        frames = [session.render_frame()]
        executed_actions: list[int] = []
        proposal_at_scheduled: int | None = None
        delivered = False
        num_actions = len(cfg.environment.action_ids)
        policy.eval()
        device = next(policy.parameters()).device
        with torch.no_grad():
            encoded = vocab.encode(observation.mission)
            mission_feature = policy.encode_mission(
                torch.tensor([encoded], dtype=torch.long, device=device),
                torch.tensor([len(encoded)], dtype=torch.long, device=device),
            )
            hidden = None
            last_token = start_action_token(num_actions)
            while not session.done:
                t = session.time
                image = torch.from_numpy(
                    observation.image.astype("int64")
                ).unsqueeze(0).to(device)
                direction = torch.tensor(
                    [observation.direction], dtype=torch.long, device=device
                )
                previous = torch.tensor([last_token], dtype=torch.long, device=device)
                logits, hidden = policy.step(
                    image, direction, previous, mission_feature, hidden
                )
                proposal = int(logits.argmax(dim=-1).item())
                if scheduled_time is not None and t == scheduled_time:
                    executed = operator.apply(proposal)
                    proposal_at_scheduled = proposal
                    delivered = True
                else:
                    executed = proposal
                observation = session.step(executed)
                executed_actions.append(executed)
                frames.append(session.render_frame())
                last_token = executed
        return RolloutCapture(
            frames=frames,
            executed=executed_actions,
            proposal_at_scheduled=proposal_at_scheduled,
            scheduled_time=scheduled_time,
            delivered=delivered,
            success=observation.terminated,
            truncated=observation.truncated,
            mission=observation.mission,
        )
    finally:
        session.close()


def capture_oracle_rollout(cfg: ExperimentConfig, entry: ManifestEntry) -> RolloutCapture:
    """Nominal scripted-oracle episode with rendering (illustrative only)."""
    session = WorldSession(cfg.environment, render_mode="rgb_array")
    try:
        observation = session.reset(entry.environment_seed)
        if canonical_scenario_hash(session.scenario_state()) != entry.canonical_scenario_hash:
            raise ManifestError("media replay does not reproduce the manifested world")
        oracle = SynchronizedOracle(session)
        frames = [session.render_frame()]
        executed_actions: list[int] = []
        last_executed: int | None = None
        while not session.done:
            recommended = oracle.recommend(last_executed, session.time)
            observation = session.step(recommended)
            executed_actions.append(recommended)
            frames.append(session.render_frame())
            last_executed = recommended
        return RolloutCapture(
            frames=frames,
            executed=executed_actions,
            proposal_at_scheduled=None,
            scheduled_time=None,
            delivered=False,
            success=observation.terminated,
            truncated=observation.truncated,
            mission=observation.mission,
        )
    finally:
        session.close()


def _annotate(
    cfg: ExperimentConfig,
    capture: RolloutCapture,
    *,
    label: str,
    color: tuple[int, int, int],
    stride: int,
    cut: IdleCut | None = None,
) -> list[Image.Image]:
    """Per-frame header/footer annotation; corruption and outcome called out."""
    names = cfg.environment.action_names
    font_big = _font(20)
    font_small = _font(15)
    images: list[Image.Image] = []
    total = len(capture.frames)
    last = cut.last_shown if cut is not None else total - 1
    kept = set(range(0, last + 1, stride))
    kept.add(last)
    if capture.scheduled_time is not None and capture.delivered:
        # The corruption call-out frame must never fall out of a time-lapse.
        kept.add(min(capture.scheduled_time + 1, last))
    kept = sorted(kept)
    for t in kept:
        frame = capture.frames[t]
        grid = Image.fromarray(frame).resize(
            (frame.shape[1] * SCALE, frame.shape[0] * SCALE), Image.NEAREST
        )
        width = grid.width
        canvas = Image.new("RGB", (width, HEADER + grid.height + FOOTER), PAPER)
        canvas.paste(grid, (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, width, HEADER], fill=color)
        draw.text((10, 10), label, font=font_big, fill=(255, 255, 255))
        status = f"step {t}/{total - 1}"
        if stride > 1:
            status += f" · {stride}x time-lapse"
        if cut is not None and t >= cut.idle_from:
            status += f" · unchanged since step {cut.idle_from}"
        corruption_frame = (
            capture.scheduled_time is not None
            and capture.delivered
            and t == capture.scheduled_time + 1
        )
        if corruption_frame:
            proposal = names[capture.proposal_at_scheduled]
            forced = names[capture.executed[capture.scheduled_time]]
            status = (
                f"t={capture.scheduled_time}: corruption  "
                f"proposal '{proposal}' -> executed '{forced}'"
            )
            draw.rectangle(
                [0, HEADER, width - 1, HEADER + grid.height - 1],
                outline=AMBER,
                width=4 * SCALE,
            )
        if t == last:
            outcome = _outcome_text(capture, cut)
            status = outcome
            draw.rectangle(
                [0, HEADER + grid.height, width, HEADER + grid.height + FOOTER],
                fill=TEAL if capture.success else RED,
            )
            # The cut wording is long; drop a size rather than let it overflow.
            font_outcome = font_big
            if font_outcome.getlength(outcome) > width - 20:
                font_outcome = font_small
            draw.text(
                (10, HEADER + grid.height + (7 if font_outcome is font_big else 9)),
                outcome, font=font_outcome, fill=(255, 255, 255),
            )
        else:
            draw.text(
                (10, HEADER + grid.height + 8), status, font=font_small,
                fill=AMBER if corruption_frame else INK,
            )
        images.append(canvas)
    images.extend([images[-1]] * HOLD_FRAMES)
    return images


def _side_by_side(
    left: list[Image.Image], right: list[Image.Image], title: str
) -> list[Image.Image]:
    length = max(len(left), len(right))
    left = left + [left[-1]] * (length - len(left))
    right = right + [right[-1]] * (length - len(right))
    gap = 8
    title_height = 36
    font = _font(17)
    composed = []
    for a, b in zip(left, right, strict=True):
        width = a.width + gap + b.width
        height = title_height + max(a.height, b.height)
        canvas = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 9), title, font=font, fill=INK)
        canvas.paste(a, (0, title_height))
        canvas.paste(b, (a.width + gap, title_height))
        composed.append(canvas)
    return composed


def _titled(frames: list[Image.Image], title: str) -> list[Image.Image]:
    title_height = 36
    font = _font(17)
    composed = []
    for frame in frames:
        canvas = Image.new("RGB", (frame.width, title_height + frame.height), PAPER)
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 9), title, font=font, fill=INK)
        canvas.paste(frame, (0, title_height))
        composed.append(canvas)
    return composed


def _even(image: Image.Image) -> Image.Image:
    width = image.width - image.width % 2
    height = image.height - image.height % 2
    return image.crop((0, 0, width, height))


def write_gif(images: list[Image.Image], path: Path, *, fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def write_mp4(images: list[Image.Image], path: Path, *, fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise ManifestError("ffmpeg is required for mp4 export")
    with tempfile.TemporaryDirectory() as tmp:
        for index, image in enumerate(images):
            _even(image).save(Path(tmp) / f"{index:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-framerate", str(fps),
                "-i", f"{tmp}/%04d.png",
                "-pix_fmt", "yuv420p",
                "-vcodec", "libx264",
                "-crf", "24",
                str(path),
            ],
            check=True,
        )


def _stride_for(*shown_lengths: int) -> int:
    """Time-lapse factor for the longest arm that is actually rendered."""
    longest = max(shown_lengths)
    if longest > 144:
        return 3
    if longest > 72:
        return 2
    return 1


def generate_result_media(
    cfg: ExperimentConfig,
    results_dir: Path,
    manifest_root: Path,
    data_root: Path,
    out_dir: Path,
    *,
    bundle_id: str = "B00",
) -> dict[str, object]:
    """Render the disclosed-rule media set from the confirmatory evaluation.

    Selection rules (deterministic, defined post-opening and disclosed here,
    they choose by ordinal, never by appearance):

    1. ``unseen_paired_contrast``: the smallest eligible-unseen scenario
       ordinal of the reference bundle where the recovery arm succeeded and
       the extra-demonstrations arm failed (both corruptions delivered).
    2. ``unseen_recovery_failure``: the smallest ordinal where the recovery
       arm failed with a delivered corruption (honesty: failures are shown).
    3. ``oracle_nominal``: visualization-split ordinal 0 under the scripted
       oracle; illustrative mechanics, labelled not-a-result.

    Every empirical replay must reproduce the stored evaluation outcome
    exactly, or generation aborts.

    Presentation, applied after selection and disclosed in the manifest under
    ``presentation``: an arm that fails by freezing is rendered two seconds
    past its freeze, with the freeze step and the true step limit on the
    banner; both panes of a pair share one time-lapse stride, which stays 1
    whenever the rendered arms are short enough.
    """
    from grounded_recovery.artifacts import file_sha256, read_json
    from grounded_recovery.config import contract_hash
    from grounded_recovery.data import (
        base_dataset_dir,
        load_eligible_entries,
        load_split_manifest,
        vocabulary_from_dataset,
    )
    from grounded_recovery.evaluate import load_arm_policy
    from grounded_recovery.perturbations import operator_from_config
    from grounded_recovery.publish import group_rows, rows_from_jsonl

    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
    grouped = group_rows(rows)
    extra_rows = {r.scenario_ordinal: r for r in
                  grouped[(bundle_id, "extra_demonstrations", "unseen")]}
    recovery_rows = {r.scenario_ordinal: r for r in
                     grouped[(bundle_id, "recovery_aggregation", "unseen")]}

    def smallest(predicate) -> int:
        for ordinal in sorted(recovery_rows):
            if predicate(extra_rows[ordinal], recovery_rows[ordinal]):
                return ordinal
        raise ManifestError("no scenario satisfies the selection rule")

    contrast_ordinal = smallest(
        lambda e, r: r.success and not e.success and e.delivered and r.delivered
    )
    failure_ordinal = smallest(lambda e, r: (not r.success) and r.delivered)

    eligible_entries, _ = load_eligible_entries(cfg, manifest_root)
    by_ordinal = {entry.ordinal: entry for entry in eligible_entries}
    unseen_operator = operator_from_config(
        cfg.perturbation.unseen_operator, cfg.environment.action_ids
    )

    summary = read_json(
        Path(data_root) / contract_hash(cfg)[:12] / bundle_id / "bundle_summary.json"
    )
    vocab = vocabulary_from_dataset(base_dataset_dir(cfg, bundle_id, data_root))
    policies = {
        "extra_demonstrations": load_arm_policy(
            cfg, summary["arms"]["extra_demonstrations"]["final_checkpoint"], vocab
        ),
        "recovery_aggregation": load_arm_policy(
            cfg, summary["arms"]["recovery_aggregation"]["final_checkpoint"], vocab
        ),
    }

    def paired_animation(ordinal: int, stem: str, rule: str) -> dict[str, object]:
        entry = by_ordinal[ordinal]
        stored_extra, stored_recovery = extra_rows[ordinal], recovery_rows[ordinal]
        scheduled = stored_recovery.scheduled_time
        captures = {}
        for arm, stored in (("extra_demonstrations", stored_extra),
                            ("recovery_aggregation", stored_recovery)):
            capture = capture_policy_rollout(
                cfg, policies[arm], vocab, entry,
                scheduled_time=scheduled, operator=unseen_operator,
            )
            if capture.success != stored.success or capture.delivered != stored.delivered:
                raise ManifestError(
                    f"media replay of {arm} on ordinal {ordinal} does not match the "
                    "stored evaluation row"
                )
            captures[arm] = capture
        cuts = {arm: idle_cut(capture) for arm, capture in captures.items()}
        shown = {
            arm: (cuts[arm].last_shown if cuts[arm] else len(capture.frames) - 1) + 1
            for arm, capture in captures.items()
        }
        # Both panes share one stride so a frame index still means a step.
        stride = _stride_for(*shown.values())
        title = (
            f'"{captures["recovery_aggregation"].mission}" · unseen corruption '
            f"(rot_minus) at t={scheduled} · bundle {bundle_id} · "
            f"scenario {ordinal} · empirical"
        )
        frames = _side_by_side(
            _annotate(cfg, captures["extra_demonstrations"],
                      label="extra demonstrations", color=BLUE, stride=stride,
                      cut=cuts["extra_demonstrations"]),
            _annotate(cfg, captures["recovery_aggregation"],
                      label="recovery aggregation", color=TEAL, stride=stride,
                      cut=cuts["recovery_aggregation"]),
            title,
        )
        gif_path = out_dir / f"{stem}.gif"
        mp4_path = out_dir / f"{stem}.mp4"
        write_gif(frames, gif_path)
        write_mp4(frames, mp4_path)
        return {
            "artifact_id": stem,
            "kind": "empirical",
            "selection_rule": rule,
            "bundle": bundle_id,
            "scenario_ordinal": ordinal,
            "environment_seed": entry.environment_seed,
            "scenario_hash": entry.canonical_scenario_hash,
            "slice": "unseen",
            "scheduled_time": scheduled,
            "operator": unseen_operator.name,
            "outcomes": {
                arm: {"success": captures[arm].success, "steps": len(captures[arm].executed)}
                for arm in captures
            },
            "presentation": {
                "time_lapse_stride": stride,
                "idle_cut": {
                    arm: None if cuts[arm] is None else {
                        "unchanged_from_step": cuts[arm].idle_from,
                        "shown_through_step": cuts[arm].last_shown,
                        "true_final_step": cuts[arm].total,
                    }
                    for arm in captures
                },
            },
            "files": {
                "gif": {"path": gif_path.name, "sha256": file_sha256(gif_path)},
                "mp4": {"path": mp4_path.name, "sha256": file_sha256(mp4_path)},
            },
        }

    entries = [
        paired_animation(
            contrast_ordinal, "unseen_paired_contrast",
            "smallest eligible-unseen ordinal with recovery success and "
            "extra-demonstrations failure (both delivered), reference bundle",
        ),
        paired_animation(
            failure_ordinal, "unseen_recovery_failure",
            "smallest eligible-unseen ordinal where the recovery arm failed "
            "with a delivered corruption, reference bundle",
        ),
    ]

    visualization_entries, _ = load_split_manifest(manifest_root, "visualization")
    oracle_capture = capture_oracle_rollout(cfg, visualization_entries[0])
    stride = _stride_for(len(oracle_capture.frames))
    oracle_frames = _annotate(
        cfg, oracle_capture,
        label="scripted oracle · illustrative protocol animation, not a result",
        color=(90, 90, 96), stride=stride,
    )
    title = (
        f'"{oracle_capture.mission}" · nominal oracle rollout · visualization '
        "split · illustrative"
    )
    oracle_frames = _titled(oracle_frames, title)
    gif_path = out_dir / "oracle_nominal.gif"
    mp4_path = out_dir / "oracle_nominal.mp4"
    write_gif(oracle_frames, gif_path)
    write_mp4(oracle_frames, mp4_path)
    from grounded_recovery.artifacts import atomic_write_json
    from grounded_recovery.artifacts import file_sha256 as fsha

    entries.append(
        {
            "artifact_id": "oracle_nominal",
            "kind": "illustrative",
            "selection_rule": "visualization-split ordinal 0 under the scripted oracle",
            "scenario_ordinal": visualization_entries[0].ordinal,
            "environment_seed": visualization_entries[0].environment_seed,
            "scenario_hash": visualization_entries[0].canonical_scenario_hash,
            "slice": "visualization",
            "outcomes": {"oracle": {"success": oracle_capture.success,
                                    "steps": len(oracle_capture.executed)}},
            "files": {
                "gif": {"path": gif_path.name, "sha256": fsha(gif_path)},
                "mp4": {"path": mp4_path.name, "sha256": fsha(mp4_path)},
            },
        }
    )
    manifest = {"media": entries, "contract_hash": contract_hash(cfg)}
    atomic_write_json(out_dir / "media_manifest.json", manifest, overwrite=True)
    return manifest
