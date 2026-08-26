"""Journey media: capture determinism, annotation, text fit, trajectory schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from gr_foundations import media_journey
from gr_foundations.common import derive_seed
from gr_foundations.training import contract_config


@pytest.fixture(scope="module")
def env_cfg():
    return contract_config(Path.cwd()).environment


@pytest.fixture(scope="module")
def expert_capture(env_cfg):
    seed = derive_seed("lab03.trajectory", 0)
    return media_journey.capture_episode(
        env_cfg, seed, lambda _r, _t, label: label, with_oracle=True, record_state=True
    )


def test_capture_is_deterministic_and_consistent(env_cfg, expert_capture) -> None:
    seed = derive_seed("lab03.trajectory", 0)
    again = media_journey.capture_episode(
        env_cfg, seed, lambda _r, _t, label: label, with_oracle=True, record_state=True
    )
    assert again.actions == expert_capture.actions
    assert again.success and expert_capture.success
    assert len(expert_capture.frames) == expert_capture.steps + 1
    # State snapshots include the terminal state for the scrubber.
    assert len(expert_capture.grids) == expert_capture.steps + 1
    assert np.array_equal(again.frames[0], expert_capture.frames[0])


def test_annotation_layout_and_outcome_banner(env_cfg, expert_capture) -> None:
    images = media_journey._annotate(
        expert_capture, title="expert", color=media_journey.STEEL,
        env_cfg=env_cfg, show_labels=True,
    )
    frame_height = expert_capture.frames[0].shape[0] * media_journey.SCALE
    assert images[0].height == frame_height + media_journey.HEADER + media_journey.FOOTER
    assert images[0].width == expert_capture.frames[0].shape[1] * media_journey.SCALE
    assert len(images) == len(expert_capture.frames) + media_journey.HOLD_FRAMES
    # The banner is sage on success (sampled clear of the caption text).
    banner = np.asarray(images[-1])[-14, -20]
    assert tuple(banner) == media_journey.SAGE
    # A mid-episode clip may replace the banner with a neutral caption.
    neutral = media_journey._annotate(
        expert_capture, title="expert", color=media_journey.STEEL,
        env_cfg=env_cfg, final_caption="frozen here", hold=1,
    )
    assert tuple(np.asarray(neutral[-1])[-14, -20]) == media_journey.GOLD


def test_corruption_flash_marks_the_frame_after_delivery(env_cfg) -> None:
    from grounded_recovery.perturbations import ActionDerangement

    operator = ActionDerangement("rot_plus", env_cfg.action_ids, (1, 2, 0))
    capture = media_journey.capture_episode(
        env_cfg, derive_seed("lab03.trajectory", 0),
        lambda _r, _t, label: label, with_oracle=True,
        corruption=(operator, 2),
    )
    assert capture.delivered and capture.corruption_time == 2
    images = media_journey._annotate(
        capture, title="x", color=media_journey.BROWN, env_cfg=env_cfg, hold=1
    )
    edge = media_journey.HEADER + 3
    flash_frame = np.asarray(images[3])[edge, 3]
    before = np.asarray(images[2])[edge, 3]
    assert tuple(flash_frame) == media_journey.GOLD  # damage visible at t+1
    assert tuple(before) != media_journey.GOLD


def test_every_caption_string_fits_the_frame() -> None:
    # Worst realistic strings at the narrowest canvas the annotator produces.
    width = 320 * media_journey.SCALE
    budget = width - 2 * media_journey.MARGIN
    probe = ImageDraw.Draw(Image.new("RGB", (width, 40)))
    for text in (
        "t=143 · forward  ⚡ corrupted at t=99",
        "the oracle, labelling as it goes",
        "world B · oracle here: forward",
        "→ byte-identical observation",
        "FAILURE · 144 steps",
    ):
        fitted, font = media_journey._fit_text(probe, text, 30, budget)
        assert probe.textlength(fitted, font=font) <= budget


def test_trajectory_document_schema(env_cfg, expert_capture) -> None:
    document = media_journey.trajectory_document(
        expert_capture, env_cfg, source={"kind": "test"}
    )
    assert document["outcome"] == "success"
    assert document["schema_version"] == "1.1.0"
    assert len(document["steps"]) == expert_capture.steps + 1
    first = document["steps"][0]
    assert set(first) == {
        "t", "grid", "agent", "observation", "visible", "action", "label", "corrupted",
    }
    grid = np.asarray(first["grid"])
    assert grid.ndim == 3 and grid.shape[2] == 3
    assert np.asarray(first["observation"]).shape == (7, 7, 3)
    # The environment's own visibility test marks the agent's cell visible.
    assert [first["agent"]["x"], first["agent"]["y"]] in first["visible"]
    assert len(first["visible"]) <= 49
    # The terminal state closes the episode without an action.
    last = document["steps"][-1]
    assert last["action"] is None and last["label"] is None and not last["corrupted"]
    assert document["legend"]["objects"]["2"] == "wall"
    assert document["action_names"] == list(env_cfg.action_names)
    with pytest.raises(Exception, match="record_state"):
        bare = media_journey.capture_episode(
            env_cfg, derive_seed("lab03.trajectory", 0), lambda _r, _t, lab: lab,
            with_oracle=True,
        )
        media_journey.trajectory_document(bare, env_cfg, source={})


def test_side_by_side_pads_to_common_length(env_cfg, expert_capture) -> None:
    images = media_journey._annotate(
        expert_capture, title="x", color=media_journey.BROWN, env_cfg=env_cfg, hold=1
    )
    combined = media_journey._side_by_side(images[:3], images)
    assert len(combined) == len(images)
    assert combined[0].width == images[0].width * 2 + 6 * media_journey.SCALE
