"""Idle detection and the disclosed cut of a frozen failing arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grounded_recovery.config import load_and_validate
from grounded_recovery.media import (
    FOOTER,
    HEADER,
    HOLD_FRAMES,
    RED,
    IdleCut,
    _annotate,
    _outcome_text,
    _stride_for,
    first_idle_frame,
    idle_cut,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    return load_and_validate(REPO_ROOT / "configs" / "pilot.yaml")


def _frames(moving: int, total: int) -> list[np.ndarray]:
    """One pixel changes for ``moving`` transitions, then the world stops."""
    frames = []
    for index in range(total):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[0, 0] = min(index, moving)
        frames.append(frame)
    return frames


def _capture(frames, *, success: bool, truncated: bool = True):
    from grounded_recovery.media import RolloutCapture

    return RolloutCapture(
        frames=frames,
        executed=[0] * (len(frames) - 1),
        proposal_at_scheduled=None,
        scheduled_time=None,
        delivered=False,
        success=success,
        truncated=truncated,
        mission="go to the grey box",
    )


def test_first_idle_frame_finds_the_freeze() -> None:
    assert first_idle_frame(_frames(moving=6, total=145)) == 6
    assert first_idle_frame(_frames(moving=144, total=145)) is None
    assert first_idle_frame(_frames(moving=0, total=5)) == 0


def test_idle_cut_only_trims_failures_that_freeze_early() -> None:
    frozen = _frames(moving=6, total=145)
    assert idle_cut(_capture(frozen, success=False)) == IdleCut(6, 16, 144)
    # A successful arm is shown to the end even if it waits at the goal.
    assert idle_cut(_capture(frozen, success=True)) is None
    # A freeze inside the hold window would cut nothing, so it does not.
    assert idle_cut(_capture(_frames(moving=140, total=145), success=False)) is None
    # An arm that keeps moving is never cut.
    assert idle_cut(_capture(_frames(moving=144, total=145), success=False)) is None


def test_outcome_text_states_the_freeze_and_the_true_limit() -> None:
    frozen = _capture(_frames(moving=6, total=145), success=False)
    assert _outcome_text(frozen, None) == "FAILURE (step limit)"
    assert (
        _outcome_text(frozen, IdleCut(6, 16, 144))
        == "FAILURE (stuck from step 6, step limit 144)"
    )
    assert _outcome_text(_capture(frozen.frames, success=True), None) == "SUCCESS"


def test_annotate_stops_at_the_cut_and_paints_the_failure_banner(cfg) -> None:
    capture = _capture(_frames(moving=6, total=145), success=False)
    cut = idle_cut(capture)
    images = _annotate(
        cfg, capture, label="extra demonstrations", color=(72, 120, 168),
        stride=1, cut=cut,
    )
    assert len(images) == cut.last_shown + 1 + HOLD_FRAMES
    banner = images[-1]
    grid_height = banner.height - HEADER - FOOTER
    assert banner.getpixel((banner.width - 5, HEADER + grid_height + 3)) == RED
    # Uncut arms still run to the end.
    full = _annotate(
        cfg, capture, label="extra demonstrations", color=(72, 120, 168), stride=1
    )
    assert len(full) == len(capture.frames) + HOLD_FRAMES


def test_annotate_respects_the_cut_under_a_time_lapse(cfg) -> None:
    capture = _capture(_frames(moving=6, total=145), success=False)
    cut = idle_cut(capture)
    images = _annotate(
        cfg, capture, label="x", color=(0, 0, 0), stride=3, cut=cut
    )
    kept = sorted({*range(0, cut.last_shown + 1, 3), cut.last_shown})
    assert len(images) == len(kept) + HOLD_FRAMES


def test_stride_follows_what_is_actually_shown() -> None:
    assert _stride_for(17, 22) == 1  # the landing pair, once the frozen arm is cut
    assert _stride_for(32, 145) == 3  # the failure pair, whose recovery arm keeps moving
    assert _stride_for(80, 10) == 2
