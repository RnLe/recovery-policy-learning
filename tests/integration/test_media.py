"""Rollout media generation: rendering, annotation, honest labelling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image, ImageSequence

from grounded_recovery.config import load_and_validate
from grounded_recovery.data import build_vocabulary, load_split_manifest
from grounded_recovery.media import (
    _annotate,
    _side_by_side,
    _stride_for,
    capture_oracle_rollout,
    capture_policy_rollout,
    write_gif,
)
from grounded_recovery.model import RecoveryPolicy
from grounded_recovery.perturbations import operator_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    return load_and_validate(REPO_ROOT / "configs" / "pilot.yaml")


@pytest.mark.gpu
def test_policy_capture_annotate_and_gif(cfg, tmp_path) -> None:
    entries, _ = load_split_manifest(REPO_ROOT / "manifests", "validation")
    entry = entries[0]
    torch.manual_seed(0)
    vocab = build_vocabulary([entry.mission])
    policy = RecoveryPolicy(
        cfg.model, vocab.size, len(cfg.environment.action_ids)
    ).to("cuda" if cfg.training.device == "cuda" else "cpu")
    operator = operator_from_config(
        cfg.perturbation.collection_operator, cfg.environment.action_ids
    )
    capture = capture_policy_rollout(
        cfg, policy, vocab, entry, scheduled_time=2, operator=operator
    )
    # frames[t] is the pre-transition world: one more frame than transitions.
    assert len(capture.frames) == len(capture.executed) + 1
    assert capture.frames[0].shape == (320, 320, 3)
    if capture.delivered:
        forced = capture.executed[2]
        assert forced == operator.apply(capture.proposal_at_scheduled)
    stride = _stride_for(len(capture.frames))
    annotated = _annotate(cfg, capture, label="test arm", color=(0, 100, 100),
                          stride=stride)
    composed = _side_by_side(annotated, annotated, "test title")
    gif_path = tmp_path / "test.gif"
    write_gif(composed[:12], gif_path)
    rendered = Image.open(gif_path)
    frames = list(ImageSequence.Iterator(rendered))
    assert len(frames) == 12
    assert frames[0].size[0] > 1200  # two annotated panels side by side


def test_oracle_capture_succeeds_and_matches_manifest_length(cfg) -> None:
    entries, _ = load_split_manifest(REPO_ROOT / "manifests", "visualization")
    capture = capture_oracle_rollout(cfg, entries[0])
    assert capture.success
    assert len(capture.executed) == entries[0].nominal_oracle_path_length
