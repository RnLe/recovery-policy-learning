"""Lab 3: policy determinism, wall-follower rules, sync-experiment accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gr_foundations import lab03_oracle
from gr_foundations.common import LabPaths, derive_seed
from gr_foundations.lab01_world import contract_environment


@pytest.fixture(scope="module")
def env_cfg():
    return contract_environment(LabPaths(lab_id="lab03", repo_root=Path.cwd()))


def _view(ahead_obj: int, right_obj: int) -> np.ndarray:
    image = np.ones((7, 7, 3), dtype=np.uint8)  # all "empty"
    image[:, :, 1] = 0
    image[:, :, 2] = 0
    image[3, 5, 0] = ahead_obj
    image[4, 6, 0] = right_obj
    return image


def _step_result(image: np.ndarray):
    from grounded_recovery.world import StepResult

    return StepResult(
        image=image, direction=0, mission="go to the red ball",
        reward=0.0, terminated=False, truncated=False,
    )


def test_wall_follower_prefers_right_then_forward_then_left() -> None:
    wall = 2
    policy = lab03_oracle.WallFollowerPolicy()
    # Right open -> turn right, then commit to forward on the next step.
    assert policy.act(_step_result(_view(ahead_obj=1, right_obj=1)), 0) == lab03_oracle.RIGHT
    assert policy.act(_step_result(_view(ahead_obj=1, right_obj=1)), 1) == lab03_oracle.FORWARD
    # Right blocked, ahead open -> forward.
    policy = lab03_oracle.WallFollowerPolicy()
    assert policy.act(_step_result(_view(ahead_obj=1, right_obj=wall)), 0) == lab03_oracle.FORWARD
    # Both blocked -> left.
    assert policy.act(_step_result(_view(ahead_obj=wall, right_obj=wall)), 1) == lab03_oracle.LEFT


def test_random_policy_is_seed_deterministic() -> None:
    seed = derive_seed("lab03.random_policy", 3)
    first = [lab03_oracle.RandomPolicy(seed).act(None, t) for t in range(10)]
    second = [lab03_oracle.RandomPolicy(seed).act(None, t) for t in range(10)]
    assert first == second
    assert set(first) <= {0, 1, 2}


def test_policy_episode_deterministic(env_cfg) -> None:
    seed = derive_seed("lab03.oracle_eval", 0)
    policy_seed = derive_seed("lab03.random_policy", 0)
    first = lab03_oracle.run_policy_episode(env_cfg, seed, lab03_oracle.RandomPolicy(policy_seed))
    second = lab03_oracle.run_policy_episode(env_cfg, seed, lab03_oracle.RandomPolicy(policy_seed))
    assert first == second


def test_sync_experiment_accounting_and_determinism(env_cfg) -> None:
    first = lab03_oracle.run_sync_experiment(env_cfg, 6)
    second = lab03_oracle.run_sync_experiment(env_cfg, 6)
    assert first == second
    assert first["pairs_delivered"] + first["pairs_undelivered"] == 6
    for protocol in lab03_oracle.SYNC_PROTOCOLS:
        assert sum(first["counts"][protocol].values()) == first["pairs_delivered"]
    # The honest protocol is the preflight-validated one: it should not fail
    # on these early forced actions.
    assert first["counts"]["honest"]["success"] == first["pairs_delivered"]


def test_run_produces_labelled_artifacts(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = LabPaths(lab_id="lab03", repo_root=tmp_path)
    lab03_oracle.run(paths, force=False, spectrum_episodes=8, sync_pairs=5)
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    assert metrics["metrics"]["spectrum"]["oracle"]["episodes"] > 0
    assert "path_lengths" not in metrics["metrics"]["spectrum"]["oracle"]
    for name in (
        "policy_spectrum.svg",
        "synchronization_experiment.svg",
        "labelled_trajectory.svg",
    ):
        assert (paths.figures_dir / name).exists()
    assert (paths.report_dir / "policy_facts.typ").exists()
