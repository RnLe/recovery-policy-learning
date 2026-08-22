"""Lab 1: observation decomposition invertibility, census determinism, run smoke."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gr_foundations import lab01_world
from gr_foundations.common import LabPaths, derive_seed
from grounded_recovery.world import WorldSession


@pytest.fixture(scope="module")
def env_cfg():
    from pathlib import Path

    return lab01_world.contract_environment(
        LabPaths(lab_id="lab01", repo_root=Path.cwd())
    )


def test_decompose_recompose_roundtrip(env_cfg) -> None:
    session = WorldSession(env_cfg)
    try:
        result = session.reset(derive_seed("lab01.census", 0))
    finally:
        session.close()
    planes = lab01_world.decompose_observation(result.image)
    assert set(planes) == {"object", "color", "state"}
    assert np.array_equal(lab01_world.recompose_observation(planes), result.image)
    # Planes are copies: mutating them must not touch the source image.
    assert not np.shares_memory(planes["object"], result.image)
    assert not np.shares_memory(planes["color"], result.image)


def test_describe_cell_known_triples() -> None:
    assert lab01_world.describe_cell(np.array([2, 5, 0])) == "wall"
    assert lab01_world.describe_cell(np.array([7, 5, 0])) == "grey box"
    assert lab01_world.describe_cell(np.array([4, 0, 1])) == "red door (closed)"


def test_parse_mission_on_real_missions(env_cfg) -> None:
    session = WorldSession(env_cfg)
    try:
        for index in range(5):
            mission = session.reset(derive_seed("lab01.census", index)).mission
            parsed = lab01_world.parse_mission(mission)
            assert parsed is not None, mission
            color, kind = parsed
            assert color in mission and kind in mission
    finally:
        session.close()
    assert lab01_world.parse_mission("open the red door") is None


def test_census_is_deterministic(env_cfg) -> None:
    first = lab01_world.run_census(env_cfg, 6)
    second = lab01_world.run_census(env_cfg, 6)
    assert first == second
    assert first["n_seeds"] == 6
    assert sum(first["mission_kind_counts"].values()) + first["unparsed_missions"] == 6


def test_run_produces_labelled_artifacts(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml", tmp_path / "configs" / "experiment_contract.yaml"
    )
    paths = LabPaths(lab_id="lab01", repo_root=tmp_path)
    summary = lab01_world.run(paths, force=False, census_seeds=6, contrast_seeds=4)
    assert summary["census_seeds"] == 6
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    assert metrics["metrics"]["census"]["n_seeds"] == 6
    assert "EXPLORATORY" in metrics["evidence_label"]
    for name in (
        "world_gallery.svg",
        "action_effects.svg",
        "observation_anatomy.svg",
        "mission_distribution.svg",
    ):
        assert (paths.figures_dir / name).exists()
        assert (paths.report_dir / "figures" / name).exists()
    assert (paths.report_dir / "world_facts.typ").exists()
    assert (paths.out_dir / "mini_report.md").exists()
