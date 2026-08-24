"""Lab 5: shape walkthrough correctness and tiny ablation run."""

from __future__ import annotations

import json
from pathlib import Path

from gr_foundations import lab05_architecture
from gr_foundations.training import contract_config
from grounded_recovery.model import RecoveryPolicy


def test_shape_walkthrough_covers_all_components_and_totals() -> None:
    contract = contract_config(Path.cwd())
    model = RecoveryPolicy(contract.model, vocab_size=21, num_actions=3)
    rows = lab05_architecture.shape_walkthrough(model, 21)
    names = [row[0] for row in rows]
    assert names[-1] == "total"
    assert set(names[:-1]) == {name for name, _ in model.named_children()}
    assert rows[-1][1] == model.parameter_count()
    component_sum = sum(row[1] for row in rows[:-1])
    assert component_sum == model.parameter_count()


def test_run_tiny_ablation(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = lab05_architecture.LabPaths(lab_id="lab05", repo_root=tmp_path)
    summary = lab05_architecture.run(
        paths,
        force=False,
        dataset_episodes=6,
        holdout_episodes=4,
        updates=12,
        batch_episodes=4,
        n_seeds=1,
    )
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    parity = metrics["metrics"]["parameter_parity"]
    assert parity["lab_policy"] == parity["recovery_policy"]
    assert set(metrics["metrics"]["ablation"]["results"]) == set(
        lab05_architecture.VARIANTS
    )
    assert (paths.figures_dir / "ablation_results.svg").exists()
    assert (paths.report_dir / "shape_walkthrough.typ").exists()
    assert (paths.report_dir / "design_rationale.typ").exists()
    assert summary["parameters"] > 10_000
