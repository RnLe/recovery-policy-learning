"""Lab 4: Q-learning convergence on a fixture, tiny end-to-end run."""

from __future__ import annotations

import json

from gr_foundations import lab04_learning
from gr_foundations.common import LabPaths


def test_qlearning_learns_the_toy_task() -> None:
    result = lab04_learning.train_qlearning(episodes=250)
    assert result["final_greedy_success"] is True
    assert result["best_greedy_steps"] <= 8  # optimal is a handful of steps
    assert result["states_in_table"] <= 60  # genuinely tabular


def test_qlearning_deterministic() -> None:
    first = lab04_learning.train_qlearning(episodes=60)
    second = lab04_learning.train_qlearning(episodes=60)
    assert first == second


def test_run_tiny_end_to_end(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = LabPaths(lab_id="lab04", repo_root=tmp_path)
    summary = lab04_learning.run(
        paths,
        force=False,
        qlearning_episodes=60,
        dataset_episodes=6,
        holdout_episodes=4,
        updates=20,
        batch_episodes=4,
        n_seeds=1,
    )
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    bc = metrics["metrics"]["behavior_cloning"]
    assert set(bc["results"]) == {"memoryless", "recurrent"}
    for rows in bc["results"].values():
        assert len(rows) == 1
        assert 0.0 <= rows[0]["open_loop_accuracy"] <= 1.0
    for name in (
        "qlearning_curve.svg",
        "bc_learning_curves.svg",
        "accuracy_vs_success.svg",
    ):
        assert (paths.figures_dir / name).exists()
    assert (paths.report_dir / "rl_contrast.typ").exists()
    assert summary["device"] in ("cuda", "cpu")
    # Trained policies are persisted for the media/demo pipeline.
    from gr_foundations.training import contract_config, load_checkpoint, model_digest
    from grounded_recovery.model import RecoveryPolicy

    checkpoint = paths.data_dir / "checkpoints" / "recurrent_s0.pt"
    assert (paths.data_dir / "checkpoints" / "memoryless_s0.pt").exists()
    stored_digest = bc["results"]["recurrent"][0]["model_digest"]
    cfg = contract_config(tmp_path)
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    model, _meta = load_checkpoint(
        checkpoint, lambda: RecoveryPolicy(cfg.model, len(meta["vocabulary"]), 3)
    )
    assert model_digest(model) == stored_digest
