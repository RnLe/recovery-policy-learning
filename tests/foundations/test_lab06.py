"""Lab 6: exact budgets, window-only labels, masked loss, tiny mini-study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr_foundations import lab06_shift
from gr_foundations.common import LabPaths
from gr_foundations.training import (
    collate_episodes,
    contract_config,
    dataset_vocabulary,
    masked_step_cross_entropy,
    resolve_device,
)
from grounded_recovery.model import RecoveryPolicy


@pytest.fixture(scope="module")
def contract():
    return contract_config(Path.cwd())


def test_extra_demo_budget_is_exact(contract) -> None:
    episodes, counters = lab06_shift.collect_extra_demos(
        contract.environment, budget_labels=25, seed_offset=0
    )
    assert counters["labels"] == 25
    assert sum(int(e.label_mask.sum()) for e in episodes) == 25
    for episode in episodes:
        assert episode.label_mask.all()
        assert np.array_equal(episode.actions, episode.target_actions)


def test_recovery_collection_budget_and_window(contract) -> None:
    env_cfg = contract.environment
    device = resolve_device()
    torch.manual_seed(0)
    model = RecoveryPolicy(contract.model, vocab_size=21, num_actions=3).to(device)
    vocab_source, _ = lab06_shift.build_bc_dataset(env_cfg, 3, "lab06.dataset")
    vocab = dataset_vocabulary(vocab_source)
    operator = lab06_shift._operator(
        contract.perturbation.collection_operator, tuple(env_cfg.action_ids)
    )
    episodes, counters = lab06_shift.collect_recovery(
        model, vocab, env_cfg, operator, budget_labels=12,
        seed_offset=0, times=(2, 3), window=4, device=device,
    )
    assert counters["labels"] == 12
    assert sum(int(e.label_mask.sum()) for e in episodes) == 12
    for episode in episodes:
        # Window starts strictly after the corruption at t* >= 2.
        assert not episode.label_mask[:3].any()
        assert episode.label_mask.sum() >= 1
    # Determinism: identical seeds and model give identical collections.
    episodes_again, counters_again = lab06_shift.collect_recovery(
        model, vocab, env_cfg, operator, budget_labels=12,
        seed_offset=0, times=(2, 3), window=4, device=device,
    )
    assert counters_again == counters
    assert [e.seed for e in episodes_again] == [e.seed for e in episodes]
    assert all(
        np.array_equal(a.label_mask, b.label_mask)
        for a, b in zip(episodes, episodes_again, strict=True)
    )


def test_loss_ignores_unlabelled_steps(contract) -> None:
    env_cfg = contract.environment
    source, _ = lab06_shift.build_bc_dataset(env_cfg, 1, "lab06.dataset")
    episode = source[0]
    sparse_mask = np.zeros(len(episode.actions), dtype=bool)
    sparse_mask[-2:] = True
    corrupted_targets = episode.target_actions.copy()
    corrupted_targets[~sparse_mask] = (corrupted_targets[~sparse_mask] + 1) % 3
    from dataclasses import replace

    sparse = replace(episode, label_mask=sparse_mask)
    sparse_garbage = replace(
        episode, label_mask=sparse_mask, target_actions=corrupted_targets
    )
    vocab = dataset_vocabulary(source)
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = RecoveryPolicy(contract.model, vocab.size, 3)
    losses = []
    for candidate in (sparse, sparse_garbage):
        batch = collate_episodes([candidate], vocab, device)
        with torch.no_grad():
            logits, _ = model(
                batch["image"], batch["direction"], batch["prev_action"],
                batch["mission_tokens"], batch["mission_lengths"], batch["step_mask"],
            )
            losses.append(
                float(masked_step_cross_entropy(logits, batch["targets"], batch["target_mask"]))
            )
    assert losses[0] == pytest.approx(losses[1])


def test_finetune_starts_from_base_and_is_deterministic(contract) -> None:
    env_cfg = contract.environment
    device = resolve_device()
    dataset, _ = lab06_shift.build_bc_dataset(env_cfg, 3, "lab06.dataset")
    vocab = dataset_vocabulary(dataset)

    def factory():
        return RecoveryPolicy(contract.model, vocab.size, 3)

    torch.manual_seed(7)
    base = factory()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    from gr_foundations.training import model_digest

    untouched, _stats = lab06_shift.finetune_arm(
        base_state, factory, dataset, dataset, vocab,
        updates=0, seed=11, device=device,
    )
    assert model_digest(untouched) == model_digest(base)
    first, _ = lab06_shift.finetune_arm(
        base_state, factory, dataset, dataset, vocab,
        updates=3, seed=11, device=device,
    )
    second, _ = lab06_shift.finetune_arm(
        base_state, factory, dataset, dataset, vocab,
        updates=3, seed=11, device=device,
    )
    assert model_digest(first) == model_digest(second)
    assert model_digest(first) != model_digest(base)


def test_run_tiny_mini_study(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = LabPaths(lab_id="lab06", repo_root=tmp_path)
    summary = lab06_shift.run(
        paths,
        force=False,
        base_episodes=6,
        holdout_scenarios=4,
        budget_labels=10,
        window=3,
        base_updates=12,
        arm_updates=8,
        n_seeds=1,
        sweep_times=(2,),
        sweep_scenarios=2,
    )
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    assert set(metrics["metrics"]["success_matrix"]) == set(lab06_shift.ARM_NAMES)
    rows = json.loads((paths.data_dir / "evaluation_rows.json").read_text())
    assert len(rows) == 4 * 3 * 3  # scenarios x slices x arms (one replicate)
    for counters in metrics["metrics"]["collection"]:
        assert counters["extra"]["labels"] == 10
        assert counters["recovery"]["labels"] == 10
    for name in ("three_arm_results.svg", "shift_anatomy.svg"):
        assert (paths.figures_dir / name).exists()
    assert (paths.report_dir / "derangements.typ").exists()
    assert summary["device"] in ("cuda", "cpu")
    for arm in lab06_shift.ARM_NAMES:
        assert (paths.data_dir / "checkpoints" / f"{arm}_r0.pt").exists()
