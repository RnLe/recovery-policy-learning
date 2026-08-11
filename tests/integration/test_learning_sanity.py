"""Tiny-overfit learning sanity and end-to-end training determinism (gate G3).

A model that cannot drive its loss to near zero on a handful of fixed windows
has a broken optimization path, and any later arm comparison would be
meaningless. Determinism across repeated runs proves that a replicate is
identified by its seed bundle and nothing else.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from grounded_recovery.artifacts import read_jsonl
from grounded_recovery.config import load_and_validate
from grounded_recovery.data import (
    base_dataset_dir,
    collate_windows,
    collect_base,
    make_manifests,
    vocabulary_from_dataset,
)
from grounded_recovery.integrity import GENESIS_HASH
from grounded_recovery.train import load_all_windows, load_checkpoint, train_base

TINY_N0 = 25
OVERRIDES = {
    "data.n0": TINY_N0,
    "training.base_updates": 250,
    "training.base_targets_per_update": 8,
    "training.learning_rate": 3.0e-3,
}


def state_dict_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_config_factory):
    cfg = load_and_validate(tiny_config_factory(**OVERRIDES))
    manifest_root = tmp_path_factory.mktemp("manifests")
    make_manifests(cfg, manifest_root)
    data_root = tmp_path_factory.mktemp("data")
    collect_base(cfg, "B00", manifest_root, data_root)
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    out_dir = tmp_path_factory.mktemp("run")
    result = train_base(cfg, "B00", dataset_dir, out_dir)
    return cfg, dataset_dir, out_dir, result


def test_tiny_fixed_dataset_overfits(trained) -> None:
    cfg, dataset_dir, _out, result = trained
    assert result.window_count == TINY_N0
    assert result.final_loss < 0.05
    # Reload the trained checkpoint and require argmax accuracy 1.0 on the
    # complete training set. If this ever flakes, raise the update budget,
    # never weaken the accuracy assertion.
    from grounded_recovery.config import contract_hash
    from grounded_recovery.model import RecoveryPolicy, model_config_hash

    vocab = vocabulary_from_dataset(dataset_dir)
    num_actions = len(cfg.environment.action_ids)
    payload = load_checkpoint(
        result.checkpoint_path,
        expected_contract_hash=contract_hash(cfg),
        expected_model_config_hash=model_config_hash(cfg.model, vocab.size, num_actions),
        expected_action_ids=cfg.environment.action_ids,
        expected_vocab=vocab.tokens,
    )
    model = RecoveryPolicy(cfg.model, vocab.size, num_actions)
    model.load_state_dict(payload["model_state"])
    model.eval()
    windows = load_all_windows(cfg, dataset_dir, vocab)
    batch = collate_windows(windows)
    with torch.no_grad():
        logits, _ = model(
            batch.image,
            batch.direction,
            batch.prev_executed_action,
            batch.mission_tokens,
            batch.mission_lengths,
            batch.step_mask,
        )
    predictions = logits[batch.target_mask].argmax(dim=-1)
    targets = batch.targets[batch.target_mask]
    assert torch.equal(predictions, targets), "trained model must fit all training targets"


def test_training_deterministic_given_seeds(trained, tmp_path_factory) -> None:
    cfg, dataset_dir, _out, result = trained
    rerun_dir = tmp_path_factory.mktemp("run-repeat")
    rerun = train_base(cfg, "B00", dataset_dir, rerun_dir)
    first = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    second = torch.load(rerun.checkpoint_path, map_location="cpu", weights_only=False)
    assert state_dict_digest(first["model_state"]) == state_dict_digest(
        second["model_state"]
    )
    assert result.final_loss == rerun.final_loss


def test_metrics_ledger_chain_and_lineage(trained) -> None:
    _cfg, _dataset, out_dir, result = trained
    rows = read_jsonl(out_dir / "training" / "base" / "metrics.jsonl")
    assert len(rows) == 250
    # Recompute the hash chain independently.
    from grounded_recovery.artifacts import canonical_json_bytes, sha256_hex

    prev = GENESIS_HASH
    for row in rows:
        payload = {k: v for k, v in row.items() if k != "row_hash"}
        assert row["prev_row_hash"] == prev
        assert row["row_hash"] == sha256_hex(
            prev.encode("ascii") + canonical_json_bytes(payload)
        )
        prev = row["row_hash"]
    assert prev == result.metrics_final_hash
    # The checkpoint's meta must reference exactly this metrics chain.
    sidecar = json.loads(
        (out_dir / "checkpoints" / "base_final.json").read_text()
    )
    assert sidecar["metrics_ledger_hash"] == result.metrics_final_hash
    assert sidecar["arm"] == "base"
    assert sidecar["update_index"] == 250
    # Exposure accounting: every update drew exactly the configured targets.
    assert all(row["loss_denominator"] == 8 for row in rows)
    assert rows[-1]["cumulative_target_exposures"] == 250 * 8


def test_train_refuses_wrong_target_count(trained, tiny_config_factory) -> None:
    from grounded_recovery.train import CheckpointMismatchError

    cfg, dataset_dir, _out, _result = trained
    import dataclasses

    wrong = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, n0=TINY_N0 + 1)
    )
    with pytest.raises(CheckpointMismatchError, match="targets"):
        train_base(wrong, "B00", dataset_dir, dataset_dir.parent / "run-wrong")
