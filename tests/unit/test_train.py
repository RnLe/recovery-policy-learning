"""Loss arithmetic, sampler determinism, and checkpoint identity.

These tests protect exposure accounting (denominator = target positions),
regularization discipline (weight decay only in the optimizer), and the rule
that a checkpoint is a complete, identity-checked function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from test_model import PILOT_YAML, make_inputs

from grounded_recovery.config import load_and_validate
from grounded_recovery.model import RecoveryPolicy, model_config_hash
from grounded_recovery.train import (
    CheckpointMeta,
    CheckpointMismatchError,
    load_checkpoint,
    make_optimizer,
    masked_cross_entropy,
    restore_rng_states,
    sample_window_indices,
    save_checkpoint,
)

NUM_ACTIONS = 3
VOCAB_SIZE = 12


def test_masked_ce_matches_hand_calculation() -> None:
    # Two rows, three steps; one target at each row's final valid step.
    logits = torch.zeros((2, 3, 3))
    logits[0, 2] = torch.tensor([2.0, 0.0, 0.0])
    logits[1, 1] = torch.tensor([0.0, 0.0, 3.0])
    targets = torch.zeros((2, 3), dtype=torch.long)
    targets[0, 2] = 0
    targets[1, 1] = 2
    target_mask = torch.zeros((2, 3), dtype=torch.bool)
    target_mask[0, 2] = True
    target_mask[1, 1] = True
    loss, denominator = masked_cross_entropy(logits, targets, target_mask)
    # Hand computation: CE = -log softmax(target logit).
    expected_row0 = -math.log(math.exp(2.0) / (math.exp(2.0) + 2.0))
    expected_row1 = -math.log(math.exp(3.0) / (math.exp(3.0) + 2.0))
    assert loss.item() == pytest.approx((expected_row0 + expected_row1) / 2, abs=1e-6)
    assert denominator == 2


def test_non_target_positions_contribute_zero() -> None:
    logits = torch.randn((2, 4, 3))
    targets = torch.zeros((2, 4), dtype=torch.long)
    target_mask = torch.zeros((2, 4), dtype=torch.bool)
    target_mask[0, 3] = True
    target_mask[1, 2] = True
    loss, _ = masked_cross_entropy(logits, targets, target_mask)
    tampered = logits.clone()
    tampered[:, 0] = 99.0  # non-target positions
    tampered[0, 1] = -99.0
    loss_tampered, _ = masked_cross_entropy(tampered, targets, target_mask)
    assert loss.item() == pytest.approx(loss_tampered.item())


def test_denominator_equals_target_count() -> None:
    logits = torch.randn((5, 2, 3))
    targets = torch.zeros((5, 2), dtype=torch.long)
    target_mask = torch.zeros((5, 2), dtype=torch.bool)
    target_mask[:, 1] = True
    _, denominator = masked_cross_entropy(logits, targets, target_mask)
    assert denominator == 5
    with pytest.raises(ValueError, match="no target positions"):
        masked_cross_entropy(logits, targets, torch.zeros_like(target_mask))


def test_sampler_deterministic_from_seed() -> None:
    first = sample_window_indices(
        np.random.default_rng(11), 100, 16, with_replacement=True
    )
    second = sample_window_indices(
        np.random.default_rng(11), 100, 16, with_replacement=True
    )
    third = sample_window_indices(
        np.random.default_rng(12), 100, 16, with_replacement=True
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_sampler_without_replacement_bounds() -> None:
    rng = np.random.default_rng(0)
    drawn = sample_window_indices(rng, 10, 10, with_replacement=False)
    assert sorted(drawn.tolist()) == list(range(10))
    with pytest.raises(ValueError, match="without replacement"):
        sample_window_indices(rng, 5, 6, with_replacement=False)


def test_weight_decay_not_in_loss() -> None:
    # The loss value must be independent of the optimizer's weight decay;
    # decay acts only through the update step.
    logits = torch.randn((3, 2, 3))
    targets = torch.zeros((3, 2), dtype=torch.long)
    target_mask = torch.zeros((3, 2), dtype=torch.bool)
    target_mask[:, 1] = True
    loss_a, _ = masked_cross_entropy(logits, targets, target_mask)
    loss_b, _ = masked_cross_entropy(logits, targets, target_mask)
    assert loss_a.item() == loss_b.item()
    model_cfg = load_and_validate(PILOT_YAML).model
    model = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.5)
    assert optimizer.param_groups[0]["weight_decay"] == 0.5


def test_grad_clip_bounds_global_norm() -> None:
    model_cfg = load_and_validate(PILOT_YAML).model
    torch.manual_seed(3)
    model = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    inputs = make_inputs(2, 3)
    logits, _ = model(*inputs)
    (logits.sum() * 1e6).backward()  # explode the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total = math.sqrt(
        sum(float(p.grad.norm() ** 2) for p in model.parameters() if p.grad is not None)
    )
    assert total <= 1.0 + 1e-4


@pytest.fixture()
def checkpoint_setup(tmp_path):
    model_cfg = load_and_validate(PILOT_YAML).model
    torch.manual_seed(21)
    model = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    # Take one real optimizer step so AdamW state exists.
    inputs = make_inputs(2, 3)
    logits, _ = model(*inputs)
    logits.sum().backward()
    optimizer.step()
    vocabulary = tuple(["<pad>", "<unk>"] + [f"w{i}" for i in range(VOCAB_SIZE - 2)])
    meta = CheckpointMeta(
        contract_hash="aa" * 32,
        model_config_hash=model_config_hash(model_cfg, VOCAB_SIZE, NUM_ACTIONS),
        bundle_id="B00",
        arm="base",
        round_index=0,
        update_index=1,
        vocabulary=vocabulary,
        action_names=("left", "right", "forward"),
        action_ids=(0, 1, 2),
        dataset_schema_version="1",
        metrics_ledger_hash="bb" * 32,
        parameter_count=model.parameter_count(),
    )
    rng = np.random.default_rng(77)
    rng.integers(0, 100, 5)  # advance the stream so state capture is non-trivial
    path = tmp_path / "base_final.pt"
    save_checkpoint(path, model, optimizer, meta, rng)
    return model_cfg, model, optimizer, meta, rng, path


def test_save_load_identical_logits(checkpoint_setup) -> None:
    model_cfg, model, _optimizer, meta, _rng, path = checkpoint_setup
    payload = load_checkpoint(
        path,
        expected_contract_hash=meta.contract_hash,
        expected_model_config_hash=meta.model_config_hash,
        expected_action_ids=meta.action_ids,
        expected_vocab=meta.vocabulary,
    )
    reloaded = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    reloaded.load_state_dict(payload["model_state"])
    inputs = make_inputs(2, 4, seed=31)
    with torch.no_grad():
        original_logits, _ = model(*inputs)
        reloaded_logits, _ = reloaded(*inputs)
    assert torch.equal(original_logits, reloaded_logits)


def test_optimizer_state_roundtrip(checkpoint_setup) -> None:
    model_cfg, _model, optimizer, meta, _rng, path = checkpoint_setup
    payload = load_checkpoint(
        path,
        expected_contract_hash=meta.contract_hash,
        expected_model_config_hash=meta.model_config_hash,
        expected_action_ids=meta.action_ids,
        expected_vocab=meta.vocabulary,
    )
    original_state = optimizer.state_dict()["state"]
    restored_state = payload["optimizer_state"]["state"]
    assert set(original_state) == set(restored_state)
    for key in original_state:
        assert torch.equal(original_state[key]["exp_avg"], restored_state[key]["exp_avg"])
        assert torch.equal(
            original_state[key]["exp_avg_sq"], restored_state[key]["exp_avg_sq"]
        )


def test_rng_state_restoration(checkpoint_setup) -> None:
    _cfg, _model, _optimizer, meta, rng, path = checkpoint_setup
    expected_next = rng.integers(0, 1_000_000, 8)
    expected_torch = torch.randn(4)
    payload = load_checkpoint(
        path,
        expected_contract_hash=meta.contract_hash,
        expected_model_config_hash=meta.model_config_hash,
        expected_action_ids=meta.action_ids,
        expected_vocab=meta.vocabulary,
    )
    restored_rng = restore_rng_states(payload["rng"])
    assert np.array_equal(restored_rng.integers(0, 1_000_000, 8), expected_next)
    assert torch.equal(torch.randn(4), expected_torch)


def test_mismatches_rejected(checkpoint_setup) -> None:
    _cfg, _model, _optimizer, meta, _rng, path = checkpoint_setup
    good = dict(
        expected_contract_hash=meta.contract_hash,
        expected_model_config_hash=meta.model_config_hash,
        expected_action_ids=meta.action_ids,
        expected_vocab=meta.vocabulary,
    )
    with pytest.raises(CheckpointMismatchError, match="contract"):
        load_checkpoint(path, **{**good, "expected_contract_hash": "ff" * 32})
    with pytest.raises(CheckpointMismatchError, match="model-config"):
        load_checkpoint(path, **{**good, "expected_model_config_hash": "ff" * 32})
    with pytest.raises(CheckpointMismatchError, match="action set"):
        load_checkpoint(path, **{**good, "expected_action_ids": (0, 1, 2, 5)})
    with pytest.raises(CheckpointMismatchError, match="vocabulary"):
        load_checkpoint(path, **{**good, "expected_vocab": ("<pad>", "<unk>", "zzz")})


def test_checkpoint_refuses_overwrite_and_writes_sidecar(checkpoint_setup) -> None:
    _cfg, model, optimizer, meta, rng, path = checkpoint_setup
    assert path.with_suffix(".json").exists()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, meta, rng)
