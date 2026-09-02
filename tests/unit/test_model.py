"""Model contract: shapes, causality, padding inertness, reset, step parity.

These tests protect the closed-loop claim: the policy trained offline must be
exactly the function evaluated online. Any future-input leakage, padding
sensitivity, hidden-state carryover, or step/forward divergence would make the
evaluation measure a different function than the one trained.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from grounded_recovery.config import load_and_validate
from grounded_recovery.model import RecoveryPolicy, model_config_hash

PILOT_YAML = Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml"

NUM_ACTIONS = 3
VOCAB_SIZE = 12


@pytest.fixture(scope="module")
def model_cfg():
    return load_and_validate(PILOT_YAML).model


@pytest.fixture(scope="module")
def model(model_cfg):
    torch.manual_seed(1234)
    policy = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    policy.eval()
    return policy


def make_inputs(batch: int, steps: int, mission_len: int = 5, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    image = torch.stack(
        (
            torch.randint(0, 11, (batch, steps, 7, 7), generator=generator),
            torch.randint(0, 6, (batch, steps, 7, 7), generator=generator),
            torch.randint(0, 4, (batch, steps, 7, 7), generator=generator),
        ),
        dim=-1,
    )
    direction = torch.randint(0, 4, (batch, steps), generator=generator)
    prev_action = torch.randint(0, NUM_ACTIONS + 1, (batch, steps), generator=generator)
    mission = torch.randint(2, VOCAB_SIZE, (batch, mission_len), generator=generator)
    mission_lengths = torch.full((batch,), mission_len, dtype=torch.long)
    step_mask = torch.ones((batch, steps), dtype=torch.bool)
    return image, direction, prev_action, mission, mission_lengths, step_mask


def test_forward_signature_is_the_privilege_boundary() -> None:
    # The model can only ever receive these inputs; goals, coordinates, oracle
    # state, or arm identity have no parameter to arrive through.
    parameters = list(inspect.signature(RecoveryPolicy.forward).parameters)
    assert parameters == [
        "self",
        "image",
        "direction",
        "prev_executed_action",
        "mission_tokens",
        "mission_lengths",
        "step_mask",
        "initial_hidden",
    ]


def test_shape_ledger(model, model_cfg) -> None:
    for batch, steps in ((1, 1), (3, 9)):
        inputs = make_inputs(batch, steps)
        logits, final_hidden = model(*inputs)
        assert logits.shape == (batch, steps, NUM_ACTIONS)
        assert logits.dtype == torch.float32
        assert final_hidden.shape == (batch, model_cfg.policy_gru)


def test_padding_invariance(model) -> None:
    # The same window alone versus padded inside a longer batch must produce
    # identical logits at valid positions: padding is inert.
    image, direction, prev_action, mission, mission_lengths, _ = make_inputs(1, 4)
    alone_logits, alone_hidden = model(
        image, direction, prev_action, mission, mission_lengths,
        torch.ones((1, 4), dtype=torch.bool),
    )
    pad = 3
    padded_image = torch.cat([image, torch.zeros((1, pad, 7, 7, 3), dtype=torch.long)], 1)
    padded_direction = torch.cat([direction, torch.zeros((1, pad), dtype=torch.long)], 1)
    padded_prev = torch.cat([prev_action, torch.zeros((1, pad), dtype=torch.long)], 1)
    padded_mask = torch.cat(
        [torch.ones((1, 4), dtype=torch.bool), torch.zeros((1, pad), dtype=torch.bool)], 1
    )
    padded_logits, padded_hidden = model(
        padded_image, padded_direction, padded_prev, mission, mission_lengths, padded_mask
    )
    # Tolerance instead of bitwise equality: CPU kernels may tile differently
    # for different batch shapes; the invariance is mathematical.
    assert torch.allclose(alone_logits, padded_logits[:, :4], atol=1e-6)
    assert torch.allclose(alone_hidden, padded_hidden, atol=1e-6)


@pytest.mark.gpu
def test_mission_padding_invariance(model) -> None:
    image, direction, prev_action, mission, mission_lengths, mask = make_inputs(1, 3)
    padded_mission = torch.cat([mission, torch.zeros((1, 4), dtype=torch.long)], 1)
    base_logits, _ = model(image, direction, prev_action, mission, mission_lengths, mask)
    padded_logits, _ = model(
        image, direction, prev_action, padded_mission, mission_lengths, mask
    )
    assert torch.equal(base_logits, padded_logits)


def test_causality_future_inputs(model) -> None:
    inputs = make_inputs(1, 6)
    logits, _ = model(*inputs)
    image, direction, prev_action, mission, mission_lengths, mask = make_inputs(
        1, 6, seed=99
    )
    # Perturb inputs strictly after t=2; logits at t<=2 must be unchanged.
    mixed_image = inputs[0].clone()
    mixed_image[:, 3:] = image[:, 3:]
    mixed_direction = inputs[1].clone()
    mixed_direction[:, 3:] = direction[:, 3:]
    mixed_prev = inputs[2].clone()
    mixed_prev[:, 3:] = prev_action[:, 3:]
    mixed_logits, _ = model(
        mixed_image, mixed_direction, mixed_prev, inputs[3], inputs[4], inputs[5]
    )
    assert torch.equal(logits[:, :3], mixed_logits[:, :3])
    assert not torch.equal(logits[:, 3:], mixed_logits[:, 3:])


def test_executed_action_causality(model) -> None:
    inputs = make_inputs(1, 6)
    logits, _ = model(*inputs)
    changed_prev = inputs[2].clone()
    changed_prev[0, 3] = (changed_prev[0, 3] + 1) % (NUM_ACTIONS + 1)
    changed_logits, _ = model(
        inputs[0], inputs[1], changed_prev, inputs[3], inputs[4], inputs[5]
    )
    # The change is visible at t=3 and later, never earlier.
    assert torch.equal(logits[:, :3], changed_logits[:, :3])
    assert not torch.equal(logits[:, 3], changed_logits[:, 3])


def test_recurrent_reset(model, model_cfg) -> None:
    inputs = make_inputs(2, 5)
    zeros = torch.zeros((2, model_cfg.policy_gru))
    default_logits, _ = model(*inputs)
    explicit_logits, _ = model(*inputs, initial_hidden=zeros)
    assert torch.equal(default_logits, explicit_logits)
    # A second forward must not be influenced by the first (no hidden carryover).
    repeat_logits, _ = model(*inputs)
    assert torch.equal(default_logits, repeat_logits)


def test_step_matches_forward(model) -> None:
    image, direction, prev_action, mission, mission_lengths, mask = make_inputs(2, 7)
    forward_logits, forward_hidden = model(
        image, direction, prev_action, mission, mission_lengths, mask
    )
    mission_feature = model.encode_mission(mission, mission_lengths)
    hidden = None
    step_logits = []
    for t in range(7):
        logits_t, hidden = model.step(
            image[:, t], direction[:, t], prev_action[:, t], mission_feature, hidden
        )
        step_logits.append(logits_t)
    stacked = torch.stack(step_logits, dim=1)
    assert torch.allclose(forward_logits, stacked, atol=1e-6)
    assert torch.allclose(forward_hidden, hidden, atol=1e-6)


def test_finite_gradients(model_cfg) -> None:
    torch.manual_seed(5)
    trainable = RecoveryPolicy(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    inputs = make_inputs(2, 4)
    logits, _ = trainable(*inputs)
    loss = logits.square().mean()
    loss.backward()
    for name, parameter in trainable.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_model_config_hash_sensitivity(model_cfg) -> None:
    base = model_config_hash(model_cfg, VOCAB_SIZE, NUM_ACTIONS)
    assert base != model_config_hash(model_cfg, VOCAB_SIZE + 1, NUM_ACTIONS)
    import dataclasses

    changed = dataclasses.replace(model_cfg, policy_gru=model_cfg.policy_gru * 2)
    assert base != model_config_hash(changed, VOCAB_SIZE, NUM_ACTIONS)
