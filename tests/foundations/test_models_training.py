"""Lab model family and BC machinery: parity, statelessness, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gr_foundations import training
from gr_foundations.models import LabPolicy
from grounded_recovery.model import RecoveryPolicy


@pytest.fixture(scope="module")
def contract():
    return training.contract_config(Path.cwd())


@pytest.fixture(scope="module")
def tiny_dataset(contract):
    episodes, counters = training.build_bc_dataset(
        contract.environment, 4, "lab04.dataset"
    )
    vocab = training.dataset_vocabulary(episodes)
    return episodes, vocab, counters


def test_full_lab_policy_matches_study_parameter_count(contract) -> None:
    lab = LabPolicy(contract.model, vocab_size=21, num_actions=3)
    study = RecoveryPolicy(contract.model, vocab_size=21, num_actions=3)
    assert lab.parameter_count() == study.parameter_count()


def test_variant_switches_change_exactly_the_intended_modules(contract) -> None:
    full = LabPolicy(contract.model, 21, 3)
    no_mission = LabPolicy(contract.model, 21, 3, use_mission=False)
    no_prev = LabPolicy(contract.model, 21, 3, use_prev_action=False)
    memoryless = LabPolicy(contract.model, 21, 3, use_memory=False)
    bow = LabPolicy(contract.model, 21, 3, mission_encoder="bow")
    assert not hasattr(no_mission, "word_embedding")
    assert not hasattr(no_prev, "action_embedding")
    assert not hasattr(memoryless, "policy_gru") and hasattr(memoryless, "trunk")
    assert hasattr(bow, "language_projection") and not hasattr(bow, "language_gru")
    counts = {m.parameter_count() for m in (full, no_mission, no_prev, memoryless, bow)}
    assert len(counts) == 5  # every switch changes the parameterization


def test_memoryless_step_ignores_hidden_state(contract, tiny_dataset) -> None:
    episodes, vocab, _ = tiny_dataset
    model = LabPolicy(contract.model, vocab.size, 3, use_memory=False)
    model.eval()
    device = torch.device("cpu")
    batch = training.collate_episodes(episodes[:1], vocab, device)
    mission_feature = model.encode_mission(
        batch["mission_tokens"], batch["mission_lengths"]
    )
    image = batch["image"][:, 0]
    direction = batch["direction"][:, 0]
    prev = batch["prev_action"][:, 0]
    with torch.no_grad():
        logits_none, _ = model.step(image, direction, prev, mission_feature, None)
        fake_hidden = torch.randn(1, contract.model.policy_gru)
        logits_fake, _ = model.step(image, direction, prev, mission_feature, fake_hidden)
    assert torch.equal(logits_none, logits_fake)


@pytest.mark.parametrize("kind", ["full", "memoryless"])
def test_step_matches_forward(contract, tiny_dataset, kind) -> None:
    episodes, vocab, _ = tiny_dataset
    model = LabPolicy(contract.model, vocab.size, 3, use_memory=(kind == "full"))
    model.eval()
    device = torch.device("cpu")
    batch = training.collate_episodes(episodes[:1], vocab, device)
    steps = int(batch["step_mask"][0].sum())
    with torch.no_grad():
        forward_logits, _ = model(
            batch["image"],
            batch["direction"],
            batch["prev_action"],
            batch["mission_tokens"],
            batch["mission_lengths"],
            batch["step_mask"],
        )
        mission_feature = model.encode_mission(
            batch["mission_tokens"], batch["mission_lengths"]
        )
        hidden = None
        for t in range(steps):
            step_logits, hidden = model.step(
                batch["image"][:, t],
                batch["direction"][:, t],
                batch["prev_action"][:, t],
                mission_feature,
                hidden,
            )
            assert torch.allclose(step_logits, forward_logits[:, t], atol=1e-6)


def test_collate_start_token_and_prev_shift(tiny_dataset) -> None:
    from gr_foundations.training import START_ACTION_TOKEN

    episodes, vocab, _ = tiny_dataset
    batch = training.collate_episodes(episodes, vocab, torch.device("cpu"))
    for row, episode in enumerate(episodes):
        steps = len(episode.actions)
        assert int(batch["prev_action"][row, 0]) == START_ACTION_TOKEN
        if steps > 1:
            assert np.array_equal(
                batch["prev_action"][row, 1:steps].numpy(), episode.actions[:-1]
            )
        assert bool(batch["step_mask"][row, :steps].all())
        assert not bool(batch["step_mask"][row, steps:].any())
        assert np.array_equal(batch["targets"][row, :steps].numpy(), episode.actions)


def test_masked_step_cross_entropy_hand_computed() -> None:
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    targets = torch.tensor([[0, 2]])
    mask_first = torch.tensor([[True, False]])
    loss_first = training.masked_step_cross_entropy(logits, targets, mask_first)
    assert loss_first.item() == pytest.approx(0.0, abs=1e-3)
    mask_both = torch.tensor([[True, True]])
    loss_both = training.masked_step_cross_entropy(logits, targets, mask_both)
    assert loss_both.item() == pytest.approx(5.0, abs=0.01)  # (0 + 10) / 2


def test_train_bc_is_bit_deterministic(contract, tiny_dataset) -> None:
    episodes, vocab, _ = tiny_dataset
    device = training.resolve_device()

    def factory():
        return LabPolicy(contract.model, vocab.size, 3, use_memory=False)

    digests = []
    for _ in range(2):
        model, _log = training.train_bc(
            factory, episodes, vocab, updates=5, batch_episodes=2, seed=1234, device=device
        )
        digests.append(training.model_digest(model))
    assert digests[0] == digests[1]
    model, _log = training.train_bc(
        factory, episodes, vocab, updates=5, batch_episodes=2, seed=999, device=device
    )
    assert training.model_digest(model) != digests[0]


def test_checkpoint_roundtrip_and_digest_guard(contract, tiny_dataset, tmp_path) -> None:
    episodes, vocab, _ = tiny_dataset
    torch.manual_seed(3)
    model = LabPolicy(contract.model, vocab.size, 3)
    path = tmp_path / "ckpt" / "model.pt"
    training.save_checkpoint(path, model, {"kind": "test", "vocabulary": list(vocab.tokens)})
    assert path.with_suffix(".json").exists()

    def factory():
        return LabPolicy(contract.model, vocab.size, 3)

    restored, meta = training.load_checkpoint(path, factory)
    assert training.model_digest(restored) == training.model_digest(model)
    assert meta["kind"] == "test"
    assert tuple(meta["vocabulary"]) == vocab.tokens
    # A tampered file must not load quietly.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    name = next(iter(bundle["state_dict"]))
    bundle["state_dict"][name] = bundle["state_dict"][name] + 1.0
    torch.save(bundle, path)
    with pytest.raises(Exception, match="digest"):
        training.load_checkpoint(path, factory)


def test_closed_loop_and_open_loop_bounds(contract, tiny_dataset) -> None:
    episodes, vocab, _ = tiny_dataset
    device = training.resolve_device()
    model = LabPolicy(contract.model, vocab.size, 3).to(device)
    outcome = training.closed_loop_success(
        model, vocab, contract.environment, [episodes[0].seed], device
    )
    assert outcome["episodes"] == 1
    assert 0.0 <= outcome["success_rate"] <= 1.0
    accuracy = training.open_loop_accuracy(model, episodes, vocab, device)
    assert 0.0 <= accuracy <= 1.0
