"""Lab-scale behavior-cloning machinery: datasets, training, evaluation.

Deliberately simpler than the study's pipeline, with full episodes and a loss on
every step instead of one-target-per-window items, because that is the
textbook formulation the labs teach first; Lab 7 explains why the study
tightens it. Determinism mirrors the study: named seeds, deterministic torch
algorithms, all tensor work on the GPU when available, environment stepping on
the CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from gr_foundations.common import FoundationsError, derive_seed
from grounded_recovery.config import EnvironmentConfig, ExperimentConfig, load_and_validate
from grounded_recovery.data import Vocabulary, build_vocabulary, start_action_token
from grounded_recovery.oracle import OracleSupportError
from grounded_recovery.world import WorldSession

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# The frozen study action set has three actions; the model-level START token
# sits one past them (the same convention as the study's window materializer).
NUM_ACTIONS = 3
START_ACTION_TOKEN = start_action_token(NUM_ACTIONS)


def contract_config(repo_root) -> ExperimentConfig:
    """The frozen study contract, loaded read-only (environment + model dims)."""
    return load_and_validate(repo_root / "configs" / "experiment_contract.yaml")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_determinism() -> None:
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class BCEpisode:
    """One training episode: executed actions plus (possibly sparse) labels.

    For plain demonstrations the oracle drove, so ``actions`` (what was
    executed, the model's previous-action input) equals ``target_actions``
    (the supervision) and ``label_mask`` is all-true. Recovery episodes
    (Lab 6) separate the three: the learner's corrupted actions were
    executed, while oracle labels exist only inside the post-corruption
    window.
    """

    seed: int
    mission: str
    images: np.ndarray  # [T, 7, 7, 3] uint8
    directions: np.ndarray  # [T] int64
    actions: np.ndarray  # [T] int64, executed
    success: bool
    target_actions: np.ndarray  # [T] int64, labels where label_mask is set
    label_mask: np.ndarray  # [T] bool


def build_bc_dataset(
    env_cfg: EnvironmentConfig, n_episodes: int, seed_component: str
) -> tuple[list[BCEpisode], dict[str, int]]:
    """Deterministic oracle demonstrations; unsupported scenarios are skipped."""
    episodes: list[BCEpisode] = []
    counters = {"requested": n_episodes, "collected": 0, "oracle_unsupported": 0}
    for index in range(n_episodes):
        seed = derive_seed(seed_component, index)
        session = WorldSession(env_cfg)
        try:
            result = session.reset(seed)
            images, directions, actions = [], [], []
            try:
                from grounded_recovery.oracle import SynchronizedOracle

                oracle = SynchronizedOracle(session)
                last: int | None = None
                while not session.done:
                    recommendation = oracle.recommend(last, session.time)
                    images.append(result.image.copy())
                    directions.append(result.direction)
                    actions.append(recommendation)
                    result = session.step(recommendation)
                    last = recommendation
            except OracleSupportError:
                counters["oracle_unsupported"] += 1
                continue
            executed = np.asarray(actions, dtype=np.int64)
            episodes.append(
                BCEpisode(
                    seed=seed,
                    mission=result.mission,
                    images=np.stack(images).astype(np.uint8),
                    directions=np.asarray(directions, dtype=np.int64),
                    actions=executed,
                    success=bool(result.terminated and result.reward > 0.0),
                    target_actions=executed.copy(),
                    label_mask=np.ones(len(executed), dtype=bool),
                )
            )
            counters["collected"] += 1
        finally:
            session.close()
    if not episodes:
        raise FoundationsError("dataset collection produced no episodes")
    return episodes, counters


def dataset_vocabulary(episodes: list[BCEpisode]) -> Vocabulary:
    return build_vocabulary([episode.mission for episode in episodes])


def collate_episodes(
    episodes: list[BCEpisode], vocab: Vocabulary, device: torch.device
) -> dict[str, torch.Tensor]:
    """Right-pad full episodes into one batch; padding exists only here."""
    batch = len(episodes)
    max_steps = max(len(episode.actions) for episode in episodes)
    tokenized = [vocab.encode(episode.mission) for episode in episodes]
    max_words = max(len(tokens) for tokens in tokenized)

    image = torch.zeros(batch, max_steps, 7, 7, 3, dtype=torch.long)
    direction = torch.zeros(batch, max_steps, dtype=torch.long)
    prev_action = torch.zeros(batch, max_steps, dtype=torch.long)
    targets = torch.zeros(batch, max_steps, dtype=torch.long)
    step_mask = torch.zeros(batch, max_steps, dtype=torch.bool)
    target_mask = torch.zeros(batch, max_steps, dtype=torch.bool)
    mission_tokens = torch.zeros(batch, max_words, dtype=torch.long)
    mission_lengths = torch.zeros(batch, dtype=torch.long)

    for row, (episode, tokens) in enumerate(zip(episodes, tokenized, strict=True)):
        steps = len(episode.actions)
        image[row, :steps] = torch.from_numpy(episode.images.astype(np.int64))
        direction[row, :steps] = torch.from_numpy(episode.directions)
        targets[row, :steps] = torch.from_numpy(episode.target_actions)
        # The model's previous-action input: START at the absolute episode
        # start, then the truly *executed* action (which differs from the
        # label in recovery episodes).
        prev_action[row, 0] = START_ACTION_TOKEN
        if steps > 1:
            prev_action[row, 1:steps] = torch.from_numpy(episode.actions[:-1])
        step_mask[row, :steps] = True
        target_mask[row, :steps] = torch.from_numpy(episode.label_mask)
        mission_tokens[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mission_lengths[row] = len(tokens)

    return {
        "image": image.to(device),
        "direction": direction.to(device),
        "prev_action": prev_action.to(device),
        "targets": targets.to(device),
        "step_mask": step_mask.to(device),
        "target_mask": target_mask.to(device),
        "mission_tokens": mission_tokens.to(device),
        "mission_lengths": mission_lengths.to(device),
    }


def masked_step_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, step_mask: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over valid steps (every active step is a target here)."""
    per_step = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    mask = step_mask.float()
    return (per_step * mask).sum() / mask.sum()


def train_bc(
    model_factory,
    episodes: list[BCEpisode],
    vocab: Vocabulary,
    *,
    updates: int,
    batch_episodes: int,
    seed: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    log_every: int = 25,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Deterministic AdamW training; returns the trained model and loss log.

    The model is constructed *after* seeding (via ``model_factory``) so that
    the initial weights are part of the seeded computation, so the same
    seed-then-construct order the study's ``train_base`` uses.
    """
    ensure_determinism()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sampler = np.random.default_rng(seed)
    model: nn.Module = model_factory()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    log: list[dict[str, float]] = []
    for update in range(updates):
        indices = sampler.integers(0, len(episodes), size=batch_episodes)
        batch = collate_episodes([episodes[i] for i in indices], vocab, device)
        logits, _ = model(
            batch["image"],
            batch["direction"],
            batch["prev_action"],
            batch["mission_tokens"],
            batch["mission_lengths"],
            batch["step_mask"],
        )
        loss = masked_step_cross_entropy(logits, batch["targets"], batch["target_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if update % log_every == 0 or update == updates - 1:
            log.append({"update": float(update), "loss": float(loss.detach().cpu())})
    model.eval()
    return model, log


def model_digest(model: nn.Module) -> str:
    """Order-stable digest of all parameters (bit-exact determinism checks)."""
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model: nn.Module, meta: dict[str, object]) -> None:
    """Persist trained weights so later steps (media, demos) can replay the
    policy without retraining. The parameter digest travels with the file and
    is re-verified on load."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["digest"] = model_digest(model)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": payload}, path)
    path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_checkpoint(path, model_factory) -> tuple[nn.Module, dict[str, object]]:
    """Rebuild a model from a checkpoint and verify its parameter digest."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FoundationsError(
            f"checkpoint {path} is missing; rerun the lab that produces it"
        )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model: nn.Module = model_factory()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    meta = dict(bundle["meta"])
    if model_digest(model) != meta["digest"]:
        raise FoundationsError(f"checkpoint {path} failed its digest check")
    return model, meta


@torch.no_grad()
def closed_loop_success(
    model: nn.Module,
    vocab: Vocabulary,
    env_cfg: EnvironmentConfig,
    seeds: list[int],
    device: torch.device,
) -> dict[str, object]:
    """Greedy closed-loop rollouts; the model drives, the oracle is absent."""
    model.eval()
    successes = 0
    steps_taken: list[int] = []
    for seed in seeds:
        session = WorldSession(env_cfg)
        try:
            result = session.reset(seed)
            tokens = vocab.encode(result.mission)
            mission_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
            mission_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
            mission_feature = model.encode_mission(mission_tokens, mission_lengths)
            hidden: torch.Tensor | None = None
            prev = torch.tensor([START_ACTION_TOKEN], dtype=torch.long, device=device)
            while not session.done:
                image = torch.from_numpy(result.image.astype(np.int64)).to(device)
                direction = torch.tensor([result.direction], dtype=torch.long, device=device)
                logits, hidden = model.step(
                    image.unsqueeze(0), direction, prev, mission_feature, hidden
                )
                action = int(torch.argmax(logits, dim=-1).item())
                result = session.step(action)
                prev = torch.tensor([action], dtype=torch.long, device=device)
            successes += int(result.terminated and result.reward > 0.0)
            steps_taken.append(session.time)
        finally:
            session.close()
    return {
        "episodes": len(seeds),
        "successes": successes,
        "success_rate": successes / len(seeds) if seeds else 0.0,
        "mean_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
    }


@torch.no_grad()
def open_loop_accuracy(
    model: nn.Module,
    episodes: list[BCEpisode],
    vocab: Vocabulary,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Teacher-forced per-step agreement with the oracle along its own paths."""
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(episodes), batch_size):
        batch = collate_episodes(episodes[start : start + batch_size], vocab, device)
        logits, _ = model(
            batch["image"],
            batch["direction"],
            batch["prev_action"],
            batch["mission_tokens"],
            batch["mission_lengths"],
            batch["step_mask"],
        )
        predictions = logits.argmax(dim=-1)
        mask = batch["step_mask"]
        correct += int((predictions.eq(batch["targets"]) & mask).sum().item())
        total += int(mask.sum().item())
    return correct / total if total else 0.0
