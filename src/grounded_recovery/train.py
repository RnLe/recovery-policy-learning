"""Masked behavioral-cloning loss, deterministic sampling, training, checkpoints.

Training is single-process and fully seeded from named streams: no DataLoader
workers, no wall-clock or unseeded randomness. Each optimizer update samples
exactly the configured number of target windows with replacement from an
explicit NumPy generator, so the later budget-matched arms can reconcile
exposures mechanically. The metrics ledger is hash-chained like the collection
ledger. Checkpoints carry the contract hash, model-config hash, vocabulary,
action mapping, and all three RNG streams; loading rejects any identity
mismatch.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from grounded_recovery.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl,
    sha256_hex,
)
from grounded_recovery.config import ExperimentConfig, contract_hash
from grounded_recovery.data import (
    Batch,
    TargetWindow,
    Vocabulary,
    collate_windows,
    enumerate_windows,
    materialize_window,
    read_episode,
    vocabulary_from_dataset,
)
from grounded_recovery.integrity import GENESIS_HASH, recount_dataset
from grounded_recovery.model import RecoveryPolicy, model_config_hash
from grounded_recovery.seeds import derive_seed


class CheckpointMismatchError(RuntimeError):
    """A checkpoint's identity stamps disagree with the current contract."""


# Required for deterministic cuBLAS kernels under
# torch.use_deterministic_algorithms(True); must be set before the first
# cuBLAS handle is created. Harmless for CPU-only runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(cfg: ExperimentConfig) -> torch.device:
    """The contract's compute device for all model computation."""
    name = cfg.training.device
    if name == "cuda" and not torch.cuda.is_available():
        raise CheckpointMismatchError(
            "the contract requires device 'cuda' but no CUDA device is available"
        )
    return torch.device(name)


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    if device.type == "cpu":
        return batch
    return Batch(
        **{
            field.name: getattr(batch, field.name).to(device, non_blocking=True)
            for field in dataclasses.fields(Batch)
        }
    )


def masked_cross_entropy(
    logits: Tensor, targets: Tensor, target_mask: Tensor
) -> tuple[Tensor, int]:
    """Cross-entropy summed at target positions, divided by their count.

    With one-target windows the denominator equals the batch size. Padding and
    context-only prefix positions never contribute.
    """
    denominator = int(target_mask.sum().item())
    if denominator == 0:
        raise ValueError("batch contains no target positions")
    flat_logits = logits[target_mask]
    flat_targets = targets[target_mask]
    loss_sum = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, reduction="sum"
    )
    return loss_sum / denominator, denominator


def sample_window_indices(
    rng: np.random.Generator, n_windows: int, batch_size: int, *, with_replacement: bool
) -> np.ndarray:
    if n_windows < 1:
        raise ValueError("no windows to sample from")
    if with_replacement:
        return rng.integers(0, n_windows, size=batch_size)
    if batch_size > n_windows:
        raise ValueError(
            f"cannot draw {batch_size} windows without replacement from {n_windows}"
        )
    return rng.choice(n_windows, size=batch_size, replace=False)


def make_optimizer(
    model: torch.nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


class MetricsWriter:
    """Hash-chained per-update metrics ledger."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"metrics ledger {self.path} already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="ascii")
        self._prev_hash = GENESIS_HASH
        self.rows = 0

    def append(self, payload: dict[str, object]) -> None:
        payload = dict(payload)
        payload["row_index"] = self.rows
        payload["prev_row_hash"] = self._prev_hash
        row_hash = sha256_hex(
            self._prev_hash.encode("ascii") + canonical_json_bytes(payload)
        )
        payload["row_hash"] = row_hash
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()
        self._prev_hash = row_hash
        self.rows += 1

    @property
    def final_hash(self) -> str:
        return self._prev_hash

    def close(self) -> None:
        self._handle.close()
        os_fsync_path(self.path)


def os_fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class CheckpointMeta:
    contract_hash: str
    model_config_hash: str
    bundle_id: str
    arm: str
    round_index: int
    update_index: int
    vocabulary: tuple[str, ...]
    action_names: tuple[str, ...]
    action_ids: tuple[int, ...]
    dataset_schema_version: str
    metrics_ledger_hash: str
    parameter_count: int


def _rng_states(sampler_rng: np.random.Generator) -> dict[str, object]:
    states: dict[str, object] = {
        "python": random.getstate(),
        "numpy_sampler": sampler_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def save_checkpoint(
    path: Path,
    model: RecoveryPolicy,
    optimizer: torch.optim.Optimizer,
    meta: CheckpointMeta,
    sampler_rng: np.random.Generator,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint {path} already exists")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "meta": asdict(meta),
        "rng": _rng_states(sampler_rng),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    sidecar = {key: value for key, value in asdict(meta).items()}
    sidecar["vocabulary_size"] = len(meta.vocabulary)
    sidecar.pop("vocabulary")
    atomic_write_json(path.with_suffix(".json"), sidecar)


def load_checkpoint(
    path: Path,
    *,
    expected_contract_hash: str,
    expected_model_config_hash: str,
    expected_action_ids: tuple[int, ...],
    expected_vocab: tuple[str, ...],
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta = CheckpointMeta(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in payload["meta"].items()
    })
    if meta.contract_hash != expected_contract_hash:
        raise CheckpointMismatchError(
            f"checkpoint contract {meta.contract_hash[:12]}... does not match "
            f"expected {expected_contract_hash[:12]}..."
        )
    if meta.model_config_hash != expected_model_config_hash:
        raise CheckpointMismatchError("checkpoint model-config hash mismatch")
    if tuple(meta.action_ids) != tuple(expected_action_ids):
        raise CheckpointMismatchError(
            f"checkpoint action set {meta.action_ids} does not match "
            f"{expected_action_ids}"
        )
    if tuple(meta.vocabulary) != tuple(expected_vocab):
        raise CheckpointMismatchError("checkpoint vocabulary mismatch")
    payload["meta"] = meta
    return payload


def restore_rng_states(states: dict[str, object]) -> np.random.Generator:
    random.setstate(states["python"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    generator = np.random.default_rng()
    generator.bit_generator.state = states["numpy_sampler"]
    return generator


@dataclass(frozen=True)
class TrainBaseResult:
    checkpoint_path: str
    updates: int
    final_loss: float
    metrics_path: str
    metrics_final_hash: str
    window_count: int


def load_all_windows(
    cfg: ExperimentConfig, dataset_dir: Path, vocab: Vocabulary
) -> list[TargetWindow]:
    """Materialize every target window of a dataset in deterministic order."""
    dataset_dir = Path(dataset_dir)
    windows: list[TargetWindow] = []
    for row in read_jsonl(dataset_dir / "episode_index.jsonl"):
        arrays, sidecar = read_episode(dataset_dir / "episodes", row["episode_id"])
        for spec in enumerate_windows(
            row["episode_id"],
            arrays,
            cfg.training.max_context_prefix,
            cfg.training.max_sequence_length,
        ):
            windows.append(
                materialize_window(
                    arrays, sidecar, spec, vocab, len(cfg.environment.action_ids)
                )
            )
    return windows


def train_base(
    cfg: ExperimentConfig, bundle_id: str, dataset_dir: Path, out_dir: Path
) -> TrainBaseResult:
    """Train the shared base policy on `D0` for the frozen update count."""
    torch.use_deterministic_algorithms(True)
    device = resolve_device(cfg)
    root = cfg.seeds.root_seed
    torch.manual_seed(derive_seed(root, bundle_id, "init"))
    random.seed(derive_seed(root, bundle_id, "optimizer"))
    sampler_rng = np.random.default_rng(derive_seed(root, bundle_id, "sampler.base"))

    recount = recount_dataset(dataset_dir)
    if cfg.data.n0 is not None and recount["targets"] != cfg.data.n0:
        raise CheckpointMismatchError(
            f"dataset holds {recount['targets']} targets, contract requires {cfg.data.n0}"
        )

    vocab = vocabulary_from_dataset(dataset_dir)
    windows = load_all_windows(cfg, dataset_dir, vocab)
    if not windows:
        raise ValueError("no target windows in dataset")

    num_actions = len(cfg.environment.action_ids)
    model = RecoveryPolicy(cfg.model, vocab.size, num_actions).to(device)
    optimizer = make_optimizer(
        model, cfg.training.learning_rate, cfg.training.weight_decay
    )
    channel_limits = (cfg.model.num_objects, cfg.model.num_colors, cfg.model.num_states)

    out_dir = Path(out_dir)
    metrics = MetricsWriter(out_dir / "training" / "base" / "metrics.jsonl")
    final_loss = float("nan")
    cumulative_exposures = 0
    try:
        model.train()
        for update in range(cfg.training.base_updates):
            indices = sample_window_indices(
                sampler_rng,
                len(windows),
                cfg.training.base_targets_per_update,
                with_replacement=cfg.training.sampling_with_replacement,
            )
            batch: Batch = batch_to_device(
                collate_windows([windows[int(i)] for i in indices], channel_limits),
                device,
            )
            logits, _ = model(
                batch.image,
                batch.direction,
                batch.prev_executed_action,
                batch.mission_tokens,
                batch.mission_lengths,
                batch.step_mask,
            )
            loss, denominator = masked_cross_entropy(
                logits, batch.targets, batch.target_mask
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.gradient_clip_norm
            )
            optimizer.step()
            cumulative_exposures += denominator
            final_loss = float(loss.item())
            metrics.append(
                {
                    "update": update,
                    "loss": final_loss,
                    "loss_denominator": denominator,
                    "gradient_norm": float(gradient_norm.item()),
                    "sampled_window_ids_checksum": sha256_hex(
                        canonical_json_bytes(
                            [windows[int(i)].spec.window_id for i in indices]
                        )
                    ),
                    "base_targets_drawn": denominator,
                    "new_targets_drawn": 0,
                    "cumulative_target_exposures": cumulative_exposures,
                    "context_steps_processed": int(batch.step_mask.sum().item()),
                    "wall_time": round(time.time(), 3),
                }
            )
    finally:
        metrics.close()

    meta = CheckpointMeta(
        contract_hash=contract_hash(cfg),
        model_config_hash=model_config_hash(cfg.model, vocab.size, num_actions),
        bundle_id=bundle_id,
        arm="base",
        round_index=0,
        update_index=cfg.training.base_updates,
        vocabulary=vocab.tokens,
        action_names=cfg.environment.action_names,
        action_ids=cfg.environment.action_ids,
        dataset_schema_version=cfg.data.dataset_schema_version,
        metrics_ledger_hash=metrics.final_hash,
        parameter_count=model.parameter_count(),
    )
    checkpoint_path = out_dir / "checkpoints" / "base_final.pt"
    save_checkpoint(checkpoint_path, model, optimizer, meta, sampler_rng)
    return TrainBaseResult(
        checkpoint_path=str(checkpoint_path),
        updates=cfg.training.base_updates,
        final_loss=final_loss,
        metrics_path=str(metrics.path),
        metrics_final_hash=metrics.final_hash,
        window_count=len(windows),
    )


# --- Arm cloning and round training ------------------------------------------

def model_state_digest(state: dict[str, torch.Tensor]) -> str:
    """Content digest of a state dict (file bytes are not stable identities)."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8") + b"\x1f")
        digest.update(str(tensor.dtype).encode("ascii") + b"\x1f")
        digest.update(repr(tuple(tensor.shape)).encode("ascii") + b"\x1f")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\x1f")
    return digest.hexdigest()


def clone_arm_from_checkpoint(
    cfg: ExperimentConfig, checkpoint_path: Path, vocab: Vocabulary
) -> tuple[RecoveryPolicy, torch.optim.Optimizer, CheckpointMeta]:
    """Instantiate one augmented arm from the shared base checkpoint.

    The model parameters are exact copies; the optimizer state follows the
    frozen ``optimizer_state_policy`` (continue from base, or reset).
    """
    from grounded_recovery.model import RecoveryPolicy, model_config_hash

    num_actions = len(cfg.environment.action_ids)
    payload = load_checkpoint(
        Path(checkpoint_path),
        expected_contract_hash=contract_hash(cfg),
        expected_model_config_hash=model_config_hash(cfg.model, vocab.size, num_actions),
        expected_action_ids=cfg.environment.action_ids,
        expected_vocab=vocab.tokens,
    )
    model = RecoveryPolicy(cfg.model, vocab.size, num_actions)
    model.load_state_dict(payload["model_state"])
    model.to(resolve_device(cfg))
    optimizer = make_optimizer(
        model, cfg.training.learning_rate, cfg.training.weight_decay
    )
    if cfg.training.optimizer_state_policy == "continue":
        optimizer.load_state_dict(payload["optimizer_state"])
    elif cfg.training.optimizer_state_policy != "reset":
        raise CheckpointMismatchError(
            f"unknown optimizer_state_policy {cfg.training.optimizer_state_policy!r}"
        )
    return model, optimizer, payload["meta"]


def assert_clone_equality(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    optimizer_a: torch.optim.Optimizer,
    optimizer_b: torch.optim.Optimizer,
) -> None:
    """Both augmented arms must start from bit-identical model and optimizer state."""
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    if sorted(state_a) != sorted(state_b):
        raise CheckpointMismatchError("arm models have different parameter sets")
    for key in state_a:
        if not torch.equal(state_a[key], state_b[key]):
            raise CheckpointMismatchError(f"arm model parameter {key} differs before training")
    opt_a, opt_b = optimizer_a.state_dict()["state"], optimizer_b.state_dict()["state"]
    if sorted(opt_a) != sorted(opt_b):
        raise CheckpointMismatchError("arm optimizers have different state sets")
    for key in opt_a:
        for field in opt_a[key]:
            value_a, value_b = opt_a[key][field], opt_b[key][field]
            if isinstance(value_a, torch.Tensor):
                if not torch.equal(value_a, value_b):
                    raise CheckpointMismatchError(
                        f"arm optimizer state {key}/{field} differs before training"
                    )
            elif value_a != value_b:
                raise CheckpointMismatchError(
                    f"arm optimizer state {key}/{field} differs before training"
                )


def train_arm_round(
    cfg: ExperimentConfig,
    bundle_id: str,
    arm: str,
    round_index: int,
    model: RecoveryPolicy,
    optimizer: torch.optim.Optimizer,
    base_windows: list[TargetWindow],
    new_windows: list[TargetWindow],
    sampler_rng: np.random.Generator,
    exposure_writer: MetricsWriter,
    cumulative: dict[str, int],
) -> None:
    """One round of equal-update training for one augmented arm.

    Every update draws exactly ``base_targets_per_update`` base windows and
    ``new_targets_per_update`` added windows (with replacement per the
    contract); the exposure ledger records the accounting per update.
    """
    if cfg.training.updates_per_round is None or cfg.training.new_targets_per_update is None:
        raise CheckpointMismatchError(
            "updates_per_round / new_targets_per_update are unresolved (PILOT_TO_FREEZE)"
        )
    if not new_windows:
        raise CheckpointMismatchError(
            f"{arm} round {round_index}: no added target windows to train on"
        )
    channel_limits = (cfg.model.num_objects, cfg.model.num_colors, cfg.model.num_states)
    device = resolve_device(cfg)
    model.train()
    for update in range(cfg.training.updates_per_round):
        checkpoint_before = model_state_digest(model.state_dict())
        base_indices = sample_window_indices(
            sampler_rng,
            len(base_windows),
            cfg.training.base_targets_per_update,
            with_replacement=cfg.training.sampling_with_replacement,
        )
        new_indices = sample_window_indices(
            sampler_rng,
            len(new_windows),
            cfg.training.new_targets_per_update,
            with_replacement=cfg.training.sampling_with_replacement,
        )
        drawn = [base_windows[int(i)] for i in base_indices] + [
            new_windows[int(i)] for i in new_indices
        ]
        batch = batch_to_device(collate_windows(drawn, channel_limits), device)
        logits, _ = model(
            batch.image,
            batch.direction,
            batch.prev_executed_action,
            batch.mission_tokens,
            batch.mission_lengths,
            batch.step_mask,
        )
        loss, denominator = masked_cross_entropy(logits, batch.targets, batch.target_mask)
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.gradient_clip_norm
        )
        optimizer.step()
        cumulative["base"] += len(base_indices)
        cumulative["new"] += len(new_indices)
        cumulative["updates"] += 1
        exposure_writer.append(
            {
                "bundle": bundle_id,
                "arm": arm,
                "round": round_index,
                "update": update,
                "checkpoint_before_hash": checkpoint_before,
                "base_unique_available": len(base_windows),
                "new_unique_available": len(new_windows),
                "base_targets_drawn": len(base_indices),
                "new_targets_drawn": len(new_indices),
                "cumulative_base_exposures": cumulative["base"],
                "cumulative_new_exposures": cumulative["new"],
                "context_steps_processed": int(batch.step_mask.sum().item()),
                "target_ids_checksum": sha256_hex(
                    canonical_json_bytes([w.spec.window_id for w in drawn])
                ),
                "loss_sum": float(loss.item()) * denominator,
                "loss_denominator": denominator,
                "gradient_norm": float(gradient_norm.item()),
                "optimizer_step": cumulative["updates"],
            }
        )
