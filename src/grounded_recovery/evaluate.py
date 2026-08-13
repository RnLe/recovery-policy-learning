"""Closed-loop policy evaluation on frozen scenario panels.

Evaluation is policy-only: the scripted oracle is never constructed, so no
oracle recommendation can reach the learned policy at test time. Rollouts are
greedy under ``model.eval()`` and ``torch.no_grad()``. Scoring is
intention-to-treat: every assigned scenario stays in the denominator, whether
or not its scheduled corruption was delivered and however early the episode
ended.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from grounded_recovery.config import ExperimentConfig, contract_hash
from grounded_recovery.data import ManifestError, Vocabulary, start_action_token
from grounded_recovery.perturbations import ActionDerangement
from grounded_recovery.schemas import ManifestEntry, canonical_scenario_hash
from grounded_recovery.seeds import derive_seed
from grounded_recovery.world import WorldSession

SLICE_CLEAN = "clean"
SLICE_MATCHED = "matched"
SLICE_UNSEEN = "unseen"


@dataclass(frozen=True)
class EvaluationRow:
    """One intention-to-treat episode outcome."""

    bundle_id: str
    arm: str
    slice_name: str
    scenario_ordinal: int
    environment_seed: int
    scenario_hash: str
    scheduled_time: int | None
    operator_name: str | None
    delivered: bool
    success: bool
    truncated: bool
    steps: int
    nominal_oracle_path_length: int
    contract_hash: str


def evaluation_row_to_json(row: EvaluationRow) -> dict[str, object]:
    return asdict(row)


def scheduled_evaluation_time(
    cfg: ExperimentConfig, slice_name: str, ordinal: int, time_set: tuple[int, ...]
) -> int:
    """Deterministic per-scenario corruption time for one evaluation slice."""
    raw = derive_seed(
        cfg.seeds.root_seed, "global", f"evaluation.schedule.{slice_name}.{ordinal}"
    )
    return time_set[raw % len(time_set)]


def rollout_policy(
    cfg: ExperimentConfig,
    session: WorldSession,
    policy,
    vocab: Vocabulary,
    entry: ManifestEntry,
    *,
    scheduled_time: int | None,
    operator: ActionDerangement | None,
) -> tuple[bool, bool, int, bool]:
    """One greedy closed-loop episode; returns (success, truncated, steps, delivered).

    The policy sees only its own observation stream; a scheduled corruption
    replaces its proposal with ``operator(proposal)`` for that one transition.
    """
    import torch

    if (scheduled_time is None) != (operator is None):
        raise ManifestError("scheduled_time and operator must be provided together")
    observation = session.reset(entry.environment_seed)
    live_hash = canonical_scenario_hash(session.scenario_state())
    if live_hash != entry.canonical_scenario_hash:
        raise ManifestError(
            f"evaluation of seed {entry.environment_seed} does not reproduce the "
            "manifested world"
        )
    num_actions = len(cfg.environment.action_ids)
    policy.eval()
    device = next(policy.parameters()).device
    delivered = False
    with torch.no_grad():
        encoded = vocab.encode(observation.mission)
        if not encoded:
            raise ManifestError("empty mission token sequence in evaluation")
        mission_feature = policy.encode_mission(
            torch.tensor([encoded], dtype=torch.long, device=device),
            torch.tensor([len(encoded)], dtype=torch.long, device=device),
        )
        hidden = None
        last_token = start_action_token(num_actions)
        while not session.done:
            t = session.time
            image = torch.from_numpy(
                observation.image.astype("int64")
            ).unsqueeze(0).to(device)
            direction = torch.tensor([observation.direction], dtype=torch.long,
                                     device=device)
            previous = torch.tensor([last_token], dtype=torch.long, device=device)
            logits, hidden = policy.step(
                image, direction, previous, mission_feature, hidden
            )
            proposal = int(logits.argmax(dim=-1).item())
            if scheduled_time is not None and t == scheduled_time:
                executed = operator.apply(proposal)
                delivered = True
            else:
                executed = proposal
            observation = session.step(executed)
            last_token = executed
    return observation.terminated, observation.truncated, session.time, delivered


def evaluate_policy_on_entries(
    cfg: ExperimentConfig,
    session: WorldSession,
    policy,
    vocab: Vocabulary,
    entries: list[ManifestEntry],
    *,
    bundle_id: str,
    arm: str,
    slice_name: str,
    operator: ActionDerangement | None,
    time_set: tuple[int, ...] | None,
) -> list[EvaluationRow]:
    """Evaluate one policy on one ordered scenario panel for one slice.

    Every entry produces exactly one row (intention-to-treat); the scenario
    order and per-scenario schedules are functions of the contract, so every
    arm and bundle sees the identical crossed panel.
    """
    if (operator is None) != (time_set is None):
        raise ManifestError("operator and time_set must be provided together")
    cfg_hash = contract_hash(cfg)
    rows: list[EvaluationRow] = []
    for entry in entries:
        scheduled: int | None = None
        if time_set is not None:
            scheduled = scheduled_evaluation_time(cfg, slice_name, entry.ordinal, time_set)
        success, truncated, steps, delivered = rollout_policy(
            cfg, session, policy, vocab, entry,
            scheduled_time=scheduled,
            operator=operator,
        )
        rows.append(
            EvaluationRow(
                bundle_id=bundle_id,
                arm=arm,
                slice_name=slice_name,
                scenario_ordinal=entry.ordinal,
                environment_seed=entry.environment_seed,
                scenario_hash=entry.canonical_scenario_hash,
                scheduled_time=scheduled,
                operator_name=operator.name if operator is not None else None,
                delivered=delivered,
                success=success,
                truncated=truncated,
                steps=steps,
                nominal_oracle_path_length=entry.nominal_oracle_path_length,
                contract_hash=cfg_hash,
            )
        )
    return rows


def load_arm_policy(cfg: ExperimentConfig, checkpoint_path: Path, vocab: Vocabulary):
    """Instantiate a policy from a checkpoint with full identity validation."""
    from grounded_recovery.model import RecoveryPolicy, model_config_hash
    from grounded_recovery.train import load_checkpoint

    num_actions = len(cfg.environment.action_ids)
    payload = load_checkpoint(
        Path(checkpoint_path),
        expected_contract_hash=contract_hash(cfg),
        expected_model_config_hash=model_config_hash(cfg.model, vocab.size, num_actions),
        expected_action_ids=cfg.environment.action_ids,
        expected_vocab=vocab.tokens,
    )
    from grounded_recovery.train import resolve_device

    policy = RecoveryPolicy(cfg.model, vocab.size, num_actions)
    policy.load_state_dict(payload["model_state"])
    policy.to(resolve_device(cfg))
    policy.eval()
    return policy
