"""Oracle/operator preflight: gate G1's executable evidence.

The preflight runs the scripted oracle, never a learned policy, over the
isolated ``operator_preflight`` split. Each episode delivers exactly one
forced corruption ``g(recommendation)`` at its manifest-scheduled time; every
other action follows the oracle. The gate requires, per operator family, at
least the configured episode count and a delivered-corruption recovery rate
at or above the frozen threshold. Failures are recorded with full traces and
are never dropped.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from grounded_recovery.artifacts import atomic_write_json, atomic_write_jsonl, hash_json
from grounded_recovery.config import (
    SPLIT_NAMES,
    ExperimentConfig,
    contract_hash,
    scenario_identity_hash,
)
from grounded_recovery.data import (
    ManifestError,
    audit_disjointness,
    load_split_manifest,
    verify_manifest_contract,
)
from grounded_recovery.oracle import OracleSupportError, run_synchronized_episode
from grounded_recovery.perturbations import ActionDerangement, operator_from_config
from grounded_recovery.schemas import ManifestEntry, PreflightEpisodeRow, preflight_row_to_json
from grounded_recovery.world import WorldSession


@dataclass(frozen=True)
class FamilyPreflightSummary:
    family: str
    operator_name: str
    episodes: int
    required_episodes: int
    episode_scale_ok: bool
    delivered: int
    recovered_successes: int
    recovery_rate_delivered: float
    undelivered: int
    truncations: int
    support_violations: int
    gate: float
    passed: bool


def _run_family(
    cfg: ExperimentConfig,
    session: WorldSession,
    family: str,
    operator: ActionDerangement,
    entries: list[ManifestEntry],
    manifest_hash: str,
) -> tuple[FamilyPreflightSummary, list[PreflightEpisodeRow], list[dict[str, object]]]:
    rows: list[PreflightEpisodeRow] = []
    failure_traces: list[dict[str, object]] = []
    cfg_hash = contract_hash(cfg)
    support_violations = 0

    for entry in entries:
        scheduled_time = entry.scheduled_intervention_times[0]
        forced_holder: dict[str, int] = {}

        def choose(
            t: int,
            recommended: int,
            _scheduled: int = scheduled_time,
            _holder: dict[str, int] = forced_holder,
        ) -> int:
            if t == _scheduled:
                forced = operator.apply(recommended)
                _holder["recommended"] = recommended
                _holder["forced"] = forced
                return forced
            return recommended

        try:
            trace = run_synchronized_episode(session, entry.environment_seed, choose)
        except OracleSupportError as error:
            support_violations += 1
            rows.append(
                PreflightEpisodeRow(
                    family=family,
                    operator_name=operator.name,
                    ordinal=entry.ordinal,
                    environment_seed=entry.environment_seed,
                    scenario_hash=entry.canonical_scenario_hash,
                    scheduled_time=scheduled_time,
                    delivered="forced" in forced_holder,
                    recommended_at_scheduled_time=forced_holder.get("recommended"),
                    forced_action=forced_holder.get("forced"),
                    success=False,
                    steps=-1,
                    nominal_oracle_path_length=entry.nominal_oracle_path_length,
                    oracle_calls=-1,
                    truncated=False,
                    termination_reason=f"oracle_support_error: {error}",
                    contract_hash=cfg_hash,
                    manifest_hash=manifest_hash,
                )
            )
            failure_traces.append(
                {
                    "family": family,
                    "seed": entry.environment_seed,
                    "scheduled_time": scheduled_time,
                    "error": str(error),
                    "transitions": [],
                }
            )
            continue

        if trace.scenario_hash != entry.canonical_scenario_hash:
            raise ManifestError(
                f"preflight replay of seed {entry.environment_seed} produced scenario hash "
                f"{trace.scenario_hash}, manifest records {entry.canonical_scenario_hash}; "
                "the environment no longer reproduces the manifested world"
            )

        delivered = any(t.t == scheduled_time for t in trace.transitions)
        if delivered:
            forced_transition = trace.transitions[scheduled_time]
            if forced_transition.executed != operator.apply(forced_transition.recommended):
                raise ManifestError(
                    f"forced action mismatch at t={scheduled_time} on seed "
                    f"{entry.environment_seed}"
                )
        termination_reason = "terminated" if trace.success else (
            "truncated" if trace.truncated else "unknown"
        )
        row = PreflightEpisodeRow(
            family=family,
            operator_name=operator.name,
            ordinal=entry.ordinal,
            environment_seed=entry.environment_seed,
            scenario_hash=entry.canonical_scenario_hash,
            scheduled_time=scheduled_time,
            delivered=delivered,
            recommended_at_scheduled_time=forced_holder.get("recommended"),
            forced_action=forced_holder.get("forced"),
            success=trace.success,
            steps=len(trace.transitions),
            nominal_oracle_path_length=entry.nominal_oracle_path_length,
            oracle_calls=trace.oracle_calls,
            truncated=trace.truncated,
            termination_reason=termination_reason,
            contract_hash=cfg_hash,
            manifest_hash=manifest_hash,
        )
        rows.append(row)
        if not trace.success:
            failure_traces.append(
                {
                    "family": family,
                    "seed": entry.environment_seed,
                    "scheduled_time": scheduled_time,
                    "error": None,
                    "transitions": [asdict(t) for t in trace.transitions],
                }
            )

    delivered_rows = [row for row in rows if row.delivered]
    recovered = sum(1 for row in delivered_rows if row.success)
    rate = recovered / len(delivered_rows) if delivered_rows else 0.0
    episodes = len(rows)
    required = cfg.perturbation.preflight_episodes_per_family
    scale_ok = episodes >= 500 and episodes >= required
    summary = FamilyPreflightSummary(
        family=family,
        operator_name=operator.name,
        episodes=episodes,
        required_episodes=max(required, 500),
        episode_scale_ok=scale_ok,
        delivered=len(delivered_rows),
        recovered_successes=recovered,
        recovery_rate_delivered=rate,
        undelivered=episodes - len(delivered_rows),
        truncations=sum(1 for row in rows if row.truncated),
        support_violations=support_violations,
        gate=cfg.environment.oracle_recovery_gate,
        passed=(
            scale_ok
            and support_violations == 0
            and rate >= cfg.environment.oracle_recovery_gate
        ),
    )
    return summary, rows, failure_traces


def run_preflight(
    cfg: ExperimentConfig, manifest_root: Path, out_dir: Path
) -> dict[str, object]:
    """Run both operator families over the preflight split and write the report."""
    manifest_root = Path(manifest_root)
    out_dir = Path(out_dir)
    started = time.time()

    verify_manifest_contract(manifest_root, "operator_preflight", cfg)
    entries, manifest_hash = load_split_manifest(manifest_root, "operator_preflight")

    # Re-audit split disjointness from the stored manifests, including the
    # preflight split against every other split.
    all_manifests = {
        split: load_split_manifest(manifest_root, split)[0] for split in SPLIT_NAMES
    }
    disjointness = audit_disjointness(all_manifests)

    operators = {
        "collection": operator_from_config(
            cfg.perturbation.collection_operator, cfg.environment.action_ids
        ),
        "unseen": operator_from_config(
            cfg.perturbation.unseen_operator, cfg.environment.action_ids
        ),
    }

    session = WorldSession(cfg.environment)
    summaries: dict[str, FamilyPreflightSummary] = {}
    all_rows: list[PreflightEpisodeRow] = []
    all_failures: list[dict[str, object]] = []
    try:
        for family, operator in operators.items():
            summary, rows, failures = _run_family(
                cfg, session, family, operator, entries, manifest_hash
            )
            summaries[family] = summary
            all_rows.extend(rows)
            all_failures.extend(failures)
    finally:
        session.close()

    observed_recommendations = sorted(
        {
            row.recommended_at_scheduled_time
            for row in all_rows
            if row.recommended_at_scheduled_time is not None
        }
    )
    action_support_hash = hash_json(
        {
            "frozen_action_ids": list(cfg.environment.action_ids),
            "observed_scheduled_recommendations": observed_recommendations,
        }
    )
    passed = all(summary.passed for summary in summaries.values())
    report = {
        "passed": passed,
        "families": {family: asdict(summary) for family, summary in summaries.items()},
        "schedule_rule": (
            "uniform in [preflight_time_min, min(preflight_time_max, "
            "nominal_oracle_path_length - 1)]; conditioned on the scenario's fixed "
            "nominal path length so every preflight corruption is deliverable"
        ),
        "action_support_hash": action_support_hash,
        "contract_hash": contract_hash(cfg),
        "manifest_hash": manifest_hash,
        "disjointness": disjointness,
        "episodes_written": len(all_rows),
        "failure_count": len(all_failures),
        "wall_time_seconds": round(time.time() - started, 3),
    }

    atomic_write_jsonl(
        out_dir / "preflight_episodes.jsonl", (preflight_row_to_json(row) for row in all_rows)
    )
    atomic_write_json(out_dir / "preflight_report.json", report)
    atomic_write_jsonl(out_dir / "failure_traces.jsonl", iter(all_failures))
    return report


# --- Three-arm vertical slice -------------------------------------------------

ARM_EXTRA = "extra_demonstrations"
ARM_RECOVERY = "recovery_aggregation"


def arm_partition(entries: list[ManifestEntry], arm: str) -> list[ManifestEntry]:
    """Deterministic disjoint split of the collection manifest between arms.

    Even ordinals feed the extra-demonstration arm, odd ordinals the recovery
    arm, so the two arms never collect from the same scenario.
    """
    if arm == ARM_EXTRA:
        return [entry for entry in entries if entry.ordinal % 2 == 0]
    if arm == ARM_RECOVERY:
        return [entry for entry in entries if entry.ordinal % 2 == 1]
    raise ManifestError(f"unknown augmented arm {arm!r}")


def _scheduled_collection_time(
    cfg: ExperimentConfig, bundle_id: str, arm: str, round_index: int, ordinal: int
) -> int:
    """Deterministic corruption time for one recovery-collection episode."""
    times = cfg.perturbation.collection_time_set
    if times is None:
        raise ManifestError("perturbation.collection_time_set is unresolved (PILOT_TO_FREEZE)")
    from grounded_recovery.seeds import derive_seed

    raw = derive_seed(
        cfg.seeds.root_seed,
        bundle_id,
        f"collection.recovery.round{round_index}.schedule.{ordinal}",
    )
    return times[raw % len(times)]


class GreedyPolicyDriver:
    """Closed-loop greedy policy rollout with one scheduled proposal corruption.

    On the scheduled step the executed action is ``operator(proposal)``; on
    every other step the policy's greedy proposal is executed unchanged. The
    driver threads the *executed* action back into its own history input, and
    the surrounding synchronized loop threads it into the oracle, so both the
    policy state and the bot plan follow what actually happened.
    """

    def __init__(self, policy, vocab, num_actions: int, session, scheduled_time: int,
                 operator) -> None:
        import torch

        from grounded_recovery.data import start_action_token

        self._torch = torch
        self.policy = policy
        self.vocab = vocab
        self.session = session
        self.scheduled_time = scheduled_time
        self.operator = operator
        self.device = next(policy.parameters()).device
        self.hidden = None
        self.mission_feature = None
        self.last_token = start_action_token(num_actions)
        self.proposals: dict[int, int] = {}

    def __call__(self, t: int, recommended: int) -> int:
        torch = self._torch
        observation = self.session.last_observation
        with torch.no_grad():
            if self.mission_feature is None:
                encoded = self.vocab.encode(observation.mission)
                if not encoded:
                    raise ManifestError("empty mission token sequence in rollout")
                tokens = torch.tensor([encoded], dtype=torch.long, device=self.device)
                lengths = torch.tensor([len(encoded)], dtype=torch.long,
                                       device=self.device)
                self.mission_feature = self.policy.encode_mission(tokens, lengths)
            image = torch.from_numpy(
                observation.image.astype("int64")
            ).unsqueeze(0).to(self.device)
            direction = torch.tensor([observation.direction], dtype=torch.long,
                                     device=self.device)
            previous = torch.tensor([self.last_token], dtype=torch.long,
                                    device=self.device)
            logits, self.hidden = self.policy.step(
                image, direction, previous, self.mission_feature, self.hidden
            )
        proposal = int(logits.argmax(dim=-1).item())
        self.proposals[t] = proposal
        executed = (
            self.operator.apply(proposal) if t == self.scheduled_time else proposal
        )
        self.last_token = executed
        return executed


@dataclass(frozen=True)
class RoundCollectionSummary:
    arm: str
    round_index: int
    budget: int
    revealed_targets: int
    episodes: int
    delivered: int
    undelivered: int
    simulator_steps: int
    oracle_calls: int
    discarded_recommendations: int
    entries_consumed: int
    dataset_dir: str


def _round_dir(cfg: ExperimentConfig, data_root: Path, bundle_id: str, arm: str,
               round_index: int) -> Path:
    return (
        Path(data_root) / contract_hash(cfg)[:12] / bundle_id / arm
        / f"round_{round_index:02d}"
    )


def collect_extra_demo_round(
    cfg: ExperimentConfig,
    bundle_id: str,
    round_index: int,
    budget: int,
    entries: list[ManifestEntry],
    start_offset: int,
    manifest_hash: str,
    data_root: Path,
) -> RoundCollectionSummary:
    """Fresh nominal oracle demonstrations until exactly ``budget`` targets.

    Identical boundary semantics to base collection: stop stepping right after
    the transition that consumes the last budgeted target.
    """
    import numpy as np

    from grounded_recovery.data import write_episode
    from grounded_recovery.integrity import LedgerWriter, recount_dataset
    from grounded_recovery.schemas import episode_from_trace
    from grounded_recovery.world import WorldSession

    dataset_dir = _round_dir(cfg, data_root, bundle_id, ARM_EXTRA, round_index)
    if dataset_dir.exists():
        raise ManifestError(f"round dataset already exists at {dataset_dir}")
    cfg_hash = contract_hash(cfg)
    session = WorldSession(cfg.environment)
    ledger = LedgerWriter(dataset_dir / "collection_ledger.jsonl")
    index_rows: list[dict[str, object]] = []
    remaining = budget
    steps_total = 0
    oracle_calls_total = 0
    consumed = 0
    try:
        for entry in entries[start_offset:]:
            if remaining == 0:
                break
            consumed += 1
            stop_budget = remaining

            def stop(completed: int, _budget: int = stop_budget) -> bool:
                return completed >= _budget

            trace = run_synchronized_episode(
                session, entry.environment_seed, lambda t, rec: rec, stop_after_step=stop
            )
            if trace.scenario_hash != entry.canonical_scenario_hash:
                raise ManifestError(
                    f"collection replay of seed {entry.environment_seed} does not "
                    "reproduce the manifested world"
                )
            length = len(trace.transitions)
            if trace.stopped_early:
                termination_reason = "budget_truncated"
            elif trace.success:
                termination_reason = "terminated"
            else:
                termination_reason = "truncated"
            episode_id = f"extra_r{round_index:02d}_{entry.ordinal:05d}"
            arrays, sidecar = episode_from_trace(
                trace,
                episode_id=episode_id,
                reveal_mask=np.ones(length, dtype=np.bool_),
                source_arm=ARM_EXTRA,
                round_index=round_index,
                termination_reason=termination_reason,
                intervention=None,
                contract_hash=cfg_hash,
                manifest_hash=manifest_hash,
            )
            write_episode(dataset_dir / "episodes", arrays, sidecar)
            ledger.append(
                episode_id=episode_id,
                episode_targets=arrays.revealed_targets,
                episode_steps=length,
                oracle_calls=trace.oracle_calls,
                budget_truncated=trace.stopped_early,
                episode_checksum=sidecar.content_checksum,
            )
            index_rows.append(
                {
                    "episode_id": episode_id,
                    "environment_seed": entry.environment_seed,
                    "canonical_scenario_hash": entry.canonical_scenario_hash,
                    "revealed_targets": arrays.revealed_targets,
                    "content_checksum": sidecar.content_checksum,
                }
            )
            remaining -= arrays.revealed_targets
            steps_total += length
            oracle_calls_total += trace.oracle_calls
        if remaining > 0:
            raise ManifestError(
                f"{ARM_EXTRA} round {round_index}: manifest partition exhausted with "
                f"{remaining} of {budget} targets uncollected"
            )
    finally:
        session.close()
        ledger.close()
    atomic_write_jsonl(dataset_dir / "episode_index.jsonl", iter(index_rows))
    ledger.finalize(contract_hash=cfg_hash, manifest_hash=manifest_hash)
    atomic_write_json(
        dataset_dir / "dataset_meta.json",
        {
            "bundle_id": bundle_id,
            "source_arm": ARM_EXTRA,
            "round_index": round_index,
            "budget": budget,
            "episodes": len(index_rows),
            "contract_hash": cfg_hash,
            "manifest_hash": manifest_hash,
            "dataset_schema_version": cfg.data.dataset_schema_version,
        },
    )
    recount = recount_dataset(dataset_dir)
    if recount["targets"] != budget:
        raise ManifestError(
            f"{ARM_EXTRA} round {round_index}: recount found {recount['targets']} targets, "
            f"exactly {budget} required"
        )
    return RoundCollectionSummary(
        arm=ARM_EXTRA,
        round_index=round_index,
        budget=budget,
        revealed_targets=budget,
        episodes=len(index_rows),
        delivered=0,
        undelivered=0,
        simulator_steps=steps_total,
        oracle_calls=oracle_calls_total,
        discarded_recommendations=steps_total - budget,
        entries_consumed=consumed,
        dataset_dir=str(dataset_dir),
    )


def collect_recovery_round(
    cfg: ExperimentConfig,
    bundle_id: str,
    round_index: int,
    budget: int,
    entries: list[ManifestEntry],
    start_offset: int,
    manifest_hash: str,
    data_root: Path,
    policy,
    vocab,
) -> RoundCollectionSummary:
    """Roll out the current recovery policy, corrupt one proposal per episode,
    and reveal at most ``H`` post-corruption oracle recommendations until the
    exact round budget is met.

    Episodes run to their natural end under policy control; recommendations
    outside the reveal window keep the bot synchronized but are discarded as
    supervision. Undelivered corruptions and zero-label episodes are stored,
    never dropped.
    """
    import dataclasses as dc

    from grounded_recovery.data import reveal_window_mask, write_episode
    from grounded_recovery.integrity import LedgerWriter, recount_dataset
    from grounded_recovery.perturbations import operator_from_config
    from grounded_recovery.schemas import episode_from_trace
    from grounded_recovery.world import WorldSession

    if cfg.data.h is None:
        raise ManifestError("data.h is unresolved (PILOT_TO_FREEZE)")
    dataset_dir = _round_dir(cfg, data_root, bundle_id, ARM_RECOVERY, round_index)
    if dataset_dir.exists():
        raise ManifestError(f"round dataset already exists at {dataset_dir}")
    operator = operator_from_config(
        cfg.perturbation.collection_operator, cfg.environment.action_ids
    )
    cfg_hash = contract_hash(cfg)
    num_actions = len(cfg.environment.action_ids)
    policy.eval()
    session = WorldSession(cfg.environment)
    ledger = LedgerWriter(dataset_dir / "collection_ledger.jsonl")
    index_rows: list[dict[str, object]] = []
    remaining = budget
    steps_total = 0
    oracle_calls_total = 0
    delivered_count = 0
    undelivered_count = 0
    discarded = 0
    consumed = 0
    try:
        for entry in entries[start_offset:]:
            if remaining == 0:
                break
            consumed += 1
            scheduled_time = _scheduled_collection_time(
                cfg, bundle_id, ARM_RECOVERY, round_index, entry.ordinal
            )
            driver = GreedyPolicyDriver(
                policy, vocab, num_actions, session, scheduled_time, operator
            )
            trace = run_synchronized_episode(session, entry.environment_seed, driver)
            if trace.scenario_hash != entry.canonical_scenario_hash:
                raise ManifestError(
                    f"recovery rollout of seed {entry.environment_seed} does not "
                    "reproduce the manifested world"
                )
            length = len(trace.transitions)
            transitions = tuple(
                dc.replace(transition, proposed=driver.proposals[transition.t])
                for transition in trace.transitions
            )
            trace = dc.replace(trace, transitions=transitions)
            delivered = length > scheduled_time
            reveal = reveal_window_mask(length, scheduled_time, cfg.data.h, remaining)
            revealed = int(reveal.sum())
            if delivered:
                delivered_count += 1
                forced = transitions[scheduled_time]
                if forced.executed != operator.apply(forced.proposed):
                    raise ManifestError(
                        f"forced proposal corruption mismatch at t={scheduled_time}"
                    )
            else:
                undelivered_count += 1
            intervention = {
                "scheduled_time": scheduled_time,
                "delivered": delivered,
                "operator": operator.name,
                "proposal_at_scheduled_time": (
                    transitions[scheduled_time].proposed if delivered else None
                ),
                "forced_action": (
                    transitions[scheduled_time].executed if delivered else None
                ),
                "revealed_targets": revealed,
            }
            episode_id = f"recovery_r{round_index:02d}_{entry.ordinal:05d}"
            arrays, sidecar = episode_from_trace(
                trace,
                episode_id=episode_id,
                reveal_mask=reveal,
                source_arm=ARM_RECOVERY,
                round_index=round_index,
                termination_reason="terminated" if trace.success else "truncated",
                intervention=intervention,
                contract_hash=cfg_hash,
                manifest_hash=manifest_hash,
            )
            write_episode(dataset_dir / "episodes", arrays, sidecar)
            ledger.append(
                episode_id=episode_id,
                episode_targets=revealed,
                episode_steps=length,
                oracle_calls=trace.oracle_calls,
                budget_truncated=False,
                episode_checksum=sidecar.content_checksum,
            )
            index_rows.append(
                {
                    "episode_id": episode_id,
                    "environment_seed": entry.environment_seed,
                    "canonical_scenario_hash": entry.canonical_scenario_hash,
                    "revealed_targets": revealed,
                    "content_checksum": sidecar.content_checksum,
                }
            )
            remaining -= revealed
            steps_total += length
            oracle_calls_total += trace.oracle_calls
            discarded += length - revealed
        if remaining > 0:
            raise ManifestError(
                f"{ARM_RECOVERY} round {round_index}: manifest partition exhausted with "
                f"{remaining} of {budget} targets unrevealed"
            )
    finally:
        session.close()
        ledger.close()
    atomic_write_jsonl(dataset_dir / "episode_index.jsonl", iter(index_rows))
    ledger.finalize(contract_hash=cfg_hash, manifest_hash=manifest_hash)
    atomic_write_json(
        dataset_dir / "dataset_meta.json",
        {
            "bundle_id": bundle_id,
            "source_arm": ARM_RECOVERY,
            "round_index": round_index,
            "budget": budget,
            "episodes": len(index_rows),
            "delivered": delivered_count,
            "undelivered": undelivered_count,
            "contract_hash": cfg_hash,
            "manifest_hash": manifest_hash,
            "dataset_schema_version": cfg.data.dataset_schema_version,
        },
    )
    recount = recount_dataset(dataset_dir)
    if recount["targets"] != budget:
        raise ManifestError(
            f"{ARM_RECOVERY} round {round_index}: recount found {recount['targets']} "
            f"targets, exactly {budget} required"
        )
    return RoundCollectionSummary(
        arm=ARM_RECOVERY,
        round_index=round_index,
        budget=budget,
        revealed_targets=budget,
        episodes=len(index_rows),
        delivered=delivered_count,
        undelivered=undelivered_count,
        simulator_steps=steps_total,
        oracle_calls=oracle_calls_total,
        discarded_recommendations=discarded,
        entries_consumed=consumed,
        dataset_dir=str(dataset_dir),
    )


# --- Pilot bundle orchestration ----------------------------------------------

REQUIRED_RESOLVED_FOR_BUNDLE: tuple[str, ...] = (
    "data.n0",
    "data.b",
    "data.k",
    "data.h",
    "perturbation.collection_time_set",
    "training.new_targets_per_update",
    "training.updates_per_round",
)


def run_pilot_bundle(
    cfg: ExperimentConfig,
    bundle_id: str,
    manifest_root: Path,
    data_root: Path,
) -> dict[str, object]:
    """Run one complete three-arm pipeline bundle under pilot settings.

    Order: shared base data -> immutable base checkpoint -> clone the two
    augmented arms (asserting bit equality) -> per round, collect each arm's
    exact budget and train both arms for exactly the same update count ->
    verify the unchanged base and write both final arm checkpoints.
    """
    import torch

    from grounded_recovery.config import unresolved_fields
    from grounded_recovery.data import (
        base_dataset_dir,
        collect_base,
        vocabulary_from_dataset,
    )
    from grounded_recovery.integrity import audit_round_fairness
    from grounded_recovery.train import (
        CheckpointMeta,
        MetricsWriter,
        assert_clone_equality,
        clone_arm_from_checkpoint,
        load_all_windows,
        model_state_digest,
        save_checkpoint,
        train_arm_round,
        train_base,
    )

    unresolved = set(unresolved_fields(cfg)) & set(REQUIRED_RESOLVED_FOR_BUNDLE)
    if unresolved:
        raise ManifestError(
            f"cannot run a bundle with unresolved fields: {sorted(unresolved)}"
        )
    assert cfg.data.b is not None and cfg.data.k is not None
    budget_per_round = cfg.data.b // cfg.data.k

    torch.use_deterministic_algorithms(True)
    verify_manifest_contract(manifest_root, "base", cfg)
    verify_manifest_contract(manifest_root, "collection", cfg)
    collection_entries, collection_hash = load_split_manifest(manifest_root, "collection")

    bundle_root = Path(data_root) / contract_hash(cfg)[:12] / bundle_id

    # 1. Shared base data and the immutable base checkpoint.
    base_summary = collect_base(cfg, bundle_id, manifest_root, data_root)
    dataset_dir = base_dataset_dir(cfg, bundle_id, data_root)
    base_result = train_base(cfg, bundle_id, dataset_dir, bundle_root)
    base_payload = torch.load(
        base_result.checkpoint_path, map_location="cpu", weights_only=False
    )
    base_digest = model_state_digest(base_payload["model_state"])

    vocab = vocabulary_from_dataset(dataset_dir)
    base_windows = load_all_windows(cfg, dataset_dir, vocab)

    # 2. Clone the two augmented arms from the identical starting state.
    arms = {}
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        model, optimizer, _meta = clone_arm_from_checkpoint(
            cfg, base_result.checkpoint_path, vocab
        )
        arms[arm] = {
            "model": model,
            "optimizer": optimizer,
            "new_windows": [],
            "offset": 0,
            "cumulative": {"base": 0, "new": 0, "updates": 0},
            "collections": [],
        }
    assert_clone_equality(
        arms[ARM_EXTRA]["model"],
        arms[ARM_RECOVERY]["model"],
        arms[ARM_EXTRA]["optimizer"],
        arms[ARM_RECOVERY]["optimizer"],
    )

    partitions = {arm: arm_partition(collection_entries, arm) for arm in arms}
    sampler_rngs = {
        ARM_EXTRA: _named_rng(cfg, bundle_id, "sampler.extra_demo"),
        ARM_RECOVERY: _named_rng(cfg, bundle_id, "sampler.recovery"),
    }
    writers = {
        arm: MetricsWriter(bundle_root / "training" / arm / "exposure_ledger.jsonl")
        for arm in arms
    }

    fairness_reports = []
    try:
        # 3. Rounds: exact collection budgets, then equal-update training.
        for round_index in range(1, cfg.data.k + 1):
            extra = arms[ARM_EXTRA]
            summary = collect_extra_demo_round(
                cfg, bundle_id, round_index, budget_per_round,
                partitions[ARM_EXTRA], extra["offset"], collection_hash, data_root,
            )
            extra["offset"] += summary.entries_consumed
            extra["collections"].append(asdict(summary))
            extra["new_windows"].extend(
                load_all_windows(cfg, Path(summary.dataset_dir), vocab)
            )

            recovery = arms[ARM_RECOVERY]
            summary = collect_recovery_round(
                cfg, bundle_id, round_index, budget_per_round,
                partitions[ARM_RECOVERY], recovery["offset"], collection_hash,
                data_root, recovery["model"], vocab,
            )
            recovery["offset"] += summary.entries_consumed
            recovery["collections"].append(asdict(summary))
            recovery["new_windows"].extend(
                load_all_windows(cfg, Path(summary.dataset_dir), vocab)
            )

            for arm in (ARM_EXTRA, ARM_RECOVERY):
                train_arm_round(
                    cfg, bundle_id, arm, round_index,
                    arms[arm]["model"], arms[arm]["optimizer"],
                    base_windows, arms[arm]["new_windows"],
                    sampler_rngs[arm], writers[arm], arms[arm]["cumulative"],
                )
            if arms[ARM_EXTRA]["cumulative"] != arms[ARM_RECOVERY]["cumulative"]:
                raise ManifestError(
                    f"round {round_index}: cumulative exposure mismatch between arms"
                )
    finally:
        for writer in writers.values():
            writer.close()

    # 4. Fairness audit over the complete exposure ledgers.
    from grounded_recovery.artifacts import read_jsonl as _read_jsonl

    ledger_rows = {
        arm: _read_jsonl(bundle_root / "training" / arm / "exposure_ledger.jsonl")
        for arm in arms
    }
    fairness_reports.append(
        audit_round_fairness(
            ledger_rows[ARM_EXTRA], ledger_rows[ARM_RECOVERY],
            arm_a=ARM_EXTRA, arm_b=ARM_RECOVERY,
        )
    )

    # 5. The base checkpoint must be untouched; write both final arm checkpoints.
    reread = torch.load(
        base_result.checkpoint_path, map_location="cpu", weights_only=False
    )
    if model_state_digest(reread["model_state"]) != base_digest:
        raise ManifestError("the shared base checkpoint changed during arm training")
    final_paths = {}
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        meta = CheckpointMeta(
            contract_hash=contract_hash(cfg),
            model_config_hash=base_payload["meta"]["model_config_hash"],
            bundle_id=bundle_id,
            arm=arm,
            round_index=cfg.data.k,
            update_index=arms[arm]["cumulative"]["updates"],
            vocabulary=vocab.tokens,
            action_names=cfg.environment.action_names,
            action_ids=cfg.environment.action_ids,
            dataset_schema_version=cfg.data.dataset_schema_version,
            metrics_ledger_hash=writers[arm].final_hash,
            parameter_count=arms[arm]["model"].parameter_count(),
        )
        path = bundle_root / "checkpoints" / f"{arm}_round_{cfg.data.k:02d}.pt"
        save_checkpoint(
            path, arms[arm]["model"], arms[arm]["optimizer"], meta, sampler_rngs[arm]
        )
        final_paths[arm] = str(path)

    summary = {
        "bundle_id": bundle_id,
        "contract_hash": contract_hash(cfg),
        "base": {
            "n0": base_summary.n0,
            "episodes": base_summary.episodes,
            "checkpoint": base_result.checkpoint_path,
            "state_digest": base_digest,
            "final_loss": base_result.final_loss,
        },
        "rounds": cfg.data.k,
        "budget_per_round": budget_per_round,
        "arms": {
            arm: {
                "collections": arms[arm]["collections"],
                "cumulative": arms[arm]["cumulative"],
                "final_checkpoint": final_paths[arm],
                "final_state_digest": model_state_digest(
                    arms[arm]["model"].state_dict()
                ),
            }
            for arm in arms
        },
        "fairness": fairness_reports,
    }
    atomic_write_json(bundle_root / "bundle_summary.json", summary)
    return summary


def _named_rng(cfg: ExperimentConfig, bundle_id: str, component: str):
    import numpy as np

    from grounded_recovery.seeds import derive_seed

    return np.random.default_rng(derive_seed(cfg.seeds.root_seed, bundle_id, component))


# --- Validation pilot ---------------------------------------------------------

def run_validation_pilot(
    cfg: ExperimentConfig, bundle_id: str, manifest_root: Path, data_root: Path
) -> dict[str, object]:
    """One pilot bundle plus a crossed validation-slice evaluation.

    Tuning evidence for gates G4/G6 only: the panel is the validation split;
    test-candidate and eligible-test scenarios are never touched here.
    """
    from dataclasses import asdict as dc_asdict

    from grounded_recovery.data import base_dataset_dir, vocabulary_from_dataset
    from grounded_recovery.evaluate import (
        SLICE_CLEAN,
        SLICE_MATCHED,
        SLICE_UNSEEN,
        evaluate_policy_on_entries,
        evaluation_row_to_json,
        load_arm_policy,
    )
    from grounded_recovery.perturbations import operator_from_config
    from grounded_recovery.statistics import paired_difference, success_summary
    from grounded_recovery.world import WorldSession

    if cfg.perturbation.unseen_time_set is None:
        raise ManifestError(
            "perturbation.unseen_time_set is unresolved (PILOT_TO_FREEZE); the pilot "
            "evaluation needs a provisional unseen schedule"
        )
    bundle_summary = run_pilot_bundle(cfg, bundle_id, manifest_root, data_root)

    entries, _ = load_split_manifest(manifest_root, "validation")
    vocab = vocabulary_from_dataset(base_dataset_dir(cfg, bundle_id, data_root))
    checkpoints = {
        "bc_base": bundle_summary["base"]["checkpoint"],
        ARM_EXTRA: bundle_summary["arms"][ARM_EXTRA]["final_checkpoint"],
        ARM_RECOVERY: bundle_summary["arms"][ARM_RECOVERY]["final_checkpoint"],
    }
    slices = {
        SLICE_CLEAN: (None, None),
        SLICE_MATCHED: (
            operator_from_config(
                cfg.perturbation.collection_operator, cfg.environment.action_ids
            ),
            cfg.perturbation.collection_time_set,
        ),
        SLICE_UNSEEN: (
            operator_from_config(
                cfg.perturbation.unseen_operator, cfg.environment.action_ids
            ),
            cfg.perturbation.unseen_time_set,
        ),
    }
    session = WorldSession(cfg.environment)
    rows_by_cell: dict[str, dict[str, list]] = {}
    try:
        for arm, checkpoint in checkpoints.items():
            policy = load_arm_policy(cfg, checkpoint, vocab)
            rows_by_cell[arm] = {}
            for slice_name, (operator, time_set) in slices.items():
                rows_by_cell[arm][slice_name] = evaluate_policy_on_entries(
                    cfg, session, policy, vocab, entries,
                    bundle_id=bundle_id, arm=arm, slice_name=slice_name,
                    operator=operator, time_set=time_set,
                )
    finally:
        session.close()

    report: dict[str, object] = {
        "phase": "validation_pilot",
        "panel": "validation",
        "panel_scenarios": len(entries),
        "bundle": bundle_summary,
        "success": {
            arm: {
                slice_name: dc_asdict(success_summary(rows))
                for slice_name, rows in by_slice.items()
            }
            for arm, by_slice in rows_by_cell.items()
        },
        "paired_recovery_minus_extra": {
            slice_name: paired_difference(
                rows_by_cell[ARM_RECOVERY][slice_name],
                rows_by_cell[ARM_EXTRA][slice_name],
            )
            for slice_name in slices
        },
    }
    bundle_root = Path(data_root) / contract_hash(cfg)[:12] / bundle_id
    atomic_write_jsonl(
        bundle_root / "validation_evaluation_rows.jsonl",
        (
            evaluation_row_to_json(row)
            for by_slice in rows_by_cell.values()
            for rows in by_slice.values()
            for row in rows
        ),
    )
    atomic_write_json(bundle_root / "pilot_report.json", report)
    return report


# --- Freeze, frozen bundle runs, and the single confirmatory opening ---------

BC_BASE = "bc_base"

FINAL_SLICES = ("clean", "matched", "unseen")

# Preregistered interpretation matrix for the primary paired contrast
# (recovery minus extra demonstrations on the eligible unseen ITT endpoint),
# applied in this order to the prespecified 95% paired t interval:
CLAIM_DECISION_RULE = (
    "support iff interval lower bound > 0; "
    "adverse iff interval upper bound < 0; "
    "rule_out iff interval upper bound < SESOI and lower bound > -SESOI; "
    "inconclusive otherwise. Confirmatory status iff the number of complete "
    "bundles >= max(5, planned R_train); the precision target is reported "
    "with a met/unmet flag and does not by itself change status."
)


def compute_code_hash(repo_root: Path) -> str:
    """SHA-256 over the research-code surface (sorted source files + pins)."""
    import hashlib

    digest = hashlib.sha256()
    repo_root = Path(repo_root)
    files = sorted((repo_root / "src" / "grounded_recovery").glob("*.py"))
    files += [repo_root / "pyproject.toml", repo_root / "uv.lock"]
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\x1f")
        digest.update(path.read_bytes())
        digest.update(b"\x1f")
    return digest.hexdigest()


def _find_preflight_evidence(
    cfg: ExperimentConfig, data_root: Path, preflight_manifest_hash: str
) -> dict[str, object]:
    from grounded_recovery.artifacts import read_json

    candidates = sorted(Path(data_root).glob("preflight/*/preflight_report.json"))
    for path in candidates:
        report = read_json(path)
        if report["manifest_hash"] != preflight_manifest_hash:
            continue
        if not report["passed"]:
            continue
        family_names = {
            family: summary["operator_name"]
            for family, summary in report["families"].items()
        }
        expected = {
            "collection": cfg.perturbation.collection_operator.name,
            "unseen": cfg.perturbation.unseen_operator.name,
        }
        if family_names != expected:
            continue
        gates = {s["gate"] for s in report["families"].values()}
        if gates != {cfg.environment.oracle_recovery_gate}:
            continue
        return {"path": str(path), "report": report}
    raise ManifestError(
        "no passing preflight report matches the current operator-preflight manifest, "
        "operators, and gate; run `gr preflight` before freezing"
    )


def run_freeze(
    pilot_path: Path,
    contract_path: Path,
    manifest_root: Path,
    data_root: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Resolve the pilot configuration into the frozen experiment contract.

    Requires: every PILOT_TO_FREEZE value resolved, all eight manifests valid
    under the current scenario identity, split disjointness recorded, passing
    at-scale preflight evidence for the current operators, and a derivable
    eligible unseen panel. Writes the frozen contract and a freeze record;
    refuses to overwrite an existing contract.
    """
    import importlib.metadata
    import time as time_module

    import yaml

    from grounded_recovery.artifacts import read_json
    from grounded_recovery.config import load_and_validate, unresolved_fields
    from grounded_recovery.data import ELIGIBLE_FILE, derive_eligible_subset, load_eligible_entries

    pilot_path = Path(pilot_path)
    contract_path = Path(contract_path)
    if contract_path.exists():
        raise ManifestError(
            f"frozen contract already exists at {contract_path}; a changed contract is a "
            "new protocol version, never an overwrite"
        )
    cfg = load_and_validate(pilot_path)
    if cfg.study.status != "PILOT":
        raise ManifestError("freeze consumes the pilot configuration (status PILOT)")
    unresolved = unresolved_fields(cfg)
    if unresolved:
        raise ManifestError(f"cannot freeze with unresolved fields: {list(unresolved)}")

    manifest_hashes = {}
    for split in SPLIT_NAMES:
        verify_manifest_contract(manifest_root, split, cfg)
        _, manifest_hashes[split] = load_split_manifest(manifest_root, split)
    disjointness = read_json(Path(manifest_root) / "disjointness_report.json")
    if not disjointness["audit"]["disjoint"]:
        raise ManifestError("split disjointness report does not verify")

    preflight = _find_preflight_evidence(
        cfg, data_root, manifest_hashes["operator_preflight"]
    )

    if (Path(manifest_root) / ELIGIBLE_FILE).exists():
        _, eligible = load_eligible_entries(cfg, manifest_root)
    else:
        eligible = derive_eligible_subset(cfg, manifest_root)

    raw = yaml.safe_load(pilot_path.read_text())
    raw["study"]["status"] = "FROZEN"
    frozen_text = yaml.safe_dump(raw, sort_keys=False)
    from grounded_recovery.artifacts import atomic_write_bytes

    atomic_write_bytes(contract_path, frozen_text.encode("utf-8"))
    frozen_cfg = load_and_validate(contract_path)

    record = {
        "protocol_version": frozen_cfg.study.protocol_version,
        "contract_hash": contract_hash(frozen_cfg),
        "scenario_identity_hash": scenario_identity_hash(frozen_cfg),
        "code_hash": compute_code_hash(repo_root),
        "manifest_hashes": manifest_hashes,
        "eligible": {
            "count": eligible["eligible_count"],
            "candidates": eligible["candidate_count"],
            "retained_fraction": eligible["retained_fraction"],
            "eligible_hash": eligible["eligible_hash"],
        },
        "preflight_report": preflight["path"],
        "preflight_manifest_hash": manifest_hashes["operator_preflight"],
        "claim_decision_rule": CLAIM_DECISION_RULE,
        "planned_bundles": list(frozen_cfg.seeds.bundle_ids),
        "r_train": min(frozen_cfg.evaluation.r_target, frozen_cfg.evaluation.r_max),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "minigrid", "gymnasium", "numpy", "scipy")
        },
        "frozen_at_unix": round(time_module.time(), 1),
    }
    atomic_write_json(contract_path.with_name("freeze_record.json"), record)
    return record


def _require_frozen(cfg: ExperimentConfig) -> None:
    from grounded_recovery.config import unresolved_fields

    if cfg.study.status != "FROZEN":
        raise ManifestError("this command accepts only the frozen experiment contract")
    if unresolved_fields(cfg):
        raise ManifestError("the frozen contract contains unresolved fields")


def run_bundle_frozen(
    contract_path: Path, bundle_id: str, manifest_root: Path, data_root: Path
) -> dict[str, object]:
    """Run one final pipeline bundle; accepts only the frozen contract."""
    from grounded_recovery.config import load_and_validate

    cfg = load_and_validate(Path(contract_path))
    _require_frozen(cfg)
    return run_pilot_bundle(cfg, bundle_id, manifest_root, data_root)


def evaluate_final(
    contract_path: Path,
    manifest_root: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, object]:
    """The single confirmatory test opening.

    Writes an opening receipt before any outcome is read, evaluates every
    bundle's three policies on the identical crossed eligible panel (clean,
    matched, and the primary eligible-unseen one-corruption ITT slice), and
    stores the raw episode rows. A second opening under the same contract is
    refused; a verified technical invalidation and rerun is a new protocol
    version, never an in-place overwrite.
    """
    import time as time_module

    from grounded_recovery.config import load_and_validate
    from grounded_recovery.data import (
        base_dataset_dir,
        load_eligible_entries,
        vocabulary_from_dataset,
    )
    from grounded_recovery.evaluate import (
        evaluate_policy_on_entries,
        evaluation_row_to_json,
        load_arm_policy,
    )
    from grounded_recovery.perturbations import operator_from_config
    from grounded_recovery.train import model_state_digest
    from grounded_recovery.world import WorldSession

    cfg = load_and_validate(Path(contract_path))
    _require_frozen(cfg)
    cfg_hash = contract_hash(cfg)
    results_dir = Path(results_root) / cfg_hash[:12]
    if results_dir.exists():
        raise ManifestError(
            f"a confirmatory opening already exists at {results_dir}; there is exactly "
            "one opening per frozen contract"
        )

    eligible_entries, eligible_meta = load_eligible_entries(cfg, manifest_root)

    import torch

    bundles = {}
    for bundle_id in cfg.seeds.bundle_ids:
        bundle_root = Path(data_root) / cfg_hash[:12] / bundle_id
        from grounded_recovery.artifacts import read_json

        summary = read_json(bundle_root / "bundle_summary.json")
        checkpoints = {
            BC_BASE: summary["base"]["checkpoint"],
            ARM_EXTRA: summary["arms"][ARM_EXTRA]["final_checkpoint"],
            ARM_RECOVERY: summary["arms"][ARM_RECOVERY]["final_checkpoint"],
        }
        digests = {}
        for arm, path in checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            digests[arm] = model_state_digest(payload["model_state"])
        if digests[BC_BASE] != summary["base"]["state_digest"]:
            raise ManifestError(f"bundle {bundle_id}: base checkpoint digest changed")
        bundles[bundle_id] = {"checkpoints": checkpoints, "digests": digests,
                              "root": bundle_root}

    # The receipt is written before any outcome is computed or read.
    receipt = {
        "state": "OPENING_STARTED",
        "opened_at_unix": round(time_module.time(), 1),
        "contract_hash": cfg_hash,
        "eligible_hash": eligible_meta["eligible_hash"],
        "eligible_count": eligible_meta["eligible_count"],
        "candidate_count": eligible_meta["candidate_count"],
        "bundles": {
            bundle_id: info["digests"] for bundle_id, info in bundles.items()
        },
        "expected_cells": len(bundles) * 3 * len(FINAL_SLICES),
        "primary_slice": "unseen",
        "primary_contrast": cfg.study.primary_contrast,
    }
    results_dir.mkdir(parents=True)
    atomic_write_json(results_dir / "opening_receipt.json", receipt)

    slices = {
        "clean": (None, None),
        "matched": (
            operator_from_config(
                cfg.perturbation.collection_operator, cfg.environment.action_ids
            ),
            cfg.perturbation.collection_time_set,
        ),
        "unseen": (
            operator_from_config(
                cfg.perturbation.unseen_operator, cfg.environment.action_ids
            ),
            cfg.perturbation.unseen_time_set,
        ),
    }
    session = WorldSession(cfg.environment)
    all_rows = []
    completed_cells = 0
    try:
        for bundle_id, info in bundles.items():
            vocab = vocabulary_from_dataset(base_dataset_dir(cfg, bundle_id, data_root))
            for arm, checkpoint in info["checkpoints"].items():
                policy = load_arm_policy(cfg, checkpoint, vocab)
                for slice_name, (operator, time_set) in slices.items():
                    rows = evaluate_policy_on_entries(
                        cfg, session, policy, vocab, eligible_entries,
                        bundle_id=bundle_id, arm=arm, slice_name=slice_name,
                        operator=operator, time_set=time_set,
                    )
                    all_rows.extend(rows)
                    completed_cells += 1
    finally:
        session.close()

    rows_hash = atomic_write_jsonl(
        results_dir / "raw_episodes.jsonl",
        (evaluation_row_to_json(row) for row in all_rows),
    )
    atomic_write_json(
        results_dir / "opening_complete.json",
        {
            "state": "COMPLETE",
            "completed_at_unix": round(time_module.time(), 1),
            "completed_cells": completed_cells,
            "expected_cells": receipt["expected_cells"],
            "raw_episodes_sha256": rows_hash,
            "rows": len(all_rows),
        },
    )
    return {"results_dir": str(results_dir), "rows": len(all_rows),
            "cells": completed_cells}
