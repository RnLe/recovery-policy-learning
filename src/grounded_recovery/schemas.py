"""Record schemas and canonical scenario identity.

The canonical scenario hash is computed from a documented serialization of the
reset world and mission, not from the seed, so that split disjointness is a
statement about worlds, and a future environment-version drift that changes
generated worlds is detectable. The serialization is version-tagged: changing
it requires a new ``SCENARIO_HASH_VERSION``, never a silent redefinition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from grounded_recovery.artifacts import canonical_json_bytes, hash_json
from grounded_recovery.world import ScenarioState

if TYPE_CHECKING:
    # Type-only: importing at runtime would be circular (oracle imports the
    # scenario hash from this module).
    from grounded_recovery.oracle import EpisodeTrace

SCENARIO_HASH_VERSION = 1

MANIFEST_VERSION = 1

DATASET_SCHEMA_VERSION = "1"

# Storage sentinel for "no value" in int8 action arrays. It is a storage
# convention only and never becomes a model token.
NULL_ACTION = -1


def canonical_scenario_hash(state: ScenarioState) -> str:
    """SHA-256 identity of a reset world under the documented serialization."""
    payload = {
        "scenario_hash_version": SCENARIO_HASH_VERSION,
        "env_id": state.env_id,
        "grid": [
            [[int(channel) for channel in cell] for cell in column]
            for column in state.grid_encoding
        ],
        "agent_pos": [int(state.agent_pos[0]), int(state.agent_pos[1])],
        "agent_dir": int(state.agent_dir),
        "mission": state.mission,
    }
    return hash_json(payload)


@dataclass(frozen=True)
class ManifestEntry:
    """One scenario of one purpose split."""

    split_name: str
    ordinal: int
    environment_id: str
    environment_seed: int
    canonical_scenario_hash: str
    mission: str
    nominal_oracle_path_length: int
    perturbation_family: str | None
    scheduled_intervention_times: tuple[int, ...]
    manifest_version: int


def manifest_entry_to_json(entry: ManifestEntry) -> dict[str, object]:
    row = asdict(entry)
    row["scheduled_intervention_times"] = list(entry.scheduled_intervention_times)
    return row


def manifest_entry_from_json(row: Mapping[str, object]) -> ManifestEntry:
    return ManifestEntry(
        split_name=str(row["split_name"]),
        ordinal=int(row["ordinal"]),
        environment_id=str(row["environment_id"]),
        environment_seed=int(row["environment_seed"]),
        canonical_scenario_hash=str(row["canonical_scenario_hash"]),
        mission=str(row["mission"]),
        nominal_oracle_path_length=int(row["nominal_oracle_path_length"]),
        perturbation_family=(
            None if row["perturbation_family"] is None else str(row["perturbation_family"])
        ),
        scheduled_intervention_times=tuple(
            int(value) for value in row["scheduled_intervention_times"]
        ),
        manifest_version=int(row["manifest_version"]),
    )


@dataclass(frozen=True)
class PreflightEpisodeRow:
    """Outcome of one oracle-only preflight episode with one forced corruption."""

    family: str
    operator_name: str
    ordinal: int
    environment_seed: int
    scenario_hash: str
    scheduled_time: int
    delivered: bool
    recommended_at_scheduled_time: int | None
    forced_action: int | None
    success: bool
    steps: int
    nominal_oracle_path_length: int
    oracle_calls: int
    truncated: bool
    termination_reason: str
    contract_hash: str
    manifest_hash: str


def preflight_row_to_json(row: PreflightEpisodeRow) -> dict[str, object]:
    return asdict(row)


# --- Episode records ---------------------------------------------------------

# Canonical array field order for checksumming; changing it is a dataset
# schema change.
EPISODE_ARRAY_FIELDS: tuple[str, ...] = (
    "images",
    "direction",
    "previous_executed_action",
    "policy_proposed_action",
    "oracle_recommended_action",
    "target_revealed",
    "executed_action",
    "perturbation_scheduled",
    "perturbation_delivered",
    "oracle_called",
    "synchronization_only",
    "terminated",
    "truncated",
    "reward",
)

_EPISODE_ARRAY_DTYPES: dict[str, np.dtype] = {
    "images": np.dtype(np.uint8),
    "direction": np.dtype(np.uint8),
    "previous_executed_action": np.dtype(np.int8),
    "policy_proposed_action": np.dtype(np.int8),
    "oracle_recommended_action": np.dtype(np.int8),
    "target_revealed": np.dtype(np.bool_),
    "executed_action": np.dtype(np.int8),
    "perturbation_scheduled": np.dtype(np.bool_),
    "perturbation_delivered": np.dtype(np.bool_),
    "oracle_called": np.dtype(np.bool_),
    "synchronization_only": np.dtype(np.bool_),
    "terminated": np.dtype(np.bool_),
    "truncated": np.dtype(np.bool_),
    "reward": np.dtype(np.float32),
}


class EpisodeSchemaError(ValueError):
    """Episode arrays or sidecar violate the dataset schema."""


@dataclass(frozen=True)
class EpisodeArrays:
    """Per-step arrays of one episode; every field has length T."""

    images: np.ndarray
    direction: np.ndarray
    previous_executed_action: np.ndarray
    policy_proposed_action: np.ndarray
    oracle_recommended_action: np.ndarray
    target_revealed: np.ndarray
    executed_action: np.ndarray
    perturbation_scheduled: np.ndarray
    perturbation_delivered: np.ndarray
    oracle_called: np.ndarray
    synchronization_only: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reward: np.ndarray

    def __post_init__(self) -> None:
        length = self.images.shape[0]
        for name in EPISODE_ARRAY_FIELDS:
            array = getattr(self, name)
            expected = _EPISODE_ARRAY_DTYPES[name]
            if array.dtype != expected:
                raise EpisodeSchemaError(f"{name}: dtype {array.dtype}, expected {expected}")
            if array.shape[0] != length:
                raise EpisodeSchemaError(
                    f"{name}: length {array.shape[0]} != episode length {length}"
                )
        if self.images.ndim != 4 or self.images.shape[1:] != (7, 7, 3):
            raise EpisodeSchemaError(f"images: shape {self.images.shape}, expected [T,7,7,3]")
        # Null-sentinel discipline: the oracle field is null exactly where the
        # oracle was not called, and the previous executed action is null
        # exactly at t=0.
        oracle_null = self.oracle_recommended_action == NULL_ACTION
        if not np.array_equal(oracle_null, ~self.oracle_called):
            raise EpisodeSchemaError(
                "oracle_recommended_action must be NULL_ACTION exactly where "
                "oracle_called is False"
            )
        if length > 0:
            prev_null = self.previous_executed_action == NULL_ACTION
            expected_prev_null = np.zeros(length, dtype=np.bool_)
            expected_prev_null[0] = True
            if not np.array_equal(prev_null, expected_prev_null):
                raise EpisodeSchemaError(
                    "previous_executed_action must be NULL_ACTION exactly at t=0"
                )
        if np.any(self.target_revealed & ~self.oracle_called):
            raise EpisodeSchemaError("a target cannot be revealed without an oracle call")
        if not np.array_equal(
            self.synchronization_only, self.oracle_called & ~self.target_revealed
        ):
            raise EpisodeSchemaError(
                "synchronization_only must equal oracle_called and not target_revealed"
            )

    @property
    def length(self) -> int:
        return int(self.images.shape[0])

    @property
    def revealed_targets(self) -> int:
        return int(self.target_revealed.sum())


@dataclass(frozen=True)
class EpisodeSidecar:
    """Human-readable identity, provenance, and checksum of one stored episode."""

    episode_id: str
    environment_seed: int
    canonical_scenario_hash: str
    mission: str
    source_arm: str
    round_index: int
    success: bool
    truncated: bool
    stopped_early: bool
    executed_length: int
    revealed_targets: int
    oracle_calls: int
    termination_reason: str
    intervention: dict[str, object] | None
    dataset_schema_version: str
    contract_hash: str
    manifest_hash: str
    content_checksum: str


_IDENTITY_SIDECAR_FIELDS: tuple[str, ...] = (
    "episode_id",
    "environment_seed",
    "canonical_scenario_hash",
    "mission",
    "source_arm",
    "round_index",
)


def episode_content_checksum(arrays: EpisodeArrays, identity: Mapping[str, object]) -> str:
    """Content-addressed identity of one episode.

    Hashes dtype, shape, and raw bytes of every array in the canonical field
    order plus the canonical JSON of the identity subset of the sidecar. File
    bytes (npz containers embed timestamps) are deliberately not the identity.
    """
    missing = [name for name in _IDENTITY_SIDECAR_FIELDS if name not in identity]
    if missing:
        raise EpisodeSchemaError(f"identity mapping missing fields: {missing}")
    digest = hashlib.sha256()
    digest.update(b"dataset_schema:" + DATASET_SCHEMA_VERSION.encode("ascii") + b"\x1f")
    for name in EPISODE_ARRAY_FIELDS:
        array = getattr(arrays, name)
        digest.update(name.encode("ascii") + b"\x1f")
        digest.update(str(array.dtype).encode("ascii") + b"\x1f")
        digest.update(repr(array.shape).encode("ascii") + b"\x1f")
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"\x1f")
    digest.update(
        canonical_json_bytes({name: identity[name] for name in _IDENTITY_SIDECAR_FIELDS})
    )
    return digest.hexdigest()


def episode_from_trace(
    trace: EpisodeTrace,
    *,
    episode_id: str,
    reveal_mask: np.ndarray,
    source_arm: str,
    round_index: int,
    termination_reason: str,
    intervention: dict[str, object] | None,
    contract_hash: str,
    manifest_hash: str,
) -> tuple[EpisodeArrays, EpisodeSidecar]:
    """Transform one synchronized episode trace into storable records.

    ``reveal_mask[t]`` marks the steps whose oracle recommendation is a
    revealed supervision target. In this collection design the oracle is
    synchronized on every active step, so ``oracle_called`` is all-True and
    non-revealed steps are synchronization-only.
    """
    length = len(trace.transitions)
    reveal = np.asarray(reveal_mask, dtype=np.bool_)
    if reveal.shape != (length,):
        raise EpisodeSchemaError(
            f"reveal_mask shape {reveal.shape} does not match episode length {length}"
        )
    images = np.stack([obs.image for obs in trace.observations[:length]]).astype(np.uint8)
    direction = np.array(
        [obs.direction for obs in trace.observations[:length]], dtype=np.uint8
    )
    previous_executed = np.full(length, NULL_ACTION, dtype=np.int8)
    for t in range(1, length):
        previous_executed[t] = trace.transitions[t - 1].executed
    proposed = np.array(
        [
            NULL_ACTION if transition.proposed is None else transition.proposed
            for transition in trace.transitions
        ],
        dtype=np.int8,
    )
    recommended = np.array(
        [transition.recommended for transition in trace.transitions], dtype=np.int8
    )
    executed = np.array(
        [transition.executed for transition in trace.transitions], dtype=np.int8
    )
    scheduled_mask = np.zeros(length, dtype=np.bool_)
    delivered_mask = np.zeros(length, dtype=np.bool_)
    if intervention is not None:
        scheduled_time = int(intervention["scheduled_time"])
        if 0 <= scheduled_time < length:
            scheduled_mask[scheduled_time] = True
            if bool(intervention.get("delivered", False)):
                delivered_mask[scheduled_time] = True
    oracle_called = np.ones(length, dtype=np.bool_)
    arrays = EpisodeArrays(
        images=images,
        direction=direction,
        previous_executed_action=previous_executed,
        policy_proposed_action=proposed,
        oracle_recommended_action=recommended,
        target_revealed=reveal,
        executed_action=executed,
        perturbation_scheduled=scheduled_mask,
        perturbation_delivered=delivered_mask,
        oracle_called=oracle_called,
        synchronization_only=oracle_called & ~reveal,
        terminated=np.array(
            [transition.terminated for transition in trace.transitions], dtype=np.bool_
        ),
        truncated=np.array(
            [transition.truncated for transition in trace.transitions], dtype=np.bool_
        ),
        reward=np.array(
            [transition.reward for transition in trace.transitions], dtype=np.float32
        ),
    )
    identity = {
        "episode_id": episode_id,
        "environment_seed": trace.seed,
        "canonical_scenario_hash": trace.scenario_hash,
        "mission": trace.mission,
        "source_arm": source_arm,
        "round_index": round_index,
    }
    sidecar = EpisodeSidecar(
        episode_id=episode_id,
        environment_seed=trace.seed,
        canonical_scenario_hash=trace.scenario_hash,
        mission=trace.mission,
        source_arm=source_arm,
        round_index=round_index,
        success=trace.success,
        truncated=trace.truncated,
        stopped_early=trace.stopped_early,
        executed_length=length,
        revealed_targets=int(reveal.sum()),
        oracle_calls=trace.oracle_calls,
        termination_reason=termination_reason,
        intervention=intervention,
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        contract_hash=contract_hash,
        manifest_hash=manifest_hash,
        content_checksum=episode_content_checksum(arrays, identity),
    )
    return arrays, sidecar


def sidecar_to_json(sidecar: EpisodeSidecar) -> dict[str, object]:
    return asdict(sidecar)


def sidecar_from_json(row: Mapping[str, object]) -> EpisodeSidecar:
    kwargs = {field.name: row[field.name] for field in fields(EpisodeSidecar)}
    kwargs["environment_seed"] = int(kwargs["environment_seed"])
    kwargs["round_index"] = int(kwargs["round_index"])
    kwargs["executed_length"] = int(kwargs["executed_length"])
    kwargs["revealed_targets"] = int(kwargs["revealed_targets"])
    kwargs["oracle_calls"] = int(kwargs["oracle_calls"])
    return EpisodeSidecar(**kwargs)
