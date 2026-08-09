"""Episode record codec: schema discipline, checksums, storage round-trip.

These tests protect the ground truth layer: every later claim reduces to
stored episode arrays, so silent corruption, container-byte identity
confusion, or sentinel misuse would invalidate everything downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from grounded_recovery.data import ManifestError, read_episode, write_episode
from grounded_recovery.schemas import (
    NULL_ACTION,
    EpisodeArrays,
    EpisodeSchemaError,
    EpisodeSidecar,
    episode_content_checksum,
)


def make_arrays(length: int = 5, revealed: int | None = None) -> EpisodeArrays:
    rng = np.random.default_rng(7)
    reveal = np.ones(length, dtype=np.bool_)
    if revealed is not None:
        reveal[:] = False
        reveal[:revealed] = True
    executed = rng.integers(0, 3, size=length).astype(np.int8)
    previous = np.empty(length, dtype=np.int8)
    previous[0] = NULL_ACTION
    previous[1:] = executed[:-1]
    terminated = np.zeros(length, dtype=np.bool_)
    terminated[-1] = True
    images = np.stack(
        (
            rng.integers(0, 11, size=(length, 7, 7)),  # object channel
            rng.integers(0, 6, size=(length, 7, 7)),  # color channel
            rng.integers(0, 4, size=(length, 7, 7)),  # state channel
        ),
        axis=-1,
    ).astype(np.uint8)
    return EpisodeArrays(
        images=images,
        direction=rng.integers(0, 4, size=length).astype(np.uint8),
        previous_executed_action=previous,
        policy_proposed_action=np.full(length, NULL_ACTION, dtype=np.int8),
        oracle_recommended_action=rng.integers(0, 3, size=length).astype(np.int8),
        target_revealed=reveal,
        executed_action=executed,
        perturbation_scheduled=np.zeros(length, dtype=np.bool_),
        perturbation_delivered=np.zeros(length, dtype=np.bool_),
        oracle_called=np.ones(length, dtype=np.bool_),
        synchronization_only=~reveal,
        terminated=terminated,
        truncated=np.zeros(length, dtype=np.bool_),
        reward=rng.random(length).astype(np.float32),
    )


def identity_for(episode_id: str) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "environment_seed": 12345,
        "canonical_scenario_hash": "ab" * 32,
        "mission": "go to the red ball",
        "source_arm": "base",
        "round_index": 0,
    }


def make_sidecar(arrays: EpisodeArrays, episode_id: str = "base_00000") -> EpisodeSidecar:
    return EpisodeSidecar(
        episode_id=episode_id,
        environment_seed=12345,
        canonical_scenario_hash="ab" * 32,
        mission="go to the red ball",
        source_arm="base",
        round_index=0,
        success=True,
        truncated=False,
        stopped_early=False,
        executed_length=arrays.length,
        revealed_targets=arrays.revealed_targets,
        oracle_calls=arrays.length,
        termination_reason="terminated",
        intervention=None,
        dataset_schema_version="1",
        contract_hash="cd" * 32,
        manifest_hash="ef" * 32,
        content_checksum=episode_content_checksum(arrays, identity_for(episode_id)),
    )


def test_write_read_roundtrip_bitexact(tmp_path) -> None:
    arrays = make_arrays()
    sidecar = make_sidecar(arrays)
    write_episode(tmp_path, arrays, sidecar)
    loaded_arrays, loaded_sidecar = read_episode(tmp_path, sidecar.episode_id)
    for name in (
        "images", "direction", "previous_executed_action", "executed_action", "reward"
    ):
        assert np.array_equal(getattr(arrays, name), getattr(loaded_arrays, name))
    assert loaded_sidecar == sidecar


def test_checksum_detects_single_element_change(tmp_path) -> None:
    arrays = make_arrays()
    sidecar = make_sidecar(arrays)
    write_episode(tmp_path, arrays, sidecar)
    # Corrupt one byte of one array inside the stored npz by rewriting it.
    import zipfile

    npz_path = tmp_path / f"{sidecar.episode_id}.npz"
    tampered = dict(np.load(npz_path))
    tampered["executed_action"] = tampered["executed_action"].copy()
    tampered["executed_action"][0] = (tampered["executed_action"][0] + 1) % 3
    np.savez_compressed(npz_path, **tampered)
    assert zipfile.is_zipfile(npz_path)
    with pytest.raises(ManifestError, match="checksum"):
        read_episode(tmp_path, sidecar.episode_id)


def test_checksum_independent_of_container_bytes(tmp_path) -> None:
    arrays = make_arrays()
    sidecar = make_sidecar(arrays)
    write_episode(tmp_path / "a", arrays, sidecar)
    write_episode(tmp_path / "b", arrays, sidecar)
    # Container bytes may differ (zip timestamps); content identity may not.
    _, sidecar_a = read_episode(tmp_path / "a", sidecar.episode_id)
    _, sidecar_b = read_episode(tmp_path / "b", sidecar.episode_id)
    assert sidecar_a.content_checksum == sidecar_b.content_checksum


def test_checksum_covers_identity_fields() -> None:
    arrays = make_arrays()
    base = episode_content_checksum(arrays, identity_for("base_00000"))
    other = episode_content_checksum(arrays, identity_for("base_00001"))
    assert base != other


def test_overwrite_refused(tmp_path) -> None:
    arrays = make_arrays()
    sidecar = make_sidecar(arrays)
    write_episode(tmp_path, arrays, sidecar)
    with pytest.raises(ManifestError, match="already exists"):
        write_episode(tmp_path, arrays, sidecar)


def test_null_sentinel_discipline_enforced() -> None:
    arrays = make_arrays()
    # Oracle answer present where oracle_called is False must be rejected.
    bad_called = arrays.oracle_called.copy()
    bad_called[2] = False
    import dataclasses

    with pytest.raises(EpisodeSchemaError, match="NULL_ACTION exactly where"):
        dataclasses.replace(arrays, oracle_called=bad_called)
    # previous_executed_action must be NULL exactly at t=0.
    bad_previous = arrays.previous_executed_action.copy()
    bad_previous[0] = 2
    with pytest.raises(EpisodeSchemaError, match="exactly at t=0"):
        dataclasses.replace(arrays, previous_executed_action=bad_previous)


def test_revealed_target_requires_oracle_call() -> None:
    import dataclasses

    arrays = make_arrays()
    bad_called = np.zeros(arrays.length, dtype=np.bool_)
    bad_recommended = np.full(arrays.length, NULL_ACTION, dtype=np.int8)
    with pytest.raises(EpisodeSchemaError, match="revealed without an oracle call"):
        dataclasses.replace(
            arrays, oracle_called=bad_called, oracle_recommended_action=bad_recommended
        )


def test_wrong_dtype_rejected() -> None:
    import dataclasses

    arrays = make_arrays()
    with pytest.raises(EpisodeSchemaError, match="dtype"):
        dataclasses.replace(arrays, reward=arrays.reward.astype(np.float64))
