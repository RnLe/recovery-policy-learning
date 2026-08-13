"""Base collection: exact N0 budget, deterministic boundary, honest storage.

These tests protect the budget-matched comparison's foundation: `D0` must
contain exactly N0 revealed targets, be a pure function of the contract, and
never contain fabricated labels.
"""

from __future__ import annotations

import shutil

import pytest

from grounded_recovery.artifacts import read_json, read_jsonl
from grounded_recovery.config import load_and_validate
from grounded_recovery.data import (
    ManifestError,
    base_dataset_dir,
    collect_base,
    load_split_manifest,
    make_manifests,
    read_episode,
)
from grounded_recovery.integrity import (
    IntegrityError,
    recount_dataset,
    verify_dataset,
    verify_episode_replay,
)

TINY_N0 = 25  # forces a mid-episode boundary for typical path lengths


@pytest.fixture(scope="module")
def collected(tmp_path_factory, tiny_config_factory):
    config_path = tiny_config_factory(**{"data.n0": TINY_N0})
    cfg = load_and_validate(config_path)
    manifest_root = tmp_path_factory.mktemp("manifests")
    make_manifests(cfg, manifest_root)
    data_root = tmp_path_factory.mktemp("data")
    summary = collect_base(cfg, "B00", manifest_root, data_root)
    return cfg, manifest_root, data_root, summary


def test_exactly_n0_targets(collected) -> None:
    cfg, _root, data_root, summary = collected
    assert summary.n0 == TINY_N0
    recount = recount_dataset(base_dataset_dir(cfg, "B00", data_root))
    assert recount["targets"] == TINY_N0
    assert recount["episodes"] == summary.episodes


def test_partial_final_episode_rule(collected) -> None:
    cfg, _root, data_root, summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    index = read_jsonl(dataset_dir / "episode_index.jsonl")
    sidecars = [
        read_episode(dataset_dir / "episodes", row["episode_id"])[1] for row in index
    ]
    # Only the final episode may be budget-truncated, and only when the
    # boundary fell inside it.
    for sidecar in sidecars[:-1]:
        assert sidecar.termination_reason in ("terminated", "truncated")
        assert not sidecar.stopped_early
    final = sidecars[-1]
    assert final.stopped_early == (final.termination_reason == "budget_truncated")
    assert final.stopped_early == summary.final_episode_budget_truncated
    # In base collection every active step is a revealed target.
    for sidecar in sidecars:
        assert sidecar.revealed_targets == sidecar.executed_length


def test_manifest_order_respected(collected) -> None:
    cfg, manifest_root, data_root, _summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    index = read_jsonl(dataset_dir / "episode_index.jsonl")
    entries, _ = load_split_manifest(manifest_root, "base")
    for row, entry in zip(index, entries[: len(index)], strict=True):
        assert row["environment_seed"] == entry.environment_seed
        assert row["canonical_scenario_hash"] == entry.canonical_scenario_hash


def test_collect_deterministic(collected, tmp_path_factory) -> None:
    cfg, manifest_root, data_root, summary = collected
    rerun_root = tmp_path_factory.mktemp("data-rerun")
    rerun = collect_base(cfg, "B00", manifest_root, rerun_root)
    assert rerun.ledger_final_hash == summary.ledger_final_hash
    assert rerun.episodes == summary.episodes
    assert rerun.steps == summary.steps


def test_refuses_existing_dataset(collected) -> None:
    cfg, manifest_root, data_root, _summary = collected
    with pytest.raises(ManifestError, match="immutable"):
        collect_base(cfg, "B00", manifest_root, data_root)


def test_refuses_unresolved_n0(collected, tiny_config_factory) -> None:
    cfg, manifest_root, data_root, _summary = collected
    unresolved = load_and_validate(
        tiny_config_factory(**{"data.n0": "PILOT_TO_FREEZE"})
    )
    with pytest.raises(ManifestError, match="PILOT_TO_FREEZE"):
        collect_base(unresolved, "B00", manifest_root, data_root)


def test_refuses_undeclared_bundle(collected) -> None:
    cfg, manifest_root, data_root, _summary = collected
    with pytest.raises(ManifestError, match="bundle"):
        collect_base(cfg, "B99", manifest_root, data_root)


def test_stored_episodes_replay_bitexact(collected) -> None:
    cfg, _root, data_root, _summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    result = verify_dataset(cfg, dataset_dir, replay_sample=3)
    assert result["targets"] == TINY_N0
    assert len(result["replayed_episodes"]) >= 1


def test_replay_detects_planted_action_edit(collected, tmp_path) -> None:
    import numpy as np

    from grounded_recovery.world import WorldSession

    cfg, _root, data_root, _summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    copy_dir = tmp_path / "tampered"
    shutil.copytree(dataset_dir, copy_dir)
    index = read_jsonl(copy_dir / "episode_index.jsonl")
    episode_id = index[0]["episode_id"]
    npz_path = copy_dir / "episodes" / f"{episode_id}.npz"
    arrays = dict(np.load(npz_path))
    edited = arrays["executed_action"].copy()
    edited[0] = (edited[0] + 1) % 3
    arrays["executed_action"] = edited
    np.savez_compressed(npz_path, **arrays)
    # The checksum catches the edit first; replay would catch it independently
    # if the checksum were also recomputed by the attacker.
    with pytest.raises((IntegrityError, ManifestError)):
        session = WorldSession(cfg.environment)
        try:
            verify_episode_replay(session, copy_dir / "episodes", episode_id)
        finally:
            session.close()


def test_recount_detects_missing_episode(collected, tmp_path) -> None:
    cfg, _root, data_root, _summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    copy_dir = tmp_path / "missing"
    shutil.copytree(dataset_dir, copy_dir)
    index = read_jsonl(copy_dir / "episode_index.jsonl")
    victim = index[-1]["episode_id"]
    (copy_dir / "episodes" / f"{victim}.npz").unlink()
    (copy_dir / "episodes" / f"{victim}.json").unlink()
    with pytest.raises(IntegrityError, match="missing"):
        recount_dataset(copy_dir)


def test_dataset_meta_stamps(collected) -> None:
    from grounded_recovery.config import contract_hash

    cfg, manifest_root, data_root, _summary = collected
    dataset_dir = base_dataset_dir(cfg, "B00", data_root)
    meta = read_json(dataset_dir / "dataset_meta.json")
    assert meta["contract_hash"] == contract_hash(cfg)
    assert meta["n0"] == TINY_N0
    _, manifest_hash = load_split_manifest(manifest_root, "base")
    assert meta["manifest_hash"] == manifest_hash
