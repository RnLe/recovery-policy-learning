"""Manifest generation, disjointness, and immutability (gate G2 machinery).

These tests protect leakage resistance: the eight purpose splits must be
deterministic functions of the contract, disjoint by seed and by canonical
scenario hash, and immutable once written.
"""

from __future__ import annotations

import dataclasses

import pytest

from grounded_recovery.config import SPLIT_NAMES, load_and_validate
from grounded_recovery.data import (
    ManifestError,
    audit_disjointness,
    load_split_manifest,
    make_manifests,
    probe_scenario,
    verify_manifest_contract,
)
from grounded_recovery.world import WorldSession


@pytest.fixture(scope="module")
def built(tmp_path_factory, tiny_config_factory):
    # Build one tiny manifest set shared by the read-only tests in this module.
    cfg = load_and_validate(tiny_config_factory())
    manifest_root = tmp_path_factory.mktemp("manifests")
    hashes = make_manifests(cfg, manifest_root)
    return cfg, manifest_root, hashes


def test_all_eight_splits_written(built) -> None:
    cfg, manifest_root, hashes = built
    assert set(hashes) == set(SPLIT_NAMES)
    for split in SPLIT_NAMES:
        entries, digest = load_split_manifest(manifest_root, split)
        assert digest == hashes[split]
        assert len(entries) == cfg.data.split_counts.for_split(split)
        verify_manifest_contract(manifest_root, split, cfg)


def test_build_deterministic(built, tmp_path) -> None:
    cfg, _root, hashes = built
    rebuilt = make_manifests(cfg, tmp_path / "again")
    assert rebuilt == hashes


def test_admissibility_and_schedule_rules(built) -> None:
    cfg, manifest_root, _hashes = built
    entries, _ = load_split_manifest(manifest_root, "operator_preflight")
    session = WorldSession(cfg.environment)
    try:
        for entry in entries:
            assert entry.nominal_oracle_path_length >= 1
            assert len(entry.scheduled_intervention_times) == 1
            scheduled = entry.scheduled_intervention_times[0]
            # The schedule rule guarantees deliverability.
            assert cfg.perturbation.preflight_time_min <= scheduled
            assert scheduled < entry.nominal_oracle_path_length
            assert scheduled <= cfg.perturbation.preflight_time_max
        # Independently re-probe two entries: the manifest must describe the
        # world the environment actually generates.
        for entry in entries[:2]:
            probe = probe_scenario(session, entry.environment_seed)
            assert probe.admissible
            assert probe.scenario_hash == entry.canonical_scenario_hash
            assert probe.path_length == entry.nominal_oracle_path_length
    finally:
        session.close()


def test_other_splits_have_empty_schedules_at_version_one(built) -> None:
    _cfg, manifest_root, _hashes = built
    for split in SPLIT_NAMES:
        if split == "operator_preflight":
            continue
        entries, _ = load_split_manifest(manifest_root, split)
        assert all(entry.scheduled_intervention_times == () for entry in entries)
        assert all(entry.manifest_version == 1 for entry in entries)


def test_refuse_overwrite_existing_manifests(built) -> None:
    cfg, manifest_root, _hashes = built
    with pytest.raises(ManifestError, match="immutable"):
        make_manifests(cfg, manifest_root)


def test_disjointness_audit_catches_planted_overlap(built) -> None:
    _cfg, manifest_root, _hashes = built
    base_entries, _ = load_split_manifest(manifest_root, "base")
    collection_entries, _ = load_split_manifest(manifest_root, "collection")
    # Plant the same scenario in both splits.
    stolen = dataclasses.replace(base_entries[0], split_name="collection", ordinal=999)
    with pytest.raises(ManifestError, match="disjointness"):
        audit_disjointness(
            {"base": base_entries, "collection": collection_entries + [stolen]}
        )


def test_load_detects_tampered_entries(built, tmp_path) -> None:
    import shutil

    _cfg, manifest_root, _hashes = built
    copy_root = tmp_path / "tampered"
    shutil.copytree(manifest_root, copy_root)
    entries_path = copy_root / "base" / "entries.jsonl"
    content = entries_path.read_text()
    entries_path.write_text(content.replace('"ordinal":0', '"ordinal":7', 1))
    with pytest.raises(ManifestError, match="hash"):
        load_split_manifest(copy_root, "base")


def test_rejects_recorded_with_reasons(built) -> None:
    from grounded_recovery.artifacts import read_json, read_jsonl

    _cfg, manifest_root, _hashes = built
    for split in SPLIT_NAMES:
        meta = read_json(manifest_root / split / "manifest_meta.json")
        rejects = read_jsonl(manifest_root / split / "rejected_probes.jsonl")
        assert meta["rejected_candidates"] == len(rejects)
        assert all(row["reason"] for row in rejects)


def test_contract_mismatch_detected(built, make_tiny_config) -> None:
    _cfg, manifest_root, _hashes = built
    other = load_and_validate(make_tiny_config(**{"seeds.root_seed": 999}))
    with pytest.raises(ManifestError, match="scenario identity"):
        verify_manifest_contract(manifest_root, "base", other)


def test_training_change_does_not_invalidate_manifests(built, make_tiny_config) -> None:
    # Manifest validity is keyed to scenario identity, not to the full
    # contract: tuning a learning setting must not force regeneration.
    _cfg, manifest_root, _hashes = built
    tuned = load_and_validate(make_tiny_config(**{"training.learning_rate": 5.0e-4}))
    verify_manifest_contract(manifest_root, "base", tuned)
