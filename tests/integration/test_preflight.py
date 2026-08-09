"""Preflight runner semantics (gate G1 machinery, tiny counts).

These tests protect the treatment definition and ITT bookkeeping rehearsal:
one forced corruption exactly at the scheduled time, synchronized recovery
afterwards, undelivered assignments kept in the denominator, and a gate that
cannot pass at miniature scale.
"""

from __future__ import annotations

import shutil

import pytest

from grounded_recovery.artifacts import ImmutableArtifactError, read_json, read_jsonl
from grounded_recovery.config import load_and_validate
from grounded_recovery.data import (
    ManifestError,
    load_split_manifest,
    make_manifests,
    write_split_manifest,
)
from grounded_recovery.experiment import run_preflight


@pytest.fixture(scope="module")
def preflight_run(tmp_path_factory, tiny_config_factory):
    cfg = load_and_validate(tiny_config_factory())
    manifest_root = tmp_path_factory.mktemp("manifests")
    make_manifests(cfg, manifest_root)
    out_dir = tmp_path_factory.mktemp("preflight-out")
    report = run_preflight(cfg, manifest_root, out_dir)
    rows = read_jsonl(out_dir / "preflight_episodes.jsonl")
    return cfg, manifest_root, out_dir, report, rows


def test_row_counts_and_families(preflight_run) -> None:
    cfg, _root, _out, report, rows = preflight_run
    per_family = cfg.data.split_counts.operator_preflight
    assert len(rows) == 2 * per_family
    assert {row["family"] for row in rows} == {"collection", "unseen"}
    assert report["episodes_written"] == len(rows)


def test_forced_action_applied_exactly_at_scheduled_time(preflight_run) -> None:
    cfg, _root, _out, _report, rows = preflight_run
    operators = {
        "collection": cfg.perturbation.collection_operator.mapping,
        "unseen": cfg.perturbation.unseen_operator.mapping,
    }
    for row in rows:
        assert row["delivered"] is True  # schedule rule guarantees deliverability
        mapping = operators[row["family"]]
        assert row["forced_action"] == mapping[row["recommended_at_scheduled_time"]]
        assert row["forced_action"] != row["recommended_at_scheduled_time"]


def test_report_rate_recomputable_from_rows(preflight_run) -> None:
    _cfg, _root, _out, report, rows = preflight_run
    for family in ("collection", "unseen"):
        family_rows = [row for row in rows if row["family"] == family]
        delivered = [row for row in family_rows if row["delivered"]]
        recovered = sum(1 for row in delivered if row["success"])
        summary = report["families"][family]
        assert summary["episodes"] == len(family_rows)
        assert summary["delivered"] == len(delivered)
        assert summary["recovered_successes"] == recovered
        expected_rate = recovered / len(delivered) if delivered else 0.0
        assert summary["recovery_rate_delivered"] == pytest.approx(expected_rate)


def test_tiny_scale_cannot_pass_the_gate(preflight_run) -> None:
    # The >=500-episode requirement is part of the gate itself: a miniature
    # run must never be able to claim G1 evidence.
    _cfg, _root, _out, report, _rows = preflight_run
    for family in ("collection", "unseen"):
        assert report["families"][family]["episode_scale_ok"] is False
        assert report["families"][family]["passed"] is False
    assert report["passed"] is False


def test_outputs_immutable(preflight_run) -> None:
    cfg, manifest_root, out_dir, _report, _rows = preflight_run
    with pytest.raises(ImmutableArtifactError):
        run_preflight(cfg, manifest_root, out_dir)


def test_disjointness_included_in_report(preflight_run) -> None:
    _cfg, _root, out_dir, report, _rows = preflight_run
    assert report["disjointness"]["disjoint"] is True
    stored = read_json(out_dir / "preflight_report.json")
    assert stored["disjointness"]["disjoint"] is True
    assert stored["contract_hash"] == report["contract_hash"]


def test_undelivered_intervention_recorded_not_dropped(
    preflight_run, tmp_path, tiny_config_factory
) -> None:
    # Plant one schedule beyond the horizon: the episode must stay in the
    # denominator with delivered=False (intention-to-treat bookkeeping).
    import dataclasses

    cfg, manifest_root, _out, _report, _rows = preflight_run
    copy_root = tmp_path / "manifests"
    shutil.copytree(manifest_root, copy_root)
    entries, _ = load_split_manifest(copy_root, "operator_preflight")
    patched = [
        dataclasses.replace(entries[0], scheduled_intervention_times=(99,))
    ] + entries[1:]
    shutil.rmtree(copy_root / "operator_preflight")
    write_split_manifest(copy_root, cfg, "operator_preflight", patched, [])
    out_dir = tmp_path / "out"
    report = run_preflight(cfg, copy_root, out_dir)
    rows = read_jsonl(out_dir / "preflight_episodes.jsonl")
    undelivered = [row for row in rows if not row["delivered"]]
    assert len(undelivered) == 2  # one per family
    assert all(row["success"] for row in undelivered)  # nominal path, oracle succeeds
    for family in ("collection", "unseen"):
        summary = report["families"][family]
        assert summary["undelivered"] == 1
        assert summary["episodes"] == cfg.data.split_counts.operator_preflight
        # Undelivered episodes are excluded from the recovery-rate denominator
        # but never from the episode count.
        assert summary["delivered"] == summary["episodes"] - 1


def test_contract_mismatch_refused(preflight_run, tmp_path, tiny_config_factory) -> None:
    _cfg, manifest_root, _out, _report, _rows = preflight_run
    other = load_and_validate(tiny_config_factory(**{"seeds.root_seed": 424242}))
    with pytest.raises(ManifestError, match="scenario identity"):
        run_preflight(other, manifest_root, tmp_path / "out2")
