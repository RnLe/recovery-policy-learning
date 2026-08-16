"""CPU end-to-end: freeze -> frozen bundles -> one opening -> frozen analysis.

Connectivity and semantics at miniature learning scale (the preflight inside
runs at its real >=500-episode scale, because the gate is not weakenable).
This test protects the lifecycle boundaries: freeze refuses placeholders and
double-freezing, final commands accept only the frozen contract, exactly one
confirmatory opening exists, and the analysis is exploratory-labelled below
the planned replication.
"""

from __future__ import annotations

import pytest

from grounded_recovery.config import contract_hash, load_and_validate
from grounded_recovery.data import ManifestError, make_manifests
from grounded_recovery.experiment import (
    evaluate_final,
    run_bundle_frozen,
    run_freeze,
    run_preflight,
)
from grounded_recovery.publish import analyze_results

pytestmark = pytest.mark.slow

REPO_ROOT_OVERRIDES = {
    "data.n0": 20,
    "data.b": 6,
    "data.k": 2,
    "data.h": 2,
    "perturbation.collection_time_set": [1, 2, 3],
    "perturbation.unseen_time_set": [2, 3],
    "training.base_updates": 60,
    "training.base_targets_per_update": 8,
    "training.new_targets_per_update": 4,
    "training.updates_per_round": 15,
    "data.split_counts.collection": 14,
    "data.split_counts.test_candidate": 6,
    "data.split_counts.operator_preflight": 500,
    "perturbation.preflight_episodes_per_family": 500,
    "evaluation.desired_interval_half_width": 0.05,
    "evaluation.r_target": 5,
    "evaluation.r_max": 5,
    "seeds.bundle_ids": ["B00", "B01"],
}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, tiny_config_factory):
    pilot_path = tiny_config_factory(**REPO_ROOT_OVERRIDES)
    cfg = load_and_validate(pilot_path)
    manifest_root = tmp_path_factory.mktemp("manifests")
    data_root = tmp_path_factory.mktemp("data")
    results_root = tmp_path_factory.mktemp("results")
    repo_root = pilot_path.parents[0]
    (repo_root / "src" / "grounded_recovery").mkdir(parents=True)
    (repo_root / "src" / "grounded_recovery" / "stub.py").write_text("# fixture\n")
    (repo_root / "pyproject.toml").write_text("# fixture\n")
    (repo_root / "uv.lock").write_text("# fixture\n")

    make_manifests(cfg, manifest_root)
    run_preflight(cfg, manifest_root, data_root / "preflight" / "x")
    contract_path = repo_root / "experiment_contract.yaml"
    record = run_freeze(pilot_path, contract_path, manifest_root, data_root, repo_root)
    return (cfg, pilot_path, contract_path, manifest_root, data_root, results_root,
            record)


def test_freeze_record_and_refusals(pipeline, tiny_config_factory, tmp_path) -> None:
    cfg, pilot_path, contract_path, manifest_root, data_root, _rroot, record = pipeline
    frozen = load_and_validate(contract_path)
    assert frozen.study.status == "FROZEN"
    assert record["contract_hash"] == contract_hash(frozen)
    assert record["r_train"] == 5
    assert 0 < record["eligible"]["count"] <= record["eligible"]["candidates"]
    # Double freeze is refused.
    with pytest.raises(ManifestError, match="already exists"):
        run_freeze(pilot_path, contract_path, manifest_root, data_root,
                   contract_path.parent)
    # Freezing an unresolved pilot is refused.
    unresolved = tiny_config_factory(**{"data.h": "PILOT_TO_FREEZE"})
    with pytest.raises(ManifestError, match="unresolved"):
        run_freeze(unresolved, tmp_path / "c.yaml", manifest_root, data_root,
                   contract_path.parent)


def test_freeze_requires_passing_preflight(pipeline, tmp_path, tiny_config_factory) -> None:
    # A tiny preflight (below the 500-episode scale) must block freezing.
    overrides = dict(REPO_ROOT_OVERRIDES)
    overrides["data.split_counts.operator_preflight"] = 4
    overrides["perturbation.preflight_episodes_per_family"] = 4
    pilot_path = tiny_config_factory(**overrides)
    cfg = load_and_validate(pilot_path)
    manifest_root = tmp_path / "manifests"
    data_root = tmp_path / "data"
    make_manifests(cfg, manifest_root)
    run_preflight(cfg, manifest_root, data_root / "preflight" / "x")
    with pytest.raises(ManifestError, match="preflight"):
        run_freeze(pilot_path, tmp_path / "contract.yaml", manifest_root, data_root,
                   tmp_path)


def test_frozen_pipeline_single_opening_and_analysis(pipeline) -> None:
    cfg, pilot_path, contract_path, manifest_root, data_root, results_root, record = (
        pipeline
    )
    # run-bundle accepts only the frozen contract.
    with pytest.raises(ManifestError, match="frozen"):
        run_bundle_frozen(pilot_path, "B00", manifest_root, data_root)
    for bundle_id in ("B00", "B01"):
        run_bundle_frozen(contract_path, bundle_id, manifest_root, data_root)

    outcome = evaluate_final(contract_path, manifest_root, data_root, results_root)
    eligible = record["eligible"]["count"]
    assert outcome["cells"] == 2 * 3 * 3
    assert outcome["rows"] == 2 * 3 * 3 * eligible

    # Exactly one opening per contract.
    with pytest.raises(ManifestError, match="one opening"):
        evaluate_final(contract_path, manifest_root, data_root, results_root)

    frozen = load_and_validate(contract_path)
    results_dir = results_root / contract_hash(frozen)[:12]
    assert (results_dir / "opening_receipt.json").exists()
    assert (results_dir / "opening_complete.json").exists()

    summary = analyze_results(frozen, results_dir, planned_r_train=record["r_train"])
    # Two completed bundles below the planned five: never confirmatory.
    assert summary["analysis_status"] == "exploratory_pilot"
    assert summary["bundles_completed"] == 2
    assert summary["claim_state"] in ("support", "adverse", "rule_out", "inconclusive")
    assert set(summary["per_bundle_deltas"]) == {"B00", "B01"}
    assert summary["scenario_denominator"] == eligible
    assert (results_dir / "figures" / "primary_paired_effect.png").exists()
    assert (results_dir / "figures" / "success_matrix.png").exists()
    assert (results_dir / "tables" / "pipeline_metrics.csv").exists()
    assert (results_dir / "statistical_summary.json").exists()


def test_integrity_phases_after_release(pipeline) -> None:
    from grounded_recovery.integrity import run_integrity

    cfg, _pilot, contract_path, manifest_root, data_root, results_root, _record = (
        pipeline
    )
    for phase in ("freeze", "preopen", "release"):
        report = run_integrity(contract_path, phase, manifest_root, data_root,
                               results_root)
        failed = [c for c in report["checks"] if not c["passed"]]
        assert report["passed"], f"phase {phase} failed: {failed}"
    # A tampered summary must fail the release recompute check.
    import json

    frozen = load_and_validate(contract_path)
    results_dir = results_root / contract_hash(frozen)[:12]
    summary_path = results_dir / "statistical_summary.json"
    original = summary_path.read_text()
    tampered = json.loads(original)
    first = next(iter(tampered["per_bundle_deltas"]))
    tampered["per_bundle_deltas"][first] += 0.25
    summary_path.write_text(json.dumps(tampered))
    try:
        report = run_integrity(contract_path, "release", manifest_root, data_root,
                               results_root)
        assert not report["passed"]
        assert any(
            c["check"] == "summary_recomputes" and not c["passed"]
            for c in report["checks"]
        )
    finally:
        summary_path.write_text(original)


def test_publish_result_both_modes(pipeline, tmp_path) -> None:
    from grounded_recovery.artifacts import read_json
    from grounded_recovery.publish import publish_result

    cfg, _pilot, contract_path, _mroot, _droot, results_root, record = pipeline
    frozen = load_and_validate(contract_path)
    results_dir = results_root / contract_hash(frozen)[:12]

    protocol_dir = tmp_path / "public_protocol"
    publish_result(frozen, protocol_dir, results_dir=None, freeze_record=record,
                   protocol_only=True)
    status = read_json(protocol_dir / "site-status.json")
    assert status["phase"] == "protocol"
    assert status["result_release"] is False
    assert not (protocol_dir / "experiment-summary.json").exists()
    manifest = read_json(protocol_dir / "artifact-manifest.json")
    assert all("sha256" in f for f in manifest["files"])

    results_out = tmp_path / "public_results"
    publish_result(frozen, results_out, results_dir=results_dir, freeze_record=record,
                   protocol_only=False)
    status = read_json(results_out / "site-status.json")
    assert status["phase"] == "results" and status["result_release"] is True
    summary = read_json(results_out / "experiment-summary.json")
    assert summary["primary_summary"]["analysis_status"] == "exploratory_pilot"
    assert summary["slices"][2] == {
        "id": "unseen", "label": "Unseen perturbation", "role": "primary"
    }
    assert len(summary["replicates"]) == 2
    for replicate in summary["replicates"]:
        assert len(replicate["outcomes"]) == 9
        clean_cells = [o for o in replicate["outcomes"] if o["slice"] == "clean"]
        assert all(o["intervention_delivered"] is None for o in clean_cells)
    assert (results_out / "figures" / "primary_paired_effect.png").exists()
    # A tampered summary must be caught before anything is published.
    import json as json_module
    import shutil

    tampered_results = tmp_path / "tampered_results"
    shutil.copytree(results_dir, tampered_results)
    summary_path = tampered_results / "statistical_summary.json"
    data = json_module.loads(summary_path.read_text())
    first = next(iter(data["per_bundle_deltas"]))
    data["per_bundle_deltas"][first] += 0.5
    summary_path.write_text(json_module.dumps(data))
    with pytest.raises(ValueError, match="recompute"):
        publish_result(frozen, tmp_path / "public_bad", results_dir=tampered_results,
                       freeze_record=record, protocol_only=False)
