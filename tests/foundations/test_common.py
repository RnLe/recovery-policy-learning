"""Foundations plumbing: seed discipline, output layout, honest labelling."""

from __future__ import annotations

import json

import pytest

from gr_foundations.common import (
    EXPLORATORY_LABEL,
    SEED_COMPONENTS,
    FoundationsError,
    LabPaths,
    derive_seed,
    export_typst_table,
    export_typst_values,
    prepare,
    write_metrics,
    write_mini_report,
)


def test_derive_seed_is_stable_and_bounded() -> None:
    first = derive_seed("lab01.census")
    assert first == derive_seed("lab01.census")
    assert 0 <= first < 2**63
    # Distinct components and indices give distinct streams.
    seeds = {derive_seed(c) for c in SEED_COMPONENTS}
    assert len(seeds) == len(SEED_COMPONENTS)
    assert derive_seed("lab01.census", 1) != first


def test_derive_seed_rejects_unknown_component_and_negative_index() -> None:
    with pytest.raises(FoundationsError, match="unknown"):
        derive_seed("lab99.made_up")
    with pytest.raises(FoundationsError, match="non-negative"):
        derive_seed("lab01.census", -1)


def test_seed_domain_disjoint_from_study() -> None:
    """Even an identical component name cannot collide with a study stream."""
    from grounded_recovery.seeds import SEED_COMPONENTS as STUDY_COMPONENTS
    from grounded_recovery.seeds import derive_seed as study_derive

    study_component = sorted(STUDY_COMPONENTS)[0]
    foundations_component = sorted(SEED_COMPONENTS)[0]
    assert derive_seed(foundations_component) != study_derive(
        20260826, "B00", study_component
    )


def test_lab_paths_layout(tmp_path) -> None:
    paths = LabPaths(lab_id="lab01", repo_root=tmp_path)
    assert paths.data_dir == tmp_path / "data" / "foundations" / "lab01"
    assert paths.out_dir == tmp_path / "foundations" / "lab01"
    assert paths.report_dir == tmp_path / "report" / "generated" / "foundations" / "lab01"
    with pytest.raises(FoundationsError, match="unknown lab"):
        LabPaths(lab_id="lab99", repo_root=tmp_path)


def test_prepare_refuses_then_replaces(tmp_path) -> None:
    paths = LabPaths(lab_id="lab02", repo_root=tmp_path)
    prepare(paths, force=False)
    marker = paths.out_dir / "stale.txt"
    marker.write_text("old run", encoding="utf-8")
    with pytest.raises(FoundationsError, match="--force"):
        prepare(paths, force=False)
    prepare(paths, force=True)
    assert not marker.exists()
    assert paths.figures_dir.is_dir() and paths.tables_dir.is_dir()


def test_metrics_document_is_labelled_and_deterministic(tmp_path) -> None:
    paths = LabPaths(lab_id="lab03", repo_root=tmp_path)
    prepare(paths, force=False)
    digest = write_metrics(paths, {"answer": 42})
    document = json.loads((paths.out_dir / "metrics.json").read_text())
    assert document["evidence_label"] == EXPLORATORY_LABEL
    assert document["lab"] == "lab03"
    assert document["metrics"] == {"answer": 42}
    prepare(paths, force=True)
    assert write_metrics(paths, {"answer": 42}) == digest


def test_typst_fragments_carry_generated_header(tmp_path) -> None:
    paths = LabPaths(lab_id="lab04", repo_root=tmp_path)
    prepare(paths, force=False)
    values_path = export_typst_values(paths, "vals", {"n-episodes": "300"})
    table_path = export_typst_table(paths, "tab", ["name", "value"], [["a", 1]])
    for path in (values_path, table_path):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("// GENERATED. DO NOT EDIT.")
    assert '#let n-episodes = "300"' in values_path.read_text(encoding="utf-8")
    with pytest.raises(FoundationsError, match="row width"):
        export_typst_table(paths, "bad", ["only"], [["a", "b"]])


def test_mini_report_states_status_and_reproduction(tmp_path) -> None:
    paths = LabPaths(lab_id="lab05", repo_root=tmp_path)
    prepare(paths, force=False)
    report = write_mini_report(
        paths, question="What?", sections=[("Findings", "Text body.")]
    )
    text = report.read_text(encoding="utf-8")
    assert EXPLORATORY_LABEL in text
    assert "grf run lab05 --force" in text
    assert "## Findings" in text
