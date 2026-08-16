"""Command-line surface: exit codes and artifact creation."""

from __future__ import annotations

import json
from pathlib import Path

from grounded_recovery.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_YAML = REPO_ROOT / "configs" / "pilot.yaml"


def test_smoke_command_writes_fingerprint(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["smoke", "--config", str(PILOT_YAML)]) == 0
    fingerprint = json.loads((tmp_path / "environment_fingerprint.json").read_text())
    assert fingerprint["environment"]["env_id"] == "BabyAI-GoToObjMazeS4-v0"
    assert fingerprint["environment"]["doors_open"] is True
    assert len(fingerprint["smoke_episodes"]) == 3
    assert fingerprint["contract_hash"]
    # The stage is recorded so the pilot-stage digest is never mistaken for the
    # frozen protocol's.
    assert fingerprint["stage"] == "pre-freeze smoke"
    assert fingerprint["config"].endswith("pilot.yaml")


def test_invalid_config_exits_nonzero(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("study: {}\n")
    assert main(["smoke", "--config", str(bad)]) == 1


def test_missing_config_exits_nonzero(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["smoke", "--config", str(tmp_path / "absent.yaml")]) == 1


def test_make_manifests_and_preflight_commands(tmp_path, monkeypatch, make_tiny_config) -> None:
    config_path = make_tiny_config()
    monkeypatch.chdir(tmp_path)
    assert main(["make-manifests", "--config", str(config_path)]) == 0
    assert (tmp_path / "manifests" / "disjointness_report.json").exists()
    # Rerunning against existing manifests must refuse.
    assert main(["make-manifests", "--config", str(config_path)]) == 1
    # Tiny scale runs the machinery but cannot pass the >=500-episode gate.
    assert main(["preflight", "--config", str(config_path)]) == 1
    reports = list((tmp_path / "data" / "preflight").glob("*/preflight_report.json"))
    assert len(reports) == 1
