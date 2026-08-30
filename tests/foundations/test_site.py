"""Site staging: fidelity to sources, hash manifest, ownership boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from gr_foundations import site
from gr_foundations.common import FoundationsError
from grounded_recovery.artifacts import atomic_write_json, file_sha256
from grounded_recovery.media import write_mp4

REPO = Path.cwd()


def _make_fixture_repo(tmp_path: Path) -> Path:
    """A miniature repo with real study/foundations artifacts and fake media."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "public_result", root / "public_result")
    for index in range(1, 8):  # journey-data reads every lab's metrics
        source = REPO / "foundations" / f"lab0{index}"
        shutil.copytree(source, root / "foundations" / f"lab0{index}")
    (root / "build").mkdir(parents=True)
    (root / "build" / "recovery-policy-learning-report.pdf").write_bytes(b"%PDF-1.7 test")

    media = root / "foundations" / "media"
    (media / "posters").mkdir(parents=True)
    (media / "trajectories").mkdir()
    (media / "network").mkdir()
    frames = [Image.new("RGB", (64, 48), (200, 100, 50))] * 3
    write_mp4(frames, media / "clip.mp4")
    frames[0].save(media / "posters" / "clip.webp", "WEBP")
    atomic_write_json(
        media / "trajectories" / "clip.json",
        {"schema_version": "1.1.0", "steps": []},
    )
    atomic_write_json(
        media / "network" / "full_r0.json",
        {"schema_version": "1.0.0", "kind": "network-flow", "steps": []},
    )
    atomic_write_json(
        media / "media_manifest.json",
        {
            "schema_version": "1.0.0",
            "fps": 5,
            "items": [
                {
                    "id": "clip",
                    "href": "media/clip.mp4",
                    "poster": "media/posters/clip.webp",
                    "sha256": file_sha256(media / "clip.mp4"),
                    "empirical": True,
                    "selection_rule": "fixture",
                    "trace": "media/trajectories/clip.json",
                }
            ],
            "trajectories": [
                {
                    "id": "clip",
                    "href": "media/trajectories/clip.json",
                    "sha256": file_sha256(media / "trajectories" / "clip.json"),
                }
            ],
        },
    )
    return root


@pytest.fixture(scope="module")
def staged_repo(tmp_path_factory):
    root = _make_fixture_repo(tmp_path_factory.mktemp("site"))
    # Author-owned content must survive staging untouched.
    illustrations = root / "site" / "public" / "illustrations"
    illustrations.mkdir(parents=True)
    (illustrations / "architecture.svg").write_text("<svg></svg>", encoding="utf-8")
    summary = site.stage(root, force=False)
    return root, summary


def test_staging_layout_and_manifest(staged_repo) -> None:
    root, summary = staged_repo
    public = root / "site" / "public"
    assert (public / "data" / "experiment-summary.json").exists()
    assert (public / "data" / "journey-data.json").exists()
    assert (public / "figures" / "lab06" / "shift_anatomy.svg").exists()
    assert (public / "figures" / "lab03" / "labelled_trajectory.svg").exists()
    # The study figures are not staged: those charts render natively from JSON.
    assert not (public / "figures" / "study").exists()
    assert (
        public / "media" / "study" / "posters" / "unseen_paired_contrast.webp"
    ).exists()
    assert (public / "media" / "journey" / "clip.mp4").exists()
    assert (public / "media" / "journey" / "posters" / "clip.webp").exists()
    assert (public / "media" / "journey" / "trajectories" / "clip.json").exists()
    assert (public / "media" / "journey" / "network" / "full_r0.json").exists()
    assert (public / "reports").is_dir()
    assert (public / "illustrations" / "architecture.svg").exists()
    assert summary["files"] == len(
        json.loads((public / "staging-manifest.json").read_text())["files"]
    )
    assert site.verify_staging(root) == summary["files"]


def test_journey_data_matches_sources(staged_repo) -> None:
    root, _summary = staged_repo
    journey = json.loads(
        (root / "site" / "public" / "data" / "journey-data.json").read_text()
    )
    lab04 = json.loads((root / "foundations" / "lab04" / "metrics.json").read_text())
    rows = lab04["metrics"]["behavior_cloning"]["results"]["recurrent"]
    expected = sum(r["holdout"]["success_rate"] for r in rows) / len(rows)
    assert journey["lab04"]["recurrent"]["unseen_success_mean"] == pytest.approx(expected)
    lab06 = json.loads((root / "foundations" / "lab06" / "metrics.json").read_text())
    assert journey["lab06"]["success_matrix"] == lab06["metrics"]["success_matrix"]
    # Media hrefs were rewritten to site-relative locations.
    assert journey["media"]["items"][0]["href"] == "media/journey/clip.mp4"
    assert journey["media"]["trajectories"][0]["href"] == (
        "media/journey/trajectories/clip.json"
    )


def test_staging_refuses_then_replaces(staged_repo) -> None:
    root, _summary = staged_repo
    with pytest.raises(FoundationsError, match="--force"):
        site.stage(root, force=False)
    again = site.stage(root, force=True)
    assert again["files"] > 0
    assert (root / "site" / "public" / "illustrations" / "architecture.svg").exists()


def test_staging_reports_missing_inputs(tmp_path) -> None:
    root = tmp_path / "empty"
    (root / "public_result").mkdir(parents=True)
    with pytest.raises(FoundationsError, match="staging inputs missing"):
        site.stage(root, force=False)


def test_verify_detects_drift(staged_repo, tmp_path) -> None:
    root, _summary = staged_repo
    target = root / "site" / "public" / "data" / "journey-data.json"
    original = target.read_text()
    try:
        target.write_text(original.replace("{", "{ ", 1), encoding="utf-8")
        with pytest.raises(FoundationsError, match="drifted"):
            site.verify_staging(root)
    finally:
        target.write_text(original, encoding="utf-8")
