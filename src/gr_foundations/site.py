"""Staging for the journey website.

`grf stage-site` is the only writer of ``site/public``'s evidence subtrees
(data/, figures/, media/, reports/): it copies the study's public result
bundle, extracts a whitelisted set of numbers from the foundations metrics
into ``journey-data.json``, generates poster frames for the study videos, and
records everything in ``staging-manifest.json`` with hashes and sources. No
number on the site is ever typed by hand. If it isn't derivable from these
sources, it doesn't render. The ``illustrations/`` subtree belongs to the
author and is never touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from gr_foundations.common import FoundationsError
from grounded_recovery.artifacts import atomic_write_json, file_sha256

SCHEMA_VERSION = "1.0.0"
OWNED_SUBTREES = ("data", "figures", "media", "reports")

STUDY_JSON = (
    "site-status.json",
    "experiment-summary.json",
    "claim-evidence.json",
    "artifact-manifest.json",
)
# Both wide study contrasts; the study figures themselves are not staged,
# the site draws those charts natively from the validated JSON.
STUDY_VIDEOS = ("unseen_paired_contrast", "unseen_recovery_failure")

# Only figures a page actually renders; the rest exist for the report alone.
LAB_FIGURES: dict[str, tuple[str, ...]] = {
    "lab01": ("observation_anatomy.svg", "action_effects.svg"),
    "lab02": ("aliasing_showcase.svg", "full_observability_contrast.svg"),
    "lab03": ("synchronization_experiment.svg", "labelled_trajectory.svg"),
    "lab04": ("qlearning_curve.svg", "bc_learning_curves.svg"),
    "lab06": ("shift_anatomy.svg",),
    "lab07": ("itt_bias.svg", "paired_power.svg", "unmatched_confound.svg"),
}

NETWORK_TRACES = ("full_r0.json",)


def _read_metrics(repo_root: Path, lab: str) -> dict:
    path = repo_root / "foundations" / lab / "metrics.json"
    if not path.exists():
        raise FoundationsError(f"{path} is missing; run `grf run {lab}` first")
    return json.loads(path.read_text())["metrics"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_journey_data(repo_root: Path, media_manifest: dict) -> dict:
    """The whitelisted per-lab numbers the journey pages are allowed to show."""
    m1 = _read_metrics(repo_root, "lab01")
    m2 = _read_metrics(repo_root, "lab02")
    m3 = _read_metrics(repo_root, "lab03")
    m4 = _read_metrics(repo_root, "lab04")
    m5 = _read_metrics(repo_root, "lab05")
    m6 = _read_metrics(repo_root, "lab06")
    m7 = _read_metrics(repo_root, "lab07")

    bc = m4["behavior_cloning"]

    def bc_summary(kind: str) -> dict:
        rows = bc["results"][kind]
        return {
            "open_loop_accuracy_mean": _mean([r["open_loop_accuracy"] for r in rows]),
            "unseen_success_mean": _mean([r["holdout"]["success_rate"] for r in rows]),
            "train_success_mean": _mean(
                [r["train_scenarios"]["success_rate"] for r in rows]
            ),
            "unseen_success_per_seed": [r["holdout"]["success_rate"] for r in rows],
        }

    ablation = m5["ablation"]["results"]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "exploratory foundations material; the confirmatory "
                          "evidence is the frozen study",
        "environment": {
            "env_id": m1["environment"]["env_id"],
            "max_steps": m1["environment"]["max_steps"],
            "action_names": m1["environment"]["frozen_action_names"],
            "observation_shape": m1["environment"]["observation_image_shape"],
        },
        "lab01": {
            "census_seeds": m1["census"]["n_seeds"],
            "unique_missions": m1["census"]["unique_missions"],
            "grid_shape": m1["census"]["grid_shapes"][0],
            "mission_colors": m1["census"]["mission_color_counts"],
            "mission_kinds": m1["census"]["mission_kind_counts"],
        },
        "lab02": {
            "episodes": m2["rollouts"]["episodes"],
            "total_states": m2["aliasing"]["total_states"],
            "observation_classes": m2["aliasing"]["observation_classes"],
            "aliased_classes": m2["aliasing"]["aliased_classes"],
            "cross_world_classes": m2["aliasing"]["cross_world_aliased_classes"],
            "conflicting_classes": m2["aliasing"]["label_heterogeneous_classes"],
            "memoryless_error_floor": m2["aliasing"]["memoryless_error_lower_bound"],
        },
        "lab03": {
            "spectrum": {
                name: {
                    "success_rate": stats["success_rate"],
                    "mean_steps": stats["mean_steps"],
                    "episodes": stats["episodes"],
                }
                for name, stats in m3["spectrum"].items()
            },
            "sync": {
                "pairs": m3["synchronization"]["pairs_delivered"],
                "success_rates": m3["synchronization"]["success_rates"],
            },
        },
        "lab04": {
            "qlearning": {
                "states": m4["qlearning"]["states_in_table"],
                "best_steps": m4["qlearning"]["best_greedy_steps"],
                "episodes": m4["qlearning"]["episodes"],
            },
            "dataset": {
                "episodes": bc["dataset_counters"]["collected"],
                "labels": bc["total_labels"],
            },
            "memoryless": bc_summary("memoryless"),
            "recurrent": bc_summary("recurrent"),
        },
        "lab05": {
            "parameters": m5["parameter_parity"]["recovery_policy"],
            "walkthrough": m5["walkthrough"],
            "variants": [
                {
                    "name": name,
                    "parameters": rows[0]["parameters"],
                    "unseen_success_mean": _mean(
                        [r["unseen"]["success_rate"] for r in rows]
                    ),
                    "unseen_success_per_seed": [
                        r["unseen"]["success_rate"] for r in rows
                    ],
                }
                for name, rows in ablation.items()
            ],
        },
        "lab06": {
            "design": m6["design"],
            "success_matrix": m6["success_matrix"],
            "per_rep_unseen_delta": m6["per_rep_unseen_delta"],
            "delivered_rate": m6["delivered_rate"],
            "exposure": m6["exposure_mean_arm_labels_per_update"],
            "sweep": m6["sweep"],
        },
        "lab07": {
            "simulation": m7["delivery_bias_simulation"],
            "pairing": m7["pairing_simulation"],
            "reanalysis": m7["lab06_reanalysis"],
            "unmatched": m7["unmatched_arm"],
            "hash_chain": m7["hash_chain_demo"],
            "contract_hash_demo": m7["contract_hash_demo"],
        },
        "media": media_manifest,
    }


def _poster_from_video(video: Path, poster: Path) -> None:
    import shutil as shutil_module

    if shutil_module.which("ffmpeg") is None:
        raise FoundationsError(
            "ffmpeg is required to extract poster frames during staging; "
            "install it (apt-get install ffmpeg, or brew install ffmpeg) "
            "and rerun `grf stage-site`"
        )
    poster.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-frames:v", "1", str(poster)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FoundationsError(f"poster extraction failed for {video}: {result.stderr}")


def stage(repo_root: Path, *, force: bool) -> dict[str, object]:
    public_result = repo_root / "public_result"
    foundations_media = repo_root / "foundations" / "media"
    report_pdf = repo_root / "build" / "recovery-policy-learning-report.pdf"
    site_public = repo_root / "site" / "public"

    missing = []
    for name in STUDY_JSON:
        if not (public_result / name).exists():
            missing.append(f"{public_result / name} (run `gr publish-result`)")
    if not (foundations_media / "media_manifest.json").exists():
        missing.append(f"{foundations_media} (run `grf media`)")
    for name in NETWORK_TRACES:
        if not (foundations_media / "network" / name).exists():
            missing.append(
                f"{foundations_media / 'network' / name} (run `grf network-trace`)"
            )
    if not report_pdf.exists():
        missing.append(f"{report_pdf} (compile the report with typst)")
    for lab, names in LAB_FIGURES.items():
        for name in names:
            path = repo_root / "foundations" / lab / "figures" / name
            if not path.exists():
                missing.append(f"{path} (run `grf run {lab}`)")
    if missing:
        raise FoundationsError("staging inputs missing:\n  " + "\n  ".join(missing))

    existing = [d for d in OWNED_SUBTREES if (site_public / d).exists()]
    if existing and not force:
        raise FoundationsError(
            f"site/public already contains {', '.join(existing)}; "
            "pass --force to restage"
        )
    for name in existing:
        shutil.rmtree(site_public / name)

    staged: list[dict[str, object]] = []

    def put(source: Path, relative: str, *, generated: bool = False) -> None:
        destination = site_public / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not generated:
            shutil.copyfile(source, destination)
        staged.append(
            {
                "path": relative,
                "sha256": file_sha256(destination),
                "source": str(source.relative_to(repo_root)) if not generated
                else "generated",
            }
        )

    for name in STUDY_JSON:
        put(public_result / name, f"data/{name}")
    for stem in STUDY_VIDEOS:
        video = public_result / "media" / f"{stem}.mp4"
        put(video, f"media/study/{stem}.mp4")
        poster = site_public / "media" / "study" / "posters" / f"{stem}.webp"
        _poster_from_video(video, poster)
        put(poster, f"media/study/posters/{stem}.webp", generated=True)

    for lab, names in LAB_FIGURES.items():
        for name in names:
            put(repo_root / "foundations" / lab / "figures" / name,
                f"figures/{lab}/{name}")

    journey_manifest = json.loads(
        (foundations_media / "media_manifest.json").read_text()
    )
    for item in journey_manifest["items"]:
        stem = item["id"]
        put(foundations_media / f"{stem}.mp4", f"media/journey/{stem}.mp4")
        put(foundations_media / "posters" / f"{stem}.webp",
            f"media/journey/posters/{stem}.webp")
        item["href"] = f"media/journey/{stem}.mp4"
        item["poster"] = f"media/journey/posters/{stem}.webp"
        if "trace" in item:
            item["trace"] = item["trace"].replace(
                "media/trajectories/", "media/journey/trajectories/"
            )
    for entry in journey_manifest["trajectories"]:
        name = Path(entry["href"]).name
        put(foundations_media / "trajectories" / name,
            f"media/journey/trajectories/{name}")
        entry["href"] = f"media/journey/trajectories/{name}"
    for name in NETWORK_TRACES:
        put(foundations_media / "network" / name, f"media/journey/network/{name}")

    put(report_pdf, "reports/Recovery_Policy_Learning_Technical_Report.pdf")

    journey_data = build_journey_data(repo_root, journey_manifest)
    journey_path = site_public / "data" / "journey-data.json"
    atomic_write_json(journey_path, journey_data, overwrite=True)
    put(journey_path, "data/journey-data.json", generated=True)

    manifest_path = site_public / "staging-manifest.json"
    atomic_write_json(
        manifest_path,
        {"schema_version": SCHEMA_VERSION, "files": staged},
        overwrite=True,
    )
    return {
        "files": len(staged),
        "site_public": str(site_public),
    }


def verify_staging(repo_root: Path) -> int:
    """Re-hash every staged file against the manifest; returns the file count."""
    site_public = repo_root / "site" / "public"
    manifest = json.loads((site_public / "staging-manifest.json").read_text())
    for entry in manifest["files"]:
        path = site_public / entry["path"]
        if not path.exists():
            raise FoundationsError(f"staged file missing: {entry['path']}")
        if file_sha256(path) != entry["sha256"]:
            raise FoundationsError(f"staged file drifted: {entry['path']}")
    return len(manifest["files"])
