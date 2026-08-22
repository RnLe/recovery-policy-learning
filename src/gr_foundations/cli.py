"""Command-line surface for the foundations labs: ``grf``.

Each lab module exposes ``run(paths, *, force) -> dict``; this module only
resolves paths, dispatches, and reports where the artifacts landed. Wall-clock
information goes to ``run_info.json`` so ``metrics.json`` stays deterministic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from gr_foundations.common import LAB_TITLES, FoundationsError, LabPaths

LAB_MODULES: dict[str, str] = {
    "lab01": "gr_foundations.lab01_world",
    "lab02": "gr_foundations.lab02_decision",
    "lab03": "gr_foundations.lab03_oracle",
    "lab04": "gr_foundations.lab04_learning",
    "lab05": "gr_foundations.lab05_architecture",
    "lab06": "gr_foundations.lab06_shift",
    "lab07": "gr_foundations.lab07_methodology",
}


def _load_runner(lab_id: str):
    try:
        module = importlib.import_module(LAB_MODULES[lab_id])
    except ModuleNotFoundError as error:
        if error.name == LAB_MODULES[lab_id]:
            raise FoundationsError(f"{lab_id} is not implemented yet") from error
        raise
    return module.run


def _run_lab(lab_id: str, repo_root: Path, *, force: bool) -> None:
    paths = LabPaths(lab_id=lab_id, repo_root=repo_root)
    runner = _load_runner(lab_id)
    print(f"[{lab_id}] {paths.title}")
    started = time.time()
    summary = runner(paths, force=force)
    elapsed = time.time() - started
    run_info = {
        "lab": lab_id,
        "elapsed_seconds": round(elapsed, 1),
        "finished_unix": round(time.time(), 1),
        "argv": sys.argv[1:],
    }
    (paths.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  outputs: {paths.out_dir}")
    print(f"  report fragments: {paths.report_dir}")
    print(f"  elapsed: {elapsed:.1f}s")


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    lab_ids = list(LAB_MODULES) if args.lab == "all" else [args.lab]
    for lab_id in lab_ids:
        _run_lab(lab_id, repo_root, force=args.force)
    return 0


def cmd_media(args: argparse.Namespace) -> int:
    from gr_foundations import media_journey

    summary = media_journey.run(Path(args.repo_root).resolve(), force=args.force)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_network_trace(args: argparse.Namespace) -> int:
    from gr_foundations import network_trace

    summary = network_trace.run(Path(args.repo_root).resolve(), force=args.force)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_study_extras(args: argparse.Namespace) -> int:
    from gr_foundations.study_extras import run_study_extras

    summary = run_study_extras(
        Path(args.contract), Path(args.manifest_root), Path(args.data_root),
        Path(args.results_root),
    )
    print(summary["status"])
    for arm, cell in summary["expert_agreement"].items():
        print(f"  {arm}: agrees with the oracle "
              f"{cell['mean_agreement_rate']:.1%} of steps, "
              f"reaches the goal {cell['mean_success_rate']:.1%} of episodes")
    interval = summary["two_corruption"]["paired_recovery_minus_extra"]["interval"]
    if interval:
        print(f"  two corruptions, recovery minus extra: {interval['mean']:+.4f} "
              f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]")
    return 0


def cmd_stage_site(args: argparse.Namespace) -> int:
    from gr_foundations import site

    summary = site.stage(Path(args.repo_root).resolve(), force=args.force)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_verify_staging(args: argparse.Namespace) -> int:
    from gr_foundations import site

    count = site.verify_staging(Path(args.repo_root).resolve())
    print(f"  verified: {count} staged files")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for lab_id, title in LAB_TITLES.items():
        try:
            _load_runner(lab_id)
            status = "ready"
        except FoundationsError:
            status = "not implemented"
        print(f"{lab_id}  {title}  [{status}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grf",
        description="Run the foundations labs (exploratory pedagogical track).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one lab or all labs in order")
    run_parser.add_argument("lab", choices=[*LAB_MODULES, "all"])
    run_parser.add_argument(
        "--force", action="store_true", help="replace existing lab outputs"
    )
    run_parser.add_argument("--repo-root", default=".")
    run_parser.set_defaults(handler=cmd_run)

    list_parser = subparsers.add_parser("list", help="list labs and their status")
    list_parser.set_defaults(handler=cmd_list)

    media_parser = subparsers.add_parser(
        "media", help="generate the journey rollout videos and trajectories"
    )
    media_parser.add_argument("--force", action="store_true")
    media_parser.add_argument("--repo-root", default=".")
    media_parser.set_defaults(handler=cmd_media)

    trace_parser = subparsers.add_parser(
        "network-trace",
        help="export real weights and activations for the site's network view",
    )
    trace_parser.add_argument("--force", action="store_true")
    trace_parser.add_argument("--repo-root", default=".")
    trace_parser.set_defaults(handler=cmd_network_trace)

    extras_parser = subparsers.add_parser(
        "study-extras",
        help="open the two reserved scenario panels (exploratory, not prespecified)",
    )
    extras_parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    extras_parser.add_argument("--manifest-root", default="manifests")
    extras_parser.add_argument("--data-root", default="data")
    extras_parser.add_argument("--results-root", default="results")
    extras_parser.set_defaults(handler=cmd_study_extras)

    stage_parser = subparsers.add_parser(
        "stage-site", help="populate site/public from the evidence bundles"
    )
    stage_parser.add_argument("--force", action="store_true")
    stage_parser.add_argument("--repo-root", default=".")
    stage_parser.set_defaults(handler=cmd_stage_site)

    verify_parser = subparsers.add_parser(
        "verify-staging", help="re-hash site/public against staging-manifest.json"
    )
    verify_parser.add_argument("--repo-root", default=".")
    verify_parser.set_defaults(handler=cmd_verify_staging)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except FoundationsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
