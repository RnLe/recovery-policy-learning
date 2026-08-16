"""Command-line entry points for the Grounded Recovery study.

Each subcommand loads and validates the configuration, calls exactly one
library function, prints the artifact paths it created, and exits nonzero on
any invariant failure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from grounded_recovery.artifacts import ImmutableArtifactError, atomic_write_json
from grounded_recovery.config import (
    ConfigError,
    ExperimentConfig,
    contract_hash,
    load_and_validate,
)
from grounded_recovery.data import ManifestError, make_manifests
from grounded_recovery.integrity import IntegrityError
from grounded_recovery.oracle import (
    OracleSupportError,
    OracleSyncError,
    run_synchronized_episode,
)
from grounded_recovery.perturbations import DerangementError
from grounded_recovery.seeds import seed_stream
from grounded_recovery.world import WorldError, WorldSession

SMOKE_EPISODES = 3
FINGERPRINT_PATH = Path("environment_fingerprint.json")
MANIFEST_ROOT = Path("manifests")

_EXPECTED_ERRORS = (
    ConfigError,
    IntegrityError,
    ManifestError,
    WorldError,
    OracleSyncError,
    OracleSupportError,
    DerangementError,
    ImmutableArtifactError,
    FileNotFoundError,
)


def _load(args: argparse.Namespace) -> ExperimentConfig:
    return load_and_validate(Path(args.config))


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = _load(args)
    session = WorldSession(cfg.environment)
    try:
        stream = seed_stream(cfg.seeds.root_seed, "global", "smoke")
        episodes = []
        for seed in itertools.islice(stream, SMOKE_EPISODES):
            trace = run_synchronized_episode(session, seed, lambda t, rec: rec)
            if not trace.success:
                print(f"smoke episode on seed {seed} did not succeed", file=sys.stderr)
                return 1
            episodes.append(
                {
                    "seed": seed,
                    "mission": trace.mission,
                    "transitions": len(trace.transitions),
                    "oracle_calls": trace.oracle_calls,
                    "scenario_hash": trace.scenario_hash,
                }
            )
        session.reset(episodes[0]["seed"])
        # `smoke` runs before the freeze, against the mutable pilot config, so
        # the digest below is the pilot's and not the frozen protocol's. The
        # stage and source are recorded beside it so the two cannot be mistaken
        # for each other by anyone comparing files.
        fingerprint = {
            "stage": "pre-freeze smoke",
            "config": str(Path(args.config)),
            "contract_hash": contract_hash(cfg),
            "environment": session.fingerprint(),
            "smoke_episodes": episodes,
        }
    finally:
        session.close()
    atomic_write_json(FINGERPRINT_PATH, fingerprint, overwrite=True)
    print(f"smoke OK: {SMOKE_EPISODES} nominal oracle episodes succeeded")
    print(f"wrote {FINGERPRINT_PATH}")
    return 0


def cmd_make_manifests(args: argparse.Namespace) -> int:
    cfg = _load(args)
    hashes = make_manifests(cfg, MANIFEST_ROOT)
    for split, digest in hashes.items():
        print(f"manifest {split}: {digest}")
    print(f"wrote {MANIFEST_ROOT}/<split>/entries.jsonl and {MANIFEST_ROOT}/"
          f"disjointness_report.json")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from grounded_recovery.experiment import run_preflight

    cfg = _load(args)
    out_dir = Path("data") / "preflight" / contract_hash(cfg)[:12]
    report = run_preflight(cfg, MANIFEST_ROOT, out_dir)
    print(json.dumps({"passed": report["passed"], "families": {
        family: {
            "episodes": summary["episodes"],
            "delivered": summary["delivered"],
            "recovery_rate_delivered": summary["recovery_rate_delivered"],
            "passed": summary["passed"],
        }
        for family, summary in report["families"].items()
    }}, indent=2))
    print(f"wrote {out_dir}/preflight_report.json")
    if not report["passed"]:
        print("preflight FAILED the oracle-recovery gate", file=sys.stderr)
        return 1
    return 0


def cmd_collect_base(args: argparse.Namespace) -> int:
    from grounded_recovery.data import collect_base

    cfg = _load(args)
    summary = collect_base(cfg, args.bundle, MANIFEST_ROOT, Path("data"))
    print(
        f"collected exactly {summary.n0} targets over {summary.episodes} episodes "
        f"({summary.steps} steps, {summary.oracle_calls} oracle calls, "
        f"final episode budget-truncated: {summary.final_episode_budget_truncated})"
    )
    print(f"wrote {summary.dataset_dir}")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    from grounded_recovery.experiment import run_validation_pilot

    cfg = _load(args)
    report = run_validation_pilot(cfg, args.bundle, MANIFEST_ROOT, Path("data"))
    print(f"validation pilot on {report['panel_scenarios']} scenarios:")
    for arm, by_slice in report["success"].items():
        cells = ", ".join(
            f"{slice_name} {values['successes']}/{values['assigned']}"
            for slice_name, values in by_slice.items()
        )
        print(f"  {arm}: {cells}")
    for slice_name, delta in report["paired_recovery_minus_extra"].items():
        print(f"  paired recovery-extra ({slice_name}): {delta:+.3f}")
    print("wrote pilot_report.json and validation_evaluation_rows.jsonl")
    return 0


def _register_pilot(subparsers) -> None:
    pilot = subparsers.add_parser(
        "pilot", help="run one pilot bundle and the validation-slice evaluation"
    )
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--bundle", required=True)
    pilot.set_defaults(func=cmd_pilot)


def cmd_freeze(args) -> int:
    from grounded_recovery.experiment import run_freeze

    record = run_freeze(
        Path(args.config), Path("configs/experiment_contract.yaml"),
        MANIFEST_ROOT, Path("data"), Path("."),
    )
    print("frozen contract written: configs/experiment_contract.yaml")
    print(f"contract hash: {record['contract_hash']}")
    print(f"eligible unseen panel: {record['eligible']['count']} of "
          f"{record['eligible']['candidates']} candidates "
          f"({record['eligible']['retained_fraction']:.2%} retained)")
    print(f"planned bundles: {record['planned_bundles']} (R_train {record['r_train']})")
    return 0


def cmd_run_bundle(args) -> int:
    from grounded_recovery.experiment import run_bundle_frozen

    summary = run_bundle_frozen(
        Path(args.contract), args.bundle, MANIFEST_ROOT, Path("data")
    )
    print(f"bundle {args.bundle} complete under contract "
          f"{summary['contract_hash'][:12]}")
    return 0


def cmd_evaluate_final(args) -> int:
    from grounded_recovery.experiment import evaluate_final

    outcome = evaluate_final(
        Path(args.contract), MANIFEST_ROOT, Path("data"), Path("results")
    )
    print(f"confirmatory opening complete: {outcome['cells']} cells, "
          f"{outcome['rows']} episode rows")
    print(f"results: {outcome['results_dir']}")
    return 0


def cmd_analyze(args) -> int:
    from grounded_recovery.artifacts import read_json
    from grounded_recovery.publish import analyze_results

    cfg = load_and_validate(Path(args.contract))
    record = read_json(Path(args.contract).with_name("freeze_record.json"))
    results_dir = Path("results") / contract_hash(cfg)[:12]
    summary = analyze_results(cfg, results_dir, planned_r_train=record["r_train"])
    interval = summary["interval"]
    print(f"analysis status: {summary['analysis_status']}")
    print(f"claim state: {summary['claim_state']}")
    print(f"mean paired difference (unseen ITT): "
          f"{summary['mean_paired_difference']:+.4f}")
    if interval:
        print(f"95% paired t interval: [{interval['lower']:+.4f}, "
              f"{interval['upper']:+.4f}]")
    print(f"wrote {results_dir}/statistical_summary.json, tables/, figures/")
    return 0


def cmd_audit(args) -> int:
    from grounded_recovery.publish import audit_results

    cfg = load_and_validate(Path(args.contract))
    results_dir = Path("results") / contract_hash(cfg)[:12]
    audits = audit_results(cfg, results_dir, data_root=Path(args.data_root))
    budget = audits["budget_exposure"]
    print(f"descriptive audits ({audits['status']})")
    print(f"  budget audit covers {budget['bundles_audited']} bundle(s); "
          f"arm exposures equal: {budget['exposures_equal_across_arms']}")
    for contrast in audits["secondary_contrasts"]:
        if contrast["mean"] is None:
            continue
        print(f"  {contrast['first_arm']} minus {contrast['second_arm']} "
              f"({contrast['slice']}): {contrast['mean']:+.4f} "
              f"[{contrast['lower']:+.4f}, {contrast['upper']:+.4f}]")
    print(f"wrote {results_dir}/descriptive_audits.json, tables/, figures/")
    return 0


def cmd_export_typst(args) -> int:
    from grounded_recovery.artifacts import read_json
    from grounded_recovery.publish import export_typst

    cfg = load_and_validate(Path(args.contract))
    record = read_json(Path(args.contract).with_name("freeze_record.json"))
    results_dir = Path("results") / contract_hash(cfg)[:12]
    export_typst(cfg, results_dir, record, Path(args.out))
    print(f"wrote generated Typst fragments to {args.out}")
    return 0


def cmd_integrity(args) -> int:
    from grounded_recovery.integrity import run_integrity

    report = run_integrity(
        Path(args.contract), args.phase, MANIFEST_ROOT, Path("data"), Path("results")
    )
    for row in report["checks"]:
        marker = "PASS" if row["passed"] else "FAIL"
        print(f"  [{marker}] {row['check']}: {row['detail']}")
    print(f"integrity phase {args.phase}: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


def cmd_publish_result(args) -> int:
    from grounded_recovery.artifacts import read_json
    from grounded_recovery.publish import publish_result

    cfg = load_and_validate(Path(args.contract))
    record = read_json(Path(args.contract).with_name("freeze_record.json"))
    results_dir = None if args.protocol_only else (
        Path("results") / contract_hash(cfg)[:12]
    )
    publish_result(
        cfg, Path(args.out),
        results_dir=results_dir,
        freeze_record=record,
        protocol_only=args.protocol_only,
    )
    mode = "protocol-only" if args.protocol_only else "results"
    print(f"published {mode} bundle to {args.out}")
    return 0


def cmd_media(args) -> int:
    from grounded_recovery.media import generate_result_media
    from grounded_recovery.publish import refresh_public_media

    cfg = load_and_validate(Path(args.contract))
    results_dir = Path("results") / contract_hash(cfg)[:12]
    manifest = generate_result_media(
        cfg, results_dir, MANIFEST_ROOT, Path(args.data_root),
        results_dir / "media", bundle_id=args.bundle,
    )
    print(f"rendered {len(manifest['media'])} animations to {results_dir / 'media'}")
    if args.refresh_public:
        copied = refresh_public_media(results_dir, Path(args.refresh_public))
        print(f"refreshed {copied} media files in {args.refresh_public}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr",
        description="Grounded Recovery experiment commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="environment/oracle smoke test + fingerprint")
    smoke.add_argument("--config", required=True)
    smoke.set_defaults(func=cmd_smoke)

    manifests = subparsers.add_parser("make-manifests", help="generate the eight split manifests")
    manifests.add_argument("--config", required=True)
    manifests.set_defaults(func=cmd_make_manifests)

    preflight = subparsers.add_parser("preflight", help="oracle/operator preflight gate (G1)")
    preflight.add_argument("--config", required=True)
    preflight.set_defaults(func=cmd_preflight)

    collect = subparsers.add_parser(
        "collect-base", help="collect the shared base dataset with exactly N0 targets"
    )
    collect.add_argument("--config", required=True)
    collect.add_argument("--bundle", required=True)
    collect.set_defaults(func=cmd_collect_base)

    _register_pilot(subparsers)

    freeze = subparsers.add_parser(
        "freeze", help="resolve the pilot config into the frozen contract (G7)"
    )
    freeze.add_argument("--config", required=True)
    freeze.set_defaults(func=cmd_freeze)

    run_bundle = subparsers.add_parser(
        "run-bundle", help="run one final pipeline bundle (frozen contract only)"
    )
    run_bundle.add_argument("--contract", required=True)
    run_bundle.add_argument("--bundle", required=True)
    run_bundle.set_defaults(func=cmd_run_bundle)

    final = subparsers.add_parser(
        "evaluate-final", help="the single confirmatory test opening (frozen only)"
    )
    final.add_argument("--contract", required=True)
    final.set_defaults(func=cmd_evaluate_final)

    analyze = subparsers.add_parser(
        "analyze", help="frozen analysis of the confirmatory results"
    )
    analyze.add_argument("--contract", required=True)
    analyze.set_defaults(func=cmd_analyze)

    audit = subparsers.add_parser(
        "audit",
        help="descriptive audits beside the frozen analysis (secondary, "
             "not prespecified)",
    )
    audit.add_argument("--contract", required=True)
    audit.add_argument("--data-root", default="data")
    audit.set_defaults(func=cmd_audit)

    export = subparsers.add_parser(
        "export-typst", help="generate the report's Typst fragments from the analysis"
    )
    export.add_argument("--contract", required=True)
    export.add_argument("--out", default="report/generated")
    export.set_defaults(func=cmd_export_typst)

    integrity = subparsers.add_parser(
        "integrity", help="scientific-integrity PASS/FAIL report for one phase"
    )
    integrity.add_argument("--contract", required=True)
    integrity.add_argument("--phase", required=True, choices=["freeze", "preopen", "release"])
    integrity.set_defaults(func=cmd_integrity)

    media = subparsers.add_parser(
        "media", help="render the rollout animations from the stored evaluation"
    )
    media.add_argument("--contract", required=True)
    media.add_argument("--bundle", default="B00")
    media.add_argument("--data-root", default="data")
    media.add_argument(
        "--refresh-public",
        metavar="BUNDLE_DIR",
        help="also replace the media of an existing published bundle",
    )
    media.set_defaults(func=cmd_media)

    publish = subparsers.add_parser(
        "publish-result", help="write the public evidence bundle (results or protocol-only)"
    )
    publish.add_argument("--contract", required=True)
    publish.add_argument("--out", default="public_result")
    publish.add_argument("--protocol-only", action="store_true")
    publish.set_defaults(func=cmd_publish_result)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except _EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
