"""Tamper-evident accounting: hash-chained ledgers and independent recounts.

The collection ledger is append-only with a per-row hash chain, fsynced on
every append; the recount validator independently re-reads the stored episode
files (verifying each content checksum) and refuses downstream use if the
ledger, the episode index, and the data disagree in any count. Integrity here
is detection, not prevention: a torn or edited ledger is discovered, and the
affected dataset is regenerated deterministically from its manifest.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from grounded_recovery.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_hex,
)

GENESIS_HASH = "0" * 64


class IntegrityError(RuntimeError):
    """Stored accounting and stored data disagree; downstream use is refused."""


@dataclass(frozen=True)
class LedgerRow:
    """One collected episode's accounting entry in the hash chain."""

    row_index: int
    episode_id: str
    episode_targets: int
    episode_steps: int
    oracle_calls: int
    budget_truncated: bool
    episode_checksum: str
    cumulative_targets: int
    cumulative_steps: int
    cumulative_episodes: int
    prev_row_hash: str
    row_hash: str


def _row_hash(prev_row_hash: str, payload: dict[str, object]) -> str:
    return sha256_hex(prev_row_hash.encode("ascii") + canonical_json_bytes(payload))


class LedgerWriter:
    """Append-only collection ledger with a per-row hash chain."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise IntegrityError(f"ledger {self.path} already exists; ledgers are append-only")
        meta_path = self.path.with_name("ledger_meta.json")
        if meta_path.exists():
            raise IntegrityError(f"finalized ledger meta {meta_path} already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="ascii")
        self._prev_hash = GENESIS_HASH
        self._rows = 0
        self._cumulative_targets = 0
        self._cumulative_steps = 0
        self._finalized = False

    def append(
        self,
        *,
        episode_id: str,
        episode_targets: int,
        episode_steps: int,
        oracle_calls: int,
        budget_truncated: bool,
        episode_checksum: str,
    ) -> LedgerRow:
        if self._finalized:
            raise IntegrityError("ledger already finalized")
        self._cumulative_targets += episode_targets
        self._cumulative_steps += episode_steps
        payload = {
            "row_index": self._rows,
            "episode_id": episode_id,
            "episode_targets": episode_targets,
            "episode_steps": episode_steps,
            "oracle_calls": oracle_calls,
            "budget_truncated": budget_truncated,
            "episode_checksum": episode_checksum,
            "cumulative_targets": self._cumulative_targets,
            "cumulative_steps": self._cumulative_steps,
            "cumulative_episodes": self._rows + 1,
            "prev_row_hash": self._prev_hash,
        }
        row = LedgerRow(**payload, row_hash=_row_hash(self._prev_hash, payload))
        self._handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._prev_hash = row.row_hash
        self._rows += 1
        return row

    def finalize(self, *, contract_hash: str, manifest_hash: str) -> str:
        if self._finalized:
            raise IntegrityError("ledger already finalized")
        self._handle.close()
        self._finalized = True
        atomic_write_json(
            self.path.with_name("ledger_meta.json"),
            {
                "row_count": self._rows,
                "final_row_hash": self._prev_hash,
                "cumulative_targets": self._cumulative_targets,
                "cumulative_steps": self._cumulative_steps,
                "contract_hash": contract_hash,
                "manifest_hash": manifest_hash,
            },
        )
        return self._prev_hash

    def close(self) -> None:
        if not self._finalized:
            self._handle.close()


def read_ledger(path: Path) -> list[LedgerRow]:
    """Read a ledger and verify its full hash chain."""
    rows: list[LedgerRow] = []
    prev_hash = GENESIS_HASH
    for index, raw in enumerate(read_jsonl(path)):
        row = LedgerRow(**raw)
        payload = {key: value for key, value in asdict(row).items() if key != "row_hash"}
        if row.row_index != index:
            raise IntegrityError(f"ledger row {index}: stored row_index {row.row_index}")
        if row.prev_row_hash != prev_hash:
            raise IntegrityError(f"ledger row {index}: hash chain broken (prev mismatch)")
        expected = _row_hash(prev_hash, payload)
        if row.row_hash != expected:
            raise IntegrityError(f"ledger row {index}: row hash mismatch (edited row?)")
        rows.append(row)
        prev_hash = row.row_hash
    return rows


def recount_dataset(dataset_dir: Path) -> dict[str, object]:
    """Independently recount stored episodes against ledger and index.

    Re-reads every episode file (verifying its content checksum), recomputes
    all counts, and compares them with the ledger rows, the ledger meta, and
    the episode index. Raises ``IntegrityError`` on any disagreement.
    """
    from grounded_recovery.data import read_episode

    dataset_dir = Path(dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    ledger_rows = read_ledger(dataset_dir / "collection_ledger.jsonl")
    meta = read_json(dataset_dir / "ledger_meta.json")
    index_rows = read_jsonl(dataset_dir / "episode_index.jsonl")

    if meta["row_count"] != len(ledger_rows):
        raise IntegrityError(
            f"ledger meta records {meta['row_count']} rows, ledger has {len(ledger_rows)}"
        )
    if ledger_rows and meta["final_row_hash"] != ledger_rows[-1].row_hash:
        raise IntegrityError("ledger meta final hash does not match the last row")
    if len(index_rows) != len(ledger_rows):
        raise IntegrityError(
            f"episode index has {len(index_rows)} rows, ledger has {len(ledger_rows)}"
        )

    stored_ids = sorted(p.stem for p in episodes_dir.glob("*.json"))
    ledger_ids = [row.episode_id for row in ledger_rows]
    if sorted(ledger_ids) != stored_ids:
        missing = set(ledger_ids) - set(stored_ids)
        extra = set(stored_ids) - set(ledger_ids)
        raise IntegrityError(
            f"episode files and ledger disagree (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )

    total_targets = 0
    total_steps = 0
    for ledger_row, index_row in zip(ledger_rows, index_rows, strict=True):
        if index_row["episode_id"] != ledger_row.episode_id:
            raise IntegrityError(
                f"index/ledger order mismatch at row {ledger_row.row_index}"
            )
        arrays, sidecar = read_episode(episodes_dir, ledger_row.episode_id)
        recounted_targets = arrays.revealed_targets
        recounted_steps = arrays.length
        if recounted_targets != ledger_row.episode_targets:
            raise IntegrityError(
                f"episode {ledger_row.episode_id}: {recounted_targets} revealed targets "
                f"in data, ledger records {ledger_row.episode_targets}"
            )
        if recounted_steps != ledger_row.episode_steps:
            raise IntegrityError(
                f"episode {ledger_row.episode_id}: {recounted_steps} steps in data, "
                f"ledger records {ledger_row.episode_steps}"
            )
        if sidecar.content_checksum != ledger_row.episode_checksum:
            raise IntegrityError(
                f"episode {ledger_row.episode_id}: checksum differs from ledger"
            )
        if sidecar.oracle_calls != ledger_row.oracle_calls:
            raise IntegrityError(
                f"episode {ledger_row.episode_id}: oracle call count differs from ledger"
            )
        total_targets += recounted_targets
        total_steps += recounted_steps

    if total_targets != meta["cumulative_targets"]:
        raise IntegrityError(
            f"recounted {total_targets} targets, ledger meta records "
            f"{meta['cumulative_targets']}"
        )
    if total_steps != meta["cumulative_steps"]:
        raise IntegrityError(
            f"recounted {total_steps} steps, ledger meta records {meta['cumulative_steps']}"
        )
    return {
        "episodes": len(ledger_rows),
        "targets": total_targets,
        "steps": total_steps,
        "final_row_hash": ledger_rows[-1].row_hash if ledger_rows else GENESIS_HASH,
    }


# --- Stored-episode replay verification --------------------------------------

def verify_episode_replay(session, episodes_dir: Path, episode_id: str) -> None:
    """Replay one stored episode and compare every step against the arrays.

    Resets from the stored seed and executes the stored actions; image bytes,
    direction, reward, and termination flags must match exactly, proving that
    the stored data is what the environment actually produced.
    """
    import numpy as np

    from grounded_recovery.data import read_episode

    arrays, sidecar = read_episode(episodes_dir, episode_id)
    observation = session.reset(sidecar.environment_seed)
    for t in range(arrays.length):
        if not np.array_equal(observation.image, arrays.images[t]):
            raise IntegrityError(f"episode {episode_id}: image divergence at t={t}")
        if observation.direction != int(arrays.direction[t]):
            raise IntegrityError(f"episode {episode_id}: direction divergence at t={t}")
        observation = session.step(int(arrays.executed_action[t]))
        if abs(observation.reward - float(arrays.reward[t])) > 1e-9:
            raise IntegrityError(f"episode {episode_id}: reward divergence at t={t}")
        if observation.terminated != bool(arrays.terminated[t]) or (
            observation.truncated != bool(arrays.truncated[t])
        ):
            raise IntegrityError(f"episode {episode_id}: termination divergence at t={t}")


def verify_dataset(cfg, dataset_dir: Path, *, replay_sample: int = 3) -> dict[str, object]:
    """Recount plus provenance stamps plus a deterministic replay sample."""
    from grounded_recovery.config import contract_hash
    from grounded_recovery.data import read_episode
    from grounded_recovery.world import WorldSession

    dataset_dir = Path(dataset_dir)
    summary = recount_dataset(dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    meta = read_json(dataset_dir / "dataset_meta.json")
    expected_contract = contract_hash(cfg)
    if meta["contract_hash"] != expected_contract:
        raise IntegrityError(
            f"dataset was collected under contract {meta['contract_hash'][:12]}..., "
            f"current contract is {expected_contract[:12]}..."
        )
    ledger_rows = read_ledger(dataset_dir / "collection_ledger.jsonl")
    for row in ledger_rows:
        _, sidecar = read_episode(episodes_dir, row.episode_id)
        if sidecar.contract_hash != expected_contract:
            raise IntegrityError(
                f"episode {row.episode_id}: contract stamp mismatch"
            )
        if sidecar.dataset_schema_version != cfg.data.dataset_schema_version:
            raise IntegrityError(
                f"episode {row.episode_id}: dataset schema version mismatch"
            )
    # Deterministic, evenly spaced replay sample including first and last.
    count = len(ledger_rows)
    sample = min(max(replay_sample, 1), count)
    indices = sorted({round(i * (count - 1) / max(sample - 1, 1)) for i in range(sample)})
    session = WorldSession(cfg.environment)
    try:
        for index in indices:
            verify_episode_replay(session, episodes_dir, ledger_rows[index].episode_id)
    finally:
        session.close()
    summary["replayed_episodes"] = [ledger_rows[index].episode_id for index in indices]
    return summary


# --- Matched-arm fairness audit ----------------------------------------------

# Exposure-ledger fields that must be pairwise equal between the two
# full-budget arms at every update. Availability counts and losses are
# logged-only quantities and deliberately excluded.
FAIRNESS_EQUAL_FIELDS: tuple[str, ...] = (
    "round",
    "update",
    "base_targets_drawn",
    "new_targets_drawn",
    "cumulative_base_exposures",
    "cumulative_new_exposures",
    "loss_denominator",
    "optimizer_step",
)


def audit_round_fairness(
    rows_a: list[dict[str, object]],
    rows_b: list[dict[str, object]],
    *,
    arm_a: str,
    arm_b: str,
) -> dict[str, object]:
    """Assert exact exposure/update matching between the two augmented arms."""
    if len(rows_a) != len(rows_b):
        raise IntegrityError(
            f"unequal optimizer update counts: {arm_a} has {len(rows_a)}, "
            f"{arm_b} has {len(rows_b)}"
        )
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b, strict=True)):
        for field in FAIRNESS_EQUAL_FIELDS:
            if row_a[field] != row_b[field]:
                raise IntegrityError(
                    f"fairness violation at update row {index}: {field} is "
                    f"{row_a[field]} for {arm_a} but {row_b[field]} for {arm_b}"
                )
    last = rows_a[-1] if rows_a else None
    return {
        "updates": len(rows_a),
        "cumulative_base_exposures": last["cumulative_base_exposures"] if last else 0,
        "cumulative_new_exposures": last["cumulative_new_exposures"] if last else 0,
    }


# --- Scientific-integrity phases ----------------------------------------------

INTEGRITY_PHASES = ("freeze", "preopen", "release")


def run_integrity(
    contract_path: Path,
    phase: str,
    manifest_root: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, object]:
    """Machine-readable PASS/FAIL integrity report for one lifecycle phase.

    Each phase includes all earlier checks: ``freeze`` verifies the frozen
    contract, manifests, disjointness, preflight evidence, and the eligible
    panel; ``preopen`` additionally verifies every planned bundle's budgets,
    exposure equality, and unchanged base; ``release`` additionally verifies
    the single opening and that the published summary recomputes from raw rows.
    """
    from grounded_recovery.config import (
        SPLIT_NAMES,
        contract_hash,
        load_and_validate,
        unresolved_fields,
    )
    from grounded_recovery.data import (
        load_eligible_entries,
        load_split_manifest,
        verify_manifest_contract,
    )
    from grounded_recovery.experiment import (
        ARM_EXTRA,
        ARM_RECOVERY,
        _find_preflight_evidence,
    )

    if phase not in INTEGRITY_PHASES:
        raise IntegrityError(f"unknown integrity phase {phase!r}")
    checks: list[dict[str, object]] = []

    def check(name: str, callable_check) -> None:
        try:
            detail = callable_check()
            checks.append({"check": name, "passed": True, "detail": detail})
        except Exception as error:  # noqa: BLE001 - every failure becomes a FAIL row
            checks.append({"check": name, "passed": False, "detail": str(error)})

    cfg = load_and_validate(Path(contract_path))
    cfg_hash = contract_hash(cfg)

    check("contract_frozen", lambda: (
        "FROZEN" if cfg.study.status == "FROZEN" and not unresolved_fields(cfg)
        else (_ for _ in ()).throw(IntegrityError("contract not frozen/resolved"))
    ))
    for split in SPLIT_NAMES:
        check(f"manifest_{split}", lambda s=split: (
            verify_manifest_contract(manifest_root, s, cfg),
            load_split_manifest(manifest_root, s)[1],
        )[1])
    check("disjointness", lambda: (
        read_json(Path(manifest_root) / "disjointness_report.json")["audit"]["disjoint"]
        or (_ for _ in ()).throw(IntegrityError("overlap recorded"))
    ))
    check("preflight_evidence", lambda: _find_preflight_evidence(
        cfg, data_root, load_split_manifest(manifest_root, "operator_preflight")[1]
    )["path"])
    check("eligible_panel", lambda: load_eligible_entries(cfg, manifest_root)[1][
        "eligible_count"
    ])

    if phase in ("preopen", "release"):
        for bundle_id in cfg.seeds.bundle_ids:
            bundle_root = Path(data_root) / cfg_hash[:12] / bundle_id

            def bundle_check(root=bundle_root, bid=bundle_id):
                summary = read_json(root / "bundle_summary.json")
                if summary["contract_hash"] != cfg_hash:
                    raise IntegrityError("bundle contract mismatch")
                extra = summary["arms"][ARM_EXTRA]["cumulative"]
                recovery = summary["arms"][ARM_RECOVERY]["cumulative"]
                if extra != recovery:
                    raise IntegrityError("arm exposures unequal")
                for arm in (ARM_EXTRA, ARM_RECOVERY):
                    budgets = [c["revealed_targets"] for c in
                               summary["arms"][arm]["collections"]]
                    if sum(budgets) != cfg.data.b:
                        raise IntegrityError(f"{arm} total revealed targets != B")
                return {"updates": extra["updates"], "b": cfg.data.b}

            check(f"bundle_{bundle_id}", bundle_check)

    if phase == "release":
        results_dir = Path(results_root) / cfg_hash[:12]

        def opening_check():
            receipt = read_json(results_dir / "opening_receipt.json")
            complete = read_json(results_dir / "opening_complete.json")
            if receipt["contract_hash"] != cfg_hash:
                raise IntegrityError("receipt contract mismatch")
            if complete["completed_cells"] != receipt["expected_cells"]:
                raise IntegrityError("incomplete crossed cells")
            return {"cells": complete["completed_cells"], "rows": complete["rows"]}

        check("single_opening", opening_check)

        def summary_recompute_check():
            from grounded_recovery.publish import group_rows, rows_from_jsonl
            from grounded_recovery.statistics import success_summary

            summary = read_json(results_dir / "statistical_summary.json")
            rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
            grouped = group_rows(rows)
            for bundle, recorded in summary["per_bundle_deltas"].items():
                recovery = success_summary(grouped[(bundle, ARM_RECOVERY, "unseen")])
                extra = success_summary(grouped[(bundle, ARM_EXTRA, "unseen")])
                if abs((recovery.rate - extra.rate) - recorded) > 1e-9:
                    raise IntegrityError(f"delta for {bundle} does not recompute")
            return {"bundles": len(summary["per_bundle_deltas"])}

        check("summary_recomputes", summary_recompute_check)

    passed = all(row["passed"] for row in checks)
    return {"phase": phase, "passed": passed, "contract_hash": cfg_hash,
            "checks": checks}
