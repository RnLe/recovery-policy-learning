"""Ledger hash chain and recount validator.

These tests protect the exact-budget claim: the accounting is only as strong
as the tamper-evidence of the ledger and the independence of the recount.
"""

from __future__ import annotations

import json

import pytest

from grounded_recovery.integrity import (
    GENESIS_HASH,
    IntegrityError,
    LedgerWriter,
    read_ledger,
)


def write_ledger(path, entries):
    writer = LedgerWriter(path)
    for index, targets in enumerate(entries):
        writer.append(
            episode_id=f"base_{index:05d}",
            episode_targets=targets,
            episode_steps=targets,
            oracle_calls=targets,
            budget_truncated=False,
            episode_checksum="aa" * 32,
        )
    final = writer.finalize(contract_hash="cc" * 32, manifest_hash="dd" * 32)
    return final


def test_chain_verifies_and_cumulatives_are_running_sums(tmp_path) -> None:
    path = tmp_path / "collection_ledger.jsonl"
    write_ledger(path, [5, 3, 7])
    rows = read_ledger(path)
    assert [row.episode_targets for row in rows] == [5, 3, 7]
    assert [row.cumulative_targets for row in rows] == [5, 8, 15]
    assert [row.cumulative_episodes for row in rows] == [1, 2, 3]
    assert rows[0].prev_row_hash == GENESIS_HASH
    assert rows[1].prev_row_hash == rows[0].row_hash


def test_edited_row_detected(tmp_path) -> None:
    path = tmp_path / "collection_ledger.jsonl"
    write_ledger(path, [5, 3, 7])
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["episode_targets"] = 4  # tamper without recomputing the chain
    lines[1] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError, match="row hash mismatch"):
        read_ledger(path)


def test_deleted_row_detected(tmp_path) -> None:
    path = tmp_path / "collection_ledger.jsonl"
    write_ledger(path, [5, 3, 7])
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(IntegrityError):
        read_ledger(path)


def test_ledger_refuses_reopening(tmp_path) -> None:
    path = tmp_path / "collection_ledger.jsonl"
    write_ledger(path, [1])
    with pytest.raises(IntegrityError, match="append-only"):
        LedgerWriter(path)


def test_finalize_twice_rejected(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "collection_ledger.jsonl")
    writer.append(
        episode_id="base_00000",
        episode_targets=1,
        episode_steps=1,
        oracle_calls=1,
        budget_truncated=False,
        episode_checksum="aa" * 32,
    )
    writer.finalize(contract_hash="cc" * 32, manifest_hash="dd" * 32)
    with pytest.raises(IntegrityError, match="finalized"):
        writer.finalize(contract_hash="cc" * 32, manifest_hash="dd" * 32)
