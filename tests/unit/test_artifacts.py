"""Canonical hashing and atomic write-once I/O.

These tests protect the identity layer: every contract, manifest, ledger, and
episode hash in the study is only meaningful if canonical serialization is
stable and immutable artifacts really are write-once.
"""

from __future__ import annotations

import os

import pytest

from grounded_recovery import artifacts


def test_canonical_json_key_order_invariance() -> None:
    # Two structurally equal objects must hash identically regardless of key
    # insertion order, otherwise contract hashes would not be identities.
    a = {"alpha": 1, "beta": [1, 2], "gamma": {"x": 1, "y": 2}}
    b = {"gamma": {"y": 2, "x": 1}, "beta": [1, 2], "alpha": 1}
    assert artifacts.canonical_json_bytes(a) == artifacts.canonical_json_bytes(b)
    assert artifacts.hash_json(a) == artifacts.hash_json(b)


def test_canonical_json_golden_bytes() -> None:
    assert artifacts.canonical_json_bytes({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


def test_canonical_json_rejects_nan_and_infinity() -> None:
    # A NaN would serialize to non-standard JSON and silently corrupt hashes.
    with pytest.raises(ValueError):
        artifacts.canonical_json_bytes({"value": float("nan")})
    with pytest.raises(ValueError):
        artifacts.canonical_json_bytes({"value": float("inf")})


def test_canonical_json_rejects_non_json_types() -> None:
    with pytest.raises(TypeError):
        artifacts.canonical_json_bytes({"value": object()})


def test_sha256_golden_value() -> None:
    # FIPS 180 test vector; catches any accidental change of hash function.
    assert (
        artifacts.sha256_hex(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_atomic_write_refuses_overwrite(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    artifacts.atomic_write_json(target, {"v": 1})
    with pytest.raises(artifacts.ImmutableArtifactError):
        artifacts.atomic_write_json(target, {"v": 2})
    # The refused write must not have altered the original.
    assert artifacts.read_json(target) == {"v": 1}
    artifacts.atomic_write_json(target, {"v": 2}, overwrite=True)
    assert artifacts.read_json(target) == {"v": 2}


def test_json_write_is_deterministic(tmp_path) -> None:
    first = artifacts.atomic_write_json(tmp_path / "a.json", {"b": 2, "a": 1})
    second = artifacts.atomic_write_json(tmp_path / "b.json", {"a": 1, "b": 2})
    assert first == second


def test_jsonl_roundtrip_and_hash(tmp_path) -> None:
    rows = [{"i": 0}, {"i": 1, "note": "x"}]
    digest = artifacts.atomic_write_jsonl(tmp_path / "rows.jsonl", rows)
    assert artifacts.read_jsonl(tmp_path / "rows.jsonl") == rows
    assert artifacts.file_sha256(tmp_path / "rows.jsonl") == digest


def test_no_partial_target_on_failure(tmp_path, monkeypatch) -> None:
    # If the final rename fails, neither the target nor the temporary file may
    # remain: an interrupted write must never look like a valid artifact.
    target = tmp_path / "artifact.bin"

    def failing_replace(src, dst):  # noqa: ARG001
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated failure"):
        artifacts.atomic_write_bytes(target, b"payload")
    monkeypatch.undo()
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
