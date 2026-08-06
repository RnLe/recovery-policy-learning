"""Canonical serialization, hashing, and atomic write-once artifact I/O.

Every scientific identity in this repository (contract hashes, manifest
hashes, episode checksums) reduces to SHA-256 over the canonical JSON bytes
defined here: sorted keys, compact separators, ASCII-only, NaN/Infinity
rejected. Artifact files are written to a temporary name in the target
directory and atomically renamed into place; overwriting an existing artifact
is an error unless explicitly requested.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path


class ImmutableArtifactError(RuntimeError):
    """An operation would overwrite an existing immutable artifact."""


def canonical_json_bytes(obj: object) -> bytes:
    """Serialize ``obj`` to the canonical JSON byte form used for hashing."""
    text = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: object) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``obj``."""
    return sha256_hex(canonical_json_bytes(obj))


def _atomic_write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise ImmutableArtifactError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> str:
    """Atomically write raw bytes; return the SHA-256 of the written bytes."""
    _atomic_write_bytes(path, data, overwrite)
    return sha256_hex(data)


def atomic_write_json(path: Path, obj: object, *, overwrite: bool = False) -> str:
    """Atomically write ``obj`` as human-readable, deterministic JSON.

    The file form is indented for inspection; it is still deterministic
    (sorted keys, ASCII-only), so the returned SHA-256 of the file bytes is
    stable across runs.
    """
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    return atomic_write_bytes(path, (text + "\n").encode("ascii"), overwrite=overwrite)


def atomic_write_jsonl(path: Path, rows: Iterable[object], *, overwrite: bool = False) -> str:
    """Atomically write rows as canonical-JSON lines; return the file SHA-256."""
    lines = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return atomic_write_bytes(path, lines, overwrite=overwrite)


def read_json(path: Path) -> object:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("ascii"))


def read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    with open(path, "rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line.decode("ascii")))
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
