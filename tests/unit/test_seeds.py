"""Named seed derivation.

These tests protect reproducibility: every dataset, model, and evaluation in
the study must be reconstructable from the contract alone, which requires the
seed map to be a stable, collision-aware function of
``(root_seed, bundle_id, component)``.
"""

from __future__ import annotations

import itertools
import subprocess
import sys

import pytest

from grounded_recovery import seeds

# Values computed with an independent implementation of the documented
# derivation (SHA-256 over "root\x1fbundle\x1fcomponent", first 8 bytes,
# masked to 63 bits). A change here means every artifact identity changes.
GOLDEN = {
    (20260826, "B00", "smoke"): 2458632096123964017,
    (20260826, "B00", "manifest.base.0"): 5384500253104970886,
    (20260826, "B01", "init"): 8075057527819335367,
    (7, "B00", "evaluation"): 3879334734855339996,
}


def test_derived_seed_golden_values() -> None:
    for (root, bundle, component), expected in GOLDEN.items():
        assert seeds.derive_seed(root, bundle, component) == expected


def test_derived_seeds_in_valid_range() -> None:
    for (root, bundle, component) in GOLDEN:
        value = seeds.derive_seed(root, bundle, component)
        assert 0 <= value < 2**63


def test_distinct_components_distinct_seeds() -> None:
    values = [seeds.derive_seed(20260826, "B00", c) for c in seeds.SEED_COMPONENTS]
    assert len(set(values)) == len(values)


def test_distinct_bundles_distinct_seeds() -> None:
    values = [seeds.derive_seed(20260826, bundle, "init") for bundle in ("B00", "B01", "B02")]
    assert len(set(values)) == len(values)


def test_seed_stream_matches_indexed_components() -> None:
    stream = seeds.seed_stream(20260826, "B00", "manifest.base")
    first_three = list(itertools.islice(stream, 3))
    assert first_three == [
        seeds.derive_seed(20260826, "B00", f"manifest.base.{i}") for i in range(3)
    ]
    assert len(set(first_three)) == 3


def test_unknown_component_rejected() -> None:
    with pytest.raises(seeds.UnknownSeedComponentError):
        seeds.derive_seed(20260826, "B00", "not.a.component")


def test_separator_in_bundle_rejected() -> None:
    # The separator byte delimits the hashed fields; allowing it inside a
    # bundle id would let two different identities collide.
    with pytest.raises(ValueError):
        seeds.derive_seed(20260826, "B\x1f00", "smoke")


def test_independent_of_pythonhashseed() -> None:
    # The derivation must not involve Python's salted hash(); identical calls
    # under different PYTHONHASHSEED values must agree.
    script = (
        "from grounded_recovery.seeds import derive_seed;"
        "print(derive_seed(20260826, 'B00', 'smoke'))"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert outputs == {str(GOLDEN[(20260826, "B00", "smoke")])}
