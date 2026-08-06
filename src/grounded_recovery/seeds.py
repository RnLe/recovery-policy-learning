"""Stable named seed derivation.

Every random stream in the study is derived from ``(root_seed, bundle_id,
component)`` with SHA-256, never with Python's salted ``hash()``. Components
form a closed registry so that a typo creates an error instead of a silently
distinct stream. Derived seeds are non-negative and fit in 63 bits, which is
accepted by NumPy, Gymnasium, and PyTorch seeding interfaces.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

_SEPARATOR = "\x1f"

SEED_COMPONENTS: tuple[str, ...] = (
    "smoke",
    "manifest.base",
    "manifest.collection",
    "manifest.validation",
    "manifest.operator_preflight",
    "manifest.operator_preflight.schedule",
    "manifest.test_candidate",
    "manifest.difficulty_shift",
    "manifest.expert_diagnostic",
    "manifest.visualization",
    "init",
    "optimizer",
    "sampler.base",
    "sampler.extra_demo",
    "sampler.recovery",
    "collection.extra_demo",
    "collection.recovery",
    "validation",
    "evaluation",
)


class UnknownSeedComponentError(ValueError):
    """A seed was requested for a component outside the closed registry."""


def _validate_component(component: str) -> None:
    for registered in SEED_COMPONENTS:
        if component == registered or component.startswith(registered + "."):
            return
    raise UnknownSeedComponentError(
        f"seed component {component!r} is not in the registry; "
        f"registered roots: {', '.join(SEED_COMPONENTS)}"
    )


def derive_seed(root_seed: int, bundle_id: str, component: str) -> int:
    """Derive the stable named seed for one component of one pipeline bundle."""
    _validate_component(component)
    if _SEPARATOR in bundle_id:
        raise ValueError("bundle_id must not contain the field separator")
    payload = _SEPARATOR.join((str(int(root_seed)), bundle_id, component)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def seed_stream(root_seed: int, bundle_id: str, component: str) -> Iterator[int]:
    """Infinite stream of derived seeds ``component.0, component.1, ...``."""
    _validate_component(component)
    index = 0
    while True:
        yield derive_seed(root_seed, bundle_id, f"{component}.{index}")
        index += 1
