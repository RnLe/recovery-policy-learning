"""Action perturbation operators.

An operator is an immutable total derangement of the frozen action set:
defined for every action, bijective, and without fixed points, so a delivered
corruption always executes a different in-set action than the one proposed.
On the three-action movement set exactly two derangements exist (the two
3-cycles), which the tests pin by exhaustive enumeration: the collection and
unseen families are exactly these two operators, and each is the other's
inverse.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from grounded_recovery.config import OperatorConfig


class DerangementError(ValueError):
    """The mapping is not a total derangement of the action set."""


def validate_derangement(mapping: tuple[int, ...], action_ids: tuple[int, ...]) -> None:
    if len(mapping) != len(action_ids):
        raise DerangementError(
            f"mapping length {len(mapping)} does not match action count {len(action_ids)}"
        )
    if sorted(mapping) != sorted(action_ids):
        raise DerangementError(
            f"mapping {mapping} is not a permutation of the action set {action_ids}"
        )
    for source, target in zip(action_ids, mapping, strict=True):
        if source == target:
            raise DerangementError(f"fixed point at action {source}: not a derangement")


@dataclass(frozen=True)
class ActionDerangement:
    """Immutable total derangement; ``mapping[i]`` is the image of ``action_ids[i]``."""

    name: str
    action_ids: tuple[int, ...]
    mapping: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_derangement(self.mapping, self.action_ids)

    def apply(self, action: int) -> int:
        try:
            index = self.action_ids.index(int(action))
        except ValueError:
            raise DerangementError(
                f"action {action} is outside the operator domain {self.action_ids}"
            ) from None
        return self.mapping[index]


def enumerate_derangements(action_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """All derangements of the action set, by brute force over permutations."""
    found = []
    for permutation in itertools.permutations(action_ids):
        if all(
            source != target
            for source, target in zip(action_ids, permutation, strict=True)
        ):
            found.append(tuple(permutation))
    return tuple(found)


def operator_from_config(
    operator: OperatorConfig, action_ids: tuple[int, ...]
) -> ActionDerangement:
    return ActionDerangement(
        name=operator.name, action_ids=tuple(action_ids), mapping=tuple(operator.mapping)
    )
