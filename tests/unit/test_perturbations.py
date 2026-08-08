"""Perturbation operator invariants.

These tests protect the treatment definition: a delivered corruption must
always replace the proposal with a different in-set action (total derangement),
and on the three-action set the collection/unseen family structure is exactly
the two 3-cycles: a mathematical fact pinned here so that any change to the
frozen action set forces a conscious protocol revision.
"""

from __future__ import annotations

import pytest

from grounded_recovery.config import load_and_validate
from grounded_recovery.perturbations import (
    ActionDerangement,
    DerangementError,
    enumerate_derangements,
    operator_from_config,
    validate_derangement,
)

ACTIONS = (0, 1, 2)


def test_exactly_two_derangements_of_three_actions() -> None:
    derangements = enumerate_derangements(ACTIONS)
    assert set(derangements) == {(1, 2, 0), (2, 0, 1)}


def test_pilot_operators_are_the_two_cycles_and_mutually_inverse(tmp_path) -> None:
    from pathlib import Path

    cfg = load_and_validate(Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml")
    collection = operator_from_config(cfg.perturbation.collection_operator, ACTIONS)
    unseen = operator_from_config(cfg.perturbation.unseen_operator, ACTIONS)
    assert {collection.mapping, unseen.mapping} == set(enumerate_derangements(ACTIONS))
    # Composition is the identity: the unseen operator is the inverse of the
    # collection operator. This structural relation is a disclosed limitation.
    for action in ACTIONS:
        assert unseen.apply(collection.apply(action)) == action


def test_identity_rejected() -> None:
    with pytest.raises(DerangementError, match="fixed point"):
        validate_derangement((0, 1, 2), ACTIONS)


def test_partial_fixed_point_rejected() -> None:
    with pytest.raises(DerangementError, match="fixed point"):
        validate_derangement((1, 0, 2), ACTIONS)


def test_non_bijective_rejected() -> None:
    with pytest.raises(DerangementError, match="permutation"):
        validate_derangement((1, 1, 0), ACTIONS)


def test_wrong_domain_rejected() -> None:
    with pytest.raises(DerangementError, match="permutation"):
        validate_derangement((1, 2, 3), ACTIONS)


def test_wrong_length_rejected() -> None:
    with pytest.raises(DerangementError, match="length"):
        validate_derangement((1, 0), ACTIONS)


def test_apply_matches_mapping() -> None:
    operator = ActionDerangement("rot_plus", ACTIONS, (1, 2, 0))
    assert [operator.apply(action) for action in ACTIONS] == [1, 2, 0]
    assert all(operator.apply(action) != action for action in ACTIONS)


def test_apply_outside_domain_rejected() -> None:
    operator = ActionDerangement("rot_plus", ACTIONS, (1, 2, 0))
    with pytest.raises(DerangementError, match="domain"):
        operator.apply(5)


def test_invalid_operator_cannot_be_constructed() -> None:
    with pytest.raises(DerangementError):
        ActionDerangement("bad", ACTIONS, (0, 1, 2))
