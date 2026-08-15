"""Reveal-window, arm-partition, and fairness-audit invariants.

These tests protect the treatment and fairness definitions of the three-arm
design: recovery labels begin strictly after the corrupted transition, the
arms draw from disjoint scenario partitions, and any exposure/update
inequality between the two full-budget arms is an error, never a warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from grounded_recovery.data import ManifestError, reveal_window_mask
from grounded_recovery.integrity import IntegrityError, audit_round_fairness
from grounded_recovery.schemas import ManifestEntry


def revealed(mask: np.ndarray) -> list[int]:
    return list(np.flatnonzero(mask))


def test_first_recovery_label_is_post_corruption() -> None:
    mask = reveal_window_mask(episode_length=10, scheduled_time=2, h=3, budget_remaining=99)
    assert revealed(mask) == [3, 4, 5]
    assert not mask[2]  # the corrupted step itself is never a target


def test_window_truncated_by_termination() -> None:
    assert revealed(reveal_window_mask(4, 2, 3, 99)) == [3]


def test_window_truncated_by_budget_partial_final_window() -> None:
    assert revealed(reveal_window_mask(10, 2, 3, 2)) == [3, 4]
    assert revealed(reveal_window_mask(10, 2, 3, 1)) == [3]


def test_episode_ending_at_or_before_corruption_reveals_nothing() -> None:
    assert revealed(reveal_window_mask(3, 2, 3, 99)) == []
    assert revealed(reveal_window_mask(2, 2, 3, 99)) == []


def test_zero_budget_reveals_nothing() -> None:
    assert revealed(reveal_window_mask(10, 2, 3, 0)) == []


def test_negative_budget_rejected() -> None:
    with pytest.raises(ManifestError):
        reveal_window_mask(10, 2, 3, -1)


def _entry(ordinal: int) -> ManifestEntry:
    return ManifestEntry(
        split_name="collection",
        ordinal=ordinal,
        environment_id="env",
        environment_seed=1000 + ordinal,
        canonical_scenario_hash=f"{ordinal:064d}",
        mission="go to the red ball",
        nominal_oracle_path_length=9,
        perturbation_family=None,
        scheduled_intervention_times=(),
        manifest_version=1,
    )


def test_arm_partition_disjoint_and_covering() -> None:
    from grounded_recovery.experiment import ARM_EXTRA, ARM_RECOVERY, arm_partition

    entries = [_entry(i) for i in range(7)]
    extra = arm_partition(entries, ARM_EXTRA)
    recovery = arm_partition(entries, ARM_RECOVERY)
    extra_ordinals = {entry.ordinal for entry in extra}
    recovery_ordinals = {entry.ordinal for entry in recovery}
    assert extra_ordinals == {0, 2, 4, 6}
    assert recovery_ordinals == {1, 3, 5}
    assert extra_ordinals.isdisjoint(recovery_ordinals)
    with pytest.raises(ManifestError):
        arm_partition(entries, "bc_base")


def _ledger_row(update: int, *, new_drawn: int = 4) -> dict[str, object]:
    return {
        "round": 1,
        "update": update,
        "base_targets_drawn": 8,
        "new_targets_drawn": new_drawn,
        "cumulative_base_exposures": 8 * (update + 1),
        "cumulative_new_exposures": new_drawn * (update + 1),
        "loss_denominator": 8 + new_drawn,
        "optimizer_step": update + 1,
        "base_unique_available": 20,
        "new_unique_available": 3,
        "loss_sum": 1.0,
    }


def test_fairness_audit_passes_on_matched_ledgers() -> None:
    rows = [_ledger_row(update) for update in range(5)]
    report = audit_round_fairness(rows, list(rows), arm_a="extra", arm_b="recovery")
    assert report["updates"] == 5
    assert report["cumulative_new_exposures"] == 20


def test_fairness_audit_ignores_logged_only_differences() -> None:
    rows_a = [_ledger_row(update) for update in range(3)]
    rows_b = [dict(row, new_unique_available=99, loss_sum=5.0) for row in rows_a]
    audit_round_fairness(rows_a, rows_b, arm_a="extra", arm_b="recovery")


def test_fairness_audit_catches_unequal_draws() -> None:
    rows_a = [_ledger_row(update) for update in range(3)]
    rows_b = [_ledger_row(update) for update in range(3)]
    rows_b[1] = _ledger_row(1, new_drawn=5)
    with pytest.raises(IntegrityError, match="new_targets_drawn"):
        audit_round_fairness(rows_a, rows_b, arm_a="extra", arm_b="recovery")


def test_fairness_audit_catches_unequal_update_counts() -> None:
    rows_a = [_ledger_row(update) for update in range(3)]
    with pytest.raises(IntegrityError, match="update counts"):
        audit_round_fairness(rows_a, rows_a[:2], arm_a="extra", arm_b="recovery")
