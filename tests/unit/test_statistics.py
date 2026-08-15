"""Paired analysis arithmetic against hand computations.

These tests protect the estimand: ITT denominators include undelivered
assignments, contrasts require the identical crossed panel, and the paired t
interval across bundles matches the textbook formula.
"""

from __future__ import annotations

import pytest

from grounded_recovery.evaluate import EvaluationRow
from grounded_recovery.statistics import (
    AnalysisError,
    paired_difference,
    paired_t_interval,
    success_summary,
)


def row(ordinal: int, *, success: bool, delivered: bool = True,
        scheduled: int | None = 2, arm: str = "recovery_aggregation") -> EvaluationRow:
    return EvaluationRow(
        bundle_id="B00",
        arm=arm,
        slice_name="unseen",
        scenario_ordinal=ordinal,
        environment_seed=1000 + ordinal,
        scenario_hash=f"{ordinal:064d}",
        scheduled_time=scheduled,
        operator_name="rot_minus",
        delivered=delivered,
        success=success,
        truncated=not success,
        steps=10,
        nominal_oracle_path_length=9,
        contract_hash="cc" * 32,
    )


def test_itt_keeps_undelivered_in_denominator() -> None:
    rows = [
        row(0, success=True),
        row(1, success=False, delivered=False),  # ended before the corruption
        row(2, success=False),
        row(3, success=True),
    ]
    summary = success_summary(rows)
    assert summary.assigned == 4
    assert summary.successes == 2
    assert summary.rate == pytest.approx(0.5)
    assert summary.delivered == 3


def test_paired_difference_requires_identical_panel() -> None:
    first = [row(0, success=True), row(1, success=False)]
    second = [row(0, success=False, arm="extra_demonstrations"),
              row(1, success=True, arm="extra_demonstrations")]
    assert paired_difference(first, second) == pytest.approx(0.0)
    mismatched = [row(0, success=False, arm="extra_demonstrations"),
                  row(2, success=True, arm="extra_demonstrations")]
    with pytest.raises(AnalysisError, match="identical ordered scenario panel"):
        paired_difference(first, mismatched)
    shifted = [row(0, success=False, scheduled=3),
               row(1, success=True, scheduled=2)]
    with pytest.raises(AnalysisError, match="identical ordered scenario panel"):
        paired_difference(first, shifted)


def test_paired_t_interval_hand_computation() -> None:
    # deltas 0.1, 0.3: mean 0.2, s^2 = 0.02, se = 0.1, t(df=1, 97.5%) = 12.7062.
    interval = paired_t_interval([0.1, 0.3])
    assert interval.mean == pytest.approx(0.2)
    assert interval.lower == pytest.approx(0.2 - 12.7062 * 0.1, abs=1e-3)
    assert interval.upper == pytest.approx(0.2 + 12.7062 * 0.1, abs=1e-3)
    assert interval.bundles == 2


def test_paired_t_interval_degenerate_and_zero_variance() -> None:
    with pytest.raises(AnalysisError, match="at least two"):
        paired_t_interval([0.2])
    flat = paired_t_interval([0.2, 0.2, 0.2])
    assert flat.mean == pytest.approx(0.2)
    assert flat.lower == pytest.approx(0.2)
    assert flat.upper == pytest.approx(0.2)
