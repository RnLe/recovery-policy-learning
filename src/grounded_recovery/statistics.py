"""Paired analysis: per-policy success, within-bundle contrasts, t interval.

The statistical unit is the complete pipeline bundle: per-policy success rates
are computed on the finite frozen panel first, then paired within-bundle
differences, then the mean and the prespecified paired t interval across
bundles. Episodes are repeated measures inside a bundle, never independent
replicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from grounded_recovery.evaluate import EvaluationRow


class AnalysisError(ValueError):
    """The evaluation rows violate an analysis precondition."""


@dataclass(frozen=True)
class SuccessSummary:
    successes: int
    assigned: int
    rate: float
    delivered: int


def success_summary(rows: list[EvaluationRow]) -> SuccessSummary:
    """Intention-to-treat success on assigned episodes; delivered reported beside."""
    if not rows:
        raise AnalysisError("no evaluation rows")
    assigned = len(rows)
    successes = sum(1 for row in rows if row.success)
    delivered = sum(1 for row in rows if row.delivered)
    return SuccessSummary(
        successes=successes,
        assigned=assigned,
        rate=successes / assigned,
        delivered=delivered,
    )


def paired_difference(
    rows_first: list[EvaluationRow], rows_second: list[EvaluationRow]
) -> float:
    """Within-bundle paired contrast: rate(first) - rate(second) on the same panel."""
    scenarios_first = [(row.scenario_ordinal, row.scheduled_time) for row in rows_first]
    scenarios_second = [(row.scenario_ordinal, row.scheduled_time) for row in rows_second]
    if scenarios_first != scenarios_second:
        raise AnalysisError(
            "paired contrast requires the identical ordered scenario panel and schedule"
        )
    return success_summary(rows_first).rate - success_summary(rows_second).rate


@dataclass(frozen=True)
class PairedTInterval:
    deltas: tuple[float, ...]
    mean: float
    lower: float
    upper: float
    level: float
    bundles: int


def paired_t_interval(deltas: list[float], level: float = 0.95) -> PairedTInterval:
    """Prespecified paired t interval across complete pipeline bundles.

    Requires at least two bundles (one paired difference has no dispersion
    estimate). With few bundles the interval is fragile by construction; every
    per-bundle point is therefore carried alongside the interval.
    """
    if len(deltas) < 2:
        raise AnalysisError(
            f"the paired t interval needs at least two complete bundles, got {len(deltas)}"
        )
    if not 0.0 < level < 1.0:
        raise AnalysisError("interval level must be in (0, 1)")
    bundles = len(deltas)
    mean = sum(deltas) / bundles
    variance = sum((delta - mean) ** 2 for delta in deltas) / (bundles - 1)
    standard_error = math.sqrt(variance / bundles)
    critical = float(stats.t.ppf(0.5 + level / 2.0, df=bundles - 1))
    return PairedTInterval(
        deltas=tuple(deltas),
        mean=mean,
        lower=mean - critical * standard_error,
        upper=mean + critical * standard_error,
        level=level,
        bundles=bundles,
    )


def crossed_cluster_bootstrap(
    success: dict[str, dict[str, list[bool]]],
    *,
    first_arm: str,
    second_arm: str,
    replicates: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, object]:
    """Crossed two-way cluster bootstrap of the paired contrast (sensitivity).

    ``success[bundle][arm]`` is the ordered per-scenario outcome vector on the
    common frozen panel. Each replicate resamples pipeline bundles and
    scenario indices independently, with the same draws applied to every arm
    and bundle, preserving the crossed structure. This is a labelled
    sensitivity analysis; it does not replace the paired t interval and cannot
    compensate for few pipeline bundles.
    """
    import numpy as np

    bundles = sorted(success)
    if len(bundles) < 2:
        raise AnalysisError("bootstrap needs at least two bundles")
    panel_sizes = {
        len(values)
        for by_arm in success.values()
        for values in by_arm.values()
    }
    if len(panel_sizes) != 1:
        raise AnalysisError("all arms and bundles must share one scenario panel")
    panel = panel_sizes.pop()
    outcome = {
        bundle: {
            arm: np.asarray(values, dtype=np.float64)
            for arm, values in by_arm.items()
        }
        for bundle, by_arm in success.items()
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bundle_draw = rng.integers(0, len(bundles), size=len(bundles))
        scenario_draw = rng.integers(0, panel, size=panel)
        deltas = []
        for bundle_index in bundle_draw:
            by_arm = outcome[bundles[bundle_index]]
            deltas.append(
                float(by_arm[first_arm][scenario_draw].mean())
                - float(by_arm[second_arm][scenario_draw].mean())
            )
        draws[index] = sum(deltas) / len(deltas)
    lower_q = (1.0 - level) / 2.0
    return {
        "method": "crossed_two_way_cluster_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "level": level,
        "lower": float(np.quantile(draws, lower_q)),
        "upper": float(np.quantile(draws, 1.0 - lower_q)),
        "mean": float(draws.mean()),
    }


# ---------------------------------------------------------------------------
# Descriptive audits.
#
# Everything below is descriptive and secondary. It is recomputed from the same
# immutable episode rows as the primary estimand and never changes it. The
# primary contrast stays the mean within-bundle difference in eligible unseen
# one-corruption intention-to-treat success.
# ---------------------------------------------------------------------------

FAILURE_SUCCESS = "success"
FAILURE_TRUNCATED = "step_limit_truncation"
FAILURE_TERMINATED = "terminated_without_goal"

FAILURE_CATEGORIES = (FAILURE_SUCCESS, FAILURE_TRUNCATED, FAILURE_TERMINATED)


def failure_category(row: EvaluationRow) -> str:
    """The terminal outcome category of one episode.

    Three categories are possible by construction: the goal was reached, the
    step limit was hit, or the episode terminated without the goal. Reporting a
    category with a zero count is deliberate; a composition that turns out to be
    degenerate is evidence, not a reason to omit the table.
    """
    if row.success:
        return FAILURE_SUCCESS
    if row.truncated:
        return FAILURE_TRUNCATED
    return FAILURE_TERMINATED


def failure_composition(rows: list[EvaluationRow]) -> dict[str, int]:
    """Counts per terminal outcome category, including categories with no rows."""
    if not rows:
        raise AnalysisError("no evaluation rows")
    counts = dict.fromkeys(FAILURE_CATEGORIES, 0)
    for row in rows:
        counts[failure_category(row)] += 1
    return counts


@dataclass(frozen=True)
class OverheadSummary:
    """Path cost relative to the oracle, conditioned on success.

    The ratio is only meaningful where the task was solved, so the denominator
    is the number of successful episodes and is always reported beside the
    estimate. A method that succeeds more often can still take longer paths when
    it does succeed, and that trade-off belongs next to the success rate.
    """

    successes: int
    assigned: int
    median_ratio: float | None
    mean_ratio: float | None


def overhead_summary(rows: list[EvaluationRow]) -> OverheadSummary:
    """Success-conditioned steps relative to the nominal oracle path length."""
    if not rows:
        raise AnalysisError("no evaluation rows")
    ratios = [
        row.steps / row.nominal_oracle_path_length
        for row in rows
        if row.success and row.nominal_oracle_path_length > 0
    ]
    if not ratios:
        return OverheadSummary(0, len(rows), None, None)
    ordered = sorted(ratios)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return OverheadSummary(
        successes=len(ratios),
        assigned=len(rows),
        median_ratio=median,
        mean_ratio=sum(ratios) / len(ratios),
    )


def delivery_summary(rows: list[EvaluationRow]) -> tuple[int, int]:
    """Assigned episodes and the subset where the scheduled corruption landed.

    Undelivered assignments stay in every denominator; this pair makes the gap
    between assignment and delivery visible rather than implicit.
    """
    if not rows:
        raise AnalysisError("no evaluation rows")
    return len(rows), sum(1 for row in rows if row.delivered)


def success_by_scheduled_time(
    rows: list[EvaluationRow],
) -> dict[int, SuccessSummary]:
    """Success split by the scheduled corruption time, for one arm and slice."""
    by_time: dict[int, list[EvaluationRow]] = {}
    for row in rows:
        if row.scheduled_time is None:
            continue
        by_time.setdefault(row.scheduled_time, []).append(row)
    return {time: success_summary(cell) for time, cell in sorted(by_time.items())}


def contrast_across_bundles(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    *,
    first_arm: str,
    second_arm: str,
    slice_name: str,
    level: float = 0.95,
) -> dict[str, object]:
    """One paired contrast across bundles, with its interval and every point.

    Used for the secondary contrasts. The statistical unit is the pipeline
    bundle, exactly as for the primary endpoint; only the arms and the slice
    differ. These contrasts are not prespecified and carry that label wherever
    they are shown.
    """
    deltas = [
        paired_difference(
            grouped[(bundle, first_arm, slice_name)],
            grouped[(bundle, second_arm, slice_name)],
        )
        for bundle in bundles
    ]
    interval = paired_t_interval(deltas, level=level) if len(deltas) >= 2 else None
    return {
        "first_arm": first_arm,
        "second_arm": second_arm,
        "slice": slice_name,
        "bundles": len(deltas),
        "per_bundle_deltas": dict(zip(bundles, deltas, strict=True)),
        "mean": interval.mean if interval else None,
        "lower": interval.lower if interval else None,
        "upper": interval.upper if interval else None,
        "level": level,
        "method": "paired_t",
        "status": "secondary_not_prespecified",
    }
