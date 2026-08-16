"""Analysis exports: tables, figures, and mini-reports, from raw rows only.

Every number in a table, figure, or report text is recomputed here from the
stored raw episode rows through the frozen analysis functions; nothing is
hand-entered. Figures state their slice, unit, denominators, and evidence
status directly in the caption text.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grounded_recovery.artifacts import atomic_write_json, read_json, read_jsonl
from grounded_recovery.config import ExperimentConfig, contract_hash
from grounded_recovery.evaluate import EvaluationRow
from grounded_recovery.seeds import derive_seed
from grounded_recovery.statistics import (
    FAILURE_CATEGORIES,
    FAILURE_SUCCESS,
    FAILURE_TERMINATED,
    FAILURE_TRUNCATED,
    contrast_across_bundles,
    crossed_cluster_bootstrap,
    delivery_summary,
    failure_composition,
    overhead_summary,
    paired_t_interval,
    success_by_scheduled_time,
    success_summary,
)

ARM_COLORS = {
    "bc_base": "#8a8a8a",
    "extra_demonstrations": "#4878a8",
    "recovery_aggregation": "#2a9d8f",
}
ARM_LABELS = {
    "bc_base": "BC base (no added labels)",
    "extra_demonstrations": "Extra demonstrations",
    "recovery_aggregation": "Recovery aggregation",
}
SLICE_ORDER = ("clean", "matched", "unseen")


def rows_from_jsonl(path: Path) -> list[EvaluationRow]:
    return [EvaluationRow(**row) for row in read_jsonl(path)]


def group_rows(
    rows: list[EvaluationRow],
) -> dict[tuple[str, str, str], list[EvaluationRow]]:
    grouped: dict[tuple[str, str, str], list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.bundle_id, row.arm, row.slice_name)].append(row)
    for cell in grouped.values():
        cell.sort(key=lambda row: row.scenario_ordinal)
    return dict(grouped)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_success_matrix(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """Success by slice and arm with one point per pipeline bundle."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    arms = list(ARM_LABELS)
    for axis, slice_name in zip(axes, SLICE_ORDER, strict=True):
        for arm_index, arm in enumerate(arms):
            rates = []
            for bundle in bundles:
                cell = grouped.get((bundle, arm, slice_name))
                if cell:
                    rates.append(success_summary(cell).rate)
            if not rates:
                continue
            mean_rate = sum(rates) / len(rates)
            axis.bar(
                arm_index, mean_rate, width=0.62, color=ARM_COLORS[arm],
                alpha=0.55, zorder=1,
            )
            axis.scatter(
                [arm_index] * len(rates), rates, color=ARM_COLORS[arm],
                edgecolor="black", linewidth=0.5, s=26, zorder=3,
            )
        denominator = 0
        any_cell = next(
            (grouped[(b, arms[0], slice_name)] for b in bundles
             if (b, arms[0], slice_name) in grouped), None,
        )
        if any_cell:
            denominator = len(any_cell)
        axis.set_title(f"{slice_name} (n={denominator} scenarios)", fontsize=10)
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(["base", "extra", "recovery"], fontsize=9)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("ITT success rate")
    fig.suptitle(
        f"Success by slice and arm, {status_label}; bars = mean over "
        f"{len(bundles)} bundle(s), points = individual bundles",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_effect(
    deltas: list[float],
    bundles: list[str],
    interval,
    sesoi: float,
    out_path: Path,
    *,
    status_label: str,
    slice_name: str = "unseen",
) -> None:
    """Primary paired-effect display: every bundle point, mean, interval."""
    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.axvline(sesoi, color="#c07c00", linestyle="--", linewidth=1.0,
                 label=f"SESOI +{sesoi:.2f}")
    axis.axvline(-sesoi, color="#c07c00", linestyle=":", linewidth=0.8)
    for index, (bundle, delta) in enumerate(zip(bundles, deltas, strict=True)):
        axis.scatter(delta, index, color="#2a9d8f", s=40, zorder=3)
        axis.annotate(bundle, (delta, index), textcoords="offset points",
                      xytext=(6, -3), fontsize=8)
    if interval is not None:
        y_mean = len(deltas) + 0.6
        axis.errorbar(
            [interval.mean], [y_mean],
            xerr=[[interval.mean - interval.lower], [interval.upper - interval.mean]],
            fmt="D", color="black", capsize=4, markersize=6,
            label=f"mean {interval.mean:+.3f} "
                  f"[{interval.lower:+.3f}, {interval.upper:+.3f}] "
                  f"(95% paired t, R={interval.bundles})",
        )
        axis.set_ylim(-0.8, y_mean + 0.9)
    axis.set_yticks([])
    axis.set_xlabel(
        f"paired difference in {slice_name} ITT success "
        "(recovery aggregation − extra demonstrations)"
    )
    axis.set_title(f"Primary paired contrast per pipeline bundle, {status_label}",
                   fontsize=10)
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curve(metrics_paths: dict[str, Path], out_path: Path,
                        *, status_label: str) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 3.0))
    for label, path in metrics_paths.items():
        rows = read_jsonl(path)
        updates = [row["update"] for row in rows]
        losses = [row.get("loss", row.get("loss_sum", 0) / max(row.get("loss_denominator", 1), 1))
                  for row in rows]
        axis.plot(updates, losses, linewidth=1.0, label=label)
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("masked cross-entropy")
    axis.set_yscale("log")
    axis.set_title(f"Training loss, {status_label}", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Descriptive audits.
#
# These exports are secondary and descriptive. They recompute from the same
# immutable episode rows and ledgers as the primary estimand and never alter it.
# Each table and figure states its denominator and carries the secondary label.
# ---------------------------------------------------------------------------

SECONDARY_LABEL = "SECONDARY, not prespecified"


def write_failure_composition(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    out_path: Path,
) -> dict[str, dict[str, int]]:
    """Terminal outcome composition per bundle, arm, and slice.

    Every category is written even when it has no rows. A degenerate
    composition is a finding about the environment, not a reason to omit it.
    """
    rows: list[list[object]] = []
    pooled: dict[str, dict[str, int]] = {}
    for slice_name in SLICE_ORDER:
        for arm in arms:
            totals = dict.fromkeys(FAILURE_CATEGORIES, 0)
            for bundle in bundles:
                cell = grouped.get((bundle, arm, slice_name))
                if not cell:
                    continue
                counts = failure_composition(cell)
                for category, value in counts.items():
                    totals[category] += value
                rows.append(
                    [bundle, arm, slice_name,
                     counts[FAILURE_SUCCESS], counts[FAILURE_TRUNCATED],
                     counts[FAILURE_TERMINATED], len(cell)]
                )
            pooled[f"{arm}|{slice_name}"] = totals
    _write_csv(
        out_path,
        ["bundle", "arm", "slice", "success", "step_limit_truncation",
         "terminated_without_goal", "assigned"],
        rows,
    )
    return pooled


def write_overhead_table(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    out_path: Path,
) -> dict[str, dict[str, float | int | None]]:
    """Success-conditioned path cost relative to the oracle, with denominators."""
    rows: list[list[object]] = []
    pooled: dict[str, dict[str, float | int | None]] = {}
    for slice_name in SLICE_ORDER:
        for arm in arms:
            cells = [
                grouped[(bundle, arm, slice_name)]
                for bundle in bundles
                if (bundle, arm, slice_name) in grouped
            ]
            for bundle, cell in zip(bundles, cells, strict=False):
                summary = overhead_summary(cell)
                rows.append([
                    bundle, arm, slice_name, summary.successes, summary.assigned,
                    None if summary.median_ratio is None
                    else round(summary.median_ratio, 6),
                    None if summary.mean_ratio is None
                    else round(summary.mean_ratio, 6),
                ])
            if cells:
                flat = [row for cell in cells for row in cell]
                summary = overhead_summary(flat)
                pooled[f"{arm}|{slice_name}"] = {
                    "successes": summary.successes,
                    "assigned": summary.assigned,
                    "median_ratio": summary.median_ratio,
                    "mean_ratio": summary.mean_ratio,
                }
    _write_csv(
        out_path,
        ["bundle", "arm", "slice", "successes", "assigned", "median_step_ratio",
         "mean_step_ratio"],
        rows,
    )
    return pooled


def write_delivery_table(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    out_path: Path,
) -> dict[str, dict[str, int]]:
    """Assigned against delivered corruptions, per bundle, arm, and slice."""
    rows: list[list[object]] = []
    pooled: dict[str, dict[str, int]] = {}
    for slice_name in SLICE_ORDER:
        if slice_name == "clean":
            continue
        for arm in arms:
            total_assigned = 0
            total_delivered = 0
            for bundle in bundles:
                cell = grouped.get((bundle, arm, slice_name))
                if not cell:
                    continue
                assigned, delivered = delivery_summary(cell)
                total_assigned += assigned
                total_delivered += delivered
                rows.append([
                    bundle, arm, slice_name, assigned, delivered,
                    round(delivered / assigned, 6),
                ])
            pooled[f"{arm}|{slice_name}"] = {
                "assigned": total_assigned, "delivered": total_delivered,
            }
    _write_csv(
        out_path,
        ["bundle", "arm", "slice", "assigned", "delivered", "delivery_rate"],
        rows,
    )
    return pooled


def write_time_profile(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    out_path: Path,
) -> dict[str, dict[int, tuple[int, int]]]:
    """Success split by the scheduled corruption time, pooled across bundles."""
    rows: list[list[object]] = []
    pooled: dict[str, dict[int, tuple[int, int]]] = {}
    for slice_name in SLICE_ORDER:
        if slice_name == "clean":
            continue
        for arm in arms:
            flat = [
                row
                for bundle in bundles
                for row in grouped.get((bundle, arm, slice_name), [])
            ]
            if not flat:
                continue
            profile = success_by_scheduled_time(flat)
            pooled[f"{arm}|{slice_name}"] = {
                time: (summary.successes, summary.assigned)
                for time, summary in profile.items()
            }
            for time, summary in profile.items():
                rows.append([
                    arm, slice_name, time, summary.successes, summary.assigned,
                    round(summary.rate, 6),
                ])
    _write_csv(
        out_path,
        ["arm", "slice", "scheduled_time", "successes", "assigned", "success_rate"],
        rows,
    )
    return pooled


SECONDARY_CONTRASTS = (
    ("recovery_aggregation", "extra_demonstrations", "clean"),
    ("recovery_aggregation", "extra_demonstrations", "matched"),
    ("extra_demonstrations", "bc_base", "unseen"),
    ("recovery_aggregation", "bc_base", "unseen"),
)


def write_secondary_contrasts(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    out_path: Path,
) -> list[dict[str, object]]:
    """Paired contrasts beside the primary one, on the same statistical unit.

    These answer the two questions a reader asks next: does the recovery
    advantage also appear without any corruption, and is the extra
    demonstration arm itself separated from the untouched base? Both are
    secondary and not prespecified.
    """
    contrasts = [
        contrast_across_bundles(
            grouped, bundles,
            first_arm=first, second_arm=second, slice_name=slice_name,
        )
        for first, second, slice_name in SECONDARY_CONTRASTS
        if all((bundle, first, slice_name) in grouped for bundle in bundles)
        and all((bundle, second, slice_name) in grouped for bundle in bundles)
    ]
    _write_csv(
        out_path,
        ["first_arm", "second_arm", "slice", "bundles", "mean", "lower", "upper",
         "status"],
        [
            [c["first_arm"], c["second_arm"], c["slice"], c["bundles"],
             None if c["mean"] is None else round(c["mean"], 6),
             None if c["lower"] is None else round(c["lower"], 6),
             None if c["upper"] is None else round(c["upper"], 6),
             c["status"]]
            for c in contrasts
        ],
    )
    return contrasts


def write_budget_exposure(
    cfg: ExperimentConfig,
    data_root: Path,
    bundles: list[str],
    out_path: Path,
) -> dict[str, object]:
    """The acquisition audit: what is matched exactly beside what is only logged.

    The two full-budget arms reveal the same number of oracle targets and take
    the same number of optimizer updates on the same target mix. What they
    cannot share is the cost of acquisition: recovery labels come from learner
    rollouts, so most oracle recommendations produced along the way fall outside
    the reveal window and are discarded. That asymmetry is the honest price of
    the method and is reported rather than called equal.
    """
    data_root = Path(data_root)
    bundle_root = data_root / contract_hash(cfg)[:12]
    rows: list[list[object]] = []
    matched: dict[str, dict[str, int]] = {}
    logged: dict[str, dict[str, int]] = {}
    for bundle in bundles:
        summary_path = bundle_root / bundle / "bundle_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        for arm, payload in sorted(summary["arms"].items()):
            cumulative = payload["cumulative"]
            matched.setdefault(arm, {"base": 0, "new": 0, "updates": 0})
            for key in ("base", "new", "updates"):
                matched[arm][key] += int(cumulative[key])
            logged.setdefault(
                arm,
                {"oracle_calls": 0, "simulator_steps": 0,
                 "discarded_recommendations": 0, "episodes": 0},
            )
            for collection in payload["collections"]:
                for key in logged[arm]:
                    logged[arm][key] += int(collection.get(key, 0))
                rows.append([
                    bundle, arm, collection["round_index"],
                    collection["revealed_targets"], collection["oracle_calls"],
                    collection["simulator_steps"],
                    collection["discarded_recommendations"],
                    collection["episodes"],
                    cumulative["base"], cumulative["new"], cumulative["updates"],
                ])
    _write_csv(
        out_path,
        ["bundle", "arm", "round", "revealed_targets", "oracle_calls",
         "simulator_steps", "discarded_recommendations", "episodes",
         "cumulative_base_exposures", "cumulative_new_exposures",
         "cumulative_updates"],
        rows,
    )
    exposures_equal = len({tuple(sorted(v.items())) for v in matched.values()}) <= 1
    return {
        "bundles_audited": len({row[0] for row in rows}),
        "matched": matched,
        "logged": logged,
        "exposures_equal_across_arms": exposures_equal,
    }


def plot_failure_composition(
    pooled: dict[str, dict[str, int]],
    arms: list[str],
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """Stacked terminal outcomes per arm on the primary slice."""
    fig, axis = plt.subplots(figsize=(8.4, 3.2))
    labels = ["reached the goal", "hit the step limit", "ended without the goal"]
    colors = ["#2a9d8f", "#c07c00", "#a03030"]
    positions = range(len(arms))
    bottoms = [0.0] * len(arms)
    totals = [sum(pooled[f"{arm}|unseen"].values()) for arm in arms]
    for category, label, color in zip(FAILURE_CATEGORIES, labels, colors, strict=True):
        values = [
            pooled[f"{arm}|unseen"][category] / total if total else 0.0
            for arm, total in zip(arms, totals, strict=True)
        ]
        axis.bar(positions, values, bottom=bottoms, width=0.6, color=color,
                 label=label, alpha=0.85)
        for position, value, base, arm in zip(
            positions, values, bottoms, arms, strict=True
        ):
            if value < 0.05:
                continue
            axis.annotate(
                f"{pooled[f'{arm}|unseen'][category]}",
                (position, base + value / 2), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    axis.set_xticks(list(positions))
    axis.set_xticklabels(["base", "extra", "recovery"], fontsize=9)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share of assigned episodes")
    axis.set_title(
        f"Terminal outcomes on the unseen slice, {status_label}\n"
        f"n={totals[0]} assigned episodes per arm, pooled over bundles",
        fontsize=9,
    )
    axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recovery_profile(
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    time_profile: dict[str, dict[int, tuple[int, int]]],
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """Success by corruption time beside the success-conditioned path cost."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    left, right = axes
    for arm in arms:
        profile = time_profile.get(f"{arm}|unseen", {})
        if not profile:
            continue
        times = sorted(profile)
        rates = [profile[t][0] / profile[t][1] for t in times]
        left.plot(times, rates, marker="o", color=ARM_COLORS[arm],
                  label=ARM_LABELS[arm], linewidth=1.4)
    left.set_xlabel("scheduled corruption time")
    left.set_ylabel("ITT success rate")
    left.set_ylim(0, 1)
    all_times = sorted({t for profile in time_profile.values() for t in profile})
    if all_times:
        left.set_xticks(all_times)
    left.grid(alpha=0.25)
    left.legend(fontsize=7)
    left.set_title("Success by corruption time, unseen slice", fontsize=9)

    data = []
    for arm in arms:
        ratios = [
            row.steps / row.nominal_oracle_path_length
            for bundle in bundles
            for row in grouped.get((bundle, arm, "unseen"), [])
            if row.success and row.nominal_oracle_path_length > 0
        ]
        data.append(ratios)
    parts = right.boxplot(data, showfliers=False, widths=0.55, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 1.3})
    for patch, arm in zip(parts["boxes"], arms, strict=True):
        patch.set_facecolor(ARM_COLORS[arm])
        patch.set_alpha(0.55)
    right.set_xticklabels(["base", "extra", "recovery"], fontsize=9)
    right.set_ylabel("steps / nominal oracle path length")
    right.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    right.grid(axis="y", alpha=0.25)
    right.set_title(
        "Path cost among successes only, denominators "
        + ", ".join(str(len(d)) for d in data),
        fontsize=9,
    )
    fig.suptitle(f"Recovery profile, {status_label}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intervention_delivery(
    delivery: dict[str, dict[str, int]],
    arms: list[str],
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """Assignments whose corruption never landed, which stay in the denominator.

    Plotting the delivered counts against the assigned counts would show two
    almost identical bars and hide the only quantity that matters here. What
    intention-to-treat accounting turns on is the small number of assignments
    that were never delivered, so that is what is drawn.
    """
    slices = [name for name in ("matched", "unseen")
              if all(f"{arm}|{name}" in delivery for arm in arms)]
    if not slices:
        return
    fig, axis = plt.subplots(figsize=(7.6, 3.0))
    width = 0.36
    offsets = [(index - (len(slices) - 1) / 2) * width for index in range(len(slices))]
    hatches = ["", "//"]
    assigned_total = 0
    for offset, slice_name, hatch in zip(offsets, slices, hatches, strict=False):
        counts = []
        for arm in arms:
            cell = delivery[f"{arm}|{slice_name}"]
            assigned_total = max(assigned_total, cell["assigned"])
            counts.append(cell["assigned"] - cell["delivered"])
        positions = [index + offset for index in range(len(arms))]
        axis.bar(positions, counts, width=width, label=slice_name,
                 color=[ARM_COLORS[arm] for arm in arms], alpha=0.85,
                 hatch=hatch, edgecolor="white")
        for position, value in zip(positions, counts, strict=True):
            axis.annotate(str(value), (position, value), ha="center", va="bottom",
                          fontsize=8, xytext=(0, 2), textcoords="offset points")
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(["base", "extra", "recovery"], fontsize=9)
    axis.set_ylabel("assignments never delivered")
    axis.set_title(
        f"Undelivered corruptions, {status_label}\n"
        f"out of {assigned_total} assigned episodes per arm and slice; "
        "every one stays in the denominator",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0", hatch=hatch,
                      edgecolor="white")
        for hatch in hatches[: len(slices)]
    ]
    axis.legend(handles, slices, fontsize=8, title="slice", title_fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_secondary_contrasts(
    contrasts: list[dict[str, object]],
    sesoi: float,
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """Every secondary paired contrast with its interval and per-bundle points."""
    usable = [c for c in contrasts if c["mean"] is not None]
    if not usable:
        return
    fig, axis = plt.subplots(figsize=(7.6, 0.62 * len(usable) + 1.5))
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.axvline(sesoi, color="#c07c00", linestyle="--", linewidth=0.9)
    labels = []
    for index, contrast in enumerate(usable):
        deltas = list(contrast["per_bundle_deltas"].values())
        axis.scatter(deltas, [index] * len(deltas), color="#8a8a8a", s=18, zorder=2)
        axis.errorbar(
            [contrast["mean"]], [index],
            xerr=[[contrast["mean"] - contrast["lower"]],
                  [contrast["upper"] - contrast["mean"]]],
            fmt="D", color="black", capsize=4, markersize=5, zorder=3,
        )
        short = {"bc_base": "base", "extra_demonstrations": "extra",
                 "recovery_aggregation": "recovery"}
        labels.append(
            f"{short[contrast['first_arm']]} - {short[contrast['second_arm']]}"
            f"  ({contrast['slice']})"
        )
    axis.set_yticks(range(len(usable)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("paired difference in ITT success rate")
    axis.set_ylim(-0.6, len(usable) - 0.4)
    axis.set_title(
        f"Secondary paired contrasts, {SECONDARY_LABEL}\n"
        f"{status_label}; points are pipeline bundles, "
        f"bars are 95% paired t intervals",
        fontsize=8.5,
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _annotate_counts(axis, positions, values) -> None:
    """Write the exact count above each bar so equality is readable, not inferred."""
    for position, value in zip(positions, values, strict=True):
        axis.annotate(
            f"{value:,}", (position, value), ha="center", va="bottom",
            fontsize=7, xytext=(0, 2), textcoords="offset points",
        )


def plot_budget_exposure(
    audit: dict[str, object],
    out_path: Path,
    *,
    status_label: str,
) -> None:
    """What the two full-budget arms share exactly, beside what they do not."""
    matched = audit["matched"]
    logged = audit["logged"]
    arms = [a for a in ("extra_demonstrations", "recovery_aggregation") if a in matched]
    if not arms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    left, right = axes
    matched_keys = ["base", "new", "updates"]
    matched_names = ["base target exposures", "new target exposures",
                     "optimizer updates"]
    width = 0.36
    for offset, arm in zip((-width / 2, width / 2), arms, strict=False):
        values = [matched[arm][key] for key in matched_keys]
        positions = [i + offset for i in range(len(matched_keys))]
        left.bar(positions, values, width=width, color=ARM_COLORS[arm],
                 label=ARM_LABELS[arm], alpha=0.85)
        _annotate_counts(left, positions, values)
    left.set_yscale("symlog")
    left.set_xticks(range(len(matched_keys)))
    left.set_xticklabels(matched_names, fontsize=8)
    left.set_ylabel("count, summed over bundles")
    left.set_title("Matched exactly", fontsize=9)
    left.legend(fontsize=7)
    left.grid(axis="y", alpha=0.25)

    logged_keys = ["oracle_calls", "simulator_steps", "discarded_recommendations"]
    logged_names = ["oracle calls", "simulator steps", "discarded recommendations"]
    for offset, arm in zip((-width / 2, width / 2), arms, strict=False):
        values = [logged[arm][key] for key in logged_keys]
        positions = [i + offset for i in range(len(logged_keys))]
        right.bar(positions, values, width=width, color=ARM_COLORS[arm],
                  alpha=0.85)
        _annotate_counts(right, positions, values)
    right.set_yscale("symlog")
    right.set_xticks(range(len(logged_keys)))
    right.set_xticklabels(logged_names, fontsize=8)
    right.set_title("Logged, deliberately not matched", fontsize=9)
    right.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Budget and exposure audit, {status_label}; "
        f"{audit['bundles_audited']} bundles, equal revealed-label budget per arm",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_descriptive_audits(
    cfg: ExperimentConfig,
    results_dir: Path,
    grouped: dict[tuple[str, str, str], list[EvaluationRow]],
    bundles: list[str],
    arms: list[str],
    *,
    status_label: str,
    data_root: Path,
) -> dict[str, object]:
    """All descriptive audit tables and figures for one opening.

    Returns the pooled summary that the report and the public bundle quote, so
    no consumer recomputes a number of its own.
    """
    tables = results_dir / "tables"
    figures = results_dir / "figures"
    failures = write_failure_composition(
        grouped, bundles, arms, tables / "failure_composition.csv"
    )
    overhead = write_overhead_table(
        grouped, bundles, arms, tables / "recovery_overhead.csv"
    )
    delivery = write_delivery_table(
        grouped, bundles, arms, tables / "intervention_delivery.csv"
    )
    time_profile = write_time_profile(
        grouped, bundles, arms, tables / "corruption_time_profile.csv"
    )
    contrasts = write_secondary_contrasts(
        grouped, bundles, tables / "secondary_contrasts.csv"
    )
    budget = write_budget_exposure(
        cfg, data_root, bundles, tables / "budget_exposure.csv"
    )

    plot_failure_composition(
        failures, arms, figures / "failure_composition.png",
        status_label=status_label,
    )
    plot_recovery_profile(
        grouped, bundles, arms, time_profile, figures / "recovery_profile.png",
        status_label=status_label,
    )
    plot_intervention_delivery(
        delivery, arms, figures / "intervention_delivery.png",
        status_label=status_label,
    )
    plot_secondary_contrasts(
        contrasts, cfg.study.sesoi_absolute_success,
        figures / "secondary_contrasts.png", status_label=status_label,
    )
    if budget["bundles_audited"]:
        plot_budget_exposure(
            budget, figures / "budget_exposure.png", status_label=status_label
        )
    return {
        "status": SECONDARY_LABEL,
        "failure_composition": failures,
        "overhead": overhead,
        "delivery": delivery,
        "corruption_time_profile": {
            key: {str(t): list(v) for t, v in profile.items()}
            for key, profile in time_profile.items()
        },
        "secondary_contrasts": contrasts,
        "budget_exposure": budget,
    }


def audit_results(
    cfg: ExperimentConfig,
    results_dir: Path,
    *,
    data_root: Path = Path("data"),
) -> dict[str, object]:
    """Descriptive audits over an opening that has already been analyzed.

    Deliberately a separate entry point from :func:`analyze_results`. The
    confirmatory summary is written once and never rewritten, so the audits
    are computed in their own pass, from the same immutable episode rows, and
    land in their own file. Nothing here can move the primary estimand.
    """
    results_dir = Path(results_dir)
    summary = read_json(results_dir / "statistical_summary.json")
    rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
    grouped = group_rows(rows)
    bundles = sorted({row.bundle_id for row in rows})
    arms = sorted({row.arm for row in rows})
    status_label = f"{summary['analysis_status'].upper()}, eligible unseen ITT"

    # The primary figures are regenerated here too. They are a rendering of the
    # stored summary, not part of it, so a presentation change must not require
    # reopening the analysis; the values they draw come from the same file the
    # release integrity check re-derives from the raw rows.
    deltas = [summary["per_bundle_deltas"][bundle] for bundle in bundles]
    interval = paired_t_interval(deltas) if len(deltas) >= 2 else None
    plot_paired_effect(
        deltas, bundles, interval, summary["sesoi_absolute_success"],
        results_dir / "figures" / "primary_paired_effect.png",
        status_label=status_label,
    )
    plot_success_matrix(
        grouped, bundles, results_dir / "figures" / "success_matrix.png",
        status_label=status_label,
    )

    audits = write_descriptive_audits(
        cfg, results_dir, grouped, bundles, arms,
        status_label=status_label, data_root=data_root,
    )
    audits["source_summary_status"] = summary["analysis_status"]
    atomic_write_json(
        results_dir / "descriptive_audits.json", audits, overwrite=True
    )
    return audits


def analyze_results(
    cfg: ExperimentConfig,
    results_dir: Path,
    *,
    planned_r_train: int,
) -> dict[str, object]:
    """Frozen analysis of one confirmatory opening, from raw rows only."""
    results_dir = Path(results_dir)
    rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
    grouped = group_rows(rows)
    bundles = sorted({row.bundle_id for row in rows})
    arms = sorted({row.arm for row in rows})

    pipeline_rows: list[list[object]] = []
    for bundle in bundles:
        for arm in arms:
            for slice_name in SLICE_ORDER:
                cell = grouped[(bundle, arm, slice_name)]
                summary = success_summary(cell)
                pipeline_rows.append(
                    [bundle, arm, slice_name, summary.successes, summary.assigned,
                     round(summary.rate, 6), summary.delivered]
                )
    _write_csv(
        results_dir / "tables" / "pipeline_metrics.csv",
        ["bundle", "arm", "slice", "successes", "assigned", "success_rate",
         "delivered"],
        pipeline_rows,
    )

    deltas = []
    paired_rows = []
    for bundle in bundles:
        recovery = success_summary(grouped[(bundle, "recovery_aggregation", "unseen")])
        extra = success_summary(grouped[(bundle, "extra_demonstrations", "unseen")])
        delta = recovery.rate - extra.rate
        deltas.append(delta)
        paired_rows.append(
            [bundle, "unseen", round(recovery.rate, 6), round(extra.rate, 6),
             round(delta, 6)]
        )
    _write_csv(
        results_dir / "tables" / "paired_effects.csv",
        ["bundle", "slice", "recovery_rate", "extra_demo_rate", "paired_difference"],
        paired_rows,
    )

    interval = paired_t_interval(deltas) if len(deltas) >= 2 else None
    success_vectors = {
        bundle: {
            arm: [row.success for row in grouped[(bundle, arm, "unseen")]]
            for arm in ("recovery_aggregation", "extra_demonstrations")
        }
        for bundle in bundles
    }
    bootstrap = None
    if len(bundles) >= 2:
        bootstrap = crossed_cluster_bootstrap(
            success_vectors,
            first_arm="recovery_aggregation",
            second_arm="extra_demonstrations",
            replicates=cfg.evaluation.bootstrap_replicates,
            seed=derive_seed(cfg.seeds.root_seed, "global", "evaluation.bootstrap"),
        )

    sesoi = cfg.study.sesoi_absolute_success
    if interval is None:
        claim_state = "inconclusive"
    elif interval.lower > 0:
        claim_state = "support"
    elif interval.upper < 0:
        claim_state = "adverse"
    elif interval.upper < sesoi and interval.lower > -sesoi:
        claim_state = "rule_out"
    else:
        claim_state = "inconclusive"

    completed = len(bundles)
    analysis_status = (
        "confirmatory" if completed >= max(5, planned_r_train) else "exploratory_pilot"
    )
    desired = cfg.evaluation.desired_interval_half_width
    achieved_half_width = (
        (interval.upper - interval.lower) / 2 if interval is not None else None
    )
    summary = {
        "estimand": (
            "mean within-bundle difference in eligible unseen one-corruption "
            "intention-to-treat success, recovery aggregation minus extra "
            "demonstrations, conditional on the frozen scenario panel"
        ),
        "primary_slice": "unseen",
        "analysis_status": analysis_status,
        "claim_state": claim_state,
        "claim_decision_rule_applied": (
            "support iff lower>0; adverse iff upper<0; rule_out iff upper<SESOI "
            "and lower>-SESOI; inconclusive otherwise"
        ),
        "bundles_completed": completed,
        "planned_r_train": planned_r_train,
        "scenario_denominator": len(next(iter(grouped.values()))),
        "mean_paired_difference": interval.mean if interval else None,
        "interval": (
            {"method": "paired_t", "level": 0.95, "lower": interval.lower,
             "upper": interval.upper}
            if interval else None
        ),
        "per_bundle_deltas": dict(zip(bundles, deltas, strict=True)),
        "sesoi_absolute_success": sesoi,
        "precision": {
            "desired_half_width": desired,
            "achieved_half_width": achieved_half_width,
            "target_met": (
                achieved_half_width is not None
                and desired is not None
                and achieved_half_width <= desired
            ),
        },
        "sensitivity_bootstrap": bootstrap,
        "contract_hash": contract_hash(cfg),
    }
    atomic_write_json(results_dir / "statistical_summary.json", summary)

    status_label = f"{analysis_status.upper()}, eligible unseen ITT"
    plot_paired_effect(
        deltas, bundles, interval, sesoi,
        results_dir / "figures" / "primary_paired_effect.png",
        status_label=status_label,
    )
    plot_success_matrix(
        grouped, bundles, results_dir / "figures" / "success_matrix.png",
        status_label=status_label,
    )
    return summary


def write_markdown_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pilot_summary(
    cfg: ExperimentConfig,
    pilot_roots: dict[str, Path],
    out_dir: Path,
) -> dict[str, object]:
    """Aggregate validation-pilot evidence across pilot bundles (G4/G6).

    Reads each pilot bundle's ``pilot_report.json`` and training ledgers,
    renders the success matrix, paired deltas, and base learning curves, and
    writes a markdown mini-report. Validation evidence only, never test data.
    """
    out_dir = Path(out_dir)
    reports = {
        bundle: read_json(Path(root) / "pilot_report.json")
        for bundle, root in pilot_roots.items()
    }
    bundles = sorted(reports)
    panel = reports[bundles[0]]["panel_scenarios"]

    # Success matrix and paired deltas from the stored evaluation rows.
    all_rows = []
    for root in pilot_roots.values():
        all_rows.extend(rows_from_jsonl(Path(root) / "validation_evaluation_rows.jsonl"))
    grouped = group_rows(all_rows)
    plot_success_matrix(
        grouped, bundles, out_dir / "figures" / "pilot_success_matrix.png",
        status_label="VALIDATION PILOT (not the primary endpoint)",
    )
    deltas = [reports[b]["paired_recovery_minus_extra"]["unseen"] for b in bundles]
    interval = paired_t_interval(deltas) if len(deltas) >= 2 else None
    plot_paired_effect(
        deltas, bundles, interval, cfg.study.sesoi_absolute_success,
        out_dir / "figures" / "pilot_paired_effect.png",
        status_label="VALIDATION PILOT (not the primary endpoint)",
    )
    plot_learning_curve(
        {
            f"{bundle} base": Path(root) / "training" / "base" / "metrics.jsonl"
            for bundle, root in pilot_roots.items()
        },
        out_dir / "figures" / "pilot_base_learning.png",
        status_label="VALIDATION PILOT",
    )

    def cell(bundle: str, arm: str, slice_name: str) -> str:
        values = reports[bundle]["success"][arm][slice_name]
        return f"{values['successes']}/{values['assigned']}"

    table_lines = ["| bundle | arm | clean | matched | unseen |",
                   "| --- | --- | --- | --- | --- |"]
    for bundle in bundles:
        for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation"):
            table_lines.append(
                f"| {bundle} | {arm} | {cell(bundle, arm, 'clean')} | "
                f"{cell(bundle, arm, 'matched')} | {cell(bundle, arm, 'unseen')} |"
            )
    deltas_lines = [
        f"- {bundle}: unseen paired delta (recovery − extra) = "
        f"{reports[bundle]['paired_recovery_minus_extra']['unseen']:+.4f}"
        for bundle in bundles
    ]
    interval_text = (
        f"mean {interval.mean:+.4f}, 95% paired t "
        f"[{interval.lower:+.4f}, {interval.upper:+.4f}] across {interval.bundles} "
        "pilot bundles"
        if interval
        else "single pilot bundle: no interval"
    )
    summary = {
        "bundles": bundles,
        "panel_scenarios": panel,
        "unseen_deltas": dict(zip(bundles, deltas, strict=True)),
        "interval": interval_text,
    }
    write_markdown_report(
        out_dir / "pilot_report.md",
        "Validation pilot mini-report (tuning evidence only)",
        [
            ("Scope",
             "All numbers on this page are from the validation split under the pilot "
             "configuration. They exist to choose and freeze design values (G4/G6) "
             "and are not the primary endpoint. The eligible unseen test panel has "
             "not been touched."),
            ("Success by slice and arm",
             "\n".join(table_lines)
             + f"\n\nPanel: {panel} validation scenarios per cell, intention-to-treat."
             + "\n\n![success matrix](figures/pilot_success_matrix.png)"),
            ("Paired unseen contrast per pilot bundle",
             "\n".join(deltas_lines) + f"\n\n{interval_text}"
             + "\n\n![paired effect](figures/pilot_paired_effect.png)"),
            ("Base training",
             "![learning curves](figures/pilot_base_learning.png)"),
        ],
    )
    atomic_write_json(out_dir / "pilot_summary.json", summary, overwrite=True)
    return summary


# --- Public result bundle -----------------------------------------------------

SCHEMA_VERSION = "1.0.0"


def publish_result(
    cfg: ExperimentConfig,
    out_dir: Path,
    *,
    results_dir: Path | None,
    freeze_record: dict[str, object],
    protocol_only: bool,
) -> dict[str, object]:
    """Write the public evidence interface consumed by report and site.

    ``protocol_only`` publishes status without any empirical summary; the
    results mode validates that the statistical summary recomputes from the
    stored raw rows before exposing a single number.
    """
    import shutil
    import time as time_module

    from grounded_recovery.statistics import success_summary

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"public bundle already exists at {out_dir}")
    (out_dir / "tables").mkdir(parents=True)
    (out_dir / "figures").mkdir()

    generated_at = time_module.strftime("%Y-%m-%dT%H:%M:%SZ", time_module.gmtime())
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase": "protocol" if protocol_only else "results",
        "protocol_version": cfg.study.protocol_version,
        "protocol_hash": freeze_record["contract_hash"],
        "code_commit": None,
        "code_hash": freeze_record["code_hash"],
        "generated_at": generated_at,
        "result_release": not protocol_only,
        "canonical_report": "reports/Recovery_Policy_Learning_Technical_Report.pdf",
    }
    atomic_write_json(out_dir / "site-status.json", status)

    published: dict[str, object] = {"status": status}
    if not protocol_only:
        if results_dir is None:
            raise ValueError("results mode requires the results directory")
        results_dir = Path(results_dir)
        summary = read_json(results_dir / "statistical_summary.json")
        rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
        grouped = group_rows(rows)
        bundles = sorted({row.bundle_id for row in rows})
        # Re-verify before publishing: every per-bundle delta must recompute.
        for bundle, recorded in summary["per_bundle_deltas"].items():
            recovery = success_summary(grouped[(bundle, "recovery_aggregation", "unseen")])
            extra = success_summary(grouped[(bundle, "extra_demonstrations", "unseen")])
            if abs((recovery.rate - extra.rate) - recorded) > 1e-9:
                raise ValueError(f"summary does not recompute for bundle {bundle}")

        replicates = []
        for bundle in bundles:
            outcomes = []
            for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation"):
                for slice_name in SLICE_ORDER:
                    cell = success_summary(grouped[(bundle, arm, slice_name)])
                    outcomes.append(
                        {
                            "method": {"bc_base": "bc_base",
                                       "extra_demonstrations": "extra_demo",
                                       "recovery_aggregation": "recovery"}[arm],
                            "slice": slice_name,
                            "assigned_episodes": cell.assigned,
                            "successful_episodes": cell.successes,
                            "success_rate": cell.rate,
                            "intervention_delivered": (
                                None if slice_name == "clean" else cell.delivered
                            ),
                        }
                    )
            replicates.append(
                {
                    "bundle_id": bundle,
                    "outcomes": outcomes,
                    "primary_paired_difference": summary["per_bundle_deltas"][bundle],
                }
            )
        experiment_summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "grounded-recovery",
            "status": "results",
            "protocol": {
                "version": cfg.study.protocol_version,
                "hash": freeze_record["contract_hash"],
                "code_commit": None,
                "primary_contrast": cfg.study.primary_contrast,
                "primary_endpoint": cfg.study.primary_endpoint,
                "sesoi_absolute_success": cfg.study.sesoi_absolute_success,
            },
            "generated_at": generated_at,
            "methods": [
                {"id": "bc_base", "label": "BC base (no added labels)"},
                {"id": "extra_demo", "label": "Extra demonstrations"},
                {"id": "recovery", "label": "Recovery aggregation"},
            ],
            "slices": [
                {"id": "clean", "label": "Clean", "role": "secondary"},
                {"id": "matched", "label": "Matched perturbation", "role": "secondary"},
                {"id": "unseen", "label": "Unseen perturbation", "role": "primary"},
            ],
            "budget": {
                "additional_revealed_targets": {
                    "extra_demo": cfg.data.b,
                    "recovery": cfg.data.b,
                },
                "optimizer_updates_matched": True,
                "target_exposures_matched": True,
                "replay_rules_matched": True,
            },
            "eligibility": {
                "candidate_scenarios": summary_eligibility(freeze_record)["candidates"],
                "eligible_scenarios": summary_eligibility(freeze_record)["count"],
                "retained_fraction": summary_eligibility(freeze_record)[
                    "retained_fraction"
                ],
                "manifest_hash": freeze_record["eligible"]["eligible_hash"],
            },
            "replicates": replicates,
            "primary_summary": {
                "analysis_status": summary["analysis_status"],
                "claim_state": summary["claim_state"],
                "mean_paired_difference": summary["mean_paired_difference"],
                "interval": {
                    "method": "paired_t",
                    "level": 0.95,
                    "lower": summary["interval"]["lower"],
                    "upper": summary["interval"]["upper"],
                },
                "pipeline_replicates": summary["bundles_completed"],
                "precision": summary["precision"],
                "sensitivity": summary["sensitivity_bootstrap"],
            },
            "warnings": [],
        }
        audits_path = results_dir / "descriptive_audits.json"
        if audits_path.exists():
            audits = read_json(audits_path)
            experiment_summary["descriptive_audits"] = {
                "status": audits["status"],
                "failure_composition": audits["failure_composition"],
                "overhead": audits["overhead"],
                "delivery": audits["delivery"],
                "corruption_time_profile": audits["corruption_time_profile"],
                "secondary_contrasts": audits["secondary_contrasts"],
                "budget_exposure": audits["budget_exposure"],
            }
        atomic_write_json(out_dir / "experiment-summary.json", experiment_summary)
        for table in (results_dir / "tables").glob("*.csv"):
            shutil.copy2(table, out_dir / "tables" / table.name)
        for figure in (results_dir / "figures").glob("*.png"):
            shutil.copy2(figure, out_dir / "figures" / figure.name)
        media_dir = results_dir / "media"
        if media_dir.is_dir():
            shutil.copytree(media_dir, out_dir / "media")

        # The two reserved panels were opened after the confirmatory result.
        # They ship in their own subtree so that nothing exploratory can be
        # mistaken for part of the primary evidence.
        exploratory_dir = results_dir.parent / "exploratory" / results_dir.name
        if exploratory_dir.is_dir():
            target = out_dir / "exploratory"
            (target / "tables").mkdir(parents=True)
            (target / "figures").mkdir()
            for table in (exploratory_dir / "tables").glob("*.csv"):
                shutil.copy2(table, target / "tables" / table.name)
            for figure in (exploratory_dir / "figures").glob("*.png"):
                shutil.copy2(figure, target / "figures" / figure.name)
            for name in ("exploratory_summary.json", "exploratory_opening.json"):
                source = exploratory_dir / name
                if source.exists():
                    shutil.copy2(source, target / name)
        published["experiment_summary"] = experiment_summary

    claim_evidence = {
        "claims": [
            {
                "claim_id": "primary_unseen_itt",
                "status": "protocol_only" if protocol_only else "results",
                "metric": cfg.study.primary_endpoint,
                "table": None if protocol_only else "tables/paired_effects.csv",
                "figure": None if protocol_only else "figures/primary_paired_effect.png",
                "contract_hash": freeze_record["contract_hash"],
                "eligible_hash": freeze_record["eligible"]["eligible_hash"],
                "code_hash": freeze_record["code_hash"],
            }
        ]
    }
    if not protocol_only:
        claim_evidence["claims"].append(
            {
                "claim_id": "exploratory_reserved_panels",
                "status": "exploratory_opened_after_the_confirmatory_result",
                "metric": "expert agreement and two-corruption stress panels",
                "table": "exploratory/tables/two_corruption.csv",
                "figure": "exploratory/figures/two_corruption.png",
                "contract_hash": freeze_record["contract_hash"],
                "eligible_hash": None,
                "code_hash": freeze_record["code_hash"],
            }
        )
        claim_evidence["claims"].append(
            {
                "claim_id": "secondary_descriptive_audits",
                "status": "secondary_not_prespecified",
                "metric": "descriptive audits over the same opening",
                "table": "tables/secondary_contrasts.csv",
                "figure": "figures/secondary_contrasts.png",
                "contract_hash": freeze_record["contract_hash"],
                "eligible_hash": freeze_record["eligible"]["eligible_hash"],
                "code_hash": freeze_record["code_hash"],
            }
        )
    atomic_write_json(out_dir / "claim-evidence.json", claim_evidence)

    write_artifact_manifest(out_dir)
    return published


def write_artifact_manifest(out_dir: Path) -> None:
    """Re-hash every file of a published bundle into its artifact manifest."""
    from grounded_recovery.artifacts import file_sha256

    out_dir = Path(out_dir)
    manifest_entries = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            manifest_entries.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "role": "status" if path.name == "site-status.json" else "evidence",
                }
            )
    atomic_write_json(
        out_dir / "artifact-manifest.json", {"files": manifest_entries}, overwrite=True
    )


def refresh_public_media(results_dir: Path, out_dir: Path) -> int:
    """Replace a published bundle's media with regenerated files.

    Rollout media is presentation, not evidence: regenerating it must not
    restamp the release status or the summary, so this replaces only the media
    subtree and re-hashes the artifact manifest around it.
    """
    import shutil

    results_dir, out_dir = Path(results_dir), Path(out_dir)
    source = results_dir / "media"
    if not source.is_dir():
        raise ValueError(f"no media to publish at {source}")
    if not (out_dir / "site-status.json").exists():
        raise ValueError(f"no published bundle at {out_dir}")
    target = out_dir / "media"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    write_artifact_manifest(out_dir)
    return sum(1 for path in target.rglob("*") if path.is_file())


def summary_eligibility(freeze_record: dict[str, object]) -> dict[str, object]:
    return freeze_record["eligible"]


def write_results_report(
    cfg: ExperimentConfig,
    results_dir: Path,
    freeze_record: dict[str, object],
    out_path: Path,
) -> None:
    """Markdown mini-report for the confirmatory opening, from stored outputs."""
    results_dir = Path(results_dir)
    summary = read_json(results_dir / "statistical_summary.json")
    receipt = read_json(results_dir / "opening_receipt.json")
    rows = rows_from_jsonl(results_dir / "raw_episodes.jsonl")
    grouped = group_rows(rows)
    bundles = sorted({row.bundle_id for row in rows})

    interval = summary["interval"]
    state_sentences = {
        "support": (
            "The prespecified interval excludes zero in favor of recovery-state "
            "allocation on the frozen endpoint in this BabyAI setting."
        ),
        "adverse": (
            "Extra nominal demonstrations performed better than recovery-state "
            "allocation under this budget and allocation."
        ),
        "rule_out": (
            "Benefits at least as large as the SESOI are ruled out under the "
            "frozen scope; this is not a claim of exact equivalence."
        ),
        "inconclusive": (
            "The interval spans meaningful benefit and no benefit: the evidence "
            "is insufficiently precise to support or rule out the effect."
        ),
    }
    matrix_lines = ["| bundle | arm | clean | matched | unseen | unseen delivered |",
                    "| --- | --- | --- | --- | --- | --- |"]
    for bundle in bundles:
        for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation"):
            cells = {}
            for slice_name in SLICE_ORDER:
                cell = success_summary(grouped[(bundle, arm, slice_name)])
                cells[slice_name] = f"{cell.successes}/{cell.assigned}"
                if slice_name == "unseen":
                    delivered = f"{cell.delivered}/{cell.assigned}"
            matrix_lines.append(
                f"| {bundle} | {arm} | {cells['clean']} | {cells['matched']} | "
                f"{cells['unseen']} | {delivered} |"
            )
    deltas_lines = [
        f"- {bundle}: {delta:+.4f}"
        for bundle, delta in sorted(summary["per_bundle_deltas"].items())
    ]
    bootstrap = summary["sensitivity_bootstrap"]
    precision = summary["precision"]

    write_markdown_report(
        Path(out_path),
        "Grounded Recovery: confirmatory results mini-report",
        [
            ("Status",
             f"**Analysis status: {summary['analysis_status']}.** "
             f"{summary['bundles_completed']} complete pipeline bundles "
             f"(planned R_train {summary['planned_r_train']}), "
             f"{summary['scenario_denominator']} eligible unseen scenarios per cell, "
             f"one opening (receipt: contract {receipt['contract_hash'][:12]}...)."),
            ("Primary endpoint (leads regardless of sign)",
             f"Estimand: {summary['estimand']}.\n\n"
             f"**Mean paired difference: {summary['mean_paired_difference']:+.4f}** "
             f"(95% paired t interval [{interval['lower']:+.4f}, "
             f"{interval['upper']:+.4f}], R={summary['bundles_completed']}). "
             f"SESOI ±{summary['sesoi_absolute_success']:.2f}.\n\n"
             f"**Claim state: {summary['claim_state']}.** "
             f"{state_sentences[summary['claim_state']]}\n\n"
             "Per-bundle paired differences (recovery − extra demonstrations):\n"
             + "\n".join(deltas_lines)
             + "\n\n![primary paired effect](figures/primary_paired_effect.png)"),
            ("Sensitivity analysis (labelled, not primary)",
             f"Crossed two-way cluster bootstrap ({bootstrap['replicates']} replicates, "
             f"seed {bootstrap['seed']}): mean {bootstrap['mean']:+.4f}, "
             f"95% interval [{bootstrap['lower']:+.4f}, {bootstrap['upper']:+.4f}]. "
             "The bootstrap resamples bundles and scenarios with common draws across "
             "arms; it does not compensate for few pipeline bundles."),
            ("Precision",
             f"Desired half-width {precision['desired_half_width']}, achieved "
             f"{precision['achieved_half_width']:.4f}, target met: "
             f"{precision['target_met']}."),
            ("Success matrix (intention-to-treat; assigned episodes in every denominator)",
             "\n".join(matrix_lines)
             + "\n\n![success matrix](figures/success_matrix.png)"),
            ("Rollout media (disclosed selection rules)",
             "Side-by-side animations of actual evaluated episodes are in "
             "`media/` (also mirrored to `public_result/media/`): "
             "`unseen_paired_contrast` (smallest eligible-unseen ordinal where "
             "recovery succeeded and extra demonstrations failed), "
             "`unseen_recovery_failure` (smallest ordinal where the recovery "
             "arm failed; failures are shown, not hidden), and "
             "`oracle_nominal` (illustrative scripted-oracle mechanics, "
             "labelled not-a-result). Each empirical replay is asserted to "
             "reproduce the stored evaluation outcome; selection is by "
             "ordinal rule, never by appearance."),
            ("Scope and limitations",
             "Evidence is in silico: a symbolic BabyAI/MiniGrid grid world with "
             "open doors, discrete actions, a scripted BabyAIBot oracle, and "
             "DAgger-style recovery-state aggregation (masked behavioral cloning; "
             "no reinforcement learning, no physical robot, no human teachers). "
             "The unseen operator is structurally the inverse of the collection "
             "operator: on a three-action set exactly two derangements exist, "
             "so 'unseen' means the unique other derangement with a disjoint time "
             "set, not an independently sampled family. Matched-only findings are "
             "perturbation-family-specific. All numbers above recompute from "
             "raw_episodes.jsonl via the frozen analysis code."),
        ],
    )


def _typst_rows(rows: list[tuple]) -> str:
    """Render rows as a Typst array of tuples, strings quoted, numbers plain."""
    def cell(value: object) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    body = ", ".join("(" + ", ".join(cell(v) for v in row) + ")" for row in rows)
    return f"({body},)" if rows else "()"


def _export_audit_typst(
    results_dir: Path, out_dir: Path, header: str, cell_count: int
) -> None:
    """Emit the descriptive audit values the results chapter quotes."""
    audits_path = results_dir / "descriptive_audits.json"
    if not audits_path.exists():
        return
    audits = read_json(audits_path)
    arms = ["bc_base", "extra_demonstrations", "recovery_aggregation"]

    mean_success: list[tuple] = []
    import csv as csv_module

    with open(results_dir / "tables" / "pipeline_metrics.csv", encoding="utf-8") as fh:
        metric_rows = list(csv_module.DictReader(fh))
    for arm in arms:
        for slice_name in SLICE_ORDER:
            rates = [
                float(row["success_rate"])
                for row in metric_rows
                if row["arm"] == arm and row["slice"] == slice_name
            ]
            if rates:
                mean_success.append((arm, slice_name, sum(rates) / len(rates)))

    failures = [
        (arm,
         audits["failure_composition"][f"{arm}|unseen"][FAILURE_SUCCESS],
         audits["failure_composition"][f"{arm}|unseen"][FAILURE_TRUNCATED],
         audits["failure_composition"][f"{arm}|unseen"][FAILURE_TERMINATED])
        for arm in arms if f"{arm}|unseen" in audits["failure_composition"]
    ]
    terminated_total = sum(
        cell[FAILURE_TERMINATED] for cell in audits["failure_composition"].values()
    )
    overhead = [
        (arm,
         audits["overhead"][f"{arm}|unseen"]["successes"],
         audits["overhead"][f"{arm}|unseen"]["assigned"],
         audits["overhead"][f"{arm}|unseen"]["median_ratio"],
         audits["overhead"][f"{arm}|unseen"]["mean_ratio"])
        for arm in arms if f"{arm}|unseen" in audits["overhead"]
    ]
    delivery = [
        (arm, slice_name, cell["assigned"], cell["delivered"],
         cell["assigned"] - cell["delivered"])
        for arm in arms
        for slice_name in ("matched", "unseen")
        if (cell := audits["delivery"].get(f"{arm}|{slice_name}")) is not None
    ]
    time_profile = [
        (arm, int(time), counts[0], counts[1])
        for arm in arms
        for time, counts in sorted(
            audits["corruption_time_profile"].get(f"{arm}|unseen", {}).items(),
            key=lambda item: int(item[0]),
        )
    ]
    contrasts = [
        (c["first_arm"], c["second_arm"], c["slice"], c["mean"], c["lower"], c["upper"])
        for c in audits["secondary_contrasts"] if c["mean"] is not None
    ]
    budget = audits["budget_exposure"]
    matched = [
        (arm, budget["matched"][arm]["base"], budget["matched"][arm]["new"],
         budget["matched"][arm]["updates"])
        for arm in arms if arm in budget["matched"]
    ]
    logged = [
        (arm, budget["logged"][arm]["oracle_calls"],
         budget["logged"][arm]["simulator_steps"],
         budget["logged"][arm]["discarded_recommendations"],
         budget["logged"][arm]["episodes"])
        for arm in arms if arm in budget["logged"]
    ]

    lines = [
        f'#let audit-status = "{audits["status"]}"',
        f"#let mean-success = {_typst_rows(mean_success)}",
        f"#let failure-unseen = {_typst_rows(failures)}",
        f"#let terminated-without-goal-total = {terminated_total}",
        f"#let overhead-unseen = {_typst_rows(overhead)}",
        f"#let delivery-rows = {_typst_rows(delivery)}",
        f"#let time-profile = {_typst_rows(time_profile)}",
        f"#let secondary-contrasts = {_typst_rows(contrasts)}",
        f"#let budget-matched = {_typst_rows(matched)}",
        f"#let budget-logged = {_typst_rows(logged)}",
        f"#let exposures-equal = {str(bool(budget['exposures_equal_across_arms'])).lower()}",
        f"#let crossed-cells = {cell_count}",
        "",
    ]
    (out_dir / "audit_result.typ").write_text(header + "\n".join(lines), encoding="utf-8")


def _export_exploratory_typst(results_dir: Path, out_dir: Path, header: str) -> None:
    """Emit the reserved-panel values, if those panels have been opened."""
    path = Path(results_dir).parent / "exploratory" / Path(results_dir).name
    summary_path = path / "exploratory_summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    arms = ["bc_base", "extra_demonstrations", "recovery_aggregation"]
    agreement = [
        (arm, summary["expert_agreement"][arm]["mean_agreement_rate"],
         summary["expert_agreement"][arm]["mean_success_rate"])
        for arm in arms if arm in summary["expert_agreement"]
    ]
    two = summary["two_corruption"]
    two_rows = [
        (arm, two["per_arm"][arm]["mean_success_rate"])
        for arm in arms if arm in two["per_arm"]
    ]
    interval = two["paired_recovery_minus_extra"]["interval"] or {}
    lines = [
        f'#let exploratory-status = "{summary["status"]}"',
        f"#let agreement-rows = {_typst_rows(agreement)}",
        f'#let agreement-scenarios = {summary["panels"]["expert_agreement"]["scenarios"]}',
        f"#let two-corruption-rows = {_typst_rows(two_rows)}",
        f'#let two-corruption-scenarios = {summary["panels"]["two_corruption"]["scenarios"]}',
        f'#let two-corruption-mean = {interval.get("mean", 0.0):.6f}',
        f'#let two-corruption-lower = {interval.get("lower", 0.0):.6f}',
        f'#let two-corruption-upper = {interval.get("upper", 0.0):.6f}',
        f"#let two-corruption-has-interval = {str(bool(interval)).lower()}",
        "",
    ]
    (out_dir / "exploratory_result.typ").write_text(
        header + "\n".join(lines), encoding="utf-8"
    )
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    import shutil as shutil_module

    for figure in (path / "figures").glob("*.png"):
        shutil_module.copy2(figure, figures / f"exploratory_{figure.name}")


def export_typst(
    cfg: ExperimentConfig,
    results_dir: Path,
    freeze_record: dict[str, object],
    out_dir: Path,
) -> None:
    """Emit generated Typst fragments from the validated analysis outputs.

    Typst formats results; it never computes them. Every fragment begins with
    a generated-file marker and carries the source hashes.
    """
    import shutil

    from grounded_recovery.artifacts import file_sha256

    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    summary = read_json(results_dir / "statistical_summary.json")
    receipt = read_json(results_dir / "opening_receipt.json")
    interval = summary["interval"]
    bootstrap = summary["sensitivity_bootstrap"]
    precision = summary["precision"]

    # Content digests stay in the machine-readable bundle and in the freeze
    # record. They are deliberately not emitted as displayable values: a page
    # of hexadecimal certifies nothing to a reader, and the verification that
    # does mean something is the release integrity check.
    header = (
        "// GENERATED. DO NOT EDIT.\n"
        "// source: statistical_summary.json for the frozen protocol\n"
    )

    deltas_typst = ", ".join(
        f'("{bundle}", {delta:.6f})'
        for bundle, delta in sorted(summary["per_bundle_deltas"].items())
    )
    metadata = header + "\n".join(
        [
            f'#let protocol-version = "{cfg.study.protocol_version}"',
            f'#let eligible-count = {receipt["eligible_count"]}',
            f'#let candidate-count = {receipt["candidate_count"]}',
            f'#let bundles-completed = {summary["bundles_completed"]}',
            f'#let planned-r-train = {summary["planned_r_train"]}',
            f'#let sesoi = {summary["sesoi_absolute_success"]}',
            f'#let n0 = {cfg.data.n0}',
            f'#let budget-b = {cfg.data.b}',
            f'#let rounds-k = {cfg.data.k}',
            f'#let window-h = {cfg.data.h}',
            "",
        ]
    )
    (out_dir / "metadata.typ").write_text(metadata, encoding="utf-8")

    primary = header + "\n".join(
        [
            f'#let analysis-status = "{summary["analysis_status"]}"',
            f'#let claim-state = "{summary["claim_state"]}"',
            f'#let mean-delta = {summary["mean_paired_difference"]:.6f}',
            f'#let interval-lower = {interval["lower"]:.6f}',
            f'#let interval-upper = {interval["upper"]:.6f}',
            f'#let per-bundle-deltas = ({deltas_typst},)',
            f'#let bootstrap-lower = {bootstrap["lower"]:.6f}',
            f'#let bootstrap-upper = {bootstrap["upper"]:.6f}',
            f'#let bootstrap-replicates = {bootstrap["replicates"]}',
            f'#let achieved-half-width = {precision["achieved_half_width"]:.6f}',
            f'#let desired-half-width = {precision["desired_half_width"]}',
            f'#let precision-met = {str(bool(precision["target_met"])).lower()}',
            "",
        ]
    )
    (out_dir / "primary_result.typ").write_text(primary, encoding="utf-8")

    # Success table rows from the pipeline metrics table.
    import csv as csv_module

    with open(results_dir / "tables" / "pipeline_metrics.csv", encoding="utf-8") as fh:
        rows = list(csv_module.DictReader(fh))
    cells = []
    for row in rows:
        cells.append(
            f'("{row["bundle"]}", "{row["arm"]}", "{row["slice"]}", '
            f'{row["successes"]}, {row["assigned"]}, {row["delivered"]})'
        )
    table = header + "#let pipeline-rows = (" + ", ".join(cells) + ",)\n"
    (out_dir / "success_table.typ").write_text(table, encoding="utf-8")

    _export_audit_typst(results_dir, out_dir, header, len(rows))
    _export_exploratory_typst(results_dir, out_dir, header)

    for figure in (results_dir / "figures").glob("*.png"):
        shutil.copy2(figure, out_dir / "figures" / figure.name)

    manifest = {
        str(path.relative_to(out_dir)): file_sha256(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "report_inputs_manifest.json"
    }
    atomic_write_json(
        out_dir / "report_inputs_manifest.json",
        {"contract_hash": summary["contract_hash"], "files": manifest},
        overwrite=True,
    )
