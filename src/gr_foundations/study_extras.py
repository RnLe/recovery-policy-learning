"""Post-hoc exploratory panels beside the confirmatory study.

Two of the eight frozen scenario splits were reserved at freeze time for
optional descriptive diagnostics and were not part of the confirmatory
opening. This module opens them, after the confirmatory result, and reports
what they show.

Everything here is exploratory and not prespecified. It cannot move the
primary estimand: it reads only the final checkpoints, writes only into
``results/exploratory/``, and never touches the confirmatory opening.

The code lives outside ``grounded_recovery`` on purpose. The agreement panel
has to construct the scripted oracle at evaluation time, and the confirmatory
evaluator's standing invariant is that it never does. Keeping that boundary
intact is worth more than sharing a module.

Panel A, expert agreement (``expert_diagnostic`` split). Each arm's final
policy runs closed loop with no corruption while the oracle runs beside it in
lockstep. At every step the policy's greedy action is compared with the
oracle's recommendation for that same state. Agreement and closed-loop success
are different things, because more than one action is often valid, and the
panel exists to show that difference rather than to rank the arms.

Panel B, two corruptions (``difficulty_shift`` split). The primary endpoint is
explicitly a one-corruption endpoint. This panel schedules two corruptions per
episode, from the held-out operator and the held-out time set, and asks whether
the advantage survives compounding.
"""

from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grounded_recovery.artifacts import atomic_write_json, atomic_write_jsonl, read_json
from grounded_recovery.config import ExperimentConfig, contract_hash, load_and_validate
from grounded_recovery.data import (
    base_dataset_dir,
    load_split_manifest,
    start_action_token,
    vocabulary_from_dataset,
)
from grounded_recovery.evaluate import load_arm_policy
from grounded_recovery.oracle import run_synchronized_episode
from grounded_recovery.perturbations import operator_from_config
from grounded_recovery.schemas import canonical_scenario_hash
from grounded_recovery.seeds import derive_seed
from grounded_recovery.world import WorldSession

BC_BASE = "bc_base"
ARM_EXTRA = "extra_demonstrations"
ARM_RECOVERY = "recovery_aggregation"
ARMS = (BC_BASE, ARM_EXTRA, ARM_RECOVERY)

ARM_LABELS = {
    BC_BASE: "BC base (no added labels)",
    ARM_EXTRA: "Extra demonstrations",
    ARM_RECOVERY: "Recovery aggregation",
}
ARM_COLORS = {
    BC_BASE: "#8a8a8a",
    ARM_EXTRA: "#4878a8",
    ARM_RECOVERY: "#2a9d8f",
}
SHORT_LABELS = ["base", "extra", "recovery"]

AGREEMENT_SPLIT = "expert_diagnostic"
DIFFICULTY_SPLIT = "difficulty_shift"

EXPLORATORY_STATUS = "EXPLORATORY, not prespecified, opened after the confirmatory result"


class ExploratoryError(RuntimeError):
    """An exploratory panel violated one of its own preconditions."""


@dataclass(frozen=True)
class AgreementRow:
    """One clean closed-loop episode with a step-by-step oracle comparison."""

    bundle_id: str
    arm: str
    scenario_ordinal: int
    environment_seed: int
    steps: int
    comparisons: int
    agreements: int
    success: bool
    truncated: bool


@dataclass(frozen=True)
class TwoCorruptionRow:
    """One episode with two scheduled corruptions, scored intention-to-treat."""

    bundle_id: str
    arm: str
    scenario_ordinal: int
    environment_seed: int
    scheduled_times: tuple[int, ...]
    delivered_count: int
    success: bool
    truncated: bool
    steps: int
    nominal_oracle_path_length: int


def two_corruption_times(cfg: ExperimentConfig, ordinal: int) -> tuple[int, int]:
    """Two distinct held-out corruption times for one scenario, from the contract.

    The pair is a deterministic function of the contract and the scenario
    ordinal, exactly as the confirmatory schedule is, so every arm and bundle
    meets the identical panel.
    """
    pairs = sorted(combinations(sorted(cfg.perturbation.unseen_time_set), 2))
    if not pairs:
        raise ExploratoryError("the unseen time set has fewer than two times")
    raw = derive_seed(
        cfg.seeds.root_seed, "global", f"evaluation.exploratory.two_corruption.{ordinal}"
    )
    return pairs[raw % len(pairs)]


def _greedy_policy_stepper(cfg: ExperimentConfig, session, policy, vocab, mission: str):
    """A closure that returns the policy's greedy action for the current state.

    The recurrent state lives in the closure, so the caller can drive the
    authoritative transition loop without the policy leaking into it.
    """
    import torch

    device = next(policy.parameters()).device
    num_actions = len(cfg.environment.action_ids)
    encoded = vocab.encode(mission)
    if not encoded:
        raise ExploratoryError("empty mission token sequence")
    with torch.no_grad():
        mission_feature = policy.encode_mission(
            torch.tensor([encoded], dtype=torch.long, device=device),
            torch.tensor([len(encoded)], dtype=torch.long, device=device),
        )
    state = {"hidden": None, "last_token": start_action_token(num_actions)}

    def propose() -> int:
        observation = session.last_observation
        with torch.no_grad():
            image = torch.from_numpy(
                observation.image.astype("int64")
            ).unsqueeze(0).to(device)
            direction = torch.tensor(
                [observation.direction], dtype=torch.long, device=device
            )
            previous = torch.tensor(
                [state["last_token"]], dtype=torch.long, device=device
            )
            logits, hidden = policy.step(
                image, direction, previous, mission_feature, state["hidden"]
            )
        state["hidden"] = hidden
        return int(logits.argmax(dim=-1).item())

    def commit(executed: int) -> None:
        state["last_token"] = executed

    return propose, commit


def _agreement_chooser(cfg: ExperimentConfig, session, policy, vocab, counters):
    """Build the executed-action callback for one agreement episode.

    A factory rather than a closure written inline in the loop: the recurrent
    state and the counters belong to this episode alone, and binding them here
    makes that explicit instead of relying on loop-variable capture.
    """
    stepper: dict[str, object] = {}

    def choose_executed(t: int, recommended: int) -> int:
        if not stepper:
            propose, commit = _greedy_policy_stepper(
                cfg, session, policy, vocab, session.last_observation.mission
            )
            stepper["propose"] = propose
            stepper["commit"] = commit
        proposed = stepper["propose"]()
        counters["comparisons"] += 1
        counters["agreements"] += int(proposed == recommended)
        stepper["commit"](proposed)
        return proposed

    return choose_executed


def run_agreement_panel(
    cfg: ExperimentConfig,
    session,
    policy,
    vocab,
    entries,
    *,
    bundle_id: str,
    arm: str,
) -> list[AgreementRow]:
    """Clean rollouts with the oracle observing every step in lockstep."""
    rows: list[AgreementRow] = []
    for entry in entries:
        counters = {"comparisons": 0, "agreements": 0}
        trace = run_synchronized_episode(
            session,
            entry.environment_seed,
            _agreement_chooser(cfg, session, policy, vocab, counters),
        )
        if trace.scenario_hash != entry.canonical_scenario_hash:
            raise ExploratoryError(
                f"seed {entry.environment_seed} does not reproduce the manifested world"
            )
        rows.append(
            AgreementRow(
                bundle_id=bundle_id,
                arm=arm,
                scenario_ordinal=entry.ordinal,
                environment_seed=entry.environment_seed,
                steps=len(trace.transitions),
                comparisons=counters["comparisons"],
                agreements=counters["agreements"],
                success=trace.success,
                truncated=trace.truncated,
            )
        )
    return rows


def run_two_corruption_panel(
    cfg: ExperimentConfig,
    session,
    policy,
    vocab,
    entries,
    operator,
    *,
    bundle_id: str,
    arm: str,
) -> list[TwoCorruptionRow]:
    """Closed-loop rollouts with two scheduled corruptions, scored ITT.

    Every assigned scenario produces exactly one row. An episode that ends
    before its second scheduled time still counts, with its delivered count
    recorded, so the denominator never depends on the outcome.
    """
    rows: list[TwoCorruptionRow] = []
    for entry in entries:
        times = two_corruption_times(cfg, entry.ordinal)
        observation = session.reset(entry.environment_seed)
        if canonical_scenario_hash(session.scenario_state()) != entry.canonical_scenario_hash:
            raise ExploratoryError(
                f"seed {entry.environment_seed} does not reproduce the manifested world"
            )
        propose, commit = _greedy_policy_stepper(
            cfg, session, policy, vocab, observation.mission
        )
        delivered = 0
        while not session.done:
            t = session.time
            proposal = propose()
            if t in times:
                executed = operator.apply(proposal)
                delivered += 1
            else:
                executed = proposal
            observation = session.step(executed)
            commit(executed)
        rows.append(
            TwoCorruptionRow(
                bundle_id=bundle_id,
                arm=arm,
                scenario_ordinal=entry.ordinal,
                environment_seed=entry.environment_seed,
                scheduled_times=times,
                delivered_count=delivered,
                success=observation.terminated,
                truncated=observation.truncated,
                steps=session.time,
                nominal_oracle_path_length=entry.nominal_oracle_path_length,
            )
        )
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _paired_interval(deltas: list[float]) -> dict[str, float] | None:
    from grounded_recovery.statistics import paired_t_interval

    if len(deltas) < 2:
        return None
    interval = paired_t_interval(deltas)
    return {
        "mean": interval.mean,
        "lower": interval.lower,
        "upper": interval.upper,
        "level": interval.level,
        "method": "paired_t",
    }


def summarize_agreement(rows: list[AgreementRow], bundles: list[str]) -> dict[str, object]:
    """Agreement and clean success per arm, with per-bundle points."""
    per_arm: dict[str, object] = {}
    for arm in ARMS:
        by_bundle = {}
        for bundle in bundles:
            cell = [r for r in rows if r.arm == arm and r.bundle_id == bundle]
            if not cell:
                continue
            comparisons = sum(r.comparisons for r in cell)
            by_bundle[bundle] = {
                "agreement_rate": sum(r.agreements for r in cell) / comparisons,
                "success_rate": sum(1 for r in cell if r.success) / len(cell),
                "episodes": len(cell),
                "compared_steps": comparisons,
            }
        if not by_bundle:
            continue
        per_arm[arm] = {
            "per_bundle": by_bundle,
            "mean_agreement_rate": sum(
                v["agreement_rate"] for v in by_bundle.values()
            ) / len(by_bundle),
            "mean_success_rate": sum(
                v["success_rate"] for v in by_bundle.values()
            ) / len(by_bundle),
        }
    return per_arm


def summarize_two_corruption(
    rows: list[TwoCorruptionRow], bundles: list[str]
) -> dict[str, object]:
    """ITT success per arm under two corruptions, with the paired contrast."""
    per_arm: dict[str, object] = {}
    for arm in ARMS:
        by_bundle = {}
        for bundle in bundles:
            cell = [r for r in rows if r.arm == arm and r.bundle_id == bundle]
            if not cell:
                continue
            by_bundle[bundle] = {
                "success_rate": sum(1 for r in cell if r.success) / len(cell),
                "assigned": len(cell),
                "both_delivered": sum(1 for r in cell if r.delivered_count == 2),
            }
        if by_bundle:
            per_arm[arm] = {
                "per_bundle": by_bundle,
                "mean_success_rate": sum(
                    v["success_rate"] for v in by_bundle.values()
                ) / len(by_bundle),
            }
    shared = [b for b in bundles
              if all(b in per_arm.get(arm, {}).get("per_bundle", {}) for arm in ARMS)]
    deltas = [
        per_arm[ARM_RECOVERY]["per_bundle"][b]["success_rate"]
        - per_arm[ARM_EXTRA]["per_bundle"][b]["success_rate"]
        for b in shared
    ]
    return {
        "per_arm": per_arm,
        "paired_recovery_minus_extra": {
            "per_bundle_deltas": dict(zip(shared, deltas, strict=True)),
            "interval": _paired_interval(deltas),
            "status": EXPLORATORY_STATUS,
        },
    }


def _percent(value: float, digits: int = 1) -> str:
    """Percentage with half-up rounding, matching the report's own formatter.

    Python rounds 0.8925 down and Typst rounds it up, which would print two
    different numbers for one value on the same page.
    """
    import math

    scale = 10 ** digits
    return f"{math.floor(value * 100 * scale + 0.5) / scale:.{digits}f}%"


def plot_agreement(summary: dict[str, object], out_path: Path, episodes: int) -> None:
    """Expert agreement beside clean success, so the gap between them is visible."""
    arms = [arm for arm in ARMS if arm in summary]
    fig, axis = plt.subplots(figsize=(8.6, 3.6))
    width = 0.36
    for offset, key, label in (
        (-width / 2, "mean_agreement_rate", "agrees with the oracle, per step"),
        (width / 2, "mean_success_rate", "reaches the goal, per episode"),
    ):
        values = [summary[arm][key] for arm in arms]
        positions = [index + offset for index in range(len(arms))]
        axis.bar(positions, values, width=width, label=label,
                 color=[ARM_COLORS[arm] for arm in arms],
                 alpha=0.85 if offset < 0 else 0.45,
                 edgecolor="white",
                 hatch="" if offset < 0 else "//")
        for position, value in zip(positions, values, strict=True):
            axis.annotate(_percent(value), (position, value), ha="center",
                          va="bottom", fontsize=8, xytext=(0, 2),
                          textcoords="offset points")
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(SHORT_LABELS[: len(arms)], fontsize=9)
    axis.set_ylim(0, 1.24)
    axis.set_ylabel("rate")
    axis.set_title(
        "Agreement with the oracle against closed-loop success, no corruption\n"
        f"exploratory, not prespecified; {episodes} episodes per arm and bundle",
        fontsize=10,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0", alpha=0.85, edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0", alpha=0.45, hatch="//",
                      edgecolor="white"),
    ]
    axis.legend(handles, ["agrees with the oracle, per step",
                          "reaches the goal, per episode"], fontsize=9,
                loc="upper center", ncols=2, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_two_corruption(
    summary: dict[str, object], out_path: Path, episodes: int
) -> None:
    """ITT success under two corruptions, with every pipeline bundle shown."""
    per_arm = summary["per_arm"]
    arms = [arm for arm in ARMS if arm in per_arm]
    fig, axis = plt.subplots(figsize=(7.6, 4.0))
    for index, arm in enumerate(arms):
        rates = [v["success_rate"] for v in per_arm[arm]["per_bundle"].values()]
        axis.bar(index, sum(rates) / len(rates), width=0.6, color=ARM_COLORS[arm],
                 alpha=0.55, zorder=1)
        axis.scatter([index] * len(rates), rates, color=ARM_COLORS[arm],
                     edgecolor="black", linewidth=0.5, s=28, zorder=3)
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(SHORT_LABELS[: len(arms)], fontsize=9)
    axis.set_ylim(0, 1)
    axis.set_ylabel("ITT success rate")
    interval = summary["paired_recovery_minus_extra"]["interval"]
    caption = (
        f"recovery minus extra: {interval['mean']:+.3f} "
        f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]"
        if interval else "interval unavailable"
    )
    axis.set_title(
        "Two held-out corruptions per episode\n"
        f"exploratory, not prespecified; {episodes} scenarios per cell\n"
        f"{caption}",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_study_extras(
    contract_path: Path,
    manifest_root: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, object]:
    """Open both reserved panels and write the exploratory bundle.

    The confirmatory opening is never read or written here. This pass records
    its own opening note, so that a post-hoc opening is itself on the record
    rather than appearing as an undated extra table.
    """
    import torch

    from grounded_recovery.train import model_state_digest

    cfg = load_and_validate(Path(contract_path))
    cfg_hash = contract_hash(cfg)
    out_dir = Path(results_root) / "exploratory" / cfg_hash[:12]
    out_dir.mkdir(parents=True, exist_ok=True)

    agreement_entries, agreement_hash = load_split_manifest(
        Path(manifest_root), AGREEMENT_SPLIT
    )
    difficulty_entries, difficulty_hash = load_split_manifest(
        Path(manifest_root), DIFFICULTY_SPLIT
    )
    operator = operator_from_config(
        cfg.perturbation.unseen_operator, cfg.environment.action_ids
    )

    bundles = list(cfg.seeds.bundle_ids)
    checkpoints: dict[str, dict[str, str]] = {}
    digests: dict[str, dict[str, str]] = {}
    for bundle_id in bundles:
        summary = read_json(
            Path(data_root) / cfg_hash[:12] / bundle_id / "bundle_summary.json"
        )
        checkpoints[bundle_id] = {
            BC_BASE: summary["base"]["checkpoint"],
            ARM_EXTRA: summary["arms"][ARM_EXTRA]["final_checkpoint"],
            ARM_RECOVERY: summary["arms"][ARM_RECOVERY]["final_checkpoint"],
        }
        digests[bundle_id] = {
            arm: model_state_digest(
                torch.load(path, map_location="cpu", weights_only=False)["model_state"]
            )
            for arm, path in checkpoints[bundle_id].items()
        }

    atomic_write_json(
        out_dir / "exploratory_opening.json",
        {
            "status": EXPLORATORY_STATUS,
            "opened_at_unix": round(time_module.time(), 1),
            "contract_hash": cfg_hash,
            "panels": {
                "expert_agreement": {
                    "split": AGREEMENT_SPLIT,
                    "manifest_hash": agreement_hash,
                    "scenarios": len(agreement_entries),
                },
                "two_corruption": {
                    "split": DIFFICULTY_SPLIT,
                    "manifest_hash": difficulty_hash,
                    "scenarios": len(difficulty_entries),
                    "operator": cfg.perturbation.unseen_operator.name,
                    "time_set": list(cfg.perturbation.unseen_time_set),
                },
            },
            "bundles": digests,
            "expected_cells": len(bundles) * len(ARMS) * 2,
        },
        overwrite=True,
    )

    session = WorldSession(cfg.environment)
    agreement_rows: list[AgreementRow] = []
    two_rows: list[TwoCorruptionRow] = []
    try:
        for bundle_id in bundles:
            vocab = vocabulary_from_dataset(
                base_dataset_dir(cfg, bundle_id, Path(data_root))
            )
            for arm, checkpoint in checkpoints[bundle_id].items():
                policy = load_arm_policy(cfg, checkpoint, vocab)
                agreement_rows.extend(
                    run_agreement_panel(
                        cfg, session, policy, vocab, agreement_entries,
                        bundle_id=bundle_id, arm=arm,
                    )
                )
                two_rows.extend(
                    run_two_corruption_panel(
                        cfg, session, policy, vocab, difficulty_entries, operator,
                        bundle_id=bundle_id, arm=arm,
                    )
                )
    finally:
        session.close()

    atomic_write_jsonl(
        out_dir / "agreement_rows.jsonl",
        [asdict(row) for row in agreement_rows],
        overwrite=True,
    )
    atomic_write_jsonl(
        out_dir / "two_corruption_rows.jsonl",
        [asdict(row) for row in two_rows],
        overwrite=True,
    )

    agreement = summarize_agreement(agreement_rows, bundles)
    two = summarize_two_corruption(two_rows, bundles)

    _write_csv(
        out_dir / "tables" / "expert_agreement.csv",
        ["bundle", "arm", "episodes", "compared_steps", "agreement_rate",
         "clean_success_rate"],
        [
            [bundle, arm, cell["episodes"], cell["compared_steps"],
             round(cell["agreement_rate"], 6), round(cell["success_rate"], 6)]
            for arm in ARMS if arm in agreement
            for bundle, cell in agreement[arm]["per_bundle"].items()
        ],
    )
    _write_csv(
        out_dir / "tables" / "two_corruption.csv",
        ["bundle", "arm", "assigned", "both_delivered", "success_rate"],
        [
            [bundle, arm, cell["assigned"], cell["both_delivered"],
             round(cell["success_rate"], 6)]
            for arm in ARMS if arm in two["per_arm"]
            for bundle, cell in two["per_arm"][arm]["per_bundle"].items()
        ],
    )
    plot_agreement(
        agreement, out_dir / "figures" / "expert_agreement.png",
        len(agreement_entries),
    )
    plot_two_corruption(
        two, out_dir / "figures" / "two_corruption.png", len(difficulty_entries)
    )

    summary = {
        "status": EXPLORATORY_STATUS,
        "contract_hash": cfg_hash,
        "expert_agreement": agreement,
        "two_corruption": two,
        "panels": {
            "expert_agreement": {"split": AGREEMENT_SPLIT,
                                 "scenarios": len(agreement_entries)},
            "two_corruption": {"split": DIFFICULTY_SPLIT,
                               "scenarios": len(difficulty_entries)},
        },
        "bundles": bundles,
    }
    atomic_write_json(out_dir / "exploratory_summary.json", summary, overwrite=True)
    return summary
