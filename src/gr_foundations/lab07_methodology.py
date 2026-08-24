"""Lab 7: measuring honestly, with budgets, intention-to-treat, frozen protocols.

The study's conclusions are only as good as its measurement discipline, and
each piece of that discipline exists because a specific bias would otherwise
creep in. This lab demonstrates the biases directly: a simulation with known
ground truth shows per-protocol analysis (dropping undelivered corruptions)
systematically missing the intention-to-treat estimand while the ITT
estimator centers on it; the same comparison is repeated on Lab 6's real
evaluation rows. A deliberately unmatched rerun of Lab 6's recovery arm at
double budget probes the attribution confound of unequal budgets, because whatever
its outcome, an unmatched comparison stops measuring the method. A
paired-versus-unpaired simulation shows why the study replicates in
pipeline bundles and analyzes paired differences. Finally, the freeze and
integrity machinery is demonstrated in miniature: a canonical-hash flip and a
tampered hash chain caught at the exact row.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from gr_foundations.common import (
    COLOR_BASE,
    COLOR_CAUTION,
    COLOR_EXTRA,
    COLOR_RECOVERY,
    FoundationsError,
    LabPaths,
    derive_seed,
    export_typst_table,
    export_typst_values,
    prepare,
    save_figure,
    write_metrics,
    write_mini_report,
    write_table_csv,
)
from grounded_recovery.artifacts import hash_json, sha256_hex

SIM_SCENARIOS = 536  # deliberately the study's eligible panel size
SIM_REPLICATES = 2000
TRUE_ARM_GAP = 0.05
CORRUPTION_HARM = 0.15
POWER_BUNDLES = 6

FREEZE_MECHANISMS: list[list[str]] = [
    ["PILOT_TO_FREEZE placeholders",
     "running confirmatory stages with unresolved design choices",
     "config.py refuses FROZEN status while placeholders remain"],
    ["contract hash (canonical JSON, SHA-256)",
     "silent post-hoc edits to the design",
     "every artifact stamps the hash; one flipped field changes it"],
    ["scenario manifests with identity hashes",
     "evaluation scenarios drifting between runs",
     "data.py manifests + disjointness audit"],
    ["hash-chained ledgers + independent recount",
     "quiet insertion, deletion, or edit of collected data",
     "integrity.py LedgerWriter / recount_dataset"],
    ["single receipted test opening",
     "rerunning the confirmatory evaluation until it looks good",
     "experiment.py opening receipt; write-once results"],
    ["preregistered claim decision rule",
     "choosing the interpretation after seeing the numbers",
     "frozen text mapping interval position to support/adverse/rule-out/"
     "inconclusive"],
]


# --------------------------------------------------------------------------
# Simulation: intention-to-treat versus per-protocol, with known truth.
# --------------------------------------------------------------------------


def simulate_delivery_bias(
    *,
    n_scenarios: int = SIM_SCENARIOS,
    n_replicates: int = SIM_REPLICATES,
    true_gap: float = TRUE_ARM_GAP,
    harm: float = CORRUPTION_HARM,
) -> dict[str, object]:
    """Two arms, one corruption per episode, delivery depending on the arm.

    Arm B is genuinely better than arm A by ``true_gap`` everywhere, and both
    lose ``harm`` success probability when the corruption is delivered. B is
    also more efficient (shorter episodes), so at late corruption times B's
    episodes are more often already over, so delivery is a *post-treatment*
    variable that differs between arms. The preregistered estimand is the ITT
    difference; conditioning on delivery estimates something else.
    """
    rng = np.random.default_rng(derive_seed("lab07.simulation"))
    difficulty = rng.integers(3, 21, size=n_scenarios)
    scheduled = rng.integers(2, 13, size=n_scenarios)

    def arm(base: float, wander: float) -> dict[str, np.ndarray]:
        clean = np.clip(base - 0.015 * difficulty, 0.05, 0.98)
        length = np.round(difficulty * (1.0 + wander)).astype(int)
        delivered = length > scheduled
        scheduled_success = np.where(delivered, clean - harm, clean)
        return {
            "clean": clean,
            "delivered": delivered,
            "p": np.clip(scheduled_success, 0.0, 1.0),
        }

    arm_a = arm(0.80, 0.50)
    arm_b = arm(0.80 + true_gap, 0.00)
    truth_itt = float(np.mean(arm_b["p"]) - np.mean(arm_a["p"]))

    itt_estimates = np.empty(n_replicates)
    per_protocol_estimates = np.empty(n_replicates)
    for replicate in range(n_replicates):
        outcomes_a = rng.random(n_scenarios) < arm_a["p"]
        outcomes_b = rng.random(n_scenarios) < arm_b["p"]
        itt_estimates[replicate] = outcomes_b.mean() - outcomes_a.mean()
        per_protocol_estimates[replicate] = (
            outcomes_b[arm_b["delivered"]].mean()
            - outcomes_a[arm_a["delivered"]].mean()
        )
    return {
        "true_itt_effect": truth_itt,
        "true_everywhere_gap": true_gap,
        "delivery_rate_a": float(arm_a["delivered"].mean()),
        "delivery_rate_b": float(arm_b["delivered"].mean()),
        "itt_mean": float(itt_estimates.mean()),
        "itt_bias": float(itt_estimates.mean() - truth_itt),
        "per_protocol_mean": float(per_protocol_estimates.mean()),
        "per_protocol_bias": float(per_protocol_estimates.mean() - truth_itt),
        "itt_estimates": itt_estimates,
        "per_protocol_estimates": per_protocol_estimates,
    }


def simulate_pairing(
    *,
    n_bundles: int = POWER_BUNDLES,
    n_replicates: int = SIM_REPLICATES,
    true_gap: float = TRUE_ARM_GAP,
    shared_noise: float = 0.03,
    arm_noise: float = 0.01,
) -> dict[str, object]:
    """Paired versus unpaired analysis of the same bundle structure.

    Each pipeline bundle re-rolls data collection and training, shifting both
    arms together (shared noise); pairing subtracts that shift out, the
    unpaired analysis pays for it.
    """
    rng = np.random.default_rng(derive_seed("lab07.power"))
    paired_widths = np.empty(n_replicates)
    unpaired_widths = np.empty(n_replicates)
    paired_significant = 0
    unpaired_significant = 0
    from scipy import stats

    critical = stats.t.ppf(0.975, df=n_bundles - 1)
    critical_unpaired = stats.t.ppf(0.975, df=2 * n_bundles - 2)
    for replicate in range(n_replicates):
        bundle_effect = rng.normal(0.0, shared_noise, size=n_bundles)
        arm_a = 0.70 + bundle_effect + rng.normal(0.0, arm_noise, size=n_bundles)
        arm_b = 0.70 + true_gap + bundle_effect + rng.normal(0.0, arm_noise, size=n_bundles)
        deltas = arm_b - arm_a
        half_paired = critical * deltas.std(ddof=1) / np.sqrt(n_bundles)
        paired_widths[replicate] = 2 * half_paired
        paired_significant += int(deltas.mean() - half_paired > 0)
        pooled = np.sqrt((arm_a.var(ddof=1) + arm_b.var(ddof=1)) / 2)
        half_unpaired = critical_unpaired * pooled * np.sqrt(2 / n_bundles)
        unpaired_widths[replicate] = 2 * half_unpaired
        unpaired_significant += int((arm_b.mean() - arm_a.mean()) - half_unpaired > 0)
    return {
        "n_bundles": n_bundles,
        "true_gap": true_gap,
        "paired_mean_width": float(paired_widths.mean()),
        "unpaired_mean_width": float(unpaired_widths.mean()),
        "paired_power": paired_significant / n_replicates,
        "unpaired_power": unpaired_significant / n_replicates,
        "paired_widths": paired_widths,
        "unpaired_widths": unpaired_widths,
    }


# --------------------------------------------------------------------------
# Real data: Lab 6 rows re-analyzed; hash and chain demonstrations.
# --------------------------------------------------------------------------


def reanalyze_lab06(rows: list[dict[str, object]]) -> dict[str, object]:
    """Recovery-minus-extra on the unseen slice: ITT versus delivered-only."""

    def rate(arm: str, *, delivered_only: bool) -> tuple[float, int]:
        selected = [
            row
            for row in rows
            if row["arm"] == arm
            and row["slice"] == "unseen"
            and (not delivered_only or row["delivered"])
        ]
        if not selected:
            return 0.0, 0
        return float(np.mean([row["success"] for row in selected])), len(selected)

    itt_recovery, n_itt_recovery = rate("recovery", delivered_only=False)
    itt_extra, n_itt_extra = rate("extra", delivered_only=False)
    pp_recovery, n_pp_recovery = rate("recovery", delivered_only=True)
    pp_extra, n_pp_extra = rate("extra", delivered_only=True)
    return {
        "itt_delta": itt_recovery - itt_extra,
        "per_protocol_delta": pp_recovery - pp_extra,
        "itt_denominators": [n_itt_recovery, n_itt_extra],
        "per_protocol_denominators": [n_pp_recovery, n_pp_extra],
        "delivered_fraction_recovery": n_pp_recovery / n_itt_recovery
        if n_itt_recovery
        else 0.0,
        "delivered_fraction_extra": n_pp_extra / n_itt_extra if n_itt_extra else 0.0,
    }


def hash_chain_demo() -> dict[str, object]:
    """A five-row hash chain, then one tampered byte, caught at its row."""
    rows = [
        {"episode": index, "labels": 10 + index, "checksum": f"c{index:02d}"}
        for index in range(5)
    ]
    def chain(all_rows: list[dict[str, object]]) -> list[str]:
        hashes = []
        previous = "0" * 64
        for row in all_rows:
            previous = sha256_hex((previous + hash_json(row)).encode("ascii"))
            hashes.append(previous)
        return hashes

    honest = chain(rows)
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[2]["labels"] = 999  # the quiet edit
    tampered = chain(tampered_rows)
    first_mismatch = next(
        index for index, (a, b) in enumerate(zip(honest, tampered, strict=True)) if a != b
    )
    return {
        "rows": len(rows),
        "tampered_row": 2,
        "first_mismatch": first_mismatch,
        "final_hash_honest": honest[-1],
        "final_hash_tampered": tampered[-1],
    }


def contract_hash_demo(repo_root) -> dict[str, object]:
    """Flipping one contract field changes the canonical hash entirely."""
    import yaml

    contract_path = repo_root / "configs" / "experiment_contract.yaml"
    document = yaml.safe_load(contract_path.read_text())
    original = hash_json(document)
    edited = json.loads(json.dumps(document))
    edited["data"]["h"] = int(edited["data"]["h"]) + 1
    return {
        "field": "data.h",
        "original_prefix": original[:16],
        "edited_prefix": hash_json(edited)[:16],
    }


# --------------------------------------------------------------------------
# The deliberately unmatched arm.
# --------------------------------------------------------------------------


def run_unmatched_arm(repo_root, budget_multiplier: int = 2) -> dict[str, object]:
    """Reproduce Lab 6's replicate 0 deterministically, then hand the recovery
    arm ``budget_multiplier`` times the label budget, the confounded design."""
    from gr_foundations import lab06_shift
    from gr_foundations.training import (
        build_bc_dataset,
        contract_config,
        dataset_vocabulary,
        resolve_device,
        train_bc,
    )
    from grounded_recovery.model import RecoveryPolicy

    contract = contract_config(repo_root)
    env_cfg = contract.environment
    device = resolve_device()
    dataset, _ = build_bc_dataset(env_cfg, lab06_shift.BASE_EPISODES, "lab06.dataset")
    holdout, _ = build_bc_dataset(
        env_cfg, lab06_shift.HOLDOUT_SCENARIOS, "lab06.holdout"
    )
    vocab = dataset_vocabulary(dataset)
    times = tuple(contract.perturbation.collection_time_set or (2, 4, 6, 8))
    op_collect = lab06_shift._operator(
        contract.perturbation.collection_operator, tuple(env_cfg.action_ids)
    )
    op_unseen = lab06_shift._operator(
        contract.perturbation.unseen_operator, tuple(env_cfg.action_ids)
    )

    def factory():
        return RecoveryPolicy(contract.model, vocab.size, len(env_cfg.action_ids))

    # Identical named seeds -> bit-identical reproduction of Lab 6's rep-0 base.
    base_model, _ = train_bc(
        factory,
        dataset,
        vocab,
        updates=lab06_shift.BASE_UPDATES,
        batch_episodes=lab06_shift.BATCH_BASE + lab06_shift.BATCH_ARM,
        seed=derive_seed("lab06.train", 0),
        device=device,
    )
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    inflated_budget = lab06_shift.BUDGET_LABELS * budget_multiplier
    recovery_data, counters = lab06_shift.collect_recovery(
        base_model, vocab, env_cfg, op_collect, inflated_budget,
        seed_offset=derive_unmatched_offset(), times=times,
        window=lab06_shift.RECOVERY_WINDOW, device=device,
    )
    arm_model, _stats = lab06_shift.finetune_arm(
        base_state, factory, dataset, recovery_data, vocab,
        updates=lab06_shift.ARM_UPDATES,
        seed=derive_seed("lab07.unmatched", 1),
        device=device,
    )
    successes = []
    for scenario_index, episode in enumerate(holdout):
        t_star = int(
            np.random.default_rng(
                derive_seed("lab06.sweep", scenario_index)
            ).choice(times)
        )
        outcome = lab06_shift.rollout_policy(
            arm_model, vocab, env_cfg, episode.seed, device,
            corruption=(op_unseen, t_star),
        )
        successes.append(float(outcome["success"]))
    return {
        "budget_labels": inflated_budget,
        "collection": counters,
        "unseen_success": float(np.mean(successes)),
        "scenarios": len(successes),
    }


def derive_unmatched_offset() -> int:
    """A collection-stream offset far outside Lab 6's replicate offsets."""
    return 900_000


# --------------------------------------------------------------------------
# The lab run.
# --------------------------------------------------------------------------


def run(
    paths: LabPaths,
    *,
    force: bool,
    skip_unmatched: bool = False,
    sim_replicates: int = SIM_REPLICATES,
) -> dict[str, object]:
    prepare(paths, force=force)

    lab06_rows_path = (
        paths.repo_root / "data" / "foundations" / "lab06" / "evaluation_rows.json"
    )
    lab06_metrics_path = paths.repo_root / "foundations" / "lab06" / "metrics.json"
    if not lab06_rows_path.exists() or not lab06_metrics_path.exists():
        raise FoundationsError(
            "lab07 re-analyzes Lab 6's evaluation rows; run `grf run lab06` first"
        )
    rows = json.loads(lab06_rows_path.read_text())
    lab06_metrics = json.loads(lab06_metrics_path.read_text())["metrics"]

    simulation = simulate_delivery_bias(n_replicates=sim_replicates)
    pairing = simulate_pairing(n_replicates=sim_replicates)
    reanalysis = reanalyze_lab06(rows)
    chain = hash_chain_demo()
    hash_demo = contract_hash_demo(paths.repo_root)
    unmatched = None if skip_unmatched else run_unmatched_arm(paths.repo_root)

    # ---- figures ----------------------------------------------------------
    fig, axis = plt.subplots(figsize=(7.8, 4.0))
    bins = np.linspace(
        min(simulation["per_protocol_estimates"].min(), simulation["itt_estimates"].min()),
        max(simulation["per_protocol_estimates"].max(), simulation["itt_estimates"].max()),
        45,
    )
    axis.hist(
        simulation["itt_estimates"], bins=bins, alpha=0.75, color=COLOR_RECOVERY,
        label="ITT estimator",
    )
    axis.hist(
        simulation["per_protocol_estimates"], bins=bins, alpha=0.75, color=COLOR_CAUTION,
        label="per-protocol estimator",
    )
    axis.axvline(
        simulation["true_itt_effect"], color="black", linewidth=1.2,
        label=f"true ITT effect ({simulation['true_itt_effect']:+.3f})",
    )
    axis.set_xlabel("estimated arm difference")
    axis.set_ylabel("simulated replicates")
    axis.legend(fontsize=12)
    axis.set_title(
        "known ground truth: conditioning on delivery biases the estimate",
        fontsize=13,
    )
    save_figure(paths, fig, "itt_bias.svg")

    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    axis.hist(
        pairing["paired_widths"], bins=40, alpha=0.75, color=COLOR_RECOVERY,
        label=f"paired (power {pairing['paired_power']:.0%})",
    )
    axis.hist(
        pairing["unpaired_widths"], bins=40, alpha=0.75, color=COLOR_BASE,
        label=f"unpaired (power {pairing['unpaired_power']:.0%})",
    )
    axis.set_xlabel(f"95% interval width at {pairing['n_bundles']} bundles")
    axis.set_ylabel("simulated replicates")
    axis.legend(fontsize=12)
    axis.set_title(
        "shared pipeline noise cancels in paired differences", fontsize=13
    )
    save_figure(paths, fig, "paired_power.svg")

    if unmatched is not None:
        matched_matrix = lab06_metrics["success_matrix"]
        fig, axis = plt.subplots(figsize=(7.4, 4.0))
        labels = [
            f"extra\n(+{lab06_metrics['design']['budget_labels']} labels)",
            f"recovery\n(+{lab06_metrics['design']['budget_labels']} labels)",
            f"recovery\n(+{unmatched['budget_labels']} labels)\nUNMATCHED",
        ]
        values = [
            matched_matrix["extra"]["unseen"],
            matched_matrix["recovery"]["unseen"],
            unmatched["unseen_success"],
        ]
        colors = [COLOR_EXTRA, COLOR_RECOVERY, COLOR_CAUTION]
        axis.bar(labels, values, color=colors)
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{value:.1%}", ha="center", fontsize=12)
        axis.set_ylim(0, 1.1)
        axis.set_ylabel("unseen-corruption ITT success")
        axis.tick_params(axis="x", labelsize=8)
        axis.set_title(
            "the unmatched design: attributing budget effects to the method",
            fontsize=13,
        )
        save_figure(paths, fig, "unmatched_confound.svg")

    # ---- tables and values ------------------------------------------------
    export_typst_table(
        paths,
        "freeze_mechanisms",
        ["mechanism", "failure it prevents", "where"],
        FREEZE_MECHANISMS,
    )
    write_table_csv(
        paths, "freeze_mechanisms.csv", ["mechanism", "prevents", "where"],
        FREEZE_MECHANISMS,
    )
    measurement_values = {
        "sim-true-itt": f"{simulation['true_itt_effect']:+.3f}",
        "sim-itt-bias": f"{simulation['itt_bias']:+.4f}",
        "sim-pp-bias": f"{simulation['per_protocol_bias']:+.4f}",
        "real-itt-delta": f"{reanalysis['itt_delta']:+.1%}",
        "real-pp-delta": f"{reanalysis['per_protocol_delta']:+.1%}",
        "paired-power": f"{pairing['paired_power']:.0%}",
        "unpaired-power": f"{pairing['unpaired_power']:.0%}",
        "chain-mismatch-row": str(chain["first_mismatch"]),
    }
    if unmatched is not None:
        matched_matrix = lab06_metrics["success_matrix"]
        measurement_values["unmatched-unseen"] = f"{unmatched['unseen_success']:.1%}"
        measurement_values["matched-recovery-unseen"] = (
            f"{matched_matrix['recovery']['unseen']:.1%}"
        )
        measurement_values["matched-extra-unseen"] = (
            f"{matched_matrix['extra']['unseen']:.1%}"
        )
    export_typst_values(paths, "measurement_facts", measurement_values)

    metrics = {
        "delivery_bias_simulation": {
            key: value
            for key, value in simulation.items()
            if not isinstance(value, np.ndarray)
        },
        "pairing_simulation": {
            key: value
            for key, value in pairing.items()
            if not isinstance(value, np.ndarray)
        },
        "lab06_reanalysis": reanalysis,
        "hash_chain_demo": chain,
        "contract_hash_demo": hash_demo,
        "unmatched_arm": unmatched,
    }
    metrics_hash = write_metrics(paths, metrics)

    if unmatched is None:
        unmatched_text = "skipped"
    else:
        matched_matrix = lab06_metrics["success_matrix"]
        unmatched_text = (
            f"We deliberately broke the match: the recovery arm reran with "
            f"{unmatched['budget_labels']} labels (twice the budget), landing "
            f"at {unmatched['unseen_success']:.1%} unseen success against "
            f"{matched_matrix['recovery']['unseen']:.1%} matched "
            f"(`figures/unmatched_confound.svg`). The feared inflation did "
            "*not* materialize here, because this base policy sits near its headroom "
            "ceiling (Lab 6), so even doubled supervision buys nothing "
            "measurable in a single replicate. That is the deeper point: an "
            "unmatched design attributes to the *method* whatever the extra "
            "*budget* did or did not do, and a small unfrozen run cannot even "
            "tell you which way the confound cuts. Budget matching is a "
            "design necessity for attribution, not an empirical convenience."
        )
    write_mini_report(
        paths,
        question="What is intention-to-treat, and why budgets, pairing, and freezing?",
        sections=[
            (
                "Intention-to-treat, defined",
                "Analyze by what was *scheduled*, not by what happened. In the "
                "study every evaluation episode has a scheduled corruption "
                "time; sometimes the episode ends first and the corruption is "
                "never delivered. Delivery depends on the policy's own "
                "behavior, so it is a post-treatment variable, and conditioning "
                "on it compares filtered, non-comparable subsets.",
            ),
            (
                "The bias, with known ground truth",
                f"A simulation ({SIM_SCENARIOS} scenarios, {SIM_REPLICATES} "
                "replicates) where the true ITT effect is "
                f"{simulation['true_itt_effect']:+.3f} by construction: the ITT "
                f"estimator's bias is {simulation['itt_bias']:+.4f}; the "
                "per-protocol estimator (delivered episodes only) is off by "
                f"{simulation['per_protocol_bias']:+.4f}, a bias of the same "
                "order as the effects being measured "
                "(`figures/itt_bias.svg`). On Lab 6's real rows the two "
                f"estimates agree ({reanalysis['itt_delta']:+.1%} ITT vs "
                f"{reanalysis['per_protocol_delta']:+.1%} delivered-only) "
                "because delivery is nearly universal there. The design still "
                "preregisters ITT, because delivery *could* differ between "
                "arms and nothing in the data would warn you.",
            ),
            (
                "Budget matching",
                "The study's currency is revealed oracle labels. " + unmatched_text,
            ),
            (
                "Why paired replicates",
                "Every pipeline bundle re-rolls collection and training, "
                "shifting both arms together; analyzing per-bundle *differences* "
                "cancels that shared noise. In simulation at "
                f"{pairing['n_bundles']} bundles the paired analysis detects a "
                f"{pairing['true_gap']:+.2f} gap with {pairing['paired_power']:.0%} "
                f"power versus {pairing['unpaired_power']:.0%} unpaired "
                "(`figures/paired_power.svg`). This is why the study runs six "
                "bundles and reports the paired-t interval, with a cluster "
                "bootstrap as sensitivity.",
            ),
            (
                "Freezing, and what it is made of",
                "A frozen protocol is a set of mechanical commitments, not a "
                "promise (`freeze_mechanisms` table). Two of them, in "
                "miniature: flipping a single contract field (`data.h`) changes "
                f"the canonical hash from `{hash_demo['original_prefix']}…` to "
                f"`{hash_demo['edited_prefix']}…`, so no quiet edit survives; "
                "and editing one row of a five-row hash-chained ledger is "
                f"caught at exactly row {chain['first_mismatch']} when the "
                "chain is recomputed.",
            ),
            (
                "Bridge to the study",
                "The study's endpoint is the eligible unseen one-corruption "
                "ITT success difference, analyzed as a paired t interval "
                "across six bundles against a 0.05 smallest-effect-of-interest, "
                "under a preregistered claim decision rule, on data whose "
                "ledgers recount cleanly and whose test set was opened exactly "
                "once against a receipt. Every one of those words is one of "
                "this lab's demonstrations, done at full scale.",
            ),
        ],
    )

    return {
        "sim_pp_bias": f"{simulation['per_protocol_bias']:+.4f}",
        "sim_itt_bias": f"{simulation['itt_bias']:+.4f}",
        "real_itt_delta": f"{reanalysis['itt_delta']:+.1%}",
        "paired_power": f"{pairing['paired_power']:.0%}",
        "unmatched_unseen": "skipped" if unmatched is None
        else f"{unmatched['unseen_success']:.1%}",
        "metrics_hash": metrics_hash,
    }
