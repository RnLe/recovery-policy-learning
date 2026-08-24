"""Lab 6: when cloning breaks, with distribution shift, corruptions, recovery.

Behavior cloning is trained on the expert's states but deployed on its own.
This lab makes that gap measurable and then previews the study's remedy
comparison at small scale. It (a) corrupts one executed action of a cloned
policy and measures how success decays with corruption time and how the
trajectory diverges from the expert path afterwards, the compounding-error
picture; (b) derives the corruption operator family (for three actions,
exactly two derangements exist, the reason the action set is frozen); and
(c) runs a mini three-arm study: the same base policy improved either with
extra nominal demonstrations or with oracle labels at learner-visited
post-corruption states, under an exactly matched additional-label budget,
evaluated clean / matched-corruption / unseen-corruption with intention-to-
treat scheduling. The per-update exposure mismatch this naive version leaks
is measured and disclosed, and closing it is exactly why the study trains on
one-target-per-window items (Lab 7).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
from gr_foundations.training import (
    START_ACTION_TOKEN,
    BCEpisode,
    build_bc_dataset,
    collate_episodes,
    contract_config,
    dataset_vocabulary,
    masked_step_cross_entropy,
    resolve_device,
    save_checkpoint,
    train_bc,
)
from grounded_recovery.artifacts import atomic_write_json
from grounded_recovery.config import EnvironmentConfig
from grounded_recovery.data import Vocabulary
from grounded_recovery.model import RecoveryPolicy
from grounded_recovery.oracle import OracleSupportError, SynchronizedOracle
from grounded_recovery.perturbations import ActionDerangement, enumerate_derangements
from grounded_recovery.world import WorldSession

BASE_EPISODES = 150
HOLDOUT_SCENARIOS = 150
BUDGET_LABELS = 400
RECOVERY_WINDOW = 8
BASE_UPDATES = 1500
ARM_UPDATES = 600
BATCH_BASE = 12
BATCH_ARM = 4
N_SEEDS = 3
SWEEP_TIMES = (2, 4, 6, 8, 10)
DIVERGENCE_HORIZON = 10

ARM_NAMES = ("base", "extra", "recovery")
ARM_COLORS = {"base": COLOR_BASE, "extra": COLOR_EXTRA, "recovery": COLOR_RECOVERY}
SLICES = ("clean", "matched", "unseen")


def _operator(config_operator, action_ids: tuple[int, ...]) -> ActionDerangement:
    return ActionDerangement(
        name=config_operator.name,
        action_ids=action_ids,
        mapping=tuple(config_operator.mapping),
    )


# --------------------------------------------------------------------------
# Rollouts under corruption.
# --------------------------------------------------------------------------


@torch.no_grad()
def rollout_policy(
    model: torch.nn.Module,
    vocab: Vocabulary,
    env_cfg: EnvironmentConfig,
    seed: int,
    device: torch.device,
    *,
    corruption: tuple[ActionDerangement, int] | None = None,
    capture_positions: bool = False,
) -> dict[str, object]:
    """Greedy closed-loop rollout with an optional one-step action corruption.

    Intention-to-treat semantics: the corruption is *scheduled*; whether the
    episode lived long enough for it to be delivered is recorded, and the
    outcome counts either way.
    """
    model.eval()
    session = WorldSession(env_cfg)
    delivered = False
    positions: list[tuple[int, int]] = []
    try:
        result = session.reset(seed)
        tokens = vocab.encode(result.mission)
        mission_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        mission_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
        mission_feature = model.encode_mission(mission_tokens, mission_lengths)
        hidden: torch.Tensor | None = None
        prev = torch.tensor([START_ACTION_TOKEN], dtype=torch.long, device=device)
        while not session.done:
            if capture_positions:
                unwrapped = session.env.unwrapped
                positions.append((int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])))
            image = torch.from_numpy(result.image.astype(np.int64)).to(device)
            direction = torch.tensor([result.direction], dtype=torch.long, device=device)
            logits, hidden = model.step(
                image.unsqueeze(0), direction, prev, mission_feature, hidden
            )
            action = int(torch.argmax(logits, dim=-1).item())
            if corruption is not None and session.time == corruption[1]:
                action = corruption[0].apply(action)
                delivered = True
            result = session.step(action)
            # The policy is told what was *executed*, corruption included.
            prev = torch.tensor([action], dtype=torch.long, device=device)
        return {
            "success": bool(result.terminated and result.reward > 0.0),
            "steps": session.time,
            "delivered": delivered,
            "positions": positions,
        }
    finally:
        session.close()


def _positions_for_actions(
    env_cfg: EnvironmentConfig, seed: int, actions: np.ndarray
) -> list[tuple[int, int]]:
    """Agent positions along a stored action sequence (deterministic replay)."""
    session = WorldSession(env_cfg)
    try:
        session.reset(seed)
        unwrapped = session.env.unwrapped
        positions = [(int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]))]
        for action in actions:
            session.step(int(action))
            positions.append((int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])))
        return positions
    finally:
        session.close()


# --------------------------------------------------------------------------
# Budgeted collection: the two remedies.
# --------------------------------------------------------------------------


def collect_extra_demos(
    env_cfg: EnvironmentConfig, budget_labels: int, seed_offset: int
) -> tuple[list[BCEpisode], dict[str, int]]:
    """Fresh nominal demonstrations, truncated at exactly ``budget_labels``."""
    episodes: list[BCEpisode] = []
    counters = {"episodes": 0, "labels": 0, "oracle_unsupported": 0, "truncated_final": 0}
    index = 0
    while counters["labels"] < budget_labels:
        seed = derive_seed("lab06.collection", seed_offset + index)
        index += 1
        session = WorldSession(env_cfg)
        try:
            result = session.reset(seed)
            oracle = SynchronizedOracle(session)
            images, directions, actions = [], [], []
            last: int | None = None
            try:
                while not session.done:
                    recommendation = oracle.recommend(last, session.time)
                    images.append(result.image.copy())
                    directions.append(result.direction)
                    actions.append(recommendation)
                    result = session.step(recommendation)
                    last = recommendation
            except OracleSupportError:
                counters["oracle_unsupported"] += 1
                continue
        finally:
            session.close()
        remaining = budget_labels - counters["labels"]
        keep = min(len(actions), remaining)
        if keep < len(actions):
            counters["truncated_final"] += 1
        executed = np.asarray(actions[:keep], dtype=np.int64)
        episodes.append(
            BCEpisode(
                seed=seed,
                mission=result.mission,
                images=np.stack(images[:keep]).astype(np.uint8),
                directions=np.asarray(directions[:keep], dtype=np.int64),
                actions=executed,
                success=bool(result.terminated and result.reward > 0.0),
                target_actions=executed.copy(),
                label_mask=np.ones(keep, dtype=bool),
            )
        )
        counters["episodes"] += 1
        counters["labels"] += keep
    if counters["labels"] != budget_labels:
        raise FoundationsError("extra-demo collection missed its exact budget")
    return episodes, counters


@torch.no_grad()
def collect_recovery(
    model: torch.nn.Module,
    vocab: Vocabulary,
    env_cfg: EnvironmentConfig,
    operator: ActionDerangement,
    budget_labels: int,
    seed_offset: int,
    times: tuple[int, ...],
    window: int,
    device: torch.device,
) -> tuple[list[BCEpisode], dict[str, int]]:
    """Learner-visited recovery states, labelled for ``window`` steps after one
    corrupted execution; exactly ``budget_labels`` labels are revealed."""
    model.eval()
    episodes: list[BCEpisode] = []
    counters = {
        "episodes_attempted": 0,
        "episodes_with_labels": 0,
        "labels": 0,
        "undelivered": 0,
        "oracle_unsupported": 0,
    }
    index = 0
    while counters["labels"] < budget_labels:
        seed = derive_seed("lab06.collection", seed_offset + index)
        schedule_rng = np.random.default_rng(derive_seed("lab06.collection", seed_offset + index))
        t_star = int(schedule_rng.choice(times))
        index += 1
        counters["episodes_attempted"] += 1
        session = WorldSession(env_cfg)
        try:
            result = session.reset(seed)
            oracle = SynchronizedOracle(session)
            tokens = vocab.encode(result.mission)
            mission_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
            mission_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
            mission_feature = model.encode_mission(mission_tokens, mission_lengths)
            hidden: torch.Tensor | None = None
            prev = torch.tensor([START_ACTION_TOKEN], dtype=torch.long, device=device)
            images, directions, executed_actions = [], [], []
            targets: list[int] = []
            mask: list[bool] = []
            last: int | None = None
            remaining = budget_labels - counters["labels"]
            try:
                while not session.done:
                    time = session.time
                    recommendation = oracle.recommend(last, time)
                    in_window = t_star < time <= t_star + window
                    reveal = in_window and sum(mask) < remaining
                    image = torch.from_numpy(result.image.astype(np.int64)).to(device)
                    direction = torch.tensor(
                        [result.direction], dtype=torch.long, device=device
                    )
                    logits, hidden = model.step(
                        image.unsqueeze(0), direction, prev, mission_feature, hidden
                    )
                    action = int(torch.argmax(logits, dim=-1).item())
                    if time == t_star:
                        action = operator.apply(action)
                    images.append(result.image.copy())
                    directions.append(result.direction)
                    executed_actions.append(action)
                    targets.append(recommendation if reveal else action)
                    mask.append(bool(reveal))
                    result = session.step(action)
                    prev = torch.tensor([action], dtype=torch.long, device=device)
                    last = action
                    if time >= t_star + window or sum(mask) >= remaining:
                        break
            except OracleSupportError:
                counters["oracle_unsupported"] += 1
                continue
        finally:
            session.close()
        revealed = int(np.sum(mask))
        if revealed == 0:
            counters["undelivered"] += 1
            continue
        episodes.append(
            BCEpisode(
                seed=seed,
                mission=result.mission,
                images=np.stack(images).astype(np.uint8),
                directions=np.asarray(directions, dtype=np.int64),
                actions=np.asarray(executed_actions, dtype=np.int64),
                success=False,  # outcome irrelevant for collection
                target_actions=np.asarray(targets, dtype=np.int64),
                label_mask=np.asarray(mask, dtype=bool),
            )
        )
        counters["episodes_with_labels"] += 1
        counters["labels"] += revealed
    if counters["labels"] != budget_labels:
        raise FoundationsError("recovery collection missed its exact budget")
    return episodes, counters


# --------------------------------------------------------------------------
# Arm training: same start, same optimization, different data.
# --------------------------------------------------------------------------


def finetune_arm(
    base_state: dict[str, torch.Tensor],
    model_factory,
    base_episodes: list[BCEpisode],
    arm_episodes: list[BCEpisode],
    vocab: Vocabulary,
    *,
    updates: int,
    seed: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Continue from the shared base weights on a fixed base/arm batch mix."""
    from gr_foundations.training import ensure_determinism

    ensure_determinism()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sampler = np.random.default_rng(seed)
    model = model_factory()
    model.load_state_dict(base_state)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    arm_label_counts: list[int] = []
    for _update in range(updates):
        base_pick = sampler.integers(0, len(base_episodes), size=BATCH_BASE)
        arm_pick = sampler.integers(0, len(arm_episodes), size=BATCH_ARM)
        batch_episodes = [base_episodes[i] for i in base_pick] + [
            arm_episodes[i] for i in arm_pick
        ]
        batch = collate_episodes(batch_episodes, vocab, device)
        logits, _ = model(
            batch["image"],
            batch["direction"],
            batch["prev_action"],
            batch["mission_tokens"],
            batch["mission_lengths"],
            batch["step_mask"],
        )
        loss = masked_step_cross_entropy(logits, batch["targets"], batch["target_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        arm_label_counts.append(
            int(sum(int(arm_episodes[i].label_mask.sum()) for i in arm_pick))
        )
    model.eval()
    return model, {"mean_arm_labels_per_update": float(np.mean(arm_label_counts))}


# --------------------------------------------------------------------------
# The lab run.
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _study_headline(repo_root: Path) -> str:
    """The confirmatory result, read from the published bundle rather than typed.

    Lab 6 previews the study's comparison at toy scale, so it has to quote the
    real answer. Reading it from the evidence bundle keeps the track's rule
    intact and means a re-analysis can never leave this sentence stale.
    """
    import json

    path = Path(repo_root) / "public_result" / "experiment-summary.json"
    if not path.exists():
        return "the study itself"
    summary = json.loads(path.read_text(encoding="utf-8"))
    primary = summary["primary_summary"]
    interval = primary["interval"]
    return (
        f"the study itself ({primary['mean_paired_difference'] * 100:+.1f}pp, "
        f"95% interval {interval['lower'] * 100:+.1f} to "
        f"{interval['upper'] * 100:+.1f}pp, over "
        f"{primary['pipeline_replicates']} pipeline bundles and "
        f"{summary['eligibility']['eligible_scenarios']} eligible scenarios)"
    )


def run(
    paths: LabPaths,
    *,
    force: bool,
    base_episodes: int = BASE_EPISODES,
    holdout_scenarios: int = HOLDOUT_SCENARIOS,
    budget_labels: int = BUDGET_LABELS,
    window: int = RECOVERY_WINDOW,
    base_updates: int = BASE_UPDATES,
    arm_updates: int = ARM_UPDATES,
    n_seeds: int = N_SEEDS,
    sweep_times: tuple[int, ...] = SWEEP_TIMES,
    sweep_scenarios: int | None = None,
) -> dict[str, object]:
    prepare(paths, force=force)
    contract = contract_config(paths.repo_root)
    env_cfg = contract.environment
    device = resolve_device()
    action_ids = tuple(env_cfg.action_ids)
    op_collect = _operator(contract.perturbation.collection_operator, action_ids)
    op_unseen = _operator(contract.perturbation.unseen_operator, action_ids)
    times = tuple(contract.perturbation.collection_time_set or (2, 4, 6, 8))
    sweep_count = sweep_scenarios if sweep_scenarios is not None else holdout_scenarios

    dataset, _ = build_bc_dataset(env_cfg, base_episodes, "lab06.dataset")
    holdout, _ = build_bc_dataset(env_cfg, holdout_scenarios, "lab06.holdout")
    vocab = dataset_vocabulary(dataset)
    base_label_count = int(sum(len(e.actions) for e in dataset))

    def factory():
        return RecoveryPolicy(contract.model, vocab.size, len(action_ids))

    # ---- replicates: base -> clone -> two budget-matched arms -------------
    rows: list[dict[str, object]] = []
    exposure: dict[str, list[float]] = {"extra": [], "recovery": []}
    collection_counters: list[dict[str, object]] = []
    rep_models: list[dict[str, torch.nn.Module]] = []
    for rep in range(n_seeds):
        base_model, _log = train_bc(
            factory,
            dataset,
            vocab,
            updates=base_updates,
            batch_episodes=BATCH_BASE + BATCH_ARM,
            seed=derive_seed("lab06.train", rep),
            device=device,
        )
        base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

        extra_data, extra_counters = collect_extra_demos(
            env_cfg, budget_labels, seed_offset=rep * 10_000
        )
        recovery_data, recovery_counters = collect_recovery(
            base_model,
            vocab,
            env_cfg,
            op_collect,
            budget_labels,
            seed_offset=rep * 10_000 + 5_000,
            times=times,
            window=window,
            device=device,
        )
        collection_counters.append(
            {"rep": rep, "extra": extra_counters, "recovery": recovery_counters}
        )

        arms: dict[str, torch.nn.Module] = {"base": base_model}
        for arm_index, (arm, arm_data) in enumerate(
            (("extra", extra_data), ("recovery", recovery_data))
        ):
            arm_model, arm_stats = finetune_arm(
                base_state,
                factory,
                dataset,
                arm_data,
                vocab,
                updates=arm_updates,
                seed=derive_seed("lab06.train", 100 + rep * 10 + arm_index),
                device=device,
            )
            exposure[arm].append(arm_stats["mean_arm_labels_per_update"])
            arms[arm] = arm_model
        rep_models.append(arms)
        for arm, arm_model in arms.items():
            save_checkpoint(
                paths.data_dir / "checkpoints" / f"{arm}_r{rep}.pt",
                arm_model,
                {
                    "lab": "lab06",
                    "arm": arm,
                    "replicate": rep,
                    "vocabulary": list(vocab.tokens),
                },
            )

        # ---- crossed ITT evaluation over identical scenarios --------------
        for scenario_index, episode in enumerate(holdout):
            t_star = int(
                np.random.default_rng(
                    derive_seed("lab06.sweep", scenario_index)
                ).choice(times)
            )
            plans = {
                "clean": None,
                "matched": (op_collect, t_star),
                "unseen": (op_unseen, t_star),
            }
            for arm, model in arms.items():
                for slice_name, corruption in plans.items():
                    outcome = rollout_policy(
                        model, vocab, env_cfg, episode.seed, device,
                        corruption=corruption,
                    )
                    rows.append(
                        {
                            "rep": rep,
                            "arm": arm,
                            "slice": slice_name,
                            "scenario": scenario_index,
                            "seed": episode.seed,
                            "scheduled_time": t_star if corruption else None,
                            "delivered": outcome["delivered"],
                            "success": outcome["success"],
                            "steps": outcome["steps"],
                            "oracle_steps": int(len(episode.actions)),
                        }
                    )

    # ---- corruption-time sweep + divergence on the first base policy ------
    sweep_base = rep_models[0]["base"]
    sweep_rows: list[dict[str, object]] = []
    divergence: dict[str, dict[int, list[float]]] = {"corrupted": {}, "clean": {}}
    for episode in holdout[:sweep_count]:
        oracle_positions = _positions_for_actions(env_cfg, episode.seed, episode.actions)
        for label, t_values in (("clean", (None,)), ("corrupted", sweep_times)):
            for t_star in t_values:
                corruption = None if t_star is None else (op_collect, int(t_star))
                outcome = rollout_policy(
                    sweep_base, vocab, env_cfg, episode.seed, device,
                    corruption=corruption, capture_positions=True,
                )
                if t_star is not None:
                    sweep_rows.append(
                        {
                            "t_star": int(t_star),
                            "delivered": outcome["delivered"],
                            "success": outcome["success"],
                        }
                    )
                align = 4 if t_star is None else int(t_star)  # clean control aligned mid-range
                if t_star is not None and not outcome["delivered"]:
                    continue
                policy_positions = outcome["positions"]
                for k in range(DIVERGENCE_HORIZON + 1):
                    t = align + k
                    if t >= len(policy_positions):
                        break
                    oracle_position = oracle_positions[min(t, len(oracle_positions) - 1)]
                    distance = float(
                        abs(policy_positions[t][0] - oracle_position[0])
                        + abs(policy_positions[t][1] - oracle_position[1])
                    )
                    divergence[label].setdefault(k, []).append(distance)

    atomic_write_json(paths.data_dir / "evaluation_rows.json", rows)

    # ---- aggregation ------------------------------------------------------
    def slice_rate(arm: str, slice_name: str, rep: int | None = None) -> float:
        selected = [
            row["success"]
            for row in rows
            if row["arm"] == arm
            and row["slice"] == slice_name
            and (rep is None or row["rep"] == rep)
        ]
        return _mean([float(s) for s in selected])

    per_rep_deltas = [
        slice_rate("recovery", "unseen", rep) - slice_rate("extra", "unseen", rep)
        for rep in range(n_seeds)
    ]
    matrix = {
        arm: {s: slice_rate(arm, s) for s in SLICES} for arm in ARM_NAMES
    }
    delivered_rate = _mean(
        [
            float(row["delivered"])
            for row in rows
            if row["slice"] != "clean"
        ]
    )

    # ---- figures ----------------------------------------------------------
    fig, axis = plt.subplots(figsize=(9.2, 4.3))
    positions_x = np.arange(len(SLICES))
    width = 0.26
    for arm_index, arm in enumerate(ARM_NAMES):
        offsets = positions_x + (arm_index - 1) * width
        means = [matrix[arm][s] for s in SLICES]
        axis.bar(offsets, means, width, color=ARM_COLORS[arm], label=arm)
        for rep in range(n_seeds):
            reps = [slice_rate(arm, s, rep) for s in SLICES]
            axis.scatter(
                offsets, reps, color="black", s=12, zorder=3,
            )
    axis.set_xticks(positions_x)
    axis.set_xticklabels(
        ["clean", f"matched corruption\n({op_collect.name})",
         f"unseen corruption\n({op_unseen.name})"],
        fontsize=11,
    )
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("ITT success rate")
    axis.legend(fontsize=12)
    axis.set_title(
        f"mini three-arm preview: +{budget_labels} labels per arm, "
        f"{n_seeds} replicates (dots), {holdout_scenarios} scenarios each",
        fontsize=13,
    )
    save_figure(paths, fig, "three_arm_results.svg")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.0))
    sweep_success = []
    for t_star in sweep_times:
        delivered = [r for r in sweep_rows if r["t_star"] == t_star and r["delivered"]]
        sweep_success.append(_mean([float(r["success"]) for r in delivered]))
    clean_rate = matrix["base"]["clean"]
    left.plot(sweep_times, sweep_success, marker="o", color=COLOR_CAUTION,
              label="one corruption at t*")
    left.axhline(clean_rate, color=COLOR_BASE, linestyle="--", linewidth=1.0,
                 label=f"clean ({clean_rate:.0%})")
    left.set_xlabel("corruption time t*")
    left.set_ylabel("success (delivered episodes)")
    left.set_ylim(0, 1.05)
    left.legend(fontsize=12)
    left.set_title("one wrong executed action costs real success", fontsize=12)
    for label, color in (("corrupted", COLOR_CAUTION), ("clean", COLOR_BASE)):
        ks = sorted(divergence[label])
        right.plot(
            ks,
            [_mean(divergence[label][k]) for k in ks],
            marker="o",
            color=color,
            label=f"{label} rollouts",
        )
    right.set_xlabel("steps after the (aligned) corruption time")
    right.set_ylabel("mean |position - expert path| (L1)")
    right.legend(fontsize=12)
    right.set_title("after the corruption, the state distribution drifts", fontsize=12)
    fig.tight_layout()
    save_figure(paths, fig, "shift_anatomy.svg")

    # ---- tables and values ------------------------------------------------
    names = env_cfg.action_names
    derangement_rows = []
    for mapping in enumerate_derangements(action_ids):
        operator_name = {
            tuple(op_collect.mapping): f"{op_collect.name} (collection)",
            tuple(op_unseen.mapping): f"{op_unseen.name} (unseen)",
        }.get(tuple(mapping), "unnamed")
        derangement_rows.append(
            [
                str(mapping),
                ", ".join(
                    f"{names[i]}->{names[mapping[i]]}" for i in range(len(names))
                ),
                operator_name,
            ]
        )
    export_typst_table(
        paths,
        "derangements",
        ["mapping", "effect", "role in the study"],
        derangement_rows,
    )
    results_rows = [
        [arm] + [f"{matrix[arm][s]:.1%}" for s in SLICES] for arm in ARM_NAMES
    ]
    export_typst_table(
        paths, "three_arm_results", ["arm", "clean", "matched", "unseen"], results_rows
    )
    write_table_csv(
        paths, "three_arm_results.csv", ["arm", "clean", "matched", "unseen"], results_rows
    )
    write_table_csv(
        paths,
        "per_rep_unseen.csv",
        ["rep", "base", "extra", "recovery", "recovery_minus_extra"],
        [
            [
                rep,
                f"{slice_rate('base', 'unseen', rep):.3f}",
                f"{slice_rate('extra', 'unseen', rep):.3f}",
                f"{slice_rate('recovery', 'unseen', rep):.3f}",
                f"{per_rep_deltas[rep]:+.3f}",
            ]
            for rep in range(n_seeds)
        ],
    )
    export_typst_values(
        paths,
        "shift_facts",
        {
            "budget-labels": str(budget_labels),
            "base-labels": str(base_label_count),
            "recovery-window": str(window),
            "delta-unseen-mean": f"{_mean(per_rep_deltas):+.1%}",
            "delta-unseen-min": f"{min(per_rep_deltas):+.1%}",
            "delta-unseen-max": f"{max(per_rep_deltas):+.1%}",
            "delivered-rate": f"{delivered_rate:.1%}",
            "base-corruption-dent": (
                f"{matrix['base']['clean'] - matrix['base']['matched']:+.1%}"
            ),
            "exposure-extra": f"{_mean(exposure['extra']):.1f}",
            "exposure-recovery": f"{_mean(exposure['recovery']):.1f}",
        },
    )

    metrics = {
        "design": {
            "base_episodes": base_episodes,
            "base_labels": base_label_count,
            "budget_labels": budget_labels,
            "window": window,
            "times": list(times),
            "operators": {"collection": op_collect.name, "unseen": op_unseen.name},
            "arm_updates": arm_updates,
            "batch_mix": [BATCH_BASE, BATCH_ARM],
            "replicates": n_seeds,
            "device": device.type,
        },
        "collection": collection_counters,
        "exposure_mean_arm_labels_per_update": {
            arm: _mean(values) for arm, values in exposure.items()
        },
        "success_matrix": matrix,
        "per_rep_unseen_delta": per_rep_deltas,
        "delivered_rate": delivered_rate,
        "sweep": {
            "times": list(sweep_times),
            "delivered_success": sweep_success,
            "clean_success": clean_rate,
        },
    }
    metrics_hash = write_metrics(paths, metrics)

    write_mini_report(
        paths,
        question="Why does cloning break under its own mistakes, and what are recovery labels?",
        sections=[
            (
                "The failure mode, measured",
                "Behavior cloning fits the expert's state distribution; at "
                "deployment the policy visits its *own* states, and each error "
                "feeds the next (Ross & Bagnell's compounding-error argument). "
                "`figures/shift_anatomy.svg`: a single corrupted executed action "
                f"drops the base policy from {clean_rate:.0%} (clean) to the "
                "shown per-t* success on delivered episodes, and the mean "
                "distance to the expert path grows step by step after the "
                "corruption while clean rollouts stay close.",
            ),
            (
                "The corruption operators, and why the action set is frozen",
                "A corruption operator must change every action it touches "
                "(otherwise some corruptions would be no-ops), i.e. it must be a "
                "*derangement* of the action set. For three actions exactly two "
                "derangements exist: the two 3-cycles, each the other's inverse "
                "(`derangements` table): one is used during collection, the "
                "other is held out as the unseen operator. Changing the action "
                "set would change this entire operator family, and enlarging it "
                "would break oracle support (with closed doors the bot emits "
                "`toggle`, outside the frozen set, the study's `doors_open` "
                "pilot discovery, Lab 1). That is why the action set is a frozen "
                "contract field, not a tuning knob.",
            ),
            (
                "Two remedies, one budget",
                f"Both arms start from the *same* base weights (trained on "
                f"{base_label_count} labels) and receive exactly "
                f"{budget_labels} additional oracle labels. The extra arm "
                "spends them on fresh nominal demonstrations; the recovery arm "
                "spends them on learner-visited states: the base policy rolls "
                "out, one action gets corrupted, and the oracle labels the next "
                f"{window} states it actually reaches. Optimization is matched "
                f"({arm_updates} updates, fixed {BATCH_BASE}+{BATCH_ARM} "
                "base/arm batch mix). One fairness leak is deliberately left "
                "open and *measured*: an extra-demo batch carries "
                f"{_mean(exposure['extra']):.1f} new labels per update versus "
                f"{_mean(exposure['recovery']):.1f} for recovery episodes, "
                "because full demonstrations are label-dense while recovery "
                "windows are sparse. The study closes exactly this leak with "
                "one-target-per-window training items, as Lab 7 explains.",
            ),
            (
                "What the mini-study shows",
                f"ITT success over {holdout_scenarios} unseen scenarios x "
                f"{n_seeds} replicates (`figures/three_arm_results.svg`): "
                f"recovery reaches {matrix['recovery']['unseen']:.1%} on the "
                f"unseen-operator slice versus {matrix['extra']['unseen']:.1%} "
                f"for extra demonstrations and {matrix['base']['unseen']:.1%} "
                "for the untouched base; per-replicate recovery-minus-extra "
                "deltas: "
                + ", ".join(f"{delta:+.1%}" for delta in per_rep_deltas)
                + ". Two honest observations. First, one corruption dents this "
                "base policy by only "
                f"{matrix['base']['clean'] - matrix['base']['matched']:+.1%} on "
                "the ITT matched slice, and a strong base leaves little headroom "
                "for either remedy, one reason the study's frozen protocol "
                "checks perturbed competence and evaluates a far larger panel. "
                "Second, three replicates at this scale cannot even fix the "
                "*sign* of the difference, the small-n lesson Lab 7 turns "
                "into design requirements. The confirmatory answer, under the "
                "frozen protocol, is "
                f"{_study_headline(paths.repo_root)}.",
            ),
            (
                "Bridge to the study",
                "Everything here is a scaled-down mirror of "
                "`grounded_recovery.experiment`: the study adds bit-exact arm "
                "cloning from a shared checkpoint, hash-chained exposure "
                "ledgers with a fairness audit, exact window-level target "
                "accounting, preregistered corruption time sets, and the "
                "one-opening evaluation discipline, the honest-measurement "
                "machinery that Lab 7 walks through.",
            ),
        ],
    )

    return {
        "unseen": {arm: f"{matrix[arm]['unseen']:.1%}" for arm in ARM_NAMES},
        "per_rep_deltas": [f"{d:+.1%}" for d in per_rep_deltas],
        "delivered_rate": f"{delivered_rate:.1%}",
        "device": device.type,
        "metrics_hash": metrics_hash,
    }
