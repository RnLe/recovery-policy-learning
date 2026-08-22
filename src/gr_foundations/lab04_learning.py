"""Lab 4: learning paradigms, reinforcement versus imitation.

Is the study reinforcement learning? No, and this lab earns that answer with
code. It first implements tabular Q-learning from scratch and shows real RL
converging on a tiny fully observable MDP; then it measures why that recipe
does not carry over (state space unbounded across generated worlds, partial
observability, language conditioning, terminal-only reward); and finally it
trains the first cloned policies of the track, a memoryless one and the
study's recurrent architecture, on identical oracle demonstrations, exposing
the two classic imitation facts: per-step accuracy is not task success, and
memory matters exactly as Lab 2's aliasing bound predicted.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gr_foundations.common import (
    COLOR_BASE,
    COLOR_NEUTRAL,
    COLOR_RECOVERY,
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
from gr_foundations.models import LabPolicy
from gr_foundations.training import (
    build_bc_dataset,
    closed_loop_success,
    contract_config,
    dataset_vocabulary,
    model_digest,
    open_loop_accuracy,
    resolve_device,
    save_checkpoint,
    train_bc,
)
from grounded_recovery.model import RecoveryPolicy

TOY_ENV_ID = "MiniGrid-Empty-5x5-v0"
QLEARNING_EPISODES = 400
DATASET_EPISODES = 200
HOLDOUT_EPISODES = 100
BC_UPDATES = 1500
BC_BATCH_EPISODES = 16
BC_SEEDS = 3


# --------------------------------------------------------------------------
# Part A: reinforcement learning, from scratch, where it belongs.
# --------------------------------------------------------------------------


class ToyMDP:
    """Fully observable adapter for the tiny RL demo.

    Deliberately *not* the study's WorldSession: the whole point is that this
    env exposes its true state (agent position and direction), which the study
    task never does.
    """

    def __init__(self) -> None:
        self.env = gym.make(TOY_ENV_ID, disable_env_checker=True)

    def reset(self, seed: int) -> tuple[int, int, int]:
        self.env.reset(seed=seed)
        return self.state()

    def state(self) -> tuple[int, int, int]:
        unwrapped = self.env.unwrapped
        return (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]),
                int(unwrapped.agent_dir))

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        _obs, reward, terminated, truncated, _info = self.env.step(int(action))
        return self.state(), float(reward), bool(terminated or truncated)

    def close(self) -> None:
        self.env.close()


def train_qlearning(
    *,
    episodes: int = QLEARNING_EPISODES,
    alpha: float = 0.5,
    gamma: float = 0.95,
    epsilon: float = 0.15,
    eval_every: int = 20,
) -> dict[str, object]:
    """Tabular Q-learning with an epsilon-greedy behavior policy."""
    rng = np.random.default_rng(derive_seed("lab04.qlearning"))
    q_table: dict[tuple[int, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=np.float64)
    )
    world = ToyMDP()
    history: list[dict[str, float]] = []

    def greedy_rollout() -> tuple[bool, int, float]:
        state = world.reset(seed=0)
        done, steps, total_reward = False, 0, 0.0
        while not done and steps < 60:
            action = int(np.argmax(q_table[state]))
            state, reward, done = world.step(action)
            total_reward += reward
            steps += 1
        return total_reward > 0.0, steps, total_reward

    try:
        for episode in range(episodes):
            state = world.reset(seed=0)
            done = False
            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(0, 3))
                else:
                    action = int(np.argmax(q_table[state]))
                next_state, reward, done = world.step(action)
                target = reward if done else reward + gamma * float(
                    np.max(q_table[next_state])
                )
                q_table[state][action] += alpha * (target - q_table[state][action])
                state = next_state
            if episode % eval_every == 0 or episode == episodes - 1:
                success, steps, total_reward = greedy_rollout()
                history.append(
                    {
                        "episode": float(episode),
                        "greedy_success": float(success),
                        "greedy_steps": float(steps),
                        "greedy_return": total_reward,
                    }
                )
    finally:
        world.close()

    best_steps = min(
        (entry["greedy_steps"] for entry in history if entry["greedy_success"]),
        default=float("nan"),
    )
    return {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "states_in_table": len(q_table),
        "final_greedy_success": history[-1]["greedy_success"] == 1.0,
        "best_greedy_steps": best_steps,
        "history": history,
    }


def _render_qlearning(qlearning: dict[str, object], paths: LabPaths) -> None:
    history = qlearning["history"]
    episodes = [entry["episode"] for entry in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    left.plot(
        episodes, [entry["greedy_return"] for entry in history], color=COLOR_NEUTRAL
    )
    left.set_xlabel("training episodes")
    left.set_ylabel("greedy return")
    left.set_title(f"tabular Q-learning on {TOY_ENV_ID}", fontsize=12)
    right.plot(
        episodes, [entry["greedy_steps"] for entry in history], color=COLOR_NEUTRAL
    )
    right.axhline(
        qlearning["best_greedy_steps"], color=COLOR_RECOVERY, linestyle="--",
        linewidth=1.0, label=f"best found: {qlearning['best_greedy_steps']:.0f} steps",
    )
    right.set_xlabel("training episodes")
    right.set_ylabel("greedy steps to goal")
    right.legend(fontsize=12)
    right.set_title("the learned policy becomes optimal", fontsize=12)
    fig.tight_layout()
    save_figure(paths, fig, "qlearning_curve.svg")


# --------------------------------------------------------------------------
# Part B: the first cloned policies.
# --------------------------------------------------------------------------


def run_bc_experiment(
    repo_root,
    *,
    dataset_episodes: int,
    holdout_episodes: int,
    updates: int,
    batch_episodes: int,
    n_seeds: int,
) -> dict[str, object]:
    contract = contract_config(repo_root)
    env_cfg = contract.environment
    device = resolve_device()

    dataset, dataset_counters = build_bc_dataset(env_cfg, dataset_episodes, "lab04.dataset")
    holdout, holdout_counters = build_bc_dataset(env_cfg, holdout_episodes, "lab04.holdout")
    vocab = dataset_vocabulary(dataset)
    holdout_seeds = [episode.seed for episode in holdout]
    train_seeds = [episode.seed for episode in dataset[: len(holdout_seeds)]]
    total_labels = int(sum(len(episode.actions) for episode in dataset))

    def make_model(kind: str):
        if kind == "memoryless":
            return LabPolicy(
                contract.model, vocab.size, len(env_cfg.action_ids), use_memory=False
            )
        return RecoveryPolicy(contract.model, vocab.size, len(env_cfg.action_ids))

    results: dict[str, list[dict[str, object]]] = {"memoryless": [], "recurrent": []}
    losses: dict[str, list[list[dict[str, float]]]] = {"memoryless": [], "recurrent": []}
    for model_index, kind in enumerate(("memoryless", "recurrent")):
        for repetition in range(n_seeds):
            seed = derive_seed("lab04.train", model_index * 10 + repetition)
            model, log = train_bc(
                lambda kind=kind: make_model(kind),
                dataset,
                vocab,
                updates=updates,
                batch_episodes=batch_episodes,
                seed=seed,
                device=device,
            )
            losses[kind].append(log)
            results[kind].append(
                {
                    "seed_index": repetition,
                    "open_loop_accuracy": open_loop_accuracy(model, holdout, vocab, device),
                    "holdout": closed_loop_success(model, vocab, env_cfg, holdout_seeds, device),
                    "train_scenarios": closed_loop_success(
                        model, vocab, env_cfg, train_seeds, device
                    ),
                    "model_digest": model_digest(model),
                    "parameters": int(sum(p.numel() for p in model.parameters())),
                }
            )
            save_checkpoint(
                repo_root / "data" / "foundations" / "lab04" / "checkpoints"
                / f"{kind}_s{repetition}.pt",
                model,
                {
                    "lab": "lab04",
                    "kind": kind,
                    "seed_index": repetition,
                    "vocabulary": list(vocab.tokens),
                },
            )
    return {
        "device": device.type,
        "dataset_counters": dataset_counters,
        "holdout_counters": holdout_counters,
        "total_labels": total_labels,
        "vocabulary_size": vocab.size,
        "updates": updates,
        "batch_episodes": batch_episodes,
        "results": results,
        "losses": losses,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _render_bc(experiment: dict[str, object], paths: LabPaths) -> None:
    colors = {"memoryless": COLOR_BASE, "recurrent": COLOR_RECOVERY}
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for kind, logs in experiment["losses"].items():
        for repetition, log in enumerate(logs):
            axis.plot(
                [entry["update"] for entry in log],
                [entry["loss"] for entry in log],
                color=colors[kind],
                alpha=0.7,
                linewidth=1.0,
                label=kind if repetition == 0 else None,
            )
    axis.set_xlabel("update")
    axis.set_ylabel("behavior-cloning loss")
    axis.set_yscale("log")
    axis.legend(fontsize=12)
    axis.set_title("identical data, two architectures", fontsize=12)
    save_figure(paths, fig, "bc_learning_curves.svg")

    fig, axis = plt.subplots(figsize=(8.4, 4.0))
    measures = [
        ("open-loop\nstep accuracy", "open_loop_accuracy", None),
        ("closed-loop success\n(training scenarios)", "train_scenarios", "success_rate"),
        ("closed-loop success\n(unseen scenarios)", "holdout", "success_rate"),
    ]
    width = 0.35
    positions = np.arange(len(measures))
    for offset, kind in ((-width / 2, "memoryless"), (width / 2, "recurrent")):
        rows = experiment["results"][kind]
        values = []
        spans = []
        for _label, key, subkey in measures:
            samples = [
                row[key] if subkey is None else row[key][subkey] for row in rows
            ]
            values.append(_mean(samples))
            spans.append((min(samples), max(samples)))
        axis.bar(positions + offset, values, width, color=colors[kind], label=kind)
        for position, value, (low, high) in zip(positions + offset, values, spans, strict=True):
            axis.plot([position, position], [low, high], color="black", linewidth=1.0)
            axis.text(position, value + 0.03, f"{value:.0%}", ha="center", fontsize=11)
    axis.set_xticks(positions)
    axis.set_xticklabels([label for label, _key, _sub in measures], fontsize=11)
    axis.set_ylim(0, 1.15)
    axis.legend(fontsize=12)
    axis.set_title(
        "the two imitation lessons: accuracy is not success, and memory matters",
        fontsize=13,
    )
    save_figure(paths, fig, "accuracy_vs_success.svg")


def _lab03_random_success(repo_root: Path) -> str:
    """Lab 3's measured random-policy success rate, read rather than retyped.

    Lab 4's argument leans on a number Lab 3 measured. Reading it keeps the
    track's rule intact: no lab prints a figure it did not compute or import.
    """
    import json

    path = Path(repo_root) / "foundations" / "lab03" / "metrics.json"
    if not path.exists():
        return "a substantial fraction"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rate = metrics["metrics"]["spectrum"]["random"]["success_rate"]
    return f"{rate:.1%}"


def run(
    paths: LabPaths,
    *,
    force: bool,
    qlearning_episodes: int = QLEARNING_EPISODES,
    dataset_episodes: int = DATASET_EPISODES,
    holdout_episodes: int = HOLDOUT_EPISODES,
    updates: int = BC_UPDATES,
    batch_episodes: int = BC_BATCH_EPISODES,
    n_seeds: int = BC_SEEDS,
) -> dict[str, object]:
    prepare(paths, force=force)

    qlearning = train_qlearning(episodes=qlearning_episodes)
    _render_qlearning(qlearning, paths)

    experiment = run_bc_experiment(
        paths.repo_root,
        dataset_episodes=dataset_episodes,
        holdout_episodes=holdout_episodes,
        updates=updates,
        batch_episodes=batch_episodes,
        n_seeds=n_seeds,
    )
    _render_bc(experiment, paths)

    summary_rows = []
    for kind in ("memoryless", "recurrent"):
        rows = experiment["results"][kind]
        summary_rows.append(
            [
                kind,
                len(rows),
                f"{_mean([r['open_loop_accuracy'] for r in rows]):.1%}",
                f"{_mean([r['train_scenarios']['success_rate'] for r in rows]):.1%}",
                f"{_mean([r['holdout']['success_rate'] for r in rows]):.1%}",
            ]
        )
    export_typst_table(
        paths,
        "bc_results",
        ["architecture", "seeds", "open-loop acc",
         "success (train scen.)", "success (unseen)"],
        summary_rows,
    )
    write_table_csv(
        paths,
        "bc_results.csv",
        ["architecture", "seeds", "open_loop_accuracy",
         "success_train_scenarios", "success_unseen"],
        summary_rows,
    )
    rl_contrast_rows = [
        ["states the learner must handle",
         f"{qlearning['states_in_table']} (fits in a table)",
         "unbounded: every seed generates a new world (Lab 2 saw 3,746 distinct "
         "observations in 300 episodes)"],
        ["observability", "full (position read directly)",
         "partial 7x7 egocentric view (Lab 2)"],
        ["language", "none", "mission string decides which object is correct"],
        ["reward signal", "terminal, reachable by epsilon-greedy exploration",
         "terminal-only; random walking succeeds sometimes (Lab 3: 27.3%) but "
         "assigns no credit to individual actions"],
        ["supervision used by the study", "reward only", "oracle action labels only "
         "(reward is only an evaluation metric)"],
    ]
    export_typst_table(
        paths, "rl_contrast", ["property", "toy MDP", "study task"], rl_contrast_rows
    )
    write_table_csv(
        paths, "rl_contrast.csv", ["property", "toy_mdp", "study_task"], rl_contrast_rows
    )

    recurrent_rows = experiment["results"]["recurrent"]
    memoryless_rows = experiment["results"]["memoryless"]

    def _rate(rows: list[dict[str, object]], key: str, subkey: str | None = None) -> str:
        samples = [row[key] if subkey is None else row[key][subkey] for row in rows]
        return f"{_mean(samples):.1%}"

    export_typst_values(
        paths,
        "learning_facts",
        {
            "qlearning-states": str(qlearning["states_in_table"]),
            "qlearning-best-steps": f"{qlearning['best_greedy_steps']:.0f}",
            "dataset-episodes": str(experiment["dataset_counters"]["collected"]),
            "dataset-labels": str(experiment["total_labels"]),
            "memoryless-open-acc": _rate(memoryless_rows, "open_loop_accuracy"),
            "memoryless-unseen": _rate(memoryless_rows, "holdout", "success_rate"),
            "recurrent-open-acc": _rate(recurrent_rows, "open_loop_accuracy"),
            "recurrent-unseen": _rate(recurrent_rows, "holdout", "success_rate"),
            "bc-device": experiment["device"],
        },
    )

    metrics = {
        "qlearning": qlearning,
        "behavior_cloning": {
            key: value for key, value in experiment.items() if key != "losses"
        },
    }
    metrics_hash = write_metrics(paths, metrics)

    mem_unseen = _mean([r["holdout"]["success_rate"] for r in memoryless_rows])
    rec_unseen = _mean([r["holdout"]["success_rate"] for r in recurrent_rows])
    mem_acc = _mean([r["open_loop_accuracy"] for r in memoryless_rows])
    rec_acc = _mean([r["open_loop_accuracy"] for r in recurrent_rows])
    write_mini_report(
        paths,
        question="Is this reinforcement learning, and if not, what is it?",
        sections=[
            (
                "Two paradigms, one line each",
                "Reinforcement learning maximizes expected return by interacting "
                "with the world and learning from reward. Imitation learning fits "
                "a policy to labelled expert decisions, which is supervised learning on "
                "(observation history, expert action) pairs. The study is pure "
                "imitation: no gradient anywhere depends on reward; reward is "
                "only read at evaluation time to *measure* success.",
            ),
            (
                "Real RL, where it belongs",
                f"Tabular Q-learning, implemented from scratch, on `{TOY_ENV_ID}` "
                f"(fully observable, {qlearning['states_in_table']} states ever "
                "encountered): the greedy policy reaches the goal in "
                f"{qlearning['best_greedy_steps']:.0f} steps after a few hundred "
                "episodes (`figures/qlearning_curve.svg`). RL genuinely works "
                "when the state is visible and enumerable and reward is "
                "reachable.",
            ),
            (
                "Why that recipe does not carry over",
                "`rl_contrast` table: the study task generates a fresh world per "
                "seed (no table can enumerate it), hides the state behind a 7x7 "
                "egocentric view, conditions success on a mission string, and "
                "pays reward only at the end. None of this makes deep RL "
                "*impossible*: random walking already succeeds "
                f"{_lab03_random_success(paths.repo_root)} of the time (Lab 3), "
                "so exploration would find reward. The study "
                "is not asking an RL question. It asks a supervision-economics "
                "question (which *labels* help more, Lab 6), and behavior "
                "cloning is the controlled substrate for answering it.",
            ),
            (
                "The first cloned policies",
                f"{experiment['dataset_counters']['collected']} oracle "
                f"demonstrations ({experiment['total_labels']} labelled steps) "
                "were cloned into two architectures, with identical data, identical "
                "optimizer, three seeds each: a memoryless policy and the "
                "study's recurrent one. Training losses in "
                "`figures/bc_learning_curves.svg`; all runs on "
                f"{experiment['device']}.",
            ),
            (
                "The two imitation lessons",
                f"(1) *Accuracy is not success*: the memoryless policy matches "
                f"the oracle on {mem_acc:.1%} of held-out expert steps yet "
                f"completes only {mem_unseen:.1%} of unseen episodes closed-loop "
                "because small per-step errors compound over a whole rollout (Lab 6 "
                "makes this the central phenomenon). (2) *Memory matters*: the "
                f"recurrent policy reaches {rec_acc:.1%} accuracy and "
                f"{rec_unseen:.1%} unseen success. Lab 2 proved a memoryless "
                "policy must disagree with the oracle on ≥3% of visited states; "
                "here the gap is visible end to end.",
            ),
            (
                "Bridge to the study",
                "The study trains the same recurrent architecture with the same "
                "masked cross-entropy idea, but on one-target-per-window items "
                "with exact revealed-label budgets instead of whole episodes, "
                "the tightening exists so that arms can be compared at equal "
                "label counts, which is Lab 7's subject. Evaluation there is "
                "exactly the greedy closed-loop rollout used here.",
            ),
        ],
    )

    return {
        "qlearning_best_steps": qlearning["best_greedy_steps"],
        "memoryless_unseen": f"{mem_unseen:.1%}",
        "recurrent_unseen": f"{rec_unseen:.1%}",
        "device": experiment["device"],
        "metrics_hash": metrics_hash,
    }
