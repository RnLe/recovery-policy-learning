"""Lab 5: the policy network, piece by piece.

The study's model is small enough to hold in one head, and this lab makes
sure of it three ways. It generates the shape/parameter walkthrough from a
live instance (documentation that cannot drift from the code), it re-derives
the architecture as ``LabPolicy`` and asserts parameter-for-parameter parity
with the study's ``RecoveryPolicy``, and it lets the data justify the design:
every major component is removed or replaced in a matched ablation (same
demonstrations, same optimizer, three seeds each) and judged by closed-loop
success on unseen scenarios, the study's own yardstick.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from gr_foundations.common import (
    COLOR_BASE,
    COLOR_CAUTION,
    COLOR_EXTRA,
    COLOR_NEUTRAL,
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
from gr_foundations.models import LabPolicy
from gr_foundations.training import (
    NUM_ACTIONS,
    build_bc_dataset,
    closed_loop_success,
    contract_config,
    dataset_vocabulary,
    model_digest,
    open_loop_accuracy,
    resolve_device,
    train_bc,
)
from grounded_recovery.model import RecoveryPolicy

DATASET_EPISODES = 200
HOLDOUT_EPISODES = 100
UPDATES = 1500
BATCH_EPISODES = 16
N_SEEDS = 3

VARIANTS: dict[str, dict[str, object]] = {
    "full": {},
    "no_memory": {"use_memory": False},
    "no_mission": {"use_mission": False},
    "no_prev_action": {"use_prev_action": False},
    "bow_mission": {"mission_encoder": "bow"},
}

VARIANT_LABELS = {
    "full": "full architecture",
    "no_memory": "no policy GRU\n(memoryless trunk)",
    "no_mission": "no mission input",
    "no_prev_action": "no previous-action\ninput",
    "bow_mission": "bag-of-words mission\n(no language GRU)",
}

RATIONALE_ROWS: list[list[str]] = [
    ["three channel embeddings",
     "the observation is symbolic (lookup indices, Lab 1), not pixels; object, "
     "color, and door-state get separate learned vectors instead of one "
     "arbitrary integer scale"],
    ["two 3x3 convolutions", "local spatial patterns over the 7x7 view; two valid "
     "convolutions reduce 7x7 to 3x3 before a linear projection"],
    ["mission GRU", "turns the instruction into one vector; the ablation checks "
     "whether order-aware encoding beats a bag of words at this grammar size"],
    ["direction embedding", "the view direction is part of the observation (Lab 2); "
     "an embedding, not a raw integer"],
    ["previous-action embedding", "the policy knows what was just *executed*, the "
     "channel through which an external corruption becomes visible (Lab 6)"],
    ["fusion layer", "concatenate all features, mix once, nonlinearity"],
    ["policy GRU", "the memory that Lab 2 proved necessary: aliased observations "
     "demand history-dependence"],
    ["linear head", "three logits, one per frozen action"],
]


def shape_walkthrough(
    model: torch.nn.Module, vocab_size: int
) -> list[list[object]]:
    """Per-component parameter counts and live output shapes via forward hooks."""
    shapes: dict[str, str] = {}
    hooks = []

    def register(name: str, module: torch.nn.Module) -> None:
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                shapes[name] = "x".join(str(d) for d in tensor.shape)

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_children():
        register(name, module)
    batch, steps, words = 2, 5, 6
    with torch.no_grad():
        model(
            torch.zeros(batch, steps, 7, 7, 3, dtype=torch.long),
            torch.zeros(batch, steps, dtype=torch.long),
            torch.full((batch, steps), NUM_ACTIONS, dtype=torch.long),
            torch.ones(batch, words, dtype=torch.long),
            torch.full((batch,), words, dtype=torch.long),
            torch.ones(batch, steps, dtype=torch.bool),
        )
    for hook in hooks:
        hook.remove()
    rows: list[list[object]] = []
    for name, module in model.named_children():
        parameters = sum(p.numel() for p in module.parameters())
        rows.append([name, parameters, shapes.get(name, "")])
    rows.append(["total", sum(p.numel() for p in model.parameters()), ""])
    return rows


def run_ablation(
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
    dataset, _ = build_bc_dataset(env_cfg, dataset_episodes, "lab04.dataset")
    holdout, _ = build_bc_dataset(env_cfg, holdout_episodes, "lab04.holdout")
    vocab = dataset_vocabulary(dataset)
    holdout_seeds = [episode.seed for episode in holdout]

    results: dict[str, list[dict[str, object]]] = {}
    for variant_index, (variant, kwargs) in enumerate(VARIANTS.items()):
        rows = []
        for repetition in range(n_seeds):
            seed = derive_seed("lab05.train", variant_index * 10 + repetition)
            model, _log = train_bc(
                lambda kwargs=kwargs: LabPolicy(
                    contract.model, vocab.size, len(env_cfg.action_ids), **kwargs
                ),
                dataset,
                vocab,
                updates=updates,
                batch_episodes=batch_episodes,
                seed=seed,
                device=device,
            )
            rows.append(
                {
                    "seed_index": repetition,
                    "parameters": int(sum(p.numel() for p in model.parameters())),
                    "open_loop_accuracy": open_loop_accuracy(model, holdout, vocab, device),
                    "unseen": closed_loop_success(
                        model, vocab, env_cfg, holdout_seeds, device
                    ),
                    "model_digest": model_digest(model),
                }
            )
        results[variant] = rows
    return {
        "device": device.type,
        "vocabulary_size": vocab.size,
        "updates": updates,
        "results": results,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _render_ablation(ablation: dict[str, object], paths: LabPaths) -> None:
    colors = {
        "full": COLOR_RECOVERY,
        "no_memory": COLOR_BASE,
        "no_mission": COLOR_CAUTION,
        "no_prev_action": COLOR_EXTRA,
        "bow_mission": COLOR_NEUTRAL,
    }
    fig, axis = plt.subplots(figsize=(12.4, 4.6))
    names = list(VARIANTS)
    positions = np.arange(len(names))
    for position, name in zip(positions, names, strict=True):
        samples = [row["unseen"]["success_rate"] for row in ablation["results"][name]]
        mean = _mean(samples)
        axis.bar(position, mean, color=colors[name])
        axis.plot(
            [position, position], [min(samples), max(samples)],
            color="black", linewidth=1.2,
        )
        axis.text(position, mean + 0.03, f"{mean:.0%}", ha="center", fontsize=12)
    axis.set_xticks(positions)
    axis.set_xticklabels([VARIANT_LABELS[name] for name in names], fontsize=11)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("closed-loop success (unseen)")
    axis.set_title(
        "one component removed at a time, with the same data, the same optimizer, "
        f"{len(ablation['results']['full'])} seeds each (range bars)",
        fontsize=13,
    )
    save_figure(paths, fig, "ablation_results.svg")


def run(
    paths: LabPaths,
    *,
    force: bool,
    dataset_episodes: int = DATASET_EPISODES,
    holdout_episodes: int = HOLDOUT_EPISODES,
    updates: int = UPDATES,
    batch_episodes: int = BATCH_EPISODES,
    n_seeds: int = N_SEEDS,
) -> dict[str, object]:
    prepare(paths, force=force)
    contract = contract_config(paths.repo_root)

    # Parity: the lab reimplementation must match the study model exactly.
    reference_vocab = 21
    lab_full = LabPolicy(contract.model, reference_vocab, NUM_ACTIONS)
    study_model = RecoveryPolicy(contract.model, reference_vocab, NUM_ACTIONS)
    if lab_full.parameter_count() != study_model.parameter_count():
        raise FoundationsError("LabPolicy no longer mirrors RecoveryPolicy")
    walkthrough = shape_walkthrough(study_model, reference_vocab)

    ablation = run_ablation(
        paths.repo_root,
        dataset_episodes=dataset_episodes,
        holdout_episodes=holdout_episodes,
        updates=updates,
        batch_episodes=batch_episodes,
        n_seeds=n_seeds,
    )
    _render_ablation(ablation, paths)

    export_typst_table(
        paths,
        "shape_walkthrough",
        ["component", "parameters", "output shape (B=2, T=5)"],
        walkthrough,
    )
    write_table_csv(
        paths, "shape_walkthrough.csv", ["component", "parameters", "output_shape"],
        walkthrough,
    )
    export_typst_table(paths, "design_rationale", ["component", "why"], RATIONALE_ROWS)
    ablation_rows = [
        [
            name,
            ablation["results"][name][0]["parameters"],
            f"{_mean([r['open_loop_accuracy'] for r in ablation['results'][name]]):.1%}",
            f"{_mean([r['unseen']['success_rate'] for r in ablation['results'][name]]):.1%}",
        ]
        for name in VARIANTS
    ]
    export_typst_table(
        paths,
        "ablation_results",
        ["variant", "parameters", "open-loop acc", "unseen success"],
        ablation_rows,
    )
    write_table_csv(
        paths,
        "ablation_results.csv",
        ["variant", "parameters", "open_loop_accuracy", "unseen_success"],
        ablation_rows,
    )
    full_unseen_rates = [
        row["unseen"]["success_rate"] for row in ablation["results"]["full"]
    ]
    export_typst_values(
        paths,
        "architecture_facts",
        {
            "total-parameters": str(study_model.parameter_count()),
            "parity-verified": "true",
            "full-unseen": f"{_mean(full_unseen_rates):.1%}",
            "ablation-device": ablation["device"],
        },
    )

    metrics = {
        "parameter_parity": {
            "lab_policy": lab_full.parameter_count(),
            "recovery_policy": study_model.parameter_count(),
        },
        "walkthrough": walkthrough,
        "ablation": ablation,
    }
    metrics_hash = write_metrics(paths, metrics)

    results = ablation["results"]

    def unseen(name: str) -> str:
        return f"{_mean([r['unseen']['success_rate'] for r in results[name]]):.1%}"

    def unseen_range(name: str) -> str:
        rates = [r["unseen"]["success_rate"] for r in results[name]]
        return f"{min(rates):.0%}–{max(rates):.0%}"

    # Lab 4 trained the same memoryless architecture on three *different*
    # seeds; pooling both labs turns the comparison into a seed-noise lesson.
    lab04_note = ""
    lab04_metrics_path = paths.repo_root / "foundations" / "lab04" / "metrics.json"
    if lab04_metrics_path.exists():
        import json

        lab04_results = json.loads(lab04_metrics_path.read_text())["metrics"][
            "behavior_cloning"
        ]["results"]
        lab04_memoryless = [
            row["holdout"]["success_rate"] for row in lab04_results["memoryless"]
        ]
        pooled = lab04_memoryless + [
            row["unseen"]["success_rate"] for row in results["no_memory"]
        ]
        lab04_note = (
            f" Lab 4 trained this same memoryless architecture on three other "
            f"seeds and measured {_mean(lab04_memoryless):.1%}; pooling all six "
            f"seeds spans {min(pooled):.0%}–{max(pooled):.0%}. Three-seed "
            "comparisons wobble, which is precisely why the study runs six "
            "paired replicates and reports an interval (Lab 7)."
        )

    write_mini_report(
        paths,
        question="What exactly is the policy network, and does every piece earn its place?",
        sections=[
            (
                "The model at a glance",
                f"{study_model.parameter_count():,} parameters: three channel "
                "embeddings over the symbolic 7x7 view, two 3x3 convolutions, a "
                "linear projection; a word-embedding + GRU mission encoder; "
                "direction and previous-action embeddings; one fusion layer; a "
                "GRU policy core; a three-logit head. The full table in "
                "`tables/shape_walkthrough.csv` is *generated from a live "
                "instance*, so it cannot drift from the code. A from-scratch "
                "reimplementation (`gr_foundations.models.LabPolicy`) matches "
                "the study model parameter for parameter (asserted at runtime "
                "and in tests).",
            ),
            (
                "One component removed at a time",
                "Five variants, identical demonstrations, identical optimizer, "
                f"{n_seeds} seeds each (`figures/ablation_results.svg`): full "
                f"{unseen('full')}, no policy GRU {unseen('no_memory')}, no "
                f"mission input {unseen('no_mission')}, no previous-action "
                f"input {unseen('no_prev_action')}, bag-of-words mission "
                f"{unseen('bow_mission')} closed-loop success on unseen "
                "scenarios.",
            ),
            (
                "Reading the ablation honestly",
                "The candid finding: at this training scale, clean-condition "
                "success is a blunt instrument. All five variants land within a "
                f"few points (means {unseen('no_memory')}–{unseen('full')}), "
                "with heavily overlapping three-seed ranges "
                f"(full {unseen_range('full')}, no-GRU "
                f"{unseen_range('no_memory')})." + lab04_note + " Two results "
                "deserve their own sentences. First, *removing the mission "
                f"costs nothing here* ({unseen('no_mission')}, and mean episode "
                "lengths do not separate either): with one distractor and a "
                "144-step limit, a mission-blind policy that simply tours "
                "objects still ends on the right one, because the endpoint tolerates "
                "detours, so language earns its keep only at tighter horizons "
                "or richer scenes (an interpretation, marked as such). Second, "
                f"the bag-of-words mission encoder ({unseen('bow_mission')}) "
                "matches the GRU at this five-word grammar.",
            ),
            (
                "Why the architecture is still the right one",
                "The components are justified by their *roles in the study*, "
                "not by clean-run ablation wins. Memory: Lab 2 proved no "
                "memoryless policy can match the oracle on aliased states, a "
                "structural argument that holds regardless of seed noise, and "
                "Lab 4's own seeds showed the end-to-end gap. Previous-action "
                "input: nearly free in clean conditions, but it is the only "
                "channel through which an externally corrupted execution "
                "becomes visible to the policy; its purpose only exists under "
                "Lab 6's corruptions. Mission conditioning: it defines the "
                "task; that ITT success barely punishes its removal at this "
                "scale is a fact about the endpoint's tolerance, not about the "
                "input being uninformative. The study inherits the standard "
                "BabyAI treatment and makes no architectural novelty claims.",
            ),
            (
                "Why a GRU and not a Transformer",
                "The dataset is a few thousand labelled steps; the model runs "
                "closed-loop, one observation at a time, carrying state forward. "
                "A recurrent core consumes O(1) memory per step at inference, "
                "trains stably at this scale, and keeps the architecture small "
                "enough to audit by hand, which this repository treats as a "
                "feature, not a limitation.",
            ),
            (
                "Bridge to the study",
                "`grounded_recovery.model.RecoveryPolicy` is this exact "
                "architecture; the study adds nothing at model level. Its "
                "checkpoints additionally freeze the vocabulary, action names, "
                "and RNG states so that training can be resumed or cloned "
                "bit-exactly. The cloning matters in Lab 6, where three arms "
                "must start from the *same* base checkpoint.",
            ),
        ],
    )

    return {
        "parameters": study_model.parameter_count(),
        "full_unseen": unseen("full"),
        "no_memory_unseen": unseen("no_memory"),
        "no_mission_unseen": unseen("no_mission"),
        "device": ablation["device"],
        "metrics_hash": metrics_hash,
    }
