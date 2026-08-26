"""Real weights and activations from a trained policy, for the site.

The architecture chapter animates the network with its actual numbers, so
this exporter loads a digest-verified lab checkpoint, lets the policy act
greedily on the same scenario the labelled demonstration uses, and records
what flows through every stage: the raw observation planes, sampled
convolution activations, the mission encoding, the fused vector, the
recurrent hidden state, and the three action logits. Weights are exported
as small slices under disclosed sampling rules. Nothing is normalized on
disk. Every block carries its own min/max, and the site maps values to
colors at display time.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from gr_foundations.common import FoundationsError, derive_seed
from gr_foundations.media_journey import _load_lab_model, _PolicyDriver, capture_episode
from gr_foundations.training import START_ACTION_TOKEN
from grounded_recovery.artifacts import atomic_write_json, file_sha256

SCHEMA_VERSION = "1.0.0"
CHECKPOINT = "lab04/checkpoints/recurrent_s0.pt"
SEED_COMPONENT = "lab03.trajectory"
MAX_STEPS = 24  # exported prefix; the scrubber does not need a full wander

# Disclosed sampling rules for the large weight blocks.
CONV_OUT_CHANNELS = (0, 1, 2, 3)
CONV_IN_CHANNELS = (0, 7)
BLOCK_ROWS = 16
HEAD_COLUMNS = 32


def _tensor(values: torch.Tensor) -> dict[str, object]:
    data = values.detach().to("cpu", torch.float64)
    return {
        "shape": list(data.shape),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "values": [round(float(v), 4) for v in data.reshape(-1)],
    }


def _weights(model: torch.nn.Module) -> dict[str, object]:
    state = {name: value for name, value in model.state_dict().items()}
    conv1 = state["observation_conv.0.weight"][list(CONV_OUT_CHANNELS)][
        :, list(CONV_IN_CHANNELS)
    ]
    conv2 = state["observation_conv.2.weight"][list(CONV_OUT_CHANNELS)][
        :, list(CONV_IN_CHANNELS)
    ]
    return {
        "word_embedding": _tensor(state["word_embedding.weight"]),
        "obs_embed_object": _tensor(state["object_embedding.weight"]),
        "obs_embed_color": _tensor(state["color_embedding.weight"]),
        "obs_embed_state": _tensor(state["state_embedding.weight"]),
        "dir_embedding": _tensor(state["direction_embedding.weight"]),
        "prev_action_embedding": _tensor(state["action_embedding.weight"]),
        "conv1_kernels": {
            "out_channels": list(CONV_OUT_CHANNELS),
            "in_channels": list(CONV_IN_CHANNELS),
            "tensor": _tensor(conv1),
        },
        "conv2_kernels": {
            "out_channels": list(CONV_OUT_CHANNELS),
            "in_channels": list(CONV_IN_CHANNELS),
            "tensor": _tensor(conv2),
        },
        "projection_block": _tensor(
            state["observation_projection.weight"][:BLOCK_ROWS, :BLOCK_ROWS]
        ),
        "fusion_block": _tensor(state["fusion.weight"][:BLOCK_ROWS, :BLOCK_ROWS]),
        "policy_gru_block": _tensor(
            state["policy_gru.weight_ih_l0"][:BLOCK_ROWS, :BLOCK_ROWS]
        ),
        "head": _tensor(state["head.weight"][:, :HEAD_COLUMNS]),
    }


def build_trace(repo_root: Path) -> dict[str, object]:
    from gr_foundations.training import contract_config

    contract = contract_config(repo_root)
    env_cfg = contract.environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = _load_lab_model(repo_root, contract, CHECKPOINT)
    model = model.to(device).eval()

    # The policy acts greedily on the demonstration scenario; the oracle rides
    # along, synchronized to the executed actions, so every step also carries
    # the label the teacher would have given.
    seed = derive_seed(SEED_COMPONENT, 0)
    driver = _PolicyDriver(model, vocab, device)
    capture = capture_episode(
        env_cfg, seed, driver, with_oracle=True, record_state=True
    )
    exported = min(capture.steps, MAX_STEPS)

    # Replay the recorded inputs through the model's own modules, harvesting
    # the intermediates with forward hooks so the exported numbers are exactly
    # what the network computed (post-ReLU where the flow is rectified).
    grabbed: dict[str, torch.Tensor] = {}

    def keep(name: str):
        def hook(_module, _inputs, output):
            grabbed[name] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [
        model.observation_conv[0].register_forward_hook(keep("conv1")),
        model.observation_conv[2].register_forward_hook(keep("conv2")),
        model.observation_projection.register_forward_hook(keep("projection")),
        model.fusion.register_forward_hook(keep("fused")),
        model.language_gru.register_forward_hook(keep("mission_tokens_hidden")),
    ]
    steps: list[dict[str, object]] = []
    try:
        with torch.no_grad():
            tokens = vocab.encode(capture.mission)
            mission_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
            mission_feature = model.encode_mission(mission_tokens, lengths)
            per_token_hidden = grabbed["mission_tokens_hidden"][0]

            hidden = None
            prev = torch.tensor([START_ACTION_TOKEN], dtype=torch.long, device=device)
            for t in range(exported):
                image = torch.from_numpy(
                    capture.observations[t].astype("int64")
                ).to(device)
                direction = torch.tensor(
                    [capture.poses[t][2]], dtype=torch.long, device=device
                )
                logits, hidden = model.step(
                    image.unsqueeze(0), direction, prev, mission_feature, hidden
                )
                chosen = int(torch.argmax(logits, dim=-1).item())
                if chosen != capture.actions[t]:
                    raise FoundationsError(
                        f"replay diverged at t={t}: {chosen} != {capture.actions[t]}"
                    )
                conv1 = F.relu(grabbed["conv1"][0, list(CONV_OUT_CHANNELS)])
                conv2 = F.relu(grabbed["conv2"][0, list(CONV_OUT_CHANNELS)])
                steps.append(
                    {
                        "t": t,
                        "obs": capture.observations[t].tolist(),
                        "direction": int(capture.poses[t][2]),
                        "prev_action": None if t == 0 else int(capture.actions[t - 1]),
                        "acts": {
                            "conv1_sample": _tensor(conv1),
                            "conv2_sample": _tensor(conv2),
                            "obs_vec": _tensor(F.relu(grabbed["projection"][0])),
                            "dir_embed": _tensor(
                                model.direction_embedding(direction)[0]
                            ),
                            "prev_action_embed": _tensor(
                                model.action_embedding(prev)[0]
                            ),
                            "fused": _tensor(F.relu(grabbed["fused"][0, 0])),
                            "hidden": _tensor(hidden[0]),
                            "logits": _tensor(logits[0]),
                            "probs": _tensor(torch.softmax(logits[0], dim=-1)),
                        },
                        "action": int(capture.actions[t]),
                        "oracle_label": capture.labels[t],
                    }
                )
                prev = torch.tensor([chosen], dtype=torch.long, device=device)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "network-flow",
        "variant": "full",
        "source": {
            "checkpoint": CHECKPOINT,
            "checkpoint_digest": _checkpoint_digest(repo_root),
            "seed_component": SEED_COMPONENT,
            "seed_index": 0,
            "policy": "greedy rollout of the lab04 recurrent policy, replicate 0",
            "selection_rule": (
                "the same scenario as the labelled demonstration "
                "(lab03.trajectory seed 0); first "
                f"{exported} of {capture.steps} steps"
            ),
            "weight_sampling": (
                f"conv kernels: output channels {list(CONV_OUT_CHANNELS)} × input "
                f"channels {list(CONV_IN_CHANNELS)}; projection/fusion/GRU blocks: "
                f"first {BLOCK_ROWS} rows × {BLOCK_ROWS} columns; head: first "
                f"{HEAD_COLUMNS} columns"
            ),
            "normalization": (
                "values are raw; each block carries its own min/max and is "
                "mapped to the color ramp at display time"
            ),
        },
        "mission": {
            "text": capture.mission,
            "tokens": [int(v) for v in vocab.encode(capture.mission)],
            "vocab": list(vocab.tokens),
        },
        "action_names": list(env_cfg.action_names),
        "outcome": "success" if capture.success else "failure",
        "steps_taken": capture.steps,
        "exported_steps": exported,
        "weights": _weights(model),
        "mission_acts": {
            "per_token_hidden": _tensor(per_token_hidden),
            "mission_vec": _tensor(mission_feature[0]),
        },
        "ranges": _activation_ranges(steps),
        "steps": steps,
    }


def _checkpoint_digest(repo_root: Path) -> str:
    import json

    meta = json.loads(
        (repo_root / "data" / "foundations" / CHECKPOINT)
        .with_suffix(".json")
        .read_text()
    )
    return str(meta["digest"])


def _activation_ranges(steps: list[dict[str, object]]) -> dict[str, object]:
    """Per-key min/max across the whole episode, so colors stay comparable."""
    ranges: dict[str, dict[str, float]] = {}
    for step in steps:
        for key, block in step["acts"].items():
            entry = ranges.setdefault(key, {"min": block["min"], "max": block["max"]})
            entry["min"] = min(entry["min"], block["min"])
            entry["max"] = max(entry["max"], block["max"])
    return ranges


def run(repo_root: Path, *, force: bool) -> dict[str, object]:
    out_dir = repo_root / "foundations" / "media" / "network"
    out_path = out_dir / "full_r0.json"
    if out_path.exists() and not force:
        raise FoundationsError(
            f"{out_path} already exists; pass --force to regenerate"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    document = build_trace(repo_root)
    atomic_write_json(out_path, document, overwrite=True)
    return {
        "out_path": str(out_path),
        "sha256": file_sha256(out_path),
        "steps": document["exported_steps"],
        "outcome": document["outcome"],
    }
