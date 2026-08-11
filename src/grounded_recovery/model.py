"""Language-conditioned recurrent policy.

The model receives only the batch fields: symbolic observation image,
direction, previous executed action, mission tokens with lengths, and the step
mask. Coordinates, goals, oracle state, perturbation identity, and any other
metadata never enter ``forward``: the signature itself is the boundary, and a
test asserts it.

Sequences are right-padded; the unidirectional GRUs run over the padded
length, and both the loss and the final hidden state are taken at valid
positions only, which makes padding exactly inert (no packed sequences).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from grounded_recovery.artifacts import hash_json
from grounded_recovery.config import ModelConfig

DIRECTION_COUNT = 4


def model_config_hash(model_cfg: ModelConfig, vocab_size: int, num_actions: int) -> str:
    import dataclasses

    return hash_json(
        {
            "model": dataclasses.asdict(model_cfg),
            "vocab_size": int(vocab_size),
            "num_actions": int(num_actions),
        }
    )


class RecoveryPolicy(nn.Module):
    """Symbolic-observation encoder + mission GRU + recurrent policy core."""

    def __init__(self, model_cfg: ModelConfig, vocab_size: int, num_actions: int) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.vocab_size = int(vocab_size)
        self.num_actions = int(num_actions)

        self.object_embedding = nn.Embedding(model_cfg.num_objects, model_cfg.tile_embedding)
        self.color_embedding = nn.Embedding(model_cfg.num_colors, model_cfg.tile_embedding)
        self.state_embedding = nn.Embedding(model_cfg.num_states, model_cfg.tile_embedding)
        tile_channels = 3 * model_cfg.tile_embedding
        self.observation_conv = nn.Sequential(
            nn.Conv2d(tile_channels, model_cfg.conv_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(model_cfg.conv_channels, model_cfg.conv_channels, kernel_size=3),
            nn.ReLU(),
        )
        conv_output = model_cfg.conv_channels * 3 * 3  # 7x7 through two valid 3x3 convs
        self.observation_projection = nn.Linear(conv_output, model_cfg.observation_projection)

        self.word_embedding = nn.Embedding(self.vocab_size, model_cfg.word_embedding)
        self.language_gru = nn.GRU(
            model_cfg.word_embedding, model_cfg.language_gru, batch_first=True
        )

        self.direction_embedding = nn.Embedding(DIRECTION_COUNT, model_cfg.direction_embedding)
        # One extra row for the START token at absolute episode start.
        self.action_embedding = nn.Embedding(
            self.num_actions + 1, model_cfg.action_embedding
        )

        fusion_input = (
            model_cfg.observation_projection
            + model_cfg.language_gru
            + model_cfg.direction_embedding
            + model_cfg.action_embedding
        )
        self.fusion = nn.Linear(fusion_input, model_cfg.fusion)
        self.policy_gru = nn.GRU(model_cfg.fusion, model_cfg.policy_gru, batch_first=True)
        self.head = nn.Linear(model_cfg.policy_gru, self.num_actions)

    def encode_observation(self, image: Tensor) -> Tensor:
        """[B, T, 7, 7, 3] long -> [B, T, observation_projection]."""
        batch, steps = image.shape[0], image.shape[1]
        tiles = torch.cat(
            (
                self.object_embedding(image[..., 0]),
                self.color_embedding(image[..., 1]),
                self.state_embedding(image[..., 2]),
            ),
            dim=-1,
        )  # [B, T, 7, 7, 3*tile]
        flat = tiles.reshape(batch * steps, 7, 7, -1).permute(0, 3, 1, 2)
        features = self.observation_conv(flat).reshape(batch * steps, -1)
        projected = torch.relu(self.observation_projection(features))
        return projected.reshape(batch, steps, -1)

    def encode_mission(self, mission_tokens: Tensor, mission_lengths: Tensor) -> Tensor:
        """[B, L] tokens with true lengths -> [B, language_gru] at the last valid step."""
        embedded = self.word_embedding(mission_tokens)
        outputs, _ = self.language_gru(embedded)
        index = (mission_lengths - 1).clamp(min=0)
        gather = index.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        return outputs.gather(1, gather).squeeze(1)

    def _fuse(
        self,
        observation_features: Tensor,
        mission_feature: Tensor,
        direction: Tensor,
        prev_executed_action: Tensor,
    ) -> Tensor:
        steps = observation_features.shape[1]
        mission_expanded = mission_feature.unsqueeze(1).expand(-1, steps, -1)
        fused = torch.cat(
            (
                observation_features,
                mission_expanded,
                self.direction_embedding(direction),
                self.action_embedding(prev_executed_action),
            ),
            dim=-1,
        )
        return torch.relu(self.fusion(fused))

    def forward(
        self,
        image: Tensor,
        direction: Tensor,
        prev_executed_action: Tensor,
        mission_tokens: Tensor,
        mission_lengths: Tensor,
        step_mask: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns logits [B, T, num_actions] and the hidden state at each
        row's last valid step [B, policy_gru]."""
        observation_features = self.encode_observation(image)
        mission_feature = self.encode_mission(mission_tokens, mission_lengths)
        fused = self._fuse(
            observation_features, mission_feature, direction, prev_executed_action
        )
        hidden0 = None if initial_hidden is None else initial_hidden.unsqueeze(0)
        outputs, _ = self.policy_gru(fused, hidden0)
        logits = self.head(outputs)
        lengths = step_mask.long().sum(dim=1)
        index = (lengths - 1).clamp(min=0)
        gather = index.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        final_hidden = outputs.gather(1, gather).squeeze(1)
        return logits, final_hidden

    def step(
        self,
        image: Tensor,
        direction: Tensor,
        prev_executed_action: Tensor,
        mission_feature: Tensor,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Single-timestep inference: [B, 7, 7, 3] -> logits [B, A], next hidden.

        ``mission_feature`` comes from ``encode_mission`` once per episode.
        """
        observation_features = self.encode_observation(image.unsqueeze(1))
        fused = self._fuse(
            observation_features,
            mission_feature,
            direction.unsqueeze(1),
            prev_executed_action.unsqueeze(1),
        )
        hidden0 = None if hidden is None else hidden.unsqueeze(0)
        outputs, _ = self.policy_gru(fused, hidden0)
        logits = self.head(outputs[:, 0])
        return logits, outputs[:, 0]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
