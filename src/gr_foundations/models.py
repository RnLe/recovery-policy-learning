"""Lab-local policy family for the foundations experiments.

``LabPolicy`` is a from-scratch reimplementation of the study architecture
with switchable components, so that Labs 4 and 5 can ablate one design choice
at a time: temporal memory (policy GRU vs feedforward trunk), mission
conditioning (present vs absent), previous-action input (present vs absent),
and the mission encoder (GRU vs bag of words). With every switch in its
default position the module mirrors ``grounded_recovery.model.RecoveryPolicy``
layer for layer, and Lab 5 asserts the parameter-count parity, which also makes
this file the reference solution for a manual reimplementation of the study
model. The ``encode_mission``/``step`` inference API is kept identical so the
same closed-loop evaluator drives both classes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from grounded_recovery.config import ModelConfig
from grounded_recovery.model import DIRECTION_COUNT


class LabPolicy(nn.Module):
    """Configurable symbolic policy: encoder + optional mission/action/memory."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        vocab_size: int,
        num_actions: int,
        *,
        use_memory: bool = True,
        use_mission: bool = True,
        use_prev_action: bool = True,
        mission_encoder: str = "gru",
    ) -> None:
        super().__init__()
        if mission_encoder not in ("gru", "bow"):
            raise ValueError(f"unknown mission encoder {mission_encoder!r}")
        self.model_cfg = model_cfg
        self.vocab_size = int(vocab_size)
        self.num_actions = int(num_actions)
        self.use_memory = use_memory
        self.use_mission = use_mission
        self.use_prev_action = use_prev_action
        self.mission_encoder = mission_encoder

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
        conv_output = model_cfg.conv_channels * 3 * 3
        self.observation_projection = nn.Linear(conv_output, model_cfg.observation_projection)

        fusion_input = model_cfg.observation_projection + model_cfg.direction_embedding
        if use_mission:
            self.word_embedding = nn.Embedding(self.vocab_size, model_cfg.word_embedding)
            if mission_encoder == "gru":
                self.language_gru = nn.GRU(
                    model_cfg.word_embedding, model_cfg.language_gru, batch_first=True
                )
            else:
                self.language_projection = nn.Linear(
                    model_cfg.word_embedding, model_cfg.language_gru
                )
            fusion_input += model_cfg.language_gru
        if use_prev_action:
            self.action_embedding = nn.Embedding(
                self.num_actions + 1, model_cfg.action_embedding
            )
            fusion_input += model_cfg.action_embedding
        self.direction_embedding = nn.Embedding(DIRECTION_COUNT, model_cfg.direction_embedding)

        self.fusion = nn.Linear(fusion_input, model_cfg.fusion)
        if use_memory:
            self.policy_gru = nn.GRU(model_cfg.fusion, model_cfg.policy_gru, batch_first=True)
        else:
            self.trunk = nn.Sequential(
                nn.Linear(model_cfg.fusion, model_cfg.policy_gru), nn.ReLU()
            )
        self.head = nn.Linear(model_cfg.policy_gru, self.num_actions)

    def encode_observation(self, image: Tensor) -> Tensor:
        batch, steps = image.shape[0], image.shape[1]
        tiles = torch.cat(
            (
                self.object_embedding(image[..., 0]),
                self.color_embedding(image[..., 1]),
                self.state_embedding(image[..., 2]),
            ),
            dim=-1,
        )
        flat = tiles.reshape(batch * steps, 7, 7, -1).permute(0, 3, 1, 2)
        features = self.observation_conv(flat).reshape(batch * steps, -1)
        projected = torch.relu(self.observation_projection(features))
        return projected.reshape(batch, steps, -1)

    def encode_mission(self, mission_tokens: Tensor, mission_lengths: Tensor) -> Tensor:
        """[B, L] -> [B, language_gru]; zeros when mission conditioning is off."""
        if not self.use_mission:
            return torch.zeros(
                mission_tokens.shape[0],
                self.model_cfg.language_gru,
                device=mission_tokens.device,
            )
        embedded = self.word_embedding(mission_tokens)
        if self.mission_encoder == "gru":
            outputs, _ = self.language_gru(embedded)
            index = (mission_lengths - 1).clamp(min=0)
            gather = index.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
            return outputs.gather(1, gather).squeeze(1)
        mask = (
            torch.arange(mission_tokens.shape[1], device=mission_tokens.device)
            .unsqueeze(0)
            .lt(mission_lengths.unsqueeze(1))
            .float()
            .unsqueeze(-1)
        )
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.relu(self.language_projection(pooled))

    def _fuse(
        self,
        observation_features: Tensor,
        mission_feature: Tensor,
        direction: Tensor,
        prev_executed_action: Tensor,
    ) -> Tensor:
        steps = observation_features.shape[1]
        parts = [observation_features]
        if self.use_mission:
            parts.append(mission_feature.unsqueeze(1).expand(-1, steps, -1))
        parts.append(self.direction_embedding(direction))
        if self.use_prev_action:
            parts.append(self.action_embedding(prev_executed_action))
        return torch.relu(self.fusion(torch.cat(parts, dim=-1)))

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
        observation_features = self.encode_observation(image)
        mission_feature = self.encode_mission(mission_tokens, mission_lengths)
        fused = self._fuse(
            observation_features, mission_feature, direction, prev_executed_action
        )
        if self.use_memory:
            hidden0 = None if initial_hidden is None else initial_hidden.unsqueeze(0)
            outputs, _ = self.policy_gru(fused, hidden0)
        else:
            outputs = self.trunk(fused)
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
        """Single-timestep inference with the same signature as the study model."""
        observation_features = self.encode_observation(image.unsqueeze(1))
        fused = self._fuse(
            observation_features,
            mission_feature,
            direction.unsqueeze(1),
            prev_executed_action.unsqueeze(1),
        )
        if self.use_memory:
            hidden0 = None if hidden is None else hidden.unsqueeze(0)
            outputs, _ = self.policy_gru(fused, hidden0)
        else:
            outputs = self.trunk(fused)
        logits = self.head(outputs[:, 0])
        return logits, outputs[:, 0]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
