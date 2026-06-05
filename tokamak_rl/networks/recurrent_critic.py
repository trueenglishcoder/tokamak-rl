from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class RecurrentCriticConfig:
    """Recurrent Q critic settings for sequence-based training experiments."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    mlp_hidden_dim: int = 256

    def __post_init__(self) -> None:
        if int(self.obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")
        if int(self.mlp_hidden_dim) <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")


class RecurrentQCritic(nn.Module):
    """LSTM Q critic over trajectory chunks.

    The critic consumes ``concat(observation, action)`` at every timestep, runs an
    LSTM, then concatenates the LSTM output back with the per-step critic input.
    It returns one scalar Q estimate per batch item and timestep.
    """

    architecture_name = "recurrent_q_critic_v1"

    def __init__(self, cfg: RecurrentCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = int(cfg.obs_dim) + int(cfg.action_dim)
        hidden_dim = int(cfg.hidden_dim)
        mlp_hidden_dim = int(cfg.mlp_hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden1 = nn.Linear(hidden_dim + input_dim, mlp_hidden_dim)
        self.hidden2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.q_head = nn.Linear(mlp_hidden_dim, 1)
        self.reset_parameters()

    @property
    def obs_dim(self) -> int:
        return int(self.cfg.obs_dim)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                std = 1.0 / param.shape[1] ** 0.5
                nn.init.trunc_normal_(param, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in (self.hidden1, self.hidden2):
            std = 1.0 / module.in_features**0.5
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(module.bias)
        nn.init.uniform_(self.q_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.q_head.bias)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._check_inputs(observation, action)
        critic_input = torch.cat([observation, action], dim=-1)
        recurrent, _hidden_out = self.lstm(critic_input, hidden)
        x = torch.cat([recurrent, critic_input], dim=-1)
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        return self.q_head(x).squeeze(-1)

    def _check_inputs(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        if observation.ndim != 3 or observation.shape[2] != self.obs_dim:
            raise ValueError(f"observation shape must be (batch, time, {self.obs_dim}), got {tuple(observation.shape)}")
        if action.ndim != 3 or action.shape[2] != self.action_dim:
            raise ValueError(f"action shape must be (batch, time, {self.action_dim}), got {tuple(action.shape)}")
        if observation.shape[:2] != action.shape[:2]:
            raise ValueError("observation and action batch/time dimensions must match")
        if not torch.all(torch.isfinite(observation)) or not torch.all(torch.isfinite(action)):
            raise ValueError("recurrent critic inputs must contain finite values")


__all__ = ["RecurrentCriticConfig", "RecurrentQCritic"]
