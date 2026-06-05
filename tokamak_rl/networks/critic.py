from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CriticConfig:
    """Feedforward Q critic settings for the first trainer smoke stage."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        if int(self.obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")


class FeedForwardQCritic(nn.Module):
    """Twin-critic building block used by the simple off-policy smoke trainer."""

    architecture_name = "feedforward_q_critic_v1"

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_dim)
        input_dim = int(cfg.obs_dim) + int(cfg.action_dim)
        self.input = nn.Linear(input_dim, hidden)
        self.hidden1 = nn.Linear(hidden, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.q_head = nn.Linear(hidden, 1)
        self.reset_parameters()

    @property
    def obs_dim(self) -> int:
        return int(self.cfg.obs_dim)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def reset_parameters(self) -> None:
        for module in (self.input, self.hidden1, self.hidden2):
            std = 1.0 / module.in_features**0.5
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(module.bias)
        nn.init.uniform_(self.q_head.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._check_inputs(observation, action)
        x = torch.cat([observation, action], dim=-1)
        x = F.elu(self.input(x))
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        return self.q_head(x).squeeze(-1)

    def _check_inputs(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[1] != self.obs_dim:
            raise ValueError(f"observation shape must be (batch, {self.obs_dim}), got {tuple(observation.shape)}")
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"action shape must be (batch, {self.action_dim}), got {tuple(action.shape)}")
        if observation.shape[0] != action.shape[0]:
            raise ValueError("observation and action batch sizes must match")
        if not torch.all(torch.isfinite(observation)) or not torch.all(torch.isfinite(action)):
            raise ValueError("critic inputs must contain finite values")


__all__ = ["CriticConfig", "FeedForwardQCritic"]
