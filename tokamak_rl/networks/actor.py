from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ActorConfig:
    """Feedforward policy architecture settings."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    min_std: float = 1.0e-6
    max_std: float = 10.0
    head_init_scale: float = 1.0e-4

    def __post_init__(self) -> None:
        if int(self.obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(self.action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")
        if float(self.min_std) <= 0.0:
            raise ValueError("min_std must be > 0")
        if float(self.max_std) < float(self.min_std):
            raise ValueError("max_std must be >= min_std")
        if float(self.head_init_scale) <= 0.0:
            raise ValueError("head_init_scale must be > 0")


class FeedForwardActor(nn.Module):
    """TCV-style feedforward actor for normalized derivative actions.

    The training path returns a tanh-squashed Gaussian action. The deterministic
    path returns ``tanh(mean)`` and is the only path intended for export.
    """

    architecture_name = "feedforward_actor_v1"

    def __init__(self, cfg: ActorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_dim)
        self.input = nn.Linear(int(cfg.obs_dim), hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.hidden1 = nn.Linear(hidden, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.hidden3 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, int(cfg.action_dim))
        self.std_head = nn.Linear(hidden, int(cfg.action_dim))
        self.reset_parameters()

    @property
    def obs_dim(self) -> int:
        return int(self.cfg.obs_dim)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    def reset_parameters(self) -> None:
        for module in (self.input, self.hidden1, self.hidden2, self.hidden3):
            std = 1.0 / module.in_features**0.5
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(module.bias)
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        for head in (self.mean_head, self.std_head):
            nn.init.uniform_(head.weight, -float(self.cfg.head_init_scale), float(self.cfg.head_init_scale))
            nn.init.zeros_(head.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        mean = self.mean_head(features)
        std = F.softplus(self.std_head(features)) + float(self.cfg.min_std)
        std = torch.clamp(std, min=float(self.cfg.min_std), max=float(self.cfg.max_std))
        return mean, std

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        obs = self._check_observation(observation)
        x = torch.tanh(self.input_norm(self.input(obs)))
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        x = F.elu(self.hidden3(x))
        return x

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _std = self.forward(observation)
        return torch.tanh(mean)

    def sample_action(self, observation: torch.Tensor, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, _log_prob, mean, std = self.sample_action_with_log_prob(observation, generator=generator)
        return action, mean, std

    def sample_action_with_log_prob(self, observation: torch.Tensor, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(observation)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        normal_log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
        squash_correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1.0e-6))
        log_prob = torch.sum(normal_log_prob - squash_correction, dim=-1)
        return action, log_prob, mean, std

    def _check_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(observation):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim != 2 or observation.shape[1] != self.obs_dim:
            raise ValueError(f"observation shape must be (batch, {self.obs_dim}), got {tuple(observation.shape)}")
        if not torch.all(torch.isfinite(observation)):
            raise ValueError("observation must contain finite values")
        return observation


__all__ = ["ActorConfig", "FeedForwardActor"]
