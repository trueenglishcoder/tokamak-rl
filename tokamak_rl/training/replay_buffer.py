from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    """Fixed-size circular transition replay buffer."""

    def __init__(self, *, capacity: int, obs_dim: int, action_dim: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be > 0")
        if int(obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.terminated = np.zeros((self.capacity,), dtype=bool)
        self.truncated = np.zeros((self.capacity,), dtype=bool)
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return int(self._size)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if obs.shape != (self.obs_dim,) or next_obs.shape != (self.obs_dim,):
            raise ValueError("observation shape does not match replay buffer obs_dim")
        if act.shape != (self.action_dim,):
            raise ValueError("action shape does not match replay buffer action_dim")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act)) or not np.all(np.isfinite(next_obs)):
            raise ValueError("replay transition arrays must be finite")
        if not np.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        i = self._pos
        self.observations[i] = obs
        self.actions[i] = act
        self.rewards[i] = float(reward)
        self.next_observations[i] = next_obs
        self.terminated[i] = bool(terminated)
        self.truncated[i] = bool(truncated)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> ReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if self.size < int(batch_size):
            raise ValueError("not enough replay samples")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=self.observations[idx].copy(),
            actions=self.actions[idx].copy(),
            rewards=self.rewards[idx].copy(),
            next_observations=self.next_observations[idx].copy(),
            terminated=self.terminated[idx].copy(),
            truncated=self.truncated[idx].copy(),
        )


__all__ = ["ReplayBatch", "ReplayBuffer"]
