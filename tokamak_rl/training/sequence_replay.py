from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def length(self) -> int:
        return int(self.rewards.shape[0])


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    mask: np.ndarray


class EpisodeReplayBuffer:
    """Replay buffer storing complete episodes and sampling padded chunks."""

    def __init__(self, *, capacity_episodes: int, obs_dim: int, action_dim: int) -> None:
        if int(capacity_episodes) <= 0:
            raise ValueError("capacity_episodes must be > 0")
        if int(obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        self.capacity_episodes = int(capacity_episodes)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._episodes: list[Episode] = []

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def total_transitions(self) -> int:
        return int(sum(ep.length for ep in self._episodes))

    def add_episode(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        episode = _episode_from_arrays(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes = self._episodes[-self.capacity_episodes :]

    def sample_sequences(self, *, batch_size: int, sequence_length: int, rng: np.random.Generator) -> SequenceBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if int(sequence_length) <= 0:
            raise ValueError("sequence_length must be > 0")
        if not self._episodes:
            raise ValueError("no episodes available")
        batch = _empty_sequence_batch(int(batch_size), int(sequence_length), self.obs_dim, self.action_dim)
        for batch_index in range(int(batch_size)):
            episode = self._episodes[int(rng.integers(0, len(self._episodes)))]
            max_start = max(episode.length - 1, 0)
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            valid_len = min(int(sequence_length), episode.length - start)
            sl = slice(start, start + valid_len)
            batch.observations[batch_index, :valid_len] = episode.observations[sl]
            batch.actions[batch_index, :valid_len] = episode.actions[sl]
            batch.rewards[batch_index, :valid_len] = episode.rewards[sl]
            batch.next_observations[batch_index, :valid_len] = episode.next_observations[sl]
            batch.terminated[batch_index, :valid_len] = episode.terminated[sl]
            batch.truncated[batch_index, :valid_len] = episode.truncated[sl]
            batch.mask[batch_index, :valid_len] = True
        return batch


def _episode_from_arrays(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    obs_dim: int,
    action_dim: int,
) -> Episode:
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    rew = np.asarray(rewards, dtype=np.float32).reshape(-1)
    next_obs = np.asarray(next_observations, dtype=np.float32)
    term = np.asarray(terminated, dtype=bool).reshape(-1)
    trunc = np.asarray(truncated, dtype=bool).reshape(-1)
    length = int(rew.shape[0])
    if length <= 0:
        raise ValueError("episode must contain at least one transition")
    if obs.shape != (length, int(obs_dim)) or next_obs.shape != (length, int(obs_dim)):
        raise ValueError("episode observation arrays must match (length, obs_dim)")
    if act.shape != (length, int(action_dim)):
        raise ValueError("episode action array must match (length, action_dim)")
    if term.shape != (length,) or trunc.shape != (length,):
        raise ValueError("episode done arrays must match length")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act)) or not np.all(np.isfinite(rew)) or not np.all(np.isfinite(next_obs)):
        raise ValueError("episode arrays must contain finite values")
    return Episode(
        observations=obs.copy(),
        actions=act.copy(),
        rewards=rew.copy(),
        next_observations=next_obs.copy(),
        terminated=term.copy(),
        truncated=trunc.copy(),
    )


def _empty_sequence_batch(batch_size: int, sequence_length: int, obs_dim: int, action_dim: int) -> SequenceBatch:
    return SequenceBatch(
        observations=np.zeros((batch_size, sequence_length, obs_dim), dtype=np.float32),
        actions=np.zeros((batch_size, sequence_length, action_dim), dtype=np.float32),
        rewards=np.zeros((batch_size, sequence_length), dtype=np.float32),
        next_observations=np.zeros((batch_size, sequence_length, obs_dim), dtype=np.float32),
        terminated=np.zeros((batch_size, sequence_length), dtype=bool),
        truncated=np.zeros((batch_size, sequence_length), dtype=bool),
        mask=np.zeros((batch_size, sequence_length), dtype=bool),
    )


__all__ = ["Episode", "EpisodeReplayBuffer", "SequenceBatch", "TensorEpisodeReplayBuffer"]


class TensorEpisodeReplayBuffer:
    """Replay buffer storing sequence chunks as tensors on the learner device."""

    def __init__(self, *, capacity_episodes: int, obs_dim: int, action_dim: int, device: torch.device | str) -> None:
        if int(capacity_episodes) <= 0:
            raise ValueError("capacity_episodes must be > 0")
        if int(obs_dim) <= 0:
            raise ValueError("obs_dim must be > 0")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be > 0")
        self.capacity_episodes = int(capacity_episodes)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self._episodes: list[Episode] = []

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def total_transitions(self) -> int:
        return int(sum(ep.length for ep in self._episodes))

    def add_episode(
        self,
        *,
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
    ) -> None:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(terminated, dtype=torch.bool, device=self.device).reshape(-1)
        trunc = torch.as_tensor(truncated, dtype=torch.bool, device=self.device).reshape(-1)
        length = int(rew.shape[0])
        if length <= 0:
            raise ValueError("episode must contain at least one transition")
        if tuple(obs.shape) != (length, self.obs_dim) or tuple(next_obs.shape) != (length, self.obs_dim):
            raise ValueError("episode observation arrays must match (length, obs_dim)")
        if tuple(act.shape) != (length, self.action_dim):
            raise ValueError("episode action array must match (length, action_dim)")
        if tuple(term.shape) != (length,) or tuple(trunc.shape) != (length,):
            raise ValueError("episode done arrays must match length")
        finite = torch.all(torch.isfinite(obs))
        finite = finite & torch.all(torch.isfinite(act))
        finite = finite & torch.all(torch.isfinite(rew))
        finite = finite & torch.all(torch.isfinite(next_obs))
        if not bool(finite.detach().cpu()):
            raise ValueError("episode arrays must contain finite values")
        self._episodes.append(
            Episode(
                observations=obs.detach().clone(),
                actions=act.detach().clone(),
                rewards=rew.detach().clone(),
                next_observations=next_obs.detach().clone(),
                terminated=term.detach().clone(),
                truncated=trunc.detach().clone(),
            )
        )
        if len(self._episodes) > self.capacity_episodes:
            self._episodes = self._episodes[-self.capacity_episodes :]

    def sample_sequences(self, *, batch_size: int, sequence_length: int, rng: np.random.Generator) -> SequenceBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if int(sequence_length) <= 0:
            raise ValueError("sequence_length must be > 0")
        if not self._episodes:
            raise ValueError("no episodes available")
        batch = _empty_tensor_sequence_batch(int(batch_size), int(sequence_length), self.obs_dim, self.action_dim, self.device)
        for batch_index in range(int(batch_size)):
            episode = self._episodes[int(rng.integers(0, len(self._episodes)))]
            max_start = max(episode.length - 1, 0)
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            valid_len = min(int(sequence_length), episode.length - start)
            sl = slice(start, start + valid_len)
            batch.observations[batch_index, :valid_len] = episode.observations[sl]
            batch.actions[batch_index, :valid_len] = episode.actions[sl]
            batch.rewards[batch_index, :valid_len] = episode.rewards[sl]
            batch.next_observations[batch_index, :valid_len] = episode.next_observations[sl]
            batch.terminated[batch_index, :valid_len] = episode.terminated[sl]
            batch.truncated[batch_index, :valid_len] = episode.truncated[sl]
            batch.mask[batch_index, :valid_len] = True
        return batch


def _empty_tensor_sequence_batch(batch_size: int, sequence_length: int, obs_dim: int, action_dim: int, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        observations=torch.zeros((batch_size, sequence_length, obs_dim), dtype=torch.float32, device=device),
        actions=torch.zeros((batch_size, sequence_length, action_dim), dtype=torch.float32, device=device),
        rewards=torch.zeros((batch_size, sequence_length), dtype=torch.float32, device=device),
        next_observations=torch.zeros((batch_size, sequence_length, obs_dim), dtype=torch.float32, device=device),
        terminated=torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device),
        truncated=torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device),
        mask=torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device),
    )
