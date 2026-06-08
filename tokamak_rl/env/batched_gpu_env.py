from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tokamak_rl.env.config import EnvConfig
from tokamak_rl.env.tokamak_env import TokamakRLEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward


@dataclass(slots=True)
class BatchedGpuTokamakEnvSlot:
    """Single-env facade backed by an in-process GPU environment pool."""

    pool: BatchedGpuTokamakEnvPool
    index: int

    @property
    def obs_dim(self) -> int:
        return self.pool.obs_dim(self.index)

    @property
    def action_dim(self) -> int:
        return self.pool.action_dim(self.index)

    def reset(self, seed: int | None = None, options: dict | None = None):
        return self.pool.reset_slot(self.index, seed=seed, options=options)

    def step(self, action: np.ndarray):
        return self.pool.step_slot(self.index, action)

    def close(self) -> None:
        self.pool.close_slot(self.index)


class BatchedGpuTokamakEnvPool:
    """In-process pool for GPU simulator sessions used by synchronous trainers.

    This deliberately avoids multiprocessing for GPU simulation. Each slot keeps
    the public single-env API expected by the existing trainers, while the pool
    owns all simulator states in one process and can be extended to true tensor
    batch stepping without changing trainer call sites.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        *,
        num_envs: int,
        reward_fn: JointCurrentBoundaryReward | None = None,
        randomizer: DomainRandomizer | None = None,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        if env_config.compute_backend != "gpu":
            raise ValueError("BatchedGpuTokamakEnvPool requires env_config.compute_backend == 'gpu'")
        self.envs = [
            TokamakRLEnv(env_config, reward_fn=reward_fn, randomizer=randomizer)
            for _ in range(int(num_envs))
        ]
        self._closed = [False for _ in self.envs]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def slot(self, index: int) -> BatchedGpuTokamakEnvSlot:
        self._validate_index(index)
        return BatchedGpuTokamakEnvSlot(pool=self, index=int(index))

    def obs_dim(self, index: int) -> int:
        self._validate_index(index)
        return self.envs[int(index)].obs_dim

    def action_dim(self, index: int) -> int:
        self._validate_index(index)
        return self.envs[int(index)].action_dim

    def reset_slot(self, index: int, *, seed: int | None = None, options: dict | None = None):
        self._validate_index(index)
        self._closed[int(index)] = False
        obs, info = self.envs[int(index)].reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), _with_batched_metadata(info, pool_size=self.num_envs, slot_index=int(index))

    def step_slot(self, index: int, action: np.ndarray):
        self._validate_index(index)
        obs, reward, terminated, truncated, info = self.envs[int(index)].step(action)
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            _with_batched_metadata(info, pool_size=self.num_envs, slot_index=int(index)),
        )

    def close_slot(self, index: int) -> None:
        self._validate_index(index)
        if not self._closed[int(index)]:
            self.envs[int(index)].close()
            self._closed[int(index)] = True

    def close(self) -> None:
        for index in range(self.num_envs):
            self.close_slot(index)

    def _validate_index(self, index: int) -> None:
        if int(index) < 0 or int(index) >= len(self.envs):
            raise IndexError("batched GPU env slot index out of range")


class BatchedGpuEnvFactory:
    """Callable factory that returns stable slots from one GPU env pool."""

    def __init__(
        self,
        env_config: EnvConfig,
        *,
        num_envs: int,
        reward_fn: JointCurrentBoundaryReward | None = None,
        randomizer: DomainRandomizer | None = None,
    ) -> None:
        self.pool = BatchedGpuTokamakEnvPool(env_config, num_envs=num_envs, reward_fn=reward_fn, randomizer=randomizer)
        self._next_index = 0

    def __call__(self) -> BatchedGpuTokamakEnvSlot:
        if self._next_index >= self.pool.num_envs:
            raise RuntimeError("BatchedGpuEnvFactory was called more times than num_envs")
        slot = self.pool.slot(self._next_index)
        self._next_index += 1
        return slot


def _with_batched_metadata(info: dict[str, Any], *, pool_size: int, slot_index: int) -> dict[str, Any]:
    out = dict(info)
    episode_metadata = dict(out.get("episode_metadata", {})) if isinstance(out.get("episode_metadata"), dict) else {}
    episode_metadata["gpu_env_pool"] = {
        "enabled": True,
        "pool_size": int(pool_size),
        "slot_index": int(slot_index),
        "process_envs": False,
    }
    out["episode_metadata"] = episode_metadata
    return out
