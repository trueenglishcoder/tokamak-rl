from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from tokamak_rl.env.config import EnvConfig
from tokamak_rl.env.tokamak_env import TokamakRLEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward


@dataclass(frozen=True, slots=True)
class WorkerTransition:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class ProcessTokamakEnv:
    """Process-owned TokamakRLEnv proxy with the normal env API.

    The learner process performs policy inference and sends actions to this
    proxy. The worker process owns the simulator session and returns transition
    data. This avoids pickling arbitrary env-factory lambdas while allowing real
    tokamak-sim environments to step in parallel across CPU processes.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        *,
        reward_fn: JointCurrentBoundaryReward | None = None,
        randomizer: DomainRandomizer | None = None,
        start_method: str = "spawn",
    ) -> None:
        self._ctx = mp.get_context(start_method)
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, env_config, reward_fn, randomizer),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        self._obs_dim: int | None = None
        self._action_dim: int | None = None
        self._closed = False

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            raise RuntimeError("reset() must be called before obs_dim is available")
        return int(self._obs_dim)

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("reset() must be called before action_dim is available")
        return int(self._action_dim)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._send(("reset", {"seed": seed, "options": options}))
        obs, info, obs_dim, action_dim = self._recv_ok()
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        self._send(("step", np.asarray(action, dtype=float)))
        obs, reward, terminated, truncated, info = self._recv_ok()
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._send(("close", None))
                self._recv_ok()
        finally:
            self._closed = True
            self._parent_conn.close()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)

    def _send(self, message: tuple[str, object]) -> None:
        if self._closed:
            raise RuntimeError("process env is closed")
        self._parent_conn.send(message)

    def _recv_ok(self):
        kind, payload = self._parent_conn.recv()
        if kind == "ok":
            return payload
        if kind == "error":
            raise RuntimeError(str(payload))
        raise RuntimeError(f"unknown worker response: {kind!r}")


class ProcessVectorEnv:
    """Small synchronous vector wrapper around process-owned TokamakRLEnv workers."""

    def __init__(
        self,
        env_config: EnvConfig,
        *,
        num_envs: int,
        reward_fn: JointCurrentBoundaryReward | None = None,
        randomizer: DomainRandomizer | None = None,
        start_method: str = "spawn",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.envs = [
            ProcessTokamakEnv(env_config, reward_fn=reward_fn, randomizer=randomizer, start_method=start_method)
            for _ in range(int(num_envs))
        ]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    def reset(self, *, seed: int) -> tuple[np.ndarray, list[dict[str, object]]]:
        observations: list[np.ndarray] = []
        infos: list[dict[str, object]] = []
        for index, env in enumerate(self.envs):
            obs, info = env.reset(seed=int(seed) + index)
            observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            infos.append(info)
        return np.stack(observations, axis=0), infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
        arr = np.asarray(actions, dtype=float)
        if arr.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.action_dim}), got {arr.shape}")
        observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, object]] = []
        for env, action in zip(self.envs, arr, strict=True):
            obs, reward, term, trunc, info = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            rewards.append(float(reward))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            infos.append(info)
        return (
            np.stack(observations, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker_main(
    conn: Connection,
    env_config: EnvConfig,
    reward_fn: JointCurrentBoundaryReward | None,
    randomizer: DomainRandomizer | None,
) -> None:
    env = TokamakRLEnv(env_config, reward_fn=reward_fn, randomizer=randomizer)
    try:
        while True:
            command, payload = conn.recv()
            try:
                if command == "reset":
                    args = dict(payload) if isinstance(payload, dict) else {}
                    obs, info = env.reset(seed=args.get("seed"), options=args.get("options"))
                    conn.send(("ok", (np.asarray(obs, dtype=np.float32), _metadata_safe(info), env.obs_dim, env.action_dim)))
                elif command == "step":
                    obs, reward, terminated, truncated, info = env.step(payload)
                    conn.send(("ok", (np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), _metadata_safe(info))))
                elif command == "close":
                    env.close()
                    conn.send(("ok", None))
                    return
                else:
                    raise ValueError(f"unknown command: {command!r}")
            except BaseException as exc:  # noqa: BLE001 - worker must report failures across process boundary.
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        env.close()
        conn.close()


def _metadata_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _metadata_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    return value


__all__ = ["ProcessTokamakEnv", "ProcessVectorEnv", "WorkerTransition"]
