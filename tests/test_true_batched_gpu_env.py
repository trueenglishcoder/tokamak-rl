from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokamak_rl.config import load_experiment_config
from tokamak_rl.env.true_batched_gpu_env import TrueBatchedGpuTokamakEnv


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")


def test_true_batched_gpu_env_finds_keep_initial_boundaries_without_legacy_boundary_calls() -> None:
    from tokamak_control.core.batched_gpu_simulator import (
        batched_gpu_simulator_profiling_snapshot,
        configure_batched_gpu_simulator_profiling,
    )
    from tokamak_control.geometry.boundary import boundary_profiling_snapshot, configure_boundary_profiling

    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment_config(repo_root / "configs/experiments/t15md_training_keep_initial_boundary.yaml")
    env_config = replace(experiment.env, compute_backend="gpu", gpu_device="cuda:0")
    configure_batched_gpu_simulator_profiling(enabled=True, reset=True)
    configure_boundary_profiling(enabled=True, reset=True)

    env = TrueBatchedGpuTokamakEnv(env_config, num_envs=4, reward_fn=experiment.reward, randomizer=experiment.randomization)
    try:
        reset = env.reset_batch(np.arange(4, dtype=int))
        assert reset.observations.shape == (4, env.obs_dim)
        assert all(bool(info["snapshot"].boundary_found) for info in reset.infos)

        for _ in range(3):
            step = env.step_batch(np.zeros((4, env.action_dim), dtype=np.float32))
            assert step.observations.shape == (4, env.obs_dim)
            assert np.all(np.isfinite(step.rewards))
            assert all(bool(info["snapshot"].boundary_found) for info in step.infos)
            assert not np.any(step.terminated)
    finally:
        env.close()

    batched_profile = batched_gpu_simulator_profiling_snapshot()
    assert batched_profile["enabled"] is True
    assert batched_profile["total"]["calls"] == 3
    assert batched_profile["blocks"]["boundary_fixed_angle"]["calls"] > 0

    legacy_boundary_profile = boundary_profiling_snapshot()
    assert legacy_boundary_profile["cpu"]["total"]["calls"] == 0
    assert legacy_boundary_profile["gpu"]["total"]["calls"] == 0


def test_true_batched_gpu_env_partial_reset_keeps_boundary_valid() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment_config(repo_root / "configs/experiments/t15md_training_keep_initial_boundary.yaml")
    env_config = replace(experiment.env, compute_backend="gpu", gpu_device="cuda:0", max_episode_steps=5)

    env = TrueBatchedGpuTokamakEnv(env_config, num_envs=4, reward_fn=experiment.reward, randomizer=experiment.randomization)
    try:
        reset = env.reset_batch(np.arange(4, dtype=int))
        assert all(bool(info["snapshot"].boundary_found) for info in reset.infos)

        last_step = None
        for _ in range(5):
            last_step = env.step_batch(np.zeros((4, env.action_dim), dtype=np.float32))
            assert all(bool(info["snapshot"].boundary_found) for info in last_step.infos)
        assert last_step is not None
        assert np.all(last_step.truncated)
        assert not np.any(last_step.terminated)

        reset_obs, reset_infos = env.reset_indices([1, 3], [1001, 1003])
        assert reset_obs.shape == (2, env.obs_dim)
        assert [info["snapshot"].step_index for info in reset_infos] == [0, 0]
        assert all(bool(info["snapshot"].boundary_found) for info in reset_infos)

        next_step = env.step_batch(np.zeros((4, env.action_dim), dtype=np.float32))
        assert bool(next_step.truncated[0])
        assert not bool(next_step.truncated[1])
        assert bool(next_step.truncated[2])
        assert not bool(next_step.truncated[3])
        assert all(bool(info["snapshot"].boundary_found) for info in next_step.infos)
    finally:
        env.close()
