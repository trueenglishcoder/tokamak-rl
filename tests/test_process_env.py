from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from tests.test_env_reset import _write_small_sim_config
from tokamak_control.realism import SensorRealismSettings
from tokamak_rl.env import EnvConfig, ProcessTokamakEnv, ProcessVectorEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.training import TCVStyleTrainerConfig, train_tcv_style_actor_critic


def _start_method() -> str:
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return "fork"
    if "spawn" in methods:
        return "spawn"
    pytest.skip("no supported multiprocessing start method available")


def test_process_tokamak_env_steps_and_returns_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env = ProcessTokamakEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=2,
            realism_enabled=False,
        ),
        randomizer=DomainRandomizer(enabled=True, sensors=SensorRealismSettings(ip_bias=5.0)),
        start_method=_start_method(),
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (env.obs_dim,)
        assert env.action_dim > 0
        assert info["episode_metadata"]["randomization"]["seed"] == 7
        assert info["snapshot"].measured_ip == pytest.approx(info["snapshot"].true_ip + 5.0)

        next_obs, reward, terminated, truncated, step_info = env.step(np.zeros((env.action_dim,), dtype=float))

        assert next_obs.shape == obs.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["snapshot"].step_index == 1
    finally:
        env.close()


def test_process_vector_env_uses_reproducible_worker_seeds(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    vector = ProcessVectorEnv(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=2,
            realism_enabled=False,
        ),
        num_envs=2,
        start_method=_start_method(),
    )
    try:
        obs, infos = vector.reset(seed=20)
        assert obs.shape == (2, vector.obs_dim)
        assert [info["episode_metadata"]["randomization"]["seed"] for info in infos] == [20, 21]

        actions = np.zeros((2, vector.action_dim), dtype=float)
        next_obs, rewards, terminated, truncated, step_infos = vector.step(actions)

        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert np.all(np.isfinite(rewards))
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert [info["snapshot"].step_index for info in step_infos] == [1, 1]
    finally:
        vector.close()


def test_tcv_trainer_can_collect_from_process_env_workers(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    env_cfg = EnvConfig(
        sim_config_path=config_path,
        scenario_name="nominal",
        angles=8,
        max_episode_steps=4,
        realism_enabled=False,
    )
    start_method = _start_method()
    trainer_cfg = TCVStyleTrainerConfig(
        total_steps=16,
        warmup_steps=4,
        batch_size=2,
        sequence_length=4,
        replay_capacity_episodes=8,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        critic_mlp_hidden_dim=16,
        num_envs=2,
        updates_per_episode=1,
        eval_episodes=1,
        eval_max_steps=4,
        seed=30,
        output_dir=tmp_path / "out",
        run_metadata={"process_envs": True, "process_start_method": start_method},
    )

    result = train_tcv_style_actor_critic(
        lambda: ProcessTokamakEnv(env_cfg, start_method=start_method),
        trainer_cfg,
    )

    assert result.total_steps == 16
    assert result.replay_episodes > 0
    assert result.critic_losses
    assert result.metrics_json is not None and result.metrics_json.exists()
