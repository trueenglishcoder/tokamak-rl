from __future__ import annotations

from pathlib import Path

from tests.test_env_reset import _write_small_sim_config
from tokamak_rl.env import EnvConfig, TrueBatchedGpuEnvFactory
from tokamak_rl.training.cli import _make_env_factory


class _Experiment:
    def __init__(self, env: EnvConfig) -> None:
        self.env = env
        self.reward = None
        self.randomization = None


def test_gpu_env_factory_uses_true_batched_gpu_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    experiment = _Experiment(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=2,
            realism_enabled=False,
            compute_backend="gpu",
        )
    )

    factory = _make_env_factory(experiment=experiment, process_envs=False, process_start_method="spawn", num_envs=2)

    assert isinstance(factory, TrueBatchedGpuEnvFactory)
    assert factory.num_envs == 2


def test_cpu_env_factory_keeps_single_env_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "small_sim.toml"
    _write_small_sim_config(config_path)
    experiment = _Experiment(
        EnvConfig(
            sim_config_path=config_path,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=2,
            realism_enabled=False,
            compute_backend="cpu",
        )
    )

    factory = _make_env_factory(experiment=experiment, process_envs=False, process_start_method="spawn", num_envs=2)

    assert not isinstance(factory, TrueBatchedGpuEnvFactory)
