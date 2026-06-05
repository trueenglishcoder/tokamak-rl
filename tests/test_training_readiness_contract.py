from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_env_reset import _write_small_sim_config
from tests.test_simple_trainer import ToyContinuousEnv
from tokamak_rl.contracts import KNOWN_TERMINATION_REASONS, REQUIRED_TRAINING_ARTIFACTS, TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.env import EnvConfig, TokamakRLEnv
from tokamak_rl.training import TCVStyleTrainerConfig, train_tcv_style_actor_critic


def test_env_reset_exposes_training_readiness_contract(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=sim_config,
            scenario_name="nominal",
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
        )
    )

    _obs, info = env.reset(seed=123)
    contract = info["episode_metadata"]["training_contract"]

    assert contract["contract_version"] == TRAINING_READINESS_CONTRACT_VERSION
    assert contract["simulator"]["config_path"] == str(sim_config)
    assert contract["simulator"]["boundary_mode"] == "limited"
    assert contract["environment"]["scenario_name"] == "nominal"
    assert contract["environment"]["angles"] == 8
    assert contract["reference"]["source_kind"] == "scenario"
    assert contract["reference"]["scenario_args"] == {}
    assert contract["randomization"]["enabled"] is False
    assert contract["randomization"]["seed"] == 123
    assert contract["randomization"]["has_nonzero_effect"] is False
    assert contract["randomization"]["simulator_realism"]["enabled"] is False
    assert contract["observation_schema"]["obs_dim"] == env.obs_dim
    assert contract["action_schema"]["action_dim"] == env.action_dim
    assert np.asarray(contract["action_schema"]["derivative_scale"], dtype=float).shape == (env.action_dim,)
    assert set(contract["termination"]["known_reasons"]) == set(KNOWN_TERMINATION_REASONS)
    env.close()


def test_synthetic_reset_contract_records_effective_reference_seeds(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    ip_path = tmp_path / "t15md_7000_ip.csv"
    _write_small_sim_config(sim_config)
    ip_path.write_text("0;0\n0.5;100000\n1.0;150000\n", encoding="utf-8")
    env = TokamakRLEnv(
        EnvConfig(
            sim_config_path=sim_config,
            scenario_name="t15_synthetic_follow",
            scenario_args={
                "seed": 5,
                "duration_s": 0.05,
                "t_step": 1.0e-3,
                "target_update_s": 0.01,
                "ip_template_csv": str(ip_path),
                "ip_seed": 7,
                "amplitude_jitter": 0.0,
                "duration_jitter": 0.0,
                "shape_jitter": 0.0,
            },
            angles=8,
            max_episode_steps=3,
            realism_enabled=False,
            resample_references_on_reset=True,
        )
    )

    _obs, info = env.reset(seed=100)
    reference = info["episode_metadata"]["training_contract"]["reference"]

    assert reference["source_kind"] == "t15_synthetic_follow"
    assert reference["resampling_enabled"] is True
    assert reference["base_seed"] == 5
    assert reference["base_ip_seed"] == 7
    assert reference["episode_seed"] == 100
    assert isinstance(reference["effective_seed"], int)
    assert isinstance(reference["effective_ip_seed"], int)
    env.close()


def test_tcv_training_writes_stage_l_artifact_contract(tmp_path: Path) -> None:
    cfg = TCVStyleTrainerConfig(
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
        eval_interval_steps=8,
        eval_episodes=1,
        eval_max_steps=4,
        seed=33,
        output_dir=tmp_path,
        checkpoint_dir=tmp_path,
        run_metadata={"experiment_name": "stage_l_contract"},
    )

    result = train_tcv_style_actor_critic(lambda: ToyContinuousEnv(max_steps=4), cfg)

    assert result.metrics_json is not None and result.metrics_json.exists()
    for name in REQUIRED_TRAINING_ARTIFACTS:
        assert (tmp_path / name).exists(), name
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["contract_version"] == TRAINING_READINESS_CONTRACT_VERSION
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == TRAINING_READINESS_CONTRACT_VERSION
    assert manifest["required_artifacts"] == list(REQUIRED_TRAINING_ARTIFACTS)
    assert all(manifest["present_artifacts"].values())
