from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_env_reset import _write_small_sim_config
from tokamak_rl.env import EnvConfig
from tokamak_rl.evaluation.rollouts import RolloutConfig, run_rollout_evaluation
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward


def test_zero_policy_rollout_writes_expected_outputs(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    out = tmp_path / "eval_out"

    result = run_rollout_evaluation(
        RolloutConfig(
            env=EnvConfig(
                sim_config_path=sim_config,
                scenario_name="nominal",
                angles=8,
                max_episode_steps=2,
                realism_enabled=False,
            ),
            episodes=1,
            policy="zero",
            seed=5,
            output_dir=out,
        )
    )

    outputs = result["outputs"]
    assert outputs.summary_json.exists()
    assert outputs.episode_metrics_csv.exists()
    assert outputs.rollouts_npz.exists()
    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert summary["policy"] == "zero"
    assert summary["episodes"] == 1
    with np.load(outputs.rollouts_npz, allow_pickle=False) as data:
        assert data["actions"].shape == (1, 2, 3)
        assert data["rewards"].shape == (1, 2)
        assert data["observations"].shape[0] == 1
        assert data["mask"].shape == (1, 2)
        assert data["true_ip"].shape == (1, 2)
        assert data["measured_ip"].shape == (1, 2)
        assert data["ip_ref"].shape == (1, 2)
        assert data["radii_ref"].shape == (1, 2, 8)
        assert data["boundary_found"].shape == (1, 2)
        assert data["measured_boundary_available"].shape == (1, 2)
        assert np.all(data["actions"] == 0.0)


def test_random_policy_rollout_is_seed_reproducible(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    env = EnvConfig(sim_config_path=sim_config, scenario_name="nominal", angles=8, max_episode_steps=2, realism_enabled=False)

    a = run_rollout_evaluation(RolloutConfig(env=env, episodes=1, policy="random", seed=7))
    b = run_rollout_evaluation(RolloutConfig(env=env, episodes=1, policy="random", seed=7))

    assert np.allclose(a["actions"][0], b["actions"][0])
    assert np.allclose(a["rewards"][0], b["rewards"][0])


def test_rollout_outputs_true_and_measured_diagnostics_under_realism(tmp_path: Path) -> None:
    from tokamak_control.realism import RealismSettings, SensorRealismSettings

    sim_config = tmp_path / "small_sim_realism.toml"
    ip_bias = 25.0
    _write_small_sim_config(
        sim_config,
        realism=RealismSettings(enabled=True, seed=19, sensors=SensorRealismSettings(ip_bias=ip_bias)),
    )
    out = tmp_path / "eval_out"

    result = run_rollout_evaluation(
        RolloutConfig(
            env=EnvConfig(
                sim_config_path=sim_config,
                scenario_name="nominal",
                angles=8,
                max_episode_steps=2,
                realism_enabled=True,
            ),
            episodes=1,
            policy="zero",
            seed=5,
            output_dir=out,
        )
    )

    outputs = result["outputs"]
    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert "boundary_failure_rate" in summary
    assert "measured_boundary_missing_rate" in summary
    with np.load(outputs.rollouts_npz, allow_pickle=False) as data:
        assert data["true_ip"].shape == (1, 2)
        assert data["measured_ip"].shape == (1, 2)
        assert data["ip_ref"].shape == (1, 2)
        assert data["radii_ref"].shape == (1, 2, 8)
        assert np.allclose(data["measured_ip"] - data["true_ip"], ip_bias)
        assert data["boundary_found"].shape == (1, 2)
        assert data["measured_boundary_available"].shape == (1, 2)


def test_synthetic_rollout_records_reference_targets_and_effective_seeds(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    ip_path = tmp_path / "t15md_7171_ip.csv"
    _write_small_sim_config(sim_config)
    ip_path.write_text("0;0\n0.5;100000\n1.0;150000\n", encoding="utf-8")
    out = tmp_path / "synthetic_eval_out"

    result = run_rollout_evaluation(
        RolloutConfig(
            env=EnvConfig(
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
                max_episode_steps=2,
                realism_enabled=False,
                resample_references_on_reset=True,
            ),
            episodes=2,
            policy="zero",
            seed=100,
            output_dir=out,
        )
    )

    assert result["episode_metrics"][0]["reference_resampling_enabled"] is True
    assert result["episode_metrics"][0]["reference_effective_seed"] != result["episode_metrics"][1]["reference_effective_seed"]
    assert result["ip_ref"][0].shape == (2,)
    assert result["radii_ref"][0].shape == (2, 8)
    outputs = result["outputs"]
    csv_text = outputs.episode_metrics_csv.read_text(encoding="utf-8")
    assert "reference_effective_seed" in csv_text
    with np.load(outputs.rollouts_npz, allow_pickle=False) as data:
        assert data["ip_ref"].shape == (2, 2)
        assert data["radii_ref"].shape == (2, 2, 8)
        assert not np.allclose(data["radii_ref"][0, 0], data["radii_ref"][1, 0])


def test_rollout_evaluation_uses_supplied_reward_config(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    env = EnvConfig(sim_config_path=sim_config, scenario_name="nominal", angles=8, max_episode_steps=2, realism_enabled=False)
    zero_reward = JointCurrentBoundaryReward(
        ip_weight=0.0,
        shape_weight=0.0,
        action_weight=0.0,
        delta_action_weight=0.0,
        current_limit_weight=0.0,
        derivative_limit_weight=0.0,
        termination_penalty=0.0,
    )

    result = run_rollout_evaluation(RolloutConfig(env=env, episodes=1, policy="zero", seed=5, reward=zero_reward))

    assert np.allclose(result["rewards"][0], 0.0)
    assert result["episode_metrics"][0]["return"] == 0.0


def test_rollout_evaluation_records_supplied_randomization_metadata(tmp_path: Path) -> None:
    sim_config = tmp_path / "small_sim.toml"
    _write_small_sim_config(sim_config)
    out = tmp_path / "eval_out"
    env = EnvConfig(sim_config_path=sim_config, scenario_name="nominal", angles=8, max_episode_steps=2, realism_enabled=False)

    result = run_rollout_evaluation(
        RolloutConfig(
            env=env,
            episodes=1,
            policy="zero",
            seed=5,
            output_dir=out,
            randomizer=DomainRandomizer(enabled=True),
        )
    )

    assert result["episode_metrics"][0]["randomization_enabled"] is True
    assert result["episode_metrics"][0]["randomization_seed"] == 5
    csv_text = result["outputs"].episode_metrics_csv.read_text(encoding="utf-8")
    assert "randomization_enabled" in csv_text
    assert "randomization_seed" in csv_text
