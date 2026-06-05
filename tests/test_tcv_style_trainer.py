from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_simple_trainer import ToyContinuousEnv
from tokamak_rl.training import TCVStyleTrainerConfig, evaluate_tcv_actor, evaluate_tcv_actor_detailed, train_tcv_style_actor_critic


def test_tcv_style_recurrent_trainer_smoke_logs_and_checkpoint(tmp_path: Path) -> None:
    cfg = TCVStyleTrainerConfig(
        total_steps=96,
        warmup_steps=12,
        batch_size=4,
        sequence_length=6,
        replay_capacity_episodes=16,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        critic_mlp_hidden_dim=32,
        mpo_action_samples=4,
        mpo_temperature_iterations=3,
        num_envs=3,
        updates_per_episode=2,
        eval_interval_steps=48,
        checkpoint_interval_steps=48,
        eval_episodes=2,
        eval_max_steps=8,
        seed=13,
        output_dir=tmp_path,
        checkpoint_dir=tmp_path,
        run_metadata={"experiment_name": "toy_tcv", "sim": {"scenario_name": "toy"}},
    )

    result = train_tcv_style_actor_critic(lambda: ToyContinuousEnv(max_steps=8), cfg)

    assert result.total_steps == 96
    assert result.replay_episodes > 0
    assert result.replay_transitions == 96
    assert result.critic_losses
    assert result.actor_losses
    assert result.mpo_temperature_losses
    assert result.mpo_temperatures
    assert result.mpo_mean_kls
    assert result.mpo_std_kls
    assert result.mpo_kl_dual_losses
    assert result.valid_steps_per_update
    assert np.all(np.isfinite(result.critic_losses))
    assert np.all(np.isfinite(result.actor_losses))
    assert np.all(np.isfinite(result.mpo_temperature_losses))
    assert np.all(np.asarray(result.mpo_temperatures) > 0.0)
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert result.latest_checkpoint_path is not None and result.latest_checkpoint_path.exists()
    assert result.best_checkpoint_path is not None and result.best_checkpoint_path.exists()
    assert (tmp_path / "step_00000048.pt").exists()
    assert result.metrics_json is not None and result.metrics_json.exists()
    assert result.losses_csv is not None and result.losses_csv.exists()
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    config_snapshot = json.loads((tmp_path / "config_snapshot.json").read_text(encoding="utf-8"))
    assert metrics["trainer"] == "tcv_style_recurrent_actor_critic_v1"
    assert metrics["algorithm"] == "tcv_mpo_recurrent_actor_critic_v1"
    assert metrics["sequence_length"] == 6
    assert metrics["critic_updates"] == len(result.critic_losses)
    assert metrics["mpo_kl_dual_updates"] == len(result.mpo_kl_dual_losses)
    assert metrics["last_mpo_temperature"] > 0.0
    assert metrics["last_mpo_mean_kl_penalty"] > 0.0
    assert metrics["last_mpo_std_kl_penalty"] > 0.0
    assert len(metrics["eval_history"]) == 2
    assert metrics["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["boundary_failure_rate"] == 0.0
    assert metrics["tracking_diagnostics"]["reference_effective_seed_count"] >= 3
    assert metrics["run_metadata"]["experiment_name"] == "toy_tcv"
    assert metrics["config"]["run_metadata"]["sim"]["scenario_name"] == "toy"
    assert metrics["checkpoint_path"] == str(result.checkpoint_path)
    assert metrics["latest_checkpoint_path"] == str(result.latest_checkpoint_path)
    assert metrics["best_checkpoint_path"] == str(result.best_checkpoint_path)
    assert metrics["device"]["requested"] == "cpu"
    assert metrics["device"]["actual"] == "cpu"
    assert config_snapshot["device"] == "cpu"
    assert metrics["throughput"]["env_steps_per_second"] > 0.0
    assert metrics["throughput"]["learner_updates_per_second"] > 0.0
    assert metrics["throughput"]["actor_inference_time_s"] >= 0.0
    assert metrics["throughput"]["env_step_time_s"] > 0.0
    assert metrics["throughput"]["replay_sampling_time_s"] >= 0.0
    assert metrics["eval_tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["eval_tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["eval_history"][0]["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    reward_csv = tmp_path / "reward_components.csv"
    reward_lines = reward_csv.read_text(encoding="utf-8").splitlines()
    assert reward_lines[0] == "step,env_index,episode,component,value"
    assert any("ip_error_norm" in line for line in reward_lines[1:])
    assert any("shape_error_norm" in line for line in reward_lines[1:])
    episodes_text = (tmp_path / "episodes.csv").read_text(encoding="utf-8")
    assert "termination_reason" in episodes_text
    assert "reference_effective_seed" in episodes_text
    termination_counts = json.loads((tmp_path / "termination_counts.json").read_text(encoding="utf-8"))
    assert termination_counts["episodes"] == len(result.episode_returns)
    with np.load(tmp_path / "reference_samples.npz", allow_pickle=False) as samples:
        assert samples["reference_effective_seed"].shape[0] >= 3
        assert samples["reference_effective_ip_seed"].shape[0] >= 3
    rollout_dir = tmp_path / "rollouts" / "eval_step_00000048"
    assert (rollout_dir / "summary.json").exists()
    assert (rollout_dir / "episode_metrics.csv").exists()
    assert (rollout_dir / "rollouts.npz").exists()
    checkpoint = __import__("torch").load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["run_metadata"]["experiment_name"] == "toy_tcv"
    assert checkpoint["algorithm"] == "tcv_mpo_recurrent_actor_critic_v1"
    assert checkpoint["total_steps"] == 96
    assert checkpoint["actor_target_state_dict"]
    assert checkpoint["q1_target_state_dict"]
    assert checkpoint["q2_target_state_dict"]
    assert checkpoint["actor_optimizer_state_dict"] is not None
    assert checkpoint["critic_optimizer_state_dict"] is not None
    assert checkpoint["log_mpo_mean_kl_penalty"] is not None
    assert checkpoint["log_mpo_std_kl_penalty"] is not None
    assert checkpoint["mpo_kl_optimizer_state_dict"] is not None
    assert checkpoint["training_torch_generator_state"] is not None
    assert checkpoint["numpy_rng_state"] is not None


def test_tcv_style_recurrent_trainer_resumes_checkpoint(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_cfg = TCVStyleTrainerConfig(
        total_steps=48,
        warmup_steps=8,
        batch_size=4,
        sequence_length=4,
        actor_hidden_dim=24,
        critic_hidden_dim=24,
        critic_mlp_hidden_dim=24,
        num_envs=2,
        eval_episodes=1,
        eval_max_steps=8,
        seed=21,
        checkpoint_dir=first_dir,
    )
    first = train_tcv_style_actor_critic(lambda: ToyContinuousEnv(max_steps=8), first_cfg)
    assert first.checkpoint_path is not None

    resumed_dir = tmp_path / "resumed"
    resumed_cfg = TCVStyleTrainerConfig(
        total_steps=32,
        warmup_steps=0,
        batch_size=4,
        sequence_length=4,
        actor_hidden_dim=24,
        critic_hidden_dim=24,
        critic_mlp_hidden_dim=24,
        num_envs=2,
        eval_episodes=1,
        eval_max_steps=8,
        seed=22,
        output_dir=resumed_dir,
        checkpoint_dir=resumed_dir,
        resume_checkpoint=first.checkpoint_path,
    )

    resumed = train_tcv_style_actor_critic(lambda: ToyContinuousEnv(max_steps=8), resumed_cfg)

    assert resumed.total_steps == 32
    assert resumed.checkpoint_path is not None and resumed.checkpoint_path.exists()
    assert resumed.metrics_json is not None and resumed.metrics_json.exists()
    assert resumed.critic_losses
    assert np.all(np.isfinite(resumed.critic_losses))


def test_tcv_style_update_cadence_respects_catchup_cap(tmp_path: Path) -> None:
    cfg = TCVStyleTrainerConfig(
        total_steps=16,
        warmup_steps=4,
        batch_size=2,
        sequence_length=4,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        critic_mlp_hidden_dim=16,
        num_envs=1,
        updates_per_episode=3,
        updates_per_env_step=0,
        max_learner_catchup_updates=1,
        eval_episodes=1,
        eval_max_steps=4,
        seed=31,
        output_dir=tmp_path,
    )

    result = train_tcv_style_actor_critic(lambda: ToyContinuousEnv(max_steps=4), cfg)

    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert len(result.critic_losses) == 4
    assert metrics["config"]["updates_per_episode"] == 3
    assert metrics["config"]["max_learner_catchup_updates"] == 1
    assert metrics["throughput"]["update_to_data_ratio"] == pytest.approx(len(result.critic_losses) / result.total_steps)


def test_tcv_style_config_validation() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        TCVStyleTrainerConfig(sequence_length=0)
    with pytest.raises(ValueError, match="gamma"):
        TCVStyleTrainerConfig(gamma=-0.1)
    with pytest.raises(ValueError, match="mpo_epsilon"):
        TCVStyleTrainerConfig(mpo_epsilon=0.0)
    with pytest.raises(ValueError, match="mpo_action_samples"):
        TCVStyleTrainerConfig(mpo_action_samples=0)
    with pytest.raises(ValueError, match="updates_per_env_step"):
        TCVStyleTrainerConfig(updates_per_env_step=-1)
    with pytest.raises(ValueError, match="max_learner_catchup_updates"):
        TCVStyleTrainerConfig(max_learner_catchup_updates=0)
    with pytest.raises(ValueError, match="max_step_checkpoints"):
        TCVStyleTrainerConfig(max_step_checkpoints=0)


def test_tcv_style_evaluate_actor_runs_loaded_policy_shape() -> None:
    from tokamak_rl.networks import ActorConfig, FeedForwardActor

    actor = FeedForwardActor(ActorConfig(obs_dim=4, action_dim=2, hidden_dim=16))
    returns = evaluate_tcv_actor(lambda: ToyContinuousEnv(max_steps=4), actor, episodes=2, max_steps=4, seed=5)
    detailed = evaluate_tcv_actor_detailed(lambda: ToyContinuousEnv(max_steps=4), actor, episodes=2, max_steps=4, seed=5)

    assert len(returns) == 2
    assert np.all(np.isfinite(returns))
    assert detailed["returns"] == returns
    assert detailed["tracking_diagnostics"]["mean_ip_error_norm"] is not None
