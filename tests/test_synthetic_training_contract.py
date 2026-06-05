from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_rl.config import load_experiment_config
from tokamak_rl.env import TokamakRLEnv
from tokamak_rl.export import load_numpy_actor
from tokamak_rl.training.cli import main as train_cli_main
from tokamak_rl.training import SimpleTrainerConfig, TCVStyleTrainerConfig, train_simple_actor_critic, train_tcv_style_actor_critic


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_EXPERIMENT = REPO_ROOT / "configs/experiments/t15md_synthetic_joint_current_boundary.yaml"
STAGE_U_SMOKE_EXPERIMENT = REPO_ROOT / "configs/experiments/t15md_training_real_replay_like_smoke.yaml"


def test_real_synthetic_reference_training_smoke_writes_diagnostics_and_provenance(tmp_path: Path) -> None:
    """Prove the real synthetic T15 config can drive a tiny trainer end to end."""
    if not SYNTHETIC_EXPERIMENT.exists():
        pytest.skip(f"Missing synthetic experiment config: {SYNTHETIC_EXPERIMENT}")
    experiment = load_experiment_config(SYNTHETIC_EXPERIMENT)
    if not Path(experiment.env.sim_config_path).exists():
        pytest.skip(f"Missing simulator config: {experiment.env.sim_config_path}")
    if "ip_template_dir" in experiment.env.scenario_args and not Path(str(experiment.env.scenario_args["ip_template_dir"])).exists():
        pytest.skip(f"Missing Ip template dir: {experiment.env.scenario_args['ip_template_dir']}")

    env_cfg = replace(experiment.env, max_episode_steps=2, angles=32, realism_enabled=False)
    out_dir = tmp_path / "train_out"
    ckpt_dir = tmp_path / "ckpt"
    run_metadata = {
        "experiment_name": experiment.name,
        "experiment_config_path": str(experiment.source_path),
        "trainer_name": "simple",
        "sim": {
            "scenario_name": env_cfg.scenario_name,
            "scenario_args": dict(env_cfg.scenario_args),
            "angles": env_cfg.angles,
            "max_episode_steps": env_cfg.max_episode_steps,
        },
    }
    trainer_cfg = SimpleTrainerConfig(
        total_steps=4,
        warmup_steps=2,
        batch_size=2,
        replay_capacity=16,
        hidden_dim=32,
        num_envs=1,
        eval_episodes=1,
        eval_max_steps=2,
        eval_interval_steps=2,
        seed=17,
        output_dir=out_dir,
        checkpoint_dir=ckpt_dir,
        run_metadata=run_metadata,
    )

    result = train_simple_actor_critic(lambda: TokamakRLEnv(env_cfg), trainer_cfg)

    assert result.total_steps == 4
    assert result.metrics_json is not None and result.metrics_json.exists()
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["run_metadata"]["experiment_name"] == "t15md_synthetic_joint_current_boundary"
    assert metrics["run_metadata"]["sim"]["scenario_name"] == "t15_synthetic_follow"
    assert "ip_template_dir" in metrics["run_metadata"]["sim"]["scenario_args"]
    assert metrics["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["reference_effective_seed_count"] >= 1
    assert metrics["eval_tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["eval_tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["eval_history"][0]["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert np.all(np.isfinite(result.eval_returns))


def test_real_synthetic_reference_tcv_style_training_smoke_writes_diagnostics_and_provenance(tmp_path: Path) -> None:
    """Prove the real synthetic T15 config can drive the main TCV-style trainer."""
    if not SYNTHETIC_EXPERIMENT.exists():
        pytest.skip(f"Missing synthetic experiment config: {SYNTHETIC_EXPERIMENT}")
    experiment = load_experiment_config(SYNTHETIC_EXPERIMENT)
    if not Path(experiment.env.sim_config_path).exists():
        pytest.skip(f"Missing simulator config: {experiment.env.sim_config_path}")
    if "ip_template_dir" in experiment.env.scenario_args and not Path(str(experiment.env.scenario_args["ip_template_dir"])).exists():
        pytest.skip(f"Missing Ip template dir: {experiment.env.scenario_args['ip_template_dir']}")

    env_cfg = replace(experiment.env, max_episode_steps=2, angles=32, realism_enabled=False)
    out_dir = tmp_path / "tcv_train_out"
    ckpt_dir = tmp_path / "tcv_ckpt"
    run_metadata = {
        "experiment_name": experiment.name,
        "experiment_config_path": str(experiment.source_path),
        "trainer_name": "tcv_style",
        "sim": {
            "scenario_name": env_cfg.scenario_name,
            "scenario_args": dict(env_cfg.scenario_args),
            "angles": env_cfg.angles,
            "max_episode_steps": env_cfg.max_episode_steps,
        },
    }
    trainer_cfg = TCVStyleTrainerConfig(
        total_steps=4,
        warmup_steps=2,
        batch_size=1,
        sequence_length=2,
        replay_capacity_episodes=8,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        critic_mlp_hidden_dim=16,
        num_envs=1,
        updates_per_episode=1,
        eval_episodes=1,
        eval_max_steps=2,
        eval_interval_steps=2,
        seed=23,
        output_dir=out_dir,
        checkpoint_dir=ckpt_dir,
        run_metadata=run_metadata,
    )

    result = train_tcv_style_actor_critic(lambda: TokamakRLEnv(env_cfg), trainer_cfg)

    assert result.total_steps == 4
    assert result.replay_episodes >= 1
    assert result.replay_transitions == 4
    assert result.metrics_json is not None and result.metrics_json.exists()
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()
    assert result.critic_losses
    metrics = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert metrics["trainer"] == "tcv_style_recurrent_actor_critic_v1"
    assert metrics["run_metadata"]["experiment_name"] == "t15md_synthetic_joint_current_boundary"
    assert metrics["run_metadata"]["sim"]["scenario_name"] == "t15_synthetic_follow"
    assert "ip_template_dir" in metrics["run_metadata"]["sim"]["scenario_args"]
    assert metrics["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["reference_effective_seed_count"] >= 1
    assert metrics["eval_tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["eval_tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["eval_history"][0]["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert np.all(np.isfinite(result.eval_returns))


def test_stage_u_real_t15_smoke_trains_resumes_exports_and_rolls_out(tmp_path: Path) -> None:
    if not STAGE_U_SMOKE_EXPERIMENT.exists():
        pytest.skip(f"Missing smoke config: {STAGE_U_SMOKE_EXPERIMENT}")
    experiment = load_experiment_config(STAGE_U_SMOKE_EXPERIMENT)
    if not Path(experiment.env.sim_config_path).exists():
        pytest.skip(f"Missing simulator config: {experiment.env.sim_config_path}")
    if "ip_template_dir" in experiment.env.scenario_args and not Path(str(experiment.env.scenario_args["ip_template_dir"])).exists():
        pytest.skip(f"Missing Ip template dir: {experiment.env.scenario_args['ip_template_dir']}")

    out_dir = tmp_path / "stage_u_train"
    ckpt_dir = tmp_path / "stage_u_checkpoints"
    assert train_cli_main([
        "--config",
        str(STAGE_U_SMOKE_EXPERIMENT),
        "--output-dir",
        str(out_dir),
        "--checkpoint-dir",
        str(ckpt_dir),
    ]) == 0

    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    latest = Path(metrics["latest_checkpoint_path"])
    best = Path(metrics["best_checkpoint_path"])
    export_dir = Path(metrics["best_actor_export_dir"])
    assert metrics["total_steps"] == experiment.training.total_steps
    assert metrics["eval_returns"] and np.all(np.isfinite(metrics["eval_returns"]))
    assert metrics["config"]["eval_seed"] == experiment.evaluation.validation_seed
    assert metrics["run_metadata"]["evaluation_randomization_mode"] == "clean"
    assert metrics["run_metadata"]["evaluation"]["randomization_mode"] == "clean"
    assert latest.exists()
    assert best.exists()
    checkpoint = __import__("torch").load(best, map_location="cpu", weights_only=True)
    assert checkpoint["training_contract"]["observation_schema"]["schema_version"] == "v2"
    assert checkpoint["observation_schema"]["obs_dim"] == checkpoint["obs_dim"]
    assert checkpoint["action_schema"]["action_dim"] == checkpoint["action_dim"]
    assert checkpoint["normalization"]["derivative_scale"]
    assert (ckpt_dir / "step_00000004.pt").exists()
    assert (ckpt_dir / "step_00000008.pt").exists()
    assert export_dir.exists()
    assert (export_dir / "policy_weights.npz").exists()
    assert (export_dir / "schema.json").exists()
    assert (export_dir / "normalization.json").exists()
    assert (out_dir / "reference_samples.npz").exists()
    assert (out_dir / "termination_counts.json").exists()
    assert (out_dir / "rollouts" / "eval_step_00000008" / "summary.json").exists()
    assert (out_dir / "rollouts" / "eval_step_00000008" / "rollouts.npz").exists()
    with np.load(out_dir / "reference_samples.npz", allow_pickle=False) as samples:
        assert samples["reference_effective_seed"].shape[0] >= 1
        assert samples["reference_effective_ip_seed"].shape[0] >= 1
    termination_counts = json.loads((out_dir / "termination_counts.json").read_text(encoding="utf-8"))
    assert termination_counts["episodes"] >= 1

    np_actor = load_numpy_actor(export_dir)
    env = TokamakRLEnv(experiment.env, reward_fn=experiment.reward, randomizer=experiment.randomization)
    obs, _info = env.reset(seed=experiment.evaluation.validation_seed)
    try:
        action = np_actor.deterministic_action(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]
        next_obs, reward, _terminated, _truncated, step_info = env.step(action)
    finally:
        env.close()
    assert action.shape == (env.action_dim,)
    assert np.all(np.isfinite(action))
    assert np.all(np.isfinite(next_obs))
    assert np.isfinite(reward)
    assert "reward_components" in step_info

    resumed_out = tmp_path / "stage_u_resumed"
    resumed_ckpt = tmp_path / "stage_u_resumed_checkpoints"
    assert train_cli_main([
        "--config",
        str(STAGE_U_SMOKE_EXPERIMENT),
        "--steps",
        "4",
        "--warmup-steps",
        "0",
        "--resume-checkpoint",
        str(latest),
        "--output-dir",
        str(resumed_out),
        "--checkpoint-dir",
        str(resumed_ckpt),
    ]) == 0
    resumed_metrics = json.loads((resumed_out / "metrics.json").read_text(encoding="utf-8"))
    assert resumed_metrics["total_steps"] == 4
    assert resumed_metrics["latest_checkpoint_path"] is not None
    assert np.all(np.isfinite(resumed_metrics["eval_returns"]))
