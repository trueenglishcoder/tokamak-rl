from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tokamak_rl.training import ReplayBuffer, SimpleTrainerConfig, evaluate_actor, evaluate_actor_detailed, load_actor_from_checkpoint, train_simple_actor_critic


class ToyContinuousEnv:
    """Tiny deterministic continuous-control env for trainer mechanics tests."""

    obs_dim = 4
    action_dim = 2

    def __init__(self, *, max_steps: int = 8) -> None:
        self.max_steps = int(max_steps)
        self.step_index = 0
        self.state = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.step_index = 0
        self.state = rng.normal(0.0, 0.05, size=(self.obs_dim,)).astype(np.float32)
        seed_value = -1 if seed is None else int(seed)
        return self.state.copy(), {
            "seed": seed,
            "episode_metadata": {
                "reference_effective_seed": seed_value + 100,
                "reference_effective_ip_seed": seed_value + 200,
            },
        }

    def step(self, action):
        act = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.step_index += 1
        self.state[:2] = 0.85 * self.state[:2] + 0.15 * act
        self.state[2:] = act
        reward = -float(np.mean(self.state[:2] ** 2) + 0.01 * np.mean(act**2))
        truncated = self.step_index >= self.max_steps
        info = {
            "reward_components": {
                "ip_error_norm": float(abs(self.state[0])),
                "shape_error_norm": float(abs(self.state[1])),
            },
            "snapshot": SimpleNamespace(boundary_found=True),
        }
        return self.state.copy(), reward, False, truncated, info

    def close(self):
        return None


def test_replay_buffer_add_sample_and_validation() -> None:
    replay = ReplayBuffer(capacity=3, obs_dim=2, action_dim=1)
    replay.add(np.array([1.0, 2.0]), np.array([0.5]), 1.0, np.array([2.0, 3.0]), False, False)
    replay.add(np.array([2.0, 3.0]), np.array([-0.5]), 0.0, np.array([3.0, 4.0]), True, False)

    batch = replay.sample(2, rng=np.random.default_rng(1))

    assert replay.size == 2
    assert batch.observations.shape == (2, 2)
    assert batch.actions.shape == (2, 1)
    with pytest.raises(ValueError, match="not enough"):
        replay.sample(3, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="action shape"):
        replay.add(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, np.array([0.0, 0.0]), False, False)


def test_simple_actor_critic_trainer_smoke_checkpoint_and_eval(tmp_path: Path) -> None:
    cfg = SimpleTrainerConfig(
        total_steps=80,
        warmup_steps=10,
        batch_size=8,
        replay_capacity=128,
        hidden_dim=32,
        eval_episodes=2,
        eval_max_steps=8,
        seed=4,
        checkpoint_dir=tmp_path,
    )

    result = train_simple_actor_critic(lambda: ToyContinuousEnv(max_steps=8), cfg)

    assert result.total_steps == 80
    assert result.replay_size == 80
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.critic_losses
    assert result.actor_losses
    assert np.all(np.isfinite(result.critic_losses))
    assert np.all(np.isfinite(result.actor_losses))
    assert len(result.eval_returns) == 2
    assert np.all(np.isfinite(result.eval_returns))

    actor = load_actor_from_checkpoint(result.checkpoint_path)
    loaded_eval = evaluate_actor(lambda: ToyContinuousEnv(max_steps=8), actor, episodes=1, max_steps=8, seed=99)
    assert len(loaded_eval) == 1
    assert np.isfinite(loaded_eval[0])
    detailed_eval = evaluate_actor_detailed(lambda: ToyContinuousEnv(max_steps=8), actor, episodes=1, max_steps=8, seed=99)
    assert detailed_eval["returns"] == loaded_eval
    assert detailed_eval["tracking_diagnostics"]["mean_ip_error_norm"] is not None


def test_simple_actor_critic_trainer_vector_env_logs_and_resume(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_cfg = SimpleTrainerConfig(
        total_steps=96,
        warmup_steps=12,
        batch_size=8,
        replay_capacity=160,
        hidden_dim=32,
        num_envs=3,
        eval_interval_steps=48,
        checkpoint_interval_steps=48,
        eval_episodes=2,
        eval_max_steps=8,
        seed=8,
        output_dir=first_dir,
        checkpoint_dir=first_dir,
        run_metadata={"experiment_name": "toy_simple", "sim": {"scenario_name": "toy"}},
    )

    first = train_simple_actor_critic(lambda: ToyContinuousEnv(max_steps=8), first_cfg)

    assert first.total_steps == 96
    assert first.replay_size == 96
    assert first.checkpoint_path is not None
    assert first.latest_checkpoint_path is not None and first.latest_checkpoint_path.exists()
    assert first.best_checkpoint_path is not None and first.best_checkpoint_path.exists()
    assert (first_dir / "step_00000048.pt").exists()
    assert first.metrics_json is not None and first.metrics_json.exists()
    assert first.losses_csv is not None and first.losses_csv.exists()
    assert len(first.eval_history) == 2
    metrics = json.loads(first.metrics_json.read_text(encoding="utf-8"))
    assert metrics["num_envs"] == 3
    assert metrics["total_steps"] == 96
    assert metrics["critic_updates"] == len(first.critic_losses)
    assert np.isfinite(metrics["last_critic_loss"])
    assert metrics["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["tracking_diagnostics"]["boundary_failure_rate"] == 0.0
    assert metrics["tracking_diagnostics"]["reference_effective_seed_count"] >= 3
    assert metrics["run_metadata"]["experiment_name"] == "toy_simple"
    assert metrics["config"]["run_metadata"]["sim"]["scenario_name"] == "toy"
    assert metrics["checkpoint_path"] == str(first.checkpoint_path)
    assert metrics["latest_checkpoint_path"] == str(first.latest_checkpoint_path)
    assert metrics["best_checkpoint_path"] == str(first.best_checkpoint_path)
    assert metrics["device"]["requested"] == "cpu"
    assert metrics["device"]["actual"] == "cpu"
    assert metrics["throughput"]["env_steps_per_second"] > 0.0
    assert metrics["throughput"]["learner_updates_per_second"] > 0.0
    assert metrics["throughput"]["actor_inference_time_s"] >= 0.0
    assert metrics["throughput"]["env_step_time_s"] > 0.0
    assert metrics["throughput"]["replay_sampling_time_s"] >= 0.0
    assert metrics["throughput"]["update_to_data_ratio"] > 0.0
    assert metrics["eval_tracking_diagnostics"]["mean_ip_error_norm"] is not None
    assert metrics["eval_tracking_diagnostics"]["mean_shape_error_norm"] is not None
    assert metrics["eval_history"][0]["tracking_diagnostics"]["mean_ip_error_norm"] is not None
    reward_csv = first_dir / "reward_components.csv"
    reward_lines = reward_csv.read_text(encoding="utf-8").splitlines()
    assert reward_lines[0] == "step,env_index,episode,component,value"
    assert any("ip_error_norm" in line for line in reward_lines[1:])
    assert any("shape_error_norm" in line for line in reward_lines[1:])
    episodes_text = (first_dir / "episodes.csv").read_text(encoding="utf-8")
    assert "termination_reason" in episodes_text
    assert "reference_effective_seed" in episodes_text
    termination_counts = json.loads((first_dir / "termination_counts.json").read_text(encoding="utf-8"))
    assert termination_counts["episodes"] == len(first.episode_returns)
    with np.load(first_dir / "reference_samples.npz", allow_pickle=False) as samples:
        assert samples["reference_effective_seed"].shape[0] >= 3
        assert samples["reference_effective_ip_seed"].shape[0] >= 3
    rollout_dir = first_dir / "rollouts" / "eval_step_00000048"
    assert (rollout_dir / "summary.json").exists()
    assert (rollout_dir / "episode_metrics.csv").exists()
    assert (rollout_dir / "rollouts.npz").exists()

    checkpoint = __import__("torch").load(first.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["run_metadata"]["experiment_name"] == "toy_simple"
    assert checkpoint["total_steps"] == 96
    assert checkpoint["actor_target_state_dict"]
    assert checkpoint["q1_target_state_dict"]
    assert checkpoint["q2_target_state_dict"]
    assert checkpoint["actor_optimizer_state_dict"] is not None
    assert checkpoint["critic_optimizer_state_dict"] is not None
    assert checkpoint["numpy_rng_state"] is not None

    resumed_dir = tmp_path / "resumed"
    resumed_cfg = SimpleTrainerConfig(
        total_steps=40,
        warmup_steps=0,
        batch_size=8,
        replay_capacity=80,
        hidden_dim=32,
        num_envs=2,
        eval_episodes=1,
        eval_max_steps=8,
        seed=9,
        output_dir=resumed_dir,
        checkpoint_dir=resumed_dir,
        resume_checkpoint=first.checkpoint_path,
    )

    resumed = train_simple_actor_critic(lambda: ToyContinuousEnv(max_steps=8), resumed_cfg)

    assert resumed.total_steps == 40
    assert resumed.replay_size == 40
    assert resumed.checkpoint_path is not None and resumed.checkpoint_path.exists()
    assert resumed.metrics_json is not None and resumed.metrics_json.exists()
    assert resumed.critic_losses
    assert resumed.actor_losses
    assert np.all(np.isfinite(resumed.critic_losses))
    assert np.all(np.isfinite(resumed.actor_losses))


def test_simple_trainer_prunes_numbered_step_checkpoints(tmp_path: Path) -> None:
    cfg = SimpleTrainerConfig(
        total_steps=24,
        warmup_steps=4,
        batch_size=4,
        replay_capacity=64,
        hidden_dim=16,
        eval_episodes=1,
        eval_max_steps=4,
        checkpoint_interval_steps=8,
        max_step_checkpoints=2,
        checkpoint_dir=tmp_path,
        seed=41,
    )

    train_simple_actor_critic(lambda: ToyContinuousEnv(max_steps=4), cfg)

    assert not (tmp_path / "step_00000008.pt").exists()
    assert (tmp_path / "step_00000016.pt").exists()
    assert (tmp_path / "step_00000024.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_simple_trainer_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        SimpleTrainerConfig(total_steps=0)
    with pytest.raises(ValueError, match="gamma"):
        SimpleTrainerConfig(gamma=1.5)
    with pytest.raises(ValueError, match="num_envs"):
        SimpleTrainerConfig(num_envs=0)
    with pytest.raises(ValueError, match="max_step_checkpoints"):
        SimpleTrainerConfig(max_step_checkpoints=0)
