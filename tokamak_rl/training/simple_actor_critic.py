from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from tokamak_rl.contracts import TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.networks import ActorConfig, CriticConfig, FeedForwardActor, FeedForwardQCritic
from tokamak_rl.training.artifact_contract import RewardComponentWriter, eval_history_with_final, write_training_contract_artifacts
from tokamak_rl.training.diagnostics import (
    TrainingDiagnostics,
    episode_artifact_record,
    json_safe,
    record_episode_step_artifacts,
    reset_artifact_record,
    termination_reason_from_step_info,
)
from tokamak_rl.training.device import resolve_training_device
from tokamak_rl.training.export_artifacts import export_best_actor_artifact
from tokamak_rl.training.progress import TrainingProgressBar
from tokamak_rl.training.replay_buffer import ReplayBatch, ReplayBuffer
from tokamak_rl.training.wandb_logging import WandBConfig, WandBLogger


EnvFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class SimpleTrainerConfig:
    """Small deterministic actor-critic trainer for pipeline smoke tests."""

    total_steps: int = 500
    warmup_steps: int = 100
    batch_size: int = 64
    replay_capacity: int = 10000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    exploration_noise: float = 0.1
    policy_delay: int = 2
    hidden_dim: int = 256
    seed: int = 0
    eval_episodes: int = 1
    eval_max_steps: int = 200
    eval_seed: int | None = None
    num_envs: int = 1
    eval_interval_steps: int | None = None
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_name: str = "checkpoint.pt"
    latest_checkpoint_name: str = "latest.pt"
    best_checkpoint_name: str = "best.pt"
    checkpoint_interval_steps: int | None = None
    max_step_checkpoints: int | None = None
    resume_checkpoint: Path | None = None
    device: str = "cpu"
    progress: bool = False
    wandb: WandBConfig = field(default_factory=WandBConfig)
    export_best_actor: bool = True
    run_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("total_steps", "batch_size", "replay_capacity", "policy_delay", "hidden_dim", "eval_episodes", "eval_max_steps", "num_envs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.eval_interval_steps is not None and int(self.eval_interval_steps) <= 0:
            raise ValueError("eval_interval_steps must be > 0 when set")
        if self.checkpoint_interval_steps is not None and int(self.checkpoint_interval_steps) <= 0:
            raise ValueError("checkpoint_interval_steps must be > 0 when set")
        if self.max_step_checkpoints is not None and int(self.max_step_checkpoints) <= 0:
            raise ValueError("max_step_checkpoints must be > 0 when set")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if float(self.gamma) < 0.0 or float(self.gamma) > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if float(self.tau) <= 0.0 or float(self.tau) > 1.0:
            raise ValueError("tau must be in (0, 1]")
        if float(self.actor_lr) <= 0.0 or float(self.critic_lr) <= 0.0:
            raise ValueError("learning rates must be > 0")
        if float(self.exploration_noise) < 0.0:
            raise ValueError("exploration_noise must be >= 0")
        if str(self.device).strip().lower() not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    total_steps: int
    replay_size: int
    critic_losses: list[float]
    actor_losses: list[float]
    episode_returns: list[float]
    episode_lengths: list[int]
    eval_returns: list[float]
    eval_history: list[dict[str, object]]
    checkpoint_path: Path | None
    metrics_json: Path | None
    losses_csv: Path | None
    latest_checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    best_actor_export_dir: Path | None = None


def train_simple_actor_critic(env_factory: EnvFactory, cfg: SimpleTrainerConfig, *, eval_env_factory: EnvFactory | None = None) -> TrainingResult:
    """Run the first lightweight off-policy trainer smoke loop."""
    eval_factory = env_factory if eval_env_factory is None else eval_env_factory
    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device, device_selection = resolve_training_device(cfg.device)
    started_at = time.perf_counter()
    collection_time_s = 0.0
    actor_inference_time_s = 0.0
    env_step_time_s = 0.0
    replay_sampling_time_s = 0.0
    learner_time_s = 0.0
    evaluation_time_s = 0.0
    envs = [env_factory() for _ in range(int(cfg.num_envs))]
    diagnostics = TrainingDiagnostics()
    observations: list[np.ndarray] = []
    training_contract: dict[str, object] | None = None
    reset_metadata: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []
    for env_index, env in enumerate(envs):
        obs, reset_info = env.reset(seed=int(cfg.seed) + env_index)
        diagnostics.record_reset_info(reset_info)
        if training_contract is None:
            training_contract = _extract_training_contract(reset_info)
        reset_record = reset_artifact_record(reset_info, env_index=env_index, episode=0)
        reset_metadata.append(reset_record)
        reference_records.append(dict(reset_record))
        observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
    obs_dim = int(envs[0].obs_dim)
    action_dim = int(envs[0].action_dim)
    for env in envs:
        if int(env.obs_dim) != obs_dim or int(env.action_dim) != action_dim:
            raise ValueError("all vectorized environments must share obs_dim and action_dim")
    actor = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    actor_target = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    q1 = FeedForwardQCritic(CriticConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    q2 = FeedForwardQCritic(CriticConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    q1_target = FeedForwardQCritic(CriticConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    q2_target = FeedForwardQCritic(CriticConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.hidden_dim))).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(cfg.actor_lr))
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=float(cfg.critic_lr))
    if cfg.resume_checkpoint is not None:
        _load_training_state(
            cfg.resume_checkpoint,
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
    else:
        actor_target.load_state_dict(actor.state_dict())
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
    replay = ReplayBuffer(capacity=int(cfg.replay_capacity), obs_dim=obs_dim, action_dim=action_dim)
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    loss_rows: list[dict[str, object]] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    eval_history: list[dict[str, object]] = []
    running_returns = [0.0 for _ in envs]
    running_lengths = [0 for _ in envs]
    running_ip_errors: list[list[float]] = [[] for _ in envs]
    running_shape_errors: list[list[float]] = [[] for _ in envs]
    running_boundary_failure_steps = [0 for _ in envs]
    episode_indices = [0 for _ in envs]
    episode_records: list[dict[str, object]] = []
    update_index = 0
    step_count = 0
    best_eval_score = float("-inf")
    best_checkpoint_path: Path | None = None
    latest_checkpoint_path: Path | None = None
    reward_writer = RewardComponentWriter(cfg.output_dir)
    progress = TrainingProgressBar(total_steps=int(cfg.total_steps), label="simple", enabled=bool(cfg.progress))
    wandb_logger = WandBLogger(
        cfg.wandb,
        config=_checkpoint_safe_config(cfg),
        run_metadata={**json_safe(cfg.run_metadata), "device": device_selection.to_metadata(), "trainer": "simple_actor_critic_v1"},
    )
    progress.update(0, status=_simple_progress_status(replay_size=0, update_count=0), force=True)

    try:
        while step_count < int(cfg.total_steps):
            batch_count = min(len(envs), int(cfg.total_steps) - step_count)
            batch_indices = list(range(batch_count))
            batch_observations = np.stack([observations[index] for index in batch_indices], axis=0)
            batch_warmup = np.asarray([step_count + offset < int(cfg.warmup_steps) for offset in range(batch_count)], dtype=bool)
            collect_t0 = time.perf_counter()
            batch_actions = _select_actions_batch(
                actor,
                batch_observations,
                action_dim=action_dim,
                warmup=batch_warmup,
                rng=rng,
                noise=float(cfg.exploration_noise),
                device=device,
            )
            action_elapsed = time.perf_counter() - collect_t0
            actor_inference_time_s += action_elapsed
            collection_time_s += action_elapsed
            for local_index, env_index in enumerate(batch_indices):
                env = envs[env_index]
                obs = observations[env_index]
                action = batch_actions[local_index]
                collect_t0 = time.perf_counter()
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                step_elapsed = time.perf_counter() - collect_t0
                env_step_time_s += step_elapsed
                collection_time_s += step_elapsed
                diagnostics.record_step_info(step_info)
                reward_components = step_info.get("reward_components") if isinstance(step_info, dict) else None
                reward_writer.record(
                    step=step_count + 1,
                    env_index=env_index,
                    episode=episode_indices[env_index],
                    components=reward_components,
                )
                wandb_logger.log(
                    {
                        "train": {
                            "reward": float(reward),
                            "env_index": int(env_index),
                            "episode": int(episode_indices[env_index]),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        },
                        "reward_components": reward_components if isinstance(reward_components, dict) else {},
                    },
                    step=step_count + 1,
                )
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                replay.add(obs, action, float(reward), next_obs, bool(terminated), bool(truncated))
                running_returns[env_index] += float(reward)
                running_lengths[env_index] += 1
                record_episode_step_artifacts(
                    step_info,
                    ip_errors=running_ip_errors[env_index],
                    shape_errors=running_shape_errors[env_index],
                    boundary_failure_counter=running_boundary_failure_steps,
                    env_index=env_index,
                )
                observations[env_index] = next_obs
                step_count += 1
                if replay.size >= int(cfg.batch_size):
                    update_index += 1
                    sample_t0 = time.perf_counter()
                    batch = replay.sample(int(cfg.batch_size), rng=rng)
                    replay_sampling_time_s += time.perf_counter() - sample_t0
                    update_t0 = time.perf_counter()
                    losses = _update_once(
                        batch,
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        gamma=float(cfg.gamma),
                        tau=float(cfg.tau),
                        update_index=update_index,
                        policy_delay=int(cfg.policy_delay),
                        device=device,
                    )
                    learner_time_s += time.perf_counter() - update_t0
                    critic_losses.append(losses["critic_loss"])
                    if losses["actor_loss"] is not None:
                        actor_losses.append(losses["actor_loss"])
                    loss_rows.append({"step": step_count, "update": update_index, "critic_loss": losses["critic_loss"], "actor_loss": losses["actor_loss"]})
                if cfg.eval_interval_steps is not None and step_count % int(cfg.eval_interval_steps) == 0:
                    eval_t0 = time.perf_counter()
                    interval_eval = evaluate_actor_detailed(eval_factory, actor, episodes=int(cfg.eval_episodes), max_steps=int(cfg.eval_max_steps), seed=_eval_seed_base(cfg) + step_count, device=device)
                    evaluation_time_s += time.perf_counter() - eval_t0
                    interval_returns = interval_eval["returns"]
                    interval_mean = float(np.mean(interval_returns)) if interval_returns else 0.0
                    eval_history.append({"step": step_count, "returns": interval_returns, "mean_return": interval_mean, "tracking_diagnostics": interval_eval["tracking_diagnostics"]})
                    wandb_logger.log_eval({"mean_return": interval_mean, "tracking_diagnostics": interval_eval["tracking_diagnostics"]}, step=step_count)
                    if cfg.checkpoint_dir is not None and interval_mean > best_eval_score:
                        best_eval_score = interval_mean
                        best_checkpoint_path = save_training_checkpoint(
                            actor=actor,
                            actor_target=actor_target,
                            q1=q1,
                            q2=q2,
                            q1_target=q1_target,
                            q2_target=q2_target,
                            actor_opt=actor_opt,
                            critic_opt=critic_opt,
                            cfg=cfg,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            total_steps=step_count,
                            update_index=update_index,
                            best_eval_score=best_eval_score,
                            numpy_rng_state=rng.bit_generator.state,
                            training_contract=training_contract,
                            path=Path(cfg.checkpoint_dir) / cfg.best_checkpoint_name,
                        )
                if cfg.checkpoint_dir is not None and cfg.checkpoint_interval_steps is not None and step_count % int(cfg.checkpoint_interval_steps) == 0:
                    step_path = Path(cfg.checkpoint_dir) / f"step_{step_count:08d}.pt"
                    save_training_checkpoint(
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        total_steps=step_count,
                        update_index=update_index,
                        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
                        numpy_rng_state=rng.bit_generator.state,
                        training_contract=training_contract,
                        path=step_path,
                    )
                    _prune_step_checkpoints(Path(cfg.checkpoint_dir), keep=cfg.max_step_checkpoints)
                    latest_checkpoint_path = save_training_checkpoint(
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        total_steps=step_count,
                        update_index=update_index,
                        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
                        numpy_rng_state=rng.bit_generator.state,
                        training_contract=training_contract,
                        path=Path(cfg.checkpoint_dir) / cfg.latest_checkpoint_name,
                    )
                if bool(terminated) or bool(truncated):
                    episode_return = float(running_returns[env_index])
                    episode_length = int(running_lengths[env_index])
                    episode_returns.append(episode_return)
                    episode_lengths.append(int(running_lengths[env_index]))
                    episode_record = episode_artifact_record(
                        env_index=env_index,
                        episode=episode_indices[env_index],
                        episode_return=running_returns[env_index],
                        episode_length=running_lengths[env_index],
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        termination_reason=termination_reason_from_step_info(step_info, terminated=bool(terminated), truncated=bool(truncated)),
                        ip_errors=running_ip_errors[env_index],
                        shape_errors=running_shape_errors[env_index],
                        boundary_failure_steps=running_boundary_failure_steps[env_index],
                        reset_record=reset_metadata[env_index],
                    )
                    episode_records.append(episode_record)
                    wandb_logger.log_episode({**episode_record, "return": episode_return, "length": episode_length}, step=step_count)
                    episode_indices[env_index] += 1
                    reset_obs, reset_info = env.reset(seed=int(cfg.seed) + 10_000 + step_count + env_index)
                    diagnostics.record_reset_info(reset_info)
                    if training_contract is None:
                        training_contract = _extract_training_contract(reset_info)
                    reset_metadata[env_index] = reset_artifact_record(reset_info, env_index=env_index, episode=episode_indices[env_index])
                    reference_records.append(dict(reset_metadata[env_index]))
                    observations[env_index] = np.asarray(reset_obs, dtype=np.float32).reshape(-1)
                    running_returns[env_index] = 0.0
                    running_lengths[env_index] = 0
                    running_ip_errors[env_index].clear()
                    running_shape_errors[env_index].clear()
                    running_boundary_failure_steps[env_index] = 0
                progress.update(
                    step_count,
                    status=_simple_progress_status(
                        replay_size=replay.size,
                        update_count=update_index,
                        critic_loss=critic_losses[-1] if critic_losses else None,
                        actor_loss=actor_losses[-1] if actor_losses else None,
                        episodes=len(episode_returns),
                    ),
                )
                wandb_logger.log(
                    {
                        "train": _simple_progress_status(
                            replay_size=replay.size,
                            update_count=update_index,
                            critic_loss=critic_losses[-1] if critic_losses else None,
                            actor_loss=actor_losses[-1] if actor_losses else None,
                            episodes=len(episode_returns),
                        )
                    },
                    step=step_count,
                )
        for env_index, length in enumerate(running_lengths):
            if int(length) > 0:
                episode_returns.append(float(running_returns[env_index]))
                episode_lengths.append(int(length))
                episode_records.append(
                    episode_artifact_record(
                        env_index=env_index,
                        episode=episode_indices[env_index],
                        episode_return=running_returns[env_index],
                        episode_length=running_lengths[env_index],
                        terminated=False,
                        truncated=True,
                        termination_reason="training_horizon",
                        ip_errors=running_ip_errors[env_index],
                        shape_errors=running_shape_errors[env_index],
                        boundary_failure_steps=running_boundary_failure_steps[env_index],
                        reset_record=reset_metadata[env_index],
                    )
                )
    finally:
        progress.close(
            status=_simple_progress_status(
                replay_size=replay.size,
                update_count=update_index,
                critic_loss=critic_losses[-1] if critic_losses else None,
                actor_loss=actor_losses[-1] if actor_losses else None,
                episodes=len(episode_returns),
            )
        )
        reward_writer.close()
        for env in envs:
            env.close()

    eval_t0 = time.perf_counter()
    final_eval = evaluate_actor_detailed(eval_factory, actor, episodes=int(cfg.eval_episodes), max_steps=int(cfg.eval_max_steps), seed=_eval_seed_base(cfg), device=device)
    evaluation_time_s += time.perf_counter() - eval_t0
    eval_returns = final_eval["returns"]
    final_mean = float(np.mean(eval_returns)) if eval_returns else 0.0
    wandb_logger.log_eval({"mean_return": final_mean, "tracking_diagnostics": final_eval["tracking_diagnostics"]}, step=step_count)
    if cfg.checkpoint_dir is not None and final_mean > best_eval_score:
        best_eval_score = final_mean
        best_checkpoint_path = save_training_checkpoint(
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            total_steps=step_count,
            update_index=update_index,
            best_eval_score=best_eval_score,
            numpy_rng_state=rng.bit_generator.state,
            training_contract=training_contract,
            path=Path(cfg.checkpoint_dir) / cfg.best_checkpoint_name,
        )
    checkpoint_path = save_training_checkpoint(
        actor=actor,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        cfg=cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_steps=step_count,
        update_index=update_index,
        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
        numpy_rng_state=rng.bit_generator.state,
        training_contract=training_contract,
    ) if cfg.checkpoint_dir is not None else None
    if cfg.checkpoint_dir is not None:
        latest_checkpoint_path = save_training_checkpoint(
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            total_steps=step_count,
            update_index=update_index,
            best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
            numpy_rng_state=rng.bit_generator.state,
            training_contract=training_contract,
            path=Path(cfg.checkpoint_dir) / cfg.latest_checkpoint_name,
        )
    best_actor_export_dir = export_best_actor_artifact(
        checkpoint_path=best_checkpoint_path,
        output_dir=cfg.output_dir,
        training_contract=training_contract,
        metadata={"trainer": "simple_actor_critic_v1", "run_metadata": cfg.run_metadata},
    ) if bool(cfg.export_best_actor) else None
    metrics_json, losses_csv = _write_training_artifacts(
        cfg=cfg,
        total_steps=step_count,
        replay_size=replay.size,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        loss_rows=loss_rows,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        eval_returns=eval_returns,
        eval_history=eval_history,
        episode_records=episode_records,
        reference_records=reference_records,
        tracking_diagnostics=diagnostics.summary(),
        eval_tracking_diagnostics=final_eval["tracking_diagnostics"],
        checkpoint_path=checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_actor_export_dir=best_actor_export_dir,
        throughput=_throughput_metrics(
            total_steps=step_count,
            update_count=update_index,
            total_elapsed_s=time.perf_counter() - started_at,
            collection_time_s=collection_time_s,
            actor_inference_time_s=actor_inference_time_s,
            env_step_time_s=env_step_time_s,
            replay_sampling_time_s=replay_sampling_time_s,
            learner_time_s=learner_time_s,
            evaluation_time_s=evaluation_time_s,
        ),
        device_metadata=device_selection.to_metadata(),
    )
    wandb_logger.log_final(
        {
            "total_steps": int(step_count),
            "replay_size": int(replay.size),
            "critic_updates": len(critic_losses),
            "actor_updates": len(actor_losses),
            "last_critic_loss": float(critic_losses[-1]) if critic_losses else None,
            "last_actor_loss": float(actor_losses[-1]) if actor_losses else None,
            "eval_mean_return": final_mean,
            "tracking_diagnostics": diagnostics.summary(),
            "eval_tracking_diagnostics": final_eval["tracking_diagnostics"],
        },
        artifact_paths={
            "metrics": metrics_json,
            "losses": losses_csv,
            "checkpoint": checkpoint_path,
            "latest_checkpoint": latest_checkpoint_path,
            "best_checkpoint": best_checkpoint_path,
            "best_actor_export": best_actor_export_dir,
        },
        step=step_count,
    )
    wandb_logger.close()
    return TrainingResult(
        total_steps=int(step_count),
        replay_size=replay.size,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        eval_returns=eval_returns,
        eval_history=eval_history,
        checkpoint_path=checkpoint_path,
        metrics_json=metrics_json,
        losses_csv=losses_csv,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_actor_export_dir=best_actor_export_dir,
    )


def evaluate_actor(env_factory: EnvFactory, actor: FeedForwardActor, *, episodes: int, max_steps: int, seed: int, device: torch.device | str = "cpu") -> list[float]:
    return list(evaluate_actor_detailed(env_factory, actor, episodes=episodes, max_steps=max_steps, seed=seed, device=device)["returns"])


def evaluate_actor_detailed(env_factory: EnvFactory, actor: FeedForwardActor, *, episodes: int, max_steps: int, seed: int, device: torch.device | str = "cpu") -> dict[str, object]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be > 0")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be > 0")
    device = torch.device(device)
    returns: list[float] = []
    diagnostics = TrainingDiagnostics()
    was_training = actor.training
    actor.eval()
    try:
        for episode_index in range(int(episodes)):
            env = env_factory()
            obs, reset_info = env.reset(seed=int(seed) + episode_index)
            diagnostics.record_reset_info(reset_info)
            total = 0.0
            try:
                for _ in range(int(max_steps)):
                    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
                    with torch.no_grad():
                        action = actor.deterministic_action(obs_t).detach().cpu().numpy()[0]
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    diagnostics.record_step_info(step_info)
                    total += float(reward)
                    if bool(terminated) or bool(truncated):
                        break
            finally:
                env.close()
            returns.append(float(total))
    finally:
        actor.train(was_training)
    return {"returns": returns, "tracking_diagnostics": diagnostics.summary()}


def _simple_progress_status(
    *,
    replay_size: int,
    update_count: int,
    critic_loss: float | None = None,
    actor_loss: float | None = None,
    episodes: int | None = None,
) -> dict[str, object]:
    return {
        "episodes": episodes,
        "replay": int(replay_size),
        "updates": int(update_count),
        "critic": critic_loss,
        "actor": actor_loss,
    }


def save_training_checkpoint(
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor | None = None,
    q1: FeedForwardQCritic,
    q2: FeedForwardQCritic,
    q1_target: FeedForwardQCritic | None = None,
    q2_target: FeedForwardQCritic | None = None,
    actor_opt: torch.optim.Optimizer | None = None,
    critic_opt: torch.optim.Optimizer | None = None,
    cfg: SimpleTrainerConfig,
    obs_dim: int,
    action_dim: int,
    total_steps: int = 0,
    update_index: int = 0,
    best_eval_score: float | None = None,
    numpy_rng_state: dict[str, object] | None = None,
    training_contract: dict[str, object] | None = None,
    path: Path | None = None,
) -> Path:
    if cfg.checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required")
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / cfg.checkpoint_name if path is None else Path(path)
    torch.save(
        {
            "trainer": "simple_actor_critic_v1",
            "config": _checkpoint_safe_config(cfg),
            "run_metadata": json_safe(cfg.run_metadata),
            "training_contract": json_safe(training_contract),
            "observation_schema": None if training_contract is None else json_safe(training_contract.get("observation_schema")),
            "action_schema": None if training_contract is None else json_safe(training_contract.get("action_schema")),
            "normalization": None if training_contract is None else json_safe(training_contract.get("normalization")),
            "total_steps": int(total_steps),
            "update_index": int(update_index),
            "best_eval_score": None if best_eval_score is None else float(best_eval_score),
            "actor_config": asdict(actor.cfg),
            "critic_config": asdict(q1.cfg),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "actor_state_dict": actor.state_dict(),
            "actor_target_state_dict": (actor_target or actor).state_dict(),
            "q1_state_dict": q1.state_dict(),
            "q2_state_dict": q2.state_dict(),
            "q1_target_state_dict": (q1_target or q1).state_dict(),
            "q2_target_state_dict": (q2_target or q2).state_dict(),
            "actor_optimizer_state_dict": None if actor_opt is None else actor_opt.state_dict(),
            "critic_optimizer_state_dict": None if critic_opt is None else critic_opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": json_safe(numpy_rng_state),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_actor_from_checkpoint(path: str | Path, *, device: torch.device | str = "cpu") -> FeedForwardActor:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "actor_config" not in checkpoint or "actor_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain an exported training actor")
    actor = FeedForwardActor(ActorConfig(**checkpoint["actor_config"]))
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(torch.device(device))
    actor.eval()
    return actor


def _load_training_state(
    path: str | Path,
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: FeedForwardQCritic,
    q2: FeedForwardQCritic,
    q1_target: FeedForwardQCritic,
    q2_target: FeedForwardQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> None:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must be a mapping")
    if int(checkpoint.get("obs_dim", -1)) != int(obs_dim) or int(checkpoint.get("action_dim", -1)) != int(action_dim):
        raise ValueError("resume checkpoint dimensions do not match current environment")
    actor.load_state_dict(checkpoint["actor_state_dict"])
    q1.load_state_dict(checkpoint["q1_state_dict"])
    q2.load_state_dict(checkpoint["q2_state_dict"])
    actor_target.load_state_dict(checkpoint.get("actor_target_state_dict", checkpoint["actor_state_dict"]))
    q1_target.load_state_dict(checkpoint.get("q1_target_state_dict", checkpoint["q1_state_dict"]))
    q2_target.load_state_dict(checkpoint.get("q2_target_state_dict", checkpoint["q2_state_dict"]))
    actor_opt_state = checkpoint.get("actor_optimizer_state_dict")
    critic_opt_state = checkpoint.get("critic_optimizer_state_dict")
    if actor_opt_state is not None:
        actor_opt.load_state_dict(actor_opt_state)
    if critic_opt_state is not None:
        critic_opt.load_state_dict(critic_opt_state)


def _select_action(actor: FeedForwardActor, observation: np.ndarray, *, action_dim: int, warmup: bool, rng: np.random.Generator, noise: float, device: torch.device) -> np.ndarray:
    if warmup:
        return rng.uniform(-1.0, 1.0, size=(int(action_dim),)).astype(np.float32)
    obs_t = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        action = actor.deterministic_action(obs_t).detach().cpu().numpy()[0]
    if float(noise) > 0.0:
        action = action + rng.normal(0.0, float(noise), size=action.shape)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _select_actions_batch(
    actor: FeedForwardActor,
    observations: np.ndarray,
    *,
    action_dim: int,
    warmup: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    device: torch.device,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError("observations must have shape (batch, obs_dim)")
    actions = np.zeros((obs.shape[0], int(action_dim)), dtype=np.float32)
    warmup_mask = np.asarray(warmup, dtype=bool).reshape(-1)
    if warmup_mask.shape != (obs.shape[0],):
        raise ValueError("warmup mask must have shape (batch,)")
    if np.any(~warmup_mask):
        obs_t = torch.as_tensor(obs[~warmup_mask], dtype=torch.float32, device=device)
        with torch.no_grad():
            actions[~warmup_mask] = actor.deterministic_action(obs_t).detach().cpu().numpy().astype(np.float32, copy=False)
        if float(noise) > 0.0:
            actions[~warmup_mask] = actions[~warmup_mask] + rng.normal(0.0, float(noise), size=actions[~warmup_mask].shape).astype(np.float32)
    if np.any(warmup_mask):
        actions[warmup_mask] = rng.uniform(-1.0, 1.0, size=(int(np.count_nonzero(warmup_mask)), int(action_dim))).astype(np.float32)
    return np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)


def _checkpoint_safe_config(cfg: SimpleTrainerConfig) -> dict[str, object]:
    data = asdict(cfg)
    for key in ("checkpoint_dir", "output_dir", "resume_checkpoint"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    data["run_metadata"] = json_safe(data.get("run_metadata", {}))
    return json_safe(data)


def _extract_training_contract(reset_info: object) -> dict[str, object] | None:
    if not isinstance(reset_info, dict):
        return None
    episode_metadata = reset_info.get("episode_metadata")
    if not isinstance(episode_metadata, dict):
        return None
    contract = episode_metadata.get("training_contract")
    return dict(contract) if isinstance(contract, dict) else None


def _eval_seed_base(cfg: SimpleTrainerConfig) -> int:
    return int(cfg.seed) + 1000 if cfg.eval_seed is None else int(cfg.eval_seed)


def _prune_step_checkpoints(checkpoint_dir: Path, *, keep: int | None) -> None:
    if keep is None:
        return
    paths = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    excess = len(paths) - int(keep)
    if excess <= 0:
        return
    for path in paths[:excess]:
        path.unlink()


def _write_training_artifacts(
    *,
    cfg: SimpleTrainerConfig,
    total_steps: int,
    replay_size: int,
    critic_losses: list[float],
    actor_losses: list[float],
    loss_rows: list[dict[str, object]],
    episode_returns: list[float],
    episode_lengths: list[int],
    eval_returns: list[float],
    eval_history: list[dict[str, object]],
    episode_records: list[dict[str, object]],
    reference_records: list[dict[str, object]],
    tracking_diagnostics: dict[str, object],
    eval_tracking_diagnostics: dict[str, object],
    checkpoint_path: Path | None,
    latest_checkpoint_path: Path | None,
    best_checkpoint_path: Path | None,
    best_actor_export_dir: Path | None,
    throughput: dict[str, object],
    device_metadata: dict[str, object],
) -> tuple[Path | None, Path | None]:
    if cfg.output_dir is None:
        return None, None
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    losses_csv = output_dir / "losses.csv"
    with losses_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "update", "critic_loss", "actor_loss"])
        writer.writeheader()
        for row in loss_rows:
            writer.writerow({"step": row["step"], "update": row["update"], "critic_loss": row["critic_loss"], "actor_loss": "" if row["actor_loss"] is None else row["actor_loss"]})
    metrics_json = output_dir / "metrics.json"
    metrics = {
        "contract_version": TRAINING_READINESS_CONTRACT_VERSION,
        "trainer": "simple_actor_critic_v1",
        "total_steps": int(total_steps),
        "num_envs": int(cfg.num_envs),
        "replay_size": int(replay_size),
        "critic_updates": len(critic_losses),
        "actor_updates": len(actor_losses),
        "last_critic_loss": float(critic_losses[-1]) if critic_losses else None,
        "last_actor_loss": float(actor_losses[-1]) if actor_losses else None,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "eval_returns": eval_returns,
        "eval_history": eval_history,
        "tracking_diagnostics": tracking_diagnostics,
        "eval_tracking_diagnostics": eval_tracking_diagnostics,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "latest_checkpoint_path": None if latest_checkpoint_path is None else str(latest_checkpoint_path),
        "best_checkpoint_path": None if best_checkpoint_path is None else str(best_checkpoint_path),
        "best_actor_export_dir": None if best_actor_export_dir is None else str(best_actor_export_dir),
        "device": device_metadata,
        "throughput": throughput,
        "simulator_profiling": _simulator_profiling_snapshot(),
        "config": _checkpoint_safe_config(cfg),
        "run_metadata": json_safe(cfg.run_metadata),
    }
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    config_json = output_dir / "config_snapshot.json"
    config_json.write_text(json.dumps(_checkpoint_safe_config(cfg), indent=2, sort_keys=True), encoding="utf-8")
    write_training_contract_artifacts(
        output_dir=output_dir,
        trainer_name="simple_actor_critic_v1",
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        eval_history=eval_history_with_final(
            eval_history,
            total_steps=total_steps,
            eval_returns=eval_returns,
            eval_tracking_diagnostics=eval_tracking_diagnostics,
        ),
        episode_records=episode_records,
        reference_records=reference_records,
        best_actor_export_dir=best_actor_export_dir,
    )
    return metrics_json, losses_csv


def _simulator_profiling_snapshot() -> dict[str, object]:
    try:
        from tokamak_control.core.plasma_model import plasma_model_profiling_snapshot
        from tokamak_control.core.gpu_plasma_model import gpu_plasma_model_profiling_snapshot
        from tokamak_control.geometry.boundary import boundary_profiling_snapshot
    except Exception:
        return {"available": False}
    return {
        "available": True,
        "plasma_model": plasma_model_profiling_snapshot(),
        "gpu_plasma_model": gpu_plasma_model_profiling_snapshot(),
        "boundary": boundary_profiling_snapshot(),
    }


def _throughput_metrics(
    *,
    total_steps: int,
    update_count: int,
    total_elapsed_s: float,
    collection_time_s: float,
    actor_inference_time_s: float,
    env_step_time_s: float,
    replay_sampling_time_s: float,
    learner_time_s: float,
    evaluation_time_s: float,
) -> dict[str, object]:
    elapsed = max(float(total_elapsed_s), 1.0e-12)
    collection = max(float(collection_time_s), 1.0e-12)
    env_step = max(float(env_step_time_s), 1.0e-12)
    learner = max(float(learner_time_s), 1.0e-12)
    overall_steps_per_second = float(total_steps) / elapsed
    return {
        "total_elapsed_s": float(total_elapsed_s),
        "collection_time_s": float(collection_time_s),
        "actor_inference_time_s": float(actor_inference_time_s),
        "env_step_time_s": float(env_step_time_s),
        "replay_sampling_time_s": float(replay_sampling_time_s),
        "learner_time_s": float(learner_time_s),
        "evaluation_time_s": float(evaluation_time_s),
        "env_steps_per_second": overall_steps_per_second,
        "overall_steps_per_second": overall_steps_per_second,
        "collection_steps_per_second": float(total_steps) / collection,
        "env_step_only_steps_per_second": float(total_steps) / env_step,
        "learner_updates_per_second": float(update_count) / learner if int(update_count) > 0 else 0.0,
        "update_to_data_ratio": float(update_count) / max(float(total_steps), 1.0),
    }


def _update_once(
    batch: ReplayBatch,
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: FeedForwardQCritic,
    q2: FeedForwardQCritic,
    q1_target: FeedForwardQCritic,
    q2_target: FeedForwardQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    gamma: float,
    tau: float,
    update_index: int,
    policy_delay: int,
    device: torch.device,
) -> dict[str, float | None]:
    obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=device)
    terminated = torch.as_tensor(batch.terminated.astype(np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        next_actions = actor_target.deterministic_action(next_obs)
        target_q = torch.minimum(q1_target(next_obs, next_actions), q2_target(next_obs, next_actions))
        target = rewards + float(gamma) * (1.0 - terminated) * target_q
    q1_pred = q1(obs, actions)
    q2_pred = q2(obs, actions)
    critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()
    actor_loss_value: float | None = None
    if int(update_index) % int(policy_delay) == 0:
        actor_loss = -q1(obs, actor.deterministic_action(obs)).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        actor_loss_value = float(actor_loss.detach().cpu().item())
    _soft_update(q1_target, q1, tau=float(tau))
    _soft_update(q2_target, q2, tau=float(tau))
    _soft_update(actor_target, actor, tau=float(tau))
    critic_loss_value = float(critic_loss.detach().cpu().item())
    if not np.isfinite(critic_loss_value) or (actor_loss_value is not None and not np.isfinite(actor_loss_value)):
        raise RuntimeError("trainer produced non-finite loss")
    return {"critic_loss": critic_loss_value, "actor_loss": actor_loss_value}


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - float(tau)).add_(source_param, alpha=float(tau))


__all__ = [
    "SimpleTrainerConfig",
    "TrainingResult",
    "evaluate_actor",
    "evaluate_actor_detailed",
    "load_actor_from_checkpoint",
    "save_training_checkpoint",
    "train_simple_actor_critic",
]
