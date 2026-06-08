from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from tokamak_rl.contracts import TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.networks import ActorConfig, FeedForwardActor, RecurrentCriticConfig, RecurrentQCritic
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
from tokamak_rl.training.recurrent_critic import RecurrentUpdateResult, recurrent_critic_update_once
from tokamak_rl.training.sequence_replay import EpisodeReplayBuffer
from tokamak_rl.training.wandb_logging import WandBConfig, WandBLogger


EnvFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class TCVStyleTrainerConfig:
    """TCV-style trainer with feedforward actor, recurrent critics, and MPO updates."""

    total_steps: int = 1000
    warmup_steps: int = 100
    batch_size: int = 16
    sequence_length: int = 64
    replay_capacity_episodes: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    mpo_kl_lr: float = 3.0e-4
    mpo_epsilon: float = 0.1
    mpo_mean_kl_epsilon: float = 0.01
    mpo_std_kl_epsilon: float = 1.0e-4
    mpo_action_samples: int = 20
    mpo_temperature_iterations: int = 10
    mpo_temperature_lr: float = 0.1
    mpo_initial_temperature: float = 1.0
    mpo_initial_mean_kl_penalty: float = 1.0
    mpo_initial_std_kl_penalty: float = 1.0
    policy_delay: int = 2
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    critic_mlp_hidden_dim: int = 256
    num_envs: int = 1
    updates_per_episode: int = 1
    updates_per_env_step: int = 0
    max_learner_catchup_updates: int | None = None
    seed: int = 0
    eval_episodes: int = 1
    eval_max_steps: int = 200
    eval_seed: int | None = None
    eval_interval_steps: int | None = None
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_name: str = "tcv_style_checkpoint.pt"
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
        for name in (
            "total_steps",
            "batch_size",
            "sequence_length",
            "replay_capacity_episodes",
            "policy_delay",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "critic_mlp_hidden_dim",
            "mpo_action_samples",
            "mpo_temperature_iterations",
            "num_envs",
            "updates_per_episode",
            "eval_episodes",
            "eval_max_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if int(self.updates_per_env_step) < 0:
            raise ValueError("updates_per_env_step must be >= 0")
        if self.max_learner_catchup_updates is not None and int(self.max_learner_catchup_updates) <= 0:
            raise ValueError("max_learner_catchup_updates must be > 0 when set")
        if self.eval_interval_steps is not None and int(self.eval_interval_steps) <= 0:
            raise ValueError("eval_interval_steps must be > 0 when set")
        if self.checkpoint_interval_steps is not None and int(self.checkpoint_interval_steps) <= 0:
            raise ValueError("checkpoint_interval_steps must be > 0 when set")
        if self.max_step_checkpoints is not None and int(self.max_step_checkpoints) <= 0:
            raise ValueError("max_step_checkpoints must be > 0 when set")
        if float(self.gamma) < 0.0 or float(self.gamma) > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if float(self.tau) <= 0.0 or float(self.tau) > 1.0:
            raise ValueError("tau must be in (0, 1]")
        if float(self.actor_lr) <= 0.0 or float(self.critic_lr) <= 0.0 or float(self.mpo_kl_lr) <= 0.0:
            raise ValueError("learning rates must be > 0")
        for name in (
            "mpo_epsilon",
            "mpo_mean_kl_epsilon",
            "mpo_std_kl_epsilon",
            "mpo_temperature_lr",
            "mpo_initial_temperature",
            "mpo_initial_mean_kl_penalty",
            "mpo_initial_std_kl_penalty",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if str(self.device).strip().lower() not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")


@dataclass(frozen=True, slots=True)
class TCVStyleTrainingResult:
    total_steps: int
    replay_episodes: int
    replay_transitions: int
    critic_losses: list[float]
    actor_losses: list[float]
    mpo_temperature_losses: list[float]
    mpo_temperatures: list[float]
    mpo_mean_kls: list[float]
    mpo_std_kls: list[float]
    mpo_kl_dual_losses: list[float]
    mpo_mean_kl_penalties: list[float]
    mpo_std_kl_penalties: list[float]
    valid_steps_per_update: list[int]
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


def train_tcv_style_actor_critic(env_factory: EnvFactory, cfg: TCVStyleTrainerConfig, *, eval_env_factory: EnvFactory | None = None) -> TCVStyleTrainingResult:
    if bool(getattr(env_factory, "is_true_batched_gpu_factory", False)):
        return _train_tcv_style_actor_critic_true_batched_gpu(env_factory, cfg, eval_env_factory=eval_env_factory)
    eval_factory = env_factory if eval_env_factory is None else eval_env_factory
    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch_generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    device, device_selection = resolve_training_device(cfg.device)
    started_at = time.perf_counter()
    collection_time_s = 0.0
    actor_inference_time_s = 0.0
    env_step_time_s = 0.0
    timing = {"replay_sampling_time_s": 0.0}
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
            raise ValueError("all training environments must share obs_dim and action_dim")

    actor = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.actor_hidden_dim))).to(device)
    actor_target = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.actor_hidden_dim))).to(device)
    critic_cfg = RecurrentCriticConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg.critic_hidden_dim),
        mlp_hidden_dim=int(cfg.critic_mlp_hidden_dim),
    )
    q1 = RecurrentQCritic(critic_cfg).to(device)
    q2 = RecurrentQCritic(critic_cfg).to(device)
    q1_target = RecurrentQCritic(critic_cfg).to(device)
    q2_target = RecurrentQCritic(critic_cfg).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(cfg.actor_lr))
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=float(cfg.critic_lr))
    log_mpo_mean_kl_penalty = torch.tensor(
        _inverse_softplus(float(cfg.mpo_initial_mean_kl_penalty)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    log_mpo_std_kl_penalty = torch.tensor(
        _inverse_softplus(float(cfg.mpo_initial_std_kl_penalty)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    mpo_kl_opt = torch.optim.Adam([log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty], lr=float(cfg.mpo_kl_lr))
    if cfg.resume_checkpoint is not None:
        _load_tcv_training_state(
            cfg.resume_checkpoint,
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
            mpo_kl_opt=mpo_kl_opt,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            torch_generator=torch_generator,
        )
    else:
        actor_target.load_state_dict(actor.state_dict())
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
    replay = EpisodeReplayBuffer(capacity_episodes=int(cfg.replay_capacity_episodes), obs_dim=obs_dim, action_dim=action_dim)
    episode_builders = [_EpisodeBuilder() for _ in envs]
    running_returns = [0.0 for _ in envs]
    running_lengths = [0 for _ in envs]
    running_ip_errors: list[list[float]] = [[] for _ in envs]
    running_shape_errors: list[list[float]] = [[] for _ in envs]
    running_boundary_failure_steps = [0 for _ in envs]
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_indices = [0 for _ in envs]
    episode_records: list[dict[str, object]] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    mpo_temperature_losses: list[float] = []
    mpo_temperatures: list[float] = []
    mpo_mean_kls: list[float] = []
    mpo_std_kls: list[float] = []
    mpo_kl_dual_losses: list[float] = []
    mpo_mean_kl_penalties: list[float] = []
    mpo_std_kl_penalties: list[float] = []
    valid_steps_per_update: list[int] = []
    loss_rows: list[dict[str, object]] = []
    eval_history: list[dict[str, object]] = []
    step_count = 0
    update_index = 0
    best_eval_score = float("-inf")
    best_checkpoint_path: Path | None = None
    latest_checkpoint_path: Path | None = None
    reward_writer = RewardComponentWriter(cfg.output_dir)
    progress = TrainingProgressBar(total_steps=int(cfg.total_steps), label="tcv_style", enabled=bool(cfg.progress))
    wandb_logger = WandBLogger(
        cfg.wandb,
        config=_checkpoint_safe_config(cfg),
        run_metadata={**json_safe(cfg.run_metadata), "device": device_selection.to_metadata(), "trainer": "tcv_style_recurrent_actor_critic_v1"},
    )
    progress.update(0, status=_tcv_progress_status(replay_episodes=0, replay_transitions=0, update_count=0), force=True)

    try:
        while step_count < int(cfg.total_steps):
            batch_count = min(len(envs), int(cfg.total_steps) - step_count)
            batch_indices = list(range(batch_count))
            batch_observations = np.stack([observations[index] for index in batch_indices], axis=0)
            batch_warmup = np.asarray([step_count + offset < int(cfg.warmup_steps) for offset in range(batch_count)], dtype=bool)
            collect_t0 = time.perf_counter()
            batch_actions = _select_training_actions_batch(
                actor,
                batch_observations,
                action_dim=action_dim,
                warmup=batch_warmup,
                rng=rng,
                torch_generator=torch_generator,
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
                episode_builders[env_index].add(obs, action, float(reward), next_obs, bool(terminated), bool(truncated))
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

                if bool(terminated) or bool(truncated):
                    replay.add_episode(**episode_builders[env_index].to_episode_arrays())
                    episode_builders[env_index].clear()
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
                    running_returns[env_index] = 0.0
                    running_lengths[env_index] = 0
                    running_ip_errors[env_index].clear()
                    running_shape_errors[env_index].clear()
                    running_boundary_failure_steps[env_index] = 0
                    update_t0 = time.perf_counter()
                    update_index = _run_recurrent_updates(
                        replay,
                        rng=rng,
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                        log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                        mpo_kl_opt=mpo_kl_opt,
                        torch_generator=torch_generator,
                        cfg=cfg,
                        updates=_bounded_update_count(int(cfg.updates_per_episode), cfg),
                        update_index=update_index,
                        step_count=step_count,
                        device=device,
                        critic_losses=critic_losses,
                        actor_losses=actor_losses,
                        mpo_temperature_losses=mpo_temperature_losses,
                        mpo_temperatures=mpo_temperatures,
                        mpo_mean_kls=mpo_mean_kls,
                        mpo_std_kls=mpo_std_kls,
                        mpo_kl_dual_losses=mpo_kl_dual_losses,
                        mpo_mean_kl_penalties=mpo_mean_kl_penalties,
                        mpo_std_kl_penalties=mpo_std_kl_penalties,
                        valid_steps_per_update=valid_steps_per_update,
                        loss_rows=loss_rows,
                        timing=timing,
                    )
                    learner_time_s += time.perf_counter() - update_t0
                    reset_obs, reset_info = env.reset(seed=int(cfg.seed) + 20_000 + step_count + env_index)
                    diagnostics.record_reset_info(reset_info)
                    if training_contract is None:
                        training_contract = _extract_training_contract(reset_info)
                    reset_metadata[env_index] = reset_artifact_record(reset_info, env_index=env_index, episode=episode_indices[env_index])
                    reference_records.append(dict(reset_metadata[env_index]))
                    observations[env_index] = np.asarray(reset_obs, dtype=np.float32).reshape(-1)

                if int(cfg.updates_per_env_step) > 0 and replay.size > 0:
                    update_t0 = time.perf_counter()
                    update_index = _run_recurrent_updates(
                        replay,
                        rng=rng,
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                        log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                        mpo_kl_opt=mpo_kl_opt,
                        torch_generator=torch_generator,
                        cfg=cfg,
                        updates=_bounded_update_count(int(cfg.updates_per_env_step), cfg),
                        update_index=update_index,
                        step_count=step_count,
                        device=device,
                        critic_losses=critic_losses,
                        actor_losses=actor_losses,
                        mpo_temperature_losses=mpo_temperature_losses,
                        mpo_temperatures=mpo_temperatures,
                        mpo_mean_kls=mpo_mean_kls,
                        mpo_std_kls=mpo_std_kls,
                        mpo_kl_dual_losses=mpo_kl_dual_losses,
                        mpo_mean_kl_penalties=mpo_mean_kl_penalties,
                        mpo_std_kl_penalties=mpo_std_kl_penalties,
                        valid_steps_per_update=valid_steps_per_update,
                        loss_rows=loss_rows,
                        timing=timing,
                    )
                    learner_time_s += time.perf_counter() - update_t0

                if cfg.eval_interval_steps is not None and step_count % int(cfg.eval_interval_steps) == 0:
                    eval_t0 = time.perf_counter()
                    interval_eval = evaluate_tcv_actor_detailed(eval_factory, actor, episodes=int(cfg.eval_episodes), max_steps=int(cfg.eval_max_steps), seed=_eval_seed_base(cfg) + step_count, device=device)
                    evaluation_time_s += time.perf_counter() - eval_t0
                    interval_returns = interval_eval["returns"]
                    interval_mean = float(np.mean(interval_returns)) if interval_returns else 0.0
                    eval_history.append({"step": step_count, "returns": interval_returns, "mean_return": interval_mean, "tracking_diagnostics": interval_eval["tracking_diagnostics"]})
                    wandb_logger.log_eval({"mean_return": interval_mean, "tracking_diagnostics": interval_eval["tracking_diagnostics"]}, step=step_count)
                    if cfg.checkpoint_dir is not None and interval_mean > best_eval_score:
                        best_eval_score = interval_mean
                        best_checkpoint_path = _save_tcv_checkpoint(
                            actor=actor,
                            actor_target=actor_target,
                            q1=q1,
                            q2=q2,
                            q1_target=q1_target,
                            q2_target=q2_target,
                            actor_opt=actor_opt,
                            critic_opt=critic_opt,
                            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
                            cfg=cfg,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            total_steps=step_count,
                            update_index=update_index,
                            best_eval_score=best_eval_score,
                            numpy_rng_state=rng.bit_generator.state,
                            torch_generator_state=torch_generator.get_state(),
                            training_contract=training_contract,
                            path=Path(cfg.checkpoint_dir) / cfg.best_checkpoint_name,
                        )

                if cfg.checkpoint_dir is not None and cfg.checkpoint_interval_steps is not None and step_count % int(cfg.checkpoint_interval_steps) == 0:
                    step_path = Path(cfg.checkpoint_dir) / f"step_{step_count:08d}.pt"
                    _save_tcv_checkpoint(
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        total_steps=step_count,
                        update_index=update_index,
                        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
                        numpy_rng_state=rng.bit_generator.state,
                        torch_generator_state=torch_generator.get_state(),
                        training_contract=training_contract,
                        path=step_path,
                    )
                    _prune_step_checkpoints(Path(cfg.checkpoint_dir), keep=cfg.max_step_checkpoints)
                    latest_checkpoint_path = _save_tcv_checkpoint(
                        actor=actor,
                        actor_target=actor_target,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        actor_opt=actor_opt,
                        critic_opt=critic_opt,
                        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        total_steps=step_count,
                        update_index=update_index,
                        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
                        numpy_rng_state=rng.bit_generator.state,
                        torch_generator_state=torch_generator.get_state(),
                        training_contract=training_contract,
                        path=Path(cfg.checkpoint_dir) / cfg.latest_checkpoint_name,
                    )
                progress.update(
                    step_count,
                    status=_tcv_progress_status(
                        replay_episodes=replay.size,
                        replay_transitions=replay.total_transitions,
                        update_count=update_index,
                        critic_loss=critic_losses[-1] if critic_losses else None,
                        actor_loss=actor_losses[-1] if actor_losses else None,
                        episodes=len(episode_returns),
                    ),
                )
                wandb_logger.log(
                    {
                        "train": _tcv_progress_status(
                            replay_episodes=replay.size,
                            replay_transitions=replay.total_transitions,
                            update_count=update_index,
                            critic_loss=critic_losses[-1] if critic_losses else None,
                            actor_loss=actor_losses[-1] if actor_losses else None,
                            episodes=len(episode_returns),
                        ),
                        "mpo": {
                            "temperature_loss": mpo_temperature_losses[-1] if mpo_temperature_losses else None,
                            "temperature": mpo_temperatures[-1] if mpo_temperatures else None,
                            "mean_kl": mpo_mean_kls[-1] if mpo_mean_kls else None,
                            "std_kl": mpo_std_kls[-1] if mpo_std_kls else None,
                            "kl_dual_loss": mpo_kl_dual_losses[-1] if mpo_kl_dual_losses else None,
                            "mean_kl_penalty": mpo_mean_kl_penalties[-1] if mpo_mean_kl_penalties else None,
                            "std_kl_penalty": mpo_std_kl_penalties[-1] if mpo_std_kl_penalties else None,
                            "valid_steps": valid_steps_per_update[-1] if valid_steps_per_update else None,
                        },
                    },
                    step=step_count,
                )
    finally:
        progress.close(
            status=_tcv_progress_status(
                replay_episodes=replay.size,
                replay_transitions=replay.total_transitions,
                update_count=update_index,
                critic_loss=critic_losses[-1] if critic_losses else None,
                actor_loss=actor_losses[-1] if actor_losses else None,
                episodes=len(episode_returns),
            )
        )
        reward_writer.close()
        for env in envs:
            env.close()

    for env_index, builder in enumerate(episode_builders):
        if builder.length > 0:
            replay.add_episode(**builder.to_episode_arrays())
            episode_returns.append(float(running_returns[env_index]))
            episode_lengths.append(int(running_lengths[env_index]))
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
            update_t0 = time.perf_counter()
            update_index = _run_recurrent_updates(
                replay,
                rng=rng,
                actor=actor,
                actor_target=actor_target,
                q1=q1,
                q2=q2,
                q1_target=q1_target,
                q2_target=q2_target,
                actor_opt=actor_opt,
                critic_opt=critic_opt,
                log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                mpo_kl_opt=mpo_kl_opt,
                torch_generator=torch_generator,
                cfg=cfg,
                updates=_bounded_update_count(int(cfg.updates_per_episode), cfg),
                update_index=update_index,
                step_count=step_count,
                device=device,
                critic_losses=critic_losses,
                actor_losses=actor_losses,
                mpo_temperature_losses=mpo_temperature_losses,
                mpo_temperatures=mpo_temperatures,
                mpo_mean_kls=mpo_mean_kls,
                mpo_std_kls=mpo_std_kls,
                mpo_kl_dual_losses=mpo_kl_dual_losses,
                mpo_mean_kl_penalties=mpo_mean_kl_penalties,
                mpo_std_kl_penalties=mpo_std_kl_penalties,
                valid_steps_per_update=valid_steps_per_update,
                loss_rows=loss_rows,
                timing=timing,
            )
            learner_time_s += time.perf_counter() - update_t0

    eval_t0 = time.perf_counter()
    final_eval = evaluate_tcv_actor_detailed(eval_factory, actor, episodes=int(cfg.eval_episodes), max_steps=int(cfg.eval_max_steps), seed=_eval_seed_base(cfg), device=device)
    evaluation_time_s += time.perf_counter() - eval_t0
    eval_returns = final_eval["returns"]
    final_mean = float(np.mean(eval_returns)) if eval_returns else 0.0
    wandb_logger.log_eval({"mean_return": final_mean, "tracking_diagnostics": final_eval["tracking_diagnostics"]}, step=step_count)
    if cfg.checkpoint_dir is not None and final_mean > best_eval_score:
        best_eval_score = final_mean
        best_checkpoint_path = _save_tcv_checkpoint(
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            total_steps=step_count,
            update_index=update_index,
            best_eval_score=best_eval_score,
            numpy_rng_state=rng.bit_generator.state,
            torch_generator_state=torch_generator.get_state(),
            training_contract=training_contract,
            path=Path(cfg.checkpoint_dir) / cfg.best_checkpoint_name,
        )
    checkpoint_path = _save_tcv_checkpoint(
        actor=actor,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
        cfg=cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_steps=step_count,
        update_index=update_index,
        best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
        numpy_rng_state=rng.bit_generator.state,
        torch_generator_state=torch_generator.get_state(),
        training_contract=training_contract,
    ) if cfg.checkpoint_dir is not None else None
    if cfg.checkpoint_dir is not None:
        latest_checkpoint_path = _save_tcv_checkpoint(
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
                            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
                            mpo_kl_opt=mpo_kl_opt,
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            total_steps=step_count,
            update_index=update_index,
            best_eval_score=None if best_eval_score == float("-inf") else best_eval_score,
            numpy_rng_state=rng.bit_generator.state,
            torch_generator_state=torch_generator.get_state(),
            training_contract=training_contract,
            path=Path(cfg.checkpoint_dir) / cfg.latest_checkpoint_name,
        )
    best_actor_export_dir = export_best_actor_artifact(
        checkpoint_path=best_checkpoint_path,
        output_dir=cfg.output_dir,
        training_contract=training_contract,
        metadata={"trainer": "tcv_style_recurrent_actor_critic_v1", "algorithm": "tcv_mpo_recurrent_actor_critic_v1", "run_metadata": cfg.run_metadata},
    ) if bool(cfg.export_best_actor) else None
    metrics_json, losses_csv = _write_tcv_artifacts(
        cfg=cfg,
        total_steps=step_count,
        replay_episodes=replay.size,
        replay_transitions=replay.total_transitions,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        mpo_temperature_losses=mpo_temperature_losses,
        mpo_temperatures=mpo_temperatures,
        mpo_mean_kls=mpo_mean_kls,
        mpo_std_kls=mpo_std_kls,
        mpo_kl_dual_losses=mpo_kl_dual_losses,
        mpo_mean_kl_penalties=mpo_mean_kl_penalties,
        mpo_std_kl_penalties=mpo_std_kl_penalties,
        valid_steps_per_update=valid_steps_per_update,
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
            replay_sampling_time_s=float(timing["replay_sampling_time_s"]),
            learner_time_s=learner_time_s,
            evaluation_time_s=evaluation_time_s,
        ),
        device_metadata=device_selection.to_metadata(),
    )
    wandb_logger.log_final(
        {
            "total_steps": int(step_count),
            "replay_episodes": int(replay.size),
            "replay_transitions": int(replay.total_transitions),
            "critic_updates": len(critic_losses),
            "actor_updates": len(actor_losses),
            "mpo_kl_dual_updates": len(mpo_kl_dual_losses),
            "last_critic_loss": float(critic_losses[-1]) if critic_losses else None,
            "last_actor_loss": float(actor_losses[-1]) if actor_losses else None,
            "last_mpo_temperature": float(mpo_temperatures[-1]) if mpo_temperatures else None,
            "last_mpo_mean_kl": float(mpo_mean_kls[-1]) if mpo_mean_kls else None,
            "last_mpo_std_kl": float(mpo_std_kls[-1]) if mpo_std_kls else None,
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
    return TCVStyleTrainingResult(
        total_steps=step_count,
        replay_episodes=replay.size,
        replay_transitions=replay.total_transitions,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        mpo_temperature_losses=mpo_temperature_losses,
        mpo_temperatures=mpo_temperatures,
        mpo_mean_kls=mpo_mean_kls,
        mpo_std_kls=mpo_std_kls,
        mpo_kl_dual_losses=mpo_kl_dual_losses,
        mpo_mean_kl_penalties=mpo_mean_kl_penalties,
        mpo_std_kl_penalties=mpo_std_kl_penalties,
        valid_steps_per_update=valid_steps_per_update,
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


def _train_tcv_style_actor_critic_true_batched_gpu(env_factory: EnvFactory, cfg: TCVStyleTrainerConfig, *, eval_env_factory: EnvFactory | None = None) -> TCVStyleTrainingResult:
    eval_factory = env_factory if eval_env_factory is None else eval_env_factory
    try:
        from tokamak_control.core.batched_gpu_simulator import configure_batched_gpu_simulator_profiling

        configure_batched_gpu_simulator_profiling(enabled=True, summary_every=0, reset=True)
    except Exception:
        pass
    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch_generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    device, device_selection = resolve_training_device(cfg.device)
    started_at = time.perf_counter()
    collection_time_s = 0.0
    actor_inference_time_s = 0.0
    env_step_time_s = 0.0
    timing = {"replay_sampling_time_s": 0.0}
    learner_time_s = 0.0
    evaluation_time_s = 0.0

    env = env_factory()
    if not hasattr(env, "reset_batch") or not hasattr(env, "step_batch"):
        raise RuntimeError("true batched GPU factory did not return a batch environment")
    seeds = np.arange(int(cfg.seed), int(cfg.seed) + int(cfg.num_envs), dtype=int)
    reset = env.reset_batch(seeds)
    observations = np.asarray(reset.observations, dtype=np.float32)
    obs_dim = int(env.obs_dim)
    action_dim = int(env.action_dim)
    if observations.shape != (int(cfg.num_envs), obs_dim):
        raise ValueError("batched reset returned observations with unexpected shape")

    diagnostics = TrainingDiagnostics()
    training_contract: dict[str, object] | None = None
    reset_metadata: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []
    for env_index, reset_info in enumerate(reset.infos):
        diagnostics.record_reset_info(reset_info)
        if training_contract is None:
            training_contract = _extract_training_contract(reset_info)
        reset_record = reset_artifact_record(reset_info, env_index=env_index, episode=0)
        reset_metadata.append(reset_record)
        reference_records.append(dict(reset_record))

    actor = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.actor_hidden_dim))).to(device)
    actor_target = FeedForwardActor(ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.actor_hidden_dim))).to(device)
    critic_cfg = RecurrentCriticConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.critic_hidden_dim), mlp_hidden_dim=int(cfg.critic_mlp_hidden_dim))
    q1 = RecurrentQCritic(critic_cfg).to(device)
    q2 = RecurrentQCritic(critic_cfg).to(device)
    q1_target = RecurrentQCritic(critic_cfg).to(device)
    q2_target = RecurrentQCritic(critic_cfg).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(cfg.actor_lr))
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=float(cfg.critic_lr))
    log_mpo_mean_kl_penalty = torch.tensor(_inverse_softplus(float(cfg.mpo_initial_mean_kl_penalty)), dtype=torch.float32, device=device, requires_grad=True)
    log_mpo_std_kl_penalty = torch.tensor(_inverse_softplus(float(cfg.mpo_initial_std_kl_penalty)), dtype=torch.float32, device=device, requires_grad=True)
    mpo_kl_opt = torch.optim.Adam([log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty], lr=float(cfg.mpo_kl_lr))
    if cfg.resume_checkpoint is not None:
        _load_tcv_training_state(
            cfg.resume_checkpoint,
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
            mpo_kl_opt=mpo_kl_opt,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            torch_generator=torch_generator,
        )
    else:
        actor_target.load_state_dict(actor.state_dict())
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

    replay = EpisodeReplayBuffer(capacity_episodes=int(cfg.replay_capacity_episodes), obs_dim=obs_dim, action_dim=action_dim)
    episode_builders = [_EpisodeBuilder() for _ in range(int(cfg.num_envs))]
    running_returns = [0.0 for _ in range(int(cfg.num_envs))]
    running_lengths = [0 for _ in range(int(cfg.num_envs))]
    running_ip_errors: list[list[float]] = [[] for _ in range(int(cfg.num_envs))]
    running_shape_errors: list[list[float]] = [[] for _ in range(int(cfg.num_envs))]
    running_boundary_failure_steps = [0 for _ in range(int(cfg.num_envs))]
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_indices = [0 for _ in range(int(cfg.num_envs))]
    episode_records: list[dict[str, object]] = []
    critic_losses: list[float] = []
    actor_losses: list[float] = []
    mpo_temperature_losses: list[float] = []
    mpo_temperatures: list[float] = []
    mpo_mean_kls: list[float] = []
    mpo_std_kls: list[float] = []
    mpo_kl_dual_losses: list[float] = []
    mpo_mean_kl_penalties: list[float] = []
    mpo_std_kl_penalties: list[float] = []
    valid_steps_per_update: list[int] = []
    loss_rows: list[dict[str, object]] = []
    eval_history: list[dict[str, object]] = []
    step_count = 0
    update_index = 0
    best_eval_score = float("-inf")
    best_checkpoint_path: Path | None = None
    latest_checkpoint_path: Path | None = None
    reward_writer = RewardComponentWriter(cfg.output_dir)
    progress = TrainingProgressBar(total_steps=int(cfg.total_steps), label="tcv_style_batched_gpu", enabled=bool(cfg.progress))
    wandb_logger = WandBLogger(cfg.wandb, config=_checkpoint_safe_config(cfg), run_metadata={**json_safe(cfg.run_metadata), "device": device_selection.to_metadata(), "trainer": "tcv_style_recurrent_actor_critic_v1", "env_backend": "true_batched_gpu"})
    progress.update(0, status=_tcv_progress_status(replay_episodes=0, replay_transitions=0, update_count=0), force=True)

    try:
        while step_count < int(cfg.total_steps):
            active_count = min(int(cfg.num_envs), int(cfg.total_steps) - step_count)
            warmup = np.asarray([step_count + i < int(cfg.warmup_steps) for i in range(int(cfg.num_envs))], dtype=bool)
            collect_t0 = time.perf_counter()
            actions = _select_training_actions_batch(actor, observations, action_dim=action_dim, warmup=warmup, rng=rng, torch_generator=torch_generator, device=device)
            action_elapsed = time.perf_counter() - collect_t0
            actor_inference_time_s += action_elapsed
            collection_time_s += action_elapsed
            collect_t0 = time.perf_counter()
            step = env.step_batch(actions)
            step_elapsed = time.perf_counter() - collect_t0
            env_step_time_s += step_elapsed
            collection_time_s += step_elapsed
            next_observations = np.asarray(step.observations, dtype=np.float32)
            done_indices: list[int] = []
            reset_seeds: list[int] = []
            for env_index in range(active_count):
                reward = float(step.rewards[env_index])
                terminated = bool(step.terminated[env_index])
                truncated = bool(step.truncated[env_index])
                info = step.infos[env_index]
                diagnostics.record_step_info(info)
                reward_components = info.get("reward_components") if isinstance(info, dict) else None
                reward_writer.record(step=step_count + env_index + 1, env_index=env_index, episode=episode_indices[env_index], components=reward_components)
                wandb_logger.log({"train": {"reward": reward, "env_index": int(env_index), "episode": int(episode_indices[env_index]), "terminated": terminated, "truncated": truncated}, "reward_components": reward_components if isinstance(reward_components, dict) else {}}, step=step_count + env_index + 1)
                episode_builders[env_index].add(observations[env_index], actions[env_index], reward, next_observations[env_index], terminated, truncated)
                running_returns[env_index] += reward
                running_lengths[env_index] += 1
                record_episode_step_artifacts(info, ip_errors=running_ip_errors[env_index], shape_errors=running_shape_errors[env_index], boundary_failure_counter=running_boundary_failure_steps, env_index=env_index)
                if terminated or truncated:
                    replay.add_episode(**episode_builders[env_index].to_episode_arrays())
                    episode_builders[env_index].clear()
                    episode_returns.append(float(running_returns[env_index]))
                    episode_lengths.append(int(running_lengths[env_index]))
                    episode_record = episode_artifact_record(env_index=env_index, episode=episode_indices[env_index], episode_return=running_returns[env_index], episode_length=running_lengths[env_index], terminated=terminated, truncated=truncated, termination_reason=termination_reason_from_step_info(info, terminated=terminated, truncated=truncated), ip_errors=running_ip_errors[env_index], shape_errors=running_shape_errors[env_index], boundary_failure_steps=running_boundary_failure_steps[env_index], reset_record=reset_metadata[env_index])
                    episode_records.append(episode_record)
                    wandb_logger.log_episode({**episode_record, "return": float(running_returns[env_index]), "length": int(running_lengths[env_index])}, step=step_count + env_index + 1)
                    episode_indices[env_index] += 1
                    running_returns[env_index] = 0.0
                    running_lengths[env_index] = 0
                    running_ip_errors[env_index].clear()
                    running_shape_errors[env_index].clear()
                    running_boundary_failure_steps[env_index] = 0
                    done_indices.append(env_index)
                    reset_seeds.append(int(cfg.seed) + 20_000 + step_count + env_index)
            observations = next_observations
            step_count += active_count
            if done_indices:
                update_t0 = time.perf_counter()
                update_index = _run_recurrent_updates(replay, rng=rng, actor=actor, actor_target=actor_target, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target, actor_opt=actor_opt, critic_opt=critic_opt, log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty=log_mpo_std_kl_penalty, mpo_kl_opt=mpo_kl_opt, torch_generator=torch_generator, cfg=cfg, updates=_bounded_update_count(int(cfg.updates_per_episode) * len(done_indices), cfg), update_index=update_index, step_count=step_count, device=device, critic_losses=critic_losses, actor_losses=actor_losses, mpo_temperature_losses=mpo_temperature_losses, mpo_temperatures=mpo_temperatures, mpo_mean_kls=mpo_mean_kls, mpo_std_kls=mpo_std_kls, mpo_kl_dual_losses=mpo_kl_dual_losses, mpo_mean_kl_penalties=mpo_mean_kl_penalties, mpo_std_kl_penalties=mpo_std_kl_penalties, valid_steps_per_update=valid_steps_per_update, loss_rows=loss_rows, timing=timing)
                learner_time_s += time.perf_counter() - update_t0
                reset_obs, reset_infos = env.reset_indices(done_indices, reset_seeds)
                for local, env_index in enumerate(done_indices):
                    reset_info = reset_infos[local]
                    diagnostics.record_reset_info(reset_info)
                    if training_contract is None:
                        training_contract = _extract_training_contract(reset_info)
                    reset_metadata[env_index] = reset_artifact_record(reset_info, env_index=env_index, episode=episode_indices[env_index])
                    reference_records.append(dict(reset_metadata[env_index]))
                    observations[env_index] = reset_obs[local]
            if int(cfg.updates_per_env_step) > 0 and replay.size > 0:
                update_t0 = time.perf_counter()
                update_index = _run_recurrent_updates(replay, rng=rng, actor=actor, actor_target=actor_target, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target, actor_opt=actor_opt, critic_opt=critic_opt, log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty=log_mpo_std_kl_penalty, mpo_kl_opt=mpo_kl_opt, torch_generator=torch_generator, cfg=cfg, updates=_bounded_update_count(int(cfg.updates_per_env_step) * active_count, cfg), update_index=update_index, step_count=step_count, device=device, critic_losses=critic_losses, actor_losses=actor_losses, mpo_temperature_losses=mpo_temperature_losses, mpo_temperatures=mpo_temperatures, mpo_mean_kls=mpo_mean_kls, mpo_std_kls=mpo_std_kls, mpo_kl_dual_losses=mpo_kl_dual_losses, mpo_mean_kl_penalties=mpo_mean_kl_penalties, mpo_std_kl_penalties=mpo_std_kl_penalties, valid_steps_per_update=valid_steps_per_update, loss_rows=loss_rows, timing=timing)
                learner_time_s += time.perf_counter() - update_t0
            progress.update(step_count, status=_tcv_progress_status(replay_episodes=replay.size, replay_transitions=replay.total_transitions, update_count=update_index, critic_loss=critic_losses[-1] if critic_losses else None, actor_loss=actor_losses[-1] if actor_losses else None, episodes=len(episode_returns)))
    finally:
        progress.close(status=_tcv_progress_status(replay_episodes=replay.size, replay_transitions=replay.total_transitions, update_count=update_index, critic_loss=critic_losses[-1] if critic_losses else None, actor_loss=actor_losses[-1] if actor_losses else None, episodes=len(episode_returns)))
        reward_writer.close()
        env.close()

    for env_index, builder in enumerate(episode_builders):
        if builder.length > 0:
            replay.add_episode(**builder.to_episode_arrays())
            episode_returns.append(float(running_returns[env_index]))
            episode_lengths.append(int(running_lengths[env_index]))
            episode_records.append(episode_artifact_record(env_index=env_index, episode=episode_indices[env_index], episode_return=running_returns[env_index], episode_length=running_lengths[env_index], terminated=False, truncated=True, termination_reason="training_horizon", ip_errors=running_ip_errors[env_index], shape_errors=running_shape_errors[env_index], boundary_failure_steps=running_boundary_failure_steps[env_index], reset_record=reset_metadata[env_index]))
    if replay.size > 0:
        update_t0 = time.perf_counter()
        update_index = _run_recurrent_updates(replay, rng=rng, actor=actor, actor_target=actor_target, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target, actor_opt=actor_opt, critic_opt=critic_opt, log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty=log_mpo_std_kl_penalty, mpo_kl_opt=mpo_kl_opt, torch_generator=torch_generator, cfg=cfg, updates=_bounded_update_count(int(cfg.updates_per_episode), cfg), update_index=update_index, step_count=step_count, device=device, critic_losses=critic_losses, actor_losses=actor_losses, mpo_temperature_losses=mpo_temperature_losses, mpo_temperatures=mpo_temperatures, mpo_mean_kls=mpo_mean_kls, mpo_std_kls=mpo_std_kls, mpo_kl_dual_losses=mpo_kl_dual_losses, mpo_mean_kl_penalties=mpo_mean_kl_penalties, mpo_std_kl_penalties=mpo_std_kl_penalties, valid_steps_per_update=valid_steps_per_update, loss_rows=loss_rows, timing=timing)
        learner_time_s += time.perf_counter() - update_t0

    eval_t0 = time.perf_counter()
    final_eval = evaluate_tcv_actor_detailed(eval_factory, actor, episodes=int(cfg.eval_episodes), max_steps=int(cfg.eval_max_steps), seed=_eval_seed_base(cfg), device=device)
    evaluation_time_s += time.perf_counter() - eval_t0
    eval_returns = final_eval["returns"]
    final_mean = float(np.mean(eval_returns)) if eval_returns else 0.0
    checkpoint_path = _save_tcv_checkpoint(actor=actor, actor_target=actor_target, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target, actor_opt=actor_opt, critic_opt=critic_opt, log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty, log_mpo_std_kl_penalty=log_mpo_std_kl_penalty, mpo_kl_opt=mpo_kl_opt, cfg=cfg, obs_dim=obs_dim, action_dim=action_dim, total_steps=step_count, update_index=update_index, best_eval_score=final_mean, numpy_rng_state=rng.bit_generator.state, torch_generator_state=torch_generator.get_state(), training_contract=training_contract) if cfg.checkpoint_dir is not None else None
    latest_checkpoint_path = checkpoint_path
    best_checkpoint_path = checkpoint_path
    best_actor_export_dir = export_best_actor_artifact(checkpoint_path=best_checkpoint_path, output_dir=cfg.output_dir, training_contract=training_contract, metadata={"trainer": "tcv_style_recurrent_actor_critic_v1", "algorithm": "tcv_mpo_recurrent_actor_critic_v1", "run_metadata": cfg.run_metadata}) if bool(cfg.export_best_actor) else None
    metrics_json, losses_csv = _write_tcv_artifacts(cfg=cfg, total_steps=step_count, replay_episodes=replay.size, replay_transitions=replay.total_transitions, critic_losses=critic_losses, actor_losses=actor_losses, mpo_temperature_losses=mpo_temperature_losses, mpo_temperatures=mpo_temperatures, mpo_mean_kls=mpo_mean_kls, mpo_std_kls=mpo_std_kls, mpo_kl_dual_losses=mpo_kl_dual_losses, mpo_mean_kl_penalties=mpo_mean_kl_penalties, mpo_std_kl_penalties=mpo_std_kl_penalties, valid_steps_per_update=valid_steps_per_update, loss_rows=loss_rows, episode_returns=episode_returns, episode_lengths=episode_lengths, eval_returns=eval_returns, eval_history=eval_history, episode_records=episode_records, reference_records=reference_records, tracking_diagnostics=diagnostics.summary(), eval_tracking_diagnostics=final_eval["tracking_diagnostics"], checkpoint_path=checkpoint_path, latest_checkpoint_path=latest_checkpoint_path, best_checkpoint_path=best_checkpoint_path, best_actor_export_dir=best_actor_export_dir, throughput=_throughput_metrics(total_steps=step_count, update_count=update_index, total_elapsed_s=time.perf_counter() - started_at, collection_time_s=collection_time_s, actor_inference_time_s=actor_inference_time_s, env_step_time_s=env_step_time_s, replay_sampling_time_s=float(timing["replay_sampling_time_s"]), learner_time_s=learner_time_s, evaluation_time_s=evaluation_time_s), device_metadata=device_selection.to_metadata())
    wandb_logger.log_final({"total_steps": int(step_count), "replay_episodes": int(replay.size), "replay_transitions": int(replay.total_transitions), "critic_updates": len(critic_losses), "actor_updates": len(actor_losses), "eval_mean_return": final_mean, "tracking_diagnostics": diagnostics.summary(), "eval_tracking_diagnostics": final_eval["tracking_diagnostics"]}, artifact_paths={"metrics": metrics_json, "losses": losses_csv, "checkpoint": checkpoint_path, "latest_checkpoint": latest_checkpoint_path, "best_checkpoint": best_checkpoint_path, "best_actor_export": best_actor_export_dir}, step=step_count)
    wandb_logger.close()
    return TCVStyleTrainingResult(total_steps=step_count, replay_episodes=replay.size, replay_transitions=replay.total_transitions, critic_losses=critic_losses, actor_losses=actor_losses, mpo_temperature_losses=mpo_temperature_losses, mpo_temperatures=mpo_temperatures, mpo_mean_kls=mpo_mean_kls, mpo_std_kls=mpo_std_kls, mpo_kl_dual_losses=mpo_kl_dual_losses, mpo_mean_kl_penalties=mpo_mean_kl_penalties, mpo_std_kl_penalties=mpo_std_kl_penalties, valid_steps_per_update=valid_steps_per_update, episode_returns=episode_returns, episode_lengths=episode_lengths, eval_returns=eval_returns, eval_history=eval_history, checkpoint_path=checkpoint_path, metrics_json=metrics_json, losses_csv=losses_csv, latest_checkpoint_path=latest_checkpoint_path, best_checkpoint_path=best_checkpoint_path, best_actor_export_dir=best_actor_export_dir)


def evaluate_tcv_actor(env_factory: EnvFactory, actor: FeedForwardActor, *, episodes: int, max_steps: int, seed: int, device: torch.device | str = "cpu") -> list[float]:
    return list(evaluate_tcv_actor_detailed(env_factory, actor, episodes=episodes, max_steps=max_steps, seed=seed, device=device)["returns"])


def evaluate_tcv_actor_detailed(env_factory: EnvFactory, actor: FeedForwardActor, *, episodes: int, max_steps: int, seed: int, device: torch.device | str = "cpu") -> dict[str, object]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be > 0")
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be > 0")
    if bool(getattr(env_factory, "is_true_batched_gpu_factory", False)):
        return _evaluate_tcv_actor_detailed_true_batched_gpu(env_factory, actor, episodes=episodes, max_steps=max_steps, seed=seed, device=device)
    device = torch.device(device)
    was_training = actor.training
    actor.eval()
    returns: list[float] = []
    diagnostics = TrainingDiagnostics()
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


def _evaluate_tcv_actor_detailed_true_batched_gpu(env_factory: EnvFactory, actor: FeedForwardActor, *, episodes: int, max_steps: int, seed: int, device: torch.device | str = "cpu") -> dict[str, object]:
    device = torch.device(device)
    env = env_factory()
    batch = int(getattr(env, "num_envs", episodes))
    remaining = int(episodes)
    returns: list[float] = []
    diagnostics = TrainingDiagnostics()
    was_training = actor.training
    actor.eval()
    try:
        while remaining > 0:
            active = min(batch, remaining)
            seeds = np.arange(int(seed) + len(returns), int(seed) + len(returns) + batch, dtype=int)
            reset = env.reset_batch(seeds)
            obs = np.asarray(reset.observations, dtype=np.float32)
            for info in reset.infos[:active]:
                diagnostics.record_reset_info(info)
            totals = np.zeros((batch,), dtype=float)
            done = np.zeros((batch,), dtype=bool)
            for _ in range(int(max_steps)):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = actor.deterministic_action(obs_t).detach().cpu().numpy()
                step = env.step_batch(action)
                obs = np.asarray(step.observations, dtype=np.float32)
                for i in range(active):
                    if done[i]:
                        continue
                    diagnostics.record_step_info(step.infos[i])
                    totals[i] += float(step.rewards[i])
                    if bool(step.terminated[i]) or bool(step.truncated[i]):
                        done[i] = True
                if bool(np.all(done[:active])):
                    break
            returns.extend(float(v) for v in totals[:active])
            remaining -= active
    finally:
        actor.train(was_training)
    return {"returns": returns, "tracking_diagnostics": diagnostics.summary()}


class _EpisodeBuilder:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_observations: list[np.ndarray] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    @property
    def length(self) -> int:
        return len(self.rewards)

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, next_observation: np.ndarray, terminated: bool, truncated: bool) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.next_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def to_episode_arrays(self) -> dict[str, np.ndarray]:
        if self.length <= 0:
            raise ValueError("cannot flush empty episode")
        return {
            "observations": np.stack(self.observations, axis=0),
            "actions": np.stack(self.actions, axis=0),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "next_observations": np.stack(self.next_observations, axis=0),
            "terminated": np.asarray(self.terminated, dtype=bool),
            "truncated": np.asarray(self.truncated, dtype=bool),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.terminated.clear()
        self.truncated.clear()


def _select_training_action(
    actor: FeedForwardActor,
    observation: np.ndarray,
    *,
    action_dim: int,
    warmup: bool,
    rng: np.random.Generator,
    torch_generator: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    if warmup:
        return rng.uniform(-1.0, 1.0, size=(int(action_dim),)).astype(np.float32)
    obs_t = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        action, _mean, _std = actor.sample_action(obs_t, generator=torch_generator if device.type == "cpu" else None)
    return action.detach().cpu().numpy()[0].astype(np.float32, copy=False)


def _select_training_actions_batch(
    actor: FeedForwardActor,
    observations: np.ndarray,
    *,
    action_dim: int,
    warmup: np.ndarray,
    rng: np.random.Generator,
    torch_generator: torch.Generator,
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
            sampled, _mean, _std = actor.sample_action(obs_t, generator=torch_generator if device.type == "cpu" else None)
        actions[~warmup_mask] = sampled.detach().cpu().numpy().astype(np.float32, copy=False)
    if np.any(warmup_mask):
        actions[warmup_mask] = rng.uniform(-1.0, 1.0, size=(int(np.count_nonzero(warmup_mask)), int(action_dim))).astype(np.float32)
    return np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)


def _update_from_sequence_replay(
    replay: EpisodeReplayBuffer,
    *,
    rng: np.random.Generator,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: RecurrentQCritic,
    q2: RecurrentQCritic,
    q1_target: RecurrentQCritic,
    q2_target: RecurrentQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    log_mpo_mean_kl_penalty: torch.Tensor,
    log_mpo_std_kl_penalty: torch.Tensor,
    mpo_kl_opt: torch.optim.Optimizer,
    torch_generator: torch.Generator,
    cfg: TCVStyleTrainerConfig,
    update_index: int,
    device: torch.device,
    timing: dict[str, float] | None = None,
) -> RecurrentUpdateResult:
    sample_t0 = time.perf_counter()
    batch = replay.sample_sequences(batch_size=int(cfg.batch_size), sequence_length=int(cfg.sequence_length), rng=rng)
    if timing is not None:
        timing["replay_sampling_time_s"] = float(timing.get("replay_sampling_time_s", 0.0)) + (time.perf_counter() - sample_t0)
    return recurrent_critic_update_once(
        batch,
        actor=actor,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
        log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
        mpo_kl_opt=mpo_kl_opt,
        mpo_epsilon=float(cfg.mpo_epsilon),
        mpo_mean_kl_epsilon=float(cfg.mpo_mean_kl_epsilon),
        mpo_std_kl_epsilon=float(cfg.mpo_std_kl_epsilon),
        mpo_action_samples=int(cfg.mpo_action_samples),
        mpo_temperature_iterations=int(cfg.mpo_temperature_iterations),
        mpo_temperature_lr=float(cfg.mpo_temperature_lr),
        mpo_initial_temperature=float(cfg.mpo_initial_temperature),
        torch_generator=torch_generator,
        gamma=float(cfg.gamma),
        tau=float(cfg.tau),
        update_index=int(update_index),
        policy_delay=int(cfg.policy_delay),
        device=device,
    )


def _run_recurrent_updates(
    replay: EpisodeReplayBuffer,
    *,
    rng: np.random.Generator,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: RecurrentQCritic,
    q2: RecurrentQCritic,
    q1_target: RecurrentQCritic,
    q2_target: RecurrentQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    log_mpo_mean_kl_penalty: torch.Tensor,
    log_mpo_std_kl_penalty: torch.Tensor,
    mpo_kl_opt: torch.optim.Optimizer,
    torch_generator: torch.Generator,
    cfg: TCVStyleTrainerConfig,
    updates: int,
    update_index: int,
    step_count: int,
    device: torch.device,
    critic_losses: list[float],
    actor_losses: list[float],
    mpo_temperature_losses: list[float],
    mpo_temperatures: list[float],
    mpo_mean_kls: list[float],
    mpo_std_kls: list[float],
    mpo_kl_dual_losses: list[float],
    mpo_mean_kl_penalties: list[float],
    mpo_std_kl_penalties: list[float],
    valid_steps_per_update: list[int],
    loss_rows: list[dict[str, object]],
    timing: dict[str, float] | None = None,
) -> int:
    for _ in range(int(updates)):
        if replay.size <= 0:
            break
        update_index += 1
        result = _update_from_sequence_replay(
            replay,
            rng=rng,
            actor=actor,
            actor_target=actor_target,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_mpo_mean_kl_penalty=log_mpo_mean_kl_penalty,
            log_mpo_std_kl_penalty=log_mpo_std_kl_penalty,
            mpo_kl_opt=mpo_kl_opt,
            torch_generator=torch_generator,
            cfg=cfg,
            update_index=update_index,
            device=device,
            timing=timing,
        )
        critic_losses.append(result.critic_loss)
        valid_steps_per_update.append(result.valid_steps)
        if result.actor_loss is not None:
            actor_losses.append(result.actor_loss)
        if result.mpo_temperature_loss is not None:
            mpo_temperature_losses.append(result.mpo_temperature_loss)
        if result.mpo_temperature is not None:
            mpo_temperatures.append(result.mpo_temperature)
        if result.mpo_mean_kl is not None:
            mpo_mean_kls.append(result.mpo_mean_kl)
        if result.mpo_std_kl is not None:
            mpo_std_kls.append(result.mpo_std_kl)
        if result.mpo_kl_dual_loss is not None:
            mpo_kl_dual_losses.append(result.mpo_kl_dual_loss)
        if result.mpo_mean_kl_penalty is not None:
            mpo_mean_kl_penalties.append(result.mpo_mean_kl_penalty)
        if result.mpo_std_kl_penalty is not None:
            mpo_std_kl_penalties.append(result.mpo_std_kl_penalty)
        loss_rows.append(
            {
                "step": step_count,
                "update": update_index,
                "critic_loss": result.critic_loss,
                "actor_loss": result.actor_loss,
                "mpo_temperature_loss": result.mpo_temperature_loss,
                "mpo_temperature": result.mpo_temperature,
                "mpo_mean_kl": result.mpo_mean_kl,
                "mpo_std_kl": result.mpo_std_kl,
                "mpo_kl_dual_loss": result.mpo_kl_dual_loss,
                "mpo_mean_kl_penalty": result.mpo_mean_kl_penalty,
                "mpo_std_kl_penalty": result.mpo_std_kl_penalty,
                "valid_steps": result.valid_steps,
            }
        )
    return update_index


def _bounded_update_count(requested: int, cfg: TCVStyleTrainerConfig) -> int:
    count = max(int(requested), 0)
    if cfg.max_learner_catchup_updates is not None:
        count = min(count, int(cfg.max_learner_catchup_updates))
    return count


def _tcv_progress_status(
    *,
    replay_episodes: int,
    replay_transitions: int,
    update_count: int,
    critic_loss: float | None = None,
    actor_loss: float | None = None,
    episodes: int | None = None,
) -> dict[str, object]:
    return {
        "episodes": episodes,
        "replay_ep": int(replay_episodes),
        "replay_steps": int(replay_transitions),
        "updates": int(update_count),
        "critic": critic_loss,
        "actor": actor_loss,
    }


def _inverse_softplus(value: float) -> float:
    positive = float(value)
    if positive <= 0.0:
        raise ValueError("value must be > 0")
    if positive > 20.0:
        return positive
    return float(np.log(np.expm1(positive)))


def _save_tcv_checkpoint(
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor | None = None,
    q1: RecurrentQCritic,
    q2: RecurrentQCritic,
    q1_target: RecurrentQCritic | None = None,
    q2_target: RecurrentQCritic | None = None,
    actor_opt: torch.optim.Optimizer | None = None,
    critic_opt: torch.optim.Optimizer | None = None,
    log_mpo_mean_kl_penalty: torch.Tensor | None = None,
    log_mpo_std_kl_penalty: torch.Tensor | None = None,
    mpo_kl_opt: torch.optim.Optimizer | None = None,
    cfg: TCVStyleTrainerConfig,
    obs_dim: int,
    action_dim: int,
    total_steps: int = 0,
    update_index: int = 0,
    best_eval_score: float | None = None,
    numpy_rng_state: dict[str, object] | None = None,
    torch_generator_state: torch.Tensor | None = None,
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
            "trainer": "tcv_style_recurrent_actor_critic_v1",
            "algorithm": "tcv_mpo_recurrent_actor_critic_v1",
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
            "log_mpo_mean_kl_penalty": None if log_mpo_mean_kl_penalty is None else log_mpo_mean_kl_penalty.detach().cpu(),
            "log_mpo_std_kl_penalty": None if log_mpo_std_kl_penalty is None else log_mpo_std_kl_penalty.detach().cpu(),
            "mpo_kl_optimizer_state_dict": None if mpo_kl_opt is None else mpo_kl_opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "training_torch_generator_state": torch_generator_state,
            "numpy_rng_state": json_safe(numpy_rng_state),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _load_tcv_training_state(
    path: str | Path,
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: RecurrentQCritic,
    q2: RecurrentQCritic,
    q1_target: RecurrentQCritic,
    q2_target: RecurrentQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    log_mpo_mean_kl_penalty: torch.Tensor,
    log_mpo_std_kl_penalty: torch.Tensor,
    mpo_kl_opt: torch.optim.Optimizer,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    torch_generator: torch.Generator,
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
    saved_log_mean_penalty = checkpoint.get("log_mpo_mean_kl_penalty")
    if isinstance(saved_log_mean_penalty, torch.Tensor):
        with torch.no_grad():
            log_mpo_mean_kl_penalty.copy_(saved_log_mean_penalty.to(device=device, dtype=log_mpo_mean_kl_penalty.dtype))
    saved_log_std_penalty = checkpoint.get("log_mpo_std_kl_penalty")
    if isinstance(saved_log_std_penalty, torch.Tensor):
        with torch.no_grad():
            log_mpo_std_kl_penalty.copy_(saved_log_std_penalty.to(device=device, dtype=log_mpo_std_kl_penalty.dtype))
    mpo_kl_opt_state = checkpoint.get("mpo_kl_optimizer_state_dict")
    if mpo_kl_opt_state is not None:
        mpo_kl_opt.load_state_dict(mpo_kl_opt_state)
    torch_state = checkpoint.get("torch_rng_state")
    if isinstance(torch_state, torch.Tensor) and torch_state.device.type == "cpu":
        torch.set_rng_state(torch_state)
    generator_state = checkpoint.get("training_torch_generator_state")
    if isinstance(generator_state, torch.Tensor) and generator_state.device.type == "cpu":
        torch_generator.set_state(generator_state)


def _write_tcv_artifacts(
    *,
    cfg: TCVStyleTrainerConfig,
    total_steps: int,
    replay_episodes: int,
    replay_transitions: int,
    critic_losses: list[float],
    actor_losses: list[float],
    mpo_temperature_losses: list[float],
    mpo_temperatures: list[float],
    mpo_mean_kls: list[float],
    mpo_std_kls: list[float],
    mpo_kl_dual_losses: list[float],
    mpo_mean_kl_penalties: list[float],
    mpo_std_kl_penalties: list[float],
    valid_steps_per_update: list[int],
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
    loss_fields = [
        "step",
        "update",
        "critic_loss",
        "actor_loss",
        "mpo_temperature_loss",
        "mpo_temperature",
        "mpo_mean_kl",
        "mpo_std_kl",
        "mpo_kl_dual_loss",
        "mpo_mean_kl_penalty",
        "mpo_std_kl_penalty",
        "valid_steps",
    ]
    with losses_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=loss_fields)
        writer.writeheader()
        for row in loss_rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in loss_fields})
    metrics_json = output_dir / "metrics.json"
    metrics = {
        "contract_version": TRAINING_READINESS_CONTRACT_VERSION,
        "trainer": "tcv_style_recurrent_actor_critic_v1",
        "algorithm": "tcv_mpo_recurrent_actor_critic_v1",
        "algorithm_note": "Feedforward stochastic actor with twin recurrent Q critics, complete-episode replay, masked sequence chunks, MPO sampled-action E-step, KL-constrained policy fitting, and deterministic actor mean for evaluation/export.",
        "total_steps": int(total_steps),
        "num_envs": int(cfg.num_envs),
        "sequence_length": int(cfg.sequence_length),
        "replay_episodes": int(replay_episodes),
        "replay_transitions": int(replay_transitions),
        "critic_updates": len(critic_losses),
        "actor_updates": len(actor_losses),
        "mpo_kl_dual_updates": len(mpo_kl_dual_losses),
        "last_critic_loss": float(critic_losses[-1]) if critic_losses else None,
        "last_actor_loss": float(actor_losses[-1]) if actor_losses else None,
        "last_mpo_temperature_loss": float(mpo_temperature_losses[-1]) if mpo_temperature_losses else None,
        "last_mpo_temperature": float(mpo_temperatures[-1]) if mpo_temperatures else None,
        "last_mpo_mean_kl": float(mpo_mean_kls[-1]) if mpo_mean_kls else None,
        "last_mpo_std_kl": float(mpo_std_kls[-1]) if mpo_std_kls else None,
        "last_mpo_kl_dual_loss": float(mpo_kl_dual_losses[-1]) if mpo_kl_dual_losses else None,
        "last_mpo_mean_kl_penalty": float(mpo_mean_kl_penalties[-1]) if mpo_mean_kl_penalties else None,
        "last_mpo_std_kl_penalty": float(mpo_std_kl_penalties[-1]) if mpo_std_kl_penalties else None,
        "mean_valid_steps_per_update": float(np.mean(valid_steps_per_update)) if valid_steps_per_update else 0.0,
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
    (output_dir / "config_snapshot.json").write_text(json.dumps(_checkpoint_safe_config(cfg), indent=2, sort_keys=True), encoding="utf-8")
    write_training_contract_artifacts(
        output_dir=output_dir,
        trainer_name="tcv_style_recurrent_actor_critic_v1",
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
        from tokamak_control.core.batched_gpu_simulator import batched_gpu_simulator_profiling_snapshot
        from tokamak_control.geometry.boundary import boundary_profiling_snapshot
    except Exception:
        return {"available": False}
    return {
        "available": True,
        "plasma_model": plasma_model_profiling_snapshot(),
        "gpu_plasma_model": gpu_plasma_model_profiling_snapshot(),
        "batched_gpu_simulator": batched_gpu_simulator_profiling_snapshot(),
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
    learner = max(float(learner_time_s), 1.0e-12)
    return {
        "total_elapsed_s": float(total_elapsed_s),
        "collection_time_s": float(collection_time_s),
        "actor_inference_time_s": float(actor_inference_time_s),
        "env_step_time_s": float(env_step_time_s),
        "replay_sampling_time_s": float(replay_sampling_time_s),
        "learner_time_s": float(learner_time_s),
        "evaluation_time_s": float(evaluation_time_s),
        "env_steps_per_second": float(total_steps) / elapsed,
        "learner_updates_per_second": float(update_count) / learner if int(update_count) > 0 else 0.0,
        "update_to_data_ratio": float(update_count) / max(float(total_steps), 1.0),
    }


def _checkpoint_safe_config(cfg: TCVStyleTrainerConfig) -> dict[str, object]:
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


def _eval_seed_base(cfg: TCVStyleTrainerConfig) -> int:
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


__all__ = ["TCVStyleTrainerConfig", "TCVStyleTrainingResult", "evaluate_tcv_actor", "evaluate_tcv_actor_detailed", "train_tcv_style_actor_critic"]
