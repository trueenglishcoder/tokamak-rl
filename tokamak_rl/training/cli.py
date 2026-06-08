from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from tokamak_rl.config import load_experiment_config
from tokamak_rl.env import ProcessTokamakEnv, TokamakRLEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.training.diagnostics import json_safe
from tokamak_rl.training.simple_actor_critic import SimpleTrainerConfig, train_simple_actor_critic
from tokamak_rl.training.tcv_style_actor_critic import TCVStyleTrainerConfig, train_tcv_style_actor_critic
from tokamak_rl.training.wandb_logging import WandBConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tokamak-rl training.")
    parser.add_argument("--trainer", choices=["tcv_style", "simple"], default=None, help="Trainer implementation to run.")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML config.")
    parser.add_argument("--steps", type=int, default=None, help="Total environment steps.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Random-action warmup steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Replay update batch size.")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence unroll length for the TCV-style recurrent critic trainer.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Actor/critic hidden width.")
    parser.add_argument("--critic-hidden-dim", type=int, default=None, help="Recurrent critic LSTM width for the TCV-style trainer.")
    parser.add_argument("--critic-mlp-hidden-dim", type=int, default=None, help="Recurrent critic MLP width for the TCV-style trainer.")
    parser.add_argument("--mpo-kl-lr", type=float, default=None, help="Optimizer learning rate for MPO KL dual variables.")
    parser.add_argument("--mpo-epsilon", type=float, default=None, help="MPO sampled-action E-step KL bound.")
    parser.add_argument("--mpo-mean-kl-epsilon", type=float, default=None, help="MPO actor mean KL bound.")
    parser.add_argument("--mpo-std-kl-epsilon", type=float, default=None, help="MPO actor standard-deviation KL bound.")
    parser.add_argument("--mpo-action-samples", type=int, default=None, help="Number of sampled actions per state for MPO policy improvement.")
    parser.add_argument("--mpo-temperature-iterations", type=int, default=None, help="Optimizer iterations for the MPO E-step temperature.")
    parser.add_argument("--mpo-temperature-lr", type=float, default=None, help="Optimizer learning rate for the MPO E-step temperature.")
    parser.add_argument("--mpo-initial-temperature", type=float, default=None, help="Initial MPO E-step temperature.")
    parser.add_argument("--mpo-initial-mean-kl-penalty", type=float, default=None, help="Initial positive penalty for actor mean KL constraint.")
    parser.add_argument("--mpo-initial-std-kl-penalty", type=float, default=None, help="Initial positive penalty for actor standard-deviation KL constraint.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None, help="Learner device for PyTorch models and updates.")
    parser.add_argument("--sim-compute-backend", choices=["cpu", "gpu"], default=None, help="Override tokamak-sim compute backend for environment stepping.")
    parser.add_argument("--sim-gpu-device", default=None, help="Override tokamak-sim GPU device when --sim-compute-backend gpu is used.")
    parser.add_argument("--process-envs", action="store_true", help="Run each training environment in its own simulator worker process.")
    parser.add_argument("--process-start-method", choices=["spawn", "fork", "forkserver"], default="spawn", help="Multiprocessing start method for --process-envs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint output directory.")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None, help="Optional interval for numbered step checkpoints and latest.pt updates.")
    parser.add_argument("--max-step-checkpoints", type=int, default=None, help="Optional retention limit for numbered step_*.pt checkpoints.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Optional checkpoint to resume actor/critic weights from.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for metrics.json, losses.csv, and config_snapshot.json.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of synchronous training environments.")
    parser.add_argument("--updates-per-episode", type=int, default=None, help="TCV-style sequence updates after each completed episode.")
    parser.add_argument("--updates-per-env-step", type=int, default=None, help="TCV-style sequence updates after each collected environment step when replay is nonempty.")
    parser.add_argument("--max-learner-catchup-updates", type=int, default=None, help="Optional cap on learner updates run by a single update trigger.")
    parser.add_argument("--eval-interval-steps", type=int, default=None, help="Optional periodic deterministic eval interval.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Deterministic eval episodes after training.")
    parser.add_argument("--eval-max-steps", type=int, default=None, help="Max steps per eval episode.")
    parser.add_argument("--eval-randomization-mode", choices=["configured", "clean"], default=None, help="Evaluation randomization mode; defaults to experiment YAML.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the terminal training progress bar.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for this run.")
    parser.add_argument("--wandb-project", default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb-name", default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--wandb-group", default=None, help="Optional Weights & Biases run group.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None, help="Weights & Biases mode.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Weights & Biases tag; can be provided multiple times.")
    parser.add_argument("--wandb-log-interval-steps", type=int, default=None, help="Environment-step interval for W&B scalar logging.")
    parser.add_argument("--wandb-no-artifacts", action="store_true", help="Disable W&B artifact upload for metrics/checkpoints/exports.")
    parser.add_argument("--no-export-best-actor", action="store_true", help="Disable automatic exports/best_actor creation for checkpointed real-env runs.")
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    if args.sim_compute_backend is not None or args.sim_gpu_device is not None:
        experiment = replace(
            experiment,
            env=replace(
                experiment.env,
                compute_backend=args.sim_compute_backend if args.sim_compute_backend is not None else experiment.env.compute_backend,
                gpu_device=args.sim_gpu_device if args.sim_gpu_device is not None else experiment.env.gpu_device,
            ),
        )
    trainer_name = args.trainer or experiment.training.trainer
    process_envs = bool(args.process_envs or experiment.training.process_envs)
    process_start_method = str(args.process_start_method or experiment.training.process_start_method)
    output_dir = args.output_dir if args.output_dir is not None else experiment.artifacts.output_dir
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else experiment.artifacts.checkpoint_dir
    checkpoint_interval_steps = args.checkpoint_interval_steps if args.checkpoint_interval_steps is not None else experiment.artifacts.checkpoint_interval_steps
    max_step_checkpoints = args.max_step_checkpoints if args.max_step_checkpoints is not None else experiment.artifacts.max_step_checkpoints
    export_best_actor = bool(experiment.artifacts.export_best_actor and not args.no_export_best_actor)
    run_metadata = _experiment_run_metadata(experiment=experiment, trainer_name=trainer_name)
    run_metadata["process_envs"] = process_envs
    run_metadata["process_start_method"] = process_start_method if process_envs else None
    eval_randomization_mode = args.eval_randomization_mode or experiment.evaluation.randomization_mode
    run_metadata["evaluation_randomization_mode"] = eval_randomization_mode
    wandb = _resolve_wandb_config(args=args, base=experiment.wandb, experiment_name=experiment.name)
    run_metadata["wandb"] = asdict(wandb)
    env_factory = _make_env_factory(experiment=experiment, process_envs=process_envs, process_start_method=process_start_method)
    eval_env_factory = _make_eval_env_factory(
        experiment=experiment,
        process_envs=process_envs,
        process_start_method=process_start_method,
        randomization_mode=eval_randomization_mode,
    )
    if trainer_name == "simple":
        cfg = SimpleTrainerConfig(
            total_steps=args.steps if args.steps is not None else experiment.training.total_steps,
            warmup_steps=args.warmup_steps if args.warmup_steps is not None else experiment.training.warmup_steps,
            batch_size=args.batch_size if args.batch_size is not None else experiment.training.batch_size,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else experiment.training.hidden_dim,
            seed=args.seed if args.seed is not None else experiment.training.seed,
            num_envs=args.num_envs if args.num_envs is not None else experiment.training.num_envs,
            eval_interval_steps=args.eval_interval_steps,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_steps=checkpoint_interval_steps,
            max_step_checkpoints=max_step_checkpoints,
            resume_checkpoint=args.resume_checkpoint,
            eval_episodes=args.eval_episodes if args.eval_episodes is not None else experiment.evaluation.episodes,
            eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else experiment.evaluation.max_steps,
            eval_seed=experiment.evaluation.validation_seed,
            device=args.device if args.device is not None else experiment.training.device,
            progress=not bool(args.no_progress),
            wandb=wandb,
            export_best_actor=export_best_actor,
            run_metadata=run_metadata,
        )
        result = train_simple_actor_critic(env_factory, cfg, eval_env_factory=eval_env_factory)
    else:
        hidden_dim = args.hidden_dim if args.hidden_dim is not None else experiment.training.hidden_dim
        cfg = TCVStyleTrainerConfig(
            total_steps=args.steps if args.steps is not None else experiment.training.total_steps,
            warmup_steps=args.warmup_steps if args.warmup_steps is not None else experiment.training.warmup_steps,
            batch_size=args.batch_size if args.batch_size is not None else experiment.training.batch_size,
            sequence_length=args.sequence_length if args.sequence_length is not None else experiment.training.sequence_length,
            actor_hidden_dim=hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim if args.critic_hidden_dim is not None else (experiment.training.critic_hidden_dim or hidden_dim),
            critic_mlp_hidden_dim=args.critic_mlp_hidden_dim if args.critic_mlp_hidden_dim is not None else (experiment.training.critic_mlp_hidden_dim or hidden_dim),
            mpo_kl_lr=args.mpo_kl_lr if args.mpo_kl_lr is not None else experiment.training.mpo_kl_lr,
            mpo_epsilon=args.mpo_epsilon if args.mpo_epsilon is not None else experiment.training.mpo_epsilon,
            mpo_mean_kl_epsilon=args.mpo_mean_kl_epsilon if args.mpo_mean_kl_epsilon is not None else experiment.training.mpo_mean_kl_epsilon,
            mpo_std_kl_epsilon=args.mpo_std_kl_epsilon if args.mpo_std_kl_epsilon is not None else experiment.training.mpo_std_kl_epsilon,
            mpo_action_samples=args.mpo_action_samples if args.mpo_action_samples is not None else experiment.training.mpo_action_samples,
            mpo_temperature_iterations=args.mpo_temperature_iterations if args.mpo_temperature_iterations is not None else experiment.training.mpo_temperature_iterations,
            mpo_temperature_lr=args.mpo_temperature_lr if args.mpo_temperature_lr is not None else experiment.training.mpo_temperature_lr,
            mpo_initial_temperature=args.mpo_initial_temperature if args.mpo_initial_temperature is not None else experiment.training.mpo_initial_temperature,
            mpo_initial_mean_kl_penalty=args.mpo_initial_mean_kl_penalty if args.mpo_initial_mean_kl_penalty is not None else experiment.training.mpo_initial_mean_kl_penalty,
            mpo_initial_std_kl_penalty=args.mpo_initial_std_kl_penalty if args.mpo_initial_std_kl_penalty is not None else experiment.training.mpo_initial_std_kl_penalty,
            seed=args.seed if args.seed is not None else experiment.training.seed,
            num_envs=args.num_envs if args.num_envs is not None else experiment.training.num_envs,
            updates_per_episode=args.updates_per_episode if args.updates_per_episode is not None else experiment.training.updates_per_episode,
            updates_per_env_step=args.updates_per_env_step if args.updates_per_env_step is not None else experiment.training.updates_per_env_step,
            max_learner_catchup_updates=args.max_learner_catchup_updates if args.max_learner_catchup_updates is not None else experiment.training.max_learner_catchup_updates,
            eval_interval_steps=args.eval_interval_steps,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_steps=checkpoint_interval_steps,
            max_step_checkpoints=max_step_checkpoints,
            resume_checkpoint=args.resume_checkpoint,
            eval_episodes=args.eval_episodes if args.eval_episodes is not None else experiment.evaluation.episodes,
            eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else experiment.evaluation.max_steps,
            eval_seed=experiment.evaluation.validation_seed,
            device=args.device if args.device is not None else experiment.training.device,
            progress=not bool(args.no_progress),
            wandb=wandb,
            export_best_actor=export_best_actor,
            run_metadata=run_metadata,
        )
        result = train_tcv_style_actor_critic(env_factory, cfg, eval_env_factory=eval_env_factory)
    if hasattr(result, "replay_episodes"):
        print(f"steps={result.total_steps}")
        print(f"replay_episodes={result.replay_episodes} replay_transitions={result.replay_transitions}")
    else:
        print(f"steps={result.total_steps} replay_size={result.replay_size}")
    if result.critic_losses:
        print(f"last_critic_loss={result.critic_losses[-1]:.6g}")
    if result.actor_losses:
        print(f"last_actor_loss={result.actor_losses[-1]:.6g}")
    print(f"eval_returns={result.eval_returns}")
    if result.checkpoint_path is not None:
        print(f"checkpoint={result.checkpoint_path}")
    if getattr(result, "latest_checkpoint_path", None) is not None:
        print(f"latest_checkpoint={result.latest_checkpoint_path}")
    if getattr(result, "best_checkpoint_path", None) is not None:
        print(f"best_checkpoint={result.best_checkpoint_path}")
    if getattr(result, "best_actor_export_dir", None) is not None:
        print(f"best_actor_export={result.best_actor_export_dir}")
    if result.metrics_json is not None:
        print(f"metrics={result.metrics_json}")
    return 0


def _experiment_run_metadata(*, experiment, trainer_name: str) -> dict[str, object]:
    return json_safe(
        {
            "experiment_name": experiment.name,
            "experiment_config_path": experiment.source_path,
            "trainer_name": trainer_name,
            "sim": asdict(experiment.env),
            "reward_config_path": experiment.reward_config_path,
            "randomization_config_path": experiment.randomization_config_path,
            "reward": asdict(experiment.reward),
            "randomization": asdict(experiment.randomization),
            "evaluation": asdict(experiment.evaluation),
        }
    )


def _resolve_wandb_config(*, args, base: WandBConfig, experiment_name: str) -> WandBConfig:
    name = args.wandb_name if args.wandb_name is not None else base.name
    if name is None and bool(args.wandb or base.enabled):
        name = experiment_name
    tags = base.tags if args.wandb_tag is None else tuple(str(tag) for tag in args.wandb_tag)
    return WandBConfig(
        enabled=bool(base.enabled or args.wandb),
        project=str(args.wandb_project if args.wandb_project is not None else base.project),
        entity=args.wandb_entity if args.wandb_entity is not None else base.entity,
        name=name,
        group=args.wandb_group if args.wandb_group is not None else base.group,
        mode=str(args.wandb_mode if args.wandb_mode is not None else base.mode),
        tags=tags,
        log_interval_steps=int(args.wandb_log_interval_steps if args.wandb_log_interval_steps is not None else base.log_interval_steps),
        log_artifacts=bool(base.log_artifacts and not args.wandb_no_artifacts),
    )


def _make_env_factory(*, experiment, process_envs: bool, process_start_method: str):
    if bool(process_envs):
        return lambda: ProcessTokamakEnv(
            experiment.env,
            reward_fn=experiment.reward,
            randomizer=experiment.randomization,
            start_method=process_start_method,
        )
    return lambda: TokamakRLEnv(experiment.env, reward_fn=experiment.reward, randomizer=experiment.randomization)


def _make_eval_env_factory(*, experiment, process_envs: bool, process_start_method: str, randomization_mode: str):
    if randomization_mode == "configured":
        return _make_env_factory(experiment=experiment, process_envs=process_envs, process_start_method=process_start_method)
    if randomization_mode != "clean":
        raise ValueError("evaluation randomization mode must be one of: configured, clean")
    clean_env = replace(experiment.env, realism_enabled=False)
    clean_randomizer = DomainRandomizer(enabled=False)
    if bool(process_envs):
        return lambda: ProcessTokamakEnv(
            clean_env,
            reward_fn=experiment.reward,
            randomizer=clean_randomizer,
            start_method=process_start_method,
        )
    return lambda: TokamakRLEnv(clean_env, reward_fn=experiment.reward, randomizer=clean_randomizer)


if __name__ == "__main__":  # pragma: no cover - exercised by shell smoke checks.
    raise SystemExit(main())
