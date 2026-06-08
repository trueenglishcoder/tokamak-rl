from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

try:  # pragma: no cover - PyYAML is a normal tokamak-rl dependency.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from tokamak_rl.config import load_experiment_config
from tokamak_rl.rewards import JointCurrentBoundaryReward
from tokamak_rl.training.cli import _experiment_run_metadata, _make_env_factory, _make_eval_env_factory
from tokamak_rl.training.diagnostics import json_safe
from tokamak_rl.training.simple_actor_critic import SimpleTrainerConfig, train_simple_actor_critic
from tokamak_rl.training.tcv_style_actor_critic import TCVStyleTrainerConfig, train_tcv_style_actor_critic
from tokamak_rl.training.wandb_logging import WandBConfig


REWARD_FIELDS = (
    "ip_weight",
    "shape_weight",
    "action_weight",
    "delta_action_weight",
    "termination_penalty",
    "ip_tolerance_norm",
    "shape_tolerance_norm",
    "current_limit_weight",
    "derivative_limit_weight",
)


@dataclass(frozen=True, slots=True)
class RewardCandidate:
    index: int
    reward: JointCurrentBoundaryReward


@dataclass(frozen=True, slots=True)
class RewardSearchScoreWeights:
    ip_error: float = 1.0
    shape_error: float = 1.0
    boundary_failure: float = 10.0
    missing_metric: float = 100.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search reward values with short tokamak-rl training runs.")
    parser.add_argument("--config", required=True, type=Path, help="Base experiment YAML config.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward_search"), help="Directory for candidate runs and ranking artifacts.")
    parser.add_argument("--trainer", choices=["tcv_style", "simple"], default=None, help="Trainer override; defaults to experiment YAML.")
    parser.add_argument("--steps", type=int, default=None, help="Training steps per candidate.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps per candidate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per candidate.")
    parser.add_argument("--sequence-length", type=int, default=None, help="TCV-style sequence length.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Actor hidden width.")
    parser.add_argument("--critic-hidden-dim", type=int, default=None, help="TCV-style recurrent critic hidden width.")
    parser.add_argument("--critic-mlp-hidden-dim", type=int, default=None, help="TCV-style critic MLP hidden width.")
    parser.add_argument("--mpo-action-samples", type=int, default=None, help="TCV-style MPO action samples.")
    parser.add_argument("--mpo-temperature-iterations", type=int, default=None, help="TCV-style MPO temperature iterations.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of synchronous environments.")
    parser.add_argument("--updates-per-episode", type=int, default=None, help="TCV-style updates per completed episode.")
    parser.add_argument("--updates-per-env-step", type=int, default=None, help="TCV-style updates per environment step.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None, help="Learner device.")
    parser.add_argument("--sim-compute-backend", choices=["cpu", "gpu"], default=None, help="Override tokamak-sim compute backend for environment stepping.")
    parser.add_argument("--sim-gpu-device", default=None, help="Override tokamak-sim GPU device when --sim-compute-backend gpu is used.")
    parser.add_argument("--seed", type=int, default=None, help="Base trainer seed. Candidate index is added to this seed.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes per candidate.")
    parser.add_argument("--eval-max-steps", type=int, default=None, help="Max evaluation steps per episode.")
    parser.add_argument("--eval-randomization-mode", choices=["configured", "clean"], default=None, help="Evaluation randomization mode.")
    parser.add_argument("--process-envs", action="store_true", help="Run each env in a worker process.")
    parser.add_argument("--process-start-method", choices=["spawn", "fork", "forkserver"], default=None, help="Process start method for --process-envs.")
    parser.add_argument("--progress", action="store_true", help="Show per-candidate trainer progress bars.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for candidate runs.")
    parser.add_argument("--wandb-project", default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb-name", default=None, help="Base W&B run name; candidate index is appended.")
    parser.add_argument("--wandb-group", default=None, help="W&B group for all candidates in this reward search.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None, help="Weights & Biases mode.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Weights & Biases tag; can be provided multiple times.")
    parser.add_argument("--wandb-log-interval-steps", type=int, default=None, help="Environment-step interval for W&B scalar logging.")
    parser.add_argument("--wandb-no-artifacts", action="store_true", help="Disable W&B artifact upload for candidate metrics/checkpoints/exports.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Write candidate checkpoints. Disabled by default to save disk.")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None, help="Optional candidate checkpoint interval when --save-checkpoints is set.")
    parser.add_argument("--max-step-checkpoints", type=int, default=1, help="Candidate numbered checkpoint retention when --save-checkpoints is set.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Maximum candidates to run. Use 0 to run the full Cartesian grid.")
    parser.add_argument("--search-seed", type=int, default=0, help="Seed used when subsampling the candidate grid.")
    parser.add_argument("--dry-run", action="store_true", help="Write candidate reward files and manifest without running training.")
    parser.add_argument("--ip-weight-values", default="0.5,1.0,2.0", help="Comma-separated candidate values.")
    parser.add_argument("--shape-weight-values", default="0.5,1.0,2.0", help="Comma-separated candidate values.")
    parser.add_argument("--action-weight-values", default="0.001,0.01", help="Comma-separated candidate values.")
    parser.add_argument("--delta-action-weight-values", default="0.001,0.01", help="Comma-separated candidate values.")
    parser.add_argument("--termination-penalty-values", default="5.0,10.0,20.0", help="Comma-separated candidate values.")
    parser.add_argument("--ip-tolerance-values", default=None, help="Comma-separated candidate values. Defaults to the base reward value.")
    parser.add_argument("--shape-tolerance-values", default=None, help="Comma-separated candidate values. Defaults to the base reward value.")
    parser.add_argument("--current-limit-weight-values", default=None, help="Comma-separated candidate values. Defaults to the base reward value.")
    parser.add_argument("--derivative-limit-weight-values", default=None, help="Comma-separated candidate values. Defaults to the base reward value.")
    parser.add_argument("--score-ip-weight", type=float, default=1.0, help="Ranking weight for eval mean Ip error.")
    parser.add_argument("--score-shape-weight", type=float, default=1.0, help="Ranking weight for eval mean boundary-shape error.")
    parser.add_argument("--score-boundary-failure-weight", type=float, default=10.0, help="Ranking weight for eval boundary failure rate.")
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates(
        base_reward=experiment.reward,
        search_space=_search_space_from_args(args, experiment.reward),
        max_candidates=int(args.max_candidates),
        seed=int(args.search_seed),
    )
    weights = RewardSearchScoreWeights(
        ip_error=float(args.score_ip_weight),
        shape_error=float(args.score_shape_weight),
        boundary_failure=float(args.score_boundary_failure_weight),
    )
    manifest = {
        "base_config": str(Path(args.config).expanduser().resolve()),
        "trainer": trainer_name,
        "candidate_count": len(candidates),
        "search_space": {key: list(values) for key, values in _search_space_from_args(args, experiment.reward).items()},
        "score_weights": asdict(weights),
    }
    (output_dir / "search_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_dir = output_dir / f"candidate_{candidate.index:04d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        reward_path = candidate_dir / "reward.yaml"
        write_reward_config(reward_path, candidate.reward)
        if args.dry_run:
            row = candidate_row(candidate=candidate, candidate_dir=candidate_dir, status="dry_run")
            rows.append(row)
            _write_rankings(output_dir, rows)
            continue
        print(f"[{candidate.index + 1}/{len(candidates)}] reward={compact_reward(candidate.reward)}", flush=True)
        try:
            metrics = run_candidate(
                experiment=replace(experiment, reward=candidate.reward),
                trainer_name=trainer_name,
                args=args,
                candidate=candidate,
                candidate_dir=candidate_dir,
            )
            score = score_metrics(metrics, weights=weights)
            row = candidate_row(candidate=candidate, candidate_dir=candidate_dir, status="ok", metrics=metrics, score=score)
        except Exception as exc:  # noqa: BLE001 - failed candidates should not kill the whole search.
            row = candidate_row(candidate=candidate, candidate_dir=candidate_dir, status="failed", error=str(exc))
        rows.append(row)
        _write_rankings(output_dir, rows)
    best = first_ok(rows)
    if best is not None:
        best_reward = JointCurrentBoundaryReward(**{field: float(best[field]) for field in REWARD_FIELDS})
        write_reward_config(output_dir / "best_reward.yaml", best_reward)
        print(f"best_candidate={best['candidate']} score={best['score']} reward={output_dir / 'best_reward.yaml'}", flush=True)
    print(f"results={output_dir / 'results.csv'}", flush=True)
    return 0


def generate_candidates(
    *,
    base_reward: JointCurrentBoundaryReward,
    search_space: Mapping[str, Iterable[float]],
    max_candidates: int,
    seed: int,
) -> list[RewardCandidate]:
    keys = list(REWARD_FIELDS)
    values = [tuple(float(v) for v in search_space.get(key, (getattr(base_reward, key),))) for key in keys]
    combos = list(itertools.product(*values))
    if int(max_candidates) > 0 and len(combos) > int(max_candidates):
        rng = np.random.default_rng(int(seed))
        keep = sorted(int(i) for i in rng.choice(len(combos), size=int(max_candidates), replace=False))
        combos = [combos[i] for i in keep]
    candidates: list[RewardCandidate] = []
    for index, combo in enumerate(combos):
        params = dict(zip(keys, combo, strict=True))
        candidates.append(RewardCandidate(index=index, reward=replace(base_reward, **params)))
    return candidates


def run_candidate(*, experiment, trainer_name: str, args, candidate: RewardCandidate, candidate_dir: Path) -> dict[str, object]:
    requested_num_envs = int(args.num_envs if args.num_envs is not None else experiment.training.num_envs)
    requested_process_envs = bool(args.process_envs or experiment.training.process_envs)
    gpu_pool_active = bool(experiment.env.compute_backend == "gpu")
    process_envs = bool(requested_process_envs and not gpu_pool_active)
    process_start_method = str(args.process_start_method or experiment.training.process_start_method)
    eval_randomization_mode = args.eval_randomization_mode or experiment.evaluation.randomization_mode
    env_factory = _make_env_factory(experiment=experiment, process_envs=process_envs, process_start_method=process_start_method, num_envs=requested_num_envs)
    eval_env_factory = _make_eval_env_factory(
        experiment=experiment,
        process_envs=process_envs,
        process_start_method=process_start_method,
        randomization_mode=eval_randomization_mode,
        num_envs=1,
    )
    run_metadata = _experiment_run_metadata(experiment=experiment, trainer_name=trainer_name)
    run_metadata["reward_search"] = {
        "candidate": candidate.index,
        "candidate_dir": str(candidate_dir),
        "eval_randomization_mode": eval_randomization_mode,
        "reward": asdict(candidate.reward),
        "requested_process_envs": requested_process_envs,
        "process_envs": process_envs,
        "gpu_env_pool_active": gpu_pool_active,
    }
    wandb = _candidate_wandb_config(args=args, experiment_name=experiment.name, candidate=candidate)
    run_metadata["wandb"] = asdict(wandb)
    checkpoint_dir = candidate_dir / "checkpoints" if bool(args.save_checkpoints) else None
    if trainer_name == "simple":
        cfg = SimpleTrainerConfig(
            total_steps=args.steps if args.steps is not None else experiment.training.total_steps,
            warmup_steps=args.warmup_steps if args.warmup_steps is not None else experiment.training.warmup_steps,
            batch_size=args.batch_size if args.batch_size is not None else experiment.training.batch_size,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else experiment.training.hidden_dim,
            seed=(args.seed if args.seed is not None else experiment.training.seed) + candidate.index,
            num_envs=requested_num_envs,
            output_dir=candidate_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_steps=args.checkpoint_interval_steps if bool(args.save_checkpoints) else None,
            max_step_checkpoints=args.max_step_checkpoints if bool(args.save_checkpoints) else None,
            eval_episodes=args.eval_episodes if args.eval_episodes is not None else experiment.evaluation.episodes,
            eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else experiment.evaluation.max_steps,
            eval_seed=experiment.evaluation.validation_seed,
            device=args.device if args.device is not None else experiment.training.device,
            progress=bool(args.progress),
            wandb=wandb,
            export_best_actor=False,
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
            mpo_action_samples=args.mpo_action_samples if args.mpo_action_samples is not None else experiment.training.mpo_action_samples,
            mpo_temperature_iterations=args.mpo_temperature_iterations if args.mpo_temperature_iterations is not None else experiment.training.mpo_temperature_iterations,
            seed=(args.seed if args.seed is not None else experiment.training.seed) + candidate.index,
            num_envs=requested_num_envs,
            updates_per_episode=args.updates_per_episode if args.updates_per_episode is not None else experiment.training.updates_per_episode,
            updates_per_env_step=args.updates_per_env_step if args.updates_per_env_step is not None else experiment.training.updates_per_env_step,
            output_dir=candidate_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_steps=args.checkpoint_interval_steps if bool(args.save_checkpoints) else None,
            max_step_checkpoints=args.max_step_checkpoints if bool(args.save_checkpoints) else None,
            eval_episodes=args.eval_episodes if args.eval_episodes is not None else experiment.evaluation.episodes,
            eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else experiment.evaluation.max_steps,
            eval_seed=experiment.evaluation.validation_seed,
            device=args.device if args.device is not None else experiment.training.device,
            progress=bool(args.progress),
            wandb=wandb,
            export_best_actor=False,
            run_metadata=run_metadata,
        )
        result = train_tcv_style_actor_critic(env_factory, cfg, eval_env_factory=eval_env_factory)
    if result.metrics_json is None:
        raise RuntimeError("candidate training did not write metrics.json")
    return json.loads(Path(result.metrics_json).read_text(encoding="utf-8"))


def score_metrics(metrics: Mapping[str, object], *, weights: RewardSearchScoreWeights = RewardSearchScoreWeights()) -> float:
    diagnostics = metrics.get("eval_tracking_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    ip_error = _metric_or_penalty(diagnostics.get("mean_ip_error_norm"), weights.missing_metric)
    shape_error = _metric_or_penalty(diagnostics.get("mean_shape_error_norm"), weights.missing_metric)
    boundary_failure = _metric_or_penalty(diagnostics.get("boundary_failure_rate"), weights.missing_metric)
    return float(weights.ip_error) * ip_error + float(weights.shape_error) * shape_error + float(weights.boundary_failure) * boundary_failure


def candidate_row(
    *,
    candidate: RewardCandidate,
    candidate_dir: Path,
    status: str,
    metrics: Mapping[str, object] | None = None,
    score: float | None = None,
    error: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate": int(candidate.index),
        "status": status,
        "score": "" if score is None else float(score),
        "candidate_dir": str(candidate_dir),
        "error": "" if error is None else error,
        **asdict(candidate.reward),
    }
    if metrics is not None:
        diagnostics = metrics.get("eval_tracking_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        eval_returns = metrics.get("eval_returns")
        row.update(
            {
                "eval_mean_ip_error_norm": _empty_if_none(diagnostics.get("mean_ip_error_norm")),
                "eval_mean_shape_error_norm": _empty_if_none(diagnostics.get("mean_shape_error_norm")),
                "eval_boundary_failure_rate": _empty_if_none(diagnostics.get("boundary_failure_rate")),
                "eval_mean_return": _mean(eval_returns) if isinstance(eval_returns, list) else "",
                "critic_updates": metrics.get("critic_updates", ""),
                "actor_updates": metrics.get("actor_updates", ""),
                "total_steps": metrics.get("total_steps", ""),
            }
        )
    return row


def write_reward_config(path: Path, reward: JointCurrentBoundaryReward) -> None:
    data = asdict(reward)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def compact_reward(reward: JointCurrentBoundaryReward) -> str:
    return ", ".join(f"{field}={getattr(reward, field):g}" for field in REWARD_FIELDS)


def _candidate_wandb_config(*, args, experiment_name: str, candidate: RewardCandidate) -> WandBConfig:
    enabled = bool(args.wandb)
    base_name = args.wandb_name or f"reward_search_{experiment_name}"
    group = args.wandb_group or base_name
    return WandBConfig(
        enabled=enabled,
        project=str(args.wandb_project or "tokamak-rl"),
        entity=args.wandb_entity,
        name=f"{base_name}_candidate_{candidate.index:04d}" if enabled else None,
        group=group if enabled else None,
        mode=str(args.wandb_mode or "online"),
        tags=tuple(str(tag) for tag in (args.wandb_tag or ())),
        log_interval_steps=int(args.wandb_log_interval_steps or 1),
        log_artifacts=not bool(args.wandb_no_artifacts),
    )


def parse_float_values(raw: str | None, *, default: float) -> tuple[float, ...]:
    if raw is None:
        return (float(default),)
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("candidate value lists must contain at least one number")
    return values


def first_ok(rows: list[dict[str, object]]) -> dict[str, object] | None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("score") != ""]
    if not ok:
        return None
    return sorted(ok, key=lambda row: float(row["score"]))[0]


def _search_space_from_args(args, base_reward: JointCurrentBoundaryReward) -> dict[str, tuple[float, ...]]:
    return {
        "ip_weight": parse_float_values(args.ip_weight_values, default=base_reward.ip_weight),
        "shape_weight": parse_float_values(args.shape_weight_values, default=base_reward.shape_weight),
        "action_weight": parse_float_values(args.action_weight_values, default=base_reward.action_weight),
        "delta_action_weight": parse_float_values(args.delta_action_weight_values, default=base_reward.delta_action_weight),
        "termination_penalty": parse_float_values(args.termination_penalty_values, default=base_reward.termination_penalty),
        "ip_tolerance_norm": parse_float_values(args.ip_tolerance_values, default=base_reward.ip_tolerance_norm),
        "shape_tolerance_norm": parse_float_values(args.shape_tolerance_values, default=base_reward.shape_tolerance_norm),
        "current_limit_weight": parse_float_values(args.current_limit_weight_values, default=base_reward.current_limit_weight),
        "derivative_limit_weight": parse_float_values(args.derivative_limit_weight_values, default=base_reward.derivative_limit_weight),
    }


def _write_rankings(output_dir: Path, rows: list[dict[str, object]]) -> None:
    ranked = sorted(rows, key=lambda row: (row.get("status") != "ok", float(row["score"]) if row.get("score") != "" else float("inf"), int(row["candidate"])))
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "candidate",
        "status",
        "score",
        "eval_mean_ip_error_norm",
        "eval_mean_shape_error_norm",
        "eval_boundary_failure_rate",
        "eval_mean_return",
        "critic_updates",
        "actor_updates",
        "total_steps",
        *REWARD_FIELDS,
        "candidate_dir",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    (output_dir / "results.json").write_text(json.dumps(json_safe(ranked), indent=2, sort_keys=True), encoding="utf-8")


def _metric_or_penalty(value: object, penalty: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(penalty)
    return out if np.isfinite(out) else float(penalty)


def _empty_if_none(value: object) -> object:
    return "" if value is None else value


def _mean(values: list[object]) -> float | str:
    floats = []
    for value in values:
        try:
            floats.append(float(value))
        except (TypeError, ValueError):
            pass
    return float(np.mean(floats)) if floats else ""


__all__ = [
    "RewardCandidate",
    "RewardSearchScoreWeights",
    "candidate_row",
    "compact_reward",
    "generate_candidates",
    "main",
    "parse_float_values",
    "score_metrics",
    "write_reward_config",
]
