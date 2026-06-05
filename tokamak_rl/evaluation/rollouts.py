from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from tokamak_rl.env import EnvConfig, TokamakRLEnv
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward


PolicyName = Literal["zero", "random"]


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Evaluation rollout settings for non-training policy checks."""

    env: EnvConfig
    episodes: int = 1
    policy: PolicyName = "zero"
    seed: int = 0
    output_dir: Path | None = None
    reward: JointCurrentBoundaryReward = field(default_factory=JointCurrentBoundaryReward)
    randomizer: DomainRandomizer = field(default_factory=DomainRandomizer)


@dataclass(frozen=True, slots=True)
class RolloutOutputs:
    """Paths written by a rollout evaluation."""

    output_dir: Path
    summary_json: Path
    episode_metrics_csv: Path
    rollouts_npz: Path


def run_rollout_evaluation(cfg: RolloutConfig) -> dict[str, object]:
    """Run deterministic rollout evaluation and optionally write artifacts."""
    if int(cfg.episodes) <= 0:
        raise ValueError("episodes must be > 0")
    rng = np.random.default_rng(int(cfg.seed))
    episode_metrics: list[dict[str, object]] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_observations: list[np.ndarray] = []
    all_terminated: list[np.ndarray] = []
    all_truncated: list[np.ndarray] = []
    all_true_ip: list[np.ndarray] = []
    all_measured_ip: list[np.ndarray] = []
    all_ip_ref: list[np.ndarray] = []
    all_radii_ref: list[np.ndarray] = []
    all_boundary_found: list[np.ndarray] = []
    all_measured_boundary_available: list[np.ndarray] = []

    for episode_index in range(int(cfg.episodes)):
        env = TokamakRLEnv(cfg.env, reward_fn=cfg.reward, randomizer=cfg.randomizer)
        obs, reset_info = env.reset(seed=int(cfg.seed) + episode_index)
        episode_metadata = dict(reset_info.get("episode_metadata", {}))
        randomization_metadata = _metadata_mapping(episode_metadata, "randomization")
        observations = [np.asarray(obs, dtype=float)]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        true_ip: list[float] = []
        measured_ip: list[float] = []
        ip_ref: list[float] = []
        radii_ref: list[np.ndarray] = []
        boundary_found: list[bool] = []
        measured_boundary_available: list[bool] = []
        termination_reason: str | None = None
        for _step_index in range(int(cfg.env.max_episode_steps)):
            action = _policy_action(cfg.policy, action_dim=env.action_dim, rng=rng)
            obs, reward, terminated, truncated, info = env.step(action)
            snapshot = info["snapshot"]
            actions.append(np.asarray(info["action_norm"], dtype=float).copy())
            rewards.append(float(reward))
            observations.append(np.asarray(obs, dtype=float).copy())
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            true_ip.append(float(snapshot.true_ip))
            measured_ip.append(float(snapshot.measured_ip))
            ip_ref.append(float(snapshot.reference.ip_ref))
            radii_ref.append(np.asarray(snapshot.reference.radii_ref, dtype=float).reshape(-1).copy())
            boundary_found.append(bool(snapshot.boundary_found))
            measured_boundary_available.append(snapshot.measured_boundary_poly is not None and snapshot.measured_radii is not None)
            if terminated or truncated:
                termination_reason = info.get("termination_reason")
                break
        env.close()
        rewards_arr = np.asarray(rewards, dtype=float)
        boundary_found_arr = np.asarray(boundary_found, dtype=bool)
        measured_boundary_available_arr = np.asarray(measured_boundary_available, dtype=bool)
        episode_metrics.append(
            {
                "episode": episode_index,
                "return": float(np.sum(rewards_arr)) if rewards_arr.size else 0.0,
                "length": int(rewards_arr.size),
                "terminated": bool(terminated_flags[-1]) if terminated_flags else False,
                "truncated": bool(truncated_flags[-1]) if truncated_flags else False,
                "termination_reason": termination_reason or "",
                "boundary_failure_steps": int(np.count_nonzero(~boundary_found_arr)),
                "measured_boundary_missing_steps": int(np.count_nonzero(~measured_boundary_available_arr)),
                "reference_resampling_enabled": bool(episode_metadata.get("reference_resampling_enabled", False)),
                "reference_episode_seed": _metadata_int(episode_metadata, "reference_episode_seed"),
                "reference_effective_seed": _metadata_int(episode_metadata, "reference_effective_seed"),
                "reference_effective_ip_seed": _metadata_int(episode_metadata, "reference_effective_ip_seed"),
                "randomization_enabled": bool(randomization_metadata.get("enabled", False)),
                "randomization_seed": _metadata_int(randomization_metadata, "seed"),
            }
        )
        all_actions.append(_stack_or_empty(actions, trailing_dim=env.action_dim))
        all_rewards.append(rewards_arr)
        all_observations.append(_stack_or_empty(observations, trailing_dim=env.obs_dim))
        all_terminated.append(np.asarray(terminated_flags, dtype=bool))
        all_truncated.append(np.asarray(truncated_flags, dtype=bool))
        all_true_ip.append(np.asarray(true_ip, dtype=float))
        all_measured_ip.append(np.asarray(measured_ip, dtype=float))
        all_ip_ref.append(np.asarray(ip_ref, dtype=float))
        all_radii_ref.append(_stack_reference_radii(radii_ref, n_angles=int(cfg.env.angles)))
        all_boundary_found.append(boundary_found_arr)
        all_measured_boundary_available.append(measured_boundary_available_arr)

    summary = _summary(policy=cfg.policy, seed=int(cfg.seed), episode_metrics=episode_metrics)
    result: dict[str, object] = {
        "summary": summary,
        "episode_metrics": episode_metrics,
        "actions": all_actions,
        "rewards": all_rewards,
        "observations": all_observations,
        "terminated": all_terminated,
        "truncated": all_truncated,
        "true_ip": all_true_ip,
        "measured_ip": all_measured_ip,
        "ip_ref": all_ip_ref,
        "radii_ref": all_radii_ref,
        "boundary_found": all_boundary_found,
        "measured_boundary_available": all_measured_boundary_available,
    }
    if cfg.output_dir is not None:
        outputs = write_rollout_outputs(result, output_dir=cfg.output_dir)
        result["outputs"] = outputs
    return result


def write_rollout_outputs(result: dict[str, object], *, output_dir: Path) -> RolloutOutputs:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    metrics_csv = output_dir / "episode_metrics.csv"
    rollouts_npz = output_dir / "rollouts.npz"
    summary_json.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    _write_metrics_csv(metrics_csv, result["episode_metrics"])
    np.savez(
        rollouts_npz,
        actions=_pad_3d(result["actions"]),
        rewards=_pad_2d(result["rewards"]),
        observations=_pad_3d(result["observations"]),
        terminated=_pad_bool_2d(result["terminated"]),
        truncated=_pad_bool_2d(result["truncated"]),
        true_ip=_pad_2d(result["true_ip"]),
        measured_ip=_pad_2d(result["measured_ip"]),
        ip_ref=_pad_2d(result["ip_ref"]),
        radii_ref=_pad_3d(result["radii_ref"]),
        boundary_found=_pad_bool_2d(result["boundary_found"]),
        measured_boundary_available=_pad_bool_2d(result["measured_boundary_available"]),
        mask=_mask_2d(result["rewards"]),
    )
    return RolloutOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        episode_metrics_csv=metrics_csv,
        rollouts_npz=rollouts_npz,
    )


def _policy_action(policy: PolicyName, *, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    if policy == "zero":
        return np.zeros((int(action_dim),), dtype=float)
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=(int(action_dim),))
    raise ValueError(f"Unknown rollout policy: {policy!r}")


def _summary(*, policy: str, seed: int, episode_metrics: list[dict[str, object]]) -> dict[str, object]:
    returns = np.asarray([m["return"] for m in episode_metrics], dtype=float)
    lengths = np.asarray([m["length"] for m in episode_metrics], dtype=float)
    terminated = np.asarray([m["terminated"] for m in episode_metrics], dtype=bool)
    total_steps = int(np.sum(lengths)) if lengths.size else 0
    boundary_failure_steps = int(np.sum([m["boundary_failure_steps"] for m in episode_metrics])) if episode_metrics else 0
    measured_boundary_missing_steps = (
        int(np.sum([m["measured_boundary_missing_steps"] for m in episode_metrics])) if episode_metrics else 0
    )
    return {
        "policy": policy,
        "seed": int(seed),
        "episodes": len(episode_metrics),
        "mean_return": float(np.mean(returns)) if returns.size else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "termination_rate": float(np.mean(terminated)) if terminated.size else 0.0,
        "boundary_failure_rate": float(boundary_failure_steps / total_steps) if total_steps else 0.0,
        "measured_boundary_missing_rate": float(measured_boundary_missing_steps / total_steps) if total_steps else 0.0,
    }


def _write_metrics_csv(path: Path, metrics: object) -> None:
    rows = list(metrics)
    fieldnames = [
        "episode",
        "return",
        "length",
        "terminated",
        "truncated",
        "termination_reason",
        "boundary_failure_steps",
        "measured_boundary_missing_steps",
        "reference_resampling_enabled",
        "reference_episode_seed",
        "reference_effective_seed",
        "reference_effective_ip_seed",
        "randomization_enabled",
        "randomization_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _metadata_mapping(metadata: dict[str, object], key: str) -> dict[str, object]:
    value = metadata.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _stack_reference_radii(values: list[np.ndarray], *, n_angles: int) -> np.ndarray:
    if not values:
        return np.zeros((0, int(n_angles)), dtype=float)
    checked: list[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape != (int(n_angles),):
            raise ValueError(f"radii_ref shape must be ({int(n_angles)},), got {arr.shape}")
        checked.append(arr)
    return np.stack(checked, axis=0).astype(float, copy=False)


def _stack_or_empty(values: list[np.ndarray], *, trailing_dim: int) -> np.ndarray:
    if not values:
        return np.zeros((0, int(trailing_dim)), dtype=float)
    return np.stack(values, axis=0).astype(float, copy=False)


def _pad_2d(arrays: object) -> np.ndarray:
    seq = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    max_len = max((a.shape[0] for a in seq), default=0)
    out = np.full((len(seq), max_len), np.nan, dtype=float)
    for i, arr in enumerate(seq):
        out[i, : arr.shape[0]] = arr
    return out


def _pad_bool_2d(arrays: object) -> np.ndarray:
    seq = [np.asarray(a, dtype=bool).reshape(-1) for a in arrays]
    max_len = max((a.shape[0] for a in seq), default=0)
    out = np.zeros((len(seq), max_len), dtype=bool)
    for i, arr in enumerate(seq):
        out[i, : arr.shape[0]] = arr
    return out


def _mask_2d(arrays: object) -> np.ndarray:
    seq = [np.asarray(a).reshape(-1) for a in arrays]
    max_len = max((a.shape[0] for a in seq), default=0)
    out = np.zeros((len(seq), max_len), dtype=bool)
    for i, arr in enumerate(seq):
        out[i, : arr.shape[0]] = True
    return out


def _pad_3d(arrays: object) -> np.ndarray:
    seq = [np.asarray(a, dtype=float) for a in arrays]
    max_len = max((a.shape[0] for a in seq), default=0)
    trailing = max((a.shape[1] if a.ndim == 2 else 0 for a in seq), default=0)
    out = np.full((len(seq), max_len, trailing), np.nan, dtype=float)
    for i, arr in enumerate(seq):
        if arr.ndim != 2:
            raise ValueError("rollout arrays must be 2D")
        out[i, : arr.shape[0], : arr.shape[1]] = arr
    return out
