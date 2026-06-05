from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tokamak_rl.contracts import CONDITIONAL_TRAINING_ARTIFACTS, REQUIRED_TRAINING_ARTIFACTS, TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.training.diagnostics import json_safe


class RewardComponentWriter:
    """Streaming writer for per-step reward component diagnostics."""

    def __init__(self, output_dir: Path | None) -> None:
        self.path: Path | None = None
        self._file = None
        self._writer: csv.DictWriter | None = None
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.path = out / "reward_components.csv"
            self._file = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=["step", "env_index", "episode", "component", "value"],
            )
            self._writer.writeheader()

    def record(self, *, step: int, env_index: int, episode: int, components: object) -> None:
        if self._writer is None or not isinstance(components, dict):
            return
        for name in sorted(components):
            value = components[name]
            if isinstance(value, (int, float)):
                self._writer.writerow(
                    {
                        "step": int(step),
                        "env_index": int(env_index),
                        "episode": int(episode),
                        "component": str(name),
                        "value": float(value),
                    }
                )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


def write_training_contract_artifacts(
    *,
    output_dir: Path,
    trainer_name: str,
    episode_returns: list[float],
    episode_lengths: list[int],
    eval_history: list[dict[str, object]],
    episode_records: list[dict[str, object]] | None = None,
    reference_records: list[dict[str, object]] | None = None,
    best_actor_export_dir: Path | None = None,
) -> Path:
    """Write required training-run artifact contract files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if episode_records is None:
        episode_records = _legacy_episode_records(episode_returns=episode_returns, episode_lengths=episode_lengths)
    if reference_records is None:
        reference_records = []
    _write_episodes_csv(output_dir / "episodes.csv", episode_records=episode_records)
    _write_eval_history_csv(output_dir / "eval_history.csv", eval_history=eval_history)
    _write_reward_components_csv(output_dir / "reward_components.csv")
    _write_reference_samples_npz(output_dir / "reference_samples.npz", reference_records=reference_records)
    _write_termination_counts_json(output_dir / "termination_counts.json", episode_records=episode_records)
    _write_eval_rollout_artifacts(output_dir / "rollouts", eval_history=eval_history)
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "contract_version": TRAINING_READINESS_CONTRACT_VERSION,
        "trainer": trainer_name,
        "required_artifacts": list(REQUIRED_TRAINING_ARTIFACTS),
        "conditional_artifacts": list(CONDITIONAL_TRAINING_ARTIFACTS),
        "present_artifacts": {
            name: True if name == "artifact_manifest.json" else (output_dir / name).exists()
            for name in REQUIRED_TRAINING_ARTIFACTS
        },
        "present_conditional_artifacts": {
            "exports/best_actor/": bool(best_actor_export_dir is not None and Path(best_actor_export_dir).exists()),
            "rollouts/": (output_dir / "rollouts").exists(),
        },
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def eval_history_with_final(
    eval_history: list[dict[str, object]],
    *,
    total_steps: int,
    eval_returns: list[float],
    eval_tracking_diagnostics: dict[str, object],
) -> list[dict[str, object]]:
    """Return eval history for artifacts, ensuring the final deterministic eval is represented."""
    rows = [dict(row) for row in eval_history]
    if any(_same_step(row.get("step"), total_steps) for row in rows):
        return rows
    final_mean = float(np.mean(np.asarray(eval_returns, dtype=float))) if eval_returns else 0.0
    rows.append(
        {
            "step": int(total_steps),
            "returns": list(eval_returns),
            "mean_return": final_mean,
            "tracking_diagnostics": dict(eval_tracking_diagnostics),
            "kind": "final",
        }
    )
    return rows


def _write_episodes_csv(path: Path, *, episode_records: list[dict[str, object]]) -> None:
    fieldnames = [
        "episode",
        "env_index",
        "return",
        "length",
        "terminated",
        "truncated",
        "termination_reason",
        "mean_ip_error_norm",
        "mean_shape_error_norm",
        "boundary_failure_steps",
        "reference_episode_seed",
        "reference_effective_seed",
        "reference_effective_ip_seed",
        "randomization_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(episode_records):
            out = {name: row.get(name, "") for name in fieldnames}
            out["episode"] = row.get("episode", index)
            writer.writerow(out)


def _write_eval_history_csv(path: Path, *, eval_history: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "step", "mean_return", "returns"])
        writer.writeheader()
        for index, row in enumerate(eval_history):
            writer.writerow(
                {
                    "index": index,
                    "step": row.get("step", ""),
                    "mean_return": row.get("mean_return", ""),
                    "returns": json.dumps(json_safe(row.get("returns", [])), sort_keys=True),
                }
            )


def _write_reward_components_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "env_index", "episode", "component", "value"])
        writer.writeheader()


def _write_reference_samples_npz(path: Path, *, reference_records: list[dict[str, object]]) -> None:
    np.savez(
        path,
        episode=np.asarray([_int_or_default(row.get("episode"), index) for index, row in enumerate(reference_records)], dtype=np.int64),
        env_index=np.asarray([_int_or_default(row.get("env_index"), -1) for row in reference_records], dtype=np.int64),
        reference_episode_seed=np.asarray([_int_or_default(row.get("reference_episode_seed"), -1) for row in reference_records], dtype=np.int64),
        reference_effective_seed=np.asarray([_int_or_default(row.get("reference_effective_seed"), -1) for row in reference_records], dtype=np.int64),
        reference_effective_ip_seed=np.asarray([_int_or_default(row.get("reference_effective_ip_seed"), -1) for row in reference_records], dtype=np.int64),
        randomization_seed=np.asarray([_int_or_default(row.get("randomization_seed"), -1) for row in reference_records], dtype=np.int64),
        reference_resampling_enabled=np.asarray([bool(row.get("reference_resampling_enabled", False)) for row in reference_records], dtype=bool),
        randomization_enabled=np.asarray([bool(row.get("randomization_enabled", False)) for row in reference_records], dtype=bool),
    )


def _write_termination_counts_json(path: Path, *, episode_records: list[dict[str, object]]) -> None:
    counts: dict[str, int] = {}
    truncated = 0
    terminated = 0
    for row in episode_records:
        if bool(row.get("truncated", False)):
            truncated += 1
        if bool(row.get("terminated", False)):
            terminated += 1
        reason = str(row.get("termination_reason", "") or "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    path.write_text(
        json.dumps(
            json_safe(
                {
                    "episodes": len(episode_records),
                    "terminated": terminated,
                    "truncated": truncated,
                    "by_reason": counts,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_eval_rollout_artifacts(path: Path, *, eval_history: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(eval_history):
        step = row.get("step", f"index_{index}")
        try:
            step_name = f"eval_step_{int(step):08d}"
        except (TypeError, ValueError):
            step_name = f"eval_{index:04d}"
        out = path / step_name
        out.mkdir(parents=True, exist_ok=True)
        returns = np.asarray(row.get("returns", []), dtype=float).reshape(-1)
        summary = {
            "step": row.get("step"),
            "episodes": int(returns.size),
            "mean_return": float(np.mean(returns)) if returns.size else 0.0,
            "tracking_diagnostics": row.get("tracking_diagnostics", {}),
        }
        (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
        with (out / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "return"])
            writer.writeheader()
            for episode, value in enumerate(returns):
                writer.writerow({"episode": episode, "return": float(value)})
        np.savez(out / "rollouts.npz", returns=returns, mask=np.ones_like(returns, dtype=bool))


def _legacy_episode_records(*, episode_returns: list[float], episode_lengths: list[int]) -> list[dict[str, object]]:
    return [
        {
            "episode": index,
            "env_index": -1,
            "return": float(episode_return),
            "length": int(episode_length),
            "terminated": False,
            "truncated": True,
            "termination_reason": "",
        }
        for index, (episode_return, episode_length) in enumerate(zip(episode_returns, episode_lengths, strict=False))
    ]


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _same_step(value: object, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


__all__ = ["RewardComponentWriter", "eval_history_with_final", "write_training_contract_artifacts"]
