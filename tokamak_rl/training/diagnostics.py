from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class TrainingDiagnostics:
    """Accumulate optional simulator target-tracking diagnostics during training."""

    ip_error_norm: list[float] = field(default_factory=list)
    shape_error_norm: list[float] = field(default_factory=list)
    boundary_found: list[bool] = field(default_factory=list)
    reference_effective_seed: list[int] = field(default_factory=list)
    reference_effective_ip_seed: list[int] = field(default_factory=list)

    def record_reset_info(self, info: Mapping[str, Any]) -> None:
        metadata = info.get("episode_metadata") if isinstance(info, Mapping) else None
        if not isinstance(metadata, Mapping):
            return
        shape_seed = _optional_int(metadata.get("reference_effective_seed"))
        ip_seed = _optional_int(metadata.get("reference_effective_ip_seed"))
        if shape_seed is not None:
            self.reference_effective_seed.append(shape_seed)
        if ip_seed is not None:
            self.reference_effective_ip_seed.append(ip_seed)

    def record_step_info(self, info: Mapping[str, Any]) -> None:
        if not isinstance(info, Mapping):
            return
        components = info.get("reward_components")
        if isinstance(components, Mapping):
            _append_finite(self.ip_error_norm, components.get("ip_error_norm"))
            _append_finite(self.shape_error_norm, components.get("shape_error_norm"))
        snapshot = info.get("snapshot")
        if snapshot is not None and hasattr(snapshot, "boundary_found"):
            self.boundary_found.append(bool(getattr(snapshot, "boundary_found")))

    def summary(self) -> dict[str, object]:
        return {
            "mean_ip_error_norm": _mean_or_none(self.ip_error_norm),
            "mean_shape_error_norm": _mean_or_none(self.shape_error_norm),
            "boundary_failure_rate": _failure_rate(self.boundary_found),
            "reference_effective_seed_count": len(self.reference_effective_seed),
            "reference_effective_ip_seed_count": len(self.reference_effective_ip_seed),
            "reference_effective_seed_first": None if not self.reference_effective_seed else int(self.reference_effective_seed[0]),
            "reference_effective_seed_last": None if not self.reference_effective_seed else int(self.reference_effective_seed[-1]),
            "reference_effective_ip_seed_first": None if not self.reference_effective_ip_seed else int(self.reference_effective_ip_seed[0]),
            "reference_effective_ip_seed_last": None if not self.reference_effective_ip_seed else int(self.reference_effective_ip_seed[-1]),
        }


def _append_finite(values: list[float], value: object) -> None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(out):
        values.append(out)


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _failure_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=bool)
    return float(np.mean(~arr))


def json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def reset_artifact_record(info: Mapping[str, Any], *, env_index: int, episode: int) -> dict[str, object]:
    metadata = info.get("episode_metadata") if isinstance(info, Mapping) else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    randomization = metadata.get("randomization")
    randomization = randomization if isinstance(randomization, Mapping) else {}
    return {
        "env_index": int(env_index),
        "episode": int(episode),
        "reference_resampling_enabled": bool(metadata.get("reference_resampling_enabled", False)),
        "reference_episode_seed": _optional_int_with_default(metadata.get("reference_episode_seed"), -1),
        "reference_effective_seed": _optional_int_with_default(metadata.get("reference_effective_seed"), -1),
        "reference_effective_ip_seed": _optional_int_with_default(metadata.get("reference_effective_ip_seed"), -1),
        "randomization_enabled": bool(randomization.get("enabled", False)),
        "randomization_seed": _optional_int_with_default(randomization.get("seed"), -1),
    }


def episode_artifact_record(
    *,
    env_index: int,
    episode: int,
    episode_return: float,
    episode_length: int,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
    ip_errors: list[float],
    shape_errors: list[float],
    boundary_failure_steps: int,
    reset_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "episode": int(episode),
        "env_index": int(env_index),
        "return": float(episode_return),
        "length": int(episode_length),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(termination_reason),
        "mean_ip_error_norm": _mean_or_empty(ip_errors),
        "mean_shape_error_norm": _mean_or_empty(shape_errors),
        "boundary_failure_steps": int(boundary_failure_steps),
        "reference_episode_seed": reset_record.get("reference_episode_seed", -1),
        "reference_effective_seed": reset_record.get("reference_effective_seed", -1),
        "reference_effective_ip_seed": reset_record.get("reference_effective_ip_seed", -1),
        "randomization_seed": reset_record.get("randomization_seed", -1),
    }


def record_episode_step_artifacts(
    info: Mapping[str, Any],
    *,
    ip_errors: list[float],
    shape_errors: list[float],
    boundary_failure_counter: list[int],
    env_index: int,
) -> None:
    components = info.get("reward_components") if isinstance(info, Mapping) else None
    if isinstance(components, Mapping):
        _append_finite(ip_errors, components.get("ip_error_norm"))
        _append_finite(shape_errors, components.get("shape_error_norm"))
    snapshot = info.get("snapshot") if isinstance(info, Mapping) else None
    if snapshot is not None and hasattr(snapshot, "boundary_found") and not bool(getattr(snapshot, "boundary_found")):
        boundary_failure_counter[int(env_index)] += 1


def termination_reason_from_step_info(info: Mapping[str, Any], *, terminated: bool, truncated: bool) -> str:
    if isinstance(info, Mapping):
        reason = info.get("termination_reason")
        if reason:
            return str(reason)
    if bool(truncated):
        return "max_episode_steps"
    if bool(terminated):
        return "simulator_terminated"
    return ""


def _optional_int_with_default(value: object, default: int) -> int:
    out = _optional_int(value)
    return int(default) if out is None else int(out)


def _mean_or_empty(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(np.mean(np.asarray(values, dtype=float)))
