from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from tokamak_rl.observations import ObservationSchema

if TYPE_CHECKING:  # pragma: no cover - typing only; NumPy inference should not import Torch.
    from tokamak_rl.networks import FeedForwardActor


@dataclass(frozen=True, slots=True)
class ExportedPolicyPaths:
    """Files produced by deterministic actor export."""

    output_dir: Path
    policy_weights_npz: Path
    schema_json: Path
    normalization_json: Path
    metadata_json: Path


class NumpyFeedForwardActor:
    """NumPy deterministic mean-path actor loaded from an exported artifact."""

    def __init__(self, weights: Mapping[str, np.ndarray], *, layer_norm_eps: float, obs_dim: int, action_dim: int) -> None:
        self.weights = {name: np.asarray(value, dtype=np.float32) for name, value in weights.items()}
        self.layer_norm_eps = float(layer_norm_eps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._validate_weights()

    @classmethod
    def from_export_dir(cls, export_dir: str | Path, *, expected_schema: Mapping[str, object] | None = None) -> NumpyFeedForwardActor:
        export_dir = Path(export_dir)
        schema = _read_json(export_dir / "schema.json")
        if expected_schema is not None:
            _check_expected_schema(schema, expected_schema)
        metadata = _read_json(export_dir / "metadata.json")
        with np.load(export_dir / "policy_weights.npz", allow_pickle=False) as data:
            weights = {name: np.asarray(data[name], dtype=np.float32) for name in data.files}
        return cls(
            weights,
            layer_norm_eps=float(metadata["layer_norm_eps"]),
            obs_dim=int(schema["obs_dim"]),
            action_dim=int(schema["action_dim"]),
        )

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"observation shape must be (batch, {self.obs_dim}), got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must contain finite values")
        x = _linear(obs, self.weights["input.weight"], self.weights["input.bias"])
        x = _layer_norm(x, self.weights["input_norm.weight"], self.weights["input_norm.bias"], eps=self.layer_norm_eps)
        x = np.tanh(x)
        x = _elu(_linear(x, self.weights["hidden1.weight"], self.weights["hidden1.bias"]))
        x = _elu(_linear(x, self.weights["hidden2.weight"], self.weights["hidden2.bias"]))
        x = _elu(_linear(x, self.weights["hidden3.weight"], self.weights["hidden3.bias"]))
        mean = _linear(x, self.weights["mean_head.weight"], self.weights["mean_head.bias"])
        return np.tanh(mean).astype(np.float32, copy=False)

    def _validate_weights(self) -> None:
        required = {
            "input.weight",
            "input.bias",
            "input_norm.weight",
            "input_norm.bias",
            "hidden1.weight",
            "hidden1.bias",
            "hidden2.weight",
            "hidden2.bias",
            "hidden3.weight",
            "hidden3.bias",
            "mean_head.weight",
            "mean_head.bias",
        }
        missing = sorted(required - set(self.weights))
        if missing:
            raise ValueError(f"exported actor is missing weights: {', '.join(missing)}")
        if self.weights["input.weight"].shape[1] != self.obs_dim:
            raise ValueError("input weight shape does not match exported obs_dim")
        if self.weights["mean_head.weight"].shape[0] != self.action_dim:
            raise ValueError("mean head shape does not match exported action_dim")


def export_actor(
    actor: "FeedForwardActor",
    output_dir: str | Path,
    *,
    schema: ObservationSchema,
    normalization: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> ExportedPolicyPaths:
    """Export the deterministic actor mean path to a NumPy-compatible artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(schema.obs_dim) != int(actor.obs_dim):
        raise ValueError("schema obs_dim does not match actor obs_dim")
    schema_data = schema.to_metadata()
    schema_data.update(
        {
            "action_dim": int(actor.action_dim),
            "action_range": [-1.0, 1.0],
            "architecture_name": actor.architecture_name,
        }
    )
    metadata_data = {
        "actor_architecture_name": actor.architecture_name,
        "obs_dim": int(actor.obs_dim),
        "action_dim": int(actor.action_dim),
        "hidden_dim": int(actor.cfg.hidden_dim),
        "layer_norm_eps": float(actor.input_norm.eps),
    }
    if metadata is not None:
        metadata_data.update(dict(metadata))
    weights_path = output_dir / "policy_weights.npz"
    np.savez(weights_path, **_actor_mean_path_weights(actor))
    schema_path = output_dir / "schema.json"
    normalization_path = output_dir / "normalization.json"
    metadata_path = output_dir / "metadata.json"
    _write_json(schema_path, schema_data)
    _write_json(normalization_path, dict(normalization))
    _write_json(metadata_path, metadata_data)
    return ExportedPolicyPaths(
        output_dir=output_dir,
        policy_weights_npz=weights_path,
        schema_json=schema_path,
        normalization_json=normalization_path,
        metadata_json=metadata_path,
    )


def load_numpy_actor(export_dir: str | Path, *, expected_schema: Mapping[str, object] | None = None) -> NumpyFeedForwardActor:
    return NumpyFeedForwardActor.from_export_dir(export_dir, expected_schema=expected_schema)


def load_actor_from_training_checkpoint(path: str | Path, *, device: object = "cpu"):
    """Load the feedforward actor stored in a simple or TCV-style trainer checkpoint."""
    import torch

    from tokamak_rl.networks import ActorConfig, FeedForwardActor

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "actor_config" not in checkpoint or "actor_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain a training actor")
    actor = FeedForwardActor(ActorConfig(**checkpoint["actor_config"]))
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(torch.device(device))
    actor.eval()
    return actor, checkpoint


def export_actor_from_training_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    schema: ObservationSchema,
    normalization: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    device: object = "cpu",
) -> ExportedPolicyPaths:
    """Export the deterministic actor mean path from a full trainer checkpoint."""
    actor, checkpoint = load_actor_from_training_checkpoint(checkpoint_path, device=device)
    export_metadata = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_trainer": checkpoint.get("trainer"),
        "checkpoint_algorithm": checkpoint.get("algorithm"),
        "checkpoint_total_steps": checkpoint.get("total_steps"),
        "checkpoint_best_eval_score": checkpoint.get("best_eval_score"),
    }
    if metadata is not None:
        export_metadata.update(dict(metadata))
    return export_actor(actor, output_dir, schema=schema, normalization=normalization, metadata=export_metadata)


def _actor_mean_path_weights(actor: FeedForwardActor) -> dict[str, np.ndarray]:
    state = actor.state_dict()
    names = (
        "input.weight",
        "input.bias",
        "input_norm.weight",
        "input_norm.bias",
        "hidden1.weight",
        "hidden1.bias",
        "hidden2.weight",
        "hidden2.bias",
        "hidden3.weight",
        "hidden3.bias",
        "mean_head.weight",
        "mean_head.bias",
    )
    return {name: state[name].detach().cpu().numpy().astype(np.float32, copy=True) for name in names}


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + float(eps))) * weight + bias


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(x))


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def _check_expected_schema(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"exported schema is missing expected key: {key}")
        if actual[key] != expected_value:
            raise ValueError(f"exported schema mismatch for {key}: expected {expected_value!r}, got {actual[key]!r}")


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "ExportedPolicyPaths",
    "NumpyFeedForwardActor",
    "export_actor",
    "export_actor_from_training_checkpoint",
    "load_actor_from_training_checkpoint",
    "load_numpy_actor",
]
