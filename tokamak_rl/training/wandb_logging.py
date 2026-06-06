from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any, Mapping

from tokamak_rl.training.diagnostics import json_safe


@dataclass(frozen=True, slots=True)
class WandBConfig:
    enabled: bool = False
    project: str = "tokamak-rl"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()
    log_interval_steps: int = 1
    log_artifacts: bool = True

    def __post_init__(self) -> None:
        if int(self.log_interval_steps) <= 0:
            raise ValueError("wandb.log_interval_steps must be > 0")
        if str(self.mode) not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be one of: online, offline, disabled")
        if self.enabled and not str(self.project).strip():
            raise ValueError("wandb.project must be non-empty when W&B is enabled")


class WandBLogger:
    """Small optional adapter around Weights & Biases.

    The trainer code should not import wandb directly. Keeping this adapter lazy
    lets ordinary local tests and non-W&B training runs work without installing
    the optional package.
    """

    def __init__(self, cfg: WandBConfig, *, config: Mapping[str, Any], run_metadata: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg
        self._wandb: Any | None = None
        self._run: Any | None = None
        if not bool(cfg.enabled) or str(cfg.mode) == "disabled":
            return
        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("W&B logging was requested, but wandb is not installed. Install tokamak-rl[wandb] or tokamak-rl[train].") from exc
        self._wandb = wandb
        init_config = json_safe({"trainer_config": config, "run_metadata": run_metadata or {}})
        self._run = wandb.init(
            project=str(cfg.project),
            entity=cfg.entity,
            name=cfg.name,
            group=cfg.group,
            mode=str(cfg.mode),
            tags=list(cfg.tags),
            config=init_config,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, values: Mapping[str, Any], *, step: int, force: bool = False) -> None:
        if self._run is None:
            return
        if not bool(force) and int(step) % int(self.cfg.log_interval_steps) != 0:
            return
        metrics = _flatten_numeric(values)
        if metrics:
            self._run.log(metrics, step=int(step))

    def log_episode(self, values: Mapping[str, Any], *, step: int) -> None:
        self.log({"episode": values}, step=step, force=True)

    def log_eval(self, values: Mapping[str, Any], *, step: int) -> None:
        self.log({"eval": values}, step=step, force=True)

    def log_final(self, values: Mapping[str, Any], *, artifact_paths: Mapping[str, str | Path | None] | None = None, step: int) -> None:
        if self._run is None:
            return
        flattened = _flatten_numeric({"final": values})
        if flattened:
            self._run.log(flattened, step=int(step))
        for key, value in _flatten_summary(values).items():
            self._run.summary[key] = value
        if bool(self.cfg.log_artifacts) and artifact_paths:
            self._log_artifacts(artifact_paths)

    def close(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None

    def _log_artifacts(self, artifact_paths: Mapping[str, str | Path | None]) -> None:
        if self._wandb is None or self._run is None:
            return
        for label, raw_path in artifact_paths.items():
            if raw_path is None:
                continue
            path = Path(raw_path)
            if not path.exists():
                continue
            artifact = self._wandb.Artifact(name=_artifact_name(label), type="training-artifact")
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)


def _artifact_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label)).strip("-")
    return safe or "artifact"


def _flatten_numeric(values: Mapping[str, Any], *, prefix: str = "") -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten_numeric(value, prefix=name))
        elif isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = float(value)
    return result


def _flatten_summary(values: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in json_safe(values).items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten_summary(value, prefix=name))
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
    return result


__all__ = ["WandBConfig", "WandBLogger"]
