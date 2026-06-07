from __future__ import annotations

import csv
from dataclasses import dataclass
import glob
from pathlib import Path
import re
from typing import Any, Mapping

try:  # pragma: no cover - exercised when PyYAML is installed in the RL env.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback is tested instead.
    yaml = None

from tokamak_rl.env import EnvConfig, ReplayInitialStateCandidate, ReplayInitialStateConfig, TerminationConfig
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward
from tokamak_rl.training.wandb_logging import WandBConfig
from tokamak_control.realism import ActuatorRealismSettings, SensorRealismSettings


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Typed top-level experiment config for staged implementation."""

    name: str
    source_path: Path
    env: EnvConfig
    reward: JointCurrentBoundaryReward
    randomization: DomainRandomizer
    evaluation: ExperimentEvaluationConfig
    training: ExperimentTrainingConfig
    artifacts: ExperimentArtifactConfig
    wandb: WandBConfig
    reward_config_path: Path | None
    randomization_config_path: Path | None


@dataclass(frozen=True, slots=True)
class ExperimentEvaluationConfig:
    deterministic: bool = True
    episodes: int = 1
    max_steps: int = 200
    validation_seed: int = 1000
    randomization_mode: str = "configured"


@dataclass(frozen=True, slots=True)
class ExperimentArtifactConfig:
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    checkpoint_interval_steps: int | None = None
    max_step_checkpoints: int | None = None
    export_best_actor: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentTrainingConfig:
    enabled: bool = False
    note: str | None = None
    trainer: str = "tcv_style"
    total_steps: int = 500
    warmup_steps: int = 100
    batch_size: int = 64
    sequence_length: int = 64
    hidden_dim: int = 256
    critic_hidden_dim: int | None = None
    critic_mlp_hidden_dim: int | None = None
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
    device: str = "cpu"
    seed: int = 0
    num_envs: int = 1
    updates_per_episode: int = 1
    updates_per_env_step: int = 0
    max_learner_catchup_updates: int | None = None
    process_envs: bool = False
    process_start_method: str = "spawn"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML and referenced reward/randomization YAML files."""
    source_path = Path(path).expanduser().resolve()
    raw = _load_yaml_mapping(source_path)
    _reject_unknown(raw, {"name", "sim", "reward_config", "randomization_config", "evaluation", "training", "artifacts", "wandb"}, "experiment")
    name = _require_str(raw, "name")
    env = _load_env_config(_require_mapping(raw, "sim"), base_dir=source_path.parent)

    reward_path = _optional_path(raw.get("reward_config"), base_dir=source_path.parent, field="reward_config")
    randomization_path = _optional_path(raw.get("randomization_config"), base_dir=source_path.parent, field="randomization_config")
    reward = _load_reward(reward_path) if reward_path is not None else JointCurrentBoundaryReward()
    randomization = _load_randomization(randomization_path) if randomization_path is not None else DomainRandomizer()
    evaluation = _load_evaluation_config(raw.get("evaluation", {}))
    training = _load_training_config(raw.get("training", {}))
    artifacts = _load_artifact_config(raw.get("artifacts", {}), base_dir=source_path.parent)
    wandb = _load_wandb_config(raw.get("wandb", {}))
    return ExperimentConfig(
        name=name,
        source_path=source_path,
        env=env,
        reward=reward,
        randomization=randomization,
        evaluation=evaluation,
        training=training,
        artifacts=artifacts,
        wandb=wandb,
        reward_config_path=reward_path,
        randomization_config_path=randomization_path,
    )


def _load_evaluation_config(value: object) -> ExperimentEvaluationConfig:
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a mapping")
    _reject_unknown(value, {"deterministic", "episodes", "max_steps", "validation_seed", "randomization_mode"}, "evaluation")
    randomization_mode = str(value.get("randomization_mode", "configured"))
    if randomization_mode not in {"configured", "clean"}:
        raise ValueError("evaluation.randomization_mode must be one of: configured, clean")
    return ExperimentEvaluationConfig(
        deterministic=_bool(value.get("deterministic", True), "evaluation.deterministic"),
        episodes=_positive_int(value.get("episodes", 1), "evaluation.episodes"),
        max_steps=_positive_int(value.get("max_steps", 200), "evaluation.max_steps"),
        validation_seed=_int(value.get("validation_seed", 1000), "evaluation.validation_seed"),
        randomization_mode=randomization_mode,
    )


def _load_artifact_config(value: object, *, base_dir: Path) -> ExperimentArtifactConfig:
    if not isinstance(value, dict):
        raise ValueError("artifacts must be a mapping")
    _reject_unknown(value, {"output_dir", "checkpoint_dir", "checkpoint_interval_steps", "max_step_checkpoints", "export_best_actor"}, "artifacts")
    return ExperimentArtifactConfig(
        output_dir=_optional_output_path(value.get("output_dir"), base_dir=base_dir, field="artifacts.output_dir"),
        checkpoint_dir=_optional_output_path(value.get("checkpoint_dir"), base_dir=base_dir, field="artifacts.checkpoint_dir"),
        checkpoint_interval_steps=None if value.get("checkpoint_interval_steps") is None else _positive_int(value.get("checkpoint_interval_steps"), "artifacts.checkpoint_interval_steps"),
        max_step_checkpoints=None if value.get("max_step_checkpoints") is None else _positive_int(value.get("max_step_checkpoints"), "artifacts.max_step_checkpoints"),
        export_best_actor=_bool(value.get("export_best_actor", True), "artifacts.export_best_actor"),
    )


def _load_wandb_config(value: object) -> WandBConfig:
    if not isinstance(value, dict):
        raise ValueError("wandb must be a mapping")
    _reject_unknown(value, {"enabled", "project", "entity", "name", "group", "mode", "tags", "log_interval_steps", "log_artifacts"}, "wandb")
    tags_raw = value.get("tags", ())
    if tags_raw is None:
        tags: tuple[str, ...] = ()
    elif isinstance(tags_raw, str):
        tags = tuple(part.strip() for part in tags_raw.split(",") if part.strip())
    elif isinstance(tags_raw, (list, tuple)):
        tags = tuple(str(item) for item in tags_raw)
    else:
        raise ValueError("wandb.tags must be a string or sequence of strings")
    return WandBConfig(
        enabled=_bool(value.get("enabled", False), "wandb.enabled"),
        project=str(value.get("project", "tokamak-rl")),
        entity=None if value.get("entity") is None else str(value.get("entity")),
        name=None if value.get("name") is None else str(value.get("name")),
        group=None if value.get("group") is None else str(value.get("group")),
        mode=str(value.get("mode", "online")),
        tags=tags,
        log_interval_steps=_positive_int(value.get("log_interval_steps", 1), "wandb.log_interval_steps"),
        log_artifacts=_bool(value.get("log_artifacts", True), "wandb.log_artifacts"),
    )


def _load_training_config(value: object) -> ExperimentTrainingConfig:
    if not isinstance(value, dict):
        raise ValueError("training must be a mapping")
    allowed = {
        "enabled",
        "note",
        "trainer",
        "total_steps",
        "steps",
        "warmup_steps",
        "batch_size",
        "sequence_length",
        "hidden_dim",
        "critic_hidden_dim",
        "critic_mlp_hidden_dim",
        "mpo_kl_lr",
        "mpo_epsilon",
        "mpo_mean_kl_epsilon",
        "mpo_std_kl_epsilon",
        "mpo_action_samples",
        "mpo_temperature_iterations",
        "mpo_temperature_lr",
        "mpo_initial_temperature",
        "mpo_initial_mean_kl_penalty",
        "mpo_initial_std_kl_penalty",
        "device",
        "seed",
        "num_envs",
        "updates_per_episode",
        "updates_per_env_step",
        "max_learner_catchup_updates",
        "process_envs",
        "process_start_method",
    }
    _reject_unknown(value, allowed, "training")
    trainer = str(value.get("trainer", "tcv_style"))
    if trainer not in {"tcv_style", "simple"}:
        raise ValueError("training.trainer must be one of: tcv_style, simple")
    device = str(value.get("device", "cpu"))
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("training.device must be one of: cpu, cuda, auto")
    start_method = str(value.get("process_start_method", "spawn"))
    if start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError("training.process_start_method must be one of: spawn, fork, forkserver")
    steps_value = value.get("total_steps", value.get("steps", 500))
    return ExperimentTrainingConfig(
        enabled=_bool(value.get("enabled", False), "training.enabled"),
        note=None if value.get("note") is None else str(value.get("note")),
        trainer=trainer,
        total_steps=_positive_int(steps_value, "training.total_steps"),
        warmup_steps=_nonnegative_int(value.get("warmup_steps", 100), "training.warmup_steps"),
        batch_size=_positive_int(value.get("batch_size", 64), "training.batch_size"),
        sequence_length=_positive_int(value.get("sequence_length", 64), "training.sequence_length"),
        hidden_dim=_positive_int(value.get("hidden_dim", 256), "training.hidden_dim"),
        critic_hidden_dim=None if value.get("critic_hidden_dim") is None else _positive_int(value.get("critic_hidden_dim"), "training.critic_hidden_dim"),
        critic_mlp_hidden_dim=None if value.get("critic_mlp_hidden_dim") is None else _positive_int(value.get("critic_mlp_hidden_dim"), "training.critic_mlp_hidden_dim"),
        mpo_kl_lr=_finite_float(value.get("mpo_kl_lr", 3.0e-4), "training.mpo_kl_lr"),
        mpo_epsilon=_finite_float(value.get("mpo_epsilon", 0.1), "training.mpo_epsilon"),
        mpo_mean_kl_epsilon=_finite_float(value.get("mpo_mean_kl_epsilon", 0.01), "training.mpo_mean_kl_epsilon"),
        mpo_std_kl_epsilon=_finite_float(value.get("mpo_std_kl_epsilon", 1.0e-4), "training.mpo_std_kl_epsilon"),
        mpo_action_samples=_positive_int(value.get("mpo_action_samples", 20), "training.mpo_action_samples"),
        mpo_temperature_iterations=_positive_int(value.get("mpo_temperature_iterations", 10), "training.mpo_temperature_iterations"),
        mpo_temperature_lr=_finite_float(value.get("mpo_temperature_lr", 0.1), "training.mpo_temperature_lr"),
        mpo_initial_temperature=_finite_float(value.get("mpo_initial_temperature", 1.0), "training.mpo_initial_temperature"),
        mpo_initial_mean_kl_penalty=_finite_float(value.get("mpo_initial_mean_kl_penalty", 1.0), "training.mpo_initial_mean_kl_penalty"),
        mpo_initial_std_kl_penalty=_finite_float(value.get("mpo_initial_std_kl_penalty", 1.0), "training.mpo_initial_std_kl_penalty"),
        device=device,
        seed=_int(value.get("seed", 0), "training.seed"),
        num_envs=_positive_int(value.get("num_envs", 1), "training.num_envs"),
        updates_per_episode=_positive_int(value.get("updates_per_episode", 1), "training.updates_per_episode"),
        updates_per_env_step=_nonnegative_int(value.get("updates_per_env_step", 0), "training.updates_per_env_step"),
        max_learner_catchup_updates=None if value.get("max_learner_catchup_updates") is None else _positive_int(value.get("max_learner_catchup_updates"), "training.max_learner_catchup_updates"),
        process_envs=_bool(value.get("process_envs", False), "training.process_envs"),
        process_start_method=start_method,
    )


def _load_env_config(raw: Mapping[str, Any], *, base_dir: Path) -> EnvConfig:
    _reject_unknown(
        raw,
        {
            "config_path",
            "initial_currents_path",
            "initial_state",
            "scenario_name",
            "scenario_args",
            "reference_source",
            "angles",
            "max_episode_steps",
            "realism_enabled",
            "resample_references_on_reset",
            "observation",
            "termination",
        },
        "sim",
    )
    sim_config_path = _required_path(raw, "config_path", base_dir=base_dir)
    initial_raw = raw.get("initial_currents_path")
    initial_path = None if initial_raw is None else _resolve_existing_path(initial_raw, base_dir=base_dir, field="initial_currents_path")
    initial_ip = None
    initial_coil_currents = "config"
    initial_ip_scale = None
    replay_initial_state = None
    if "initial_state" in raw:
        initial_state = _load_initial_state_config(_require_mapping(raw, "initial_state"), base_dir=base_dir)
        initial_ip = initial_state["ip"]
        initial_coil_currents = initial_state["coil_currents"]
        initial_ip_scale = initial_state["ip_scale"]
        replay_initial_state = initial_state["replay_initial_state"]
    scenario_args_raw = raw.get("scenario_args", {})
    if not isinstance(scenario_args_raw, dict):
        raise ValueError("sim.scenario_args must be a mapping")
    scenario_name = str(raw.get("scenario_name", "nominal"))
    scenario_args = dict(scenario_args_raw)
    if "reference_source" in raw:
        if scenario_name != "nominal" or scenario_args:
            raise ValueError("sim.reference_source cannot be combined with scenario_name/scenario_args")
        scenario_name, scenario_args = _load_reference_source(_require_mapping(raw, "reference_source"), base_dir=base_dir)
    angles = _positive_int(raw.get("angles", 32), "sim.angles")
    max_episode_steps = _positive_int(raw.get("max_episode_steps", 1000), "sim.max_episode_steps")
    realism_enabled = _bool(raw.get("realism_enabled", True), "sim.realism_enabled")
    resample_references_on_reset = _bool(raw.get("resample_references_on_reset", True), "sim.resample_references_on_reset")
    observation_version = "v1"
    target_preview_steps = 0
    target_preview_stride = 1
    if "observation" in raw:
        observation_raw = _require_mapping(raw, "observation")
        _reject_unknown(observation_raw, {"version", "target_preview_steps", "target_preview_stride"}, "sim.observation")
        observation_version = str(observation_raw.get("version", "v1"))
        target_preview_steps = _positive_int(observation_raw.get("target_preview_steps", 0), "sim.observation.target_preview_steps") if observation_version == "v2" else _int(observation_raw.get("target_preview_steps", 0), "sim.observation.target_preview_steps")
        if int(target_preview_steps) < 0:
            raise ValueError("sim.observation.target_preview_steps must be >= 0")
        target_preview_stride = _positive_int(observation_raw.get("target_preview_stride", 1), "sim.observation.target_preview_stride")
    termination = TerminationConfig()
    if "termination" in raw:
        termination = _load_termination_config(_require_mapping(raw, "termination"))
    return EnvConfig(
        sim_config_path=sim_config_path,
        initial_currents_path=initial_path,
        initial_ip=initial_ip,
        initial_coil_currents=initial_coil_currents,
        initial_ip_scale=initial_ip_scale,
        replay_initial_state=replay_initial_state,
        scenario_name=scenario_name,
        scenario_args=scenario_args,
        angles=angles,
        max_episode_steps=max_episode_steps,
        realism_enabled=realism_enabled,
        resample_references_on_reset=resample_references_on_reset,
        observation_version=observation_version,
        target_preview_steps=target_preview_steps,
        target_preview_stride=target_preview_stride,
        termination=termination,
    )


def _load_initial_state_config(raw: Mapping[str, Any], *, base_dir: Path) -> dict[str, object]:
    _reject_unknown(raw, {"ip", "coil_currents", "ip_scale", "replay"}, "sim.initial_state")
    coil_currents = str(raw.get("coil_currents", "config"))
    if coil_currents not in {"config", "zero", "sample_replay"}:
        raise ValueError("sim.initial_state.coil_currents must be one of: config, zero, sample_replay")
    ip_raw = raw.get("ip")
    if ip_raw is None or str(ip_raw) == "sample_replay":
        ip = None
    else:
        ip = _finite_float(ip_raw, "sim.initial_state.ip")
    ip_scale = None if raw.get("ip_scale") is None else _positive_float(raw.get("ip_scale"), "sim.initial_state.ip_scale")
    replay_initial_state = None
    if "replay" in raw:
        replay_initial_state = _load_replay_initial_state_config(_require_mapping(raw, "replay"), base_dir=base_dir)
    if coil_currents == "sample_replay" and replay_initial_state is None:
        raise ValueError("sim.initial_state.replay is required when coil_currents is sample_replay")
    if str(ip_raw) == "sample_replay" and replay_initial_state is None:
        raise ValueError("sim.initial_state.replay is required when ip is sample_replay")
    return {"ip": ip, "coil_currents": coil_currents, "ip_scale": ip_scale, "replay_initial_state": replay_initial_state}


def _load_replay_initial_state_config(raw: Mapping[str, Any], *, base_dir: Path) -> ReplayInitialStateConfig:
    _reject_unknown(raw, {"initial_currents_glob", "boundary_parameters_glob"}, "sim.initial_state.replay")
    currents_glob = _require_str(raw, "initial_currents_glob")
    boundary_glob = _require_str(raw, "boundary_parameters_glob")
    current_paths = _resolve_glob(currents_glob, base_dir=base_dir, field="sim.initial_state.replay.initial_currents_glob")
    boundary_by_shot = _load_initial_boundary_rows(
        _resolve_glob(boundary_glob, base_dir=base_dir, field="sim.initial_state.replay.boundary_parameters_glob")
    )
    candidates: list[ReplayInitialStateCandidate] = []
    for path in current_paths:
        shot = _extract_shot_id(path)
        if shot is None:
            continue
        boundary = boundary_by_shot.get(shot)
        if boundary is None:
            continue
        candidates.append(
            ReplayInitialStateCandidate(
                shot=shot,
                initial_currents_path=path,
                initial_ip=float(boundary["Ip"]),
                initial_boundary_parameters={
                    key: float(boundary[key])
                    for key in ("R0", "Z0", "A0", "kappa", "delta")
                    if key in boundary
                },
            )
        )
    if not candidates:
        raise ValueError("sim.initial_state.replay produced no matched replay initial-state candidates")
    candidates.sort(key=lambda item: item.shot)
    return ReplayInitialStateConfig(candidates=tuple(candidates))


def _resolve_glob(pattern: str, *, base_dir: Path, field: str) -> list[Path]:
    raw = Path(pattern).expanduser()
    candidate_patterns = [str(raw)] if raw.is_absolute() else [str(base_dir / raw), str(base_dir.parent.parent / raw)]
    paths: list[Path] = []
    resolved_pattern = candidate_patterns[0]
    for candidate_pattern in candidate_patterns:
        matched = sorted(Path(p).resolve() for p in glob.glob(candidate_pattern))
        if matched:
            paths = matched
            resolved_pattern = candidate_pattern
            break
    if not paths:
        raise FileNotFoundError(f"{field} matched no files: {resolved_pattern}")
    return paths


def _extract_shot_id(path: Path) -> str | None:
    match = re.search(r"(\d{4,6})", path.stem)
    return None if match is None else match.group(1)


def _load_initial_boundary_rows(paths: list[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("fit_status") != "ok":
                    continue
                shot = str(row.get("shot") or _extract_shot_id(path) or "")
                if not shot:
                    continue
                out[shot] = {
                    key: _finite_float(row[key], f"{path.name}.{key}")
                    for key in ("Ip", "R0", "Z0", "A0", "kappa", "delta")
                    if key in row and row[key] not in {None, ""}
                }
                break
    return out


def _load_termination_config(raw: Mapping[str, Any]) -> TerminationConfig:
    allowed = {
        "terminate_on_boundary_loss",
        "boundary_loss_grace_steps",
        "terminate_on_nonfinite_observation",
        "terminate_on_nonfinite_reward",
        "current_limit_margin_min",
        "derivative_limit_margin_min",
        "measured_boundary_missing_steps",
    }
    _reject_unknown(raw, allowed, "sim.termination")
    return TerminationConfig(
        terminate_on_boundary_loss=_bool(raw.get("terminate_on_boundary_loss", True), "sim.termination.terminate_on_boundary_loss"),
        boundary_loss_grace_steps=_nonnegative_int(raw.get("boundary_loss_grace_steps", 0), "sim.termination.boundary_loss_grace_steps"),
        terminate_on_nonfinite_observation=_bool(raw.get("terminate_on_nonfinite_observation", True), "sim.termination.terminate_on_nonfinite_observation"),
        terminate_on_nonfinite_reward=_bool(raw.get("terminate_on_nonfinite_reward", True), "sim.termination.terminate_on_nonfinite_reward"),
        current_limit_margin_min=None if raw.get("current_limit_margin_min") is None else _finite_float(raw.get("current_limit_margin_min"), "sim.termination.current_limit_margin_min"),
        derivative_limit_margin_min=None if raw.get("derivative_limit_margin_min") is None else _finite_float(raw.get("derivative_limit_margin_min"), "sim.termination.derivative_limit_margin_min"),
        measured_boundary_missing_steps=None if raw.get("measured_boundary_missing_steps") is None else _positive_int(raw.get("measured_boundary_missing_steps"), "sim.termination.measured_boundary_missing_steps"),
    )


def _load_reference_source(raw: Mapping[str, Any], *, base_dir: Path) -> tuple[str, dict[str, object]]:
    kind = _require_str(raw, "kind")
    if kind != "t15_synthetic_follow":
        raise ValueError("sim.reference_source.kind must be 't15_synthetic_follow'")
    allowed = {
        "kind",
        "seed",
        "duration_s",
        "t_step",
        "target_update_s",
        "theta_count",
        "ip",
        "boundary",
        "preset",
    }
    _reject_unknown(raw, allowed, "sim.reference_source")
    args: dict[str, object] = {}
    if "preset" in raw:
        args["reference_preset"] = _require_str(raw, "preset")
    _copy_optional_int(raw, args, "seed", "sim.reference_source.seed")
    _copy_optional_float(raw, args, "duration_s", "sim.reference_source.duration_s")
    _copy_optional_float(raw, args, "t_step", "sim.reference_source.t_step")
    _copy_optional_float(raw, args, "target_update_s", "sim.reference_source.target_update_s")
    _copy_optional_int(raw, args, "theta_count", "sim.reference_source.theta_count")
    if "ip" in raw:
        ip_raw = raw["ip"]
        if not isinstance(ip_raw, dict):
            raise ValueError("sim.reference_source.ip must be a mapping")
        args.update(_load_reference_ip_source(ip_raw, base_dir=base_dir))
    if "boundary" in raw:
        boundary_raw = raw["boundary"]
        if not isinstance(boundary_raw, dict):
            raise ValueError("sim.reference_source.boundary must be a mapping")
        args.update(_load_reference_boundary_source(boundary_raw))
    return "t15_synthetic_follow", args


def _load_reference_ip_source(raw: Mapping[str, Any], *, base_dir: Path) -> dict[str, object]:
    kind = _require_str(raw, "kind")
    args: dict[str, object] = {}
    if kind == "template_dir":
        _reject_unknown(raw, {"kind", "path", "seed", "amplitude_jitter", "duration_jitter", "shape_jitter", "start"}, "sim.reference_source.ip")
        args["ip_template_dir"] = str(_required_reference_path(raw, "path", base_dir=base_dir, field="sim.reference_source.ip.path"))
        _copy_optional_int(raw, args, "seed", "sim.reference_source.ip.seed", target="ip_seed")
        _copy_optional_float(raw, args, "start", "sim.reference_source.ip.start", target="ip_start")
        _copy_ip_jitter(raw, args)
        return args
    if kind == "template_csv":
        _reject_unknown(raw, {"kind", "path", "seed", "amplitude_jitter", "duration_jitter", "shape_jitter", "start"}, "sim.reference_source.ip")
        args["ip_template_csv"] = str(_required_reference_path(raw, "path", base_dir=base_dir, field="sim.reference_source.ip.path"))
        _copy_optional_int(raw, args, "seed", "sim.reference_source.ip.seed", target="ip_seed")
        _copy_optional_float(raw, args, "start", "sim.reference_source.ip.start", target="ip_start")
        _copy_ip_jitter(raw, args)
        return args
    if kind == "csv":
        _reject_unknown(raw, {"kind", "path", "time_offset"}, "sim.reference_source.ip")
        args["ip_csv"] = str(_required_reference_path(raw, "path", base_dir=base_dir, field="sim.reference_source.ip.path"))
        _copy_optional_float(raw, args, "time_offset", "sim.reference_source.ip.time_offset")
        return args
    if kind == "ramp":
        _reject_unknown(raw, {"kind", "start", "end", "ramp_s"}, "sim.reference_source.ip")
        _copy_optional_float(raw, args, "start", "sim.reference_source.ip.start", target="ip_start")
        _copy_optional_float(raw, args, "end", "sim.reference_source.ip.end", target="ip_end")
        _copy_optional_float(raw, args, "ramp_s", "sim.reference_source.ip.ramp_s", target="ip_ramp_s")
        return args
    if kind == "segmented":
        _reject_unknown(
            raw,
            {
                "kind",
                "seed",
                "min",
                "max",
                "segment_min_steps",
                "segment_max_steps",
                "segment_count_min",
                "segment_count_max",
                "max_steps",
                "rate_limit",
                "hold_probability",
                "start",
            },
            "sim.reference_source.ip",
        )
        args["ip_segmented"] = True
        _copy_optional_int(raw, args, "seed", "sim.reference_source.ip.seed", target="ip_seed")
        _copy_optional_float(raw, args, "min", "sim.reference_source.ip.min", target="ip_min")
        _copy_optional_float(raw, args, "max", "sim.reference_source.ip.max", target="ip_max")
        _copy_optional_int(raw, args, "segment_min_steps", "sim.reference_source.ip.segment_min_steps", target="ip_segment_min_steps")
        _copy_optional_int(raw, args, "segment_max_steps", "sim.reference_source.ip.segment_max_steps", target="ip_segment_max_steps")
        _copy_optional_int(raw, args, "segment_count_min", "sim.reference_source.ip.segment_count_min", target="ip_segment_count_min")
        _copy_optional_int(raw, args, "segment_count_max", "sim.reference_source.ip.segment_count_max", target="ip_segment_count_max")
        _copy_optional_int(raw, args, "max_steps", "sim.reference_source.ip.max_steps", target="ip_max_steps")
        _copy_optional_float(raw, args, "rate_limit", "sim.reference_source.ip.rate_limit", target="ip_rate_limit")
        _copy_optional_float(raw, args, "hold_probability", "sim.reference_source.ip.hold_probability", target="ip_hold_probability")
        _copy_optional_float(raw, args, "start", "sim.reference_source.ip.start", target="ip_start")
        return args
    raise ValueError("sim.reference_source.ip.kind must be one of: template_dir, template_csv, csv, ramp, segmented")


def _load_reference_boundary_source(raw: Mapping[str, Any]) -> dict[str, object]:
    kind = _require_str(raw, "kind")
    args: dict[str, object] = {}
    if kind == "generated_parameters":
        _reject_unknown(raw, {"kind", "bounds", "rate_limits"}, "sim.reference_source.boundary")
        args["boundary_kind"] = "generated_parameters"
        if "bounds" in raw:
            args["boundary_bounds"] = _load_boundary_bounds(_require_mapping(raw, "bounds"), field="sim.reference_source.boundary.bounds")
        if "rate_limits" in raw:
            args["boundary_rate_limits"] = _load_boundary_rates(_require_mapping(raw, "rate_limits"), field="sim.reference_source.boundary.rate_limits")
        return args
    if kind == "static_parameters":
        _reject_unknown(raw, {"kind", "parameters"}, "sim.reference_source.boundary")
        args["boundary_kind"] = "static_parameters"
        args["boundary_parameters"] = _load_boundary_parameters(_require_mapping(raw, "parameters"), field="sim.reference_source.boundary.parameters")
        return args
    raise ValueError("sim.reference_source.boundary.kind must be one of: generated_parameters, static_parameters")


def _load_boundary_parameters(raw: Mapping[str, Any], *, field: str) -> dict[str, float]:
    _reject_unknown(raw, {"R0", "Z0", "A0", "kappa", "delta"}, field)
    return {name: _finite_float(raw[name], f"{field}.{name}") for name in ("R0", "Z0", "A0", "kappa", "delta")}


def _load_boundary_rates(raw: Mapping[str, Any], *, field: str) -> dict[str, float]:
    _reject_unknown(raw, {"R0", "Z0", "A0", "kappa", "delta"}, field)
    return {name: _finite_float(raw[name], f"{field}.{name}") for name in ("R0", "Z0", "A0", "kappa", "delta")}


def _load_boundary_bounds(raw: Mapping[str, Any], *, field: str) -> dict[str, dict[str, float]]:
    _reject_unknown(raw, {"R0", "Z0", "A0", "kappa", "delta"}, field)
    return {name: _load_min_max(_require_mapping(raw, name), field=f"{field}.{name}") for name in ("R0", "Z0", "A0", "kappa", "delta")}


def _load_min_max(raw: Mapping[str, Any], *, field: str) -> dict[str, float]:
    _reject_unknown(raw, {"min", "max"}, field)
    lo = _finite_float(raw.get("min"), f"{field}.min")
    hi = _finite_float(raw.get("max"), f"{field}.max")
    if hi <= lo:
        raise ValueError(f"{field} must satisfy max > min")
    return {"min": lo, "max": hi}


def _copy_ip_jitter(raw: Mapping[str, Any], args: dict[str, object]) -> None:
    _copy_optional_float(raw, args, "amplitude_jitter", "sim.reference_source.ip.amplitude_jitter")
    _copy_optional_float(raw, args, "duration_jitter", "sim.reference_source.ip.duration_jitter")
    _copy_optional_float(raw, args, "shape_jitter", "sim.reference_source.ip.shape_jitter")


def _copy_optional_float(raw: Mapping[str, Any], args: dict[str, object], key: str, field: str, *, target: str | None = None) -> None:
    if key not in raw:
        return
    value = _finite_float(raw[key], field)
    args[target or key] = value


def _copy_optional_int(raw: Mapping[str, Any], args: dict[str, object], key: str, field: str, *, target: str | None = None) -> None:
    if key not in raw:
        return
    args[target or key] = _int(raw[key], field)


def _required_reference_path(raw: Mapping[str, Any], key: str, *, base_dir: Path, field: str) -> Path:
    if key not in raw:
        raise ValueError(f"Missing required path field: {field}")
    return _resolve_existing_path(raw[key], base_dir=base_dir, field=field)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown fields in {field}: {', '.join(unknown)}")


def _finite_float(value: object, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np_isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def _positive_float(value: object, field: str) -> float:
    out = _finite_float(value, field)
    if out <= 0.0:
        raise ValueError(f"{field} must be > 0")
    return out


def _int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def np_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _load_reward(path: Path) -> JointCurrentBoundaryReward:
    raw = _load_yaml_mapping(path)
    allowed = set(JointCurrentBoundaryReward.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown reward config fields in {path}: {', '.join(unknown)}")
    return JointCurrentBoundaryReward(**{k: float(v) for k, v in raw.items()})


def _load_randomization(path: Path) -> DomainRandomizer:
    raw = _load_yaml_mapping(path)
    _reject_unknown(raw, {"enabled", "note", "actuators", "sensors"}, "randomization")
    actuators = ActuatorRealismSettings()
    sensors = SensorRealismSettings()
    if "actuators" in raw:
        actuators = _load_randomization_actuators(_require_mapping(raw, "actuators"))
    if "sensors" in raw:
        sensors = _load_randomization_sensors(_require_mapping(raw, "sensors"))
    return DomainRandomizer(
        enabled=_bool(raw.get("enabled", False), "randomization.enabled"),
        actuators=actuators,
        sensors=sensors,
        note=None if raw.get("note") is None else str(raw.get("note")),
    )


def _load_randomization_actuators(raw: Mapping[str, Any]) -> ActuatorRealismSettings:
    _reject_unknown(
        raw,
        {
            "pfc_delay_steps",
            "sol_delay_steps",
            "pfc_gain_sigma",
            "sol_gain_sigma",
            "pfc_bias_sigma",
            "sol_bias_sigma",
            "pfc_command_noise_sigma",
            "sol_command_noise_sigma",
        },
        "randomization.actuators",
    )
    out = ActuatorRealismSettings(
        pfc_delay_steps=_int(raw.get("pfc_delay_steps", 0), "randomization.actuators.pfc_delay_steps"),
        sol_delay_steps=_int(raw.get("sol_delay_steps", 0), "randomization.actuators.sol_delay_steps"),
        pfc_gain_sigma=_finite_float(raw.get("pfc_gain_sigma", 0.0), "randomization.actuators.pfc_gain_sigma"),
        sol_gain_sigma=_finite_float(raw.get("sol_gain_sigma", 0.0), "randomization.actuators.sol_gain_sigma"),
        pfc_bias_sigma=_finite_float(raw.get("pfc_bias_sigma", 0.0), "randomization.actuators.pfc_bias_sigma"),
        sol_bias_sigma=_finite_float(raw.get("sol_bias_sigma", 0.0), "randomization.actuators.sol_bias_sigma"),
        pfc_command_noise_sigma=_finite_float(raw.get("pfc_command_noise_sigma", 0.0), "randomization.actuators.pfc_command_noise_sigma"),
        sol_command_noise_sigma=_finite_float(raw.get("sol_command_noise_sigma", 0.0), "randomization.actuators.sol_command_noise_sigma"),
    )
    out.validate()
    return out


def _load_randomization_sensors(raw: Mapping[str, Any]) -> SensorRealismSettings:
    _reject_unknown(
        raw,
        {
            "ip_noise_sigma",
            "ip_bias",
            "ip_bias_sigma",
            "ip_delay_steps",
            "active_current_noise_sigma",
            "active_current_bias_sigma",
            "active_current_delay_steps",
            "radii_noise_sigma",
            "radii_bias_sigma",
            "radii_delay_steps",
            "boundary_xy_noise_sigma",
            "boundary_delay_steps",
            "psi_noise_sigma",
        },
        "randomization.sensors",
    )
    out = SensorRealismSettings(
        ip_noise_sigma=_finite_float(raw.get("ip_noise_sigma", 0.0), "randomization.sensors.ip_noise_sigma"),
        ip_bias=_finite_float(raw.get("ip_bias", 0.0), "randomization.sensors.ip_bias"),
        ip_bias_sigma=_finite_float(raw.get("ip_bias_sigma", 0.0), "randomization.sensors.ip_bias_sigma"),
        ip_delay_steps=_int(raw.get("ip_delay_steps", 0), "randomization.sensors.ip_delay_steps"),
        active_current_noise_sigma=_finite_float(raw.get("active_current_noise_sigma", 0.0), "randomization.sensors.active_current_noise_sigma"),
        active_current_bias_sigma=_finite_float(raw.get("active_current_bias_sigma", 0.0), "randomization.sensors.active_current_bias_sigma"),
        active_current_delay_steps=_int(raw.get("active_current_delay_steps", 0), "randomization.sensors.active_current_delay_steps"),
        radii_noise_sigma=_finite_float(raw.get("radii_noise_sigma", 0.0), "randomization.sensors.radii_noise_sigma"),
        radii_bias_sigma=_finite_float(raw.get("radii_bias_sigma", 0.0), "randomization.sensors.radii_bias_sigma"),
        radii_delay_steps=_int(raw.get("radii_delay_steps", 0), "randomization.sensors.radii_delay_steps"),
        boundary_xy_noise_sigma=_finite_float(raw.get("boundary_xy_noise_sigma", 0.0), "randomization.sensors.boundary_xy_noise_sigma"),
        boundary_delay_steps=_int(raw.get("boundary_delay_steps", 0), "randomization.sensors.boundary_delay_steps"),
        psi_noise_sigma=_finite_float(raw.get("psi_noise_sigma", 0.0), "randomization.sensors.psi_noise_sigma"),
    )
    out.validate()
    return out


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = _safe_yaml_load(f.read())
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def _safe_yaml_load(text: str) -> object:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Tiny fallback parser for this repo's simple config YAML files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith(" "):
            indent = len(raw_line) - len(raw_line.lstrip(" "))
        else:
            indent = 0
        line = raw_line.strip()
        if ":" not in line:
            if indent > 0:
                continue
            raise ValueError(f"Unsupported YAML line {lineno}: {raw_line!r}")
        key, value_raw = line.split(":", 1)
        key = key.strip()
        value_raw = value_raw.strip()
        if not key:
            raise ValueError(f"Empty YAML key at line {lineno}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_raw == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_raw)
    return root


def _parse_scalar(value: str) -> object:
    if value in {"null", "None", "~"}:
        return None
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(c in value for c in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _require_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_path(raw: Mapping[str, Any], key: str, *, base_dir: Path) -> Path:
    if key not in raw:
        raise ValueError(f"Missing required path field: sim.{key}")
    return _resolve_existing_path(raw[key], base_dir=base_dir, field=f"sim.{key}")


def _optional_path(value: object, *, base_dir: Path, field: str) -> Path | None:
    if value is None:
        return None
    return _resolve_existing_path(value, base_dir=base_dir, field=field)


def _optional_output_path(value: object, *, base_dir: Path, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir.parent.parent / path
    return path.resolve()


def _resolve_existing_path(value: object, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidates = [base_dir / path, base_dir.parent.parent / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out < 0:
        raise ValueError(f"{field} must be >= 0")
    return out


def _bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{field} must be a boolean")
