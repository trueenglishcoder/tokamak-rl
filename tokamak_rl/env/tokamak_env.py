from __future__ import annotations

import numpy as np

from tokamak_rl.actions import ActionScaler
from tokamak_rl.contracts import KNOWN_TERMINATION_REASONS, TRAINING_READINESS_CONTRACT_VERSION
from tokamak_rl.env.config import EnvConfig
from tokamak_rl.observations import ObservationSchema
from tokamak_rl.observations.builder import build_observation
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward, boundary_points_from_radii


class TokamakRLEnv:
    """Gymnasium-style environment backed by tokamak-sim's bridge API.

    The concrete `reset`/`step` contract is implemented here while keeping the
    dependency on Gymnasium optional at import time.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: EnvConfig,
        reward_fn: JointCurrentBoundaryReward | None = None,
        randomizer: DomainRandomizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.session = None
        self.machine = None
        self.schema: ObservationSchema | None = None
        self.action_scaler: ActionScaler | None = None
        self.reward_fn = JointCurrentBoundaryReward() if reward_fn is None else reward_fn
        self.randomizer = DomainRandomizer() if randomizer is None else randomizer
        self.previous_action_norm: np.ndarray | None = None
        self._reset_count = 0
        self._terminated = False
        self._termination_reason: str | None = None
        self._measured_boundary_missing_count = 0
        self._configure_simulator_profiling()

    def _configure_simulator_profiling(self) -> None:
        try:
            from tokamak_control.core.plasma_model import configure_plasma_model_profiling
            from tokamak_control.core.gpu_plasma_model import configure_gpu_plasma_model_profiling
            from tokamak_control.geometry.boundary import configure_boundary_profiling
        except Exception:
            return
        configure_plasma_model_profiling(enabled=True, summary_every=0, reset=False)
        configure_gpu_plasma_model_profiling(enabled=True, summary_every=0, reset=False)
        configure_boundary_profiling(enabled=True, summary_every=0, reset=False)

    def reset(self, seed: int | None = None, options: dict | None = None):
        from tokamak_control.bridge import InitialStateOverride, SimulationSession

        _ = options
        initial_currents_path, initial_ip, initial_metadata, initial_pfc_currents, initial_sol_currents = self._initial_state_for_reset(seed)
        scenario_args, reference_metadata = self._scenario_args_for_reset(seed, initial_metadata=initial_metadata)
        randomization_sample = self.randomizer.sample_episode(seed=seed)
        randomization_metadata = dict(randomization_sample.metadata)
        self.session = SimulationSession.from_paths(
            config_path=self.cfg.sim_config_path,
            initial_currents_path=initial_currents_path,
            scenario_name=self.cfg.scenario_name,
            scenario_args=scenario_args,
            angles=self.cfg.angles,
            steps=self.cfg.max_episode_steps,
            seed=seed,
            realism_enabled=self.cfg.realism_enabled,
            realism_settings=randomization_sample.realism_settings,
            initial_state_override=InitialStateOverride(
                ip=initial_ip,
                coil_currents=(
                    "explicit"
                    if initial_pfc_currents is not None and initial_sol_currents is not None
                    else ("config" if self.cfg.initial_coil_currents == "sample_replay" else self.cfg.initial_coil_currents)
                ),
                ip_scale=self.cfg.initial_ip_scale,
                pfc_currents=initial_pfc_currents,
                sol_currents=initial_sol_currents,
            ),
            compute_backend=self.cfg.compute_backend,
            gpu_device=self.cfg.gpu_device,
        )
        reset = self.session.reset(seed=seed, realism_settings=randomization_sample.realism_settings)
        self._reset_count += 1
        self.machine = reset.machine
        self.schema = ObservationSchema(
            reset.machine.n_active_total,
            len(reset.machine.angles_rad),
            version=self.cfg.observation_version,
            target_preview_steps=int(self.cfg.target_preview_steps),
        )
        self.action_scaler = ActionScaler(reset.machine.derivative_scale)
        self.previous_action_norm = np.zeros((reset.machine.n_active_total,), dtype=float)
        self._terminated = False
        self._termination_reason = None
        self._measured_boundary_missing_count = 0
        episode_metadata = {
            **dict(reset.episode_metadata),
            **reference_metadata,
            "sampled_initial_state": initial_metadata,
            "randomization": randomization_metadata,
        }
        episode_metadata["training_contract"] = self._training_contract_metadata(
            machine=reset.machine,
            reference_metadata=reference_metadata,
            randomization_metadata=randomization_metadata,
        )
        obs = self._observation(reset.observation_snapshot)
        info = {
            "snapshot": reset.observation_snapshot,
            "machine": reset.machine,
            "episode_metadata": episode_metadata,
            "config_path": self.cfg.sim_config_path,
            "initial_currents_path": initial_currents_path,
        }
        return obs, info

    @property
    def action_dim(self) -> int:
        if self.action_scaler is None:
            raise RuntimeError("reset() must be called before action_dim is available")
        return self.action_scaler.action_dim

    @property
    def obs_dim(self) -> int:
        if self.schema is None:
            raise RuntimeError("reset() must be called before obs_dim is available")
        return self.schema.obs_dim

    def step(self, action_norm: np.ndarray):
        from tokamak_control.bridge import DerivativeAction

        if self.session is None or self.action_scaler is None or self.previous_action_norm is None or self.machine is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError(f"environment is terminated: {self._termination_reason}")
        action = np.asarray(action_norm, dtype=float).reshape(-1)
        physical = self.action_scaler.to_physical(action)
        clipped_action = self.action_scaler.to_normalized(physical)
        previous_action = self.previous_action_norm.copy()
        result = self.session.step_derivatives(DerivativeAction(physical))
        reward, components = self.reward_fn(
            true_ip=result.snapshot.true_ip,
            ip_ref=result.snapshot.reference.ip_ref,
            ip_scale=self.machine.ip_scale,
            true_boundary_poly=result.snapshot.true_boundary_poly,
            reference_boundary_points=boundary_points_from_radii(
                center=self.machine.center,
                angles_rad=self.machine.angles_rad,
                radii=result.snapshot.reference.radii_ref,
            ),
            radius_scale=self.machine.radius_scale,
            action_norm=clipped_action,
            previous_action_norm=previous_action,
            current_limit_margin=result.snapshot.current_limit_margin,
            derivative_limit_margin=result.snapshot.derivative_limit_margin,
            terminated=bool(result.terminated),
        )
        self.previous_action_norm = clipped_action
        obs = self._observation(result.snapshot)
        termination_reason, termination_detail = self._termination_from_step_result(result, reward=reward, observation=obs)
        terminated = bool(result.terminated or termination_reason is not None)
        if terminated:
            self._terminated = True
            self._termination_reason = termination_reason
        info = {
            "snapshot": result.snapshot,
            "reward_components": components,
            "termination_reason": termination_reason,
            "termination_detail": termination_detail,
            "action_norm": clipped_action.copy(),
            "physical_derivatives": physical.copy(),
        }
        return obs, float(reward), terminated, bool(result.truncated), info

    def _scenario_args_for_reset(self, seed: int | None, *, initial_metadata: dict[str, object] | None = None) -> tuple[dict[str, object], dict[str, object]]:
        args = dict(self.cfg.scenario_args)
        static_boundary_from_initial = bool(args.pop("boundary_static_from_initial_state", False))
        if initial_metadata is not None and initial_metadata.get("mode") in {"sample_replay", "sample_ranges"}:
            if self.cfg.scenario_name == "t15_synthetic_follow":
                args["ip_start"] = float(initial_metadata["initial_ip"])
                boundary_initial = initial_metadata.get("initial_boundary_parameters")
                if isinstance(boundary_initial, dict):
                    if static_boundary_from_initial and str(args.get("boundary_kind", "generated_parameters")) == "static_parameters":
                        args["boundary_parameters"] = dict(boundary_initial)
                    elif str(args.get("boundary_kind", "generated_parameters")) == "generated_parameters":
                        args["boundary_initial_parameters"] = _clip_boundary_parameters_to_bounds(
                            dict(boundary_initial),
                            args.get("boundary_bounds"),
                        )
        if self.cfg.scenario_name != "t15_synthetic_follow" or not self.cfg.resample_references_on_reset:
            return args, {"reference_resampling_enabled": False, "boundary_static_from_initial_state": static_boundary_from_initial}

        episode_seed = int(self._reset_count if seed is None else seed)
        base_shape_seed = _coerce_seed(args.get("seed", 0), "scenario_args.seed")
        base_ip_seed = _coerce_seed(args.get("ip_seed", base_shape_seed + 1_000_003), "scenario_args.ip_seed")
        shape_seed = _mix_seed(base_shape_seed, episode_seed, salt=0x13579BDF)
        ip_seed = _mix_seed(base_ip_seed, episode_seed, salt=0x2468ACE0)
        args["seed"] = shape_seed
        args["ip_seed"] = ip_seed
        return args, {
            "reference_resampling_enabled": True,
            "reference_base_seed": base_shape_seed,
            "reference_base_ip_seed": base_ip_seed,
            "reference_episode_seed": episode_seed,
            "reference_effective_seed": shape_seed,
            "reference_effective_ip_seed": ip_seed,
            "boundary_static_from_initial_state": static_boundary_from_initial,
        }

    def _initial_state_for_reset(self, seed: int | None):
        if self.cfg.initial_coil_currents == "sample_ranges":
            if self.cfg.range_initial_state is None:
                raise RuntimeError("sample_ranges initial state requires range_initial_state config")
            episode_seed = int(self._reset_count if seed is None else seed)
            rng = np.random.default_rng(episode_seed + 9_104_729)
            ranges = self.cfg.range_initial_state
            initial_ip = _sample_interval(rng, ranges.ip)
            pfc_currents = np.asarray([_sample_interval(rng, item) for item in ranges.pfc_currents], dtype=float)
            sol_currents = np.asarray([_sample_interval(rng, item) for item in ranges.sol_currents], dtype=float)
            boundary_parameters = {
                name: _sample_interval(rng, interval)
                for name, interval in ranges.boundary_parameters.items()
            }
            metadata = {
                "mode": "sample_ranges",
                "shot": "synthetic",
                "initial_currents_path": None,
                "initial_ip": float(initial_ip),
                "initial_pfc_currents": pfc_currents.tolist(),
                "initial_sol_currents": sol_currents.tolist(),
                "initial_boundary_parameters": boundary_parameters,
            }
            return self.cfg.initial_currents_path, float(initial_ip), metadata, pfc_currents, sol_currents

        if self.cfg.initial_coil_currents != "sample_replay":
            return self.cfg.initial_currents_path, self.cfg.initial_ip, {
                "mode": str(self.cfg.initial_coil_currents),
                "shot": None,
                "initial_currents_path": None if self.cfg.initial_currents_path is None else str(self.cfg.initial_currents_path),
                "initial_ip": None if self.cfg.initial_ip is None else float(self.cfg.initial_ip),
            }, None, None
        if self.cfg.replay_initial_state is None:
            raise RuntimeError("sample_replay initial state requires replay_initial_state config")
        candidates = self.cfg.replay_initial_state.candidates
        episode_seed = int(self._reset_count if seed is None else seed)
        rng = np.random.default_rng(episode_seed + 9_104_729)
        candidate = candidates[int(rng.integers(0, len(candidates)))]
        metadata = {
            "mode": "sample_replay",
            "shot": str(candidate.shot),
            "initial_currents_path": str(candidate.initial_currents_path),
            "initial_ip": float(candidate.initial_ip),
            "initial_boundary_parameters": dict(candidate.initial_boundary_parameters),
        }
        return candidate.initial_currents_path, float(candidate.initial_ip), metadata, None, None

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
        self.session = None

    def _training_contract_metadata(
        self,
        *,
        machine,
        reference_metadata: dict[str, object],
        randomization_metadata: dict[str, object],
    ) -> dict[str, object]:
        if self.schema is None or self.action_scaler is None:
            raise RuntimeError("environment schema/action scaler must be initialized before contract metadata")
        return {
            "contract_version": TRAINING_READINESS_CONTRACT_VERSION,
            "simulator": {
                "config_path": str(self.cfg.sim_config_path),
                "initial_currents_path": None if self.cfg.initial_currents_path is None else str(self.cfg.initial_currents_path),
                "replay_initial_state_candidates": 0 if self.cfg.replay_initial_state is None else len(self.cfg.replay_initial_state.candidates),
                "range_initial_state_enabled": self.cfg.range_initial_state is not None,
                "boundary_mode": str(machine.boundary_mode),
                "limiter_name": machine.limiter_name,
                "t_step": float(machine.t_step),
            },
            "environment": {
                "scenario_name": str(self.cfg.scenario_name),
                "scenario_args": dict(self.cfg.scenario_args),
                "angles": int(self.cfg.angles),
                "max_episode_steps": int(self.cfg.max_episode_steps),
                "realism_enabled": bool(self.cfg.realism_enabled),
                "resample_references_on_reset": bool(self.cfg.resample_references_on_reset),
                "initial_ip": None if self.cfg.initial_ip is None else float(self.cfg.initial_ip),
                "initial_coil_currents": str(self.cfg.initial_coil_currents),
                "initial_ip_scale": None if self.cfg.initial_ip_scale is None else float(self.cfg.initial_ip_scale),
            },
            "reference": _reference_contract_metadata(
                scenario_name=self.cfg.scenario_name,
                scenario_args=self.cfg.scenario_args,
                reference_metadata=reference_metadata,
            ),
            "randomization": dict(randomization_metadata),
            "observation_schema": self.schema.to_metadata(),
            "normalization": {
                "ip_scale": float(machine.ip_scale),
                "radius_scale": float(machine.radius_scale),
                "current_scale": np.asarray(machine.current_scale, dtype=float).tolist(),
                "derivative_scale": np.asarray(machine.derivative_scale, dtype=float).tolist(),
                "phase": "step_index / max_episode_steps",
            },
            "target_preview": {
                "steps": int(self.cfg.target_preview_steps),
                "stride": int(self.cfg.target_preview_stride),
            },
            "action_schema": {
                "action_dim": int(self.action_scaler.action_dim),
                "action_range": [-1.0, 1.0],
                "active_order": list(machine.active_order),
                "derivative_scale": np.asarray(machine.derivative_scale, dtype=float).tolist(),
            },
            "termination": {"known_reasons": list(KNOWN_TERMINATION_REASONS)},
            "termination_config": _termination_config_metadata(self.cfg.termination),
        }

    def _observation(self, snapshot) -> np.ndarray:
        if self.schema is None or self.machine is None or self.previous_action_norm is None:
            raise RuntimeError("environment is not initialized")
        preview = self._reference_preview(snapshot)
        return build_observation(
            schema=self.schema,
            step_index=snapshot.step_index,
            max_episode_steps=self.cfg.max_episode_steps,
            measured_ip=snapshot.measured_ip,
            ip_ref=snapshot.reference.ip_ref,
            ip_scale=self.machine.ip_scale,
            measured_active_currents=snapshot.measured_active_currents,
            current_scale=self.machine.current_scale,
            measured_radii=snapshot.measured_radii,
            radii_ref=snapshot.reference.radii_ref,
            target_preview_time_norm=preview["time_norm"],
            ip_ref_preview=preview["ip_ref"],
            radii_ref_preview=preview["radii_ref"],
            radius_scale=self.machine.radius_scale,
            previous_action_norm=self.previous_action_norm,
        )

    def _reference_preview(self, snapshot) -> dict[str, np.ndarray | None]:
        if self.schema is None or self.schema.version == "v1":
            return {"time_norm": None, "ip_ref": None, "radii_ref": None}
        if self.session is None or self.machine is None:
            raise RuntimeError("environment session is not initialized")
        preview_steps = int(self.schema.target_preview_steps)
        stride = int(self.cfg.target_preview_stride)
        step_offsets = np.arange(1, preview_steps + 1, dtype=float) * float(stride)
        times = float(snapshot.time_s) + step_offsets * float(self.machine.t_step)
        frames = [self.session.reference_at_time(float(t)) for t in times]
        return {
            "time_norm": step_offsets / max(float(self.cfg.max_episode_steps), 1.0),
            "ip_ref": np.asarray([frame.ip_ref for frame in frames], dtype=float),
            "radii_ref": np.stack([np.asarray(frame.radii_ref, dtype=float).reshape(-1) for frame in frames], axis=0),
        }

    def _termination_from_step_result(self, result, *, reward: float, observation: np.ndarray) -> tuple[str | None, str | None]:
        cfg = self.cfg.termination
        snapshot = result.snapshot
        if snapshot.measured_boundary_poly is None or snapshot.measured_radii is None:
            self._measured_boundary_missing_count += 1
        else:
            self._measured_boundary_missing_count = 0
        boundary_loss_grace_active = int(snapshot.step_index) <= int(cfg.boundary_loss_grace_steps)
        if (not bool(snapshot.boundary_found)) and bool(cfg.terminate_on_boundary_loss) and not boundary_loss_grace_active:
            return "boundary_not_found", snapshot.boundary_reason or result.termination_reason
        if bool(result.terminated):
            return "simulator_terminated", result.termination_reason
        if (not boundary_loss_grace_active) and cfg.measured_boundary_missing_steps is not None and self._measured_boundary_missing_count >= int(cfg.measured_boundary_missing_steps):
            return "measured_boundary_missing", f"measured boundary missing for {self._measured_boundary_missing_count} consecutive steps"
        if cfg.current_limit_margin_min is not None and _margin_below(snapshot.current_limit_margin, float(cfg.current_limit_margin_min)):
            return "current_limit_breach", f"current limit margin below {float(cfg.current_limit_margin_min):.6g}"
        if cfg.derivative_limit_margin_min is not None and _margin_below(snapshot.derivative_limit_margin, float(cfg.derivative_limit_margin_min)):
            return "derivative_limit_breach", f"derivative limit margin below {float(cfg.derivative_limit_margin_min):.6g}"
        if bool(cfg.terminate_on_nonfinite_reward) and not np.isfinite(float(reward)):
            return "invalid_reward", "reward is nonfinite"
        if bool(cfg.terminate_on_nonfinite_observation) and not np.all(np.isfinite(np.asarray(observation, dtype=float))):
            return "invalid_observation", "observation contains nonfinite values"
        return None, None


def _coerce_seed(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer seed")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer seed") from exc


def _mix_seed(base_seed: int, episode_seed: int, *, salt: int) -> int:
    value = (int(base_seed) & 0xFFFFFFFF)
    value ^= (int(episode_seed) + int(salt)) & 0xFFFFFFFF
    value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return int(value)


def _clip_boundary_parameters_to_bounds(parameters: dict[str, object], bounds: object) -> dict[str, float]:
    out = {str(key): float(value) for key, value in parameters.items()}
    if not isinstance(bounds, dict):
        return out
    for key in ("R0", "Z0", "A0", "kappa", "delta"):
        raw_bound = bounds.get(key)
        if not isinstance(raw_bound, dict) or key not in out:
            continue
        if "min" in raw_bound and "max" in raw_bound:
            out[key] = float(np.clip(out[key], float(raw_bound["min"]), float(raw_bound["max"])))
    return out


def _reference_contract_metadata(
    *,
    scenario_name: str,
    scenario_args: dict[str, object],
    reference_metadata: dict[str, object],
) -> dict[str, object]:
    source_kind = "t15_synthetic_follow" if scenario_name == "t15_synthetic_follow" else "scenario"
    return {
        "source_kind": source_kind,
        "scenario_name": str(scenario_name),
        "scenario_args": dict(scenario_args),
        "resampling_enabled": bool(reference_metadata.get("reference_resampling_enabled", False)),
        "base_seed": reference_metadata.get("reference_base_seed"),
        "base_ip_seed": reference_metadata.get("reference_base_ip_seed"),
        "episode_seed": reference_metadata.get("reference_episode_seed"),
        "effective_seed": reference_metadata.get("reference_effective_seed"),
        "effective_ip_seed": reference_metadata.get("reference_effective_ip_seed"),
    }


def _termination_config_metadata(config) -> dict[str, object]:
    return {
        "terminate_on_boundary_loss": bool(config.terminate_on_boundary_loss),
        "boundary_loss_grace_steps": int(config.boundary_loss_grace_steps),
        "terminate_on_nonfinite_observation": bool(config.terminate_on_nonfinite_observation),
        "terminate_on_nonfinite_reward": bool(config.terminate_on_nonfinite_reward),
        "current_limit_margin_min": config.current_limit_margin_min,
        "derivative_limit_margin_min": config.derivative_limit_margin_min,
        "measured_boundary_missing_steps": config.measured_boundary_missing_steps,
    }


def _margin_below(margin: np.ndarray | None, threshold: float) -> bool:
    if margin is None:
        return False
    values = np.asarray(margin, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.any(finite < float(threshold)))


def _sample_interval(rng: np.random.Generator, interval: tuple[float, float]) -> float:
    low = float(interval[0])
    high = float(interval[1])
    if low == high:
        return low
    return float(rng.uniform(low, high))
