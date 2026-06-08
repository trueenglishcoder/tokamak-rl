from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tokamak_rl.env.config import EnvConfig
from tokamak_rl.env.tokamak_env import _clip_boundary_parameters_to_bounds, _coerce_seed, _margin_below, _mix_seed, _sample_interval, _termination_config_metadata
from tokamak_rl.observations import ObservationSchema
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward


@dataclass(slots=True)
class BatchedGpuReset:
    observations: np.ndarray
    infos: list[dict[str, Any]]


@dataclass(slots=True)
class BatchedGpuStep:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]


class TrueBatchedGpuTokamakEnv:
    """True batched CUDA environment for tokamak RL training."""

    is_true_batched_gpu_env = True

    def __init__(self, cfg: EnvConfig, *, num_envs: int, reward_fn: JointCurrentBoundaryReward | None = None, randomizer: DomainRandomizer | None = None) -> None:
        if cfg.compute_backend != "gpu":
            raise ValueError("TrueBatchedGpuTokamakEnv requires compute_backend == 'gpu'")
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be > 0")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.reward_fn = JointCurrentBoundaryReward() if reward_fn is None else reward_fn
        self.randomizer = DomainRandomizer() if randomizer is None else randomizer
        self.device = torch.device(cfg.gpu_device or "cuda:0")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("true batched GPU environment requires a usable CUDA device")
        self._load_machine()
        self.schema = ObservationSchema(self.n_active_total, int(cfg.angles), version=cfg.observation_version, target_preview_steps=int(cfg.target_preview_steps))
        self.previous_action_norm = torch.zeros((self.num_envs, self.n_active_total), dtype=torch.float64, device=self.device)
        self.episode_indices = np.zeros((self.num_envs,), dtype=int)
        self.measured_boundary_missing = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self.terminated_flags = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._initial_metadata: list[dict[str, object]] = [{} for _ in range(self.num_envs)]
        self._reference_metadata: list[dict[str, object]] = [{} for _ in range(self.num_envs)]
        self._reference_args: list[dict[str, object]] = [dict(self.cfg.scenario_args) for _ in range(self.num_envs)]

    @property
    def obs_dim(self) -> int:
        return self.schema.obs_dim

    @property
    def action_dim(self) -> int:
        return self.n_active_total

    def reset_batch(self, seeds: np.ndarray | list[int]) -> BatchedGpuReset:
        seeds_arr = np.asarray(seeds, dtype=int).reshape(-1)
        if seeds_arr.shape != (self.num_envs,):
            raise ValueError(f"seeds must have shape ({self.num_envs},)")
        initial_ip = np.zeros((self.num_envs,), dtype=float)
        pfc = np.zeros((self.num_envs, self.n_pfc), dtype=float)
        sol = np.zeros((self.num_envs, self.n_sol), dtype=float)
        initial_metadata: list[dict[str, object]] = []
        reference_metadata: list[dict[str, object]] = []
        reference_args: list[dict[str, object]] = []
        for index, seed in enumerate(seeds_arr):
            ip, pfc_i, sol_i, meta = self._sample_initial_state(int(seed))
            args, ref_meta = self._scenario_args_for_reset(int(seed), initial_metadata=meta)
            initial_ip[index] = ip
            pfc[index] = pfc_i
            sol[index] = sol_i
            initial_metadata.append(meta)
            reference_metadata.append(ref_meta)
            reference_args.append(args)
        self._initial_metadata = initial_metadata
        self._reference_metadata = reference_metadata
        self._reference_args = [dict(args) for args in reference_args]
        self.previous_action_norm.zero_()
        self.measured_boundary_missing.zero_()
        self.terminated_flags.zero_()
        self.episode_indices[:] = 0
        reset_result = self.sim.reset(ip=initial_ip, pfc_currents=pfc, sol_currents=sol)
        self._build_references(reference_args, initial_ip)
        obs_t = self._build_observations(reset_result.boundary)
        infos = [self._reset_info(i, reset_result.boundary) for i in range(self.num_envs)]
        return BatchedGpuReset(observations=self._cpu(obs_t).astype(np.float32, copy=False), infos=infos)

    def reset_indices(self, indices: list[int], seeds: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if len(indices) != len(seeds):
            raise ValueError("indices and seeds must have the same length")
        if not indices:
            return np.zeros((0, self.obs_dim), dtype=np.float32), []
        ref_args_all = [dict(args) for args in self._reference_args]
        reset_ip = np.zeros((len(indices),), dtype=float)
        reset_pfc = np.zeros((len(indices), self.n_pfc), dtype=float)
        reset_sol = np.zeros((len(indices), self.n_sol), dtype=float)
        for local, (idx, seed) in enumerate(zip(indices, seeds, strict=True)):
            initial_ip, pfc_i, sol_i, meta = self._sample_initial_state(int(seed))
            args, ref_meta = self._scenario_args_for_reset(int(seed), initial_metadata=meta)
            reset_ip[local] = initial_ip
            reset_pfc[local] = pfc_i
            reset_sol[local] = sol_i
            self._initial_metadata[int(idx)] = meta
            self._reference_metadata[int(idx)] = ref_meta
            self.episode_indices[int(idx)] += 1
            ref_args_all[int(idx)] = args
            self._reference_args[int(idx)] = dict(args)
        reset_result = self.sim.reset_indices(indices, ip=reset_ip, pfc_currents=reset_pfc, sol_currents=reset_sol)
        initial_ip_all = self._cpu(self.sim.state.ip)
        self._build_references(ref_args_all, initial_ip_all)
        self.previous_action_norm[indices] = 0.0
        self.measured_boundary_missing[indices] = 0
        self.terminated_flags[indices] = False
        obs_t = self._build_observations(reset_result.boundary)
        infos = [self._reset_info(int(i), reset_result.boundary) for i in indices]
        return self._cpu(obs_t[indices]).astype(np.float32, copy=False), infos

    def step_batch(self, actions: np.ndarray) -> BatchedGpuStep:
        action_norm = torch.as_tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float64, device=self.device).reshape(self.num_envs, self.n_active_total)
        action_norm = torch.clamp(action_norm, -1.0, 1.0)
        physical = action_norm * self.derivative_scale_t[None, :]
        previous_action = self.previous_action_norm.clone()
        result = self.sim.step(physical)
        step = result.state.step.long()
        ref_ip, ref_radii, ref_points = self._reference_at_steps(step)
        reward, components = self._reward_tensor(result, ref_ip=ref_ip, ref_points=ref_points, action_norm=action_norm, previous_action=previous_action)
        self.previous_action_norm = action_norm
        obs_t = self._build_observations(result.boundary)
        reward, terminated, truncated, reasons = self._termination(result, reward=reward, obs=obs_t)
        self.terminated_flags |= terminated
        infos = [self._step_info(i, result, components, reasons) for i in range(self.num_envs)]
        return BatchedGpuStep(
            observations=self._cpu(obs_t).astype(np.float32, copy=False),
            rewards=self._cpu(reward).astype(float, copy=False),
            terminated=self._cpu(terminated).astype(bool, copy=False),
            truncated=self._cpu(truncated).astype(bool, copy=False),
            infos=infos,
        )

    def close(self) -> None:
        return None

    def _load_machine(self) -> None:
        from tokamak_control.compute import ComputeSettings
        from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
        from tokamak_control.io.config_io import load_config

        cfg = load_config(self.cfg.sim_config_path, initial_currents_path=self.cfg.initial_currents_path)
        if cfg.limiter_shape is None:
            raise RuntimeError("true batched GPU env requires limiter geometry")
        if str(cfg.boundary_mode) != "limited":
            raise RuntimeError("true batched GPU env currently supports limited boundary mode only; diverted GPU fixed-angle training is not implemented")
        self.loaded_cfg = cfg
        self.compute_metadata = {
            "backend": "gpu",
            "plant_backend": "batched_gpu",
            "boundary_backend": "fixed_angle_gpu",
            "observation_backend": "gpu_tensor",
            "reward_backend": "gpu_tensor",
            "gpu_device": str(self.device),
        }
        self.n_pfc = int(cfg.pfc.n_coils)
        self.n_sol = int(cfg.sol.n_coils)
        self.n_active_total = self.n_pfc + self.n_sol
        self.center = (float(cfg.physics.R0), float(cfg.physics.Z0))
        self.angles_rad = np.linspace(-np.pi, np.pi, int(self.cfg.angles), endpoint=False, dtype=float)
        self.angles_t = torch.as_tensor(self.angles_rad, dtype=torch.float64, device=self.device)
        self.dir_t = torch.stack((torch.cos(self.angles_t), torch.sin(self.angles_t)), dim=1)
        self.sim = BatchedGpuTokamakSimulator.from_settings(
            grid=cfg.grid,
            pfc=cfg.pfc,
            sol=cfg.sol,
            settings=cfg.physics,
            batch_size=self.num_envs,
            angles_rad=self.angles_rad,
            limiter_shape=cfg.limiter_shape,
            gpu_device=str(self.device),
        )
        self.current_limits = np.concatenate([_limit_vector(cfg.physics.pfc_current_limit, self.n_pfc), _limit_vector(cfg.physics.sol_current_limit, self.n_sol)])
        self.derivative_limits = np.concatenate([_limit_vector(cfg.physics.pfc_deriv_limit, self.n_pfc), _limit_vector(cfg.physics.sol_deriv_limit, self.n_sol)])
        current_scale = np.where(np.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, 1.0)
        derivative_scale = np.where(np.isfinite(self.derivative_limits) & (self.derivative_limits > 0.0), self.derivative_limits, 1.0)
        self.current_scale = current_scale.astype(float)
        self.derivative_scale = derivative_scale.astype(float)
        self.current_scale_t = torch.as_tensor(self.current_scale, dtype=torch.float64, device=self.device)
        self.derivative_scale_t = torch.as_tensor(self.derivative_scale, dtype=torch.float64, device=self.device)
        self.ip_scale = float(self.cfg.initial_ip_scale or max(abs(float(cfg.physics.Ip0)), 1.0))
        limiter_pts = np.asarray(cfg.limiter_shape, dtype=float).reshape(-1, 2)
        self.radius_scale = max(float(np.max(np.linalg.norm(limiter_pts - np.asarray(self.center, dtype=float), axis=1))), 1.0)

    def _sample_initial_state(self, seed: int) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
        if self.cfg.initial_coil_currents != "sample_ranges":
            pfc = np.zeros((self.n_pfc,), dtype=float) if self.cfg.initial_coil_currents == "zero" else np.asarray(self.loaded_cfg.pfc.initial_currents, dtype=float)
            sol = np.zeros((self.n_sol,), dtype=float) if self.cfg.initial_coil_currents == "zero" else np.asarray(self.loaded_cfg.sol.initial_currents, dtype=float)
            ip = float(self.loaded_cfg.physics.Ip0 if self.cfg.initial_ip is None else self.cfg.initial_ip)
            return ip, pfc, sol, {"mode": str(self.cfg.initial_coil_currents), "shot": None, "initial_currents_path": None, "initial_ip": ip}
        if self.cfg.range_initial_state is None:
            raise RuntimeError("sample_ranges initial state requires range_initial_state")
        rng = np.random.default_rng(int(seed) + 9_104_729)
        ranges = self.cfg.range_initial_state
        ip = _sample_interval(rng, ranges.ip)
        pfc = np.asarray([_sample_interval(rng, item) for item in ranges.pfc_currents], dtype=float)
        sol = np.asarray([_sample_interval(rng, item) for item in ranges.sol_currents], dtype=float)
        boundary = {name: _sample_interval(rng, interval) for name, interval in ranges.boundary_parameters.items()}
        return ip, pfc, sol, {"mode": "sample_ranges", "shot": "synthetic", "initial_currents_path": None, "initial_ip": float(ip), "initial_pfc_currents": pfc.tolist(), "initial_sol_currents": sol.tolist(), "initial_boundary_parameters": boundary}

    def _scenario_args_for_reset(self, seed: int, *, initial_metadata: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        args = dict(self.cfg.scenario_args)
        static_boundary_from_initial = bool(args.pop("boundary_static_from_initial_state", False))
        if self.cfg.scenario_name == "t15_synthetic_follow":
            args["ip_start"] = float(initial_metadata["initial_ip"])
            boundary_initial = initial_metadata.get("initial_boundary_parameters")
            if isinstance(boundary_initial, dict):
                if static_boundary_from_initial and str(args.get("boundary_kind", "generated_parameters")) == "static_parameters":
                    args["boundary_parameters"] = dict(boundary_initial)
                elif str(args.get("boundary_kind", "generated_parameters")) == "generated_parameters":
                    args["boundary_initial_parameters"] = _clip_boundary_parameters_to_bounds(dict(boundary_initial), args.get("boundary_bounds"))
        if self.cfg.scenario_name != "t15_synthetic_follow" or not self.cfg.resample_references_on_reset:
            return args, {"reference_resampling_enabled": False, "boundary_static_from_initial_state": static_boundary_from_initial}
        episode_seed = int(seed)
        base_shape_seed = _coerce_seed(args.get("seed", 0), "scenario_args.seed")
        base_ip_seed = _coerce_seed(args.get("ip_seed", base_shape_seed + 1_000_003), "scenario_args.ip_seed")
        shape_seed = _mix_seed(base_shape_seed, episode_seed, salt=0x13579BDF)
        ip_seed = _mix_seed(base_ip_seed, episode_seed, salt=0x2468ACE0)
        args["seed"] = shape_seed
        args["ip_seed"] = ip_seed
        return args, {"reference_resampling_enabled": True, "reference_base_seed": base_shape_seed, "reference_base_ip_seed": base_ip_seed, "reference_episode_seed": episode_seed, "reference_effective_seed": shape_seed, "reference_effective_ip_seed": ip_seed, "boundary_static_from_initial_state": static_boundary_from_initial}

    def _build_references(self, args_by_env: list[dict[str, object]], initial_ip: np.ndarray) -> None:
        from tokamak_control.config.scenarios import make_scenario

        horizon = int(self.cfg.max_episode_steps) + int(self.cfg.target_preview_steps) * int(self.cfg.target_preview_stride) + 2
        times = np.arange(horizon, dtype=float) * float(self.loaded_cfg.physics.t_step)
        ip = np.zeros((self.num_envs, horizon), dtype=float)
        radii = np.zeros((self.num_envs, horizon, int(self.cfg.angles)), dtype=float)
        for env_index, args in enumerate(args_by_env):
            scenario = make_scenario(self.cfg.scenario_name, np.ones((int(self.cfg.angles),), dtype=float), float(initial_ip[env_index]), params=args, center=self.center)
            for step, t in enumerate(times):
                ip[env_index, step] = float(scenario.Ip_ref(float(t)))
                radii[env_index, step] = np.asarray(scenario.ref_radii(self.angles_rad, float(t)), dtype=float).reshape(-1)
        self.ip_ref = torch.as_tensor(ip, dtype=torch.float64, device=self.device)
        self.radii_ref = torch.as_tensor(radii, dtype=torch.float64, device=self.device)
        center = torch.tensor(self.center, dtype=torch.float64, device=self.device)
        self.points_ref = center[None, None, None, :] + self.radii_ref[:, :, :, None] * self.dir_t[None, None, :, :]

    def _reference_at_steps(self, steps):
        idx = torch.clamp(steps.long(), 0, self.ip_ref.shape[1] - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self.ip_ref[env_idx, idx], self.radii_ref[env_idx, idx], self.points_ref[env_idx, idx]

    def _build_observations(self, boundary) -> torch.Tensor:
        state = self.sim.state
        step = state.step.long()
        ip_ref, radii_ref, _points_ref = self._reference_at_steps(step)
        b = self.num_envs
        boundary_valid = boundary.found.to(torch.float64)
        measured_radii = torch.where(boundary.found[:, None], boundary.radii, torch.zeros_like(boundary.radii))
        radii_error = torch.where(boundary.found[:, None], measured_radii - radii_ref, torch.zeros_like(radii_ref))
        currents = torch.cat((state.pfc_currents, state.sol_currents), dim=1)
        parts = [
            (step.to(torch.float64) / max(float(self.cfg.max_episode_steps), 1.0)).reshape(b, 1),
            boundary_valid.reshape(b, 1),
            (state.ip / max(float(self.ip_scale), 1.0)).reshape(b, 1),
            (ip_ref / max(float(self.ip_scale), 1.0)).reshape(b, 1),
            ((state.ip - ip_ref) / max(float(self.ip_scale), 1.0)).reshape(b, 1),
            currents / self.current_scale_t[None, :],
            measured_radii / max(float(self.radius_scale), 1.0),
            radii_ref / max(float(self.radius_scale), 1.0),
            radii_error / max(float(self.radius_scale), 1.0),
            torch.clamp(self.previous_action_norm, -1.0, 1.0),
        ]
        if self.schema.version == "v2":
            offsets = torch.arange(1, int(self.cfg.target_preview_steps) + 1, dtype=torch.int64, device=self.device) * int(self.cfg.target_preview_stride)
            preview_idx = torch.clamp(step[:, None] + offsets[None, :], 0, self.ip_ref.shape[1] - 1)
            env_idx = torch.arange(b, device=self.device)[:, None]
            parts.extend([
                offsets.to(torch.float64).reshape(1, -1).expand(b, -1) / max(float(self.cfg.max_episode_steps), 1.0),
                self.ip_ref[env_idx, preview_idx] / max(float(self.ip_scale), 1.0),
                (self.radii_ref[env_idx, preview_idx].reshape(b, -1) / max(float(self.radius_scale), 1.0)),
            ])
        with torch.no_grad():
            obs = torch.cat(parts, dim=1).to(dtype=torch.float32)
        if obs.shape != (b, self.schema.obs_dim):
            raise RuntimeError(f"batched observation shape {tuple(obs.shape)} != {(b, self.schema.obs_dim)}")
        return obs

    def _reward_tensor(self, result, *, ref_ip, ref_points, action_norm, previous_action):
        state = result.state
        boundary = result.boundary
        ip_error_norm = torch.abs(state.ip - ref_ip) / max(float(self.ip_scale), 1.0)
        r_ip = torch.exp(-((ip_error_norm / max(float(self.reward_fn.ip_tolerance_norm), 1.0e-12)) ** 2))
        distances = torch.linalg.norm(torch.where(boundary.found[:, None, None], boundary.points, ref_points) - ref_points, dim=2)
        shape_error_norm = torch.sqrt(torch.mean(distances**2, dim=1)) / max(float(self.radius_scale), 1.0)
        shape_mean = torch.mean(distances, dim=1) / max(float(self.radius_scale), 1.0)
        shape_max = torch.max(distances, dim=1).values / max(float(self.radius_scale), 1.0)
        r_shape = torch.where(boundary.found, torch.exp(-((shape_error_norm / max(float(self.reward_fn.shape_tolerance_norm), 1.0e-12)) ** 2)), torch.zeros_like(shape_error_norm))
        action_rms = torch.sqrt(torch.mean(action_norm**2, dim=1))
        delta_action_rms = torch.sqrt(torch.mean((action_norm - previous_action) ** 2, dim=1))
        current_limit_penalty = _margin_penalty_tensor(result.current_margin, weight=float(self.reward_fn.current_limit_weight), like=action_rms)
        derivative_limit_penalty = _margin_penalty_tensor(result.derivative_margin, weight=float(self.reward_fn.derivative_limit_weight), like=action_rms)
        reward = float(self.reward_fn.ip_weight) * r_ip + float(self.reward_fn.shape_weight) * r_shape - float(self.reward_fn.action_weight) * action_rms**2 - float(self.reward_fn.delta_action_weight) * delta_action_rms**2 - current_limit_penalty - derivative_limit_penalty
        components = {
            "ip_error_norm": ip_error_norm,
            "shape_error_norm": shape_error_norm,
            "shape_distance_mean_norm": shape_mean,
            "shape_distance_max_norm": shape_max,
            "shape_target_point_count": torch.full_like(shape_error_norm, float(self.cfg.angles)),
            "r_ip": r_ip,
            "r_shape": r_shape,
            "action_rms": action_rms,
            "delta_action_rms": delta_action_rms,
            "current_limit_penalty": current_limit_penalty,
            "derivative_limit_penalty": derivative_limit_penalty,
        }
        return reward, components

    def _termination(self, result, *, reward, obs):
        cfg = self.cfg.termination
        boundary = result.boundary
        step = result.state.step.long()
        self.measured_boundary_missing = torch.where(boundary.found, torch.zeros_like(self.measured_boundary_missing), self.measured_boundary_missing + 1)
        grace = step <= int(cfg.boundary_loss_grace_steps)
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        reasons = [None for _ in range(self.num_envs)]
        boundary_loss = (~boundary.found) & bool(cfg.terminate_on_boundary_loss) & (~grace)
        terminated |= boundary_loss
        nonfinite_reward = (~torch.isfinite(reward)) & bool(cfg.terminate_on_nonfinite_reward)
        nonfinite_obs = (~torch.all(torch.isfinite(obs), dim=1)) & bool(cfg.terminate_on_nonfinite_observation)
        terminated |= nonfinite_reward | nonfinite_obs
        truncated = step >= int(self.cfg.max_episode_steps)
        for i in range(self.num_envs):
            if bool(boundary_loss[i].detach().cpu()):
                reasons[i] = "boundary_not_found"
            elif bool(nonfinite_reward[i].detach().cpu()):
                reasons[i] = "invalid_reward"
            elif bool(nonfinite_obs[i].detach().cpu()):
                reasons[i] = "invalid_observation"
            elif bool(truncated[i].detach().cpu()):
                reasons[i] = "max_episode_steps"
        reward = torch.where(terminated, reward - float(self.reward_fn.termination_penalty), reward)
        return reward, terminated, truncated, reasons

    def _reset_info(self, index: int, boundary) -> dict[str, Any]:
        return {
            "snapshot": self._snapshot(index, boundary),
            "machine": self._machine_metadata(),
            "episode_metadata": self._episode_metadata(index),
        }

    def _step_info(self, index: int, result, components: dict[str, torch.Tensor], reasons: list[str | None]) -> dict[str, Any]:
        comp = {name: float(values[index].detach().cpu()) for name, values in components.items()}
        return {
            "snapshot": self._snapshot(index, result.boundary),
            "reward_components": comp,
            "termination_reason": reasons[index],
            "termination_detail": reasons[index],
            "action_norm": self._cpu(self.previous_action_norm[index]),
        }

    def _snapshot(self, index: int, boundary):
        state = self.sim.state
        step = int(state.step[index].detach().cpu())
        ref_ip, ref_radii, _ = self._reference_at_steps(state.step.long())
        found = bool(boundary.found[index].detach().cpu())
        return SimpleNamespace(
            step_index=step,
            time_s=float(state.time_s[index].detach().cpu()),
            true_ip=float(state.ip[index].detach().cpu()),
            measured_ip=float(state.ip[index].detach().cpu()),
            measured_active_currents=self._cpu(torch.cat((state.pfc_currents[index], state.sol_currents[index]), dim=0)),
            measured_radii=None if not found else self._cpu(boundary.radii[index]),
            true_radii=None if not found else self._cpu(boundary.radii[index]),
            true_boundary_poly=None,
            measured_boundary_poly=None,
            boundary_found=found,
            boundary_reason=None if found else "No fixed-angle limited boundary found",
            current_limit_margin=None if self.sim.current_margin() is None else self._cpu(self.sim.current_margin()[index]),
            derivative_limit_margin=None if self.sim.derivative_margin() is None else self._cpu(self.sim.derivative_margin()[index]),
            reference=SimpleNamespace(ip_ref=float(ref_ip[index].detach().cpu()), radii_ref=self._cpu(ref_radii[index])),
        )

    def _episode_metadata(self, index: int) -> dict[str, object]:
        return {
            **self._reference_metadata[index],
            "sampled_initial_state": self._initial_metadata[index],
            "randomization": {"enabled": False, "seed": index},
            "compute": dict(self.compute_metadata),
            "training_contract": {
                "contract_version": "training_readiness_v1",
                "simulator": {"boundary_mode": str(self.loaded_cfg.boundary_mode), "t_step": float(self.loaded_cfg.physics.t_step), "compute_backend": "gpu"},
                "environment": {"scenario_name": str(self.cfg.scenario_name), "angles": int(self.cfg.angles), "max_episode_steps": int(self.cfg.max_episode_steps)},
                "observation_schema": self.schema.to_metadata(),
                "target_preview": {"steps": int(self.cfg.target_preview_steps), "stride": int(self.cfg.target_preview_stride)},
                "gpu_env_pool": {"enabled": True, "backend": "true_batched_gpu", "pool_size": int(self.num_envs)},
            },
            "gpu_env_pool": {"enabled": True, "backend": "true_batched_gpu", "pool_size": int(self.num_envs), "slot_index": int(index), "process_envs": False},
        }

    def _machine_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(n_active_total=self.n_active_total, angles_rad=self.angles_rad, compute_backend="gpu")

    def _cpu(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().copy()


class TrueBatchedGpuEnvFactory:
    is_true_batched_gpu_factory = True

    def __init__(self, env_config: EnvConfig, *, num_envs: int, reward_fn: JointCurrentBoundaryReward | None = None, randomizer: DomainRandomizer | None = None) -> None:
        self.env_config = env_config
        self.num_envs = int(num_envs)
        self.reward_fn = reward_fn
        self.randomizer = randomizer
        self.env: TrueBatchedGpuTokamakEnv | None = None

    def __call__(self) -> TrueBatchedGpuTokamakEnv:
        if self.env is None:
            self.env = TrueBatchedGpuTokamakEnv(self.env_config, num_envs=self.num_envs, reward_fn=self.reward_fn, randomizer=self.randomizer)
        return self.env


def _limit_vector(limit: float | None, size: int) -> np.ndarray:
    if int(size) <= 0:
        return np.zeros((0,), dtype=float)
    if limit is None or not np.isfinite(float(limit)) or float(limit) <= 0.0:
        return np.full((int(size),), np.nan, dtype=float)
    return np.full((int(size),), float(limit), dtype=float)


def _margin_penalty_tensor(margin, *, weight: float, like):
    if margin is None or float(weight) == 0.0:
        return torch.zeros_like(like)
    finite = torch.isfinite(margin)
    deficits = torch.where(finite, torch.clamp(0.10 - margin, min=0.0) ** 2, torch.zeros_like(margin))
    counts = torch.clamp(torch.sum(finite.to(dtype=margin.dtype), dim=1), min=1.0)
    return float(weight) * torch.sum(deficits, dim=1) / counts
