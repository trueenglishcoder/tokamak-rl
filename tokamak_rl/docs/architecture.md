# Architecture

This document summarizes the current `tokamak-rl` architecture.

## Repository Role

`tokamak-rl` is the companion research and training repository for reinforcement-learning tokamak control.

`tokamak-sim` owns the simulator, machine configuration loading, physical boundary extraction, run artifact writing, realism support, and programmatic bridge.

`tokamak-rl` owns training-side logic:

- environment wrappers
- action normalization
- observation construction
- reward calculation
- training-side randomization
- replay buffers
- policy and critic networks
- training loops
- rollout evaluation
- deterministic policy export

## Runtime Pipeline

The current environment flow is:

```text
load experiment config
construct EnvConfig
create TokamakRLEnv
reset SimulationSession
receive simulator snapshot and machine metadata
build measured observation
step with normalized action
clip and scale action to physical derivative vector
send DerivativeAction to tokamak-sim bridge
receive next simulator snapshot
compute reward from true simulator channels and references
return observation, reward, terminated, truncated, info
```

The environment exposes a Gym-style API:

```text
obs, info = env.reset(seed=...)
obs, reward, terminated, truncated, info = env.step(action)
```

The environment info dictionary includes simulator snapshot data, machine metadata, physical derivatives, normalized action, reward components, and termination reason when applicable.

`training_readiness_contract.md` defines the runtime and artifact contract. Environment reset metadata includes a nested `training_contract` block with simulator, environment, reference, randomization, observation, action, and termination metadata. Trainer runs with `output_dir` write the artifact filenames required by that contract.

Termination rules are configurable under `sim.termination`. Environment steps return stable `termination_reason` names and preserve raw simulator/rule text in `termination_detail`.

## Simulator Dependency

`tokamak-rl` depends on `tokamak-sim` as a Python package.

The simulator is accessed through public bridge objects from `tokamak_control.bridge`, especially `SimulationSession`, `DerivativeAction`, and bridge snapshot/result dataclasses.

The reward uses true simulator values. Actor observations use measured simulator channels when realism is active.

## Configuration

Experiment config loading is handled by `tokamak_rl.config.load_experiment_config`.

The config package exports:

```python
from tokamak_rl.config import ExperimentConfig, load_experiment_config
```

The top-level experiment config requires a non-empty `name` and a `sim` mapping. The `sim` mapping is converted into `EnvConfig`.

Supported simulator config fields include:

```text
config_path
initial_currents_path
initial_state
scenario_name
scenario_args
reference_source
angles
max_episode_steps
realism_enabled
resample_references_on_reset
observation
termination
```

`initial_state` can override simulator startup state for training. Debug presets may start from `Ip = 0` and zero active coil currents. Real-replay-like training presets use `ip: sample_replay` and `coil_currents: sample_replay` to sample initial `Ip` and active coil currents from the fitted T15MD replay-start pool, while keeping an explicit Ip normalization scale.

`reference_source` is the training-facing trajectory-source block. The current supported source is `kind: t15_synthetic_follow`, which is translated to the simulator's `t15_synthetic_follow` scenario. It is mutually exclusive with manual `scenario_name`/`scenario_args`. When `resample_references_on_reset` is true, `TokamakRLEnv.reset(seed=...)` derives effective shape and Ip seeds from the configured base seeds and the episode reset seed.

Ip generation can use `template_dir`, `template_csv`, `csv`, `ramp`, or `segmented`. The `segmented` mode creates continuous no-jump trajectories from ramp/hold segments with configured value bounds, segment duration bounds, segment count bounds, maximum steps, rate limit, start value, and seed. Boundary generation can use `generated_parameters` with configured `R0`, `Z0`, `A0`, `kappa`, and `delta` bounds/rate limits, or `static_parameters` for fixed shapes such as the circular-boundary starter preset.

The main training presets are:

```text
configs/experiments/t15md_training_real_replay_like.yaml
configs/experiments/t15md_training_circle_static_boundary.yaml
```

The loader also supports referenced reward and randomization config files:

```text
reward_config
randomization_config
```

Reward config fields are checked against `JointCurrentBoundaryReward` dataclass fields, and unknown reward fields are rejected. The loaded reward instance is passed into `TokamakRLEnv` for rollout evaluation and training CLI runs; direct environment construction defaults to `JointCurrentBoundaryReward()` unless a reward is supplied.

Randomization config supports `enabled`, `actuators`, and `sensors` blocks matching tokamak-sim's runtime realism/noise settings. The loaded `DomainRandomizer` is passed into `TokamakRLEnv` for rollout evaluation and training CLI runs. Each reset records the sampled randomization contract in metadata and passes simulator-backed sensor/actuator perturbations into `SimulationSession`.

The current experiment config path used by tests is:

```text
configs/experiments/t15md_joint_current_boundary.yaml
```

Reward presets live under:

```text
configs/rewards/
```

Randomization presets live under:

```text
configs/randomization/
```

The config loader uses PyYAML when available and has a minimal fallback parser for simple YAML mappings.

## Environment

Environment code lives in `tokamak_rl.env`.

The environment package exports:

```python
from tokamak_rl.env import EnvConfig, TokamakRLEnv
```

`EnvConfig` contains:

```text
sim_config_path
initial_currents_path
initial_ip
initial_coil_currents
initial_ip_scale
scenario_name
scenario_args
angles
max_episode_steps
realism_enabled
resample_references_on_reset
observation_version
target_preview_steps
target_preview_stride
termination
```

`TokamakRLEnv` keeps Gymnasium optional. It implements the concrete reset/step contract directly.

On reset, it constructs `SimulationSession.from_paths`, resets the simulator, stores machine metadata, builds `ObservationSchema`, creates `ActionScaler` from the simulator derivative scale, initializes previous normalized action to zero, and returns the first observation plus metadata.

On step, it converts normalized policy action to physical derivatives, sends them to the simulator bridge, computes reward from true channels and references, updates the previous normalized action, builds the next observation, and returns a Gym-style tuple.

## Actions

Action normalization lives in `tokamak_rl.actions`.

`ActionScaler` maps normalized action vectors to physical active-coil derivative commands. Inputs are clipped to `[-1, 1]`, validated for shape and finiteness, and scaled by derivative limits. Invalid derivative scales require an explicit fallback scale.

## Observations

Observation construction lives in `tokamak_rl.observations`.

The package exports:

```python
from tokamak_rl.observations import ObservationSchema
```

`ObservationSchema` defines the fixed flat observation layout for schema version `v1`.

The current field order is:

```text
phase_norm
boundary_valid
ip_meas_norm
ip_ref_norm
ip_error_norm
active_currents_meas_norm
radii_meas_norm
radii_ref_norm
radii_error_norm
previous_action_norm
```

The observation dimension is:

```text
5 + 2 * n_active_total + 3 * n_angles
```

Field sizes are:

```text
phase_norm                 1
boundary_valid             1
ip_meas_norm               1
ip_ref_norm                1
ip_error_norm              1
active_currents_meas_norm  n_active_total
radii_meas_norm            n_angles
radii_ref_norm             n_angles
radii_error_norm           n_angles
previous_action_norm       n_active_total
```

`build_observation` validates scalar, active-current, radii, reference, and previous-action inputs. It normalizes Ip by `ip_scale`, active currents by `current_scale`, radii by `radius_scale`, and clips previous normalized actions to `[-1, 1]`.

Missing measured boundary/radii data sets `boundary_valid` to zero and fills measured radii and radii error fields with zeros. Stale boundary values are not reused.

`ObservationSchema.to_metadata()` exports schema version, observation dimension, active actuator count, angle count, field order, and field sizes for policy export.

Schema version `v2` adds target-trajectory preview. V2 keeps all V1 fields and appends:

```text
target_preview_time_norm
ip_ref_preview_norm
radii_ref_preview_norm
```

`target_preview_steps` and `target_preview_stride` are configured under `sim.observation` in experiment YAML. `TokamakRLEnv` obtains future reference frames from the simulator bridge without stepping physics, normalizes preview time offsets by episode step count, and records preview settings in `training_contract.target_preview` metadata.

## Rewards

Reward calculation lives in `tokamak_rl.rewards`.

The rewards package exports:

```python
from tokamak_rl.rewards import JointCurrentBoundaryReward
```

`JointCurrentBoundaryReward` is the current reward component calculator for Ip tracking and target-point boundary tracking.

It computes normalized plasma-current error:

```text
abs(true_ip - ip_ref) / ip_scale
```

and converts it to a smooth reward term using `ip_tolerance_norm`.

For boundary shape, it converts simulator reference radii into Cartesian target shape points using the machine center and measurement angles. It then computes the shortest distance from each target point to the true plasma boundary polyline, normalizes the target-point RMSE by `radius_scale`, and converts that geometric error to a smooth reward term using `shape_tolerance_norm`. If the true boundary polyline is missing, the shape reward is zero.

The reward also includes penalties for normalized action RMS, normalized action-change RMS, optional low-margin current and derivative penalties, and termination.

The returned reward components include:

```text
ip_error_norm
shape_error_norm
shape_distance_mean_norm
shape_distance_max_norm
shape_target_point_count
r_ip
r_shape
action_rms
delta_action_rms
current_limit_penalty
derivative_limit_penalty
```

## Randomization

Training-side randomization lives in `tokamak_rl.randomization`.

The package exports:

```python
from tokamak_rl.randomization import DomainRandomizer
```

`DomainRandomizer` provides an episode-level simulator-randomization contract. `sample_episode(seed=...)` returns a `RandomizationSample` with JSON-safe metadata plus optional tokamak-sim `RealismSettings`. Environment reset records this metadata under `info["episode_metadata"]["randomization"]`, passes the settings to `SimulationSession`, and rollout evaluation writes `randomization_enabled` plus `randomization_seed` into `episode_metrics.csv`.

Supported randomization is limited to sensor and actuator perturbations already implemented by tokamak-sim. Plant parameter randomization is intentionally absent until tokamak-sim exposes explicit neutral plant perturbation hooks.

## Evaluation

Rollout evaluation lives in `tokamak_rl.evaluation`.

The evaluation package currently includes a CLI and rollout writer.

The CLI entry point parses:

```text
--config
--out
--episodes
--policy
--seed
```

`run_rollout_evaluation` supports at least zero and random policy modes. It writes summary JSON, episode metrics CSV, and rollout NPZ artifacts when an output directory is provided.

The rollout NPZ includes:

```text
actions
rewards
observations
terminated
truncated
true_ip
measured_ip
boundary_found
measured_boundary_available
mask
```

The rollout summary includes mean return, mean length, termination rate, boundary failure rate, and measured-boundary missing rate. Rollout artifacts also include reference-target channels (`ip_ref`, `radii_ref`) and per-episode effective reference seeds so synthetic-target variation can be audited after evaluation or training smoke runs.

## Networks

Network modules live in `tokamak_rl.networks`.

The package exports:

```python
from tokamak_rl.networks import (
    ActorConfig,
    CriticConfig,
    FeedForwardActor,
    FeedForwardQCritic,
    RecurrentCriticConfig,
    RecurrentQCritic,
)
```

`FeedForwardActor` is the current policy network. It uses a linear input layer, layer normalization, tanh activation, three ELU hidden layers, and separate mean and standard-deviation heads. The training path samples tanh-squashed Gaussian actions for exploration and MPO action-candidate evaluation. The deterministic action path returns `tanh(mean)` and is the evaluation/export path.

`FeedForwardQCritic` is the current feedforward Q critic for the simple off-policy trainer. It concatenates observation and action, applies three feedforward layers, and returns one scalar Q estimate per batch item.

`RecurrentQCritic` is the sequence critic for the TCV-style trainer. It consumes `(observation, action)` sequences, runs an LSTM, concatenates the LSTM output back with the per-step critic input, and returns one Q estimate per batch item and timestep.

## Replay Buffers

Transition replay is implemented by `ReplayBuffer`. It is a fixed-size circular buffer storing observations, actions, rewards, next observations, terminated flags, and truncated flags. It samples random transition batches for the simple trainer.

Sequence replay is implemented by `EpisodeReplayBuffer`. It stores complete episodes and samples padded fixed-length sequence chunks with a boolean mask. The mask marks valid timesteps and prevents padded timesteps from contributing to recurrent critic losses.

## Training

Training modules live in `tokamak_rl.training`.

The training package exports:

```python
ReplayBatch
ReplayBuffer
Episode
EpisodeReplayBuffer
RecurrentUpdateResult
SequenceBatch
SimpleTrainerConfig
TCVStyleTrainerConfig
TCVStyleTrainingResult
TrainingResult
evaluate_actor
evaluate_tcv_actor
load_actor_from_checkpoint
save_training_checkpoint
recurrent_critic_update_once
train_simple_actor_critic
train_tcv_style_actor_critic
```

The training CLI supports two trainer names:

```text
simple
tcv_style
```

CLI training runs show a dependency-free terminal progress bar by default. The bar reports step count, percent complete, step rate, ETA, replay/update counts, episode count, and latest losses; `--no-progress` disables it for log-only runs.

The simple actor-critic path uses `FeedForwardActor`, two `FeedForwardQCritic` instances, target networks, a circular transition replay buffer, random warmup actions, deterministic actor actions with optional Gaussian exploration noise, delayed actor updates, soft target updates, vectorized synchronous environments, checkpoint writing, metrics JSON, losses CSV, and checkpoint resume.

The TCV-style path uses a feedforward actor with twin recurrent critics. It stores complete episodes, samples padded sequence chunks, applies masked recurrent MPO updates, samples stochastic actor actions after warmup, runs updates after completed episodes, writes checkpoints, writes metrics JSON and losses CSV, supports periodic evaluation, and supports checkpoint resume.

Both trainer configs expose `device` as `cpu`, `cuda`, or `auto`. Device resolution is explicit and recorded in `metrics.json`; requesting CUDA on a machine without CUDA raises a clear error. Trainer metrics also include throughput timing for actor inference, environment stepping, replay sampling, learner updates, evaluation, environment steps per second, learner updates per second, and update-to-data ratio. For synchronous `num_envs > 1` collection, actor inference is batched before environment stepping. The training CLI can run real tokamak-sim environments through `ProcessTokamakEnv` worker processes with `--process-envs`; each worker owns one simulator session, while the main learner batches policy inference and updates. TCV-style update cadence is configurable through `updates_per_episode`, `updates_per_env_step`, and `max_learner_catchup_updates`.

The TCV-style trainer's implemented algorithm identity is `tcv_mpo_recurrent_actor_critic_v1`: a deployable feedforward stochastic actor, twin recurrent Q critics used during training, complete-episode replay, masked sequence chunks, sampled-action MPO E-step, KL-constrained actor fitting, and deterministic actor mean for evaluation/export.

Trainer `metrics.json` includes tracking diagnostics accumulated from environment step `reward_components`, simulator snapshots, and reset metadata when available. These diagnostics include mean normalized Ip error, mean normalized boundary-shape error, boundary failure rate, and first/last effective reference seeds observed during training. The same diagnostic path is used for deterministic trainer evaluation through `eval_tracking_diagnostics` and per-interval `eval_history[*].tracking_diagnostics`. Metrics, checkpoints, and config snapshots also store JSON-safe `run_metadata` for experiment provenance.

## Recurrent Critic Update

`recurrent_critic_update_once` performs one masked sequence update over a `SequenceBatch`.

It validates `gamma`, `tau`, `policy_delay`, MPO bounds, action sample count, and the sequence mask. It estimates target values from sampled target-actor actions, trains both recurrent critics with masked MSE, then runs the MPO policy-improvement path on the configured actor-update cadence: sample candidate actions, weight them by critic values through the MPO temperature solve, fit the actor with weighted likelihood, apply mean/std KL constraints through learned dual penalties, soft-update critic and actor targets, and return finite loss diagnostics.

## Export

Policy export lives in `tokamak_rl.export`.

The export package exposes:

```python
from tokamak_rl.export import ExportedPolicyPaths, NumpyFeedForwardActor, export_actor, load_numpy_actor
```

`export_actor` writes deterministic actor export artifacts:

```text
policy_weights.npz
schema.json
normalization.json
metadata.json
```

The exported weights include only the deterministic mean path:

```text
input
input_norm
hidden1
hidden2
hidden3
mean_head
```

The stochastic `std_head` is not included in the runtime export.

`NumpyFeedForwardActor` loads the exported artifact and evaluates deterministic actions with NumPy. It validates observation shape, finiteness, expected schema values, required weights, input dimension, and action dimension.

The export CLI accepts observation/action dimensions, active coil count, angle count, output directory, and an optional PyTorch checkpoint path.
