# Training Readiness Contract

This document defines the runtime and artifact contract for training runs.

The contract version is `training_readiness_v1`. Future changes may extend this contract, but they must not silently remove fields or claim behavior that is outside the current simulator and training interfaces.

## Environment Reset Metadata

Every `TokamakRLEnv.reset(seed=...)` result must return `info["episode_metadata"]` with a nested `training_contract` mapping.

Required `training_contract` fields:

- `contract_version`
- `simulator`
- `environment`
- `reference`
- `randomization`
- `observation_schema`
- `action_schema`
- `termination`

The `simulator` block records:

- `config_path`
- `initial_currents_path`
- `boundary_mode`
- `limiter_name`
- `t_step`

The `environment` block records:

- `scenario_name`
- `scenario_args`
- `angles`
- `max_episode_steps`
- `realism_enabled`
- `resample_references_on_reset`
- `initial_ip`
- `initial_coil_currents`
- `initial_ip_scale`

The `reference` block records the current truth available to the environment:

- `source_kind`
- `scenario_name`
- `scenario_args`
- `resampling_enabled`
- `base_seed`
- `base_ip_seed`
- `episode_seed`
- `effective_seed`
- `effective_ip_seed`

Reference metadata records the configured source and effective per-episode reference seeds. Detailed generated trajectory tables are stored through training/evaluation artifacts rather than embedded wholesale in reset metadata.

The `randomization` block records the sampled randomization contract. Supported randomization settings are connected to tokamak-sim's simulator-side sensor and actuator perturbation hooks. Plant-parameter randomization is not claimed unless explicit simulator plant hooks are added.

The `observation_schema` block must come from `ObservationSchema.to_metadata()`.

The `action_schema` block records:

- `action_dim`
- `action_range`
- `active_order`
- `derivative_scale`

The `termination` block records the known termination reason vocabulary.

`termination_config` records:

- `terminate_on_boundary_loss`
- `boundary_loss_grace_steps`
- `terminate_on_nonfinite_observation`
- `terminate_on_nonfinite_reward`
- `current_limit_margin_min`
- `derivative_limit_margin_min`
- `measured_boundary_missing_steps`

## Step Info

Every `TokamakRLEnv.step(action)` result must return `info` with:

- `snapshot`
- `reward_components`
- `termination_reason`
- `termination_detail`
- `action_norm`
- `physical_derivatives`

Limit margins currently live on `info["snapshot"].current_limit_margin` and `info["snapshot"].derivative_limit_margin`.

`termination_reason` is a stable machine-readable name. `termination_detail` preserves the simulator or rule-specific text.

## Training Artifacts

Every trainer run with `output_dir` must write these artifact files:

```text
metrics.json
config_snapshot.json
losses.csv
episodes.csv
eval_history.csv
reward_components.csv
reference_samples.npz
termination_counts.json
artifact_manifest.json
```

`metrics.json` must include `contract_version`.

`artifact_manifest.json` must include:

- `contract_version`
- `trainer`
- `required_artifacts`
- `present_artifacts`
- `conditional_artifacts`
- `present_conditional_artifacts`

`episodes.csv` records return, length, termination/truncation state, termination reason, mean tracking errors, boundary failure steps, reference seeds, and randomization seed. `reference_samples.npz` records reset-level reference/randomization samples. `termination_counts.json` aggregates episode endings. `rollouts/eval_step_*/` records deterministic evaluation summaries, per-episode returns, and an NPZ payload for each periodic or final evaluation event.

Trainer runs with `checkpoint_dir` use these checkpoint semantics:

- numbered `step_XXXXXXXX.pt` checkpoints when `checkpoint_interval_steps` is configured
- optional numbered checkpoint retention through `max_step_checkpoints`
- `latest.pt` after cadence checkpoints and at the end of training
- `best.pt` selected by deterministic evaluation mean return
- checkpoint payloads with actor, critic, target-network, optimizer, counter, config, metadata, and RNG fields
- real tokamak environment checkpoint payloads also carry `training_contract`, `observation_schema`, `action_schema`, and `normalization`
- `metrics.json` fields for `checkpoint_path`, `latest_checkpoint_path`, and `best_checkpoint_path`

Evaluation randomization mode is explicit in experiment YAML. `evaluation.randomization_mode: configured` evaluates with the configured experiment randomizer; `clean` evaluates with simulator realism disabled and a disabled `DomainRandomizer`. The selected mode is recorded in trainer run metadata.

Real tokamak environment runs that provide the reset training contract also support best-actor export:

- `exports/best_actor/` exported from the selected `best.pt` checkpoint
- export files `policy_weights.npz`, `schema.json`, `normalization.json`, and `metadata.json`
- `metrics.json` field `best_actor_export_dir`
- exported normalization metadata with Ip, radius, active-current, and derivative scales

## Current Boundaries

The current project does not claim:

- plant-parameter randomization from `tokamak-rl` randomization config
- plant model perturbations beyond tokamak-sim's explicit realism hooks
- proven controller quality before a real long training run and evaluation campaign
