# Workflows

This document records current user-facing `tokamak-rl` workflows.

## Install

Expected sibling layout:

```text
tokamak/
  tokamak-sim/
  tokamak-rl/
```

`tokamak-rl` installs or mounts `tokamak-sim` as a Python dependency.

Local editable install from the `tokamak/` sibling layout:

```bash
cd tokamak-rl
python -m pip install -e ../tokamak-sim
python -m pip install -e '.[train,dev]'
```

## Docker Training

`tokamak-rl` has a dedicated training Docker image. It installs `tokamak-sim` during image build and the compose file mounts the sibling simulator repository read-only at runtime. RL training does not require launching the `tokamak-sim` Docker service.

Build the RL image from the `tokamak-rl` directory:

```bash
docker compose build tokamak-rl
```

Run a short no-noise static-circle smoke training on CPU:

```bash
docker compose run --rm tokamak-rl \
  python scripts/train.py \
    --config configs/experiments/t15md_training_circle_static_boundary.yaml \
    --trainer tcv_style \
    --steps 10 \
    --num-envs 2 \
    --warmup-steps 2 \
    --batch-size 1 \
    --sequence-length 2 \
    --hidden-dim 16 \
    --critic-hidden-dim 16 \
    --critic-mlp-hidden-dim 16 \
    --mpo-action-samples 4 \
    --mpo-temperature-iterations 3 \
    --updates-per-env-step 1 \
    --updates-per-episode 1 \
    --eval-episodes 2 \
    --eval-max-steps 10 \
    --eval-randomization-mode clean \
    --device cpu \
    --output-dir outputs/zero_start_circle_10x2 \
    --checkpoint-dir outputs/zero_start_circle_10x2/checkpoints \
    --checkpoint-interval-steps 5 \
    --max-step-checkpoints 2
```

On a GPU server with the NVIDIA container runtime installed, use the GPU profile:

```bash
docker compose --profile gpu run --rm tokamak-rl-gpu \
  python scripts/train.py \
    --config configs/experiments/t15md_training_real_replay_like.yaml \
    --device cuda
```

Artifacts are written into mounted local paths such as `outputs/` and `checkpoints/`.

## Run Tests

Run the regression and smoke tests from the `tokamak-rl` root:

```bash
python -m pytest -q
```

Focused checks can be run by test file:

```bash
python -m pytest -q tests/test_env_reset.py
python -m pytest -q tests/test_rollouts.py
python -m pytest -q tests/test_actor_export.py
python -m pytest -q tests/test_tcv_style_trainer.py
python -m pytest -q tests/test_synthetic_training_contract.py
```

Some tests create a small temporary simulator config and use `tokamak-sim` APIs directly. The config-loader test expects the sibling `tokamak-sim` repository and local simulator config files to exist. `test_synthetic_training_contract.py` covers tiny real synthetic-reference training smoke paths for both the simple trainer and the TCV-style recurrent-critic trainer. These tests load `configs/experiments/t15md_synthetic_joint_current_boundary.yaml`, shorten the episode in memory, and verify diagnostics plus provenance artifacts.

## Load Experiment Config

The tested experiment config is:

```text
configs/experiments/t15md_joint_current_boundary.yaml
```

It resolves the simulator config from the sibling `tokamak-sim` repository and points to reward and randomization presets.

The loader accepts referenced config files:

```text
reward_config
randomization_config
```

The loaded `reward_config` is passed into `TokamakRLEnv` through evaluation and training CLIs, so reward weights in YAML affect actual rollout/training rewards. The loaded `randomization_config` is passed into the same runtime paths, records sampled episode metadata, and passes supported sensor/actuator perturbations into tokamak-sim. Plant-parameter perturbations remain absent until tokamak-sim exposes explicit hooks for them.

and simulator fields:

```text
config_path
initial_currents_path
scenario_name
scenario_args
angles
max_episode_steps
realism_enabled
```

## Observation Schema

The current actor observation is schema version `v1`.

For `n_active_total` active actuators and `n_angles` boundary/radii sample angles, the observation dimension is:

```text
5 + 2 * n_active_total + 3 * n_angles
```

The current fields are:

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

For T15-style use with `n_active_total = 18` and `n_angles = 32`, the observation dimension is:

```text
82
```

## Run A Single Rollout

Rollout evaluation is implemented through `tokamak_rl.evaluation.rollouts.run_rollout_evaluation`.

The current tested policy modes are:

```text
zero
random
```

The rollout writer produces:

```text
summary.json
episode_metrics.csv
rollouts.npz
```

The rollout NPZ stores actions, rewards, observations, terminated/truncated flags, mask, true/measured Ip, `ip_ref`, `radii_ref`, boundary-found flags, and measured-boundary availability flags.

## Run Evaluation

Use the evaluation script for zero-policy or random-policy rollouts:

```bash
python scripts/evaluate.py   --config configs/experiments/t15md_joint_current_boundary.yaml   --out outputs/evaluation   --episodes 1   --policy zero   --seed 0
```

The script prints:

```text
output directory
summary.json path
episode_metrics.csv path
rollouts.npz path
```

Use the random policy by changing:

```bash
--policy random
```

## Train A TCV-Style Policy

Use the TCV-style recurrent-critic trainer:

```bash
python scripts/train.py   --trainer tcv_style   --config configs/experiments/t15md_joint_current_boundary.yaml   --steps 500   --warmup-steps 100   --batch-size 16   --sequence-length 64   --hidden-dim 256   --critic-hidden-dim 256   --critic-mlp-hidden-dim 256   --num-envs 1   --updates-per-episode 1   --device auto   --output-dir outputs/train_tcv_style   --checkpoint-dir checkpoints/train_tcv_style   --eval-episodes 1   --eval-max-steps 200   --seed 0
```

The experiment YAML can also provide the trainer, evaluation, artifact, checkpoint, device, and worker settings directly. For the real T15 end-to-end smoke path:

```bash
python scripts/train.py   --config configs/experiments/t15md_training_real_replay_like_smoke.yaml
```

This trainer uses a deployable feedforward actor and twin recurrent critics during training. Its update rule is `tcv_mpo_recurrent_actor_critic_v1`, with sampled-action MPO policy improvement, KL-constrained actor fitting, and recurrent critic updates over sequence chunks. Use `--device cpu`, `--device cuda`, or `--device auto` to control learner placement. Use `--process-envs --num-envs N` to run each real tokamak-sim training environment in its own CPU worker process while the main process batches actor inference and learner updates. It writes training metrics and losses when `--output-dir` is set, including device and throughput fields, and writes final/latest/best checkpoints when `--checkpoint-dir` is set. For real tokamak environments that expose the training contract, checkpointed runs also export `exports/best_actor/` from the selected `best.pt` checkpoint.

The training CLI shows a terminal progress bar by default with step count, percent complete, step rate, ETA, replay/update counts, episode count, and latest losses. Use `--no-progress` for log-only or noninteractive runs.

Validation randomization is explicit. Set `evaluation.randomization_mode: clean` to run deterministic validation with simulator realism disabled and a disabled `DomainRandomizer`, or `configured` to reuse the experiment randomization settings. The CLI override is `--eval-randomization-mode clean|configured`.

Resume from a checkpoint:

```bash
python scripts/train.py   --trainer tcv_style   --config configs/experiments/t15md_joint_current_boundary.yaml   --steps 500   --resume-checkpoint checkpoints/train_tcv_style/tcv_style_checkpoint.pt   --output-dir outputs/train_tcv_style_resumed   --checkpoint-dir checkpoints/train_tcv_style_resumed
```

## Train A Simple Actor-Critic Policy

Use the simple feedforward actor-critic trainer for smoke testing:

```bash
python scripts/train.py   --trainer simple   --config configs/experiments/t15md_joint_current_boundary.yaml   --steps 500   --warmup-steps 100   --batch-size 64   --hidden-dim 256   --num-envs 1   --output-dir outputs/train_simple   --checkpoint-dir checkpoints/train_simple   --eval-episodes 1   --eval-max-steps 200   --seed 0
```

Resume from a checkpoint:

```bash
python scripts/train.py   --trainer simple   --config configs/experiments/t15md_joint_current_boundary.yaml   --steps 500   --resume-checkpoint checkpoints/train_simple/checkpoint.pt   --output-dir outputs/train_simple_resumed   --checkpoint-dir checkpoints/train_simple_resumed
```

## Training Outputs

When `--output-dir` is provided, training writes:

```text
metrics.json
losses.csv
config_snapshot.json
episodes.csv
eval_history.csv
reward_components.csv
reference_samples.npz
termination_counts.json
artifact_manifest.json
rollouts/
```

`metrics.json` includes `tracking_diagnostics`. When the environment provides simulator reward components and snapshots, this block includes mean normalized Ip error, mean normalized boundary-shape error, boundary failure rate, and first/last effective reference seeds seen during training. Final and periodic deterministic evaluations add `eval_tracking_diagnostics` and `tracking_diagnostics` entries inside `eval_history` rows. Metrics, config snapshots, and checkpoints include `run_metadata` so artifacts record the experiment config path, translated simulator scenario/reference args, reward config, and randomization config.

`reward_components.csv` is a streamed artifact with columns:

```text
step,env_index,episode,component,value
```

`episodes.csv` is a per-episode diagnostic artifact with return, length, termination/truncation fields, mean tracking errors, boundary failure steps, reference seeds, and randomization seed. Training also writes `reference_samples.npz`, `termination_counts.json`, and `rollouts/eval_step_*/summary.json`, `episode_metrics.csv`, and `rollouts.npz` for deterministic evaluation events. If no periodic evaluation is configured, the final deterministic evaluation is still represented under `rollouts/eval_step_<total_steps>/`.

When `--checkpoint-dir` is provided, the simple trainer writes:

```text
checkpoint.pt
latest.pt
best.pt
step_XXXXXXXX.pt when --checkpoint-interval-steps is configured
```

and the TCV-style trainer writes:

```text
tcv_style_checkpoint.pt
latest.pt
best.pt
step_XXXXXXXX.pt when --checkpoint-interval-steps is configured
```

Use `--max-step-checkpoints N` or `artifacts.max_step_checkpoints` to retain only the newest N numbered `step_*.pt` checkpoints. `latest.pt`, `best.pt`, and the final checkpoint are not pruned by that setting. Real tokamak environment checkpoints include the reset training contract plus observation schema, action schema, and normalization metadata used for later export.

The training CLI prints steps, replay size or replay episode counts, latest losses, evaluation returns, final/latest/best checkpoint paths, and metrics path.
For exported runs it also prints `best_actor_export=...` and records `best_actor_export_dir` in `metrics.json`.

Optional W&B logging is enabled with `--wandb` or a top-level experiment YAML `wandb:` block. The trainer logs scalar train rewards, reward components, replay/update counters, critic and actor losses, TCV-style MPO diagnostics, per-episode summaries, deterministic evaluation summaries, and final metrics. By default W&B artifact logging uploads the final metrics/losses files plus checkpoint/export paths that exist; disable that with `--wandb-no-artifacts` or `wandb.log_artifacts: false` when checkpoints are too large. Use `--wandb-mode offline` for server runs that should sync later, and `--wandb-log-interval-steps N` to reduce scalar logging frequency.

`scripts/search_rewards.py` runs short candidate trainings for reward-value search. It writes one candidate directory per reward config, ranks candidates by deterministic evaluation tracking diagnostics rather than raw return, and emits `results.csv`, `results.json`, `search_manifest.json`, and `best_reward.yaml`. Candidate checkpoints are disabled unless `--save-checkpoints` is provided.

## Export A Policy

Use the export script to write a deterministic NumPy-compatible actor artifact:

```bash
python scripts/export_policy.py   --out outputs/exported_policy   --obs-dim 82   --action-dim 18   --n-active-total 18   --n-angles 32
```

Add a checkpoint to export a trained actor state dict or a full trainer checkpoint:

```bash
python scripts/export_policy.py   --out outputs/exported_policy   --obs-dim 82   --action-dim 18   --n-active-total 18   --n-angles 32   --checkpoint checkpoints/train_tcv_style/best.pt
```

The export script validates that `--obs-dim` matches the dimension produced by `ObservationSchema(n_active_total, n_angles)`.

The export path writes:

```text
policy_weights.npz
schema.json
normalization.json
metadata.json
```

The exported NumPy actor is tested for parity against the PyTorch deterministic actor.

## Inspect Artifacts

Expected local artifact folders:

```text
runs/
output/
outputs/
checkpoints/
```

Rollout artifacts include summary JSON, episode metrics CSV, and rollout NPZ.

Training artifacts include checkpoints, metrics JSON, losses CSV, config snapshot JSON, evaluation returns, evaluation history, episode summaries, an artifact manifest, and streamed reward component rows.

Export artifacts include NumPy weights and metadata JSON files.

## Generate Or Load Reference Trajectories

Reference generation uses simulator-side trajectory logic instead of duplicating it in `tokamak-rl`. Experiment configs may provide `sim.reference_source`; the loader translates it into the `tokamak-sim` `t15_synthetic_follow` scenario.

Example using real T15 Ip tables as templates:

```yaml
sim:
  config_path: ../tokamak-sim/configs/T15MD_new_data.toml
  initial_currents_path: ../tokamak-sim/configs/initial_currents/T15MD_new_data_3864.toml
  reference_source:
    kind: t15_synthetic_follow
    seed: 11
    duration_s: 1.0
    t_step: 0.001
    target_update_s: 0.20
    theta_count: 512
    ip:
      kind: template_dir
      path: ../tokamak-sim/data/t15_data_new/ip
      seed: 101
      amplitude_jitter: 0.05
      duration_jitter: 0.05
      shape_jitter: 0.02
```

Supported Ip source kinds are:

```text
template_dir  -> ip_template_dir plus optional seed/jitter
template_csv  -> ip_template_csv plus optional seed/jitter
csv           -> direct ip_csv plus optional time_offset
ramp          -> ip_start/ip_end/ip_ramp_s
segmented     -> continuous ramp/hold segments with value, duration, count, length, and rate bounds
```

Boundary source configuration lives under `reference_source.boundary`:

```yaml
boundary:
  kind: generated_parameters
  bounds:
    R0: {min: 1.3266, max: 1.4756}
    Z0: {min: -0.0496, max: 0.0137}
    A0: {min: 0.5200, max: 0.6641}
    kappa: {min: 1.1212, max: 1.4966}
    delta: {min: 0.1044, max: 0.3656}
  rate_limits:
    R0: 0.30
    Z0: 0.70
    A0: 0.45
    kappa: 1.20
    delta: 0.80
```

For static starter curricula, use fixed boundary parameters:

```yaml
boundary:
  kind: static_parameters
  parameters:
    R0: 1.40
    Z0: 0.0
    A0: 0.55
    kappa: 1.0
    delta: 0.0
```

The main training presets are:

```text
configs/experiments/t15md_training_real_replay_like.yaml
configs/experiments/t15md_training_circle_static_boundary.yaml
```

`reference_source` is mutually exclusive with manual `scenario_name` and `scenario_args`; this keeps training configs from defining two competing reference contracts.

By default, deterministic per-reset resampling is enabled for `t15_synthetic_follow`. The environment combines the base shape/Ip seeds from the config with the reset seed and passes effective `seed`/`ip_seed` values into `tokamak-sim`. Resetting with the same seed reproduces the same reference; resetting with a different seed produces a different synthetic reference. Disable this with:

```yaml
sim:
  resample_references_on_reset: false
```

The reset `info["episode_metadata"]` records `reference_base_seed`, `reference_base_ip_seed`, `reference_episode_seed`, `reference_effective_seed`, and `reference_effective_ip_seed` when resampling is active. Rollout/evaluation artifacts carry these effective seeds into `episode_metrics.csv`, and store `ip_ref` plus `radii_ref` in `rollouts.npz` for post-run target diagnostics. Reset metadata also records `info["episode_metadata"]["randomization"]`, passes supported noise settings into tokamak-sim, and carries `randomization_enabled` plus `randomization_seed` into rollout episode metrics.

Target-preview observations are configured with:

```yaml
sim:
  observation:
    version: v2
    target_preview_steps: 8
    target_preview_stride: 10
```

V2 observations include future reference time offsets, future Ip targets, and future boundary radii targets. The simulator bridge supplies these future references without advancing the plasma state.

Termination rules are configured with:

```yaml
sim:
  termination:
    terminate_on_boundary_loss: true
    terminate_on_nonfinite_observation: true
    terminate_on_nonfinite_reward: true
    current_limit_margin_min: null
    derivative_limit_margin_min: null
    measured_boundary_missing_steps: null
```

`termination_reason` is a stable name such as `boundary_not_found`, `measured_boundary_missing`, `current_limit_breach`, or `derivative_limit_breach`. `termination_detail` keeps the simulator or rule-specific detail string.
