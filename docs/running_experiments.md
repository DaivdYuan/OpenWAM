# Running Experiments

Open-WAM exposes package-owned CLIs for supported training, evaluation, config
inspection, and static validation. Root scripts are limited to documented
utilities and maintained launch wrappers.

## Static Validation First

Validate configs before launching compute:

```bash
open-wam-validate-config configs/experiments/<experiment>.yaml
```

The static validator does not import model code. It is meant to catch missing
sections, enum typos, bad path placeholders, and incompatible public config
choices before GPU time is allocated.

## Training

Training uses the same generic stack across method families:

```bash
open-wam-train --cfg configs/experiments/<experiment>.yaml
```

Before real training, check:

- local dataset paths are configured through `configs/local_paths.yaml`
- checkpoint/artifact aliases are present when required
- the selected optional extras are installed
- WandB or local tracking policy is documented for the run
- the config has an experiment card if it is intended to be reproducible

For current fixed-128 LIBERO post-training runs, two shell launchers remain as
documented convenience wrappers around `open_wam.training.train`:

```bash
CONFIG_NAME=mot_libero_latent_local_video_then_action_heng_compatible \
OPEN_WAM_PRINT_TRAIN_ARGV=1 \
scripts/run_mot_nonjoint_posttrain_libero.sh
```

```bash
CONFIG_NAME=parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible \
OPEN_WAM_PRINT_TRAIN_ARGV=1 \
scripts/run_parallel_stream_posttrain_libero.sh
```

`OPEN_WAM_PRINT_TRAIN_ARGV=1` prints the resolved train argv without launching
training. Remove it only when local data, checkpoint paths, and compute are
ready.

## Offline Evaluation

Use experiment configs for batch-level metrics in this minimal snapshot:

```bash
open-wam-eval \
  --cfg configs/experiments/<experiment>.yaml \
  --device cpu \
  --max-batches 1
```

For real policies, switch the device and batch limits according to the
available compute. Result files should use the versioned result envelope
described in [Reproducibility](reproducibility.md).

## Sanity Checks

This minimal snapshot does not include a package-owned end-to-end sanity
console command. Use static config validation and the public release pytest
subset for CPU-safe checks until a maintained sanity implementation is added.

GPU or simulator checks should be marked and documented as resource gated. They
should skip clearly when the required resource is missing.

## Realtime And Simulator Rollouts

Closed-loop rollouts must avoid future information. The simulator should step
forward in wall-clock-aware time, and the policy should only consume
observations available at the current control step.

Use rollout code only when a simulator adapter and matching config are present.
This minimal import does not include a package-owned closed-loop rollout CLI or
checked-in closed-loop rollout config.

Report:

- target action Hz
- achieved non-fallback action Hz
- fallback action count
- rollout success/failure
- output video path
- exact checkpoint/config used

Do not compare realtime results without documenting planner mode, diffusion
step count, fallback policy, and simulator task/episode identity.
