# Open-WAM

Open-WAM is a research codebase for studying **where and how to attach action
policy logic** in a world action model while keeping the **video backbone
fixed**.

The current implementation is organized around one constraint:

- the shared visual path should remain LingBot-compatible
- policy attachment structure and placement are the main research variable

## Current Status

The repo currently includes:

- a stage-aware `VisualTower + PolicyVariant + ActionDecoder` stack
- runnable `parallel_stream`, `register_attached`, `video_sequence_policy`,
  `post_latent`, `post_decoded`, `mot`, and `causal_video_prediction` variants
- a LingBot replica backbone as the default shared-core family for real
  multimodal variants
- a shared runtime backbone knob under `backbone.implementation`:
  - `shared_transformer` (default)
  - `dummy` (smoke/legacy only)
- an optional `backbone.load_reference_core_weights` path that loads LingBot
  backbone weights into the shared replica core for
  `register_attached`, `post_latent`, and `post_decoded`
- an exact LingBot parallel-stream runtime path that executes on the same
  shared backbone object used by the other real variants
- a uniform data contract for all sources and policy variants
- a config-driven canonical RGB layout builder
- a dataset registry keyed by `data.dataset_type`
- a real LeRobot-v2 adapter for `physical-intelligence/libero`
- legacy `contract_only` compatibility via config migration into the new stack
- Lightning train/eval wrappers and root experiment YAMLs

The first real dataset path is:

- `physical-intelligence/libero`

## Repo Layout

```text
configs/         runnable experiment YAMLs and local path samples
docs/            public quickstart, CLI, testing, artifact, and release docs
AGENTS.md        repo-level contributor and agent style guide
src/open_wam/third_party/  vendored external modules kept inside the repo
scripts/         thin wrappers, smoke tests, and inspection scripts
src/open_wam/    all source code
```

Important source packages:

- `src/open_wam/configs`: typed config contracts
- `src/open_wam/data`: dataset adapters, collation, and canonical RGB preprocessing
- `src/open_wam/models/visual_tower`: shared visual frontend, core, decode
  boundary, and exact LingBot reference loader
- `src/open_wam/models/policy_variants`: method-specific train/infer behavior
  for `parallel_stream`, `register_attached`, `video_sequence_policy`,
  `post_latent`, `post_decoded`, `mot`, and `causal_video_prediction`
- `src/open_wam/models/action_decoders`: action decoders and losses
- `src/open_wam/models/video_backbone`: backbone config and compatibility contracts
- `src/open_wam/pipelines`: variant pipeline, exact LingBot runner, and rollout helpers
- `src/open_wam/lightning`: Lightning module and datamodule
- `src/open_wam/training`: train entrypoint
- `src/open_wam/evals`: eval entrypoint

## Public Docs

- [Quickstart](docs/quickstart.md): fresh clone to CPU smoke, local path setup,
  and resource matrix
- [Architecture](docs/architecture.md): stable runtime boundary and extension
  contracts
- [Method families](docs/method_families.md): current policy-attachment
  families and how they share runtime infrastructure
- [Benchmarks and data](docs/benchmarks.md): public fixture, LIBERO,
  RoboTwin, CALVIN, action dimensions, and visual layout contracts
- [Running experiments](docs/running_experiments.md): training, evaluation,
  static validation, and resource-gated rollout workflow
- [CLI reference](docs/cli.md): package-owned commands and root script policy
- [Testing](docs/testing.md): pytest markers and CI tiers
- [Artifacts](docs/artifacts.md): local path registry, checkpoint manifests, and
  layout conventions
- [Deployment namespace](docs/deployment_namespace.md): public snapshot boundary
  for deployment-only code
- [Reproducibility](docs/reproducibility.md): result schemas, experiment cards,
  and WandB naming
- [Extension SDK](docs/extension_sdk.md): dataset, policy-variant, and decoder
  registry extension points
- [Experiment cards](docs/experiment_cards.md): method-family result card
  template and current public-card status
- [GitHub Pages](docs/github_pages.md): generated MkDocs site and required
  repository settings

## Design Rules

- Raw-video ingestion lives in the data layer, not in the backbone.
- The shared visual tower should stay stable across policy-attachment experiments.
- Policy variants interact with the backbone through explicit stage contracts, not ad hoc internals.
- Camera names, camera count, layout, action dimension, action horizon, and state dimension should be configurable from YAML.
- Dataset-specific parsing should stay inside dataset adapters registered by `data.dataset_type`.
- Dataset adapters may expose transformed action supervision, not just raw controller deltas.
- All method families should continue to share the same top-level `VariantPipeline -> VisualTower` boundary even when their within-core runtimes differ.
- For the canonical multimodal methods, differences should come from runtime
  programs, sequence semantics, cache policy, and decoders rather than from
  swapping out the transformer object underneath them.

## Trainer and Variant Flow

Training uses one generic Lightning stack:

- [src/open_wam/training/train.py](src/open_wam/training/train.py) loads a root
  experiment config and instantiates one `OpenWAMLightningModule` and one
  `OpenWAMDataModule`
- [src/open_wam/lightning/module.py](src/open_wam/lightning/module.py) converts
  `WAMBatch` into `PolicyTrainBatch` and always calls
  `pipeline.forward_train(...)`
- [src/open_wam/pipelines/variant_pipeline.py](src/open_wam/pipelines/variant_pipeline.py)
  is where the variant actually changes behavior:
  - prepare visual stages
  - let the policy variant prepare train-time artifacts
  - run the variant forward
  - let the action decoder compute the final loss

That means the trainer itself is not variant-specific. Variants change training
semantics by implementing:

- `required_visual_stages()`
- `prepare_train_inputs()`
- `forward_train()`
- `prepare_infer_state()`
- `forward_infer_step()`

inside `src/open_wam/models/policy_variants/`.

The current method split is:

- `parallel_stream` / method 1: exact LingBot train/infer semantics through
  shared-backbone exact runtime programs
- `register_attached` / method 2: shared runtime-program executor with
  structured sequence adapters, structured attention kernels, shared stream
  adapters, and shared stream output heads
- `post_latent` / `post_decoded`: simple feature-attached baselines over the
  same stage-aware pipeline

## Current Diffusion Granularity

Current diffusion behavior is split into three buckets.

Method 1, LingBot:

- `parallel_stream`
- separate video and action schedulers
- one sampled diffusion timestep per **frame**
- video sigma is broadcast across latent channels and spatial positions of that
  frame
- action sigma is broadcast across action channels and `action_per_frame`
  positions of that frame
- loss is reduced and normalized per frame
- the shared backbone executes method 1 through exact runtime programs rather
  than a sidecar transformer module

Method 2, DreamZero-style register-attached on LingBot backbone:

- video latents get their own noise scheduler, targets, and weighted loss
- actions get their own noise scheduler, targets, and weighted loss
- the shared core sees noisy video and noisy action tokens together
- training loss is `video_diffusion_loss + action_diffusion_loss`
- `register_attached`
  - full clean-video teacher-forcing prefix during training
  - one sampled timestep per video frame
  - action timesteps are coupled to future video blocks by default
  - joint inference rollout: update video and action in the same denoising loop
  - `inference.joint_sampler: unipc` by default for DreamZero-style multistep sampling
  - optional shared denoising count via `inference.joint_num_inference_steps`
  - per-stream CFG stays generic:
    - `inference.video_cfg_mode: guided`
    - `inference.action_cfg_mode: conditioned`
  - cache warmup stays generic:
    - `inference.joint_cache_warmup_source`
    - `inference.joint_cache_initial_warmup_anchor`
    - `inference.joint_cache_rollout_warmup_anchor`
  - `inference.joint_observed_video_prefix_frames: 1` keeps the observed
    first frame fixed during inference-time denoising
  - stream tokenizers and flow heads are now backbone-owned shared runtime
    components rather than variant-local modules

Action-only diffusion variants:

- `post_latent`
- `post_decoded`

These still use LingBot-style action flow matching:

- action tensor is `[B, H_action, D_action]`
- one sampled diffusion timestep per **action horizon slot**
- that sigma is broadcast across all `D_action` channels at that slot
- diffusion loss is reduced per slot across action dims, then averaged over
  slots and batch

## Quick Start

Set up the minimal development environment. This installs the core package
surface only; it does not install Torch, Lightning, simulator packages, or
video codecs:

```bash
uv sync --group dev
uv run python -c "import open_wam; print(open_wam.__version__)"
```

Run static config validation without launching model code:

```bash
uv run open-wam-validate-config \
  configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml \
  configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml
```

Inspect a config through the stable package CLI:

```bash
uv run open-wam-inspect-config \
  --cfg configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml
```

For real datasets/checkpoints, create a local path registry:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Replace the `/path/to/...` placeholders. `configs/local_paths.yaml` is
gitignored. See [docs/quickstart.md](docs/quickstart.md) and
[docs/artifacts.md](docs/artifacts.md) for the public path workflow.

Install optional extras only when needed:

```bash
uv sync --extra torch
uv sync --extra train
uv sync --extra eval
uv sync --extra tracking
uv sync --extra viz
uv sync --extra libero
uv sync --extra robotwin
uv sync --extra calvin
uv sync --extra sim
uv sync --extra deployment
uv sync --extra docs
uv sync --extra full
```

For a local LIBERO simulator rollout the `[libero]` extra alone is **not**
enough — it only pins the LIBERO-specific runtime deps (`gym`, `robosuite`,
`bddl`, etc.) and not the model stack. Use `[sim]` (or `[full]`), and add
the upstream LIBERO source plus a one-line config so LIBERO can locate its
bddl / init / asset folders:

```bash
# 1. Install model + simulator deps in one shot
uv sync --extra sim

# 2. Clone upstream LIBERO; it is not on PyPI
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ../LIBERO
# Empty __init__.py so editable installs see `libero` as a real package
# instead of an empty PEP-660 namespace finder.
touch ../LIBERO/libero/__init__.py
uv pip install -e ../LIBERO

# 3. Tell LIBERO where its asset/bddl/init directories live
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<'EOF'
benchmark_root: /absolute/path/to/LIBERO/libero/libero
bddl_files:    /absolute/path/to/LIBERO/libero/libero/bddl_files
init_states:   /absolute/path/to/LIBERO/libero/libero/init_files
datasets:      /absolute/path/to/LIBERO/libero/datasets
assets:        /absolute/path/to/LIBERO/libero/libero/assets
EOF

# 4. Point the local checkpoint registry at the trained Method 1 ckpt
cp configs/local_paths.sample.yaml configs/local_paths.yaml
# Replace the parallel_stream_exact_libero_step_400 placeholder with the
# absolute path to your local checkpoint_step_400 directory.
```

The keys in `~/.libero/config.yaml` must be exactly `benchmark_root`,
`bddl_files`, `init_states`, `datasets`, `assets` — without the `_folder`
suffix LIBERO's loader rejects them.

Use the release pytest subset after installing Torch/runtime extras when you
want fast CPU checks of the imported strict-old paths:

```bash
uv run --group dev --extra train python -m pytest \
  tests/test_config_loader.py \
  tests/test_static_config_schema.py \
  tests/test_mot_runtime_routing.py \
  tests/test_mot_modules.py::test_mot_action_then_video_action_only_rollout_skips_predicted_video \
  tests/test_mot_modules.py::test_mot_decoupled_action_only_rollout_skips_split_cache_video_denoise \
  -q
```

Train the current Method-5 strict-old LIBERO path:

```bash
uv run python -m open_wam.training.train \
  --cfg configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml
```

Historical LIBERO visualization wrappers are not part of this public snapshot.
Use the package evaluator with an included public experiment config for local
CPU/GPU smoke checks.

Run eval from an included experiment YAML:

```bash
uv run python -m open_wam.evals.evaluate \
  --cfg configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml
```

Trajectory eval modes:

- `trajectory`: teacher-forced visual rollout over episode windows
- `trajectory_open_loop`: reuses predicted video latents across later windows
  by aligning overlapping frame indices and seeding newly entered frames from
  the current clean observation window

The current generic evaluator now:

- loads an experiment YAML
- builds the current `VariantPipeline`
- supports three modes:
  - `batch`: independent one-window inference on each sampled batch
  - `trajectory`: stateful rollout over episode-ordered windows, carrying
    `PolicyInferState` and previous predictions across the trajectory
  - `trajectory_open_loop`: same stateful rollout, but the next step may
    consume predicted video latents instead of rereading GT RGB
- runs the standard inference path in both modes, so each evaluation step still
  includes the variant's full denoising loop
- reports action-prediction shape and mean masked action MSE
- reports video latent MSE whenever the active variant exposes predicted
  latents
- reports mean per-trajectory action MSE and video latent MSE for trajectory
  modes when available
- optionally loads a checkpoint passed with `--checkpoint`

Trajectory mode requires an episode-aware dataset adapter, i.e. one that can
group windows by `episode_index` and `observation_start`. The current LIBERO
adapters support this; generic synthetic adapters do not.

Run trajectory eval on LIBERO:

```bash
uv run python -m open_wam.evals.evaluate \
  --cfg configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml
```

## Backbone Sharing Clarification

All canonical multimodal methods now run through the same top-level owner:

- `VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`

With `backbone.implementation = shared_transformer`, methods 1, 2, and 4 run
through the same shared `VisualTower` frontend and shared transformer-core
object.

What differs between the methods is the runtime program:

- method 1 uses exact LingBot-compatible runtime programs, chunk/window
  attention, and slot-pool cache semantics
- method 2 uses structured register-sequence runtime programs, structured
  branchwise attention, and structured rollout-cache semantics
- method 4 uses the same shared core with a lightweight decoded-feature policy
  head

`post_latent` is the intentional exception: when configured with
`attach_site=post_frontend_latents`, it may stop at the shared frontend and
bypass the transformer core by design.

LingBot-compatible weights can initialize the shared backbone by setting:

- `backbone.implementation: shared_transformer`
- `backbone.load_reference_core_weights: true`
- `backbone.pretrained_model_name_or_path: /path/to/checkpoint-root`

Exact method-1 execution uses that same shared backbone object, but drives it
through the LingBot-compatible exact runtime programs exposed by the shared
runtime executor rather than a sidecar transformer module.

Historical internal execution notes are not part of this public snapshot. Use
the included experiment YAMLs, CLI docs, and local path sample as the supported
public starting points.

## Current Dataset Contract

All dataset adapters should return the same artifact shape after collation:

- `views`: `dict[str, Tensor]`, each view `[B, T, H, W, 3]`
- `actions`: `[B, H_action, D_action]`
- `action_mask`: optional mask aligned to `actions`
- `state`: optional `[B, H_state, D_state]`
- `state_mask`: optional mask aligned to `state`
- `task_text`: optional tuple of task strings
- `metadata`: tuple of per-sample metadata dicts

The shared visual path canonicalizes `views` into one RGB canvas and emits
stageful `VisualStageOutputs`.

For LIBERO specifically, `actions` default to a transformed 7D
reference-relative EEF target `[rel_xyz, rel_axis_angle, gripper_1d_command]`.
The pose part comes from dataset state, while the last scalar is copied from
the raw LIBERO action command rather than from finger-joint state. That public
target is now supported by:

- exact original-vs-reconstructed trajectory comparison over either a sampled
  horizon or a full episode
- closed-loop conversion back into LIBERO `OSC_POSE` actions for simulator
  replay / evaluation

## Public Documentation

The durable public docs live under `docs/`. Raw internal notes, deployment
workspaces, rig logs, and private checkpoint instructions are intentionally not
included in this first public snapshot.

## Current Caveat

`physical-intelligence/libero` is structurally a LeRobot-format dataset, but the
installed `lerobot` package in this environment does not safely load the repo
revision currently on Hugging Face. The current adapter therefore reads the
repo's metadata and episode parquet files directly.
