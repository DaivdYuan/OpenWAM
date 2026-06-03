# Experiment Cards

Experiment cards are the public reproducibility layer for method families 1
through 5. Each card should point to an artifact manifest entry once a public
checkpoint exists.

## Required Fields

- method family
- variant name
- benchmark and task split
- train config
- eval config
- rollout command, if applicable
- checkpoint artifact id or local path alias
- dataset artifact id or local path alias
- hardware
- expected metrics
- known limitations

## Method Matrix

| Method | Current variant names | Public card status |
| --- | --- | --- |
| 1 | `parallel_stream`, `parallel_stream_lingbot_exact` | layout card added; public checkpoint pending |
| 2 | `register_attached` | scaffolded; public checkpoint pending |
| 3 | `video_sequence_policy` | layout card added; public checkpoint pending |
| 4 | `post_latent`, `post_decoded` with video-conditioned decoder | scaffolded; public checkpoint pending |
| 5 | `mot` | scaffolded; public checkpoint pending |
| fixture | `public_tiny_synthetic_contract` | public structural fixture card added |

## Current Cards

Detailed card pages are not included in this minimal snapshot. Add them under
`docs/` when public configs, artifacts, and reproduction commands are stable.

## Template

```yaml
method_family: method1
variant: parallel_stream
benchmark: libero_10
train_config: configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml
eval_config: null
checkpoint_artifact_id: null
dataset_artifact_id: null
hardware:
  gpu: null
  num_gpus: null
expected_metrics:
  mean_action_mse: null
  rollout_success_rate: null
commands:
  train: null
  eval: null
  rollout: null
limitations:
  - Public checkpoint hosting is not filled yet.
```
