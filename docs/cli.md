# CLI Reference

Open-WAM exposes package-owned console commands as the stable public CLI
surface. Root scripts are limited to documented utilities and maintained launch
wrappers.

## Stable Commands

| Command | Purpose | Backing implementation |
| --- | --- | --- |
| `open-wam-train` | Train from an experiment YAML | `scripts/train.py` |
| `open-wam-eval` | Offline eval from experiment or eval YAML | `src/open_wam/evals/evaluate.py` |
| `open-wam-inspect-config` | Load and print typed config | `scripts/inspect_config.py` |
| `open-wam-validate-config` | Static YAML validation without model imports | `scripts/validate_configs_static.py` |

## Root Script Policy

New docs should prefer `open-wam-*` commands for train/eval/config workflows.
Any `scripts/...` entrypoint kept in the public snapshot must have:

1. a documented purpose
2. a CPU-safe help, dry-run, or syntax check
3. public configs or clear local-resource prerequisites
4. no dependency on omitted notes, configs, or deployment workspaces

## Command Examples

```bash
uv run open-wam-validate-config \
  configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml
```

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml
```

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml \
  --device cpu \
  --max-batches 1
```

This minimal snapshot does not include package-owned simulator rollout or
end-to-end sanity console commands. Add those commands only after their
implementations and public configs are included.
