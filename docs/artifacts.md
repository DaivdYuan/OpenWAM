# Artifacts And Checkpoints

Open-WAM separates public experiment configs from machine-local artifact paths.

## Local Path Registry

Use `configs/local_paths.yaml` for local paths. Start from:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

That file is gitignored. It can also live outside the repo:

```bash
OPEN_WAM_LOCAL_PATHS=/path/to/local_paths.yaml uv run open-wam-eval ...
```

## Artifact Manifest

`configs/artifacts.sample.yaml` documents the public manifest schema for data
and checkpoints. Machine-local or private manifests should use
`configs/artifacts.yaml`, which is gitignored.

- `artifact_id`
- `method_family`
- `variant`
- `benchmark`
- `config`
- `local_path_alias`
- `expected_layout`
- `download_url`
- `checksum`
- `license`
- `source`
- `notes`

Entries with `download_url: null` are layout documentation only. They should not
be advertised as reproducible public checkpoints until hosting, checksum, and
license fields are filled. The sample manifest therefore uses a
`checkpoint-layout-reference` entry to document the expected fields without
claiming that a public checkpoint or checked-in fixture exists.

## Checkpoint Layout Convention

Full training checkpoint roots should use this layout when possible:

```text
checkpoint_step_N/
  full_training_state.pt
  model_state.pt
  transformer/
    config.json
    ...
```

Transformer-only runtime paths may point directly at `checkpoint_step_N/transformer`.
Code that accepts checkpoint roots should also accept roots containing a
`transformer/` child when possible.

## Checkpoint Utilities

Download a released checkpoint snapshot from a user-provided Hugging Face repo:

```bash
uv run --extra train python scripts/download_checkpoint.py \
  --repo-id <org-or-user>/<repo> \
  --mode inference
```

Extract a lightweight `model_state.pt` from a full training checkpoint:

```bash
uv run --extra train python scripts/extract_model_state_checkpoint.py \
  --input /path/to/checkpoint_step_N
```
