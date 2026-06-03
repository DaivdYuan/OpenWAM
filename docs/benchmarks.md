# Benchmarks And Data

Open-WAM keeps benchmark-specific loading behind adapters while exposing one
uniform model-facing batch contract.

## Supported Sources

| Source | Status | Primary use |
| --- | --- | --- |
| LIBERO | Dataset and simulator paths | Manipulation policy training, evaluation, and realtime rollout experiments. |
| RoboTwin | Dataset and simulator adapter path | Simulated robotic manipulation with configurable action schema. |
| CALVIN | Dataset and simulator adapter path | Simulated language-conditioned manipulation with 7D relative actions. |

Private datasets, local simulator checkouts, and large checkpoints should be
provided through the local path registry, not hard-coded in public configs.

## Action Dimensions

Benchmarks expose different native action spaces. The model-facing action
dimension is configured separately from the source action dimension.

| Benchmark | Common source action | Model-facing examples |
| --- | --- | --- |
| LIBERO | 7D EEF delta plus gripper | 7D or sparse 30D mapping depending on config. |
| RoboTwin | 16D or 30D modes | Native 16D, native 30D, or mapped sparse 30D. |
| CALVIN | 7D `rel_actions` | Native 7D or sparse 30D compatibility mapping. |

Action mapping should be explicit in the dataset adapter/config. A model should
not infer missing dimensions silently.

## Visual Layout

The data layer builds canonical RGB layouts before the visual backbone sees the
batch. Public configs should make these choices visible:

- camera names
- camera count
- frame window
- target image size
- layout policy
- channel order

This keeps visual packing controlled across methods and benchmarks.

## Public Snapshot Checks

The strict-old config set in this snapshot is useful for:

- static config validation
- release-gate pytest routing
- artifact manifest layout validation
- new contributor onboarding

Run the current validation path with:

```bash
open-wam-validate-config \
  configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml \
  configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml
```

After local LeRobot/LIBERO paths are configured, inspect the dataset adapter
without starting a training run:

```bash
uv run --extra train python scripts/inspect_libero_adapter.py \
  --cfg configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml
```

Use real benchmark cards and experiment cards for claims about policy quality.
