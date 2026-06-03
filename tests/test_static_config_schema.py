from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.configs.static_schema import validate_config_file, validate_config_files


REPO_ROOT = Path(__file__).resolve().parents[1]

STRICT_OLD_CONFIG_PATHS = tuple(
    REPO_ROOT / "configs" / "experiments" / name
    for name in (
        "mot_libero_latent_local_video_then_action_heng_compatible.yaml",
        "mot_libero_latent_local_joint_heng_compatible.yaml",
        "mot_libero_latent_local_action_then_video_heng_compatible.yaml",
        "mot_libero_latent_local_decoupled_same_step_heng_compatible.yaml",
        "mot_libero_latent_local_video_noisy_to_action_heng_compatible.yaml",
        "mot_libero_latent_local_action_noisy_to_video_heng_compatible.yaml",
        "mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_action_then_video_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_video_noisy_to_action_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_action_noisy_to_video_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml",
        "causal_video_prediction_libero_latent_local.yaml",
    )
)


@pytest.mark.unit
def test_static_validator_accepts_strict_old_configs() -> None:
    reports = validate_config_files(STRICT_OLD_CONFIG_PATHS)

    assert all(report.ok for report in reports)


@pytest.mark.unit
def test_static_validator_catches_enum_typos(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
name: bad
data:
  dataset_name: bad
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 4
    action_horizon: 2
    state_dim: 3
    state_horizon: 1
backbone:
  implementation: not_a_backbone
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
action_decoder:
  name: parallel_stream_action_decoder
  action_dim: 4
  action_horizon: 2
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid BackboneImplementation" in issue.message for issue in report.errors)
