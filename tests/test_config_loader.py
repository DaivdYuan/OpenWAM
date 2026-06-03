from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.configs import CurrentBlockCoupling, JointTimestepCoupling, ParallelSequenceContract
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]

M5_STRICT_OLD_CONFIGS = {
    "mot_libero_latent_local_video_then_action_heng_compatible.yaml": CurrentBlockCoupling.VIDEO_THEN_ACTION,
    "mot_libero_latent_local_joint_heng_compatible.yaml": CurrentBlockCoupling.JOINT,
    "mot_libero_latent_local_action_then_video_heng_compatible.yaml": CurrentBlockCoupling.ACTION_THEN_VIDEO,
    "mot_libero_latent_local_decoupled_same_step_heng_compatible.yaml": CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    "mot_libero_latent_local_video_noisy_to_action_heng_compatible.yaml": CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
    "mot_libero_latent_local_action_noisy_to_video_heng_compatible.yaml": CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    "mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml": CurrentBlockCoupling.JOINT,
}

M1_STRICT_OLD_CONFIGS = {
    "parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml": CurrentBlockCoupling.VIDEO_THEN_ACTION,
    "parallel_stream_libero_lingbot_m1_joint_heng_compatible.yaml": CurrentBlockCoupling.JOINT,
    "parallel_stream_libero_lingbot_m1_action_then_video_heng_compatible.yaml": CurrentBlockCoupling.ACTION_THEN_VIDEO,
    "parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml": CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    "parallel_stream_libero_lingbot_m1_video_noisy_to_action_heng_compatible.yaml": CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
    "parallel_stream_libero_lingbot_m1_action_noisy_to_video_heng_compatible.yaml": CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml": CurrentBlockCoupling.JOINT,
}


@pytest.mark.parametrize("config_name,expected_coupling", sorted(M5_STRICT_OLD_CONFIGS.items()))
def test_m5_strict_old_configs_load(config_name: str, expected_coupling: CurrentBlockCoupling) -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / config_name)

    assert config.policy_variant.name == "mot"
    assert config.policy_variant.parallel_sequence_contract == ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    assert config.policy_variant.current_block_coupling == expected_coupling
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.policy_variant.noisy_video_condition_prob == pytest.approx(0.5)
    assert config.training.enabled_objectives == ("action", "latent")


@pytest.mark.parametrize("config_name,expected_coupling", sorted(M1_STRICT_OLD_CONFIGS.items()))
def test_m1_strict_old_configs_load(config_name: str, expected_coupling: CurrentBlockCoupling) -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / config_name)

    assert config.policy_variant.name == "parallel_stream"
    assert config.policy_variant.parallel_sequence_contract == ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    assert config.policy_variant.current_block_coupling == expected_coupling
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.policy_variant.noisy_video_condition_prob == pytest.approx(0.5)
    assert config.training.enabled_objectives == ("latent", "action")


def test_video_only_pretrain_config_loads_for_strict_old_initialization() -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / "causal_video_prediction_libero_latent_local.yaml")

    assert config.policy_variant.name == "causal_video_prediction"
    assert "latent" in config.training.enabled_objectives
