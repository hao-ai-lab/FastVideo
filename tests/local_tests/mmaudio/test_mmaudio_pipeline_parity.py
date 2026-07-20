# SPDX-License-Identifier: Apache-2.0
"""MMAudio pipeline contracts and opt-in real end-to-end parity."""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO = REPO_ROOT.parent / "MMAudio"
CONVERTED_MODEL = REPO_ROOT / "converted_weights/mmaudio/large_44k_v2"
DFN5B_DIR = REPO_ROOT / "official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384"
BIGVGAN_DIR = REPO_ROOT / "official_weights/mmaudio/bigvgan_v2_44khz_128band_512x"


def test_mmaudio_pipeline_config_registry_and_preset() -> None:
    from fastvideo.configs.pipelines.mmaudio import MMAudioV2AConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.registry import get_model_info, get_preset_selection

    assert WorkloadType.from_string("v2a") is WorkloadType.V2A
    assert WorkloadType.from_string("t2a") is WorkloadType.T2A
    config = MMAudioV2AConfig()
    assert config.dit_config.prefix == "MMAudio"
    assert len(config.image_encoder_configs or ()) == 2
    assert config.text_encoder_precisions == ("bf16",)
    assert config.image_encoder_precisions == ("bf16", "bf16")
    assert config.audio_decoder_precision == "bf16"
    assert config.vocoder_precision == "bf16"
    assert config.duration_s == 8.0
    assert config.max_audio_duration_s is None
    assert get_preset_selection("FastVideo/MMAudio-large-44k-v2-Diffusers") == (
        "mmaudio_large_44k_v2",
        "mmaudio",
    )

    if CONVERTED_MODEL.is_dir():
        info = get_model_info(str(CONVERTED_MODEL), workload_type=WorkloadType.V2A)
        assert info.pipeline_cls.__name__ == "MMAudioPipeline"
        assert info.pipeline_config_cls is MMAudioV2AConfig


def test_mmaudio_published_sequence_lengths() -> None:
    from fastvideo.configs.pipelines.mmaudio import MMAudioV2AConfig
    from fastvideo.pipelines.basic.mmaudio.stages import mmaudio_sequence_lengths

    assert mmaudio_sequence_lengths(8.0, MMAudioV2AConfig()) == (345, 64, 192)
    assert mmaudio_sequence_lengths(10.0, MMAudioV2AConfig()) == (431, 80, 240)
    assert mmaudio_sequence_lengths(2.0, MMAudioV2AConfig()) == (87, 16, 40)


def test_mmaudio_bf16_euler_stage_matches_official() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MMAudio BF16 scheduler parity requires CUDA")

    from mmaudio.model.flow_matching import FlowMatching

    from fastvideo.configs.pipelines.mmaudio import MMAudioV2AConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from fastvideo.pipelines.basic.mmaudio.stages import MMAudioDenoisingStage
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    class ToyTransformer:
        @staticmethod
        def guided_flow(timestep, latent, conditions, empty_conditions, guidance_scale):
            del conditions, empty_conditions, guidance_scale
            return torch.tanh(latent * 0.125 + timestep)

        @staticmethod
        def unnormalize(latent):
            return latent

    device = torch.device("cuda:0")
    initial = torch.randn(
        (1, 87, 40),
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    official = FlowMatching(min_sigma=0.0, inference_mode="euler", num_steps=25)
    expected = official.to_data(lambda time, latent: torch.tanh(latent * 0.125 + time), initial.clone())

    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=1.0,
        invert_sigmas=True,
        sigma_min=0.0,
        use_reference_discrete_timesteps=True,
    )
    batch = ForwardBatch(data_type="audio", latents=initial.clone(), num_inference_steps=25, guidance_scale=4.5)
    batch.extra["mmaudio_conditions"] = object()
    batch.extra["mmaudio_empty_conditions"] = object()
    actual = MMAudioDenoisingStage(ToyTransformer(), scheduler)(
        batch,
        FastVideoArgs(model_path="test", pipeline_config=MMAudioV2AConfig()),
    ).latents
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    os.environ.get("MMAUDIO_RUN_PIPELINE_PARITY") != "1",
    reason="Set MMAUDIO_RUN_PIPELINE_PARITY=1 and MMAUDIO_PARITY_VIDEO to run the 25-step real-weight gate.",
)
def test_mmaudio_real_v2a_pipeline_waveform_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare official and FastVideo waveforms with the same 2s video/seed."""
    video_value = os.environ.get("MMAUDIO_PARITY_VIDEO")
    if not video_value:
        pytest.skip("Set MMAUDIO_PARITY_VIDEO to a video containing at least two seconds.")
    video_path = Path(video_value)
    required = (
        OFFICIAL_REPO / "weights/mmaudio_large_44k_v2.pth",
        OFFICIAL_REPO / "ext_weights/v1-44.pth",
        OFFICIAL_REPO / "ext_weights/synchformer_state_dict.pth",
        CONVERTED_MODEL / "model_index.json",
        DFN5B_DIR / "open_clip_pytorch_model.bin",
        BIGVGAN_DIR / "bigvgan_generator.pt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"MMAudio real pipeline assets are missing: {missing}")
    if not torch.cuda.is_available():
        pytest.skip("MMAudio real pipeline parity requires CUDA")

    import open_clip
    import mmaudio.ext.autoencoder.autoencoder as official_autoencoder
    import mmaudio.model.utils.features_utils as official_features_module
    from mmaudio.eval_utils import generate as official_generate
    from mmaudio.eval_utils import load_video as official_load_video
    from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as OfficialBigVGAN
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    monkeypatch.setattr(
        official_features_module,
        "create_model_from_pretrained",
        lambda *args, **kwargs: open_clip.create_model_from_pretrained(
            f"local-dir:{DFN5B_DIR}", return_transform=False
        ),
    )

    class LocalBigVGAN:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            del model_id
            return OfficialBigVGAN.from_pretrained(str(BIGVGAN_DIR), **kwargs)

    monkeypatch.setattr(official_autoencoder, "BigVGANv2", LocalBigVGAN)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    with torch.inference_mode():
        official_model = get_my_mmaudio("large_44k_v2").to(device, dtype).eval()
        official_model.load_weights(
            torch.load(required[0], map_location=device, weights_only=True)
        )
        official_features = FeaturesUtils(
            tod_vae_ckpt=required[1],
            synchformer_ckpt=required[2],
            enable_conditions=True,
            mode="44k",
            need_vae_encoder=False,
        ).to(device, dtype).eval()
        video = official_load_video(video_path, 2.0, load_all_frames=False)
        duration = video.duration_sec
        official_model.update_seq_lengths(
            math.ceil(duration * 44100 / 512 / 2),
            int(duration * 8),
            int((((duration * 25) - 16) // 8 + 1) * 16 / 2),
        )
        expected = official_generate(
            video.clip_frames[None],
            video.sync_frames[None],
            ["A dog runs past a red car!"],
            negative_text=[""],
            feature_utils=official_features,
            net=official_model,
            fm=FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25),
            rng=torch.Generator(device=device).manual_seed(42),
            cfg_strength=4.5,
        ).float().cpu()
    del official_features, official_model
    gc.collect()
    torch.cuda.empty_cache()

    from fastvideo.configs.pipelines.mmaudio import MMAudioV2AConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
    from fastvideo.pipelines.basic.mmaudio import MMAudioPipeline
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    args = FastVideoArgs(
        model_path=str(CONVERTED_MODEL),
        workload_type=WorkloadType.V2A,
        pipeline_config=MMAudioV2AConfig(),
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        num_gpus=1,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
    )
    try:
        pipeline = MMAudioPipeline(str(CONVERTED_MODEL), args)
        pipeline.post_init()
        output = pipeline.forward(
            ForwardBatch(
                data_type="video",
                video_path=str(video_path),
                prompt="A dog runs past a red car!",
                negative_prompt="",
                audio_start_in_s=0.0,
                audio_end_in_s=2.0,
                num_inference_steps=25,
                guidance_scale=4.5,
                seed=42,
                num_videos_per_prompt=1,
                height=8,
                width=8,
                num_frames=1,
                save_video=False,
                return_frames=False,
            ),
            args,
        )
        actual = output.extra["decoded_audio"]
        assert output.extra["audio_sample_rate"] == 44100
        assert output.extra["audio_only"] is True
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        pipeline.close()
    finally:
        cleanup_dist_env_and_memory()
