# SPDX-License-Identifier: Apache-2.0
"""CPU checks for MiniMax-H3-Image packing, the 2D VAE, PDD heads, and NVFP4 names."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29531")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import pytest
import torch

from fastvideo.configs.models.dits.minimax_h3 import (
    MiniMaxH3ArchConfig,
    MiniMaxH3Config,
    MiniMaxH3ImageArchConfig,
    MiniMaxH3ImageConfig,
)
from fastvideo.configs.models.vaes.minimax_h3_image import MiniMaxH3ImageVAEArchConfig, MiniMaxH3ImageVAEConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.quantization.nvfp4_config import is_minimax_h3_nvfp4_linear_prefix
from fastvideo.models.dits.minimax_h3 import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Transformer3DModel
from fastvideo.models.vaes.minimax_h3_image import AutoencoderKLMiniMaxH3Image
from fastvideo.pipelines.basic.minimax_h3_image.minimax_h3_image_pipeline import MiniMaxH3ImageModularPipeline
from fastvideo.pipelines.basic.minimax_h3_image.packing import (
    MINIMAX_H3_IMAGE_SAMPLE_SIGMAS,
    build_image_packed_sequence,
    edit_label,
    get_image_canvas_size,
    rounded_sample_sigmas,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.registry import _is_minimax_h3_image, _is_minimax_h3_video, get_model_family


def _tiny_dit_arch(**overrides: object) -> MiniMaxH3ArchConfig:
    fields = dict(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    fields.update(overrides)
    return MiniMaxH3ArchConfig(**fields)


def test_canvas_scales_small_images_and_keeps_large_ones() -> None:
    assert get_image_canvas_size(1024, 1024, 2048 * 2048, 32) == (2048, 2048)
    assert get_image_canvas_size(4096, 4096, 2048 * 2048, 32) == (4096, 4096)
    height, width = get_image_canvas_size(100, 100, 2048 * 2048, 32)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width >= 2048 * 2048


def test_edit_label_has_no_angle_brackets() -> None:
    assert edit_label(1) == "Picture 1: "
    assert "<" not in edit_label(2)


def test_packed_layout_puts_references_before_the_target() -> None:
    tags = torch.tensor([1, 1, 0, 1], dtype=torch.long)
    position_ids, token_tags, image_indices, text_indices, num_reference_rows = build_image_packed_sequence(
        tags,
        reference_grids=[(4, 4)],
        latent_height=4,
        latent_width=4,
        patch_size=(1, 2, 2),
    )
    assert num_reference_rows == 4
    assert text_indices.tolist() == [0, 1, 2, 3]
    assert image_indices[0].item() == 4
    assert token_tags[image_indices].eq(0).all()
    assert position_ids.dtype == torch.float64
    assert position_ids.shape == (4 + 4 + 4, 3)
    # Text occupies axis 0. Each image is one later rotary time slot.
    assert position_ids[:4, 1:].eq(0).all()
    assert position_ids[4:8, 0].eq(position_ids[4, 0]).all()
    assert position_ids[8:, 0].eq(position_ids[4, 0] + 1).all()


def test_sigma_round_trip_keeps_six_steps() -> None:
    sigmas = rounded_sample_sigmas()
    assert sigmas.shape == (len(MINIMAX_H3_IMAGE_SAMPLE_SIGMAS), )
    assert sigmas[-1].item() == 0.0
    assert torch.all(sigmas[1:] < sigmas[:-1])


def test_video_detector_does_not_claim_the_image_repo() -> None:
    assert get_model_family("MiniMaxAI/MiniMax-H3-Image") == "minimax_h3_image"
    assert get_model_family("MiniMaxAI/MiniMax-H3") == "minimax_h3"
    assert get_model_family("FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2") == "minimax_h3"
    assert _is_minimax_h3_image("MiniMaxAI/MiniMax-H3-Image-PE") is False
    assert _is_minimax_h3_video("MiniMaxAI/MiniMax-H3") is True
    assert _is_minimax_h3_video("exports/minimax-h3-image") is False
    assert _is_minimax_h3_image("minimaxh3imagepipeline") is True
    assert _is_minimax_h3_video("minimaxh3modularpipeline") is True


def test_local_image_manifest_is_not_classified_as_video(tmp_path) -> None:
    model_dir = tmp_path / "custom-export"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": "MiniMaxH3ImagePipeline",
            "_diffusers_version": "0.39.0",
        }),
        encoding="utf-8",
    )
    assert get_model_family(str(model_dir)) == "minimax_h3_image"


def test_image_dit_pdd_heads_replace_audio_and_proj_out(distributed_setup) -> None:
    del distributed_setup
    image = MiniMaxH3ImageTransformer2DModel(
        MiniMaxH3ImageConfig(arch_config=_tiny_image_arch(), prefix="minimax_h3"),
        {},
    )
    video = MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=_tiny_dit_arch(), prefix="minimax_h3"), {})
    assert image.audio_proj_in is None and image.audio_proj_out is None
    assert image.proj_out is None and image.pdd_proj_out is not None
    assert len(image.pdd_proj_out) == 2
    assert video.audio_proj_in is not None and video.proj_out is not None
    assert video.pdd_proj_out is None
    assert AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3 not in image.supported_attention_backends
    assert AttentionBackendEnum.ATTN_QAT_INFER in image.supported_attention_backends

    main_ffn = [
        name for name, _module in image.named_modules()
        if name.startswith("transformer_blocks.") and (name.endswith("ff.fc_in") or name.endswith("ff.fc_out"))
    ]
    assert main_ffn
    for name in main_ffn:
        assert is_minimax_h3_nvfp4_linear_prefix(f"minimax_h3.{name}")
    assert not is_minimax_h3_nvfp4_linear_prefix("minimax_h3.token_refiner.refiner_blocks.0.ff.fc_in")
    assert not is_minimax_h3_nvfp4_linear_prefix("minimax_h3.pdd_proj_out.0")
    assert not is_minimax_h3_nvfp4_linear_prefix("minimax_h3.proj_in")

    text = torch.randn(1, 2, 16)
    hidden = torch.randn(1, 4, 16)
    position_ids = torch.zeros(6, 3)
    position_ids[:2, 0] = torch.arange(2)
    tags = torch.tensor([1, 1, 0, 0, 0, 0])
    from fastvideo.forward_context import set_forward_context

    timestep_indices = torch.zeros(6, dtype=torch.long)
    forward_kwargs = dict(
        hidden_states=hidden,
        audio_hidden_states=hidden.new_empty(1, 0, 1),
        encoder_hidden_states=text,
        timestep=torch.tensor([0.5]),
        timestep_indices=timestep_indices,
        token_tags=tags,
        position_ids=position_ids,
        video_indices=torch.arange(2, 6),
        audio_indices=torch.empty(0, dtype=torch.long),
        text_indices=torch.arange(2),
    )
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        first, _audio = image(**forward_kwargs, pdd_step_index=0)
        second, _audio = image(**forward_kwargs, pdd_step_index=1)
    assert first.shape == (1, 4, 16)
    assert not torch.equal(first, second)


def _tiny_image_arch() -> MiniMaxH3ImageArchConfig:
    base = _tiny_dit_arch()
    values = {name: getattr(base, name) for name in base.__dataclass_fields__ if name != "_supported_attention_backends"}
    values["num_pdd_heads"] = 2
    values["audio_enabled"] = False
    return MiniMaxH3ImageArchConfig(**values)


def test_image_ffn_still_selects_nvfp4_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("NVFP4QuantizeMethod construction requires CUDA")
    from fastvideo.layers.linear import UnquantizedLinearMethod
    from fastvideo.layers.quantization.nvfp4_config import NVFP4Config, NVFP4QuantizeMethod

    model = MiniMaxH3ImageTransformer2DModel(
        MiniMaxH3ImageConfig(
            arch_config=_tiny_image_arch(),
            prefix="minimax_h3",
            quant_config=NVFP4Config(),
        ),
        {},
    )
    feed_forward = model.transformer_blocks[0].ff
    assert isinstance(feed_forward.fc_in.quant_method, NVFP4QuantizeMethod)
    assert isinstance(feed_forward.fc_out.quant_method, NVFP4QuantizeMethod)
    assert isinstance(model.pdd_proj_out[0].quant_method, UnquantizedLinearMethod)
    assert isinstance(model.proj_in.quant_method, UnquantizedLinearMethod)


def test_image_vae_encoder_decoder_shapes() -> None:
    arch = MiniMaxH3ImageVAEArchConfig(
        latent_channels=4,
        block_out_channels=(32, ),
        spatial_downsample_factors=(2, ),
        temporal_downsample_factors=(1, ),
        layers_per_block=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_rope_dim_ratio=0.5,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    vae = AutoencoderKLMiniMaxH3Image(MiniMaxH3ImageVAEConfig(arch_config=arch, use_tiling=True))
    pixels = torch.rand(1, 3, 32, 32)
    latents = vae.encode(vae.normalize_pixels(pixels)).latent_dist.mode()
    assert latents.shape == (1, 4, 16, 16)
    decoded = vae.decode(latents).sample
    assert decoded.shape == (1, 3, 32, 32)
    assert vae.post_quant_conv.__class__.__name__ == "Conv2d"
    assert isinstance(vae.encoder, torch.nn.Module)


def test_sequential_load_defaults_on_for_the_image_pipeline() -> None:
    pipeline = MiniMaxH3ImageModularPipeline.__new__(MiniMaxH3ImageModularPipeline)
    enabled = FastVideoArgs(
        model_path="MiniMaxAI/MiniMax-H3-Image",
        lazy_module_load=False,
        h3_sequential_load=None,
    )
    disabled = FastVideoArgs(
        model_path="MiniMaxAI/MiniMax-H3-Image",
        lazy_module_load=False,
        h3_sequential_load=False,
    )
    assert pipeline._defer_denoise_modules(enabled) is True
    assert pipeline._defer_denoise_modules(disabled) is False
