# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")
# Force TORCH_SDPA backend for parity testing - both FastVideo and LTX-2 reference
# will use PyTorch's scaled_dot_product_attention for consistent results
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.configs.models.dits import LTX2VideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.ltx2 import Modality as FastVideoModality
from fastvideo.models.loader.component_loader import TransformerLoader
from .test_ltx2 import (
    _attach_block_detail_logging,
    _attach_block_sum_logging,
    _infer_patch_params,
    _read_transformer_config,
)


def test_ltx2_transformer_audio_parity():
    torch.manual_seed(42)
    diffusers_root = Path(
        os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    )
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    fastvideo_path = Path(
        os.getenv(
            "LTX2_FASTVIDEO_PATH",
            str(diffusers_root / "transformer"),
        )
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 official weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"FastVideo converted weights not found at {fastvideo_path}")

    try:
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import (LTXModelConfigurator,
                                                LTXV_MODEL_COMFY_RENAMING_MAP)
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.types import AudioLatentShape, VideoLatentShape
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    # Load config from metadata using same approach as test_ltx2.py
    config_loader = SafetensorsModelStateDictLoader()
    metadata = config_loader.metadata(str(official_path))
    transformer_config = _read_transformer_config(metadata)

    config = LTX2VideoConfig()
    cfg = config.arch_config
    cfg.num_attention_heads = transformer_config.get("num_attention_heads",
                                                     cfg.num_attention_heads)
    cfg.attention_head_dim = transformer_config.get("attention_head_dim",
                                                    cfg.attention_head_dim)
    cfg.num_layers = transformer_config.get("num_layers", cfg.num_layers)
    cfg.cross_attention_dim = transformer_config.get(
        "cross_attention_dim", cfg.cross_attention_dim)
    cfg.caption_channels = transformer_config.get("caption_channels",
                                                  cfg.caption_channels)
    cfg.norm_eps = transformer_config.get("norm_eps", cfg.norm_eps)
    cfg.attention_type = transformer_config.get("attention_type",
                                                cfg.attention_type)
    cfg.positional_embedding_theta = transformer_config.get(
        "positional_embedding_theta", cfg.positional_embedding_theta)
    cfg.positional_embedding_max_pos = transformer_config.get(
        "positional_embedding_max_pos", cfg.positional_embedding_max_pos)
    cfg.timestep_scale_multiplier = transformer_config.get(
        "timestep_scale_multiplier", cfg.timestep_scale_multiplier)
    cfg.use_middle_indices_grid = transformer_config.get(
        "use_middle_indices_grid", cfg.use_middle_indices_grid)
    cfg.rope_type = transformer_config.get("rope_type", cfg.rope_type)
    cfg.double_precision_rope = transformer_config.get(
        "double_precision_rope",
        transformer_config.get("frequencies_precision", "")
        == "float64",
    )
    cfg.audio_num_attention_heads = transformer_config.get(
        "audio_num_attention_heads", cfg.audio_num_attention_heads)
    cfg.audio_attention_head_dim = transformer_config.get(
        "audio_attention_head_dim", cfg.audio_attention_head_dim)
    cfg.audio_in_channels = transformer_config.get("audio_in_channels",
                                                   cfg.audio_in_channels)
    cfg.audio_out_channels = transformer_config.get("audio_out_channels",
                                                    cfg.audio_out_channels)
    cfg.audio_cross_attention_dim = transformer_config.get(
        "audio_cross_attention_dim", cfg.audio_cross_attention_dim)
    cfg.audio_positional_embedding_max_pos = transformer_config.get(
        "audio_positional_embedding_max_pos",
        cfg.audio_positional_embedding_max_pos,
    )
    cfg.av_ca_timestep_scale_multiplier = transformer_config.get(
        "av_ca_timestep_scale_multiplier", cfg.av_ca_timestep_scale_multiplier)
    cfg.in_channels = transformer_config.get("in_channels", cfg.in_channels)
    cfg.out_channels = transformer_config.get("out_channels", cfg.out_channels)

    patch_size, num_channels_latents = _infer_patch_params(cfg.in_channels)
    cfg.patch_size = (1, patch_size, patch_size)
    cfg.num_channels_latents = num_channels_latents

    if not torch.cuda.is_available():
        pytest.skip("LTX-2 transformer parity test requires CUDA for attention backends.")

    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=config, dit_precision=precision_str),
    )
    args.device = device

    loader = TransformerLoader()
    fastvideo_model = loader.load(str(fastvideo_path), args).to(device=device, dtype=precision)

    # Use SingleGPUModelBuilder to load the reference model (same as test_ltx2.py)
    reference_builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=str(official_path),
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    reference_model = reference_builder.build(
        device=device, dtype=precision).to(device=device, dtype=precision)
    reference_model.set_gradient_checkpointing(False)

    fastvideo_model.eval()
    reference_model.eval()

    debug_logs = os.getenv("LTX2_DEBUG_LOGS", "0") == "1"
    _attach_block_sum_logging(
        fastvideo_model.model,
        repo_root / "ltx2_debug" / "fastvideo_audio.log",
        "fastvideo",
        debug_logs,
    )
    _attach_block_sum_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference_audio.log",
        "reference",
        debug_logs,
    )
    _attach_block_detail_logging(
        fastvideo_model.model,
        repo_root / "ltx2_debug" / "fastvideo_audio_detail.log",
        "fastvideo",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )
    _attach_block_detail_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference_audio_detail.log",
        "reference",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )

    patchifier = VideoLatentPatchifier(patch_size=cfg.patch_size[1])
    audio_patchifier = AudioPatchifier(patch_size=1)
    batch_size = 1
    frames = 4
    height = cfg.patch_size[1] * 4
    width = cfg.patch_size[2] * 4
    hidden_states = torch.randn(
        batch_size,
        cfg.num_channels_latents,
        frames,
        height,
        width,
        device=device,
        dtype=precision,
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        16,
        cfg.caption_channels,
        device=device,
        dtype=precision,
    )
    timestep = torch.tensor([500], device=device, dtype=precision)

    video_shape = VideoLatentShape.from_torch_shape(hidden_states.shape)
    positions = patchifier.get_patch_grid_bounds(video_shape, device=hidden_states.device)
    latents = patchifier.patchify(hidden_states)

    audio_frames = 16
    audio_channels = 8
    audio_mel_bins = 16
    audio_latents = torch.randn(
        batch_size,
        audio_channels,
        audio_frames,
        audio_mel_bins,
        device=device,
        dtype=precision,
    )
    audio_shape = AudioLatentShape.from_torch_shape(audio_latents.shape)
    audio_positions = audio_patchifier.get_patch_grid_bounds(audio_shape, device=audio_latents.device)
    audio_tokens = audio_patchifier.patchify(audio_latents)

    video = Modality(
        enabled=True,
        latent=latents,
        timesteps=timestep,
        positions=positions,
        context=encoder_hidden_states,
        context_mask=None,
    )
    audio = Modality(
        enabled=True,
        latent=audio_tokens,
        timesteps=timestep,
        positions=audio_positions,
        context=encoder_hidden_states,
        context_mask=None,
    )

    fastvideo_video = FastVideoModality(
        enabled=True,
        latent=latents,
        timesteps=timestep,
        positions=positions,
        context=encoder_hidden_states,
        context_mask=None,
    )
    fastvideo_audio = FastVideoModality(
        enabled=True,
        latent=audio_tokens,
        timesteps=timestep,
        positions=audio_positions,
        context=encoder_hidden_states,
        context_mask=None,
    )

    with torch.no_grad():
        _, ref_audio_out = reference_model(
            video=video,
            audio=audio,
            perturbations=BatchedPerturbationConfig.empty(batch_size),
        )
        ref_audio_out = audio_patchifier.unpatchify(ref_audio_out, output_shape=audio_shape)
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            _, fastvideo_audio_out = fastvideo_model.model(
                video=fastvideo_video,
                audio=fastvideo_audio,
            )
            fastvideo_audio_out = audio_patchifier.unpatchify(fastvideo_audio_out, output_shape=audio_shape)

    assert ref_audio_out.shape == fastvideo_audio_out.shape
    assert ref_audio_out.dtype == fastvideo_audio_out.dtype
    # With TORCH_SDPA backend for both, use same tolerance as video parity test
    assert_close(ref_audio_out, fastvideo_audio_out, atol=1e-4, rtol=1e-4)
