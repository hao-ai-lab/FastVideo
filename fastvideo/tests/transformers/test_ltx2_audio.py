# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("FASTVIDEO_LIGHT_IMPORT", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

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
from fastvideo.tests.transformers.test_ltx2 import (
    _attach_block_detail_logging,
    _attach_block_sum_logging,
    _infer_arch,
    _load_into_model,
    _load_safetensors,
    _normalize_keys,
    _select_transformer_weights,
)


def test_ltx2_transformer_audio_parity():
    torch.manual_seed(42)
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    fastvideo_path = Path(
        os.getenv(
            "LTX2_FASTVIDEO_PATH",
            "converted/ltx2/transformer",
        )
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 official weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"FastVideo converted weights not found at {fastvideo_path}")

    try:
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.model.transformer import LTXModel
        from ltx_core.model.transformer.attention import AttentionFunction
        from ltx_core.model.transformer.model import LTXModelType
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.model.transformer.rope import LTXRopeType
        from ltx_core.types import AudioLatentShape, VideoLatentShape
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    ref_raw_weights = _load_safetensors(official_path)
    ref_weights = _normalize_keys(ref_raw_weights)
    ref_weights = _select_transformer_weights(ref_weights)
    if not ref_weights:
        pytest.skip("No transformer weights found in safetensors file.")
    arch = _infer_arch(ref_weights)
    if arch["num_layers"] == 0:
        pytest.skip("Could not infer transformer block count from LTX-2 weights.")

    config = LTX2VideoConfig()
    cfg = config.arch_config
    cfg.in_channels = arch["in_channels"]
    cfg.out_channels = arch["out_channels"]
    cfg.num_layers = arch["num_layers"]
    cfg.num_attention_heads = arch["num_attention_heads"]
    cfg.attention_head_dim = arch["attention_head_dim"]
    cfg.cross_attention_dim = arch["inner_dim"]
    cfg.caption_channels = arch["caption_channels"]
    cfg.num_channels_latents = arch["num_channels_latents"]
    cfg.patch_size = (1, arch["patch_size"], arch["patch_size"])

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

    reference_attention = AttentionFunction.PYTORCH
    reference_attn_env = os.getenv("LTX2_REFERENCE_ATTN")
    if reference_attn_env:
        reference_attention = AttentionFunction(reference_attn_env)
    else:
        fastvideo_attn_backend = os.getenv("FASTVIDEO_ATTENTION_BACKEND")
        if fastvideo_attn_backend == "FLASH_ATTN":
            reference_attention = AttentionFunction.FLASH_ATTENTION_3

    reference_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_layers=cfg.num_layers,
        cross_attention_dim=cfg.cross_attention_dim,
        norm_eps=cfg.norm_eps,
        caption_channels=cfg.caption_channels,
        positional_embedding_theta=cfg.positional_embedding_theta,
        positional_embedding_max_pos=cfg.positional_embedding_max_pos,
        timestep_scale_multiplier=cfg.timestep_scale_multiplier,
        use_middle_indices_grid=cfg.use_middle_indices_grid,
        attention_type=reference_attention,
        rope_type=LTXRopeType(cfg.rope_type),
        double_precision_rope=cfg.double_precision_rope,
        audio_num_attention_heads=cfg.audio_num_attention_heads,
        audio_attention_head_dim=cfg.audio_attention_head_dim,
        audio_in_channels=cfg.audio_in_channels,
        audio_out_channels=cfg.audio_out_channels,
        audio_cross_attention_dim=cfg.audio_cross_attention_dim,
        audio_positional_embedding_max_pos=cfg.audio_positional_embedding_max_pos,
        av_ca_timestep_scale_multiplier=cfg.av_ca_timestep_scale_multiplier,
    ).to(device=device, dtype=precision)

    loaded_reference = _load_into_model(reference_model, ref_weights)
    if loaded_reference == 0:
        pytest.skip("No matching keys loaded into LTX-2 transformer models.")

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
    assert_close(ref_audio_out, fastvideo_audio_out, atol=1e-2, rtol=1e-2)
