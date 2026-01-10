# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

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
from fastvideo.models.loader.component_loader import TransformerLoader


def _read_transformer_config(config: dict) -> dict:
    transformer_config = config.get("transformer", {})
    if not transformer_config:
        raise ValueError("Missing transformer config in LTX-2 metadata.")
    return transformer_config


def _infer_patch_params(in_channels: int) -> tuple[int, int]:
    patch_size = 1
    num_channels_latents = 128
    for candidate in (8, 16, 32, 64, 128):
        if in_channels % candidate != 0:
            continue
        patch_volume = in_channels // candidate
        root = int(round(patch_volume**0.5))
        if root * root == patch_volume:
            patch_size = root
            num_channels_latents = candidate
            break
    print(
        f"[LTX2 TEST] Inferred patch_size={patch_size}, "
        f"num_channels_latents={num_channels_latents}"
    )
    return patch_size, num_channels_latents


def _attach_block_sum_logging(
    model: torch.nn.Module,
    log_path: Path,
    label: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    def _format_sum(tensor: torch.Tensor | None) -> str:
        if tensor is None:
            return "None"
        return f"{tensor.float().sum().item():.6f}"

    def _hook(module, inputs, outputs):  # noqa: ANN001
        if isinstance(outputs, tuple):
            video_args, audio_args = outputs
            video_sum = _format_sum(video_args.x if video_args is not None else None)
            audio_sum = _format_sum(audio_args.x if audio_args is not None else None)
        else:
            video_sum = _format_sum(outputs)
            audio_sum = "None"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{label}:{module.idx}:video_sum={video_sum},audio_sum={audio_sum}\n")

    for block in model.transformer_blocks:
        block.register_forward_hook(_hook)


def _attach_block_detail_logging(
    model: torch.nn.Module,
    log_path: Path,
    label: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    def _format_sum(tensor: torch.Tensor | None) -> str:
        if tensor is None:
            return "None"
        return f"{tensor.float().sum().item():.6f}"

    def _hook_factory(block_idx: int, name: str):
        def _hook(_module, _inputs, outputs):  # noqa: ANN001
            if isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs
            out_sum = _format_sum(out if torch.is_tensor(out) else None)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{label}:{block_idx}:{name}:out_sum={out_sum}\n")
        return _hook

    for block in model.transformer_blocks:
        idx = block.idx
        for name in (
            "attn1",
            "attn2",
            "ff",
            "audio_attn1",
            "audio_attn2",
            "audio_ff",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ):
            if hasattr(block, name):
                getattr(block, name).register_forward_hook(_hook_factory(idx, name))

    def _output_hook(name: str):
        def _hook(_module, _inputs, outputs):  # noqa: ANN001
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            out_sum = _format_sum(out if torch.is_tensor(out) else None)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{label}:output:{name}:out_sum={out_sum}\n")
        return _hook

    for name in ("proj_out", "audio_proj_out"):
        if hasattr(model, name):
            getattr(model, name).register_forward_hook(_output_hook(name))


def test_ltx2_transformer_parity():
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
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import (LTXModelConfigurator,
                                                LTXV_MODEL_COMFY_RENAMING_MAP)
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.types import VideoLatentShape
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

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
        repo_root / "ltx2_debug" / "fastvideo.log",
        "fastvideo",
        debug_logs,
    )
    _attach_block_sum_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference.log",
        "reference",
        debug_logs,
    )
    _attach_block_detail_logging(
        fastvideo_model.model,
        repo_root / "ltx2_debug" / "fastvideo_detail.log",
        "fastvideo",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )
    _attach_block_detail_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference_detail.log",
        "reference",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )

    patchifier = VideoLatentPatchifier(patch_size=cfg.patch_size[1])
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

    video = Modality(
        enabled=True,
        latent=latents,
        timesteps=timestep,
        positions=positions,
        context=encoder_hidden_states,
        context_mask=None,
    )

    with torch.no_grad():
        ref_out, _ = reference_model(
            video=video,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(batch_size),
        )
        ref_out = patchifier.unpatchify(ref_out, output_shape=video_shape)
        print(f"[LTX2 TEST] Reference model output shape: {ref_out.shape}")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            fastvideo_out = fastvideo_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            print(f"[LTX2 TEST] FastVideo model output shape: {fastvideo_out.shape}")
    assert ref_out.shape == fastvideo_out.shape
    assert ref_out.dtype == fastvideo_out.dtype
    assert_close(ref_out, fastvideo_out, atol=1e-4, rtol=1e-4)
