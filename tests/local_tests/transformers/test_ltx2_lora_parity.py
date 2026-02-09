# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.configs.models.dits import LTX2VideoConfig
from fastvideo.configs.pipelines import LTX2T2VConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from .test_ltx2 import _infer_patch_params, _read_transformer_config


def test_ltx2_transformer_lora_parity():
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
    lora_path = Path(
        os.getenv(
            "LTX2_LORA_PATH",
            "official_ltx_weights/ltx-2-19b-distilled-lora-384.safetensors",
        )
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 official weights not found at {official_path}")
    if not lora_path.exists():
        pytest.skip(f"LTX-2 distilled LoRA not found at {lora_path}")
    if not diffusers_root.exists():
        pytest.skip(f"LTX-2 diffusers weights not found at {diffusers_root}")

    try:
        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
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
        pytest.skip("LTX-2 LoRA parity test requires CUDA for attention backends.")

    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"

    pipeline_config = LTX2T2VConfig()
    pipeline_config.dit_config = config
    pipeline_config.dit_precision = precision_str

    args = FastVideoArgs(
        model_path=str(diffusers_root),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        pin_cpu_memory=False,
        lora_path=str(lora_path),
        lora_nickname="refine",
        pipeline_config=pipeline_config,
    )
    args.device = device

    fastvideo_pipeline = LTX2Pipeline(
        model_path=str(diffusers_root),
        fastvideo_args=args,
        required_config_modules=["transformer"],
    )
    fastvideo_model = fastvideo_pipeline.modules["transformer"].to(
        device=device, dtype=precision)
    fastvideo_model.eval()

    lora_layers = fastvideo_pipeline.lora_layers.get("transformer")
    assert lora_layers is not None, "LoRA layers were not created."
    applied_layers = [
        layer for _, layer in lora_layers.all_lora_layers()
        if not layer.disable_lora
    ]
    assert applied_layers, "LoRA adapter did not match any layers."

    fastvideo_state = fastvideo_model.state_dict()

    def _canonical(name: str) -> str:
        while name.startswith("model."):
            name = name[len("model."):]
        return name

    fastvideo_weights: dict[str, torch.Tensor] = {}
    for key, value in fastvideo_state.items():
        if key.endswith(".base_layer.weight"):
            trimmed = key[:-len(".base_layer.weight")]
            fastvideo_weights[_canonical(trimmed)] = value
            continue
        if key.endswith(".weight") and ".base_layer." not in key:
            fastvideo_weights[_canonical(key[:-len(".weight")])] = value

    def _iter_lora_layers(path: Path) -> list[str]:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            layers = set()
            for key in f.keys():
                if key.endswith(".lora_A.weight"):
                    layer = key.replace("diffusion_model.", "").replace(
                        ".lora_A.weight", "")
                    layers.add(layer)
            return sorted(layers)

    def _get_tensor(path: Path, key: str) -> torch.Tensor | None:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            if key not in f.keys():
                return None
            return f.get_tensor(key)

    lora_layers = _iter_lora_layers(lora_path)
    assert lora_layers, "No LoRA layers found in the adapter."

    checked = 0
    for layer in lora_layers:
        base_key = f"model.diffusion_model.{layer}.weight"
        lora_a_key = f"diffusion_model.{layer}.lora_A.weight"
        lora_b_key = f"diffusion_model.{layer}.lora_B.weight"
        fast_weight = fastvideo_weights.get(layer)
        if fast_weight is None:
            continue
        base_weight = _get_tensor(official_path, base_key)
        lora_a = _get_tensor(lora_path, lora_a_key)
        lora_b = _get_tensor(lora_path, lora_b_key)
        if base_weight is None or lora_a is None or lora_b is None:
            continue
        expected = (base_weight.to(torch.bfloat16) +
                    torch.matmul(lora_b.to(torch.bfloat16),
                                 lora_a.to(torch.bfloat16)))
        actual = fast_weight.detach().cpu().to(torch.bfloat16)
        assert_close(actual.float(), expected.float(), atol=5e-4, rtol=5e-4)
        checked += 1
        if checked >= 10:
            break

    assert checked > 0, "No matching LoRA layers found for parity checks."

    del fastvideo_model
    del fastvideo_pipeline
    LoRAPipeline.lora_layers.clear()
    LoRAPipeline.lora_adapters.clear()
    LoRAPipeline.cur_adapter_name = ""
    LoRAPipeline.cur_adapter_path = ""
    LoRAPipeline.lora_initialized = False
    import gc
    gc.collect()
    torch.cuda.empty_cache()
