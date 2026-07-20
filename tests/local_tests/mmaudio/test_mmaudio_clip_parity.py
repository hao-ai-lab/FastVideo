# SPDX-License-Identifier: Apache-2.0
"""MMAudio DFN5B OpenCLIP conditioner parity tests."""

from __future__ import annotations

import os
import importlib.util
from functools import cache
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
DFN5B_DIR = Path(
    os.environ.get(
        "MMAUDIO_DFN5B_DIR",
        REPO_ROOT / "official_weights/mmaudio/DFN5B-CLIP-ViT-H-14-384",
    )
)


@cache
def _conversion_module():
    path = REPO_ROOT / "scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("mmaudio_converter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converted_open_clip_tokenizer_parity(tmp_path: Path) -> None:
    import open_clip
    from transformers import CLIPTokenizer

    _conversion_module().write_open_clip_tokenizer(tmp_path)
    converted = CLIPTokenizer.from_pretrained(tmp_path / "tokenizer")
    prompts = ["", "A dog runs past a red car!", "雨の日, cinematic"]
    encoded = converted(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    actual = encoded.input_ids.masked_fill(encoded.attention_mask == 0, 0)
    expected = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")(prompts)
    torch.testing.assert_close(actual, expected)


def _map_open_clip_text(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return _conversion_module().map_open_clip_text_state(state)


def _map_open_clip_vision(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return _conversion_module().map_open_clip_vision_state(state)


def test_mmaudio_dfn_clip_implementation_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("DFN CLIP implementation parity requires CUDA")

    from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg

    from fastvideo.configs.models.encoders.mmaudio_clip import (
        MMAudioDFNCLIPTextArchConfig,
        MMAudioDFNCLIPTextConfig,
        MMAudioDFNCLIPVisionArchConfig,
        MMAudioDFNCLIPVisionConfig,
    )
    from fastvideo.models.encoders.mmaudio_clip import (
        MMAudioDFNCLIPTextEncoder,
        MMAudioDFNCLIPVisionEncoder,
    )
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.forward_context import set_forward_context
    from fastvideo.platforms import AttentionBackendEnum

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    embed_dim = 16
    official = CLIP(
        embed_dim=embed_dim,
        vision_cfg=CLIPVisionCfg(layers=2, width=32, head_width=8, patch_size=8, image_size=16),
        text_cfg=CLIPTextCfg(context_length=8, vocab_size=32, width=32, heads=4, layers=2, eos_id=31),
        quick_gelu=True,
    )
    backends = (AttentionBackendEnum.TORCH_SDPA,)
    text_arch = MMAudioDFNCLIPTextArchConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=128,
        projection_dim=embed_dim,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=8,
        text_len=8,
        eos_token_id=31,
        _supported_attention_backends=backends,
    )
    vision_arch = MMAudioDFNCLIPVisionArchConfig(
        hidden_size=32,
        intermediate_size=128,
        projection_dim=embed_dim,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=16,
        patch_size=8,
        _supported_attention_backends=backends,
    )
    text_encoder = MMAudioDFNCLIPTextEncoder(MMAudioDFNCLIPTextConfig(arch_config=text_arch))
    vision_encoder = MMAudioDFNCLIPVisionEncoder(MMAudioDFNCLIPVisionConfig(arch_config=vision_arch))
    state = official.state_dict()
    text_encoder.load_state_dict(_map_open_clip_text(state), strict=True)
    vision_encoder.load_state_dict(_map_open_clip_vision(state), strict=True)

    device = torch.device("cuda:0")
    official.to(device).eval()
    text_encoder.to(device).eval()
    vision_encoder.to(device).eval()
    tokens = torch.tensor([[1, 2, 3, 31, 0, 0, 0, 0]], device=device)
    images = torch.randn((1, 3, 16, 16), generator=torch.Generator(device=device).manual_seed(1234), device=device)
    with torch.inference_mode():
        expected_text = official.token_embedding(tokens)
        expected_text = expected_text + official.positional_embedding
        expected_text = official.transformer(expected_text, attn_mask=official.attn_mask)
        expected_text = F.normalize(official.ln_final(expected_text), dim=-1)
        expected_image = official.encode_image(images, normalize=True)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            actual_text = text_encoder(tokens).last_hidden_state
            actual_image = vision_encoder(images).last_hidden_state

    torch.testing.assert_close(actual_text, expected_text, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_image, expected_image, atol=2e-5, rtol=2e-5)
    cleanup_dist_env_and_memory()


def test_mmaudio_dfn5b_real_weight_parity() -> None:
    if not DFN5B_DIR.is_dir():
        pytest.skip("DFN5B assets are absent. Set MMAUDIO_DFN5B_DIR to a local apple/DFN5B-CLIP-ViT-H-14-384 snapshot.")
    if not torch.cuda.is_available():
        pytest.skip("DFN5B parity requires CUDA")

    import open_clip

    from fastvideo.configs.models.encoders.mmaudio_clip import (
        MMAudioDFNCLIPTextConfig,
        MMAudioDFNCLIPVisionConfig,
    )
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.encoders.mmaudio_clip import (
        MMAudioDFNCLIPTextEncoder,
        MMAudioDFNCLIPVisionEncoder,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29592")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        official = open_clip.create_model_from_pretrained(
            f"local-dir:{DFN5B_DIR}", return_transform=False)
        state = official.state_dict()
        text_encoder = MMAudioDFNCLIPTextEncoder(
            MMAudioDFNCLIPTextConfig())
        vision_encoder = MMAudioDFNCLIPVisionEncoder(
            MMAudioDFNCLIPVisionConfig())
        text_encoder.load_state_dict(_map_open_clip_text(state), strict=True)
        vision_encoder.load_state_dict(_map_open_clip_vision(state),
                                       strict=True)

        device = torch.device("cuda:0")
        official.to(device).eval()
        text_encoder.to(device).eval()
        vision_encoder.to(device).eval()
        tokens = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")(
            ["A dog runs past a red car!"]).to(device)
        frames = torch.rand(
            (1, 3, 384, 384),
            generator=torch.Generator(device=device).manual_seed(9012),
            device=device,
        )
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=device).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        with torch.inference_mode():
            expected_text = official.token_embedding(tokens)
            expected_text = expected_text + official.positional_embedding
            expected_text = official.transformer(
                expected_text, attn_mask=official.attn_mask)
            expected_text = F.normalize(official.ln_final(expected_text),
                                        dim=-1)
            expected_image = official.encode_image(frames, normalize=True)
            with set_forward_context(current_timestep=0,
                                     attn_metadata=None):
                actual_text = text_encoder(tokens).last_hidden_state
                actual_image = vision_encoder(frames).last_hidden_state

        text_error = (actual_text.float() - expected_text.float()).abs()
        image_error = (actual_image.float() - expected_image.float()).abs()
        print("text_max_abs", text_error.max().item())
        print("image_max_abs", image_error.max().item())
        torch.testing.assert_close(actual_text,
                                   expected_text,
                                   atol=2e-5,
                                   rtol=2e-5)
        torch.testing.assert_close(actual_image,
                                   expected_image,
                                   atol=2e-5,
                                   rtol=2e-5)
    finally:
        cleanup_dist_env_and_memory()
