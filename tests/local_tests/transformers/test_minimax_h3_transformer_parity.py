# SPDX-License-Identifier: Apache-2.0
"""CPU synthetic parity for the native MiniMax H3 Transformer."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch.testing import assert_close


os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
os.environ["DIFFUSERS_ATTN_BACKEND"] = "native"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29613")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SRC = REPO_ROOT / "DiffusersMiniMaxH3" / "src"
OFFICIAL_TRANSFORMER = OFFICIAL_SRC / "diffusers" / "models" / "transformers" / "transformer_minimax_h3.py"
OFFICIAL_TRANSFORMER_SHA256 = "45f1d4a4e9e72128f0d6c5b5d66379b61131a354b58d2776280a83f6c89f8d51"

assert OFFICIAL_TRANSFORMER.is_file(), f"Pinned Diffusers MiniMax H3 source is missing: {OFFICIAL_TRANSFORMER}"
assert hashlib.sha256(OFFICIAL_TRANSFORMER.read_bytes()).hexdigest() == OFFICIAL_TRANSFORMER_SHA256, (
    "Diffusers MiniMax H3 reference changed; refresh the pinned source audit before updating parity."
)
sys.path.insert(0, str(OFFICIAL_SRC))

from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel as OfficialMiniMaxH3Transformer,
)

from fastvideo.configs.models.dits.minimax_h3 import (  # noqa: E402
    MiniMaxH3ArchConfig,
    MiniMaxH3Config,
)
from fastvideo.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.dits.minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel as FastVideoMiniMaxH3Transformer,
)
from fastvideo.platforms import current_platform  # noqa: E402


TINY_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}
NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8


@pytest.fixture(scope="module", autouse=True)
def distributed_runtime():
    """Initialize the SP=1 process groups required by DistributedAttention."""
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _build_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    """Strict-load one deterministic official random state into both models."""
    torch.manual_seed(20260802)
    official = OfficialMiniMaxH3Transformer(**TINY_CONFIG).eval()
    arch_config = MiniMaxH3ArchConfig(**TINY_CONFIG)
    fastvideo_config = MiniMaxH3Config(arch_config=arch_config)

    # FastVideo's CPU platform intentionally has no production backend
    # resolver. This local test selects the existing SDPA implementation only
    # while constructing DistributedAttention; no production selector changes.
    with patch.object(
        current_platform,
        "get_attn_backend_cls",
        return_value="fastvideo.attention.backends.sdpa.SDPABackend",
    ):
        fastvideo = FastVideoMiniMaxH3Transformer(fastvideo_config, dict(TINY_CONFIG)).eval()

    official_state = official.state_dict()
    fastvideo_state = fastvideo.state_dict()
    assert list(fastvideo_state) == list(official_state)
    assert all(fastvideo_state[name].shape == official_state[name].shape for name in official_state)
    incompatible = fastvideo.load_state_dict(official_state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return official, fastvideo


def _make_inputs() -> dict[str, torch.Tensor]:
    """Build one padless packed layout with three modalities and three times."""
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    timestep_indices[video_indices[:2]] = 2

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 4
    position_ids[video_indices, 2] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32) % 2

    video_patch_dim = TINY_CONFIG["in_channels"] * 4
    return {
        "hidden_states": torch.randn(2, NUM_VIDEO_TOKENS, video_patch_dim, generator=generator),
        "audio_hidden_states": torch.randn(
            2,
            NUM_AUDIO_TOKENS,
            TINY_CONFIG["audio_in_channels"],
            generator=generator,
        ),
        "encoder_hidden_states": torch.randn(
            2,
            NUM_TEXT_TOKENS,
            TINY_CONFIG["text_dim"],
            generator=generator,
        ),
        "timestep": torch.tensor([0.7, 0.3, 0.999], dtype=torch.float32),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "return_dict": False,
    }


def _capture_activations(model: torch.nn.Module, names: tuple[str, ...]) -> tuple[dict[str, list[torch.Tensor]], list]:
    activations: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def capture(name: str):
        def hook(_module, _inputs, output) -> None:
            assert isinstance(output, torch.Tensor)
            activations[name].append(output.detach().clone())

        return hook

    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture(name)) for name in names]
    return activations, handles


def test_minimax_h3_transformer_matches_pinned_diffusers() -> None:
    """Compare both output heads numerically with identical weights and rows."""
    source_file = Path(inspect.getsourcefile(OfficialMiniMaxH3Transformer) or "").resolve()
    assert source_file == OFFICIAL_TRANSFORMER.resolve()
    official, fastvideo = _build_models()
    inputs = _make_inputs()
    activation_names = (
        "token_refiner.refiner_blocks.0",
        "transformer_blocks.0",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    )
    official_activations, official_handles = _capture_activations(official, activation_names)
    fastvideo_activations, fastvideo_handles = _capture_activations(fastvideo, activation_names)

    with torch.inference_mode():
        official_video, official_audio = official(**inputs)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fastvideo_video, fastvideo_audio = fastvideo(**inputs)
    for handle in official_handles + fastvideo_handles:
        handle.remove()

    assert_close(fastvideo_video, official_video, atol=1e-5, rtol=1e-5)
    assert_close(fastvideo_audio, official_audio, atol=1e-5, rtol=1e-5)
    for name in activation_names:
        assert len(fastvideo_activations[name]) == len(official_activations[name])
        for result, expected in zip(fastvideo_activations[name], official_activations[name], strict=True):
            assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_minimax_h3_transformer_preserves_mixed_dtype_islands() -> None:
    """Keep only the released input/time/output modules in FP32."""
    _, fastvideo = _build_models()
    fastvideo.to(dtype=torch.bfloat16)

    assert fastvideo.proj_in.weight.dtype == torch.float32
    assert fastvideo.audio_proj_in.weight.dtype == torch.float32
    assert fastvideo.time_embedder.linear_1.weight.dtype == torch.float32
    assert fastvideo.proj_out.weight.dtype == torch.float32
    assert fastvideo.audio_proj_out.weight.dtype == torch.float32
    assert fastvideo.rope.inv_freq.dtype == torch.float32
    assert fastvideo.context_embedder.weight.dtype == torch.bfloat16
    assert fastvideo.token_refiner.refiner_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert fastvideo.transformer_blocks[0].adaln_proj.linear.weight.dtype == torch.bfloat16
    assert fastvideo.norm_out.norm.weight.dtype == torch.bfloat16


def test_minimax_h3_arch_defaults_match_pinned_reference() -> None:
    """Pin the released 33B architecture independently of the tiny fixture."""
    config = MiniMaxH3ArchConfig()
    assert config.num_layers == 50
    assert config.num_refiner_layers == 2
    assert config.hidden_size == 5376
    assert config.num_attention_heads == 56
    assert config.attention_head_dim == 128
    assert config.ffn_dim == 14336
    assert config.text_dim == 5120
    assert config.in_channels == 24
    assert config.audio_in_channels == 32
    assert 2 * 3 * config.rope_freq_dim == 96


def test_minimax_h3_transformer_rejects_semantic_padding() -> None:
    """Keep tag -1 out of the model-owned sequence-parallel padding path."""
    _, fastvideo = _build_models()
    inputs = _make_inputs()
    inputs["token_tags"] = inputs["token_tags"].clone()
    inputs["token_tags"][-1] = -1

    with pytest.raises(ValueError, match="semantically padless"):
        fastvideo(**inputs)
