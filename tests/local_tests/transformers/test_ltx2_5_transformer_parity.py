# SPDX-License-Identifier: Apache-2.0
"""Focused parity coverage for the config-gated LTX-2.5 DiT extension.

The CPU tests use random weights to compare the new stateful/functional delta
against pinned ``ltx-core`` v1.2.0. The real-weight test exercises the official
builder and FastVideo's production ``TransformerLoader`` when the gated assets
have been provisioned.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.environ.get("LTX2_5_OFFICIAL_REF_DIR", REPO_ROOT / "LTX-2-Reference"))
OFFICIAL_SRC = OFFICIAL_REF_DIR / "packages" / "ltx-core" / "src"
OFFICIAL_WEIGHTS_DIR = Path(
    os.environ.get(
        "LTX2_5_OFFICIAL_WEIGHTS_DIR",
        REPO_ROOT / "official_weights" / "LTX-2.5" / "diffusion_models",
    )
)
CONVERTED_WEIGHTS_DIR = Path(
    os.environ.get(
        "LTX2_5_CONVERTED_WEIGHTS_DIR",
        REPO_ROOT / "converted_weights" / "ltx2_5",
    )
)
PARITY_SCOPE = "both"


@pytest.fixture(autouse=True)
def _setup_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set default environment variables for all tests in this module."""
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29525")
    monkeypatch.setenv("DISABLE_SP", "1")
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _import_official(module_name: str):
    if not OFFICIAL_SRC.is_dir():
        pytest.skip(f"Pinned LTX-2 v1.2.0 reference is missing: {OFFICIAL_REF_DIR}")
    if str(OFFICIAL_SRC) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_SRC))
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report the exact missing reference dependency.
        pytest.skip(f"Cannot import pinned official module {module_name}: {exc}")


def _tiny_model_kwargs() -> dict:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "cross_attention_dim": 8,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 4,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "audio_cross_attention_dim": 8,
        "cross_attention_adaln": True,
    }


@pytest.mark.parametrize("bias", (False, True))
def test_feed_forward_random_weight_parity(bias: bool) -> None:
    """The 2.5 bias gate must change both FFN linears, not their math/layout."""
    official_module = _import_official("ltx_core.model.transformer.feed_forward")
    from fastvideo.models.dits.ltx2 import FeedForward as FastVideoFeedForward

    torch.manual_seed(20260812)
    official = official_module.FeedForward(dim=8, dim_out=8, bias=bias).eval()
    fastvideo = FastVideoFeedForward(dim=8, dim_out=8, bias=bias).eval()

    official_state = official.state_dict()
    fastvideo_state = fastvideo.state_dict()
    assert official_state.keys() == fastvideo_state.keys()
    fastvideo.load_state_dict(official_state, strict=True)

    inputs = torch.randn(2, 5, 8)
    with torch.inference_mode():
        expected = official(inputs)
        actual = fastvideo(inputs)
    assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_keyframe_absolute_embedding_random_weight_parity() -> None:
    """Match upstream marker semantics for positive, zero, and negative mask values."""
    official_module = _import_official("ltx_core.model.transformer.transformer_args")
    from fastvideo.models.dits.ltx2 import apply_keyframes_absolute_embedding

    torch.manual_seed(20260812)
    hidden_states = torch.randn(2, 4, 8)
    embedding = torch.randn(1, 8)
    keyframes_mask = torch.tensor(
        [[[1.0], [0.0], [-1.0], [2.0]], [[0.0], [1.0], [0.0], [0.5]]]
    )
    provider = lambda: embedding

    expected = official_module.apply_keyframes_absolute_embedding(hidden_states, keyframes_mask, provider)
    actual = apply_keyframes_absolute_embedding(hidden_states, keyframes_mask, provider)
    assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert apply_keyframes_absolute_embedding(hidden_states, None, provider) is hidden_states


@pytest.mark.parametrize("dynamic_prompt", (False, True))
def test_cross_attention_adaln_static_and_dynamic_prompt_parity(dynamic_prompt: bool) -> None:
    """Exercise the static 2.5 K/V table and legacy dynamic prompt AdaLN paths."""
    official_module = _import_official("ltx_core.model.transformer.transformer")
    from fastvideo.models.dits.ltx2 import _rms_norm_dispatch
    from fastvideo.models.dits.ltx2 import apply_cross_attention_adaln

    torch.manual_seed(20260812)
    batch_size, sequence_length, dim = 2, 3, 8
    x = torch.randn(batch_size, sequence_length, dim)
    context = torch.randn_like(x)
    q_shift = torch.randn_like(x)
    q_scale = torch.randn_like(x)
    q_gate = torch.randn_like(x)
    prompt_scale_shift_table = torch.randn(2, dim)
    prompt_timestep = torch.randn(batch_size, sequence_length, 2 * dim) if dynamic_prompt else None

    def attention(inputs, *, context, mask):
        assert mask is None
        return inputs + context

    expected = official_module.apply_cross_attention_adaln(
        _rms_norm_dispatch(x, eps=1e-6),
        context,
        attention,
        q_shift,
        q_scale,
        q_gate,
        prompt_scale_shift_table,
        prompt_timestep,
    )
    actual = apply_cross_attention_adaln(
        x,
        context,
        attention,
        q_shift,
        q_scale,
        q_gate,
        prompt_scale_shift_table,
        prompt_timestep,
    )
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("use_prompt_adaln_single", "ff_bias", "audio_ff_bias", "use_keyframes_abs_pos_embedding"),
    (
        (True, True, True, False),  # LTX-2/2.3-compatible defaults.
        (False, False, False, True),  # LTX-2.5-style architecture delta.
    ),
)
def test_variant_structure_matches_official(
    use_prompt_adaln_single: bool,
    ff_bias: bool,
    audio_ff_bias: bool,
    use_keyframes_abs_pos_embedding: bool,
) -> None:
    """Compare the exact parameter surface controlled by the four new flags."""
    official_model_module = _import_official("ltx_core.model.transformer.model")
    from fastvideo.models.dits.ltx2 import LTXModel as FastVideoLTXModel
    from fastvideo.models.dits.ltx2 import LTXModelType as FastVideoLTXModelType

    common = _tiny_model_kwargs()
    variant = {
        "use_prompt_adaln_single": use_prompt_adaln_single,
        "ff_bias": ff_bias,
        "audio_ff_bias": audio_ff_bias,
        "use_keyframes_abs_pos_embedding": use_keyframes_abs_pos_embedding,
    }
    official = official_model_module.LTXModel(
        model_type=official_model_module.LTXModelType.AudioVideo,
        **common,
        **variant,
    )
    fastvideo = FastVideoLTXModel(
        model_type=FastVideoLTXModelType.AudioVideo,
        caption_proj_before_connector=True,
        **common,
        **variant,
    )

    assert (official.prompt_adaln_single is not None) == (fastvideo.prompt_adaln_single is not None)
    assert (official.audio_prompt_adaln_single is not None) == (fastvideo.audio_prompt_adaln_single is not None)

    official_state = official.state_dict()
    fastvideo_state = fastvideo.state_dict()
    for prefix in ("transformer_blocks.0.ff.", "transformer_blocks.0.audio_ff."):
        official_keys = {name for name in official_state if name.startswith(prefix)}
        fastvideo_keys = {name for name in fastvideo_state if name.startswith(prefix)}
        assert official_keys == fastvideo_keys

    keyframe_name = "keyframes_abs_pos_embedding"
    assert (keyframe_name in official_state) == use_keyframes_abs_pos_embedding
    assert (keyframe_name in fastvideo_state) == use_keyframes_abs_pos_embedding
    if use_keyframes_abs_pos_embedding:
        assert official_state[keyframe_name].shape == fastvideo_state[keyframe_name].shape == (1, 8)


def test_arch_config_defaults_preserve_pre_2_5_checkpoints() -> None:
    from fastvideo.configs.models.dits.ltx2 import LTX2VideoArchConfig

    config = LTX2VideoArchConfig()
    assert config.use_prompt_adaln_single is True
    assert config.ff_bias is True
    assert config.audio_ff_bias is True
    assert config.use_keyframes_abs_pos_embedding is False


def _read_transformer_metadata(official_path: Path) -> dict:
    loader_module = _import_official("ltx_core.loader.sft_loader")
    metadata = loader_module.SafetensorsModelStateDictLoader().metadata(str(official_path))
    transformer = metadata.get("config", {}).get("transformer", {})
    if not transformer:
        pytest.fail(f"LTX-2.5 checkpoint has no config.transformer metadata: {official_path}", pytrace=False)
    return transformer


def _fastvideo_config_from_metadata(metadata: dict):
    from fastvideo.configs.models.dits import LTX2VideoConfig

    config = LTX2VideoConfig()
    arch = config.arch_config
    for name in (
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "cross_attention_dim",
        "caption_channels",
        "norm_eps",
        "positional_embedding_theta",
        "positional_embedding_max_pos",
        "timestep_scale_multiplier",
        "use_middle_indices_grid",
        "rope_type",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_in_channels",
        "audio_out_channels",
        "audio_cross_attention_dim",
        "audio_positional_embedding_max_pos",
        "av_ca_timestep_scale_multiplier",
        "in_channels",
        "out_channels",
        "cross_attention_adaln",
        "caption_proj_before_connector",
        "apply_gated_attention",
        "use_prompt_adaln_single",
        "ff_bias",
        "audio_ff_bias",
        "use_keyframes_abs_pos_embedding",
    ):
        if name in metadata:
            setattr(arch, name, metadata[name])
    arch.double_precision_rope = metadata.get("frequencies_precision", "") == "float64"
    return config


@pytest.mark.parametrize("variant", ("dev", "distilled"))
def test_real_weights_load_strictly_through_production_loader(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict-load both gated 2.5 transformer variants when provisioned."""
    official_path = OFFICIAL_WEIGHTS_DIR / f"ltx-2.5-22b-{variant}-transformer-bf16.safetensors"
    converted_dir = CONVERTED_WEIGHTS_DIR / variant / "transformer"
    missing = [str(path) for path in (official_path, converted_dir) if not path.exists()]
    if missing:
        pytest.skip(f"Gated LTX-2.5 transformer assets are not provisioned: {missing}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.fail("Provisioned LTX-2.5 weights require a bf16-capable CUDA GPU", pytrace=False)

    official_loader = _import_official("ltx_core.loader.single_gpu_model_builder")
    official_transformer = _import_official("ltx_core.model.transformer")

    metadata = _read_transformer_metadata(official_path)
    official = official_loader.SingleGPUModelBuilder(
        model_class_configurator=official_transformer.LTXModelConfigurator,
        model_path=str(official_path),
        model_sd_ops=official_transformer.LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=torch.device("cuda:0"), dtype=torch.bfloat16)
    assert official.use_prompt_adaln_single == metadata.get("use_prompt_adaln_single", True)
    assert official.use_keyframes_abs_pos_embedding == metadata.get("use_keyframes_abs_pos_embedding", False)
    del official
    torch.cuda.empty_cache()

    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits import ltx2 as fastvideo_ltx2
    from fastvideo.models.loader.component_loader import TransformerLoader

    monkeypatch.setattr(fastvideo_ltx2, "get_sp_world_size", lambda: 1)

    config = _fastvideo_config_from_metadata(metadata)
    args = FastVideoArgs(
        model_path=str(converted_dir),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=config, dit_precision="bf16"),
    )
    args.device = torch.device("cuda:0")
    model = TransformerLoader().load(str(converted_dir), args)
    assert model.model.use_prompt_adaln_single == metadata.get("use_prompt_adaln_single", True)
    assert model.model.use_keyframes_abs_pos_embedding == metadata.get("use_keyframes_abs_pos_embedding", False)
