# SPDX-License-Identifier: Apache-2.0
"""Z-Image transformer parity.

Coverage scope: both. The tiny CPU tests compare the pinned implementation and
FastVideo model directly. The production-loader test activates when the real
transformer subfolder is available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.dits import ZImageDiTConfig
from fastvideo.models.dits.zimage import ZImageAttention, ZImageTransformer2DModel, _prepare_attention_mask


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("ZIMAGE_OFFICIAL_REF_DIR", REPO_ROOT / "Z-Image"))
OFFICIAL_SRC = OFFICIAL_REF_DIR / "src"
TRANSFORMER_DIR = Path(
    os.getenv("ZIMAGE_TRANSFORMER_DIR", REPO_ROOT / "official_weights" / "Z-Image" / "transformer"))
REFERENCE_REVISION = "26f23eda626ffadda020b04ff79488e1d72004cd"
HF_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
HF_KEY_COUNT = 521
HF_KEY_SHA256 = "3a9216f208c1873b2cf06394411a53e1e95e10fae3b01dca0f7223556e47c354"
PARITY_SCOPE = "both"

PRODUCTION_CONFIG = {
    "all_patch_size": (2, ),
    "all_f_patch_size": (1, ),
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2560,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": (32, 48, 48),
    "axes_lens": (1536, 512, 512),
}

TINY_CONFIG = {
    "all_patch_size": (2, ),
    "all_f_patch_size": (1, ),
    "in_channels": 2,
    "dim": 12,
    "n_layers": 1,
    "n_refiner_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 8,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": (2, 2, 2),
    "axes_lens": (192, 16, 16),
}


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_transformer_cls():
    source_file = OFFICIAL_SRC / "zimage" / "transformer.py"
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Pinned Z-Image reference clone not found: {OFFICIAL_REF_DIR}")
    if not source_file.is_file():
        pytest.fail(f"Z-Image reference clone is incomplete; missing {source_file}")

    try:
        revision = subprocess.run(
            ["git", "-C", str(OFFICIAL_REF_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"Cannot verify Z-Image reference revision: {exc}")
    assert revision == REFERENCE_REVISION, (
        f"expected Z-Image reference {REFERENCE_REVISION}, got {revision}")

    module_names = (
        "config",
        "config.inference",
        "config.model",
        "utils",
        "utils.import_utils",
        "utils.attention",
        "_zimage_reference_transformer",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    added_path = str(OFFICIAL_SRC) not in sys.path
    if added_path:
        sys.path.insert(0, str(OFFICIAL_SRC))

    try:
        for name in module_names:
            sys.modules.pop(name, None)
        utils_package = ModuleType("utils")
        utils_package.__path__ = [str(OFFICIAL_SRC / "utils")]
        sys.modules["utils"] = utils_package
        _load_source_module("utils.import_utils", OFFICIAL_SRC / "utils" / "import_utils.py")
        _load_source_module("utils.attention", OFFICIAL_SRC / "utils" / "attention.py")
        module = _load_source_module("_zimage_reference_transformer", source_file)
        assert Path(module.__file__).resolve() == source_file.resolve()
        yield module.ZImageTransformer2DModel
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]
        if added_path:
            sys.path.remove(str(OFFICIAL_SRC))


def _build_fastvideo(config_values: dict) -> ZImageTransformer2DModel:
    config = ZImageDiTConfig()
    config.update_model_arch(config_values)
    return ZImageTransformer2DModel(
        config=config,
        hf_config={"_class_name": "ZImageTransformer2DModel", **config_values},
    )


def _state_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def test_zimage_transformer_production_meta_key_surface(reference_transformer_cls):
    from fastvideo.models.registry import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls("ZImageTransformer2DModel")
    assert model_cls is ZImageTransformer2DModel

    with torch.device("meta"):
        reference = reference_transformer_cls(**PRODUCTION_CONFIG)
        fastvideo = _build_fastvideo(PRODUCTION_CONFIG)

    reference_shapes = _state_shapes(reference)
    fastvideo_shapes = _state_shapes(fastvideo)
    assert fastvideo_shapes == reference_shapes
    keys = sorted(reference_shapes)
    key_digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
    assert len(keys) == HF_KEY_COUNT
    assert key_digest == HF_KEY_SHA256
    print(f"Z-Image production transformer key/shape surface: {len(reference_shapes)} tensors")


def test_zimage_transformer_tiny_cpu_parity(reference_transformer_cls):
    reference = reference_transformer_cls(**TINY_CONFIG).eval()
    generator = torch.Generator(device="cpu").manual_seed(123)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)

    fastvideo = _build_fastvideo(TINY_CONFIG).eval()
    assert _state_shapes(fastvideo) == _state_shapes(reference)
    fastvideo.load_state_dict(reference.state_dict(), strict=True)

    generator.manual_seed(456)
    images = [
        torch.randn((2, 1, 4, 4), generator=generator),
        torch.randn((2, 1, 8, 10), generator=generator),
    ]
    caption_features = [
        torch.randn((3, 8), generator=generator),
        torch.randn((35, 8), generator=generator),
    ]
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32)

    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        reference_outputs = reference(images, timestep, caption_features)[0]
        positional_outputs = fastvideo(images, caption_features, timestep)[0]
        keyword_outputs = fastvideo(
            hidden_states=images,
            encoder_hidden_states=caption_features,
            timestep=timestep,
        )[0]

    assert len(positional_outputs) == len(keyword_outputs) == len(reference_outputs) == 2
    for positional, actual, expected in zip(positional_outputs, keyword_outputs, reference_outputs):
        assert actual.shape == expected.shape
        assert expected.abs().max() > 0
        assert_close(positional, expected, atol=1e-6, rtol=1e-6)
        assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_zimage_attention_additive_mask_parity(reference_transformer_cls):
    reference_module = sys.modules[reference_transformer_cls.__module__]
    reference = reference_module.ZImageAttention(12, 2, 2).eval()
    generator = torch.Generator(device="cpu").manual_seed(321)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)

    fastvideo = ZImageAttention(12, 2, 2).eval()
    fastvideo.load_state_dict(reference.state_dict(), strict=True)
    hidden_states = torch.randn((2, 5, 12), generator=generator)
    attention_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    additive_mask = _prepare_attention_mask(attention_mask, hidden_states.dtype)
    assert additive_mask is not None
    assert additive_mask.dtype == hidden_states.dtype
    assert additive_mask.shape == (2, 1, 1, 5)
    assert additive_mask[0, 0, 0, 0] == 0
    assert torch.isneginf(additive_mask[0, 0, 0, -1])

    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = reference(hidden_states, attention_mask=attention_mask)
        actual = fastvideo(hidden_states, attention_mask=attention_mask)
    assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_zimage_transformer_rejects_sequence_parallelism(monkeypatch: pytest.MonkeyPatch):
    model = _build_fastvideo(TINY_CONFIG)
    monkeypatch.setattr("fastvideo.models.dits.zimage.model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr("fastvideo.models.dits.zimage.get_sp_world_size", lambda: 2)

    with pytest.raises(NotImplementedError, match="sp_size=1"):
        model(
            hidden_states=[torch.zeros(2, 1, 4, 4)],
            encoder_hidden_states=[torch.zeros(3, 8)],
            timestep=torch.tensor([0.5]),
        )


def _has_real_weights() -> bool:
    return any(TRANSFORMER_DIR.glob("*.safetensors"))


def _load_real_config() -> dict:
    with (TRANSFORMER_DIR / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    assert config.pop("_class_name") == "ZImageTransformer2DModel"
    config.pop("_diffusers_version", None)
    config.pop("_name_or_path", None)
    for key, expected in PRODUCTION_CONFIG.items():
        actual = config[key]
        if isinstance(expected, tuple):
            actual = tuple(actual)
        assert actual == expected, f"unexpected production config {key}={actual!r}"
    assert set(config) == set(PRODUCTION_CONFIG)
    return config


def _run_fastvideo_production(inputs: tuple, dtype: torch.dtype) -> list[torch.Tensor]:
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader

    precision = "bf16" if dtype == torch.bfloat16 else "fp16"
    args = FastVideoArgs(
        model_path=str(TRANSFORMER_DIR),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=ZImageDiTConfig(), dit_precision=precision),
    )
    model = TransformerLoader().load(str(TRANSFORMER_DIR), args).eval()
    assert isinstance(model, ZImageTransformer2DModel)
    assert args.model_paths["transformer"] == str(TRANSFORMER_DIR)
    assert next(model.parameters()).device.type == "cuda"
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        outputs = model(
            hidden_states=inputs[0],
            encoder_hidden_states=inputs[1],
            timestep=inputs[2],
        )[0]
    result = [output.detach().float().cpu() for output in outputs]
    del model
    torch.cuda.empty_cache()
    return result


def _run_reference_production(reference_transformer_cls, config: dict, inputs: tuple,
                              dtype: torch.dtype) -> list[torch.Tensor]:
    from safetensors.torch import load_file

    state_dict = {}
    for shard in sorted(TRANSFORMER_DIR.glob("*.safetensors")):
        state_dict.update(load_file(str(shard), device="cpu"))
    with torch.device("meta"):
        model = reference_transformer_cls(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model = model.to(device="cuda", dtype=dtype).eval()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        outputs = model(inputs[0], inputs[2], inputs[1])[0]
    result = [output.detach().float().cpu() for output in outputs]
    del model
    torch.cuda.empty_cache()
    return result


@pytest.mark.skipif(
    not _has_real_weights(),
    reason=(
        "Pinned Z-Image transformer weights are absent; set ZIMAGE_TRANSFORMER_DIR "
        f"to Tongyi-MAI/Z-Image-Turbo@{HF_REVISION}/transformer"),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real-weight Z-Image transformer parity requires CUDA")
def test_zimage_transformer_production_loader_forward_parity(reference_transformer_cls):
    config = _load_real_config()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    generator = torch.Generator(device="cuda").manual_seed(789)
    inputs = (
        [torch.randn((16, 1, 4, 4), device="cuda", dtype=dtype, generator=generator)],
        [torch.randn((3, 2560), device="cuda", dtype=dtype, generator=generator)],
        torch.tensor([0.25], device="cuda", dtype=dtype),
    )

    fastvideo_outputs = _run_fastvideo_production(inputs, dtype)
    reference_outputs = _run_reference_production(reference_transformer_cls, config, inputs, dtype)
    assert len(fastvideo_outputs) == len(reference_outputs) == 1

    actual, expected = fastvideo_outputs[0], reference_outputs[0]
    diff = (actual - expected).abs()
    mean_scale = expected.abs().mean().clamp_min(1e-6)
    print(f"Z-Image real-weight parity: max={diff.max():.6f}, mean={diff.mean():.6f}")
    assert diff.mean() / mean_scale < 0.01
    assert_close(actual, expected, atol=1e-2, rtol=1e-2)
