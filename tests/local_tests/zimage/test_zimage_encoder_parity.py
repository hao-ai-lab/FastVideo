# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image Qwen3 text encoder support in FastVideo.

This compares:
1) direct transformers AutoModel output, and
2) FastVideo's production ``TextEncoderLoader`` passthrough, and
3) FastVideo's shared native Qwen3 encoder (``Qwen3ForCausalLM``, added for Flux2
   Klein and reused here — Z-Image-Turbo's ``Qwen3Model`` checkpoint routes
   to it via the model registry),

using identical local checkpoint, tokenization, and inputs.

Usage:
    pytest tests/local_tests/zimage/test_zimage_encoder_parity.py -v -s
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import gc
import json
import os

import pytest
import torch


def _reclaim_vram() -> None:
    """Actually reclaim VRAM after the caller has ``del``'d its model refs.

    HF transformer models hold reference cycles, so ``del`` alone does not free
    them — a ``gc.collect()`` is required before the CUDA caching allocator will
    release the blocks. Without this the parity test keeps two full encoder
    copies resident and OOMs on a 44 GB GPU.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0
from torch.testing import assert_close
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import safe_open

from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.layers.quantization.absmax_fp8 import AbsMaxFP8MergedParameter
from fastvideo.models.encoders.qwen3 import Qwen3ForCausalLM
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.models.loader.utils import set_default_torch_dtype
from fastvideo.models.registry import ModelRegistry

PARITY_SCOPE = "both"

# Strict-load contract: the shared Qwen3 encoder is body-only (embed_tokens +
# layers + norm, no lm_head). Z-Image-Turbo ships a full Qwen3 checkpoint, so
# `lm_head.weight` is the only key that may go unmatched; anything else means a
# real silent drop. Enforced here in the test since the shared encoder's loader
# is intentionally lenient (it's used by multiple models).
_ALLOWED_UNEXPECTED_KEYS = {"lm_head.weight"}


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_TEXT_ENCODER_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "text_encoder"
ZIMAGE_TOKENIZER_DIR = REPO_ROOT / "official_weights" / "Z-Image" / "tokenizer"


@pytest.fixture(scope="module", autouse=True)
def _init_dist_and_tp_groups():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_safetensors(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def _iter_pretrained_safetensors(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.exists():
        yield from _iter_safetensors(single)
        return

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        idx = _load_json(index)
        shard_names = sorted(set(idx["weight_map"].values()))
        for shard in shard_names:
            yield from _iter_safetensors(model_dir / shard)
        return

    raise FileNotFoundError(
        f"Missing safetensors checkpoint in {model_dir} (expected model.safetensors or model.safetensors.index.json)"
    )


def _pretrained_safetensor_keys(model_dir: Path) -> set[str]:
    return {name for name, _ in _iter_pretrained_safetensors(model_dir)}


def _load_qwen3_config() -> Qwen3TextConfig:
    cfg_raw = _load_json(ZIMAGE_TEXT_ENCODER_DIR / "config.json")
    for key in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(key, None)
    config = Qwen3TextConfig()
    config.update_model_arch(cfg_raw)
    assert config.num_hidden_layers == 36
    assert config.hidden_size == 2560
    assert config.intermediate_size == 9728
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    return config


def _loader_args(cpu_offload: bool) -> SimpleNamespace:
    return SimpleNamespace(
        text_encoder_cpu_offload=cpu_offload,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
    )


def _precision_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Unsupported parity dtype: {dtype}")


def _tokenize_official_prompts(tokenizer, prompts: list[str]):
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    return tokenizer(
        formatted,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )


def _assert_native_load_surface(
    model,
    checkpoint_keys: set[str],
    loaded_params: set[str],
) -> None:
    param_names = {name for name, _ in model.named_parameters()}
    missing = param_names - loaded_params
    assert not missing, f"FastVideo parameters not loaded: {sorted(missing)}"

    normalized_keys = {
        name[len("model."):] if name.startswith("model.") else name
        for name in checkpoint_keys
    }
    mapped_keys: set[str] = set()
    for checkpoint_name in normalized_keys:
        for param_name, weight_name, _ in model.config.arch_config.stacked_params_mapping:
            if weight_name in checkpoint_name:
                mapped_keys.add(checkpoint_name.replace(weight_name, param_name))
                break
        else:
            mapped_keys.add(checkpoint_name)

    unexpected = mapped_keys - param_names - {
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    }
    assert unexpected <= _ALLOWED_UNEXPECTED_KEYS, (
        "Unexpected checkpoint keys not in allowlist: "
        f"{sorted(unexpected - _ALLOWED_UNEXPECTED_KEYS)}"
    )


def _fake_qwen3_weight_target(include_quant_scales: bool = False):
    stacked = Qwen3TextConfig().arch_config.stacked_params_mapping
    params = {
        "layers.0.self_attn.qkv_proj.weight": torch.nn.Parameter(torch.empty(1)),
        "layers.0.mlp.gate_up_proj.weight": torch.nn.Parameter(torch.empty(1)),
    }
    if include_quant_scales:
        qkv_scale = AbsMaxFP8MergedParameter(torch.zeros(3), requires_grad=False)
        qkv_scale.output_partition_sizes = [1, 1, 1]
        gate_up_scale = AbsMaxFP8MergedParameter(torch.zeros(2), requires_grad=False)
        gate_up_scale.output_partition_sizes = [1, 1]
        params.update({
            "layers.0.self_attn.qkv_proj.scale_weight": qkv_scale,
            "layers.0.mlp.gate_up_proj.scale_weight": gate_up_scale,
        })
    for param in params.values():
        if not hasattr(param, "weight_loader"):
            param.weight_loader = (
                lambda target, loaded, *args: target.data.copy_(loaded)
            )

    model = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(stacked_params_mapping=stacked)
        ),
        named_parameters=lambda: params.items(),
    )
    return model, params, stacked


# The provisional bf16 thresholds below come from the historical unpinned
# 35-layer run. Per add-model-02-parity's calibration block, bf16 max-based
# asserts are meaningless for deep encoders — fused-vs-unfused linear order
# alone produces a long tail, so the test uses mean + median instead. The
# pinned snapshot has 36 blocks and must be rerun before these thresholds are
# treated as current evidence.
#
# Historical empirical numbers (unrecorded snapshot, 35 layers, CUDA):
#   last_hidden_state (post-norm):  mean=0.0152, median=0.0117
#   hidden_states[-2] (pre-norm):   mean=0.0754, median=0.0625
# Thresholds below add ~1.6x headroom over observed.
_BF16_MEAN_DRIFT_POST_NORM = 0.025
_BF16_MEAN_DRIFT_PRE_NORM = 0.120
_BF16_MEDIAN_DRIFT_POST_NORM = 0.020
_BF16_MEDIAN_DRIFT_PRE_NORM = 0.100
_FP32_ATOL = 1e-4
_FP32_RTOL = 1e-4


def _print_diag(label: str, ref: torch.Tensor, fv: torch.Tensor) -> torch.Tensor:
    diff = (ref.float() - fv.float()).abs()
    flat = diff.flatten()
    p99 = flat.kthvalue(max(1, int(0.99 * flat.numel()))).values
    print(
        f"[{label}] max_diff={diff.max():.4f}  mean_diff={diff.mean():.4f}  "
        f"median_diff={diff.median():.4f}  p99_diff={p99:.4f}",
        flush=True,
    )
    return diff


@pytest.mark.parametrize(
    ("cpu_offload", "target_device", "expected_device"),
    [
        (True, torch.device("cuda"), torch.device("cpu")),
        (False, torch.device("meta"), torch.device("meta")),
    ],
)
def test_qwen3_production_loader_avoids_device_context_and_honors_placement(
    monkeypatch: pytest.MonkeyPatch,
    cpu_offload: bool,
    target_device: torch.device,
    expected_device: torch.device,
):
    """HF loading stays ambient while final placement honors target/offload."""
    import fastvideo.platforms as platforms

    placements: list[torch.device] = []
    auto_model_calls: list[dict] = []

    class FakeHFModel:
        def eval(self):
            return self

        def to(self, device: torch.device):
            placements.append(torch.device(device))
            return self

    fake_model = FakeHFModel()

    ambient_device = torch.empty(0).device
    default_devices: list[torch.device] = []

    def fake_auto_model_from_pretrained(*args, **kwargs):
        auto_model_calls.append(kwargs)
        default_devices.append(torch.empty(0).device)
        return fake_model

    def fail_causal_lm_from_pretrained(*args, **kwargs):
        raise AssertionError("Qwen text encoding must not load AutoModelForCausalLM")

    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        staticmethod(fake_auto_model_from_pretrained),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fail_causal_lm_from_pretrained),
    )
    monkeypatch.setattr(
        platforms,
        "_current_platform",
        SimpleNamespace(
            is_mps=lambda: False,
            verify_model_arch=lambda _arch: None,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "fastvideo.distributed.get_local_torch_device",
        lambda: torch.device("cuda"),
    )

    config = Qwen3TextConfig()
    # The pinned checkpoint names the causal wrapper even though the official
    # Z-Image loader intentionally asks Transformers for the body-only model.
    config.arch_config.architectures = ["Qwen3ForCausalLM"]
    loaded = TextEncoderLoader().load_model(
        "unused-by-mock",
        config,
        target_device,
        _loader_args(cpu_offload=cpu_offload),
        dtype="fp32",
        use_text_encoder_override=True,
    )

    assert loaded is fake_model
    assert auto_model_calls == [{
        "local_files_only": True,
        "dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }]
    assert default_devices == [ambient_device]
    assert placements == [expected_device]
    assert loaded._fastvideo_input_device == expected_device


def test_set_default_torch_dtype_restores_after_exception():
    original = torch.get_default_dtype()

    with pytest.raises(RuntimeError, match="expected failure"):
        with set_default_torch_dtype(torch.bfloat16):
            raise RuntimeError("expected failure")

    assert torch.get_default_dtype() == original


@pytest.mark.parametrize("checkpoint_prefix", ["", "model."])
def test_qwen3_native_load_accepts_already_fused_weights_and_scales(
    checkpoint_prefix: str,
):
    """Direct FastVideo/state-dict keys satisfy each fused target atomically."""
    fake_model, params, _ = _fake_qwen3_weight_target(include_quant_scales=True)
    expected = {
        name: torch.arange(1, param.numel() + 1, dtype=param.dtype)
        for name, param in params.items()
    }
    weights = [
        (f"{checkpoint_prefix}{name}", expected[name])
        for name in params
    ]

    loaded = Qwen3ForCausalLM.load_weights(fake_model, weights)

    assert loaded == set(params)
    for name, param in params.items():
        assert torch.equal(param.data, expected[name])
    _assert_native_load_surface(
        fake_model,
        {name for name, _ in weights},
        loaded,
    )


def test_qwen3_native_load_rejects_incompatible_fused_quant_scale_shape():
    fake_model, params, _ = _fake_qwen3_weight_target(include_quant_scales=True)
    weights = [
        (name, torch.ones_like(param))
        for name, param in params.items()
        if name != "layers.0.self_attn.qkv_proj.scale_weight"
    ]
    weights.append(("layers.0.self_attn.qkv_proj.scale_weight", torch.ones(1)))

    with pytest.raises(AssertionError, match="Attempted to load weight"):
        Qwen3ForCausalLM.load_weights(fake_model, weights)


def test_qwen3_native_load_uses_real_shard_loader_for_split_quant_scales():
    fake_model, params, stacked = _fake_qwen3_weight_target(include_quant_scales=True)
    weights = [
        (name, torch.ones_like(param))
        for name, param in params.items()
        if name.endswith(".weight")
    ]
    weights.extend(
        (f"model.layers.0.self_attn{weight_name}.scale_weight", torch.tensor(float(index + 1)))
        for index, (_, weight_name, _) in enumerate(stacked[:3])
    )
    weights.extend(
        (f"model.layers.0.mlp{weight_name}.scale_weight", torch.tensor(float(index + 4)))
        for index, (_, weight_name, _) in enumerate(stacked[3:])
    )

    loaded = Qwen3ForCausalLM.load_weights(fake_model, weights)

    assert loaded == set(params)
    assert torch.equal(
        params["layers.0.self_attn.qkv_proj.scale_weight"].data,
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        params["layers.0.mlp.gate_up_proj.scale_weight"].data,
        torch.tensor([4.0, 5.0]),
    )


@pytest.mark.parametrize(
    ("missing_weight_name", "missing_shard_id"),
    [
        (".q_proj", "q"),
        (".k_proj", "k"),
        (".v_proj", "v"),
        (".gate_proj", 0),
        (".up_proj", 1),
    ],
)
def test_qwen3_native_load_rejects_missing_fused_weight_shard(
    missing_weight_name: str,
    missing_shard_id: str | int,
):
    fake_model, _, stacked = _fake_qwen3_weight_target()
    weights = [
        (f"model.layers.0.self_attn{weight_name}.weight", torch.empty(1))
        for _, weight_name, _ in stacked[:3]
        if weight_name != missing_weight_name
    ]
    weights.extend(
        (f"model.layers.0.mlp{weight_name}.weight", torch.empty(1))
        for _, weight_name, _ in stacked[3:]
        if weight_name != missing_weight_name
    )

    with pytest.raises(ValueError, match="Missing required stacked checkpoint shards") as exc_info:
        Qwen3ForCausalLM.load_weights(fake_model, weights)

    destination = (
        "layers.0.self_attn.qkv_proj.weight"
        if isinstance(missing_shard_id, str)
        else "layers.0.mlp.gate_up_proj.weight"
    )
    assert f"{destination}[{missing_shard_id}]" in str(exc_info.value)


@pytest.mark.parametrize(
    ("missing_weight_name", "missing_shard_id"),
    [
        (".q_proj", "q"),
        (".k_proj", "k"),
        (".v_proj", "v"),
        (".gate_proj", 0),
        (".up_proj", 1),
    ],
)
def test_qwen3_native_load_rejects_missing_fused_quant_scale_shard(
    missing_weight_name: str,
    missing_shard_id: str | int,
):
    fake_model, params, stacked = _fake_qwen3_weight_target(include_quant_scales=True)
    weights = [
        (name, torch.empty(1))
        for name in params
        if name.endswith(".weight")
    ]
    weights.extend(
        (f"model.layers.0.self_attn{weight_name}.scale_weight", torch.empty(1))
        for _, weight_name, _ in stacked[:3]
        if weight_name != missing_weight_name
    )
    weights.extend(
        (f"model.layers.0.mlp{weight_name}.scale_weight", torch.empty(1))
        for _, weight_name, _ in stacked[3:]
        if weight_name != missing_weight_name
    )

    with pytest.raises(ValueError, match="Missing required stacked checkpoint shards") as exc_info:
        Qwen3ForCausalLM.load_weights(fake_model, weights)

    destination = (
        "layers.0.self_attn.qkv_proj.scale_weight"
        if isinstance(missing_shard_id, str)
        else "layers.0.mlp.gate_up_proj.scale_weight"
    )
    assert f"{destination}[{missing_shard_id}]" in str(exc_info.value)


@pytest.mark.skipif(not ZIMAGE_TEXT_ENCODER_DIR.exists(), reason="Z-Image text encoder checkpoint required")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(
            torch.bfloat16,
            id="bf16",
            marks=pytest.mark.skipif(
                not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                reason="bf16 parity requires a bf16-capable CUDA device",
            ),
        ),
    ],
)
def test_zimage_qwen3_encoder_parity_forward(dtype: torch.dtype):
    if not ZIMAGE_TOKENIZER_DIR.exists():
        pytest.skip(f"Z-Image tokenizer dir not found: {ZIMAGE_TOKENIZER_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)

    tokenizer = AutoTokenizer.from_pretrained(str(ZIMAGE_TOKENIZER_DIR), local_files_only=True)

    prompts = [
        "A cinematic shot of a rainy neon street at night.",
        "A watercolor illustration of a fox in autumn leaves.",
    ]
    toks = _tokenize_official_prompts(tokenizer, prompts)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    config = _load_qwen3_config()

    ref = AutoModel.from_pretrained(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device=device)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        ref_last = ref_out.last_hidden_state.detach().float().cpu()
        assert ref_out.hidden_states is not None
        ref_hs_m2 = ref_out.hidden_states[-2].detach().float().cpu()

    print(f"[mem] after HF ref forward: {_gpu_mem_mb():.0f} MiB", flush=True)
    # Keep only CPU outputs so each full encoder is resident one at a time.
    del ref, ref_out
    _reclaim_vram()
    print(f"[mem] after freeing HF ref:  {_gpu_mem_mb():.0f} MiB", flush=True)

    production = TextEncoderLoader().load_model(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        config,
        device,
        _loader_args(cpu_offload=False),
        dtype=_precision_name(dtype),
        use_text_encoder_override=True,
    )
    assert production.__class__.__name__ == "Qwen3Model"
    assert not hasattr(production, "lm_head")
    assert production._fastvideo_input_device == device

    with torch.no_grad():
        production_out = production(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        production_last = production_out.last_hidden_state.detach().float().cpu()
        assert production_out.hidden_states is not None
        production_hs_m2 = production_out.hidden_states[-2].detach().float().cpu()

    # Production loading must preserve the independent Transformers oracle
    # exactly; native-kernel drift is assessed separately below.
    assert_close(ref_last, production_last, atol=0.0, rtol=0.0)
    assert_close(ref_hs_m2, production_hs_m2, atol=0.0, rtol=0.0)
    del production, production_out, production_last, production_hs_m2
    _reclaim_vram()

    fv_cls, _ = ModelRegistry.resolve_model_cls("Qwen3Model")
    fv = fv_cls(config).eval()
    loaded = fv.load_weights(_iter_pretrained_safetensors(ZIMAGE_TEXT_ENCODER_DIR))
    assert loaded, "No Qwen3 weights were loaded into FastVideo model"
    fv = fv.to(device=device, dtype=dtype)
    print(f"[mem] after FastVideo load:  {_gpu_mem_mb():.0f} MiB", flush=True)

    _assert_native_load_surface(
        fv,
        _pretrained_safetensor_keys(ZIMAGE_TEXT_ENCODER_DIR),
        loaded,
    )

    with torch.no_grad():
        fv_out = fv(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        fv_last = fv_out.last_hidden_state.detach().float().cpu()
        assert fv_out.hidden_states is not None
        fv_hs_m2 = fv_out.hidden_states[-2].detach().float().cpu()

    # Z-Image uses only valid token embeddings selected by attention mask.
    # Compare parity on that exact path to avoid padded-token artifacts.
    mask_cpu = attention_mask.detach().bool().cpu()
    is_bf16 = dtype == torch.bfloat16

    for i in range(mask_cpu.shape[0]):
        valid = mask_cpu[i]
        ref_last_v = ref_last[i][valid]
        fv_last_v = fv_last[i][valid]
        ref_hs_m2_v = ref_hs_m2[i][valid]
        fv_hs_m2_v = fv_hs_m2[i][valid]

        last_diff = _print_diag(f"batch{i} last_hidden_state {dtype}", ref_last_v, fv_last_v)
        hs_m2_diff = _print_diag(f"batch{i} hidden_states[-2] {dtype}", ref_hs_m2_v, fv_hs_m2_v)

        if is_bf16:
            # Distribution checks instead of element-wise assert_close: cross-
            # kernel bf16 produces a long max tail (fused vs unfused linears),
            # but a real op bug pushes mean *and* median high. Mean catches
            # systematic drift; median catches "wrong weights / swapped
            # layers" where most elements diverge. Pre-norm tensors get a
            # looser bound because the final RMSNorm compresses drift ~5x.
            assert last_diff.mean().item() < _BF16_MEAN_DRIFT_POST_NORM, (
                f"batch{i} last_hidden_state bf16 mean drift "
                f"{last_diff.mean():.4f} >= {_BF16_MEAN_DRIFT_POST_NORM}"
            )
            assert last_diff.median().item() < _BF16_MEDIAN_DRIFT_POST_NORM, (
                f"batch{i} last_hidden_state bf16 median drift "
                f"{last_diff.median():.4f} >= {_BF16_MEDIAN_DRIFT_POST_NORM}"
            )
            assert hs_m2_diff.mean().item() < _BF16_MEAN_DRIFT_PRE_NORM, (
                f"batch{i} hidden_states[-2] bf16 mean drift "
                f"{hs_m2_diff.mean():.4f} >= {_BF16_MEAN_DRIFT_PRE_NORM}"
            )
            assert hs_m2_diff.median().item() < _BF16_MEDIAN_DRIFT_PRE_NORM, (
                f"batch{i} hidden_states[-2] bf16 median drift "
                f"{hs_m2_diff.median():.4f} >= {_BF16_MEDIAN_DRIFT_PRE_NORM}"
            )
        else:
            assert_close(ref_last_v, fv_last_v, atol=_FP32_ATOL, rtol=_FP32_RTOL)
            assert_close(ref_hs_m2_v, fv_hs_m2_v, atol=_FP32_ATOL, rtol=_FP32_RTOL)


@pytest.mark.skipif(not ZIMAGE_TEXT_ENCODER_DIR.exists(), reason="Z-Image text encoder checkpoint required")
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="bf16 per-layer diagnostic requires a bf16-capable CUDA device",
)
def test_zimage_qwen3_encoder_per_layer_bf16_diagnostic():
    """Diagnostic-only: prints per-layer FastVideo-vs-HF drift for the bf16
    forward to distinguish monotonic accumulation (bf16 tail) from a
    single-layer spike (real op mismatch).

    This test always passes; review the captured stdout to interpret.
    Expected pattern for healthy bf16 accumulation: max/mean/median grow
    smoothly with layer depth. Suspect pattern: one layer's diff is
    >>2x the previous, suggesting a real bug at that block.
    """
    if not ZIMAGE_TOKENIZER_DIR.exists():
        pytest.skip(f"Z-Image tokenizer dir not found: {ZIMAGE_TOKENIZER_DIR}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(11)

    tokenizer = AutoTokenizer.from_pretrained(str(ZIMAGE_TOKENIZER_DIR), local_files_only=True)
    toks = _tokenize_official_prompts(
        tokenizer,
        ["A cinematic shot of a rainy neon street at night."],
    )
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    ref = AutoModel.from_pretrained(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device=device)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Cache the reference hidden states on CPU, then free the HF model so
        # only one encoder is resident when FastVideo runs (avoids OOM).
        ref_hidden_cpu = [h.detach().float().cpu() for h in ref_out.hidden_states]
    del ref, ref_out
    _reclaim_vram()

    fv_cls, _ = ModelRegistry.resolve_model_cls("Qwen3Model")
    cfg_raw = _load_json(ZIMAGE_TEXT_ENCODER_DIR / "config.json")
    for k in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(k, None)
    cfg = Qwen3TextConfig()
    cfg.update_model_arch(cfg_raw)
    fv = fv_cls(cfg).eval()
    fv.load_weights(_iter_pretrained_safetensors(ZIMAGE_TEXT_ENCODER_DIR))
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        fv_out = fv(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        fv_hidden_cpu = [h.detach().float().cpu() for h in fv_out.hidden_states]

    assert fv_out.hidden_states is not None
    assert len(ref_hidden_cpu) == 37, (
        f"expected embeddings plus 36 block outputs, got {len(ref_hidden_cpu)} entries"
    )
    assert len(ref_hidden_cpu) == len(fv_hidden_cpu), (
        f"len mismatch: HF={len(ref_hidden_cpu)} FV={len(fv_hidden_cpu)}"
    )

    mask_cpu = attention_mask.detach().bool().cpu()[0]
    print(
        f"\n[per-layer bf16 diagnostic] num_hidden_states={len(ref_hidden_cpu)}  "
        f"valid_tokens={mask_cpu.sum().item()}/{mask_cpu.numel()}",
        flush=True,
    )
    print(
        f"{'idx':>3}  {'kind':<10}  {'max':>8}  {'mean':>8}  {'median':>8}  {'p99':>8}",
        flush=True,
    )
    for idx, (ref_h, fv_h) in enumerate(zip(ref_hidden_cpu, fv_hidden_cpu)):
        ref_v = ref_h[0][mask_cpu]
        fv_v = fv_h[0][mask_cpu]
        diff = (ref_v - fv_v).abs()
        flat = diff.flatten()
        p99 = torch.quantile(flat, 0.99).item()
        if idx == 0:
            kind = "embedding"
        elif idx == len(ref_hidden_cpu) - 1:
            kind = "post-norm"
        else:
            kind = f"layer{idx - 1}-out"
        print(
            f"{idx:>3}  {kind:<10}  "
            f"{diff.max().item():>8.4f}  {diff.mean().item():>8.4f}  "
            f"{diff.median().item():>8.4f}  {p99:>8.4f}",
            flush=True,
        )
