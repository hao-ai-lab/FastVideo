# SPDX-License-Identifier: Apache-2.0
"""
Parity test for Z-Image Qwen3 text encoder support in FastVideo.

This compares:
1) direct transformers AutoModel output, and
2) FastVideo's shared Qwen3 encoder (``Qwen3ForCausalLM``, added for Flux2
   Klein and reused here — Z-Image-Turbo's ``Qwen3Model`` checkpoint routes
   to it via the model registry),

using identical local checkpoint, tokenization, and inputs.

Usage:
    pytest tests/local_tests/zimage/test_zimage_encoder_parity.py -v -s
"""

from __future__ import annotations

from pathlib import Path
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
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import safe_open

from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.registry import ModelRegistry

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


# bf16 tolerance reflects accumulated drift across the Qwen3 encoder
# (Z-Image-Turbo ships a 35-layer Qwen3 variant). Per
# add-model-02-parity's calibration block, bf16 max-based asserts are
# meaningless for deep encoders — fused-vs-unfused linear order alone
# produces a long tail. We use distribution checks (mean + median)
# instead. The per-layer diagnostic test in this file confirmed the
# growth is monotonic with layer depth (no single-layer spike), and
# fp32 forwards are bit-exact, so this is the textbook bf16-tail
# signature and not a code bug.
#
# Empirical numbers (Z-Image-Turbo Qwen3, 35 layers, observed on CUDA):
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
    toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    ref = AutoModel.from_pretrained(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        ref_last = ref_out.last_hidden_state.detach().float().cpu()
        ref_hs_m2 = ref_out.hidden_states[-2].detach().float().cpu()

    print(f"[mem] after HF ref forward: {_gpu_mem_mb():.0f} MiB", flush=True)
    # Free the HF reference before loading the FastVideo model — its outputs
    # are already on CPU, so keeping it resident just doubles peak VRAM (two
    # full encoder copies OOMs a 44 GB L40S in fp32).
    del ref, ref_out
    _reclaim_vram()
    print(f"[mem] after freeing HF ref:  {_gpu_mem_mb():.0f} MiB", flush=True)

    fv_cls, _ = ModelRegistry.resolve_model_cls("Qwen3Model")
    cfg_raw = _load_json(ZIMAGE_TEXT_ENCODER_DIR / "config.json")
    for k in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(k, None)

    cfg = Qwen3TextConfig()
    cfg.update_model_arch(cfg_raw)
    fv = fv_cls(cfg).eval()
    loaded = fv.load_weights(_iter_pretrained_safetensors(ZIMAGE_TEXT_ENCODER_DIR))
    assert loaded, "No Qwen3 weights were loaded into FastVideo model"
    fv = fv.to(device=device, dtype=dtype)
    print(f"[mem] after FastVideo load:  {_gpu_mem_mb():.0f} MiB", flush=True)

    # Strict-load contract: assert the checkpoint surface matches the model's
    # params modulo the allowlist. Non-encoder heads like `lm_head.weight`
    # (the body-only encoder has no lm_head) are the only acceptable unmatched
    # keys; anything else means a real silent drop.
    checkpoint_keys = {name for name, _ in _iter_pretrained_safetensors(ZIMAGE_TEXT_ENCODER_DIR)}
    checkpoint_keys = {k[len("model."):] if k.startswith("model.") else k for k in checkpoint_keys}
    param_names = {name for name, _ in fv.named_parameters()}
    # Map stacked-param checkpoint keys to their merged FastVideo names so the
    # diff below only surfaces *unexpected* keys, not the qkv/gate_up fusion.
    stacked = fv.config.arch_config.stacked_params_mapping
    mapped_keys: set[str] = set()
    for ckpt_name in checkpoint_keys:
        for param_name, weight_name, _ in stacked:
            if weight_name in ckpt_name:
                mapped_keys.add(ckpt_name.replace(weight_name, param_name))
                break
        else:
            mapped_keys.add(ckpt_name)
    unexpected = mapped_keys - param_names - {"rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached"}
    assert unexpected <= _ALLOWED_UNEXPECTED_KEYS, (
        f"Unexpected checkpoint keys not in allowlist: "
        f"{sorted(unexpected - _ALLOWED_UNEXPECTED_KEYS)}"
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
    toks = tokenizer(
        ["A cinematic shot of a rainy neon street at night."],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    ref = AutoModel.from_pretrained(
        str(ZIMAGE_TEXT_ENCODER_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device=device, dtype=dtype)

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
