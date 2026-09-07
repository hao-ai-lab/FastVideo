# SPDX-License-Identifier: Apache-2.0
"""Low-overhead performance accounting for modular training.

The monitor counts logical Transformer invocations at the role boundary. This
is important for distillation: one optimizer step may contain several student
rollouts plus critic and teacher forwards. FLOPs are an analytic estimate of
the Transformer core; measured step and sample throughput remain exact wall
clock metrics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.forward_context import get_forward_context

_FLOPS_PER_TFLOP = 1.0e12


@dataclass(slots=True)
class _ForwardWork:
    role: str
    forward_flops: float
    useful_flops: float
    causal_chunks: int
    query_frames: int
    query_tokens: int
    attention_pairs: float
    dense_attention_pairs: float
    backward_expected: bool


def infer_peak_bf16_tflops(device_name: str) -> float | None:
    """Infer dense BF16 tensor-core peak TFLOP/s from a CUDA device name.

    Values intentionally exclude structured sparsity. Explicit configuration
    should be used when board form factor or clocks differ from the common
    data-center variant.
    """
    name = device_name.upper()
    # Order matters: GB200 contains B200.
    peaks = (
        ("GB200", 2500.0),
        ("B200", 2250.0),
        ("H200", 989.5),
        ("H100", 989.5),
        ("L40S", 362.05),
        ("A100", 312.0),
        ("A40", 149.7),
    )
    return next((peak for token, peak in peaks if token in name), None)


def _arch_config(module: torch.nn.Module) -> Any | None:
    config = getattr(module, "config", None)
    return getattr(config, "arch_config", config)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list | tuple):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _output_requires_grad(output: Any) -> bool:
    if isinstance(output, torch.Tensor):
        return bool(output.requires_grad)
    if isinstance(output, dict):
        return any(_output_requires_grad(value) for value in output.values())
    if isinstance(output, list | tuple):
        return any(_output_requires_grad(value) for value in output)
    return False


def _blockwise_causal_frame_pairs(
    num_frames: int,
    frames_per_block: int,
    local_attn_size: int,
) -> int:
    """Frame-level pairs used by full-sequence block-causal training."""
    total = 0
    block = max(1, frames_per_block)
    for start in range(0, num_frames, block):
        end = min(start + block, num_frames)
        attended_start = 0 if local_attn_size < 0 else max(0, end - local_attn_size)
        total += (end - start) * (end - attended_start)
    return total


def _teacher_forcing_frame_pairs(
    num_frames: int,
    frames_per_block: int,
) -> int:
    """Pairs for causal Wan's concatenated ``[clean | noisy]`` mask."""
    total = 0
    block = max(1, frames_per_block)
    for start in range(0, num_frames, block):
        end = min(start + block, num_frames)
        # Clean and noisy query blocks each attend ``end`` frames.
        total += 2 * (end - start) * end
    return total


def _vsa_attention_pairs(
    seq_len: int,
    metadata: Any,
) -> tuple[float, int]:
    """Return nominal sparse token pairs and tile count.

    Selected key tiles are data-dependent. Their mean valid size gives the
    expected token-pair count while retaining the kernel's ceil-and-clamp top-k
    behavior.
    """
    block_sizes = getattr(metadata, "variable_block_sizes", None)
    if block_sizes is None:
        return float(seq_len * seq_len), 0
    num_blocks = int(block_sizes.numel())
    if num_blocks <= 0:
        return float(seq_len * seq_len), 0
    sparsity = float(getattr(metadata, "VSA_sparsity", 0.0))
    topk = max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))
    return float(seq_len * seq_len) * topk / num_blocks, num_blocks


def estimate_transformer_forward(
    module: torch.nn.Module,
    kwargs: dict[str, Any],
    output: Any,
    *,
    role: str,
    attention_metadata: Any | None,
    cross_attention_cached: bool = False,
) -> _ForwardWork | None:
    """Estimate one Wan-style Transformer invocation from its logical input."""
    hidden_states = _first_tensor(kwargs.get("hidden_states"))
    if hidden_states is None or hidden_states.ndim != 5:
        return None

    arch = _arch_config(module)
    if arch is None:
        return None
    required = ("hidden_size", "ffn_dim", "num_layers", "patch_size")
    if any(not hasattr(arch, name) for name in required):
        return None

    batch_size = int(hidden_states.shape[0])
    raw_frames = int(hidden_states.shape[2])
    raw_height = int(hidden_states.shape[3])
    raw_width = int(hidden_states.shape[4])
    patch_size = arch.patch_size
    patch: tuple[int,
                 ...] = ((1, patch_size,
                          patch_size) if isinstance(patch_size, int) else tuple(int(value) for value in patch_size))
    if len(patch) != 3 or any(value <= 0 for value in patch):
        return None

    frames = raw_frames // patch[0]
    spatial_tokens = (raw_height // patch[1]) * (raw_width // patch[2])
    seq_len = frames * spatial_tokens
    if seq_len <= 0:
        return None

    hidden_size = int(arch.hidden_size)
    ffn_dim = int(arch.ffn_dim)
    num_layers = int(arch.num_layers)

    context = _first_tensor(kwargs.get("encoder_hidden_states"))
    context_tokens = int(context.shape[-2]) if context is not None and context.ndim >= 2 else 0
    image_context = _first_tensor(kwargs.get("encoder_hidden_states_image"))
    if image_context is not None and image_context.ndim >= 2:
        context_tokens += int(image_context.shape[-2])

    is_causal = hasattr(module, "num_frame_per_block")
    teacher_forcing = is_causal and _first_tensor(kwargs.get("clean_x")) is not None
    query_tokens = seq_len
    dense_pairs = float(seq_len * seq_len)
    vsa_tiles = 0
    causal_chunks = 0

    if is_causal:
        frames_per_block = int(getattr(module, "num_frame_per_block", 1))
        causal_chunks = math.ceil(frames / max(1, frames_per_block))
        local_attn_size = int(getattr(module, "local_attn_size", -1))
        kv_cache = kwargs.get("kv_cache")
        if kv_cache is not None:
            current_start = int(kwargs.get("current_start", 0))
            max_frames = local_attn_size if local_attn_size >= 0 else 21
            key_tokens = min(current_start + seq_len, max_frames * spatial_tokens)
            attention_pairs = float(seq_len * key_tokens)
            dense_pairs = float(seq_len * (current_start + seq_len))
        elif teacher_forcing:
            query_tokens = 2 * seq_len
            attention_pairs = float(_teacher_forcing_frame_pairs(frames, frames_per_block) * spatial_tokens**2)
            dense_pairs = float(query_tokens * query_tokens)
        else:
            attention_pairs = float(
                _blockwise_causal_frame_pairs(frames, frames_per_block, local_attn_size) * spatial_tokens**2)
    else:
        attention_pairs, vsa_tiles = _vsa_attention_pairs(seq_len, attention_metadata)

    # Causal Wan pads text to ``text_len`` before every cross-attention block.
    if is_causal:
        context_tokens = max(context_tokens, int(getattr(module, "text_len", 0)))

    b = batch_size
    seq_length = query_tokens
    dim = hidden_size
    # Four self-attention projections, two MLP projections, cross-attention
    # Q/K/V/out projections, and the two attention matmuls.
    per_layer = (8.0 * b * seq_length * dim * dim + 4.0 * b * seq_length * dim * ffn_dim +
                 4.0 * b * seq_length * dim * dim +
                 (0.0 if cross_attention_cached else 4.0 * b * context_tokens * dim * dim) +
                 4.0 * b * seq_length * context_tokens * dim + 4.0 * b * attention_pairs * dim)

    if vsa_tiles:
        # Wan VSA adds one gate projection and dense pooled QK/AV matmuls.
        per_layer += 2.0 * b * seq_length * dim * dim
        per_layer += 4.0 * b * vsa_tiles * vsa_tiles * dim

    forward_flops = per_layer * num_layers
    backward_expected = _output_requires_grad(output)
    # Standard MFU counts model-algorithm FLOPs, excluding activation
    # checkpoint recompute: training is approximately F + 2B = 3F.
    useful_flops = forward_flops * (3.0 if backward_expected else 1.0)
    return _ForwardWork(
        role=role,
        forward_flops=forward_flops,
        useful_flops=useful_flops,
        causal_chunks=causal_chunks,
        query_frames=b * (2 * frames if teacher_forcing else frames),
        query_tokens=b * query_tokens,
        attention_pairs=b * attention_pairs,
        dense_attention_pairs=b * dense_pairs,
        backward_expected=backward_expected,
    )


class TrainingPerformanceMonitor:
    """Count per-role Transformer work for one optimizer step at a time."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._work: list[_ForwardWork] = []
        self._forward_calls: dict[str, int] = defaultdict(int)
        self._backward_forwards: dict[str, int] = defaultdict(int)

    def attach(self, role_models: dict[str, Any]) -> None:
        self.close()
        for role, model in role_models.items():
            transformer = getattr(model, "transformer", None)
            if not isinstance(transformer, torch.nn.Module):
                continue

            cached_cross_attention: list[bool] = []

            def pre_hook(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                cache_state: list[bool] = cached_cross_attention,
            ) -> None:
                del module, args
                cache = kwargs.get("crossattn_cache")
                initialized = (isinstance(cache, list) and bool(cache)
                               and all(isinstance(item, dict) and bool(item.get("is_init", False)) for item in cache))
                cache_state.append(initialized)

            def hook(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                role_name: str = role,
                cache_state: list[bool] = cached_cross_attention,
            ) -> None:
                del args
                self._forward_calls[role_name] += 1
                if _output_requires_grad(output):
                    self._backward_forwards[role_name] += 1
                try:
                    metadata = get_forward_context().attn_metadata
                except AssertionError:
                    metadata = None
                work = estimate_transformer_forward(
                    module,
                    kwargs,
                    output,
                    role=role_name,
                    attention_metadata=metadata,
                    cross_attention_cached=(cache_state.pop() if cache_state else False),
                )
                if work is not None:
                    self._work.append(work)

            self._handles.append(transformer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._handles.append(transformer.register_forward_hook(hook, with_kwargs=True))

    def reset(self) -> None:
        self._work.clear()
        self._forward_calls.clear()
        self._backward_forwards.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def metrics(
        self,
        *,
        step_time_sec: float,
        local_batch_size: int,
        grad_accum: int,
        world_size: int,
        sp_size: int,
        peak_tflops_per_gpu: float | None,
    ) -> dict[str, float]:
        if step_time_sec <= 0.0:
            return {}
        world_size = max(1, int(world_size))
        sp_size = max(1, int(sp_size))
        data_parallel_size = max(1, world_size // sp_size)
        accumulation = max(1, int(grad_accum))
        sample_scale_to_world = data_parallel_size * accumulation
        work_scale_to_world = data_parallel_size

        metrics = {
            "perf/steps_per_sec": 1.0 / step_time_sec,
            "perf/samples_per_sec": (max(0, int(local_batch_size)) * sample_scale_to_world / step_time_sec),
            "perf/model_forward_calls": float(sum(self._forward_calls.values())),
        }

        for role, calls in self._forward_calls.items():
            prefix = f"perf/role/{role}"
            metrics[f"{prefix}/forward_calls"] = float(calls)
            metrics[f"{prefix}/backward_forwards"] = float(self._backward_forwards[role])

        if not self._work:
            return metrics

        useful_flops = sum(item.useful_flops for item in self._work) * work_scale_to_world
        query_frames = sum(item.query_frames for item in self._work) * work_scale_to_world
        query_tokens = sum(item.query_tokens for item in self._work) * work_scale_to_world
        attention_pairs = sum(item.attention_pairs for item in self._work)
        dense_pairs = sum(item.dense_attention_pairs for item in self._work)
        achieved_tflops_per_gpu = useful_flops / step_time_sec / world_size / _FLOPS_PER_TFLOP

        metrics.update({
            "perf/causal_chunks": float(sum(item.causal_chunks for item in self._work)),
            "perf/query_latent_frames_per_sec": query_frames / step_time_sec,
            "perf/query_tokens_per_sec": query_tokens / step_time_sec,
            "perf/attention_density": (attention_pairs / dense_pairs if dense_pairs > 0.0 else 1.0),
            "perf/estimated_tflops_per_gpu": achieved_tflops_per_gpu,
        })
        if peak_tflops_per_gpu is not None:
            metrics["perf/peak_tflops_per_gpu"] = peak_tflops_per_gpu
            metrics["perf/estimated_mfu"] = achieved_tflops_per_gpu / peak_tflops_per_gpu

        by_role: dict[str, list[_ForwardWork]] = defaultdict(list)
        for item in self._work:
            by_role[item.role].append(item)
        for role, items in by_role.items():
            prefix = f"perf/role/{role}"
            metrics[f"{prefix}/causal_chunks"] = float(sum(item.causal_chunks for item in items))
            role_flops = sum(item.useful_flops for item in items) * work_scale_to_world
            metrics[f"{prefix}/estimated_tflops_per_gpu"] = (role_flops / step_time_sec / world_size / _FLOPS_PER_TFLOP)
        return metrics
