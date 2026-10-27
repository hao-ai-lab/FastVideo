# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for modular training performance accounting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.utils.performance import (
    TrainingPerformanceMonitor,
    _blockwise_causal_frame_pairs,
    _teacher_forcing_frame_pairs,
    estimate_transformer_forward,
    infer_peak_bf16_tflops,
)


def _arch() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        ffn_dim=16,
        num_layers=2,
        patch_size=(1, 2, 2),
    )


class _FakeTransformer(torch.nn.Module):

    def __init__(self, *, causal: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(arch_config=_arch())
        self.weight = torch.nn.Parameter(torch.ones(()))
        if causal:
            self.num_frame_per_block = 2
            self.local_attn_size = 3
            self.text_len = 4

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del encoder_hidden_states, kwargs
        return hidden_states * self.weight


class _UnsupportedTransformer(torch.nn.Module):

    def forward(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def _kwargs(*, frames: int = 4) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.ones(2, 3, frames, 4, 4),
        "encoder_hidden_states": torch.ones(2, 3, 8),
    }


def test_dense_forward_flops_and_backward_factor() -> None:
    module = _FakeTransformer()
    output = torch.ones(2, 3, 4, 4, 4, requires_grad=True)

    work = estimate_transformer_forward(
        module,
        _kwargs(),
        output,
        role="student",
        attention_metadata=None,
    )

    assert work is not None
    # B=2, L=16, C=3, d=8, ffn=16, layers=2.
    expected_per_layer = (
        8 * 2 * 16 * 8**2
        + 4 * 2 * 16 * 8 * 16
        + 4 * 2 * 16 * 8**2
        + 4 * 2 * 3 * 8**2
        + 4 * 2 * 16 * 3 * 8
        + 4 * 2 * 16**2 * 8
    )
    assert work.forward_flops == expected_per_layer * 2
    assert work.useful_flops == work.forward_flops * 3
    assert work.query_frames == 8
    assert work.query_tokens == 32
    assert work.attention_pairs == 2 * 16**2


def test_causal_chunk_and_teacher_forcing_pair_counts() -> None:
    assert _blockwise_causal_frame_pairs(5, 2, -1) == 17
    assert _blockwise_causal_frame_pairs(5, 2, 3) == 13
    assert _teacher_forcing_frame_pairs(4, 2) == 24


def test_streaming_causal_forward_uses_cache_window() -> None:
    module = _FakeTransformer(causal=True)
    kwargs = _kwargs(frames=2)
    # Four spatial tokens/frame, starting after two frames. The three-frame
    # local window gives 8 query tokens x 12 key tokens.
    kwargs.update({"kv_cache": [{}], "current_start": 8})

    work = estimate_transformer_forward(
        module,
        kwargs,
        torch.ones(2, 3, 2, 4, 4),
        role="student",
        attention_metadata=None,
    )

    assert work is not None
    assert work.causal_chunks == 1
    assert work.attention_pairs == 2 * 8 * 12
    assert work.dense_attention_pairs == 2 * 8 * 16


def test_vsa_uses_kernel_topk_density_and_overhead() -> None:
    module = _FakeTransformer()
    metadata = SimpleNamespace(
        variable_block_sizes=torch.tensor([4, 4, 4, 4]),
        VSA_sparsity=0.75,
    )

    work = estimate_transformer_forward(
        module,
        _kwargs(),
        torch.ones(2, 3, 4, 4, 4),
        role="student",
        attention_metadata=metadata,
    )

    assert work is not None
    assert work.attention_pairs == pytest.approx(2 * 16**2 * 0.25)
    dense_work = estimate_transformer_forward(
        module,
        _kwargs(),
        torch.ones(2, 3, 4, 4, 4),
        role="student",
        attention_metadata=None,
    )
    assert dense_work is not None
    # VSA includes its gate projection and pooled dense attention overhead.
    assert work.forward_flops > dense_work.forward_flops - (
        dense_work.attention_pairs - work.attention_pairs) * 4 * 8 * 2


def test_streaming_cross_attention_cache_skips_kv_projections() -> None:
    module = _FakeTransformer(causal=True)
    kwargs = _kwargs(frames=2)
    kwargs.update({"kv_cache": [{}], "current_start": 8})
    output = torch.ones(2, 3, 2, 4, 4)

    uncached = estimate_transformer_forward(
        module,
        kwargs,
        output,
        role="student",
        attention_metadata=None,
    )
    cached = estimate_transformer_forward(
        module,
        kwargs,
        output,
        role="student",
        attention_metadata=None,
        cross_attention_cached=True,
    )

    assert uncached is not None and cached is not None
    expected_saved = 4 * 2 * 4 * 8**2 * 2
    assert uncached.forward_flops - cached.forward_flops == expected_saved


def test_monitor_counts_multiple_rollouts_and_roles_without_double_counting_accumulation() -> None:
    models = {
        role: SimpleNamespace(transformer=_FakeTransformer())
        for role in ("student", "teacher", "critic")
    }
    monitor = TrainingPerformanceMonitor()
    monitor.attach(models)
    monitor.reset()

    with set_forward_context(current_timestep=0, attn_metadata=None):
        models["student"].transformer(**_kwargs())
        models["student"].transformer(**_kwargs())
        models["critic"].transformer(**_kwargs())
        with torch.no_grad():
            models["teacher"].transformer(**_kwargs())
            models["teacher"].transformer(**_kwargs())

    metrics = monitor.metrics(
        step_time_sec=2.0,
        local_batch_size=2,
        grad_accum=4,
        world_size=8,
        sp_size=4,
        peak_tflops_per_gpu=100.0,
    )
    monitor.close()

    assert metrics["perf/model_forward_calls"] == 5
    assert metrics["perf/causal_chunks"] == 0
    assert metrics["perf/role/student/forward_calls"] == 2
    assert metrics["perf/role/teacher/forward_calls"] == 2
    assert metrics["perf/role/teacher/backward_forwards"] == 0
    assert metrics["perf/role/critic/backward_forwards"] == 1
    # 2 local samples x DP=2 x grad-accum=4 / 2 seconds.
    assert metrics["perf/samples_per_sec"] == 8
    assert metrics["perf/estimated_mfu"] > 0


def test_monitor_counts_forward_when_flop_estimator_does_not_support_model() -> None:
    monitor = TrainingPerformanceMonitor()
    model = SimpleNamespace(transformer=_UnsupportedTransformer())
    monitor.attach({"student": model})
    monitor.reset()

    model.transformer(hidden_states=torch.ones(1))
    metrics = monitor.metrics(
        step_time_sec=1.0,
        local_batch_size=1,
        grad_accum=1,
        world_size=1,
        sp_size=1,
        peak_tflops_per_gpu=None,
    )
    monitor.close()

    assert metrics["perf/model_forward_calls"] == 1
    assert metrics["perf/role/student/forward_calls"] == 1
    assert "perf/estimated_tflops_per_gpu" not in metrics


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA A40", 149.7),
        ("NVIDIA L40S", 362.05),
        ("NVIDIA H100 80GB HBM3", 989.5),
        ("NVIDIA H200", 989.5),
        ("NVIDIA B200", 2250.0),
        ("NVIDIA GB200", 2500.0),
        ("Some Future GPU", None),
    ],
)
def test_infer_peak_bf16_tflops(device_name: str, expected: float | None) -> None:
    assert infer_peak_bf16_tflops(device_name) == expected
