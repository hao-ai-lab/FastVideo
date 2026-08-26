# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Sol-Engine-aligned MiniMax-H3 optimizations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


class _TimeProjection(nn.Module):

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.stack((timesteps, timesteps.square()), dim=-1)


class _TimeEmbedder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc_in.weight.copy_(torch.eye(2))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc_in(embeddings)


class _Projection(nn.Module):

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 18)
        with torch.no_grad():
            values = torch.arange(36, dtype=torch.float32).reshape(18, 2)
            self.linear.weight.copy_(values / 37 + offset)
            self.linear.bias.copy_(torch.arange(18, dtype=torch.float32) / 19 + offset)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows = self.linear(embeddings).view(-1, 6)
        return rows.chunk(6, dim=-1)


class _ProjectionBlock(nn.Module):

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.adaln_proj = _Projection(offset)


class _TrajectoryTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.time_proj = _TimeProjection()
        self.time_embedder = _TimeEmbedder()
        self.transformer_blocks = nn.ModuleList([_ProjectionBlock(0.0), _ProjectionBlock(0.25)])
        self.adaln_rank = None


class _IdentityAttentionImpl(nn.Module):

    def preprocess_qkv(self, qkv: torch.Tensor, metadata: object) -> torch.Tensor:
        del metadata
        return qkv

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: object) -> torch.Tensor:
        del k, v, metadata
        return q

    def postprocess_output(self, output: torch.Tensor, metadata: object) -> torch.Tensor:
        del metadata
        return output


def test_minimax_h3_relayout_is_bit_exact_and_compile_safe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")

    from fastvideo.models.dits.minimax_h3_fusions.relayout import (
        merge_heads,
        pack_qkv_destination_major,
    )

    rows, heads, head_dim, world = 17, 8, 16, 4
    fused = torch.randn(rows, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, v = fused[:, 0], fused[:, 1], fused[:, 2]
    heads_local = heads // world

    expected_pack = torch.empty(
        world,
        rows,
        heads_local,
        3 * head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    for index, tensor in enumerate((q, k, v)):
        shard = tensor.reshape(rows, world, heads_local, head_dim).permute(1, 0, 2, 3)
        expected_pack[..., index * head_dim:(index + 1) * head_dim].copy_(shard)

    compiled_pack = torch.compile(pack_qkv_destination_major, fullgraph=True, dynamic=False)
    actual_pack = compiled_pack(q, k, v, world)
    assert torch.equal(actual_pack, expected_pack)

    packed_output = torch.randn_like(expected_pack[..., :head_dim])
    expected_merge = packed_output.permute(1, 0, 2, 3).contiguous().reshape(rows, heads, head_dim)
    compiled_merge = torch.compile(merge_heads, fullgraph=True, dynamic=False)
    actual_merge = compiled_merge(packed_output)
    assert torch.equal(actual_merge, expected_merge)


def test_adaln_precompute_rejects_a_different_trajectory() -> None:
    from fastvideo.models.dits.minimax_h3 import (
        MiniMaxH3Transformer3DModel,
        _MiniMaxH3StepCursor,
    )

    cursor = _MiniMaxH3StepCursor(torch.device("cpu"), ((1.0,),))
    transformer = SimpleNamespace(
        _h3_adaln_cursor=cursor,
        transformer_blocks=[object()],
    )
    same_plan = [(torch.tensor([1.0]), torch.tensor([0]))]
    stats = MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, same_plan)
    assert stats["installed"] == 0

    different_plan = [(torch.tensor([0.5]), torch.tensor([0]))]
    with pytest.raises(RuntimeError, match="same denoising schedule"):
        MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, different_plan)


def test_adaln_precompute_matches_projection_tables_and_reuses_schedule() -> None:
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel

    transformer = _TrajectoryTransformer()
    plan = [
        (torch.tensor([1.0, 0.5]), torch.tensor([0, 1])),
        (torch.tensor([0.25]), torch.tensor([0])),
    ]
    original_projections = [block.adaln_proj for block in transformer.transformer_blocks]
    embeddings = [transformer.time_embedder(transformer.time_proj(timestep)) for timestep, _ in plan]
    expected = []
    for projection in original_projections:
        per_step = [torch.cat(projection(embedding), dim=-1) for embedding in embeddings]
        per_step[1] = torch.cat((per_step[1], torch.zeros_like(per_step[1])))
        expected.append(per_step)

    stats = MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, plan)
    assert stats["installed"] == 1
    assert stats["steps"] == 2
    assert stats["blocks"] == 2

    for step in range(2):
        MiniMaxH3Transformer3DModel.set_adaln_step(transformer, step)
        for block, expected_steps in zip(transformer.transformer_blocks, expected, strict=True):
            actual = torch.cat(block.adaln_proj(torch.empty(0)), dim=-1)
            assert torch.equal(actual, expected_steps[step])

    reused = MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, plan)
    assert reused["installed"] == 0
    with pytest.raises(IndexError, match="trajectory step"):
        MiniMaxH3Transformer3DModel.set_adaln_step(transformer, 2)


def test_adaln_precompute_failure_does_not_partially_replace_blocks() -> None:
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel

    transformer = _TrajectoryTransformer()
    originals = [block.adaln_proj for block in transformer.transformer_blocks]

    def fail_projection(embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        del embeddings
        raise RuntimeError("injected projection failure")

    transformer.transformer_blocks[1].adaln_proj.forward = fail_projection
    plan = [(torch.tensor([1.0]), torch.tensor([0]))]
    with pytest.raises(RuntimeError, match="injected projection failure"):
        MiniMaxH3Transformer3DModel.prepare_adaln_trajectory(transformer, plan)
    assert [block.adaln_proj for block in transformer.transformer_blocks] == originals
    assert not hasattr(transformer, "_h3_adaln_cursor")


@pytest.mark.parametrize(
    ("layerwise", "fsdp", "message"),
    [
        (True, False, "dit_layerwise_offload=True"),
        (False, True, "FSDP inference"),
    ],
)
def test_adaln_precompute_rejects_incompatible_loader_lifecycles(layerwise: bool, fsdp: bool,
                                                                  message: str) -> None:
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
        _validate_adaln_precompute_configuration,
    )

    args = SimpleNamespace(dit_layerwise_offload=layerwise, use_fsdp_inference=fsdp)
    with pytest.raises(RuntimeError, match=message):
        _validate_adaln_precompute_configuration(args)


def test_adaln_precompute_accepts_replicated_materialized_weights() -> None:
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
        _validate_adaln_precompute_configuration,
    )

    _validate_adaln_precompute_configuration(
        SimpleNamespace(dit_layerwise_offload=False, use_fsdp_inference=False))


def test_packed_sp_is_inert_on_one_rank() -> None:
    from fastvideo.models.dits.minimax_h3 import _packed_sp_active

    assert not _packed_sp_active(False, 1)
    assert not _packed_sp_active(True, 1)
    assert _packed_sp_active(True, 4)
    with pytest.raises(ValueError, match="must be positive"):
        _packed_sp_active(True, 0)


def test_packed_sp_falls_back_to_autograd_aware_collective_when_grad_enabled() -> None:
    from fastvideo.attention.layer import DistributedAttention
    from fastvideo.forward_context import set_forward_context

    attention = DistributedAttention.__new__(DistributedAttention)
    nn.Module.__init__(attention)
    attention.attn_impl = _IdentityAttentionImpl()
    attention.head_size = 2
    attention.packed_qkv_relayout = True
    attention._compile_forward_enabled = True

    q = torch.randn(1, 5, 4, 2, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    with (
        patch("fastvideo.attention.layer.get_sp_world_size", return_value=2),
        patch("fastvideo.attention.layer.get_sp_parallel_rank", return_value=0),
        patch("fastvideo.attention.layer.sequence_model_parallel_all_to_all_4D", side_effect=lambda tensor, **_: tensor)
        as generic,
        patch("fastvideo.attention.layer.sequence_model_parallel_direct_all_to_all",
              side_effect=AssertionError("packed collective must not run with gradients")) as direct,
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        output, replicated = attention(q, k, v)
    output.sum().backward()
    assert replicated is None
    assert torch.equal(q.grad, torch.ones_like(q))
    assert generic.call_count == 2
    direct.assert_not_called()


def test_direct_packed_collective_rejects_unsupported_inputs_before_launch() -> None:
    from fastvideo.distributed.communication_op import sequence_model_parallel_direct_all_to_all

    group = SimpleNamespace(world_size=4)
    with patch("fastvideo.distributed.communication_op.get_sp_group", return_value=group):
        with pytest.raises(RuntimeError, match="inference-only"):
            sequence_model_parallel_direct_all_to_all(torch.randn(4, 2, requires_grad=True))
        with torch.no_grad(), pytest.raises(ValueError, match="leading dimension"):
            sequence_model_parallel_direct_all_to_all(torch.randn(5, 2))
        with torch.no_grad(), pytest.raises(ValueError, match="contiguous"):
            sequence_model_parallel_direct_all_to_all(torch.randn(2, 4).transpose(0, 1))


def test_direct_packed_collective_captures_as_a_fullgraph_custom_op(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.distributed.is_initialized():
        pytest.skip("test owns a temporary world-1 process group")

    from fastvideo.distributed import parallel_state

    class Group:
        unique_name = "minimax_h3_direct_compile_test"
        device_group = None

    group = Group()
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{tmp_path / 'store'}",
        rank=0,
        world_size=1,
    )
    group.device_group = torch.distributed.group.WORLD
    parallel_state._register_group(group)
    try:
        def direct_collective(tensor: torch.Tensor) -> torch.Tensor:
            return torch.ops.fastvideo.direct_all_to_all_single(tensor, group.unique_name)

        tensor = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            compiled = torch.compile(direct_collective, fullgraph=True, dynamic=False)
            output = compiled(tensor)
        assert torch.equal(output, tensor)
    finally:
        torch.distributed.destroy_process_group()


def test_minimax_h3_fusions_capture_in_fullgraph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")

    from fastvideo.models.dits.minimax_h3_fusions import (
        fused_qknorm_rope,
        fused_residual_gate_rmsnorm_modulate,
        fused_rmsnorm_modulate,
        minimax_h3_swiglu,
    )

    device, dtype = torch.device("cuda"), torch.bfloat16
    batch, rows, hidden = 1, 12, 128
    x = torch.randn(batch, rows, hidden, device=device, dtype=dtype)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    table = torch.randn(18, hidden, device=device, dtype=dtype)
    index = torch.arange(rows, device=device).remainder(table.shape[0])
    q = torch.randn(batch, rows, 1, hidden, device=device, dtype=dtype)
    cos = torch.randn(rows, 96, device=device, dtype=dtype)
    sin = torch.randn_like(cos)

    cases = (
        (fused_rmsnorm_modulate, (x, weight, table, table, index, 1e-5)),
        (fused_residual_gate_rmsnorm_modulate, (x, branch, table, weight, table, table, index, 1e-5)),
        (fused_qknorm_rope, (q, weight, cos, sin, 1e-5)),
        (minimax_h3_swiglu, (torch.randn(batch, rows, 512, device=device, dtype=dtype),)),
    )
    with torch.inference_mode():
        for function, args in cases:
            expected = function(*args)
            actual = torch.compile(function, fullgraph=True, dynamic=False)(*args)
            expected_items = expected if isinstance(expected, tuple) else (expected,)
            actual_items = actual if isinstance(actual, tuple) else (actual,)
            for expected_item, actual_item in zip(expected_items, actual_items, strict=True):
                torch.testing.assert_close(actual_item, expected_item, atol=0, rtol=0)
