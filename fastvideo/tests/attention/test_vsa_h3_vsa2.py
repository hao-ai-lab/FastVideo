# SPDX-License-Identifier: Apache-2.0
"""VSA-H3's ``vsa2`` variant: PAN2's joint ``[video; text]`` sequence on the fused VSA2 kernels (fastvideo-kernel/vsa2_train).

The layout tests run anywhere: the backend's VSA2 rows must be exactly the layout VSA2's reference defines (text blocks,
then 4 x 4 x 8 tiles with every token at its own slot), and enumerating the frames last to first must give the time axis
VSA-H3's ``first`` partition. The GPU tests (sm_100a / sm_103a) run the attention layer as PAN2 calls it: at sparsity 0.9
its output is bit for bit the kernels' on rows placed independently, and its forward and backward match the kernels'
fp32 reference under their own selection (overall, on the first frame and on the padded tiles); at sparsity 0 (every
tile pair kept) its forward and backward match an fp32 oracle that knows only the partition into blocks: dense
attention over every token, plus the gate times the attention between the blocks' mean-pooled rows.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import os
from pathlib import Path

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAMetadataBuilder, _axis_tile_sizes,
                                                               _vsa2_segment_geometry)
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.platforms import AttentionBackendEnum

_REFERENCE = Path(__file__).resolve().parents[3] / "fastvideo-kernel" / "vsa2_train" / "reference.py"
_HEADS, _HEAD_DIM = 4, 128
_VSA2 = dict(tile_size=128, video_tile_policy=("first", "last", "last"), variant="vsa2")
_on_vsa2_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="The VSA2 kernels run on sm_100a / sm_103a (B200 / GB200, B300)")


def _reference():
    spec = importlib.util.spec_from_file_location("vsa2_reference", _REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_rows(thw: tuple[int, int, int], text: int, reverse_time: bool) -> torch.Tensor:
    """The tiled row of every packed ``[video; text]`` row, from the reference's slot positions."""
    reference = _reference()
    layout = reference.Layout(text, thw)
    positions, valid = reference.tile_positions(layout)
    t, h, w = thw
    tiled = torch.empty(t * h * w + text, dtype=torch.long)
    for tile, slot in valid.nonzero().tolist():
        pt, ph, pw = positions[tile, slot].tolist()
        frame = t - 1 - pt if reverse_time else pt
        tiled[(frame * h + ph) * w + pw] = (layout.prefix_blocks + tile) * reference.ROWS + slot
    tiled[t * h * w:] = torch.arange(text)
    return tiled


@pytest.mark.parametrize("reverse_time", [False, True])
@pytest.mark.parametrize("thw, text", [((9, 22, 39), 300), ((49, 22, 39), 512), ((31, 6, 17), 128), ((4, 4, 8), 1)])
def test_vsa2_layout_is_the_reference_layout(thw, text, reverse_time) -> None:
    sizes, rows, num_prefix_tiles, num_video_tiles = _vsa2_segment_geometry((thw, text), torch.device("cpu"),
                                                                            reverse_time)
    assert torch.equal(rows, _expected_rows(thw, text, reverse_time))
    reference = _reference()
    layout = reference.Layout(text, thw)
    assert (num_prefix_tiles, num_video_tiles) == (layout.prefix_blocks, layout.video_blocks)
    assert torch.equal(sizes, reference.row_valid(layout).sum(-1))


@pytest.mark.parametrize("frames", [9, 31, 49, 61])
def test_vsa2_reversed_time_is_the_first_partition(frames) -> None:
    # Which frames share a tile row: under reversal, the same groups as VSA-H3's ``first`` policy, the first frames in
    # the short group.
    _, rows, num_prefix_tiles, _ = _vsa2_segment_geometry(((frames, 4, 8), 0), torch.device("cpu"), True)
    tile_of_frame = (rows.reshape(frames, 32)[:, 0] // 128 - num_prefix_tiles).tolist()
    groups = [len(list(group)) for _, group in itertools.groupby(tile_of_frame)]
    assert groups == list(_axis_tile_sizes(frames, 4, "first"))


def test_vsa2_build_rejects_what_its_kernels_cannot_hold() -> None:
    builder = MiniMaxH3VSAMetadataBuilder()
    base = dict(current_timestep=0, patch_size=(1, 2, 2), VSA_sparsity=0.9, packed_segments=((9, 44, 78), 300),
                device=torch.device("cpu"))
    metadata = builder.build(**base, **_VSA2)
    assert metadata.variant == "vsa2" and metadata.vsa2_thw == (9, 22, 39) and metadata.vsa2_text == 300
    assert metadata.video_tile_spans == ((3, 3 + 3 * 6 * 5), )
    for bad in (dict(tile_size=64), dict(video_tile_policy=("first", "center", "center")),
                dict(compress_prefix_keys=False), dict(exempt=False), dict(variant="vsa3")):
        with pytest.raises(ValueError):
            builder.build(**base, **{**_VSA2, **bad})
    with pytest.raises(ValueError):
        builder.build(**{**base, "packed_segments": ((9, 44, 78), (9, 44, 78), 300)}, ref_keep_rate=0.5, **_VSA2)


@pytest.fixture()
def single_process_group(env_overrides):
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    env_overrides.enter_context(envs.override_external("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", os.environ.get("MASTER_PORT") or "29596"))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


# 9 x 22 x 39 video tokens (a partial tile on every axis; 9 = 1 + 4 + 4 frames under the ``first`` time partition) and
# 300 text rows (three blocks, the last one partial), joined as [video; text].
_LATENT, _TEXT = (9, 44, 78), 300
_THW = (9, 22, 39)
_VIDEO = math.prod(_THW)


def _layer(prefix: str = ""):
    from fastvideo.attention.layer import DistributedAttention_VSA

    with envs.FASTVIDEO_ATTENTION_BACKEND.override("VIDEO_SPARSE_ATTN_H3"):
        layer = DistributedAttention_VSA(num_heads=_HEADS, head_size=_HEAD_DIM, causal=False,
                                         supported_attention_backends=(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, ),
                                         prefix=prefix)
    assert layer.backend == AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
    return layer


def _inputs(seed: int) -> list[torch.Tensor]:
    """q, k, v, text q, k, v and a gate on the video rows, [1, L, H, D] bf16."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(length, scale=1.0):
        return (scale * torch.randn(1, length, _HEADS, _HEAD_DIM, device="cuda", generator=generator)).to(torch.bfloat16)

    return [rand(_VIDEO), rand(_VIDEO), rand(_VIDEO), rand(_TEXT), rand(_TEXT), rand(_TEXT), rand(_VIDEO, 0.5)]


def _attend(layer, metadata, q, k, v, text_q, text_k, text_v, gate):
    # The text joins as a sequence-parallel shard, as PAN2's student runs it at SP > 1.
    with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        return layer(q, k, v, _VIDEO, secondary_q=text_q, secondary_k=text_k, secondary_v=text_v,
                     secondary_seq_len=_TEXT, attn_metadata=metadata, gate_compress=gate)


def _build(sparsity: float, policy: tuple[str, str, str], dense_layers: tuple[int, ...] = ()):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, patch_size=(1, 2, 2), VSA_sparsity=sparsity,
                                               packed_segments=(_LATENT, _TEXT), device=torch.device("cuda"),
                                               tile_size=128, video_tile_policy=policy, variant="vsa2",
                                               dense_layers=dense_layers)


@_on_vsa2_gpu
def test_pan2_joint_vsa2_is_the_kernels_on_their_layout(single_process_group) -> None:
    from fastvideo.attention.backends.video_sparse_attn import compute_topk
    from fastvideo.attention.backends.video_sparse_attn_h3 import _vsa2_attention

    metadata = _build(0.9, ("first", "last", "last"))
    inputs = _inputs(0)
    with torch.no_grad():
        video, text = _attend(_layer(), metadata, *inputs)

    # The same rows placed from the reference's layout, heads first, the text's gate zero.
    rows = _expected_rows(_THW, _TEXT, reverse_time=True).cuda()
    seq = metadata.variable_block_sizes.numel() * 128
    q, k, v, text_q, text_k, text_v, gate = inputs
    placed = []
    for video_rows, text_rows in ((q, text_q), (k, text_k), (v, text_v), (gate, torch.zeros_like(text_q))):
        tiled = torch.zeros(1, seq, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        tiled[:, rows] = torch.cat([video_rows, text_rows], dim=1)
        placed.append(tiled.transpose(1, 2).contiguous())
    with torch.no_grad():
        out, _ = _vsa2_attention()(*placed, *_THW, _TEXT, compute_topk(0.9, metadata.num_video_tiles))
    expected = out.transpose(1, 2)[:, rows]
    assert torch.equal(video, expected[:, :_VIDEO]) and torch.equal(text, expected[:, _VIDEO:])


@_on_vsa2_gpu
def test_pan2_joint_vsa2_dense_layers_keep_every_video_tile(single_process_group) -> None:
    # PAN2's blocks are prefixed transformer_blocks.<i>: a listed block runs the kernels at full selection, exactly as
    # an all-dense build does, while the other blocks keep the build's sparsity.
    policy = ("first", "last", "last")
    with_dense = _build(0.9, policy, dense_layers=(3, ))
    inputs = _inputs(3)
    with torch.no_grad():
        listed = _attend(_layer("transformer_blocks.3.attn"), with_dense, *inputs)
        unlisted = _attend(_layer("transformer_blocks.4.attn"), with_dense, *inputs)
        full = _attend(_layer("transformer_blocks.3.attn"), _build(0.0, policy), *inputs)
        sparse = _attend(_layer("transformer_blocks.4.attn"), _build(0.9, policy), *inputs)
    assert all(torch.equal(actual, expected) for actual, expected in zip(listed, full))
    assert all(torch.equal(actual, expected) for actual, expected in zip(unlisted, sparse))
    assert not torch.equal(listed[0], unlisted[0])


@_on_vsa2_gpu
def test_pan2_joint_vsa2_compiles_into_a_fullgraph_region(single_process_group) -> None:
    # Regional compile traces the attention layer into each block's graph (fullgraph, dynamic shapes): the VSA2
    # kernels must be one opaque custom op there, forward and backward, and give what the eager layer gives.
    metadata = _build(0.9, ("first", "last", "last"))
    layer = _layer()
    layer._set_compile_forward_enabled(True)

    def attend(q, k, v, text_q, text_k, text_v, gate):
        return layer(q, k, v, _VIDEO, secondary_q=text_q, secondary_k=text_k, secondary_v=text_v,
                     secondary_seq_len=_TEXT, attn_metadata=metadata, gate_compress=gate)

    compiled = torch.compile(attend, fullgraph=True, dynamic=True)
    inputs = _inputs(2)

    def run(fn):
        leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
            video, text = fn(*leaves)
        (video.float().pow(2).mean() + text.float().pow(2).mean()).backward()
        return video.detach(), text.detach(), [leaf.grad for leaf in leaves]

    eager_video, eager_text, eager_grads = run(attend)
    video, text, grads = run(compiled)
    assert torch.equal(video, eager_video) and torch.equal(text, eager_text)
    # dQ is reduced across key blocks in no fixed order: equal up to the last bf16 bit.
    for name, actual, expected in zip(("dq", "dk", "dv", "dq_text", "dk_text", "dv_text", "dgate"), grads,
                                      eager_grads):
        assert float((actual.float() - expected.float()).abs().max()) <= 1e-2, name


def _oracle(inputs: list[torch.Tensor], policy: tuple[str, str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 dense attention plus the gated attention between mean-pooled blocks, from the partition alone."""
    q, k, v, text_q, text_k, text_v, gate = inputs
    q, k, v = (torch.cat([a, b], dim=1)[0].float().transpose(0, 1) for a, b in ((q, text_q), (k, text_k), (v, text_v)))
    scale = _HEAD_DIM**-0.5
    fine = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v  # [H, L, D]
    # Each token's block: the text in runs of 128 rows (blocks first), the video tiles of the partition.
    sizes = [_axis_tile_sizes(n, b, p) for n, b, p in zip(_THW, (4, 4, 8), policy)]
    axis_tile = [torch.tensor([i for i, size in enumerate(axis) for _ in range(size)]) for axis in sizes]
    tile = ((axis_tile[0][:, None, None] * len(sizes[1]) + axis_tile[1][None, :, None]) * len(sizes[2]) +
            axis_tile[2][None, None, :]).reshape(-1)
    text_blocks = math.ceil(_TEXT / 128)
    block = torch.cat([tile + text_blocks, torch.arange(_TEXT) // 128]).cuda()
    one_hot = torch.nn.functional.one_hot(block).float()  # [L, N]
    pooled = [(one_hot.T @ x) / one_hot.sum(0)[:, None] for x in (q, k, v)]
    coarse = torch.softmax(pooled[0] @ pooled[1].transpose(-1, -2) * scale, dim=-1) @ pooled[2]  # [H, N, D]
    full_gate = torch.cat([gate, torch.zeros_like(text_q)], dim=1)[0].float().transpose(0, 1)
    out = (fine + full_gate * coarse[:, block]).transpose(0, 1)[None]
    return out[:, :_VIDEO], out[:, _VIDEO:]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


@_on_vsa2_gpu
@pytest.mark.parametrize("policy", [("first", "last", "last"), ("last", "last", "last")])
def test_pan2_joint_vsa2_dense_matches_the_partition_oracle(single_process_group, policy) -> None:
    metadata = _build(0.0, policy)
    layer = _layer()

    def run(fn, inputs):
        leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
        video, text = fn(leaves)
        (video.float().pow(2).mean() + text.float().pow(2).mean()).backward()
        return video.detach(), text.detach(), [leaf.grad for leaf in leaves]

    inputs = _inputs(1)
    video, text, grads = run(lambda leaves: _attend(layer, metadata, *leaves), inputs)
    oracle_video, oracle_text, oracle_grads = run(lambda leaves: _oracle(leaves, policy), inputs)
    assert _relative_error(video, oracle_video) < 2e-2
    assert _relative_error(text, oracle_text) < 2e-2
    for name, actual, expected in zip(("dq", "dk", "dv", "dq_text", "dk_text", "dv_text", "dgate"), grads,
                                      oracle_grads):
        assert _relative_error(actual, expected) < 5e-2, name


def _region_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Relative errors over all video rows, the first frame (its own time tile) and the padded edge tiles."""
    t, h, w = _THW
    frame = torch.arange(_VIDEO) // (h * w)
    ph, pw = torch.arange(_VIDEO) // w % h, torch.arange(_VIDEO) % w
    # The partial tiles at the end of each axis under the "last" policies: 22 = 4 x 5 + 2 rows, 39 = 8 x 4 + 7 columns.
    edge = (ph >= h - h % 4) | (pw >= w - w % 8)
    regions = {"video": torch.ones(_VIDEO, dtype=torch.bool), "first frame": frame == 0, "edge tiles": edge}
    return {name: _relative_error(actual[:, mask.cuda()], expected[:, mask.cuda()]) for name, mask in regions.items()}


@_on_vsa2_gpu
def test_pan2_joint_vsa2_sparse_backward_matches_the_reference(single_process_group) -> None:
    # At the student's sparsity 0.9 the layer's forward and backward on PAN2's rows (the first frame in a time tile of
    # its own, end-padded height and width tiles, text in the compression keys) are the kernels' fp32 reference's,
    # judged under the kernels' own selection, overall and on the first frame and the padded tiles alone. On B300 every
    # error is at bf16 rounding (0.23-0.27%; dgate 0.002%), so the 1% limit catches a fault in one region.
    from fastvideo.attention.backends.video_sparse_attn import compute_topk
    from fastvideo.attention.backends.video_sparse_attn_h3 import _vsa2_attention

    reference = _reference()
    metadata = _build(0.9, ("first", "last", "last"))
    inputs = _inputs(3)
    generator = torch.Generator(device="cuda").manual_seed(4)
    dvideo, dtext = (torch.randn(1, length, _HEADS, _HEAD_DIM, device="cuda", generator=generator).to(torch.bfloat16)
                     for length in (_VIDEO, _TEXT))
    leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
    video, text = _attend(_layer(), metadata, *leaves)
    torch.autograd.backward([video, text], [dvideo, dtext])

    rows = _expected_rows(_THW, _TEXT, reverse_time=True).cuda()
    seq = metadata.variable_block_sizes.numel() * 128

    def place(video_rows: torch.Tensor, text_rows: torch.Tensor) -> torch.Tensor:
        tiled = torch.zeros(1, seq, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        tiled[:, rows] = torch.cat([video_rows, text_rows], dim=1)
        return tiled.transpose(1, 2).contiguous()

    q, k, v, text_q, text_k, text_v, gate = inputs
    placed = [place(a, b) for a, b in ((q, text_q), (k, text_k), (v, text_v), (gate, torch.zeros_like(text_q)))]
    topk = compute_topk(0.9, metadata.num_video_tiles)
    assert 0 < topk < metadata.num_video_tiles
    with torch.no_grad():
        out, selection = _vsa2_attention()(*placed, *_THW, _TEXT, topk)
    ref_out, ref_selection, _, ref_dq, ref_dk, ref_dv, ref_dgate = reference.run(
        *placed, place(dvideo, dtext), *_THW, _TEXT, topk, outputs=(out, selection))
    judged = ~torch.isnan(ref_selection)
    assert bool((selection[judged] == ref_selection[judged]).all()), "the kernels' selection is not this top-k"

    def unplace(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = x.transpose(1, 2)[:, rows]
        return packed[:, :_VIDEO], packed[:, _VIDEO:]

    expected = {"out": unplace(ref_out), "dq": unplace(ref_dq), "dk": unplace(ref_dk), "dv": unplace(ref_dv)}
    actual = {"out": (video.detach(), text.detach()), "dq": (leaves[0].grad, leaves[3].grad),
              "dk": (leaves[1].grad, leaves[4].grad), "dv": (leaves[2].grad, leaves[5].grad)}
    for name, (video_actual, text_actual) in actual.items():
        video_expected, text_expected = expected[name]
        for region, error in _region_errors(video_actual, video_expected).items():
            assert error < 1e-2, f"{name} on the {region}: relative error {error:.3g}"
        assert _relative_error(text_actual, text_expected) < 1e-2, f"{name} on the text"
    # The text rows' gate is zero by contract, so only the video rows' gate gradient is judged.
    for region, error in _region_errors(leaves[6].grad, unplace(ref_dgate)[0]).items():
        assert error < 1e-2, f"dgate on the {region}: relative error {error:.3g}"
