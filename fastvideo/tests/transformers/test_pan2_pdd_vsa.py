# SPDX-License-Identifier: Apache-2.0
"""PAN2 Parallel Decoding Distillation heads and the joint ``[video; text]`` VSA-H3 attention, on CPU.

The widened ``proj_out`` must reproduce the released head when it is expanded from a single-head checkpoint, a fused block
must equal the weighted mean of its heads, the VSA gate must exist only where the VSA-H3 backend resolves, and the joint
attention must give the dense answer once the backend tiles the packed ``[video; text]`` sequence. Tile 128 is the VSA128
geometry; the kernels themselves are checked on a GPU host.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.attention import DistributedAttention
from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder,
                                                               token_tile_and_valid)
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig, PAN2VideoConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.pdd import PDDModalitySchedule, PDDReplicatedLinear, pdd_fine_grid
from fastvideo.models.dits import pan2 as pan2_module
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.platforms import AttentionBackendEnum

PDD_STEPS = 4
LATENT_CHANNELS = 4
TINY_ARCH = dict(
    patch_size=(1, 2, 2),
    in_channels=2 * LATENT_CHANNELS + 1,
    out_channels=LATENT_CHANNELS,
    num_attention_heads=4,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    context_mlp_ratio=2.0,
    refiner_mlp_ratio=2.0,
    text_embed_dim=16,
    rope_axes_dim=(4, 6, 6),
)


@pytest.fixture()
def cpu_attention(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))


@pytest.fixture()
def single_process_group(env_overrides):
    """A 1x1 sequence/tensor-parallel group, which DistributedAttention needs even on one CPU process."""
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    env_overrides.enter_context(envs.override_external("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", os.environ.get("MASTER_PORT") or "29593"))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


def _model(pdd_steps: int | None = None) -> pan2_module.PAN2Transformer3DModel:
    arch = PAN2ArchConfig(**TINY_ARCH, pdd_steps=pdd_steps)
    return pan2_module.PAN2Transformer3DModel(PAN2VideoConfig(arch_config=arch), hf_config={})


def _inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return dict(
        hidden_states=torch.randn(1, TINY_ARCH["in_channels"], 3, 4, 6, generator=generator),
        encoder_hidden_states=torch.randn(1, 5, TINY_ARCH["text_embed_dim"], generator=generator),
        timestep=torch.tensor([600.0]),
    )


def _randomize(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                              forward_batch=ForwardBatch(data_type="dummy")):
        return model(**inputs)


def _widen_from(single: torch.nn.Module, widened: pan2_module.PAN2Transformer3DModel) -> None:
    """Load a single-head state dict into the widened model the way the loader does."""
    state = {name: widened.preprocess_loaded_weight(name, tensor) for name, tensor in single.state_dict().items()}
    assert widened.load_state_dict(state, strict=True).missing_keys == []


def test_widened_heads_reproduce_the_released_head(cpu_attention, single_process_group):
    single, widened = _model(), _model(PDD_STEPS)
    _randomize(single, seed=1)
    _widen_from(single, widened)

    assert isinstance(widened.proj_out, PDDReplicatedLinear) and widened.proj_out.channel_layout == "channel_major"
    inputs = _inputs()
    reference = _forward(single, inputs)
    heads = _forward(widened, inputs)

    assert reference.shape == (1, LATENT_CHANNELS, 3, 4, 6)
    # The channel axis of a PDD student reads (pdd_steps, C).
    assert heads.shape == (1, PDD_STEPS * LATENT_CHANNELS, 3, 4, 6)
    for head in heads.unflatten(1, (PDD_STEPS, LATENT_CHANNELS)).unbind(1):
        torch.testing.assert_close(head, reference, rtol=1e-5, atol=1e-6)


def test_fused_block_is_the_weighted_mean_of_its_heads(cpu_attention, single_process_group):
    widened = _model(PDD_STEPS)
    _randomize(widened, seed=2)
    inputs = _inputs(seed=3)
    weights = {"video": PDDModalitySchedule(shift=9.0).integration_weights(pdd_fine_grid(PDD_STEPS))}

    heads = _forward(widened, inputs).unflatten(1, (PDD_STEPS, LATENT_CHANNELS))
    with widened.fuse_pdd_block(1, 3, weights, torch.float32):
        fused = _forward(widened, inputs)

    alpha = weights["video"][1:3] / weights["video"][1:3].sum()
    expected = torch.einsum("n,bnc...->bc...", alpha.float(), heads[:, 1:3].float())
    assert fused.shape == expected.shape
    torch.testing.assert_close(fused.float(), expected, rtol=1e-4, atol=1e-5)


def test_pdd_export_helpers(cpu_attention, single_process_group):
    widened = _model(PDD_STEPS)
    assert widened.inference_config_overrides() == {"pdd_steps": PDD_STEPS}
    assert _model().inference_config_overrides() == {}
    rows = widened.proj_out.head_output_size
    assert widened.inference_export_expected_shape("proj_out.weight", (rows, 7)) == (PDD_STEPS * rows, 7)
    # A continued run may start from an export that is already widened.
    assert widened.inference_export_expected_shape("proj_out.weight", (PDD_STEPS * rows, 7)) == (PDD_STEPS * rows, 7)
    with pytest.raises(ValueError, match="output rows"):
        widened.inference_export_expected_shape("proj_out.weight", (rows + 1, 7))
    assert widened.inference_export_expected_shape("transformer_blocks.0.ff.fc_in.weight", (3, 5)) == (3, 5)
    with pytest.raises(RuntimeError, match="no PDD heads"):
        _ = _model().pdd_linears


class _VSAH3Selection:
    """What the selector returns when VIDEO_SPARSE_ATTN_H3 resolves on a CUDA host."""

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN_H3"


def test_vsa_gate_exists_only_where_vsa_h3_resolves(monkeypatch, cpu_attention, single_process_group):
    dense = _model()
    assert not any("gate_compress" in name for name, _ in dense.named_parameters())
    assert all(type(block.attn.attn) is DistributedAttention for block in dense.transformer_blocks)

    select = pan2_module.get_attn_backend

    def select_vsa_h3(head_size, dtype, supported_attention_backends=None, **kwargs):
        if supported_attention_backends and AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3 in supported_attention_backends:
            return _VSAH3Selection
        return select(head_size, dtype, supported_attention_backends=supported_attention_backends, **kwargs)

    monkeypatch.setattr(pan2_module, "get_attn_backend", select_vsa_h3)
    sparse = _model()
    gates = [name for name, _ in sparse.named_parameters() if "gate_compress" in name]
    assert gates == [f"transformer_blocks.{index}.attn.to_gate_compress.weight" for index in range(2)]
    assert all(isinstance(block.attn.attn, DistributedAttention_VSA) for block in sparse.transformer_blocks)
    # The refiner attends over the text alone: it never sees the VSA backend, so it builds neither gate nor VSA layer.
    for refiner in sparse.context_refiner:
        assert refiner.attn.attn.backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
        assert not hasattr(refiner.attn, "to_gate_compress")


# ---- joint [video; text] attention under the VSA-H3 backend ---------------------------------------------------------


class _DenseOverTilesImpl:
    """The H3 backend's real tiling around a dense, bidirectional kernel: the answer sparse attention gives when it
    keeps every tile. Pad rows of partial tiles never serve as keys."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        self.real = MiniMaxH3VSAImpl(num_heads=num_heads,
                                     head_size=head_size,
                                     causal=False,
                                     softmax_scale=head_size**-0.5)
        self.gate_seen: torch.Tensor | None = None

    def preprocess_qkv(self, qkv, metadata):
        return self.real.preprocess_qkv(qkv, metadata)

    def postprocess_output(self, output, metadata):
        return self.real.postprocess_output(output, metadata)

    def forward(self, q, k, v, gate_compress, metadata):
        self.gate_seen = gate_compress
        _, valid = token_tile_and_valid(metadata.variable_block_sizes, metadata.tile_elems)
        keep = torch.zeros(q.shape[1], dtype=torch.bool)
        keep[:valid.numel()] = valid
        mask = keep[None, None, None, :]
        out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2),
                                                               k.transpose(1, 2),
                                                               v.transpose(1, 2),
                                                               attn_mask=mask)
        return out.transpose(1, 2)


def _tile128_metadata(latent_shape: tuple[int, int, int], text_len: int):
    # PAN2 packs ``[video; text]``: the video is one sparse region and the text a dense segment after it.
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               patch_size=(1, 2, 2),
                                               VSA_sparsity=0.0,
                                               packed_segments=(latent_shape, text_len),
                                               device=torch.device("cpu"),
                                               tile_size=128)


@pytest.mark.parametrize("with_gate", [False, True])
def test_joint_vsa_attention_matches_dense_joint_attention(cpu_attention, single_process_group, with_gate):
    # 6 x 5 x 7 video tokens (the grid is not a multiple of the 4 x 4 x 8 tile) and 9 text rows: partial tiles on both.
    latent_shape, text_len, num_heads, head_size = (6, 10, 14), 9, 4, 16
    video_len = 6 * 5 * 7
    metadata = _tile128_metadata(latent_shape, text_len)
    assert metadata.total_seq_length == video_len + text_len

    generator = torch.Generator().manual_seed(4)

    def rand(length):
        return torch.randn(1, length, num_heads, head_size, generator=generator)

    q, k, v = rand(video_len), rand(video_len), rand(video_len)
    text_q, text_k, text_v = rand(text_len), rand(text_len), rand(text_len)

    dense = DistributedAttention(num_heads=num_heads, head_size=head_size, causal=False,
                                 supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    sparse = DistributedAttention_VSA(num_heads=num_heads, head_size=head_size, causal=False,
                                      supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    impl = _DenseOverTilesImpl(num_heads, head_size)
    sparse.backend, sparse.attn_impl = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, impl

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                              forward_batch=ForwardBatch(data_type="dummy")):
        expected_video, expected_text = dense(q, k, v, video_len, replicated_q=text_q, replicated_k=text_k,
                                              replicated_v=text_v)
        gate = rand(video_len) if with_gate else None
        video, text = sparse(q, k, v, video_len, replicated_q=text_q, replicated_k=text_k, replicated_v=text_v,
                             gate_compress=gate, attn_metadata=metadata)

    assert video.shape == expected_video.shape and text.shape == expected_text.shape
    torch.testing.assert_close(video, expected_video, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(text, expected_text, rtol=1e-4, atol=1e-5)
    if with_gate:
        # The backend receives one gate row per packed row; the text rows' gate is zero.
        assert impl.gate_seen is not None
        untiled = impl.real.postprocess_output(impl.gate_seen, metadata)
        torch.testing.assert_close(untiled[:, :video_len], gate)
        assert not bool(untiled[:, video_len:].any())


def test_joint_vsa_attention_rejects_replicated_rows_on_other_backends(cpu_attention, single_process_group):
    layer = DistributedAttention_VSA(num_heads=4, head_size=16, causal=False,
                                     supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    x = torch.zeros(1, 4, 4, 16)
    with pytest.raises(AssertionError, match="VIDEO_SPARSE_ATTN_H3"):
        layer(x, x, x, 4, replicated_q=x, replicated_k=x, replicated_v=x)


def test_vsa128_geometry_of_a_pan2_clip():
    """An 8 s clip: 49 latent frames of 22 x 39 tokens, then the text, in 4 x 4 x 8 tiles."""
    text_len = 1100
    metadata = _tile128_metadata((49, 44, 78), text_len)
    video_tokens = 49 * 22 * 39
    assert video_tokens == 42042
    assert metadata.total_seq_length == video_tokens + text_len
    assert metadata.tile_elems == 128
    assert metadata.num_prefix_tiles == math.ceil(text_len / 128)
    assert metadata.num_video_tiles == math.ceil(49 / 4) * math.ceil(22 / 4) * math.ceil(39 / 8)
    # Every packed row has exactly one place in the tiled sequence.
    assert metadata.untile_combined_index.unique().numel() == metadata.total_seq_length
    assert int(metadata.variable_block_sizes.sum()) == metadata.total_seq_length


# ---- PAN2's VSA layout: tile placement, compression keys, native tile 128 ------------------------------------------


def test_axis_tile_policies_place_the_partial_tile():
    from fastvideo.attention.backends.video_sparse_attn_h3 import _axis_tile_sizes

    # The 8 s PAN2 grid of 49 x 22 x 39 tokens in 4 x 4 x 8 tiles.
    assert _axis_tile_sizes(49, 4, "last") == (4, ) * 12 + (1, )
    assert _axis_tile_sizes(49, 4, "first") == (1, ) + (4, ) * 12
    assert _axis_tile_sizes(22, 4, "center") == (4, 4, 3, 3, 4, 4)
    assert _axis_tile_sizes(39, 8, "center") == (8, 8, 7, 8, 8)
    # A deficit wider than the tile count spreads more than one row per tile and never empties one.
    assert _axis_tile_sizes(9, 8, "center") == (4, 5)
    assert _axis_tile_sizes(1, 4, "first") == (1, )
    assert _axis_tile_sizes(40, 8, "center") == (8, ) * 5
    with pytest.raises(ValueError, match="axis policy"):
        _axis_tile_sizes(10, 4, "middle")


def test_last_policy_partition_is_the_upstream_layout():
    from fastvideo.attention.backends.video_sparse_attn import (construct_variable_block_sizes,
                                                                get_tile_partition_indices)
    from fastvideo.attention.backends.video_sparse_attn_h3 import _policy_tile_partition

    grid, tile = (7, 6, 11), (4, 4, 8)
    index, sizes = _policy_tile_partition(grid, tile, ("last", "last", "last"), torch.device("cpu"))
    num_tiles = tuple(math.ceil(length / edge) for length, edge in zip(grid, tile))
    assert torch.equal(index, get_tile_partition_indices(grid, tile, torch.device("cpu")))
    assert torch.equal(sizes, construct_variable_block_sizes(grid, num_tiles, torch.device("cpu"), tile).long())


def _pan2_clip_metadata(**layout):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               patch_size=(1, 2, 2),
                                               VSA_sparsity=0.9,
                                               packed_segments=((49, 44, 78), 1100),
                                               device=torch.device("cpu"),
                                               tile_size=128,
                                               **layout)


def test_pan2_layout_gives_the_conditioning_frame_its_own_tile_row():
    upstream = _pan2_clip_metadata()
    pan2 = _pan2_clip_metadata(video_tile_policy=("first", "center", "center"))
    tokens_per_frame = 22 * 39
    for metadata in (upstream, pan2):
        # Same tile count and coverage either way; only the partition moves.
        assert metadata.num_video_tiles == 13 * 6 * 5 and metadata.num_prefix_tiles == 9
        assert metadata.untile_combined_index.unique().numel() == metadata.total_seq_length == 42042 + 1100
    # Tile row 0 of the PAN2 layout is exactly latent frame 0; upstream mixes it with frames 1-3.
    first, last = pan2.video_tile_spans[0][0], pan2.video_tile_spans[0][0] + 6 * 5
    tile_of_row = pan2.untile_combined_index // 128
    frame0_tiles = tile_of_row[:tokens_per_frame]
    assert int(frame0_tiles.min()) == first and int(frame0_tiles.max()) == last - 1
    assert int(pan2.variable_block_sizes[first:last].sum()) == tokens_per_frame
    assert int(upstream.variable_block_sizes[first:last].sum()) == 4 * tokens_per_frame
    # Height spreads its shortfall: no 2-row sliver tiles remain.
    assert int(pan2.variable_block_sizes[first + 30:].min()) >= 4 * 3 * 7
    assert int(upstream.variable_block_sizes.min()) < int(pan2.variable_block_sizes[first + 30:].min())
    with pytest.raises(ValueError, match="video_tile_policy"):
        _pan2_clip_metadata(video_tile_policy=("first", "center"))


def test_compression_keys_exclude_the_text_prefix_only_when_asked():
    from fastvideo.attention.backends.video_sparse_attn_h3 import _compression_scores

    scores = torch.randn(1, 2, 5, 5)
    keep = _tile128_metadata((4, 8, 16), 200)  # 2 prefix tiles of text
    assert keep.num_prefix_tiles == 2 and keep.compress_prefix_keys
    assert _compression_scores(scores, keep) is scores
    video_only = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, patch_size=(1, 2, 2), VSA_sparsity=0.0,
                                                     packed_segments=((4, 8, 16), 200), device=torch.device("cpu"),
                                                     tile_size=128, compress_prefix_keys=False)
    masked = _compression_scores(scores, video_only)
    assert bool(torch.isneginf(masked[..., :2]).all())
    torch.testing.assert_close(masked[..., 2:], scores[..., 2:])
    weights = torch.softmax(masked, dim=-1)
    assert not bool(weights[..., :2].any()) and torch.allclose(weights.sum(-1), torch.ones(1, 2, 5))


@pytest.mark.parametrize("with_gate", [False, True])
def test_sharded_text_rows_match_replicated_text_rows(cpu_attention, single_process_group, with_gate):
    """The secondary (sequence-parallel sharded) text path gives the replicated path's answer, dense and sparse."""
    latent_shape, text_len, num_heads, head_size = (6, 10, 14), 9, 4, 16
    video_len = 6 * 5 * 7
    metadata = _tile128_metadata(latent_shape, text_len)
    generator = torch.Generator().manual_seed(8)

    def rand(length):
        return torch.randn(1, length, num_heads, head_size, generator=generator)

    q, k, v = rand(video_len), rand(video_len), rand(video_len)
    text_q, text_k, text_v = rand(text_len), rand(text_len), rand(text_len)
    gate = rand(video_len) if with_gate else None
    dense = DistributedAttention(num_heads=num_heads, head_size=head_size, causal=False,
                                 supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    sparse = DistributedAttention_VSA(num_heads=num_heads, head_size=head_size, causal=False,
                                      supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    sparse.backend, sparse.attn_impl = AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, _DenseOverTilesImpl(num_heads, head_size)
    replicated = dict(replicated_q=text_q, replicated_k=text_k, replicated_v=text_v)
    secondary = dict(secondary_q=text_q, secondary_k=text_k, secondary_v=text_v, secondary_seq_len=text_len)

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                              forward_batch=ForwardBatch(data_type="dummy")):
        for layer, extra in ((dense, {}), (sparse, dict(gate_compress=gate, attn_metadata=metadata))):
            expected_video, expected_text = layer(q, k, v, video_len, **replicated, **extra)
            video, text = layer(q, k, v, video_len, **secondary, **extra)
            torch.testing.assert_close(video, expected_video)
            torch.testing.assert_close(text, expected_text)
    with pytest.raises(AssertionError, match="exclusive"):
        dense(q, k, v, video_len, **replicated, **secondary)
