# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax H3 VAE compilation and profiling ranges.

The final test is a CUDA regression gate for the reduce-overhead tile path
(real tiled decode/encode with an unmocked ``_stitch_tiles``).
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioBigVGANDecoder,
    MiniMaxH3AudioVAE,
)
from fastvideo.models.vaes.minimax_h3_video import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoAttention,
    MiniMaxH3VideoViTDecoder3d,
)
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


def _empty_typed_module(module_type: type[nn.Module]) -> nn.Module:
    """Create a weightless instance that retains its production module type."""
    module = object.__new__(module_type)
    nn.Module.__init__(module)
    return module


def _assert_dynamic_compile_selects_decoder(
    vae_type: type[nn.Module],
    decoder_type: type[nn.Module],
) -> None:
    """Verify one H3 VAE compiles only its top-level decoder in place."""
    vae = _empty_typed_module(vae_type)
    decoder = _empty_typed_module(decoder_type)
    same_type_under_another_name = _empty_typed_module(decoder_type)
    unrelated_submodule = nn.Identity()
    vae.decoder = decoder
    vae.same_type_under_another_name = same_type_under_another_name
    vae.unrelated_submodule = unrelated_submodule

    compiled_forward = Mock(name="compiled_forward")
    compile_kwargs = {"backend": "inductor", "dynamic": False}
    with patch(
        "fastvideo.pipelines.composed_pipeline_base.torch.compile",
        return_value=compiled_forward,
    ) as compile_mock:
        compiled_count = ComposedPipelineBase._compile_with_conditions(vae, compile_kwargs)

    assert compiled_count == 1
    compile_mock.assert_called_once()
    selected_forward = compile_mock.call_args.args[0]
    assert selected_forward.__self__ is decoder
    assert selected_forward.__func__ is decoder_type.forward
    assert compile_mock.call_args.kwargs == compile_kwargs
    assert decoder.forward is compiled_forward
    assert "forward" not in same_type_under_another_name.__dict__
    assert "forward" not in unrelated_submodule.__dict__

    wrong_type_vae = _empty_typed_module(vae_type)
    wrong_type_vae.decoder = nn.Identity()
    with patch("fastvideo.pipelines.composed_pipeline_base.torch.compile") as wrong_type_compile:
        wrong_type_count = ComposedPipelineBase._compile_with_conditions(wrong_type_vae, compile_kwargs)
    assert wrong_type_count == 0
    wrong_type_compile.assert_not_called()


def _assert_reduce_overhead_compile(compiled_function: Any) -> None:
    """Verify a class-owned compile boundary enables CUDA Graph replay."""
    assert hasattr(compiled_function, "get_compiler_config")
    assert compiled_function.get_compiler_config()["triton.cudagraphs"] is True


def test_video_attention_uses_selected_fastvideo_backend() -> None:
    """Pass BSHD tensors to the selected dense backend without forward metadata."""
    backend_call: dict[str, Any] = {}

    class RecordingAttentionImpl:
        """Record the backend construction and forward contracts."""

        def __init__(self, **kwargs: Any) -> None:
            backend_call["init"] = kwargs

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_metadata: Any,
        ) -> torch.Tensor:
            """Return values unchanged after recording the backend inputs."""
            backend_call["shapes"] = (query.shape, key.shape, value.shape)
            backend_call["metadata"] = attention_metadata
            return value

    class RecordingAttentionBackend:
        """Supply the recording implementation through the backend API."""

        @staticmethod
        def get_impl_cls() -> type[RecordingAttentionImpl]:
            return RecordingAttentionImpl

    with (
        patch("fastvideo.platforms.current_platform") as current_platform,
        patch(
            "fastvideo.models.vaes.minimax_h3_video.get_attn_backend",
            return_value=RecordingAttentionBackend,
        ) as get_backend,
    ):
        current_platform.is_cuda_alike.return_value = True
        attention = MiniMaxH3VideoAttention(dim=8, heads=2, dim_head=4)

    attention.to_q = nn.Identity()
    attention.to_k = nn.Identity()
    attention.to_v = nn.Identity()
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()
    attention.to_out[0] = nn.Identity()
    output = attention(torch.empty((1, 3, 8), device="meta"))

    get_backend.assert_called_once_with(
        4,
        torch.bfloat16,
        supported_attention_backends=(
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
        ),
    )
    assert backend_call["init"] == {
        "num_heads": 2,
        "head_size": 4,
        "softmax_scale": 0.5,
        "num_kv_heads": 2,
        "causal": False,
        # The FP32-pinned VAE opts out of the FASTVIDEO_NVFP4_FA4 env opt-in.
        "nvfp4_fa4": False,
    }
    assert backend_call["shapes"] == ((1, 3, 2, 4), ) * 3
    assert backend_call["metadata"] is None
    assert output.shape == (1, 3, 8)


def test_video_attention_cpu_uses_torch_sdpa() -> None:
    """Use PyTorch SDPA when H3 VAE attention receives CPU tensors."""
    with (
        patch("fastvideo.platforms.current_platform") as current_platform,
        patch("fastvideo.models.vaes.minimax_h3_video.get_attn_backend") as get_backend,
    ):
        current_platform.is_cuda_alike.return_value = False
        attention = MiniMaxH3VideoAttention(dim=8, heads=2, dim_head=4)

    get_backend.assert_not_called()
    assert attention.attn_impl is None

    attention.to_q = nn.Identity()
    attention.to_k = nn.Identity()
    attention.to_v = nn.Identity()
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()
    attention.to_out[0] = nn.Identity()
    hidden_states = torch.randn(1, 3, 8)
    query = hidden_states.unflatten(2, (2, 4)).permute(0, 2, 1, 3)
    expected = F.scaled_dot_product_attention(query, query, query).permute(0, 2, 1, 3).flatten(2, 3)

    torch.testing.assert_close(attention(hidden_states), expected)


def test_compile_with_conditions_selects_minimax_h3_video_decoder() -> None:
    """Compile the registered video decoder with the VAE runtime kwargs."""
    assert not hasattr(MiniMaxH3VideoViTDecoder3d.forward, "get_compiler_config")
    _assert_dynamic_compile_selects_decoder(AutoencoderKLMiniMaxH3, MiniMaxH3VideoViTDecoder3d)


def test_tile_helpers_stay_eager_by_default() -> None:
    """Leave the tile helpers uncompiled unless the VAE compile opt-in runs.

    Default users must keep pre-PR eager behavior: no inductor/triton
    requirement, no first-decode compile latency, and no permanent cudagraph
    memory pools unless ``enable_torch_compile_vae`` was requested.
    """
    assert AutoencoderKLMiniMaxH3._tile_helpers_compiled is False
    for helper in (AutoencoderKLMiniMaxH3._stitch_tiles, AutoencoderKLMiniMaxH3._project_decoder_tile):
        assert not hasattr(helper, "get_compiler_config")
        assert not hasattr(helper, "_torchdynamo_orig_callable")


def test_prepare_for_compile_installs_reduce_overhead_tile_helpers() -> None:
    """Compile both tile helpers with CUDA Graph replay under the opt-in hook.

    ``ComposedPipelineBase._maybe_compile_pipeline_module`` invokes
    ``prepare_for_compile`` only when ``enable_torch_compile_vae`` is set,
    right before the decoder is compiled through ``_compile_conditions``.
    """
    vae = _empty_typed_module(AutoencoderKLMiniMaxH3)
    assert vae._tile_helpers_compiled is False

    vae.prepare_for_compile()

    assert vae._tile_helpers_compiled is True
    _assert_reduce_overhead_compile(vae._stitch_tiles)
    _assert_reduce_overhead_compile(vae._project_decoder_tile)

    # Idempotent: the pipeline hook may run more than once per instance.
    compiled_stitch = vae._stitch_tiles
    compiled_project = vae._project_decoder_tile
    vae.prepare_for_compile()
    assert vae._stitch_tiles is compiled_stitch
    assert vae._project_decoder_tile is compiled_project

    # The opt-in mutates only this instance; the class (and therefore every
    # default instance) stays eager.
    assert AutoencoderKLMiniMaxH3._tile_helpers_compiled is False
    assert not hasattr(AutoencoderKLMiniMaxH3._stitch_tiles, "get_compiler_config")
    assert not hasattr(AutoencoderKLMiniMaxH3._project_decoder_tile, "get_compiler_config")


def test_tile_drivers_return_caller_owned_tensors_when_compiled() -> None:
    """Clone the stitched canvas out of cudagraph-pooled storage under the opt-in.

    With ``mode="reduce-overhead"`` the ``_stitch_tiles`` output is a
    CUDA-graph static buffer that the next replay overwrites, while the
    collect-then-``torch.cat`` consumers (``_decode``/``_encode``/
    ``_encode_pixels``/``encode_keyframe``) hold each chunk or clip result
    across replays. The tile drivers therefore hand back a caller-owned copy
    when the helpers are compiled — and return the stitched tensor by identity
    when eager, keeping default behavior and peak memory unchanged.
    """

    def _mock_tiled_vae(stitched: torch.Tensor) -> nn.Module:
        vae = _empty_typed_module(AutoencoderKLMiniMaxH3)
        vae.use_tiling = True
        vae.spatial_compression_ratio = 1
        vae.tile_sample_min_height = 1
        vae.tile_sample_min_width = 1
        vae.tile_sample_min_overlap_height = 0
        vae.tile_sample_min_overlap_width = 0
        vae._split_tiles = Mock(return_value=([0, 1], [1, 1], [0]))
        vae.post_quant_conv = nn.Identity()
        vae._project_decoder_tile = vae.post_quant_conv
        vae.quant_conv = nn.Identity()
        vae.encoder = nn.Identity()
        vae.decoder = nn.Identity()
        vae._stitch_tiles = Mock(return_value=stitched)
        return vae

    latent_clip = torch.zeros((1, 1, 1, 2, 2))
    for driver in ("_decode_clip", "_encode_clip"):
        stitched = torch.zeros((1, 1, 1, 2, 2))

        # Default instances return the stitched tensor without copying.
        eager_vae = _mock_tiled_vae(stitched)
        assert getattr(eager_vae, driver)(latent_clip) is stitched

        compiled_vae = _mock_tiled_vae(stitched)
        compiled_vae._tile_helpers_compiled = True
        owned = getattr(compiled_vae, driver)(latent_clip)
        assert owned is not stitched
        assert torch.equal(owned, stitched)


def test_compile_with_conditions_selects_minimax_h3_audio_decoder() -> None:
    """Compile the audio VAE decoder that the H3 waveform decode path calls."""
    _assert_dynamic_compile_selects_decoder(MiniMaxH3AudioVAE, MiniMaxH3AudioBigVGANDecoder)


def test_decode_emits_indexed_temporal_chunk_ranges() -> None:
    """Nest frame-segment ranges under each temporal decoder chunk range."""
    vae = _empty_typed_module(AutoencoderKLMiniMaxH3)
    vae.tokens_chunk_size = 1
    vae.token_overlap = 1
    vae.temporal_compression_ratio = 1
    vae.frame_pre_padding = 0
    vae.frame_overlap = 1
    vae.config = SimpleNamespace(token_drop=1)
    vae._decode_clip = Mock(return_value=torch.zeros((1, 1, 2, 1, 1)))
    range_events = []

    @contextmanager
    def record_range(name: str):
        range_events.append(("enter", name))
        try:
            yield
        finally:
            range_events.append(("exit", name))

    with patch("fastvideo.models.vaes.minimax_h3_video.nvtx_range", record_range):
        decoded = vae._decode(torch.zeros((1, 1, 2, 1, 1)))

    assert decoded.shape == (1, 1, 3, 1, 1)
    assert vae._decode_clip.call_count == 2
    assert range_events == [
        ("enter", "minimax_h3.vae.temporal_chunk.0"),
        ("enter", "minimax_h3.vae.temporal_chunk.0.frame_segment.0"),
        ("exit", "minimax_h3.vae.temporal_chunk.0.frame_segment.0"),
        ("enter", "minimax_h3.vae.temporal_chunk.0.frame_segment.1"),
        ("exit", "minimax_h3.vae.temporal_chunk.0.frame_segment.1"),
        ("exit", "minimax_h3.vae.temporal_chunk.0"),
        ("enter", "minimax_h3.vae.temporal_chunk.1"),
        ("enter", "minimax_h3.vae.temporal_chunk.1.frame_segment.0"),
        ("exit", "minimax_h3.vae.temporal_chunk.1.frame_segment.0"),
        ("enter", "minimax_h3.vae.temporal_chunk.1.frame_segment.1"),
        ("exit", "minimax_h3.vae.temporal_chunk.1.frame_segment.1"),
        ("exit", "minimax_h3.vae.temporal_chunk.1"),
    ]


def test_decode_clip_no_spatial_tiling_stage_ranges() -> None:
    """Separate untiled latent projection and decoder ranges."""
    vae = _empty_typed_module(AutoencoderKLMiniMaxH3)
    vae.use_tiling = False
    range_events = []
    vae.post_quant_conv = nn.Identity()
    vae.post_quant_conv.register_forward_hook(
        lambda _module, _args, _output: range_events.append(("call", "post_quant_conv")))
    vae.decoder = nn.Identity()
    vae.decoder.register_forward_hook(
        lambda _module, _args, _output: range_events.append(("call", "decoder_forward")))
    latent_clip = torch.zeros((1, 1, 1, 2, 2))

    @contextmanager
    def record_range(name: str):
        range_events.append(("enter", name))
        try:
            yield
        finally:
            range_events.append(("exit", name))

    with patch("fastvideo.models.vaes.minimax_h3_video.nvtx_range", record_range):
        decoded_clip = vae._decode_clip(latent_clip)

    assert decoded_clip is latent_clip
    assert range_events == [
        ("enter", "minimax_h3.vae.decode_clip"),
        ("enter", "minimax_h3.vae.decode_clip.no_s_tile.post_quant_conv"),
        ("call", "post_quant_conv"),
        ("exit", "minimax_h3.vae.decode_clip.no_s_tile.post_quant_conv"),
        ("enter", "minimax_h3.vae.decode_clip.no_s_tile.decoder_forward"),
        ("call", "decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.no_s_tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip"),
    ]


def test_decode_clip_emits_tiled_stage_ranges() -> None:
    """Nest indexed decoder tiles between tile-splitting and stitching ranges."""
    vae = _empty_typed_module(AutoencoderKLMiniMaxH3)
    vae.use_tiling = True
    vae.spatial_compression_ratio = 1
    vae.tile_sample_min_height = 1
    vae.tile_sample_min_width = 1
    vae.tile_sample_min_overlap_height = 0
    vae.tile_sample_min_overlap_width = 0
    vae._split_tiles = Mock(side_effect=[
        ([0, 1], [1, 1], [0]),
        ([0, 1], [1, 1], [0]),
    ])
    vae.post_quant_conv = nn.Identity()
    vae._project_decoder_tile = Mock(side_effect=vae.post_quant_conv)
    vae.decoder = nn.Identity()
    stitched_clip = torch.zeros((1, 1, 1, 2, 2))
    vae._stitch_tiles = Mock(return_value=stitched_clip)
    range_events = []

    @contextmanager
    def record_range(name: str):
        range_events.append(("enter", name))
        try:
            yield
        finally:
            range_events.append(("exit", name))

    with patch("fastvideo.models.vaes.minimax_h3_video.nvtx_range", record_range):
        decoded_clip = vae._decode_clip(torch.zeros((1, 1, 1, 2, 2)))

    # Default (eager) instances return the stitched canvas by identity; the
    # caller-owned copy under the compile opt-in is pinned by
    # test_tile_drivers_return_caller_owned_tensors_when_compiled.
    assert decoded_clip is stitched_clip
    assert vae._split_tiles.call_count == 2
    assert vae._project_decoder_tile.call_count == 4
    assert vae._stitch_tiles.call_count == 1
    assert range_events == [
        ("enter", "minimax_h3.vae.decode_clip"),
        ("enter", "minimax_h3.vae.decode_clip.split_tiles"),
        ("exit", "minimax_h3.vae.decode_clip.split_tiles"),
        ("enter", "minimax_h3.vae.decode_clip.decode_tiles"),
        ("enter", "minimax_h3.vae.decode_clip.tile.0.0"),
        ("enter", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.0.0"),
        ("enter", "minimax_h3.vae.decode_clip.tile.0.1"),
        ("enter", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.0.1"),
        ("enter", "minimax_h3.vae.decode_clip.tile.1.0"),
        ("enter", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.1.0"),
        ("enter", "minimax_h3.vae.decode_clip.tile.1.1"),
        ("enter", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.decoder_forward"),
        ("exit", "minimax_h3.vae.decode_clip.tile.1.1"),
        ("exit", "minimax_h3.vae.decode_clip.decode_tiles"),
        ("enter", "minimax_h3.vae.decode_clip.stitch_tiles"),
        ("exit", "minimax_h3.vae.decode_clip.stitch_tiles"),
        ("exit", "minimax_h3.vae.decode_clip"),
    ]


def _tiny_real_vae() -> AutoencoderKLMiniMaxH3:
    """Random-weight VAE small enough for a real tiled decode/encode on GPU."""
    from fastvideo.configs.models.vaes.minimax_h3_video import (
        MiniMaxH3VideoVAEArchConfig,
        MiniMaxH3VideoVAEConfig,
    )

    arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=1,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    return AutoencoderKLMiniMaxH3(
        MiniMaxH3VideoVAEConfig(
            arch_config=arch,
            use_tiling=False,
            use_temporal_tiling=False,
            use_parallel_tiling=False,
        )).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduce-overhead tile compile requires CUDA graphs")
@torch.inference_mode()
def test_tiled_decode_and_encode_survive_cudagraph_buffer_reuse_on_cuda() -> None:
    """Real tiled decode()/encode() with an unmocked reduce-overhead ``_stitch_tiles``.

    Regression gate for the CUDA-graph output-clobbering bug under the
    ``enable_torch_compile_vae`` opt-in (``prepare_for_compile``): the
    stitched canvas is a cudagraph static buffer, and the
    collect-then-``torch.cat`` consumers (``_decode``/``_encode``/
    ``_encode_pixels``) hold chunk/clip results across subsequent
    ``_stitch_tiles`` replays. Without the eager ``.clone()`` at the
    tile-driver returns, the first tiled ``decode()`` with >=2 temporal
    chunks raises ``accessing tensor output of CUDAGraphs that has been
    overwritten by a subsequent run``. This test needs >=2 chunks (decode),
    >=2 clips (encode), and a >=2x2 spatial tile grid.
    """
    torch.manual_seed(20260821)
    vae = _tiny_real_vae().to("cuda")
    vae.enable_tiling(16, 16, 4, 4)
    # The hook ComposedPipelineBase._maybe_compile_pipeline_module runs for
    # the opt-in; it installs the reduce-overhead compiled tile helpers.
    vae.prepare_for_compile()
    assert vae._tile_helpers_compiled

    # 8 latent tokens = 2 temporal chunks (tokens_chunk_size 5); 8x8 latents =
    # 32x32 pixels = a 2x2 grid of 16px tiles.
    z = torch.randn(1, 4, 8, 8, 8, device="cuda")
    pad_tokens, num_chunks, _ = vae._temporal_decode_plan(z.shape[2])
    assert num_chunks >= 2, "decode workload must span multiple stitch replays"

    decoded_first = vae.decode(z).sample
    decoded_second = vae.decode(z).sample
    assert torch.equal(decoded_first, decoded_second)

    # 34 frames = 2 encode clips of clip_length 17 -> 2 stitch replays.
    pixels = torch.rand(1, 3, 34, 32, 32, device="cuda")
    encoded_first = vae.encode(pixels).latent_dist.parameters
    encoded_second = vae.encode(pixels).latent_dist.parameters
    assert torch.equal(encoded_first, encoded_second)

    uint8_pixels = torch.randint(0, 256, (1, 3, 34, 32, 32), dtype=torch.uint8)
    streamed_first = vae.encode_pixels(uint8_pixels).latent_dist.parameters
    streamed_second = vae.encode_pixels(uint8_pixels).latent_dist.parameters
    assert torch.equal(streamed_first, streamed_second)

    # Output parity vs a default (fully eager, never-prepared) instance with
    # the same weights and tiling — the tolerance absorbs inductor fusion
    # reassociation only.
    eager_vae = _tiny_real_vae().to("cuda")
    eager_vae.load_state_dict(vae.state_dict())
    eager_vae.enable_tiling(16, 16, 4, 4)
    assert not eager_vae._tile_helpers_compiled
    torch.testing.assert_close(decoded_first, eager_vae.decode(z).sample, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(encoded_first, eager_vae.encode(pixels).latent_dist.parameters, atol=2e-4, rtol=2e-4)
