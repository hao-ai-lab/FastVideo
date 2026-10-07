# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sequence-parallel MiniMax-H3 VAE chunk decode / clip encode.

The collective transport is simulated with a threaded fake group (one thread
per simulated rank, barrier-synchronized slots), so the REAL drivers in
``fastvideo.models.vaes.minimax_h3_parallel`` — chunk assignment, placeholder
rounds, metadata broadcast, gathered-segment assembly, halo/blend math — run
end to end on CPU and are checked bit-exactly against the serial APIs.
"""

import threading

import pytest
import torch
from torch.testing import assert_close

import fastvideo.envs as envs
from fastvideo.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from fastvideo.models.vaes import minimax_h3_parallel
from fastvideo.models.vaes.minimax_h3_parallel import (
    DECODE_GATHER_STRATEGIES,
    DEFAULT_DECODE_GATHER_STRATEGY,
    decode_to_pixels_parallel,
    encode_pixels_parallel,
    parallel_chunk_indices,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3


def _tiny_vae(token_drop: int = 3) -> AutoencoderKLMiniMaxH3:
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
        token_drop=token_drop,
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


class _ThreadedFakeGroup:
    """Barrier-synchronized in-process stand-in for a GroupCoordinator.

    One thread per simulated rank runs the SPMD driver; ``gather`` /
    ``all_gather`` / ``broadcast_object`` rendezvous through shared slots
    with a double barrier (all writes land, everyone reads, then slots are
    reusable). Matches the GroupCoordinator call signatures the drivers use.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._local = threading.local()
        self._barrier = threading.Barrier(world_size)
        self._slots: list = [None] * world_size
        self._object = None

    @property
    def rank_in_group(self) -> int:
        return self._local.rank

    def broadcast_object(self, obj=None, src: int = 0):
        if self.world_size == 1:
            return obj
        if self.rank_in_group == src:
            self._object = obj
        self._barrier.wait()
        received = self._object
        self._barrier.wait()
        return received

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        self._slots[self.rank_in_group] = input_
        self._barrier.wait()
        gathered = torch.cat([slot for slot in self._slots], dim=dim)
        self._barrier.wait()
        return gathered

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1):
        if self.world_size == 1:
            return input_
        self._slots[self.rank_in_group] = input_
        self._barrier.wait()
        gathered = torch.cat([slot for slot in self._slots], dim=dim) if self.rank_in_group == dst else None
        self._barrier.wait()
        return gathered

    def run(self, fn) -> list:
        """Run ``fn(rank)`` on one thread per rank; re-raise the first error."""
        results: list = [None] * self.world_size
        errors: list = [None] * self.world_size

        def _target(rank: int) -> None:
            self._local.rank = rank
            try:
                # inference_mode is thread-local; the drivers run inference-only.
                with torch.inference_mode():
                    results[rank] = fn(rank)
            except BaseException as error:  # noqa: BLE001 - propagate to the test
                errors[rank] = error
                self._barrier.abort()

        threads = [threading.Thread(target=_target, args=(rank, )) for rank in range(self.world_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None and not isinstance(error, threading.BrokenBarrierError):
                raise error
        for error in errors:
            if error is not None:
                raise error
        return results


@pytest.mark.parametrize("num_chunks,world_size", ((0, 4), (1, 4), (7, 4), (8, 4), (20, 4), (5, 3), (2, 5)))
def test_parallel_chunk_indices_partition(num_chunks: int, world_size: int) -> None:
    """Round-robin ownership covers every chunk exactly once, in order."""
    owned = [parallel_chunk_indices(num_chunks, world_size, rank) for rank in range(world_size)]
    flattened = sorted(index for indices in owned for index in indices)
    assert flattened == list(range(num_chunks))
    for rank, indices in enumerate(owned):
        assert indices == sorted(indices)
        assert all(index % world_size == rank for index in indices)
        # Round-robin balance: no rank holds more than one extra chunk.
        assert len(indices) in (num_chunks // world_size, -(-num_chunks // world_size))


def test_parallel_chunk_indices_validates() -> None:
    with pytest.raises(ValueError, match="world_size"):
        parallel_chunk_indices(4, 0, 0)
    with pytest.raises(ValueError, match="rank_in_group"):
        parallel_chunk_indices(4, 2, 2)
    with pytest.raises(ValueError, match="num_chunks"):
        parallel_chunk_indices(-1, 2, 0)


# Latent frames cover: one padded chunk (3), pad on the intra-clip tail (6),
# two blended chunks (12), three chunks plus pad trim (13). World sizes cover
# fewer chunks than ranks, uneven rounds, and the exact-multiple case.
@pytest.mark.parametrize("world_size", (2, 3, 4, 5))
@pytest.mark.parametrize("latent_frames", (3, 6, 12, 13))
@pytest.mark.parametrize("strategy", DECODE_GATHER_STRATEGIES)
@torch.inference_mode()
def test_parallel_decode_matches_serial(world_size: int, latent_frames: int, strategy: str) -> None:
    torch.manual_seed(20260821 + latent_frames)
    vae = _tiny_vae()
    latents = torch.randn(1, 4, latent_frames, 4, 4)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    group = _ThreadedFakeGroup(world_size)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32) if rank == 0 else None
        return decode_to_pixels_parallel(vae, latents.clone(), output, group, strategy=strategy)

    results = group.run(_rank_main)
    assert all(result is None for result in results[1:])
    assert_close(results[0], expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_parallel_decode_without_token_drop() -> None:
    """token_drop == 0 has no overlap halo; the assembler must skip blending."""
    torch.manual_seed(20260822)
    vae = _tiny_vae(token_drop=0)
    assert vae.frame_overlap == 0
    latents = torch.randn(1, 4, 10, 4, 4)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    group = _ThreadedFakeGroup(3)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32) if rank == 0 else None
        return decode_to_pixels_parallel(vae, latents.clone(), output, group)

    results = group.run(_rank_main)
    assert_close(results[0], expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_parallel_decode_batched_slicing_matches_serial() -> None:
    torch.manual_seed(20260823)
    vae = _tiny_vae()
    vae.enable_slicing()
    latents = torch.randn(2, 4, 7, 4, 4)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    group = _ThreadedFakeGroup(2)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32) if rank == 0 else None
        return decode_to_pixels_parallel(vae, latents.clone(), output, group)

    results = group.run(_rank_main)
    assert_close(results[0], expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_parallel_decode_world_size_one_is_serial() -> None:
    torch.manual_seed(20260824)
    vae = _tiny_vae()
    latents = torch.randn(1, 4, 7, 4, 4)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    group = _ThreadedFakeGroup(1)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
        return decode_to_pixels_parallel(vae, latents, output, group)

    results = group.run(_rank_main)
    assert_close(results[0], expected, atol=0.0, rtol=0.0)


def test_parallel_decode_validates_buffers_and_strategy() -> None:
    vae = _tiny_vae()
    latents = torch.randn(1, 4, 7, 4, 4)
    output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    group = _ThreadedFakeGroup(1)
    group._local.rank = 0

    with pytest.raises(ValueError, match="strategy"):
        decode_to_pixels_parallel(vae, latents, output, group, strategy="scatter")
    with pytest.raises(ValueError, match="must provide the CPU output buffer"):
        decode_to_pixels_parallel(vae, latents, None, group)
    with pytest.raises(ValueError, match="CPU float32 tensor"):
        decode_to_pixels_parallel(vae, latents, output[:, :, :-1], group)

    group._local.rank = 1  # simulate a non-leader passing a buffer
    group.world_size = 2
    with pytest.raises(ValueError, match="Only the first sequence-parallel rank"):
        decode_to_pixels_parallel(vae, latents, output, group)


@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.parametrize("num_frames", (16, 22, 40))
@torch.inference_mode()
def test_parallel_encode_matches_serial(world_size: int, num_frames: int) -> None:
    """Every rank must hold the full serial moments, bit for bit."""
    torch.manual_seed(20260825 + num_frames)
    vae = _tiny_vae()
    pixels = torch.randint(0, 256, (1, 3, num_frames, 16, 16), dtype=torch.uint8)
    expected = vae.encode_pixels(pixels).latent_dist.parameters

    group = _ThreadedFakeGroup(world_size)
    results = group.run(lambda rank: encode_pixels_parallel(vae, pixels, group).latent_dist.parameters)
    for moments in results:
        assert_close(moments, expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_parallel_encode_float_and_batched_slicing() -> None:
    torch.manual_seed(20260826)
    vae = _tiny_vae()
    vae.enable_slicing()
    pixels = torch.rand(2, 3, 22, 16, 16)
    expected = vae.encode_pixels(pixels).latent_dist.parameters

    group = _ThreadedFakeGroup(3)
    results = group.run(lambda rank: encode_pixels_parallel(vae, pixels, group).latent_dist.parameters)
    for moments in results:
        assert_close(moments, expected, atol=0.0, rtol=0.0)


def test_parallel_encode_validates_input() -> None:
    vae = _tiny_vae()
    group = _ThreadedFakeGroup(1)
    group._local.rank = 0
    with pytest.raises(ValueError, match="must remain on CPU"):
        encode_pixels_parallel(vae, torch.empty(1, 3, 4, 16, 16, device="meta"), group)
    with pytest.raises(TypeError, match="uint8 or a floating-point"):
        encode_pixels_parallel(vae, torch.zeros(1, 3, 4, 16, 16, dtype=torch.int32), group)
    with pytest.raises(ValueError, match="must have shape"):
        encode_pixels_parallel(vae, torch.zeros(1, 4, 4, 16, 16), group)


def test_fastvideo_args_strategy_literals_match_module() -> None:
    """fastvideo_args mirrors the strategy literals to avoid importing model
    modules at args construction; keep the two in sync."""
    from fastvideo.fastvideo_args import FastVideoArgs

    args = FastVideoArgs(model_path="test/parallel-vae")
    assert args.vae_parallel_decode is False
    assert args.vae_parallel_encode is False
    assert args.vae_parallel_decode_strategy == DEFAULT_DECODE_GATHER_STRATEGY
    assert args.vae_parallel_decode_strategy in DECODE_GATHER_STRATEGIES

    for strategy in DECODE_GATHER_STRATEGIES:
        assert FastVideoArgs(model_path="test/parallel-vae",
                             vae_parallel_decode_strategy=strategy).vae_parallel_decode_strategy == strategy
    with pytest.raises(ValueError, match="vae_parallel_decode_strategy"):
        FastVideoArgs(model_path="test/parallel-vae", vae_parallel_decode_strategy="scatter")


@pytest.fixture
def tile_parallel():
    with envs.FASTVIDEO_H3_VAE_TILE_PARALLEL.override(True):
        yield


def _chunk_split_must_not_run(*args, **kwargs):
    raise AssertionError("the tile split fell back to the chunk split")


def _tiled_vae(token_drop: int = 3) -> AutoencoderKLMiniMaxH3:
    vae = _tiny_vae(token_drop=token_drop)
    # 32x48 pixels split into a 3x5 grid of 16x16 tiles with 8-pixel overlaps.
    vae.enable_tiling(tile_sample_min_height=16,
                      tile_sample_min_width=16,
                      tile_sample_min_overlap_height=8,
                      tile_sample_min_overlap_width=8)
    return vae


# 3 latent frames is one chunk (fewer chunks than ranks: only the tile split
# keeps every rank busy); 13 is three chunks with pad trim and blending.
@pytest.mark.parametrize("world_size", (2, 4, 7))
@pytest.mark.parametrize("latent_frames", (3, 13))
@pytest.mark.parametrize("strategy", DECODE_GATHER_STRATEGIES)
@torch.inference_mode()
def test_tile_parallel_decode_matches_serial(monkeypatch, tile_parallel, world_size: int, latent_frames: int,
                                             strategy: str) -> None:
    torch.manual_seed(20261002 + latent_frames)
    vae = _tiled_vae()
    latents = torch.randn(1, 4, latent_frames, 8, 12)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    monkeypatch.setattr(minimax_h3_parallel, "_decode_segment", _chunk_split_must_not_run)
    group = _ThreadedFakeGroup(world_size)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32) if rank == 0 else None
        return decode_to_pixels_parallel(vae, latents.clone(), output, group, strategy=strategy)

    results = group.run(_rank_main)
    assert all(result is None for result in results[1:])
    assert_close(results[0], expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_tile_parallel_decode_falls_back_with_fewer_tiles_than_ranks(monkeypatch, tile_parallel) -> None:
    """One chunk of a 1x1 tile grid on 3 ranks: the chunk split serves it unchanged."""
    torch.manual_seed(20261003)
    vae = _tiny_vae()
    vae.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16)
    latents = torch.randn(1, 4, 3, 4, 4)
    expected = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32)
    vae.decode_to_pixels(latents, expected)

    group = _ThreadedFakeGroup(3)

    def _rank_main(rank: int):
        output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32) if rank == 0 else None
        return decode_to_pixels_parallel(vae, latents.clone(), output, group)

    assert_close(group.run(_rank_main)[0], expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("world_size", (2, 4, 15, 16))
@torch.inference_mode()
def test_tile_parallel_keyframe_encode_matches_serial(monkeypatch, tile_parallel, world_size: int) -> None:
    """Every rank gets the serial keyframe moments; a grid smaller than the group refuses (None)."""
    from fastvideo.models.vaes.minimax_h3_parallel import encode_keyframe_tile_parallel

    torch.manual_seed(20261004 + world_size)
    vae = _tiled_vae()
    keyframe = torch.rand(1, 3, 1, 32, 48) * 2 - 1
    expected = vae._encode_clip(keyframe)

    group = _ThreadedFakeGroup(world_size)
    results = group.run(lambda rank: encode_keyframe_tile_parallel(vae, keyframe, group))
    if world_size > 15:
        assert all(result is None for result in results)
        return
    for moments in results:
        assert_close(moments, expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_tile_parallel_is_off_by_default() -> None:
    from fastvideo.models.vaes.minimax_h3_parallel import encode_keyframe_tile_parallel

    with envs.FASTVIDEO_H3_VAE_TILE_PARALLEL.override(None):
        vae = _tiled_vae()
        group = _ThreadedFakeGroup(2)
        group._local.rank = 0
        assert encode_keyframe_tile_parallel(vae, torch.zeros(1, 3, 1, 32, 48), group) is None
        assert vae._encode_keyframe_tile_parallel(torch.zeros(1, 3, 1, 32, 48)) is None


@torch.inference_mode()
def test_tile_parallel_stands_down_on_a_clamped_grid(tile_parallel) -> None:
    """A grid whose last tile clamps cannot be gathered; the serial encode serves it."""
    from fastvideo.models.vaes.minimax_h3_parallel import encode_keyframe_tile_parallel

    vae = _tiny_vae()
    vae.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16, tile_sample_min_overlap_height=8,
                      tile_sample_min_overlap_width=8)
    group = _ThreadedFakeGroup(2)
    group._local.rank = 0
    # 31 px split into 16/16/15: the clamped tile breaks the equal-shape gather.
    assert encode_keyframe_tile_parallel(vae, torch.zeros(1, 3, 1, 31, 48), group) is None


@torch.inference_mode()
def test_encode_keyframe_routes_through_the_tile_split(monkeypatch, tile_parallel) -> None:
    import fastvideo.distributed as distributed

    torch.manual_seed(20261005)
    vae = _tiled_vae()
    keyframe = torch.rand(1, 3, 1, 32, 48) * 2 - 1
    expected = vae.encode_keyframe(keyframe).latent_dist.parameters

    group = _ThreadedFakeGroup(4)
    monkeypatch.setattr(distributed, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_sp_group", lambda: group)
    monkeypatch.setattr(vae, "_encode_clip", _chunk_split_must_not_run)
    results = group.run(lambda rank: vae.encode_keyframe(keyframe).latent_dist.parameters)
    for moments in results:
        assert_close(moments, expected, atol=0.0, rtol=0.0)
