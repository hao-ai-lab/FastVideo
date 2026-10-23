# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel chunk scheduling for the MiniMax-H3 video VAE.

The H3 video VAE decodes a video as a series of temporal-chunk decoder
forwards whose outputs are joined by a short deterministic frame blend
(``AutoencoderKLMiniMaxH3._decode_chunks``), and encodes videos as fully
independent ``clip_length``-frame encoder forwards. Neither the chunk decode
nor the clip encode has any cross-chunk data dependency — only the *joining*
of decoded chunks (overlap blending, frame trimming) is sequential. This
module round-robins the chunk/clip forwards across the ranks of a
sequence-parallel group and replays the serial joining logic on the
assembling rank, reproducing the serial result bit for bit.

Bit-exactness contract:
- every rank holds an identical copy of the inputs (the H3 DiT all-gathers
  its outputs, and reference pixels are prepared identically on all ranks);
- a chunk decoded on any rank is bitwise the tensor the serial loop would
  produce (identical weights, inputs, and deterministic kernels on identical
  GPUs), and NCCL transports it bitwise;
- every serialization point of the serial algorithm (overlap blending, frame
  trimming, pixel denormalization, output-buffer copies, moment
  concatenation and token-drop trimming) runs on the assembling rank in
  serial order via the same VAE methods the serial path uses.

Collective safety: all group ranks must call these functions together with
identically shaped inputs. Work proceeds in rounds of one collective each;
ranks without a chunk in the final round contribute a placeholder tensor, so
participation is uniform by construction and no rank-dependent branch guards
a collective.

Caveat — compiled decoders (``enable_torch_compile_vae``): inductor autotunes
kernel configs per process at first call, so a compiled decoder is only
deterministic WITHIN a process, not across processes. Chunks decoded on other
ranks then differ from the serial rank's decode of the same chunk exactly as
two serial runs in different processes would. Direct decoder tensors measured
on GB200 at 124f had max absolute error 0.00268358 (0.684/255), mean absolute
error 4.213e-05 (0.0107/255), and 24.59% nonzero values; the first chunk was
bit-identical. A separate decoded-MP4 comparison reached 63/255 on <0.5% of
pixels, but that includes lossy MP4 encoding and is not the decoder-tensor
error envelope. With the eager decoder — the pipeline default — parallel
output is bitwise equal to serial ``decode_to_pixels``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import fastvideo.envs as envs
from fastvideo.models.vaes.minimax_h3_video import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLOutput,
    DiagonalGaussianDistribution,
    _tile_batch_size,
)
from fastvideo.profiler import nvtx_range

if TYPE_CHECKING:
    from fastvideo.distributed.parallel_state import GroupCoordinator

# Collective used to move decoded chunk segments to the assembling rank.
# "gather" moves each segment once (destination-only); "all_gather" also
# leaves every rank with every segment. Both are exact; the default is the
# faster one measured on GB200 NVL72 (see the PR notes).
DECODE_GATHER_STRATEGIES = ("gather", "all_gather")
DEFAULT_DECODE_GATHER_STRATEGY = "gather"


def parallel_chunk_indices(num_chunks: int, world_size: int, rank_in_group: int) -> list[int]:
    """Round-robin chunk ownership: chunk ``i`` belongs to rank ``i % world_size``."""
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be non-negative, got {num_chunks}.")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank_in_group < world_size:
        raise ValueError(f"rank_in_group {rank_in_group} out of range for world_size {world_size}.")
    return list(range(rank_in_group, num_chunks, world_size))


def _num_rounds(num_chunks: int, world_size: int) -> int:
    return -(-num_chunks // world_size)


def _decode_segment(vae: AutoencoderKLMiniMaxH3, z_padded: torch.Tensor, chunk_index: int) -> torch.Tensor:
    """Decode one temporal chunk's clip and keep the frames the join consumes.

    The serial loop uses two spans of each decoded clip: the chunk body
    ``clip[:, :, frame_pre_padding:chunk_num_frames]`` and (when
    ``token_drop > 0``) the blend tail
    ``clip[:, :, chunk_num_frames + frame_pre_padding:]``. Everything from
    ``frame_pre_padding`` on covers both, so one contiguous slice per chunk
    travels over the wire. ``.contiguous()`` also detaches the segment from
    any decoder-owned storage (e.g. a compiled decoder's reuse pools) before
    the next chunk decode can overwrite it.
    """
    start = chunk_index * vae.tokens_chunk_size
    with nvtx_range(f"minimax_h3.vae.parallel_chunk.{chunk_index}"):
        clip = vae._decode_clip(z_padded[:, :, start:start + vae.tokens_chunk_size + vae.token_overlap])
        return clip[:, :, vae.frame_pre_padding:].contiguous()


class _ChunkAssembler:
    """Replay the serial chunk-joining semantics of ``_decode_chunks`` +
    ``_decode_to_pixels`` on gathered chunk segments, in chunk order.

    On CUDA the joining kernels and output copies run on a dedicated side
    stream: they depend only on already-gathered segments, so running them
    off the main stream keeps the assembling rank's next chunk decode (and
    therefore every other rank's next collective) off the assembly's tail.
    Stream placement cannot change values — the ops and their order are
    identical — so bit-exactness with the serial path is unaffected.
    """

    def __init__(self, vae: AutoencoderKLMiniMaxH3, output: torch.Tensor, output_num_frames: int,
                 non_blocking: bool, device: torch.device) -> None:
        self._vae = vae
        self._output = output
        self._output_num_frames = output_num_frames
        self._non_blocking = non_blocking
        self._body_frames = vae.tokens_chunk_size * vae.temporal_compression_ratio - vae.frame_pre_padding
        self._overlap: torch.Tensor | None = None
        self._frame_start = 0
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def push(self, segment: torch.Tensor) -> None:
        """Consume the next chunk's segment (``clip[:, :, frame_pre_padding:]``)."""
        if self._stream is None:
            self._push(segment)
            return
        # The segment is produced on the current (collective) stream; hand it
        # to the assembly stream and pin its storage until assembly reads it.
        self._stream.wait_stream(torch.cuda.current_stream(segment.device))
        segment.record_stream(self._stream)
        with torch.cuda.stream(self._stream):
            self._push(segment)

    def _push(self, segment: torch.Tensor) -> None:
        vae = self._vae
        chunk = segment[:, :, :self._body_frames]
        if self._overlap is not None:
            chunk = vae._blend(self._overlap, chunk, vae.frame_overlap, dim=-3)
        num_frames = min(chunk.shape[2], self._output_num_frames - self._frame_start)
        chunk = chunk[:, :, :num_frames]
        # The tail past the body (and its pre-padding gap) is the next
        # chunk's blend overlap — the serial loop's ``next_overlap``.
        self._overlap = segment[:, :, self._body_frames + vae.frame_pre_padding:] if vae.config.token_drop > 0 else None
        if num_frames > 0:
            self._emit(chunk)

    def finalize(self) -> None:
        """Emit the final overlap tail exactly as the serial generator does."""
        if self._overlap is not None and self._frame_start < self._output_num_frames:
            tail = self._overlap[:, :, :self._output_num_frames - self._frame_start]
            if self._stream is None:
                self._emit(tail)
            else:
                with torch.cuda.stream(self._stream):
                    self._emit(tail)
        if self._frame_start != self._output.shape[2]:
            raise RuntimeError(
                f"MiniMax-H3 decode wrote {self._frame_start} frames into an output buffer expecting "
                f"{self._output.shape[2]}.")

    def synchronize(self) -> None:
        """Drain assembly kernels and output copies before the buffer is read."""
        if self._stream is not None:
            self._stream.synchronize()

    def _emit(self, chunk: torch.Tensor) -> None:
        pixels = self._vae.denormalize_pixels(chunk.float()).clamp_(0, 1)
        self._vae._copy_chunk_pixels(pixels, self._output, self._frame_start, self._non_blocking)
        self._frame_start += pixels.shape[2]


def _broadcast_segment_meta(group: "GroupCoordinator",
                            segment: torch.Tensor | None) -> tuple[torch.dtype, tuple[int, ...]]:
    """Share the leader's real segment dtype/shape so placeholder tensors match.

    The decoder's output dtype depends on the surrounding autocast context;
    deriving it on the leader from an actually decoded segment (instead of
    predicting it) keeps collective dtypes correct by construction.
    """
    meta = (segment.dtype, tuple(segment.shape)) if segment is not None else None
    meta = group.broadcast_object(meta, src=0)
    if meta is None:
        raise RuntimeError("MiniMax-H3 parallel VAE meta broadcast returned no leader metadata.")
    return meta


def decode_to_pixels_parallel(
    vae: AutoencoderKLMiniMaxH3,
    z: torch.Tensor,
    output: torch.Tensor | None,
    group: "GroupCoordinator",
    strategy: str = DEFAULT_DECODE_GATHER_STRATEGY,
) -> torch.Tensor | None:
    """Chunk-parallel ``decode_to_pixels`` across a sequence-parallel group.

    All group ranks call this together with identical ``z``. Temporal chunks
    are decoded round-robin across the group and their segments move to the
    group's first rank, which assembles bitwise the serial
    ``decode_to_pixels`` result into ``output``. Only the first rank passes
    ``output`` (validated exactly like the serial API); other ranks pass
    ``None`` and receive ``None``.
    """
    if strategy not in DECODE_GATHER_STRATEGIES:
        raise ValueError(f"Unknown parallel-decode strategy {strategy!r}; expected one of {DECODE_GATHER_STRATEGIES}.")
    is_leader = group.rank_in_group == 0
    if is_leader:
        if output is None:
            raise ValueError("The first sequence-parallel rank must provide the CPU output buffer.")
        expected_shape = vae.decoded_pixel_shape(z.shape)
        if output.device.type != "cpu" or output.dtype != torch.float32 or tuple(output.shape) != expected_shape:
            raise ValueError(
                "`output` must be a CPU float32 tensor with shape "
                f"{expected_shape}, got device={output.device}, dtype={output.dtype}, shape={tuple(output.shape)}.")
    elif output is not None:
        raise ValueError("Only the first sequence-parallel rank may provide an output buffer.")
    if group.world_size == 1:
        return vae.decode_to_pixels(z, output)

    try:
        if vae.use_slicing and z.shape[0] > 1:
            for batch_index, z_slice in enumerate(z.split(1)):
                slice_output = output[batch_index:batch_index + 1] if output is not None else None
                _decode_single_parallel(vae, z_slice, slice_output, group, strategy)
        else:
            _decode_single_parallel(vae, z, output, group, strategy)
    finally:
        # Drain the leader's async chunk copies before the caller (or an
        # exception handler) can read or release the pinned buffer.
        if output is not None and vae._streams_chunk_copies(z, output):
            torch.cuda.current_stream(z.device).synchronize()
    return output


# -- spatial-tile split (FASTVIDEO_H3_VAE_TILE_PARALLEL) ---------------------
#
# The chunk split above leaves ranks idle when a clip has few temporal chunks
# (a 124-frame 768p clip has fewer chunks than eight ranks hold work for) and
# cannot split a one-frame keyframe at all. With spatial tiling on, every
# chunk (or keyframe) is itself a grid of independent tile forwards that the
# serial path stitches afterwards. The tile split hands out (chunk, tile)
# forwards round-robin instead, gathers them, and stitches each chunk on the
# assembling rank with the serial ``_stitch_tiles`` before the unchanged chunk
# join, so the result is bitwise the serial one under the same contract as the
# chunk split.


def _tile_split_applies(vae: AutoencoderKLMiniMaxH3, x: torch.Tensor) -> bool:
    # The batched tile decode (FASTVIDEO_H3_VAE_TILE_BATCH) is a different
    # serial path; the tile split replays only the per-tile one.
    return (envs.FASTVIDEO_H3_VAE_TILE_PARALLEL.get() and vae.use_tiling and not vae._tile_helpers_compiled
            and x.shape[0] == 1 and _tile_batch_size() == 1 and not torch.is_grad_enabled())


def _tile_grid(vae: AutoencoderKLMiniMaxH3, height: int, width: int):
    """The serial tile split of a ``height`` x ``width`` pixel canvas: rows, columns, and their boxes."""
    rows = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    columns = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    boxes = [(y, h, x, w) for y, h in zip(rows[0], rows[1]) for x, w in zip(columns[0], columns[1])]
    return rows, columns, boxes


def _as_grid(tiles: list[torch.Tensor], num_columns: int) -> list[list[torch.Tensor]]:
    return [tiles[start:start + num_columns] for start in range(0, len(tiles), num_columns)]


def _tile_grid_is_exact(rows, columns, height: int, width: int, ratio: int) -> bool:
    """True when the grid tiles the canvas into equal shapes with ratio-aligned tile sizes.

    The gathers need equal tile shapes across ranks, and the decode slice must
    replay the serial ``y // ratio:y // ratio + h // ratio`` exactly. Grids whose
    last tile clamps (an unabsorbed ``_split_tiles`` residual) or whose tile size
    is not a multiple of the spatial ratio stand down to the serial path.
    """
    for starts, lengths, axis in ((rows[0], rows[1], height), (columns[0], columns[1], width)):
        if starts[-1] + lengths[-1] != axis:
            return False
        if len(lengths) > 1 and lengths[0] % ratio != 0:
            return False
    return True


def encode_keyframe_tile_parallel(vae: AutoencoderKLMiniMaxH3, x: torch.Tensor,
                                  group: "GroupCoordinator") -> torch.Tensor | None:
    """``vae._encode_clip(x)`` for a one-frame keyframe with its tiles split across ``group``.

    All group ranks call this together with identical ``x`` and all receive
    the moments. Returns ``None`` (on every rank alike) when the split does not
    apply; the caller then runs the serial encode.
    """
    if group.world_size == 1 or not _tile_split_applies(vae, x):
        return None
    rows, columns, boxes = _tile_grid(vae, *x.shape[-2:])
    if (len(boxes) < group.world_size
            or not _tile_grid_is_exact(rows, columns, *x.shape[-2:], vae.spatial_compression_ratio)):
        return None
    world_size, rank = group.world_size, group.rank_in_group
    tiles: list[torch.Tensor] = []
    placeholder = None
    for round_start in range(0, len(boxes), world_size):
        index = round_start + rank
        if index < len(boxes):
            y, h, x0, w = boxes[index]
            with nvtx_range(f"minimax_h3.vae.parallel_keyframe_tile.{index}"):
                mine = vae.quant_conv(vae.encoder(x[..., y:y + h, x0:x0 + w])).contiguous()
            placeholder = mine
        else:
            # Only the last round can be short, and every rank owned a tile in
            # round 0, so the placeholder shape is the tile shape of this grid.
            mine = torch.zeros_like(placeholder)
        gathered = group.all_gather(mine.unsqueeze(0), dim=0)
        tiles.extend(gathered[slot] for slot in range(min(world_size, len(boxes) - round_start)))
    ratio = vae.spatial_compression_ratio
    return vae._stitch_tiles(_as_grid(tiles, len(columns[0])), [o // ratio for o in rows[2]],
                             [o // ratio for o in columns[2]])


def _decode_single_tile_parallel(
    vae: AutoencoderKLMiniMaxH3,
    z: torch.Tensor,
    output: torch.Tensor | None,
    group: "GroupCoordinator",
    strategy: str,
) -> bool:
    """The tile split of ``_decode_single_parallel``; False (on every rank alike) when it does not apply."""
    if not _tile_split_applies(vae, z):
        return False
    pad_tokens, num_chunks, output_num_frames = vae._temporal_decode_plan(z.shape[2])
    ratio = vae.spatial_compression_ratio
    rows, columns, boxes = _tile_grid(vae, z.shape[-2] * ratio, z.shape[-1] * ratio)
    jobs = [(chunk, box) for chunk in range(num_chunks) for box in range(len(boxes))]
    world_size, rank = group.world_size, group.rank_in_group
    if (len(jobs) < world_size
            or not _tile_grid_is_exact(rows, columns, z.shape[-2] * ratio, z.shape[-1] * ratio, ratio)):
        return False
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    assembler = None
    if output is not None:
        assembler = _ChunkAssembler(vae, output, output_num_frames, vae._streams_chunk_copies(z, output), z.device)
    pending: dict[tuple[int, int], torch.Tensor] = {}
    next_chunk = 0
    placeholder = None
    try:
        for round_start in range(0, len(jobs), world_size):
            index = round_start + rank
            if index < len(jobs):
                chunk, box = jobs[index]
                start = chunk * vae.tokens_chunk_size
                y, h, x, w = boxes[box]
                tile = z[:, :, start:start + vae.tokens_chunk_size + vae.token_overlap, y // ratio:(y + h) // ratio,
                         x // ratio:(x + w) // ratio]
                with nvtx_range(f"minimax_h3.vae.parallel_tile.{chunk}.{box}"):
                    mine = vae.decoder(vae._project_decoder_tile(tile)).contiguous()
                placeholder = mine
            else:
                mine = torch.zeros_like(placeholder)
            with nvtx_range(f"minimax_h3.vae.parallel_tile_{strategy}.{round_start // world_size}"):
                if strategy == "gather":
                    gathered = group.gather(mine.unsqueeze(0), dst=0, dim=0)
                else:
                    gathered = group.all_gather(mine.unsqueeze(0), dim=0)
            if assembler is None or gathered is None:
                continue
            for slot in range(min(world_size, len(jobs) - round_start)):
                pending[jobs[round_start + slot]] = gathered[slot]
            while next_chunk < num_chunks and all((next_chunk, box) in pending for box in range(len(boxes))):
                tiles = [pending.pop((next_chunk, box)) for box in range(len(boxes))]
                clip = vae._stitch_tiles(_as_grid(tiles, len(columns[0])), rows[2], columns[2])
                assembler.push(clip[:, :, vae.frame_pre_padding:].contiguous())
                next_chunk += 1
        if assembler is not None:
            assembler.finalize()
    finally:
        if assembler is not None:
            assembler.synchronize()
    return True


def _decode_single_parallel(
    vae: AutoencoderKLMiniMaxH3,
    z: torch.Tensor,
    output: torch.Tensor | None,
    group: "GroupCoordinator",
    strategy: str,
) -> None:
    if _decode_single_tile_parallel(vae, z, output, group, strategy):
        return
    pad_tokens, num_chunks, output_num_frames = vae._temporal_decode_plan(z.shape[2])
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    world_size = group.world_size
    rank = group.rank_in_group

    # Every rank decodes its round-0 chunk BEFORE the metadata rendezvous so
    # the first decodes run concurrently (a rank that waited on the broadcast
    # first would idle a full chunk-decode behind the leader). The leader
    # owns chunk 0 under round-robin assignment, so its segment supplies real
    # dtype/shape for placeholder rounds instead of guessing autocast state.
    first_segment = _decode_segment(vae, z, rank) if rank < num_chunks else None
    segment_dtype, segment_shape = _broadcast_segment_meta(group, first_segment if rank == 0 else None)

    assembler = None
    if output is not None:
        non_blocking = vae._streams_chunk_copies(z, output)
        assembler = _ChunkAssembler(vae, output, output_num_frames, non_blocking, z.device)

    try:
        segment_frames = segment_shape[2]
        for round_index in range(_num_rounds(num_chunks, world_size)):
            chunk_index = round_index * world_size + rank
            if chunk_index >= num_chunks:
                segment = torch.zeros(segment_shape, dtype=segment_dtype, device=z.device)
            elif round_index == 0 and first_segment is not None:
                segment = first_segment
            else:
                segment = _decode_segment(vae, z, chunk_index)
            with nvtx_range(f"minimax_h3.vae.parallel_{strategy}.{round_index}"):
                if strategy == "gather":
                    gathered = group.gather(segment, dst=0, dim=2)
                else:
                    gathered = group.all_gather(segment, dim=2)
            if assembler is None or gathered is None:
                continue
            for slot in range(world_size):
                if round_index * world_size + slot >= num_chunks:
                    break
                assembler.push(gathered.narrow(2, slot * segment_frames, segment_frames))
        if assembler is not None:
            assembler.finalize()
    finally:
        # Drain assembly-stream copies into ``output`` even on the error path
        # so an exception cannot leave an in-flight DMA into a buffer the
        # caller may release.
        if assembler is not None:
            assembler.synchronize()


def _encode_clip_moments(vae: AutoencoderKLMiniMaxH3, pixels: torch.Tensor, clip_index: int) -> torch.Tensor:
    """Encode one ``clip_length``-frame clip exactly as ``_encode_pixels`` does."""
    clip_length = vae.config.clip_length
    frame_start = clip_index * clip_length
    with nvtx_range(f"minimax_h3.vae.parallel_encode_clip.{clip_index}"):
        clip = pixels[:, :, frame_start:frame_start + clip_length].to(
            device=vae.pixel_mean.device,
            dtype=torch.float32,
        )
        if pixels.dtype == torch.uint8:
            clip = clip / 255.0
        if clip.shape[2] < clip_length:
            pad_frames = clip[:, :, -1:].repeat(1, 1, clip_length - clip.shape[2], 1, 1)
            clip = torch.cat([clip, pad_frames], dim=2)
        clip = vae.normalize_pixels(clip)
        return vae._encode_clip(clip).contiguous()


def encode_pixels_parallel(
    vae: AutoencoderKLMiniMaxH3,
    pixels: torch.Tensor,
    group: "GroupCoordinator",
) -> AutoencoderKLOutput:
    """Clip-parallel ``encode_pixels`` across a sequence-parallel group.

    Encoder clips have no cross-clip dependency (no overlap, no blending), so
    ranks encode disjoint clips and all-gather the per-clip moment tensors.
    Every rank returns the identical full posterior — preserving the serial
    contract that all ranks hold the same encoded latents — bitwise equal to
    ``vae.encode_pixels(pixels)``. Moments are latent-sized (a few MB per
    clip), so the all-gather is negligible next to the clip forwards.
    """
    if pixels.ndim != 5 or pixels.shape[1] != vae.config.in_channels or pixels.shape[2] <= 0:
        raise ValueError(
            f"`pixels` must have shape [B, {vae.config.in_channels}, T, H, W] with T > 0, "
            f"got {tuple(pixels.shape)}.")
    if pixels.device.type != "cpu":
        raise ValueError(f"`pixels` must remain on CPU, got device={pixels.device}.")
    if pixels.dtype != torch.uint8 and not pixels.is_floating_point():
        raise TypeError(f"`pixels` must use uint8 or a floating-point dtype, got {pixels.dtype}.")
    if group.world_size == 1:
        return vae.encode_pixels(pixels)
    if vae.use_slicing and pixels.shape[0] > 1:
        moments = torch.cat([_encode_single_parallel(vae, pixel_slice, group) for pixel_slice in pixels.split(1)])
    else:
        moments = _encode_single_parallel(vae, pixels, group)
    return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(moments))


def _encode_single_parallel(vae: AutoencoderKLMiniMaxH3, pixels: torch.Tensor,
                            group: "GroupCoordinator") -> torch.Tensor:
    clip_length = vae.config.clip_length
    num_clips = -(-pixels.shape[2] // clip_length)
    world_size = group.world_size
    rank = group.rank_in_group

    # Same first-work-then-rendezvous ordering as the decode path: encode the
    # round-0 clip before the metadata broadcast so first encodes overlap.
    first_moments = _encode_clip_moments(vae, pixels, rank) if rank < num_clips else None
    moment_dtype, moment_shape = _broadcast_segment_meta(group, first_moments if rank == 0 else None)

    moment_tokens = moment_shape[2]
    parts: list[torch.Tensor] = []
    for round_index in range(_num_rounds(num_clips, world_size)):
        clip_index = round_index * world_size + rank
        if clip_index >= num_clips:
            moments = torch.zeros(moment_shape, dtype=moment_dtype, device=vae.pixel_mean.device)
        elif round_index == 0 and first_moments is not None:
            moments = first_moments
        else:
            moments = _encode_clip_moments(vae, pixels, clip_index)
        gathered = group.all_gather(moments, dim=2)
        for slot in range(world_size):
            if round_index * world_size + slot >= num_clips:
                break
            parts.append(gathered.narrow(2, slot * moment_tokens, moment_tokens))
    encoded = torch.cat(parts, dim=2)
    if vae.config.token_drop > 0:
        encoded = encoded[:, :, :-vae.config.token_drop]
    return encoded


__all__ = [
    "DECODE_GATHER_STRATEGIES",
    "DEFAULT_DECODE_GATHER_STRATEGY",
    "decode_to_pixels_parallel",
    "encode_pixels_parallel",
    "parallel_chunk_indices",
]
