# Copyright (c) 2026 SandAI. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack MAGI-2 preview video, audio, text, and image-conditioning tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from typing import Any, Literal

import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn import functional as F

from fastvideo.models.dits.magi2_runtime.psm import psm


class Modality(IntEnum):
    """Modality identifiers consumed by the MAGI-2 preview transformer."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    TIME = 3


@dataclass
class VarlenHandler:
    """Variable-length attention boundaries for a packed token sequence."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class Magi2DataProxyConfig:
    """Configuration that controls MAGI-2 preview token packing."""

    t_patch_size: int = 1
    patch_size: int = 1
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    add_time_token: bool = False
    time_channel_dim: int = 64
    time_aligned_rope: bool = False
    audio_latent_fps: float = 25.0
    time_pos_fps: float = 3.125
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0


@dataclass
class ModelInput:
    """Multimodal tensors and lengths consumed by ``Magi2DataProxy``."""

    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]
    t: torch.Tensor
    ref_audio_feat: torch.Tensor | None = None
    ref_audio_feat_len: torch.Tensor | list[int] | None = None
    ref_video_feat: torch.Tensor | None = None
    ref_video_feat_len: torch.Tensor | list[int] | None = None
    per_token_video_t: torch.Tensor | None = None
    per_token_audio_t: torch.Tensor | None = None
    ref_image_feat: torch.Tensor | None = None
    ref_image_feat_len: torch.Tensor | None = None
    ref_image_special_token_embedding: torch.Tensor | None = None


def _to_int(value: int | torch.Tensor) -> int:
    """Convert a scalar tensor or Python integer into an integer."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _pad_cat(
    tensors: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Right-pad feature channels and concatenate token segments."""
    if not tensors:
        return torch.empty(0, 0, device=device, dtype=dtype)
    max_channel = max(tensor.shape[-1] for tensor in tensors)
    return torch.cat(
        [
            F.pad(
                tensor.to(device=device, dtype=dtype),
                (0, max_channel - tensor.shape[-1]),
            )
            for tensor in tensors
        ],
        dim=0,
    )


def _segment_paint(
    values: list[int],
    seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
    output_size: int,
) -> torch.Tensor:
    """Expand one modality identifier over each packed token segment."""
    mapping = torch.empty(output_size, dtype=dtype, device=device)
    offset = 0
    for value, seqlen in zip(values, seqlens, strict=True):
        mapping[offset:offset + seqlen] = int(value)
        offset += seqlen
    return mapping


def _seqlens2cu_seqlens(
    seqlens: list[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert sequence lengths into int32 cumulative boundaries."""
    seqlens_tensor = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return F.pad(torch.cumsum(seqlens_tensor, dim=0), (1, 0))


def _ceil_div(dividend: int, divisor: int) -> int:
    """Return integer ceiling division."""
    return (dividend + divisor - 1) // divisor


def _len_to_list(value: torch.Tensor) -> list[int]:
    """Convert one packed reference-image grid shape into Python integers."""
    return [int(entry) for entry in value.detach().to(torch.long).reshape(-1).tolist()]


def _get_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    time_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build token coordinates with source-grid and reference-grid metadata."""
    if device is None:
        device = torch.device("cpu")
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw
    if time_positions is None:
        time_range = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    else:
        time_range = time_positions.to(device=device, dtype=dtype) + offset_t
    height_range = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_range = torch.arange(ori_w, device=device, dtype=dtype) + offset_w
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_range,
        height_range,
        width_range,
        indexing="ij",
    )
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)
    metadata = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w],
        device=device,
        dtype=dtype,
    )
    return torch.cat([coords_flat, metadata.expand(coords_flat.size(0), -1)], dim=-1)


def _sinusoidal_embedding_1d(
    dim: int,
    position: torch.Tensor,
) -> torch.Tensor:
    """Encode normalized diffusion times into cosine-sine feature channels."""
    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            start=0,
            end=half,
            dtype=torch.float32,
            device=position.device,
        )
        / half
    )
    arguments = position[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )
    return embedding


@dataclass
class SingleData:
    """One sample's token segments and packing metadata."""

    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    diffusion_t: torch.Tensor | None = None
    per_token_video_t: torch.Tensor | None = None
    per_token_audio_t: torch.Tensor | None = None
    time_channel_dim: int = 0
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0
    time_pos_fps: float = 3.125
    ref_image_feats: list[torch.Tensor] | None = None
    ref_image_feat_lens: list[list[int]] | None = None
    ref_image_special_tokens: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Trim variable-length segments and derive immutable packing sizes."""
        self.video_token_num = self.video_x_t.shape[0]
        self.origin_audio_feat_len = self.audio_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[:self.audio_feat_len]
        self.txt_feat = self.txt_feat[:self.txt_feat_len]
        if self.per_token_audio_t is not None:
            self.per_token_audio_t = self.per_token_audio_t[:self.audio_feat_len]

        self.ref_image_feats = self.ref_image_feats or []
        self.ref_image_feat_lens = self.ref_image_feat_lens or []
        self.ref_image_special_tokens = self.ref_image_special_tokens or []
        self.ref_image_token_nums = [
            int(math.prod(feat_len)) for feat_len in self.ref_image_feat_lens
        ]
        self.ref_image_feats = [
            feat[:token_num]
            for feat, token_num in zip(
                self.ref_image_feats,
                self.ref_image_token_nums,
                strict=True,
            )
        ]
        self.num_ref_images = len(self.ref_image_feats)
        self.total_ref_image_feat_len = sum(self.ref_image_token_nums)
        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]

    @property
    def device(self) -> torch.device:
        """Return the device that owns this sample's packed tokens."""
        return self.video_x_t.device

    @property
    def default_dtype(self) -> torch.dtype:
        """Return the video-token dtype used for the packed sequence."""
        return self.video_x_t.dtype

    @property
    def add_time_token(self) -> bool:
        """Return whether this sample appends a standalone time token."""
        return self.diffusion_t is not None

    @property
    def total_token_num(self) -> int:
        """Return the complete token count for one sample."""
        total = self.video_token_num + self.audio_feat_len + self.txt_feat_len
        total += self.total_ref_image_feat_len + self.num_ref_images
        return total + (1 if self.add_time_token else 0)

    @property
    def feat_to_cat(self) -> list[torch.Tensor]:
        """Return token segments in transformer consumption order."""
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        for image_index in range(self.num_ref_images):
            tensors.append(self.ref_image_special_tokens[image_index])
            tensors.append(self.ref_image_feats[image_index])
        if self.add_time_token:
            assert self.diffusion_t is not None
            tensors.append(
                self.diffusion_t.to(
                    device=self.device,
                    dtype=self.default_dtype,
                ).reshape(1, 1)
            )
        return tensors

    @property
    def token_sequence(self) -> torch.Tensor:
        """Return channel-padded tokens for one sample."""
        return _pad_cat(self.feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_map_seqlens(self) -> tuple[list[int], list[int]]:
        """Return segment lengths and modality identifiers in token order."""
        seqlens = [self.video_token_num, self.audio_feat_len, self.txt_feat_len]
        modalities: list[int] = [
            int(Modality.VIDEO),
            int(Modality.AUDIO),
            int(Modality.TEXT),
        ]
        for image_index in range(self.num_ref_images):
            seqlens.append(1)
            modalities.append(int(Modality.TEXT))
            seqlens.append(self.ref_image_token_nums[image_index])
            modalities.append(int(Modality.VIDEO))
        if self.add_time_token:
            seqlens.append(1)
            modalities.append(int(Modality.TIME))
        return seqlens, modalities

    @property
    def modality_mapping(self) -> torch.Tensor:
        """Expand segment modality identifiers to one value per token."""
        seqlens, modalities = self.modality_map_seqlens
        return _segment_paint(
            modalities,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    def _default_coords(
        self,
        shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        offset_thw: tuple[int, int, int] = (0, 0, 0),
        time_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build coordinate rows on this sample's device and dtype."""
        return _get_coords(
            shape,
            ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
            time_positions=time_positions,
        )

    @property
    def coords_to_cat(self) -> list[torch.Tensor]:
        """Return coordinate segments in the same order as feature segments."""
        t_steps = self.t // self.t_patch_size
        h_steps = self.h // self.patch_size
        w_steps = self.w // self.patch_size
        if self.spatial_rope_interpolation == "inter":
            video_h_ref, video_w_ref = 32, 32
        else:
            video_h_ref, video_w_ref = h_steps, w_steps

        video_coords = self._default_coords(
            (t_steps, h_steps, w_steps),
            (t_steps, video_h_ref, video_w_ref),
        )
        magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self._default_coords(
            (self.audio_feat_len, 1, 1),
            (magic_audio_ref_t // self.t_patch_size, 1, 1),
        )
        coords = [
            video_coords,
            audio_coords,
            self._default_coords(
                (self.txt_feat_len, 1, 1),
                (1, 1, 1),
                offset_thw=(-self.txt_feat_len, 0, 0),
            ),
        ]

        for image_index in range(self.num_ref_images):
            token_len = self.ref_image_token_nums[image_index]
            if len(self.ref_image_feat_lens[image_index]) >= 2:
                image_h = int(self.ref_image_feat_lens[image_index][0])
                image_w = int(self.ref_image_feat_lens[image_index][1])
            else:
                image_h = image_w = int(math.ceil(math.sqrt(token_len)))
            time_offset = t_steps + 2 + image_index
            coords.append(
                torch.tensor(
                    [
                        [
                            time_offset,
                            -1,
                            -1,
                            1,
                            image_h,
                            image_w,
                            1,
                            image_h,
                            image_w,
                        ]
                    ],
                    device=self.device,
                    dtype=self.default_dtype,
                )
            )
            coords.append(
                self._default_coords(
                    (1, image_h, image_w),
                    (1, image_h, image_w),
                    offset_thw=(time_offset, 0, 0),
                )[:token_len]
            )

        if self.add_time_token:
            coords.append(self._default_coords((1, 1, 1), (1, 1, 1))[:1])
        return coords

    @property
    def coords_mapping(self) -> torch.Tensor:
        """Return one nine-value coordinate row per token."""
        return torch.cat(self.coords_to_cat, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        """Return per-token diffusion-time features in feature-segment order."""
        if self.time_channel_dim == 0:
            return torch.empty(self.total_token_num, 0, device=self.device)
        assert self.per_token_video_t is not None
        assert self.per_token_audio_t is not None
        time_parts = [
            self.per_token_video_t.squeeze(-1),
            self.per_token_audio_t.squeeze(-1),
            torch.zeros(self.txt_feat_len, device=self.device),
        ]
        for image_index in range(self.num_ref_images):
            time_parts.append(torch.zeros(1, device=self.device))
            time_parts.append(
                torch.zeros(
                    self.ref_image_token_nums[image_index],
                    device=self.device,
                )
            )
        if self.add_time_token:
            assert self.diffusion_t is not None
            time_parts.append(self.diffusion_t.reshape(1).to(self.device))
        raw_time = torch.cat(time_parts, dim=0)
        if self.time_channel_dim == 1:
            return raw_time.unsqueeze(-1)
        return _sinusoidal_embedding_1d(self.time_channel_dim, raw_time)


@dataclass
class SimplePackedData:
    """A batch represented as adjacent variable-length sample sequences."""

    items: list[SingleData]

    def __post_init__(self) -> None:
        """Cache token-length summaries shared by attention and depacking."""
        if len(self.items) == 0:
            raise ValueError("SimplePackedData must contain at least one item.")
        self._total_token_num_list = [item.total_token_num for item in self.items]
        self._total_token_num_sum = sum(self._total_token_num_list)
        self._total_token_num_max = max(self._total_token_num_list)
        self._total_token_cu_seqlens = _seqlens2cu_seqlens(
            self._total_token_num_list
        )

    @property
    def device(self) -> torch.device:
        """Return the device used by every packed sample."""
        return self.items[0].device

    @property
    def default_dtype(self) -> torch.dtype:
        """Return the token dtype used by every packed sample."""
        return self.items[0].default_dtype

    @property
    def token_sequence(self) -> torch.Tensor:
        """Return every sample's feature segments as one packed tensor."""
        feature_segments = list(
            chain.from_iterable(item.feat_to_cat for item in self.items)
        )
        return _pad_cat(feature_segments, self.device, self.default_dtype)

    @property
    def modality_mapping(self) -> torch.Tensor:
        """Return one modality identifier per token across all samples."""
        seqlens: list[int] = []
        modalities: list[int] = []
        for item in self.items:
            item_seqlens, item_modalities = item.modality_map_seqlens
            seqlens.extend(item_seqlens)
            modalities.extend(item_modalities)
        return _segment_paint(
            modalities,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    @property
    def coords_mapping(self) -> torch.Tensor:
        """Return coordinate rows across all packed samples."""
        coordinate_segments = list(
            chain.from_iterable(item.coords_to_cat for item in self.items)
        )
        return torch.cat(coordinate_segments, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        """Return diffusion-time features across all packed samples."""
        return torch.cat(
            [item.time_token_sequence for item in self.items],
            dim=0,
        )

    @property
    def total_token_num(self) -> int:
        """Return the total number of unpadded tokens."""
        return self._total_token_num_sum

    @property
    def cu_seqlen(self) -> torch.Tensor:
        """Return cumulative token boundaries for each packed sample."""
        return self._total_token_cu_seqlens.clone()

    @property
    def max_seqlen(self) -> int:
        """Return the longest unpadded sample sequence."""
        return self._total_token_num_max

    def __getitem__(self, index: int) -> SingleData:
        """Return one packed sample descriptor."""
        return self.items[index]

    def depack_token_sequence(
        self,
        token_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore video and audio tensors from model-output token segments."""
        videos: list[torch.Tensor] = []
        audios: list[torch.Tensor] = []
        token_counts = [item.total_token_num for item in self.items]
        for item, token_slice in zip(
            self.items,
            torch.split(token_sequence, token_counts, dim=0),
            strict=True,
        ):
            video_flat = token_slice[:item.video_token_num, :item.video_channel]
            output_channel = item.video_channel // (
                item.t_patch_size * item.patch_size * item.patch_size
            )
            video = rearrange(
                video_flat,
                "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
                T=item.t // item.t_patch_size,
                H=item.h // item.patch_size,
                W=item.w // item.patch_size,
                pT=item.t_patch_size,
                pH=item.patch_size,
                pW=item.patch_size,
                C=output_channel,
            ).contiguous()
            audio = torch.zeros(
                item.origin_audio_feat_len,
                item.audio_channel,
                device=token_sequence.device,
                dtype=token_sequence.dtype,
            )
            audio[:item.audio_feat_len] = token_slice[
                item.video_token_num:item.video_token_num + item.audio_feat_len,
                :item.audio_channel,
            ]
            videos.append(video)
            audios.append(audio)
        return torch.stack(videos, dim=0), torch.stack(audios, dim=0)


class Magi2DataProxy:
    """Pack MAGI-2 preview inputs and restore preview-model outputs."""

    def __init__(self, config: Magi2DataProxyConfig) -> None:
        self.config = config
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs: Any) -> None:
        """Save packing metadata that is required to depack model outputs."""
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str) -> Any:
        """Return saved packing metadata by name."""
        return self._saved_data[key]

    def _reduce_max_token_num_for_ep_cp(self, total_token_num: int) -> int:
        """Find the maximum token count across active expert and context groups."""
        if not (dist.is_available() and dist.is_initialized()):
            return total_token_num
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        local_token_count = torch.tensor(
            [total_token_num],
            device=device,
            dtype=torch.int64,
        )
        maximum_counts = [local_token_count]
        ep_size = psm.get_world_size("ep")
        if ep_size > 1:
            ep_maximum = local_token_count.clone()
            dist.all_reduce(
                ep_maximum,
                op=dist.ReduceOp.MAX,
                group=psm.get_parallel_group("ep"),
            )
            maximum_counts.append(ep_maximum)
        cp_size = psm.get_world_size("cp")
        if cp_size > 1:
            cp_maximum = local_token_count.clone()
            dist.all_reduce(
                cp_maximum,
                op=dist.ReduceOp.MAX,
                group=psm.get_parallel_group("cp"),
            )
            maximum_counts.append(cp_maximum)
        return max(int(maximum.item()) for maximum in maximum_counts)

    def _pad_for_ep_cp(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        time_token_sequence: torch.Tensor,
        varlen_handler: VarlenHandler,
        align_to: int = 48,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        VarlenHandler,
        int,
    ]:
        """Pad distributed packed inputs to a shared multiple-of-48 length."""
        cp_size = max(1, psm.get_world_size("cp"))
        ep_size = max(1, psm.get_world_size("ep"))
        if cp_size <= 1 and ep_size <= 1:
            return (
                x,
                coords_mapping,
                modality_mapping,
                time_token_sequence,
                varlen_handler,
                0,
            )

        total_token_num = x.shape[0]
        target_size = self._reduce_max_token_num_for_ep_cp(total_token_num)
        padded_size = _ceil_div(target_size, align_to) * align_to
        pad_size = padded_size - total_token_num
        if pad_size <= 0:
            return (
                x,
                coords_mapping,
                modality_mapping,
                time_token_sequence,
                varlen_handler,
                0,
            )

        x = F.pad(x, (0, 0, 0, pad_size), value=0)
        pad_coords = _get_coords(
            shape=(pad_size, 1, 1),
            ref_feat_shape=(pad_size, 1, 1),
            offset_thw=(0, 0, 0),
            device=x.device,
            dtype=torch.float32,
        )
        coords_mapping = torch.cat([coords_mapping, pad_coords], dim=0)
        modality_mapping = F.pad(
            modality_mapping,
            (0, pad_size),
            value=int(Modality.TEXT),
        )
        if time_token_sequence.shape[-1] == 0:
            time_token_sequence = time_token_sequence.new_zeros(padded_size, 0)
        else:
            time_token_sequence = F.pad(
                time_token_sequence,
                (0, 0, 0, pad_size),
                value=0,
            )
        padded_boundary = torch.tensor(
            [padded_size],
            device=x.device,
            dtype=torch.int32,
        )
        varlen_handler = VarlenHandler(
            cu_seqlens_q=torch.cat(
                [varlen_handler.cu_seqlens_q, padded_boundary]
            ),
            cu_seqlens_k=torch.cat(
                [varlen_handler.cu_seqlens_k, padded_boundary]
            ),
            max_seqlen_q=max(varlen_handler.max_seqlen_q, pad_size),
            max_seqlen_k=max(varlen_handler.max_seqlen_k, pad_size),
        )
        return (
            x,
            coords_mapping,
            modality_mapping,
            time_token_sequence,
            varlen_handler,
            pad_size,
        )

    def img2tokens(self, x_t: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping 3D patches with channel-major features."""
        kernel_size = (self.t_patch_size, self.patch_size, self.patch_size)
        if not all(
            size >= kernel
            for size, kernel in zip(x_t.shape[2:], kernel_size, strict=True)
        ):
            return torch.empty(
                x_t.shape[0],
                0,
                x_t.shape[1] * math.prod(kernel_size),
                device=x_t.device,
                dtype=x_t.dtype,
            )
        patches = (
            x_t.unfold(2, self.t_patch_size, self.t_patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .unfold(4, self.patch_size, self.patch_size)
        )
        batch_size, channels, t_steps, h_steps, w_steps, _, _, _ = patches.shape
        return (
            patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
            .reshape(batch_size, t_steps * h_steps * w_steps, -1)
            .contiguous()
        )

    def process_input(
        self,
        data: ModelInput,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        VarlenHandler,
        torch.Tensor,
    ]:
        """Pack a batch without allowing conditional samples to attend each other."""
        batch_size, input_video_channel, video_t, video_h, video_w = data.x_t.shape
        video_tokens = self.img2tokens(data.x_t)
        audio_tokens = data.audio_x_t.contiguous()
        text_tokens = data.txt_feat.contiguous()

        per_token_video_tokens = None
        per_token_audio = None
        if self.config.time_channel_dim > 0 and data.per_token_video_t is not None:
            per_token_video_tokens = self.img2tokens(data.per_token_video_t)[:, :, :1]
            per_token_audio = data.per_token_audio_t

        ref_image_data = data.ref_image_feat
        ref_image_feat_len = data.ref_image_feat_len
        ref_image_special_tokens = data.ref_image_special_token_embedding
        has_ref_image = (
            ref_image_data is not None
            and ref_image_data.ndim >= 5
            and ref_image_feat_len is not None
            and ref_image_feat_len.ndim >= 2
            and ref_image_special_tokens is not None
            and ref_image_special_tokens.ndim >= 3
        )
        num_ref_images = ref_image_data.shape[1] if has_ref_image else 0
        ref_device = text_tokens.device
        ref_dtype = text_tokens.dtype

        items: list[SingleData] = []
        for batch_index in range(batch_size):
            image_features: list[torch.Tensor] = []
            image_feature_lengths: list[list[int]] = []
            image_special_tokens: list[torch.Tensor] = []
            for image_index in range(num_ref_images):
                assert ref_image_data is not None
                assert ref_image_feat_len is not None
                assert ref_image_special_tokens is not None
                image_features.append(
                    self.img2tokens(
                        ref_image_data[batch_index, image_index].unsqueeze(0)
                    ).squeeze(0)
                )
                image_feature_lengths.append(
                    _len_to_list(ref_image_feat_len[batch_index, image_index])
                )
                image_special_tokens.append(
                    ref_image_special_tokens[batch_index, image_index]
                    .to(device=ref_device, dtype=ref_dtype)
                    .unsqueeze(0)
                )

            items.append(
                SingleData(
                    video_x_t=video_tokens[batch_index],
                    audio_x_t=audio_tokens[batch_index],
                    audio_feat_len=_to_int(data.audio_feat_len[batch_index]),
                    txt_feat=text_tokens[batch_index],
                    txt_feat_len=_to_int(data.txt_feat_len[batch_index]),
                    ref_image_feats=image_features,
                    ref_image_feat_lens=image_feature_lengths,
                    ref_image_special_tokens=image_special_tokens,
                    t=video_t,
                    h=video_h,
                    w=video_w,
                    patch_size=self.patch_size,
                    t_patch_size=self.t_patch_size,
                    spatial_rope_interpolation=self.config.spatial_rope_interpolation,
                    diffusion_t=(
                        data.t[batch_index] if self.config.add_time_token else None
                    ),
                    per_token_video_t=(
                        per_token_video_tokens[batch_index]
                        if per_token_video_tokens is not None
                        else None
                    ),
                    per_token_audio_t=(
                        per_token_audio[batch_index]
                        if per_token_audio is not None
                        else None
                    ),
                    time_channel_dim=self.config.time_channel_dim,
                    time_pos_fps=self.config.time_pos_fps,
                    vae_first_latent_is_image=self.config.vae_first_latent_is_image,
                    video_fps=self.config.video_fps,
                )
            )

        packed = SimplePackedData(items)
        packed_cu_seqlens = packed.cu_seqlen.to(
            device=data.x_t.device,
            dtype=torch.int32,
        )
        varlen_handler = VarlenHandler(
            cu_seqlens_q=packed_cu_seqlens,
            cu_seqlens_k=packed_cu_seqlens.clone(),
            max_seqlen_q=packed.max_seqlen,
            max_seqlen_k=packed.max_seqlen,
        )
        (
            token_sequence,
            coords_mapping,
            modality_mapping,
            time_token_sequence,
            varlen_handler,
            pad_size,
        ) = self._pad_for_ep_cp(
            packed.token_sequence,
            packed.coords_mapping,
            packed.modality_mapping,
            packed.time_token_sequence,
            varlen_handler,
        )
        self.saved_for_output(
            simple_packed_data=packed,
            input_video_channel=input_video_channel,
            pad_size=pad_size,
        )
        return (
            token_sequence,
            coords_mapping,
            modality_mapping,
            varlen_handler,
            time_token_sequence,
        )

    def process_output(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove distributed padding and restore video and audio tensors."""
        packed: SimplePackedData = self.get_saved_data("simple_packed_data")
        pad_size = self.get_saved_data("pad_size")
        if pad_size > 0:
            x = x[:-pad_size]
        return packed.depack_token_sequence(x)
