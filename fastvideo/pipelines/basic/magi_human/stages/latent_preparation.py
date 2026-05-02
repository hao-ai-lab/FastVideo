# SPDX-License-Identifier: Apache-2.0
"""Latent preparation stage for daVinci-MagiHuman base text-to-AV.

Produces:
  - random video latent of shape `[1, z_dim, latent_T, latent_H, latent_W]`,
  - random audio latent of shape `[1, num_frames, 64]` (the DiT jointly
    denoises both modalities),
  - padded T5-Gemma text embedding (target length 640) plus the original
    (pre-pad) context length, which the UniPC + CFG loop needs so the
    unconditional path sees the same padded length.

Also stakes out the per-token coords / modality map that the DiT consumes
(replicates the reference `MagiDataProxy.process_input`).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult

# Matches inference/common/sequence_schema.py in the reference.
MODALITY_VIDEO = 0
MODALITY_AUDIO = 1
MODALITY_TEXT = 2

# Audio temporal compression ratio: 1 audio frame → 1/4 latent frame.
# Mirrors data_proxy.py:206 `(audio_feat_len - 1) // 4 + 1` where 4 is
# the audio VAE's temporal stride (same as vae_stride[0] for video).
_AUDIO_TEMPORAL_COMPRESSION = 4

# v1 text-coord reference shape: (T=2, H=1, W=1).
# Mirrors data_proxy.py:202 `ref_feat_shape=(2, 1, 1)` for coords_style=="v1".
_V1_TEXT_REF_SHAPE: tuple[int, int, int] = (2, 1, 1)


def _build_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    h_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    w_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w
    tg, hg, wg = torch.meshgrid(time_rng, h_rng, w_rng, indexing="ij")
    coords = torch.stack([tg, hg, wg], dim=-1).reshape(-1, 3)
    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w],
        device=device,
        dtype=dtype,
    ).expand(coords.size(0), -1)
    return torch.cat([coords, meta], dim=-1)


def _pad_or_trim_dim1(t: torch.Tensor, target: int) -> tuple[torch.Tensor, int]:
    """Pad-or-trim along dim 1. Returns (new_tensor, original_length)."""
    current = t.size(1)
    if current < target:
        pad = [0, 0, 0, target - current]
        return F.pad(t, pad, "constant", 0.0), current
    return t[:, :target], target


def _img2tokens(x_t: torch.Tensor, t_patch: int, patch: int) -> torch.Tensor:
    """Pack a video latent [B, C, T, H, W] -> [B, L, C * t_patch * patch^2].

    Per-token feature ordering is channel-major ``(C pT pH pW)``: the DiT's
    ``video_embedder`` weight was trained on the layout produced by
    upstream's grouped-conv ``UnfoldNd`` packer (channel slowest, patch
    elements fastest). Spatial-major ``(pT pH pW C)`` silently permutes the
    in-features and produces noise output. Asymmetric with
    ``unpack_tokens`` which uses ``(pT pH pW C)`` to match
    ``final_linear_video``'s trained output layout.
    """
    B, C, T, H, W = x_t.shape
    assert T % t_patch == 0 and H % patch == 0 and W % patch == 0, (
        f"Latent dims {T,H,W} must divide ({t_patch}, {patch}, {patch})")
    return rearrange(
        x_t,
        "B C (T pT) (H pH) (W pW) -> B (T H W) (C pT pH pW)",
        pT=t_patch,
        pH=patch,
        pW=patch,
    ).contiguous()


class MagiHumanLatentPreparationStage(PipelineStage):
    """Prepare latents, coords, modality maps, and padded text embed."""

    def __init__(
        self,
        vae_stride: tuple[int, int, int] = (4, 16, 16),
        z_dim: int = 48,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        fps: int = 25,
        t5_gemma_target_length: int = 640,
        coords_style: Literal["v1", "v2"] = "v2",
        text_offset: int = 0,
        audio_in_channels: int = 64,
    ) -> None:
        super().__init__()
        self.vae_stride = vae_stride
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.fps = fps
        self.t5_gemma_target_length = t5_gemma_target_length
        self.coords_style = coords_style
        self.text_offset = text_offset
        self.audio_in_channels = audio_in_channels

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seconds = int(getattr(batch, "num_seconds", None) or 4)
        fps = self.fps
        num_frames = seconds * fps + 1
        latent_T = (num_frames - 1) // 4 + 1

        # Match upstream pipeline.py:61-64 + video_generate.py:254-261:
        # the requested 272p height snaps to 256, while width stays 480.
        br_h = int(batch.height) if batch.height else 256
        br_w = int(batch.width) if batch.width else 480
        pT, pH, pW = self.patch_size
        vt, vh, vw = self.vae_stride
        # Snap to patch granularity (matches reference).
        latent_H = (br_h // vh // pH) * pH
        latent_W = (br_w // vw // pW) * pW
        actual_H = latent_H * vh
        actual_W = latent_W * vw
        batch.height = actual_H
        batch.width = actual_W

        generator = torch.Generator(device=device)
        if batch.seed is not None:
            generator.manual_seed(int(batch.seed))

        # Video latent: [1, z_dim, latent_T, latent_H, latent_W]
        video_latent = torch.randn(
            (1, self.z_dim, latent_T, latent_H, latent_W),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        # Audio latent: [1, num_frames, audio_in_channels]
        audio_latent = torch.randn(
            (1, num_frames, self.audio_in_channels),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # Prompt embeds: the upstream TextEncodingStage already ran. It
        # produced a list of [1, L, D] tensors per prompt. Pad/trim each
        # to the target length and store the original length so the DiT
        # stage can build the correct modality-map slices.
        padded_prompt_embeds: list[torch.Tensor] = []
        padded_prompt_lens: list[int] = []
        for embed in batch.prompt_embeds:
            # embed: [1, L, 3584]
            padded, original = _pad_or_trim_dim1(
                embed.to(torch.float32),
                target=self.t5_gemma_target_length,
            )
            padded_prompt_embeds.append(padded)
            padded_prompt_lens.append(original)
        batch.prompt_embeds = padded_prompt_embeds
        # Stash the original text length list on the batch for the denoise
        # stage — FastVideo's ForwardBatch doesn't have a first-class field
        # for this so we attach it.
        batch.magi_original_text_lens = padded_prompt_lens

        # Matching negative prompts.
        if batch.negative_prompt_embeds is not None and batch.negative_prompt_embeds:
            padded_neg: list[torch.Tensor] = []
            padded_neg_lens: list[int] = []
            for embed in batch.negative_prompt_embeds:
                padded, original = _pad_or_trim_dim1(
                    embed.to(torch.float32),
                    target=self.t5_gemma_target_length,
                )
                padded_neg.append(padded)
                padded_neg_lens.append(original)
            batch.negative_prompt_embeds = padded_neg
            batch.magi_original_neg_text_lens = padded_neg_lens

        batch.latents = video_latent
        batch.audio_latents = audio_latent
        batch.num_frames = num_frames
        batch.magi_latent_T = latent_T
        batch.magi_latent_H = latent_H
        batch.magi_latent_W = latent_W
        return batch


class StaticPackedInputs:
    """Step-invariant packed inputs: video+audio tokens, coords, modality map.

    Computed once before the denoise loop; reused for every cond/uncond call.
    Text tokens are NOT included here because cond/uncond have different lengths.
    """

    __slots__ = (
        "video_tokens",
        "audio_tokens",
        "video_coords",
        "audio_coords",
        "video_mm",
        "audio_mm",
        "video_token_num",
        "audio_feat_len",
        "max_ch",
    )

    def __init__(
        self,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        video_mm: torch.Tensor,
        audio_mm: torch.Tensor,
        max_ch: int,
    ) -> None:
        self.video_tokens = video_tokens
        self.audio_tokens = audio_tokens
        self.video_coords = video_coords
        self.audio_coords = audio_coords
        self.video_mm = video_mm
        self.audio_mm = audio_mm
        self.video_token_num = video_tokens.size(0)
        self.audio_feat_len = audio_tokens.size(0)
        self.max_ch = max_ch


def build_static_packed_inputs(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    audio_feat_len: int,
    patch_size: tuple[int, int, int],
    coords_style: Literal["v1", "v2"] = "v2",
) -> StaticPackedInputs:
    """Build the step-invariant portion of the packed token stream.

    Returns video+audio tokens (padded to a common channel width), their
    coords, and their modality slices. Text is excluded because cond/uncond
    differ in length; call assemble_packed_inputs to attach text per call.

    Mirrors SingleData.token_sequence / coords_mapping / modality_mapping in
    inference/pipeline/data_proxy.py, minus the text portion.
    """
    pT, pH, pW = patch_size
    _, z_dim, T, H, W = video_latent.shape
    assert video_latent.size(0) == 1, "batch size 1 required for MagiHuman base"

    video_tokens = _img2tokens(video_latent, t_patch=pT, patch=pH)[0]
    audio_tokens = audio_latent[0, :audio_feat_len].contiguous()

    max_ch = max(video_tokens.size(-1), audio_tokens.size(-1))
    video_tokens = F.pad(video_tokens, (0, max_ch - video_tokens.size(-1)))
    audio_tokens = F.pad(audio_tokens, (0, max_ch - audio_tokens.size(-1)))

    device = video_tokens.device
    dtype = video_tokens.dtype
    video_token_num = video_tokens.size(0)

    video_mm = torch.full((video_token_num, ), MODALITY_VIDEO, dtype=torch.int64, device=device)
    audio_mm = torch.full((audio_feat_len, ), MODALITY_AUDIO, dtype=torch.int64, device=device)

    video_ref_shape = (T // pT, H // pH, W // pW)
    video_coords = _build_coords(
        shape=video_ref_shape,
        ref_feat_shape=video_ref_shape,
        device=device,
        dtype=dtype,
    )

    if coords_style == "v2":
        audio_ref_t = (audio_feat_len - 1) // _AUDIO_TEMPORAL_COMPRESSION + 1
        audio_coords = _build_coords(
            shape=(audio_feat_len, 1, 1),
            ref_feat_shape=(audio_ref_t // pT, 1, 1),
            device=device,
            dtype=dtype,
        )
    else:
        audio_coords = _build_coords(
            shape=(audio_feat_len, 1, 1),
            ref_feat_shape=(T // pT, 1, 1),
            device=device,
            dtype=dtype,
        )

    return StaticPackedInputs(
        video_tokens=video_tokens,
        audio_tokens=audio_tokens,
        video_coords=video_coords,
        audio_coords=audio_coords,
        video_mm=video_mm,
        audio_mm=audio_mm,
        max_ch=max_ch,
    )


def assemble_packed_inputs(
    static: StaticPackedInputs,
    txt_feat: torch.Tensor,
    txt_feat_len: int,
    coords_style: Literal["v1", "v2"] = "v2",
    text_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach per-call text tokens to the precomputed static packed inputs.

    Returns (token_seq, coords, modality_map) ready for the DiT.
    """
    text_tokens = txt_feat[0, :txt_feat_len].contiguous()
    max_ch = max(static.max_ch, text_tokens.size(-1))

    video_tokens = F.pad(static.video_tokens, (0, max_ch - static.video_tokens.size(-1)))
    audio_tokens = F.pad(static.audio_tokens, (0, max_ch - static.audio_tokens.size(-1)))
    text_tokens = F.pad(text_tokens, (0, max_ch - text_tokens.size(-1)))
    token_seq = torch.cat([video_tokens, audio_tokens, text_tokens], dim=0)

    device = token_seq.device
    dtype = token_seq.dtype
    text_mm = torch.full((txt_feat_len, ), MODALITY_TEXT, dtype=torch.int64, device=device)
    mm = torch.cat([static.video_mm, static.audio_mm, text_mm], dim=0)

    if coords_style == "v2":
        text_coords = _build_coords(
            shape=(txt_feat_len, 1, 1),
            ref_feat_shape=(1, 1, 1),
            offset_thw=(-txt_feat_len, 0, 0),
            device=device,
            dtype=dtype,
        )
    else:
        text_coords = _build_coords(
            shape=(txt_feat_len, 1, 1),
            ref_feat_shape=_V1_TEXT_REF_SHAPE,
            offset_thw=(text_offset, 0, 0),
            device=device,
            dtype=dtype,
        )

    coords = torch.cat([static.video_coords, static.audio_coords, text_coords], dim=0)
    return token_seq, coords, mm


def build_packed_inputs(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    audio_feat_len: int,
    txt_feat: torch.Tensor,
    txt_feat_len: int,
    patch_size: tuple[int, int, int],
    coords_style: Literal["v1", "v2"] = "v2",
    text_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the full packed token stream in one call (backwards-compat wrapper).

    Equivalent to assemble_packed_inputs(build_static_packed_inputs(...), ...).
    Prefer calling the two helpers separately when the static portion can be
    reused across multiple calls (e.g. cond/uncond in the denoise loop).
    """
    static = build_static_packed_inputs(
        video_latent=video_latent,
        audio_latent=audio_latent,
        audio_feat_len=audio_feat_len,
        patch_size=patch_size,
        coords_style=coords_style,
    )
    return assemble_packed_inputs(
        static=static,
        txt_feat=txt_feat,
        txt_feat_len=txt_feat_len,
        coords_style=coords_style,
        text_offset=text_offset,
    )


def unpack_tokens(
    output: torch.Tensor,  # [L, max(V_ch, A_ch)]
    video_token_num: int,
    audio_feat_len: int,
    video_in_channels: int,
    audio_in_channels: int,
    latent_shape: tuple[int, int, int, int, int],  # [1, z_dim, T, H, W]
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `build_packed_inputs` for the DiT output.

    Splits the flat output back into a video latent (un-patched into
    B C T H W) and an audio latent (B, L, 64).
    """
    pT, pH, pW = patch_size
    _, z_dim, T, H, W = latent_shape
    tH, tW = H // pH, W // pW

    video_flat = output[:video_token_num, :video_in_channels]
    video_latent = rearrange(
        video_flat,
        "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
        H=tH,
        W=tW,
        pT=pT,
        pH=pH,
        pW=pW,
    ).contiguous().unsqueeze(0)

    audio_latent = output[
        video_token_num:video_token_num + audio_feat_len,
        :audio_in_channels,
    ].unsqueeze(0)

    return video_latent, audio_latent
