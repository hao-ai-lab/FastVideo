# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-Animate-14B: character animation / replacement.

Ported against both references: the official ``Wan-Video/Wan2.2``
(``wan/modules/animate/``) and diffusers' merged ``WanAnimateTransformer3DModel``.
The official Diffusers-format checkpoint (``Wan-AI/Wan2.2-Animate-14B-Diffusers``)
uses the diffusers naming, which this port loads directly.

Animate required no new tower: it is the stock Wan-I2V transformer
(channel-concat conditioning, 36-channel input, CLIP image branch, single
global timestep, standard RoPE grid), so this class *subclasses*
``WanTransformer3DModel`` -- the base ``__init__`` builds the entire tower from
``WanAnimateArchConfig`` -- and adds three things:

1. ``pose_patch_embedding``: a second patchifier whose output is **added** to
   the patchified video tokens, *skipping the reference latent frame* (frame 0
   of the sequence is the reference image and carries no pose). Adding pose to
   frame 0 corrupts identity conditioning -- the off-by-one is load-bearing.
2. The face-driving stack (``wan_animate_face.py``): 512x512 face crops ->
   LIA motion vectors -> causal 4x funnel -> 4+1 face tokens per latent frame.
3. ``face_adapter``: per-frame cross-attention applied residually **after**
   every ``inject_face_latents_blocks``-th block (blocks 0, 5, ..., 35 -- the
   adapter index i serves block i * stride; asserted in the arch config).

Sequence parallelism is not wired up in this first version: the face adapter's
per-frame reshape needs the full token sequence, and an SP shard splits frames
across ranks mid-frame. Single-GPU and FSDP-style weight sharding work
(``__init__`` raises on sp_world_size > 1).
"""
from typing import Any

import torch
from torch import nn

import fastvideo.envs as envs
from fastvideo.configs.models.dits.wan_animate import WanAnimateConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits.wan_animate_face import (MotionConv2d, WanAnimateFaceCrossAttention,
                                                    WanAnimateFaceEncoder, WanAnimateMotionEncoder)
from fastvideo.models.dits.wanvideo import WanTransformer3DModel
from fastvideo.platforms import current_platform


class WanAnimateTransformer3DModel(WanTransformer3DModel):
    """Wan2.2-Animate-14B transformer."""

    _fsdp_shard_conditions = WanAnimateConfig()._fsdp_shard_conditions
    _compile_conditions = WanAnimateConfig()._compile_conditions
    _supported_attention_backends = WanAnimateConfig()._supported_attention_backends
    param_names_mapping = WanAnimateConfig().param_names_mapping
    reverse_param_names_mapping = WanAnimateConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanAnimateConfig().lora_param_names_mapping

    def __init__(self, config: WanAnimateConfig, hf_config: dict[str, Any]) -> None:
        if envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN":
            raise ValueError(
                "Wan-Animate has no VSA checkpoint. The base __init__ would swap in "
                "WanTransformerBlock_VSA, whose to_gate_compress weights are absent here and get "
                "silently zero-filled by the loader (fsdp_load ALLOWED_NEW_PARAM_PATTERNS). Unset "
                "FASTVIDEO_ATTENTION_BACKEND.")
        if get_sp_world_size() > 1:
            raise NotImplementedError(
                "Wan-Animate does not support sequence parallelism yet: the face adapter's "
                "per-frame attention needs the whole token sequence on one rank. Run with "
                "sp_world_size=1 (FSDP weight sharding is unaffected).")
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        inner_dim = arch.hidden_size
        self.inject_face_latents_blocks = arch.inject_face_latents_blocks
        self.motion_encoder_batch_size = arch.motion_encoder_batch_size

        # Second patchifier for the pose video: same PatchEmbed wrapper as the
        # base patch_embedding, so the checkpoint's pose_patch_embedding.*
        # lands on .proj via the mapping.
        self.pose_patch_embedding = PatchEmbed(in_chans=arch.latent_channels,
                                               embed_dim=inner_dim,
                                               patch_size=arch.patch_size,
                                               flatten=False)

        self.motion_encoder = WanAnimateMotionEncoder(size=arch.motion_encoder_size,
                                                      style_dim=arch.motion_style_dim,
                                                      motion_dim=arch.motion_dim,
                                                      out_dim=arch.motion_encoder_dim,
                                                      channels=arch.motion_encoder_channel_sizes)
        self.face_encoder = WanAnimateFaceEncoder(in_dim=arch.motion_encoder_dim,
                                                  out_dim=inner_dim,
                                                  hidden_dim=arch.face_encoder_hidden_dim,
                                                  num_heads=arch.face_encoder_num_heads,
                                                  eps=arch.eps)
        # Dense, positional: face_adapter.{i} serves block i * inject stride.
        self.face_adapter = nn.ModuleList([
            WanAnimateFaceCrossAttention(inner_dim, arch.num_attention_heads, eps=arch.eps)
            for _ in range(arch.num_layers // arch.inject_face_latents_blocks)
        ])

        # The Animate checkpoint ships no attn2.norm_added_q (verified on the
        # real weights: 40 missing keys under strict load). FastVideo's
        # WanI2VCrossAttention allocates it only because base Wan-I2V
        # checkpoints ship it; it is never used in forward. Drop it so the
        # strict load has neither missing nor unexpected keys.
        for block in self.blocks:
            if hasattr(block.attn2, "norm_added_q"):
                block.attn2.norm_added_q = nn.Identity()

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """Diffusers pins this tensor fp32 (_keep_in_fp32_modules); match it.

        The QR basis is re-orthonormalised from it every forward, so bf16
        rounding changes the basis itself, not just its precision -- upcasting
        inside forward is too late.
        """
        return torch.float32 if name == "motion_encoder.motion_synthesis_weight" else default_dtype

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        """Rebuild the motion encoder's anti-aliasing blur kernels after meta load.

        TransformerLoader builds the model under ``torch.device("meta")`` and
        streams checkpoint weights in; non-persistent buffers are not in the
        checkpoint and stay on meta until this hook (called by fsdp_load)
        recreates them. Kernels stay fp32 -- forward casts them to the
        activation dtype.
        """
        for module in self.modules():
            if isinstance(module, MotionConv2d) and module.blur and module.blur_kernel.is_meta:
                module.blur_kernel = module._build_blur_kernel().to(device)

    def _patchify_with_pose(self, hidden_states: torch.Tensor, pose_latents: torch.Tensor) -> torch.Tensor:
        """Patchify the 36-channel input and add pose tokens to every frame but the first.

        Returns the pre-flatten conv grid [B, dim, T, h, w]. Frame 0 is the
        reference latent and receives no pose -- kept as a separate method so a
        test can pin that isolation directly.
        """
        grid = self.patch_embedding(hidden_states)
        pose = self.pose_patch_embedding(pose_latents.to(hidden_states.dtype))
        grid[:, :, 1:] = grid[:, :, 1:] + pose
        return grid

    def _encode_face(self, face_pixel_values: torch.Tensor) -> torch.Tensor:
        """Face video [B, 3, F, H, W] -> face tokens [B, F' + 1, N + 1, dim].

        The motion encoder runs frame-by-frame in chunks of
        ``motion_encoder_batch_size`` (a memory knob, not architecture). A zero
        "pad face" row is prepended for the reference latent frame, which has
        no driving face.
        """
        batch_size, channels, num_face_frames = face_pixel_values.shape[:3]
        frames = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(-1, channels, *face_pixel_values.shape[3:])
        motion_vec = torch.cat(
            [self.motion_encoder(chunk) for chunk in torch.split(frames, self.motion_encoder_batch_size)])
        motion_vec = self.face_encoder(motion_vec.view(batch_size, num_face_frames, -1))
        pad_face = torch.zeros_like(motion_vec[:, :1])
        return torch.cat([pad_face, motion_vec], dim=1)

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                pose_latents: torch.Tensor | None = None,
                face_pixel_values: torch.Tensor | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        """Denoise one step of an animation/replacement segment.

        hidden_states    [B, 36, T+1, H, W]: noise | 4ch mask | conditional
                         latent y, channel-concatenated by the pipeline. Frame
                         0 is the reference latent slot.
        pose_latents     [B, 16, T, H, W] VAE-encoded skeleton video; exactly
                         one frame fewer than hidden_states (no pose on ref).
        face_pixel_values [B, 3, F, 512, 512] raw face crops,
                         F = 4T - 3 pixel frames for T latent frames.
        encoder_hidden_states_image  CLIP features of the reference image
                         (257 tokens) -- required, the I2V cross-attention
                         splits them off positionally.

        Returns the full [B, 16, T+1, H, W] velocity; the decode stage drops
        the reference slot (and any guidance frames).
        """
        if pose_latents is None or face_pixel_values is None:
            raise ValueError(
                "Wan-Animate needs both a pose video and a face video. Got "
                f"pose_latents={'set' if pose_latents is not None else 'None'}, "
                f"face_pixel_values={'set' if face_pixel_values is not None else 'None'}. The "
                "pipeline supplies these from the preprocessed src_pose/src_face inputs.")
        if encoder_hidden_states_image is None or (isinstance(encoder_hidden_states_image, list)
                                                   and not encoder_hidden_states_image):
            raise ValueError("Wan-Animate needs CLIP features of the reference image "
                             "(encoder_hidden_states_image): the I2V cross-attention splits off "
                             "the first 257 context tokens as image tokens positionally, so this "
                             "must fail here rather than misread the text tokens.")
        if pose_latents.shape[2] + 1 != hidden_states.shape[2]:
            raise ValueError(
                f"pose_latents has {pose_latents.shape[2]} frames but must have exactly one fewer "
                f"than hidden_states ({hidden_states.shape[2]}): the reference latent frame "
                "carries no pose.")

        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = encoder_hidden_states_image[0]

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        freqs_cis = self._rotary_embeddings(post_patch_num_frames, post_patch_height, post_patch_width,
                                            hidden_states.device)

        # 1. Patchify, then add pose tokens to every frame but the reference.
        hidden_states = self._patchify_with_pose(hidden_states, pose_latents)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        original_seq_len = hidden_states.shape[1]

        # 2. Face tokens: one 4+1 group per latent frame, zero row for the ref.
        motion_vec = self._encode_face(face_pixel_values)
        if motion_vec.shape[1] != post_patch_num_frames:
            raise ValueError(
                f"face/video misalignment: the face video produced {motion_vec.shape[1]} latent-"
                f"frame token groups (incl. the reference pad) but the video has "
                f"{post_patch_num_frames} latent frames. The face video needs 4*T - 3 pixel "
                f"frames for T video latent frames (e.g. 77 for 20).")

        # 3. Condition embeddings (time, text, CLIP image) -- Wan2.1 timestep logic.
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        # The blocks add cross-attention output straight onto hidden_states, so
        # a context in another dtype would silently upcast the residual stream.
        assert encoder_hidden_states.dtype == orig_dtype, (
            f"context dtype {encoder_hidden_states.dtype} != latent dtype {orig_dtype}")

        # 4. The tower, with a face cross-attention residual after every
        #    inject-stride-th block.
        for idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)
            if idx % self.inject_face_latents_blocks == 0:
                adapter = self.face_adapter[idx // self.inject_face_latents_blocks]
                hidden_states = hidden_states + adapter(hidden_states, motion_vec)

        # 5. Output norm, projection & unpatchify (single-timestep path).
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height,
                                              post_patch_width, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _rotary_embeddings(self, frames: int, height: int, width: int,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy of the base model's inline RoPE block; keep the dim split in sync."""
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (frames, height, width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000)
        return freqs_cos.to(device).float(), freqs_sin.to(device).float()


# Entry point for model registry: auto-discovered by class name, which matches
# the checkpoint's _class_name -- no registry alias table entry is needed.
EntryClass = WanAnimateTransformer3DModel
