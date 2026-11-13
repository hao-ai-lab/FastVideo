# SPDX-License-Identifier: Apache-2.0
"""PAN2 reference implementation for local parity tests.

The diffusers ``PAN2Transformer3DModel`` and ``PAN2ModularPipeline`` are not public yet. The reference here is
assembled from diffusers' and transformers' own building blocks (``AdaLayerNormZero``, ``AdaLayerNormContinuous``,
``FeedForward`` with ``gelu-approximate``/``swiglu``, ``Timesteps``/``TimestepEmbedding``,
``get_1d_rotary_pos_embed``/``apply_rotary_emb``, ``Qwen3_5TextModel``, ``AutoencoderKLWan``,
``FlowMatchEulerDiscreteScheduler``) and follows the vLLM-Omni PAN2 port (vllm-project/vllm-omni#8602), which was
validated against the diffusers implementation (upstream revision ``e0c4c731``). Module names are the checkpoint's,
so checkpoints load strictly. Its unconditional SP padding mask is equivalent to unpadded reference attention.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm

TINY_REPO_ID = "wuqing157/tiny-pan2-modular-pipe"
TINY_REVISION = "3ff3130b4527f61aaab7bde01465c9884dc84cdd"
SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."
TEXT_ENCODER_LAYERS_TO_SKIP = 2
MAX_SEQUENCE_LENGTH = 1536


def tiny_model_dir() -> Path | None:
    """``PAN2_TINY_MODEL_DIR``, else the pinned tiny checkpoint in the local HF cache."""
    local = os.getenv("PAN2_TINY_MODEL_DIR")
    if local:
        return Path(local)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(TINY_REPO_ID, revision=TINY_REVISION, local_files_only=True))
    except Exception:
        return None


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2).flatten(2)


class _RefRefinerAttention(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, dim, bias=False)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        value = self.to_v(x).unflatten(-1, (self.heads, -1))
        return self.to_out[0](_sdpa(query, key, value))


class _RefRefinerBlock(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = _RefRefinerAttention(dim, heads, dim_head, eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="swiglu", bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class _RefJointAttention(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, heads * dim_head)
        self.to_k = nn.Linear(dim, heads * dim_head)
        self.to_v = nn.Linear(dim, heads * dim_head)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, dim)])
        self.add_q_proj = nn.Linear(dim, heads * dim_head)
        self.add_k_proj = nn.Linear(dim, heads * dim_head)
        self.add_v_proj = nn.Linear(dim, heads * dim_head)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)
        self.to_add_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                rope: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        value = self.to_v(x).unflatten(-1, (self.heads, -1))
        # Interleaved (GPT-J) rotary embedding on the video tokens only.
        query = apply_rotary_emb(query, rope, use_real=True, use_real_unbind_dim=-1, sequence_dim=1)
        key = apply_rotary_emb(key, rope, use_real=True, use_real_unbind_dim=-1, sequence_dim=1)

        context_query = self.norm_added_q(self.add_q_proj(context).unflatten(-1, (self.heads, -1)))
        context_key = self.norm_added_k(self.add_k_proj(context).unflatten(-1, (self.heads, -1)))
        context_value = self.add_v_proj(context).unflatten(-1, (self.heads, -1))

        out = _sdpa(torch.cat([query, context_query], dim=1), torch.cat([key, context_key], dim=1),
                    torch.cat([value, context_value], dim=1))
        out, context_out = out.split([x.shape[1], context.shape[1]], dim=1)
        return self.to_out[0](out), self.to_add_out(context_out)


class _RefBlock(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, context_mlp_ratio: float):
        super().__init__()
        eps = 1e-6
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))
        self.attn = _RefJointAttention(dim, heads, dim_head, eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(dim, inner_dim=int(dim * context_mlp_ratio), activation_fn="gelu-approximate")

    def forward(self, x, context, temb, rope):
        norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, emb=temb)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.scale_shift_table_context.chunk(6, dim=-1))
        norm_context = self.norm1_context(context) * (1 + c_scale_msa) + c_shift_msa
        attn_out, context_attn_out = self.attn(norm_x, norm_context, rope)
        x = x + attn_out * gate_msa.unsqueeze(1)
        context = context + context_attn_out * c_gate_msa
        norm_x = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_context = self.norm2_context(context) * (1 + c_scale_mlp) + c_shift_mlp
        x = x + self.ff(norm_x) * gate_mlp.unsqueeze(1)
        context = context + self.ff_context(norm_context) * c_gate_mlp
        return x, context


class ReferencePAN2Transformer(nn.Module):

    def __init__(self, patch_size=(1, 2, 2), in_channels=97, out_channels=48, num_attention_heads=32,
                 attention_head_dim=128, num_layers=44, num_refiner_layers=4, mlp_ratio=8.0, context_mlp_ratio=4.0,
                 refiner_mlp_ratio=4.0, text_embed_dim=4096, rope_axes_dim=(16, 56, 56), rope_theta=256.0,
                 rope_temporal_scale=16 / 24, **_):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.rope_axes_dim = tuple(rope_axes_dim)
        self.rope_theta = rope_theta
        self.rope_temporal_scale = rope_temporal_scale
        inner_dim = num_attention_heads * attention_head_dim
        self.x_embedder = nn.Conv3d(in_channels, inner_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.context_embedder = nn.Sequential(RMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, inner_dim))
        self.context_refiner = nn.ModuleList([
            _RefRefinerBlock(inner_dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5)
            for _ in range(num_refiner_layers)
        ])
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))
        self.transformer_blocks = nn.ModuleList([
            _RefBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels)

    def _rope(self, num_frames: int, height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        axes = [torch.arange(size, device=device, dtype=torch.float32) for size in (num_frames, height, width)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        grid[0] = grid[0] * self.rope_temporal_scale
        cos, sin = zip(*[
            get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), self.rope_theta, use_real=True,
                                    repeat_interleave_real=True) for i, dim in enumerate(self.rope_axes_dim)
        ])
        return torch.cat(cos, dim=1).float(), torch.cat(sin, dim=1).float()

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        t, h, w = num_frames // p_t, height // p_h, width // p_w
        rope = self._rope(t, h, w, hidden_states.device)
        x = self.x_embedder(hidden_states).flatten(2).transpose(1, 2)
        temb = self.time_embedder(self.time_proj(timestep).to(x.dtype))
        context = self.context_embedder(encoder_hidden_states)
        for block in self.context_refiner:
            context = block(context)
        context = context + self.context_type_embedding
        for block in self.transformer_blocks:
            x, context = block(x, context, temb, rope)
        x = self.proj_out(self.norm_out(x, temb))
        x = x.reshape(batch_size, t, h, w, -1, p_t, p_h, p_w).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def reference_encode_prompt(tokenizer, text_encoder, prompt: str, device) -> torch.Tensor:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt or " "}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True).input_ids
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    crop_start = next(i + len(header_ids) for i in range(len(input_ids))
                      if input_ids[i:i + len(header_ids)] == header_ids)
    input_ids = torch.tensor([input_ids[:crop_start + MAX_SEQUENCE_LENGTH]], device=device)
    hidden_states = text_encoder(input_ids=input_ids, output_hidden_states=True).hidden_states
    return hidden_states[-(TEXT_ENCODER_LAYERS_TO_SKIP + 1)][:, crop_start:]


def reference_condition_latents(vae, latents: torch.Tensor, image: PIL.Image.Image | None, height: int,
                                width: int) -> torch.Tensor:
    batch_size, num_channels, num_frames, latent_height, latent_width = latents.shape
    condition = latents.new_zeros(batch_size, num_channels + 1, num_frames, latent_height, latent_width)
    if image is None:
        return condition
    from diffusers.video_processor import VideoProcessor

    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
    left, top = round((image.width - width) / 2), round((image.height - height) / 2)
    image = image.crop((left, top, left + width, top + height))
    pixels = VideoProcessor(vae_scale_factor=16).preprocess(image, height=height, width=width)
    pixels = pixels.unsqueeze(2).to(device=latents.device, dtype=vae.dtype)
    image_latents = vae.encode(pixels).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(image_latents)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(image_latents)
    condition[:, :num_channels, :1] = (image_latents - mean) / std
    condition[:, num_channels:, :1] = 1.0
    return condition


@torch.no_grad()
def reference_generate(components: dict, *, prompt: str, negative_prompt: str, image: PIL.Image.Image | None,
                       height: int, width: int, num_frames: int, num_inference_steps: int, guidance_scale: float,
                       seed: int, dit_dtype: torch.dtype, device="cpu",
                       skip_first_latent_cfg: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Final latents and the decoded video in ``[0, 1]``, following the vLLM-Omni pipeline step by step.

    ``skip_first_latent_cfg`` is the reference sampler's ``--i2v_skip_first_latent_cfg``: with an image, the first
    latent frame takes the positive prediction instead of the guided one.
    """
    from diffusers.utils.torch_utils import randn_tensor

    tokenizer, text_encoder = components["tokenizer"], components["text_encoder"]
    transformer, vae, scheduler = components["transformer"], components["vae"], components["scheduler"]
    do_cfg = guidance_scale > 1.0
    prompt_embeds = reference_encode_prompt(tokenizer, text_encoder, prompt, device).to(dit_dtype)
    negative_embeds = (reference_encode_prompt(tokenizer, text_encoder, negative_prompt, device).to(dit_dtype)
                       if do_cfg else None)

    shape = (1, vae.config.z_dim, (num_frames - 1) // vae.config.scale_factor_temporal + 1,
             height // vae.config.scale_factor_spatial, width // vae.config.scale_factor_spatial)
    generator = [torch.Generator("cpu").manual_seed(seed)]
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=torch.float32)
    condition = reference_condition_latents(vae, latents, image, height, width).to(dit_dtype)

    scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1], device=device)
    for t in scheduler.timesteps:
        model_input = torch.cat([latents.to(dit_dtype), condition], dim=1)
        timestep = t.expand(model_input.shape[0])
        noise_pred = transformer(model_input, timestep, prompt_embeds)
        if do_cfg:
            noise_uncond = transformer(model_input, timestep, negative_embeds)
            guided = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            if image is not None and skip_first_latent_cfg:
                guided[:, :, :1] = noise_pred[:, :, :1]
            noise_pred = guided
        latents = scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    video = vae.decode((latents * std + mean).to(vae.dtype), return_dict=False)[0]
    return latents, (video.float() / 2 + 0.5).clamp(0, 1)


def load_reference_components(model_dir: Path, dit_dtype: torch.dtype) -> dict:
    import json

    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5TextModel

    transformer_dir = model_dir / "transformer"
    transformer = ReferencePAN2Transformer(**json.loads((transformer_dir / "config.json").read_text()))
    transformer.load_state_dict(load_file(str(transformer_dir / "diffusion_pytorch_model.safetensors")), strict=True)
    return {
        "tokenizer": AutoTokenizer.from_pretrained(model_dir / "tokenizer"),
        "text_encoder": Qwen3_5TextModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.float32).eval(),
        "transformer": transformer.to(dit_dtype).eval(),
        "vae": AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).eval(),
        "scheduler": FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler"),
    }
