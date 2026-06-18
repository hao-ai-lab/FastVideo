"""Real torch adapters for LTX-2 (design_v3 §15 two-stage distilled). Imported lazily on a GPU box.

LTX-2 differs from Wan in three ways the adapters absorb so the v2 loops keep their numpy
``dit(latent,text,sigma)->velocity`` / ``vae.decode`` / ``text_encoder.encode`` surface:
  * DiT (``LTX2Transformer3DModel``) patchifies internally and takes a PER-TOKEN timestep
    ``ones(B, token_count, 1) * sigma`` (sigma direct, 0..1 — NOT sigma*1000) plus a per-sample
    ``video_sigma``; it predicts ``denoised`` (x0), so the adapter returns the flow-match velocity
    ``(x_t - x0)/sigma`` the loop integrates.
  * VAE (``LTX2CausalVideoAutoencoder``) un-normalizes internally on decode (per-channel stats).
  * Text encoder (``LTX2GemmaTextEncoderModel``) runs Gemma + feature-extractor + connectors
    (the projection) inside its forward, returning the projected ``last_hidden_state``.
All run at the model's native precision (bf16 per LTX2T2VConfig). BRINGUP: confirm on box.
"""
from __future__ import annotations

import torch

from .torch_adapters import _to_numpy, _to_torch


class TorchLTX2DiT:
    """dit(latent[C,T,H,W], text_embed[seq,dim], sigma) -> velocity[C,T,H,W] (numpy in/out)."""

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        from fastvideo.models.dits.ltx2 import VideoLatentShape
        self._VideoLatentShape = VideoLatentShape

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None):
        hs = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0)   # [1,C,T,H,W]
        ehs = _to_torch(text_embed, device=self.device, dtype=self.dtype)
        if ehs is not None:
            ehs = ehs.unsqueeze(0)
        s = float(sigma)
        vshape = self._VideoLatentShape.from_torch_shape(tuple(hs.shape))
        token_count = self.module.patchifier.get_token_count(vshape)
        # per-token timestep = template * sigma (sigma direct, 0..1); per-sample video_sigma.
        timestep = torch.full((1, token_count, 1), s, device=self.device, dtype=torch.float32)
        video_sigma = torch.tensor([s], device=self.device, dtype=torch.float32)
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=s, attn_metadata=None):
            denoised = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                                   video_sigma=video_sigma, encoder_attention_mask=None)
        if isinstance(denoised, tuple):
            denoised = denoised[0]                       # t2v: video_out only (audio path unused)
        # LTX-2 DiT predicts x0; the v2 loop integrates velocity. velocity = (x_t - x0)/sigma.
        velocity = (hs.float() - denoised.float()) / max(s, 1e-6)
        return _to_numpy(velocity.squeeze(0))

    def copy_from(self, other):
        self.module.load_state_dict(other.module.state_dict())

    def clone(self):
        import copy
        c = TorchLTX2DiT.__new__(TorchLTX2DiT)
        c.module = copy.deepcopy(self.module)
        c.device, c.dtype, c._VideoLatentShape = self.device, self.dtype, self._VideoLatentShape
        return c

    def mse_grad_step(self, *a, **k):
        raise NotImplementedError("GPU training surface (mse_grad_step) is a separate workstream (Risk F)")


class TorchLTX2VAE:
    """vae.decode(latent[C,T,H,W]) -> video[3,T,H,W] in [-1,1]. The LTX2 VideoDecoder un-normalizes
    internally (per-channel statistics), so the adapter just marshals numpy<->torch."""

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def decode(self, latent):
        z = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0)
        video = self.module.decode(z)
        if hasattr(video, "sample"):
            video = video.sample
        return _to_numpy(video.squeeze(0))

    @torch.no_grad()
    def encode(self, video):
        x = _to_torch(video, device=self.device, dtype=self.dtype).unsqueeze(0)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return _to_numpy(z.squeeze(0))


class TorchGemma:
    """text_encoder.encode(text) -> embedding[seq,dim] (numpy out). LTX2GemmaTextEncoderModel runs
    Gemma + feature-extractor + connectors (projection) in its forward and returns last_hidden_state."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 256):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.device, self.dtype, self.max_length = device, dtype, max_length

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding="max_length")
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return _to_numpy(hidden.squeeze(0))
