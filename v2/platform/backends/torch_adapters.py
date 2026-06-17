"""Real torch component adapters for the GPU backend (design_v3 §17; README "Running the real models").

WRITTEN-NOT-RUN. This module imports ``torch`` and the real ``fastvideo`` package — NEITHER is
available in the CPU-test environment, so this code cannot be executed or verified here. It is
grounded in the verbatim real APIs (cited inline), and every place the real API must be confirmed on
a GPU box is marked ``# BRINGUP``. See ``GPU_BRINGUP.md`` for the ordered checklist and the risk list.

It is imported ONLY lazily — inside the cuda builder bodies in ``torch_cuda.py`` — so importing the
platform package on a CPU box never imports torch and the mini stays green.

Each adapter wraps a real module (resolved from ``spec.load_id`` = ``"module:Class"``) to expose the
mini's narrow duck-typed surface the loops call:
    dit(latent, text_embed, sigma) -> velocity   ·   vae.decode(latent)/encode(video)   ·   text_encoder.encode(text)
The loops / CFG / scheduler math stay numpy fp32 (lowest-risk bring-up); the adapter marshals
numpy<->torch at its boundary. A torch-native end-to-end surface is the perf follow-up (Risk G).
"""
from __future__ import annotations

import numpy as np
import torch

# Flow-match timestep convention: the loop hands the adapter the raw sigma (1→0); the diffusers /
# FastVideo Wan convention embeds ``timestep = sigma * num_train_timesteps``. BRINGUP risk B: confirm
# the exact convention (continuous ``sigma*1000`` vs a discrete scheduler index) on the box.
NUM_TRAIN_TIMESTEPS = 1000


def _fastvideo_args(spec):
    # BRINGUP risk A: the real loaders take a FastVideoArgs (model config / precision / parallelism),
    # not just a path. Confirm the exact constructor + required fields on the box.
    from fastvideo.fastvideo_args import FastVideoArgs
    return FastVideoArgs(model_path=spec.checkpoint)


def _load_via_fastvideo(loader_attr: str, spec):
    """Construct a real component via the FastVideo loaders (models/loader/component_loader.py) — the
    REAL entry point. ``WanTransformer3DModel`` / ``AutoencoderKLWan`` have NO ``from_pretrained``; the
    loader reads the checkpoint config, resolves the right class (so e.g. UMT5 vs T5 is chosen from the
    config, not hardcoded), instantiates with that config, and loads weights (incl. FSDP). The card's
    ``load_id`` is therefore a hint, not the instantiation path."""
    from fastvideo.models.loader import component_loader as _cl
    loader = getattr(_cl, loader_attr)()
    return loader.load(spec.checkpoint, _fastvideo_args(spec))


def _torch_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(name, torch.float32)


def _require_checkpoint(spec) -> str:
    ckpt = getattr(spec, "checkpoint", "") or ""
    if not ckpt:
        raise RuntimeError(
            f"component {spec.component_id!r}: set ComponentSpec.checkpoint to the weights path / HF id "
            f"for the GPU backend (load_id={spec.load_id!r}). See GPU_BRINGUP.md, risk A.")
    return ckpt


def _to_torch(a, *, device, dtype):
    return None if a is None else torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _to_numpy(t):
    # loop / CFG / scheduler math stay numpy fp32; marshal back at the boundary.
    if hasattr(t, "sample"):            # BRINGUP: some heads wrap output in an object with .sample
        t = t.sample
    return t.detach().to("cpu", torch.float32).numpy()


# --------------------------------------------------------------------------- #
# DiT — WanTransformer3DModel (fastvideo/models/dits/wanvideo.py:561)          #
# --------------------------------------------------------------------------- #
class TorchWanDiT:
    """dit(latent[C,T,H,W], text_embed[seq,dim], sigma) -> velocity[C,T,H,W] (numpy in/out).

    Bridges to the real forward (wanvideo.py:632):
        forward(hidden_states[B,C,T,H,W], encoder_hidden_states[B,seq,dim], timestep,
                encoder_hidden_states_image=None, guidance=None) -> velocity[B,C,T,H,W]
    """

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None):
        hs = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0)      # add batch dim
        ehs = _to_torch(text_embed, device=self.device, dtype=self.dtype)
        if ehs is not None:
            ehs = ehs.unsqueeze(0)
        # BRINGUP risk B: timestep = sigma * 1000. Confirm dtype/convention for Wan on the box.
        timestep = torch.tensor([float(sigma) * NUM_TRAIN_TIMESTEPS], device=self.device)
        img = _to_torch(context, device=self.device, dtype=self.dtype)
        if img is not None:
            img = img.unsqueeze(0)
        velocity = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                               encoder_hidden_states_image=img)
        # BRINGUP risk C: confirm the output is rectified-flow velocity (noise − clean), matching the
        # mini sampler's x0 = x_t − sigma·v. A sign/orientation flip denoises backward.
        return _to_numpy(velocity.squeeze(0))

    # --- weight surface used by serving weight-sync + training (design_v3 §10) ----------------- #
    def copy_from(self, other: "TorchWanDiT") -> None:
        self.module.load_state_dict(other.module.state_dict())

    def blend_from(self, other: "TorchWanDiT", decay: float) -> None:    # EMA / decayed-old-policy
        with torch.no_grad():
            for p, q in zip(self.module.parameters(), other.module.parameters()):
                p.mul_(decay).add_(q, alpha=1.0 - decay)

    def clone(self) -> "TorchWanDiT":
        import copy
        c = TorchWanDiT.__new__(TorchWanDiT)
        c.module = copy.deepcopy(self.module)
        c.device, c.dtype = self.device, self.dtype
        return c

    def mse_grad_step(self, *a, **k):
        # BRINGUP risk F: the toy did an inline numpy SGD step; a torch module needs an
        # optimizer/autograd training loop. RL/distill on the cuda rung is a separate workstream.
        raise NotImplementedError(
            "GPU training surface (mse_grad_step) is a separate workstream — see GPU_BRINGUP.md risk F")


# --------------------------------------------------------------------------- #
# VAE — AutoencoderKLWan (fastvideo/models/vaes/wanvae.py)                     #
# --------------------------------------------------------------------------- #
class TorchWanVAE:
    """vae.decode(latent[C,T,H,W]) -> video; vae.encode(video) -> latent (numpy in/out).

    The DiT operates in NORMALIZED latent space, so encode applies ``(z - mean) * inv_std`` and decode
    inverts it. FastVideo stores ``latents_std`` as its RECIPROCAL — the encode transform is
    ``(z - latents_mean) * latents_std`` with ``latents_std = 1/std`` (training_utils.py:801-806).
    Skipping this yields washed-out / saturated video, not an error (BRINGUP risk D).
    """

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype

    def _mean_invstd(self, like):
        mean = torch.tensor(self.module.latents_mean, device=like.device,
                            dtype=torch.float32).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(self.module.latents_std, device=like.device,
                                      dtype=torch.float32)).view(1, -1, 1, 1, 1)
        return mean, inv_std

    @torch.no_grad()
    def encode(self, video):
        x = _to_torch(video, device=self.device, dtype=self.dtype).unsqueeze(0)
        dist = self.module.encode(x)                  # DiagonalGaussianDistribution (wanvae.py:1178)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        mean, inv_std = self._mean_invstd(z)
        z = (z.float() - mean) * inv_std              # → the normalized latent the DiT expects
        return _to_numpy(z.squeeze(0))

    @torch.no_grad()
    def decode(self, latent):
        z = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0).float()
        mean, inv_std = self._mean_invstd(z)
        z = z / inv_std + mean                        # invert the encode-side normalization
        shift = getattr(self.module, "shift_factor", None)   # BRINGUP risk D: confirm placement/sign
        if shift is not None:
            z = z + (shift.to(z) if torch.is_tensor(shift) else shift)
        video = self.module.decode(z.to(self.dtype))  # → video [B,3,T,H,W] in [-1,1] (wanvae.py:1240)
        return _to_numpy(video.squeeze(0))


# --------------------------------------------------------------------------- #
# Text encoder — (U)MT5 (fastvideo/models/encoders/t5.py)                      #
# --------------------------------------------------------------------------- #
class TorchT5Encoder:
    """text_encoder.encode(text) -> embedding[seq,dim] (numpy out)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 512):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.device, self.dtype, self.max_length = device, dtype, max_length

    @torch.no_grad()
    def encode(self, text):
        # BRINGUP risk E: the exact tokenizer kwargs come from the model config (text_len/max_length,
        # special tokens). Don't force padding="max_length" — that is not in the real kwargs.
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        # REQUIRED: the FastVideo T5/UMT5 attention reads global state via get_forward_context(), so the
        # forward must run inside set_forward_context(...) (forward_context.py:54; text_encoding.py:273).
        # A bare call reads stale/None context and fails or mis-encodes.
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return _to_numpy(hidden.squeeze(0))


# --------------------------------------------------------------------------- #
# Builders (called lazily by torch_cuda.py; receive (spec, instance, platform)) #
# --------------------------------------------------------------------------- #
def _device_dtype(spec, instance, platform):
    device = "cuda" if platform.device == "cuda" else platform.device
    dtype = _torch_dtype(instance.card.precision.dtype_for(spec.component_id))
    return device, dtype


def build_torch_dit(spec, instance, platform):
    _require_checkpoint(spec)
    device, dtype = _device_dtype(spec, instance, platform)
    module = _load_via_fastvideo("TransformerLoader", spec)     # NOT from_pretrained (it has none)
    return TorchWanDiT(module, device=device, dtype=dtype)


def build_torch_vae(spec, instance, platform):
    _require_checkpoint(spec)
    device, dtype = _device_dtype(spec, instance, platform)
    module = _load_via_fastvideo("VAELoader", spec)
    return TorchWanVAE(module, device=device, dtype=dtype)


def build_torch_text_encoder(spec, instance, platform):
    _require_checkpoint(spec)
    device, dtype = _device_dtype(spec, instance, platform)
    module = _load_via_fastvideo("TextEncoderLoader", spec)     # resolves UMT5 vs T5 from the config
    tokenizer = _load_via_fastvideo("TokenizerLoader", spec)
    return TorchT5Encoder(module, tokenizer, device=device, dtype=dtype)
