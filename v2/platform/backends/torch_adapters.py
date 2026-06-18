"""Real torch component adapters for the GPU backend (design_v3 §17; README "Running the real models").

Wraps the real ``fastvideo.models.*`` modules (resolved from each card's ``load_id``, weights from
``ComponentSpec.checkpoint``) to the mini's narrow duck-typed surface the loops call:
    dit(latent, text_embed, sigma) -> velocity   ·   vae.decode(latent)/encode(video)   ·   text_encoder.encode(text)
The loops / CFG / scheduler math stay numpy fp32 (lowest-risk bring-up); the adapter marshals
numpy<->torch at its boundary. A torch-native end-to-end surface is the perf follow-up (Risk G).

Imported ONLY lazily — inside the cuda builder bodies in ``torch_cuda.py`` — so importing the platform
package on a CPU box never imports torch and the mini stays green.

Bring-up status: the construction layer (FastVideoArgs / loaders / dist-init / latent normalization)
and the interface contracts (DiT bare velocity, timestep=sigma*1000, UMT5-from-config) were CONFIRMED
on a 1x H100 box against Wan2.1-T2V-1.3B-Diffusers. Remaining ``# BRINGUP`` notes mark on-box-verified
facts and the few items still box-dependent (multi-GPU FSDP, full-resolution SSIM parity).
"""
from __future__ import annotations

import os

import numpy as np
import torch

# Flow-match timestep convention: the loop hands the adapter the raw sigma (1->0); the diffusers /
# FastVideo Wan convention embeds ``timestep = sigma * num_train_timesteps``. BRINGUP risk B: confirmed
# matching the FlowUniPC scheduler's continuous timestep on the box.
NUM_TRAIN_TIMESTEPS = 1000

_RUNTIME_READY = False


def _ensure_fastvideo_runtime() -> None:
    """Single-process distributed env the real fastvideo loaders require (mirrors
    ``fastvideo/worker/gpu_worker.py:init_device``). The loaders call ``get_local_torch_device()`` and
    ``maybe_load_fsdp_model`` builds a 1x1 device mesh, which needs an initialized process group.
    Idempotent — ``maybe_init_*`` returns early once initialized. BRINGUP risk A (single-GPU; multi-GPU
    FSDP sharding via tp/sp>1 is the follow-up)."""
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    _RUNTIME_READY = True


def _require_checkpoint(spec) -> str:
    ckpt = getattr(spec, "checkpoint", "") or ""
    if not ckpt:
        raise RuntimeError(
            f"component {spec.component_id!r}: set ComponentSpec.checkpoint to the weights path / HF id "
            f"for the GPU backend (load_id={spec.load_id!r}). See GPU_BRINGUP.md, risk A.")
    return ckpt


def _model_root(spec) -> str:
    """Model root = the dir holding ``model_index.json`` + component subfolders. ``spec.checkpoint`` is
    the component subfolder (e.g. ``<root>/transformer``); its parent is the root the pipeline-config
    registry resolves from (``_get_config_info`` reads ``model_index.json`` for a local path). BRINGUP
    risk A."""
    return os.path.dirname(os.path.normpath(_require_checkpoint(spec)))


def _fastvideo_args(spec):
    """Build the FastVideoArgs the real loaders need (BRINGUP risk A). ``from_kwargs`` populates
    ``pipeline_config`` (dit/vae/text-encoder configs + precisions) from the model root via the
    registry. Single-GPU inference, all CPU-offload / layerwise-offload / FSDP OFF — weights stay
    resident on one device, the simplest correct bring-up."""
    from fastvideo.fastvideo_args import FastVideoArgs
    return FastVideoArgs.from_kwargs(
        model_path=_model_root(spec), num_gpus=1, tp_size=1, sp_size=1,
        dit_cpu_offload=False, text_encoder_cpu_offload=False, vae_cpu_offload=False,
        image_encoder_cpu_offload=False, dit_layerwise_offload=False,
        use_fsdp_inference=False, pin_cpu_memory=False)


def _load_component(loader_attr: str, path: str, args):
    """Construct a real component via the FastVideo loaders (models/loader/component_loader.py) — the
    REAL entry point. ``WanTransformer3DModel`` / ``AutoencoderKLWan`` have NO ``from_pretrained``; the
    loader reads the checkpoint config, resolves the class (so UMT5 vs T5 is chosen from config, not a
    hardcode), instantiates, and loads weights. ``path`` is the component subfolder; ``load_id`` on the
    card is a hint, not the instantiation path."""
    from fastvideo.models.loader import component_loader as _cl
    return getattr(_cl, loader_attr)().load(path, args)


def _torch_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(name, torch.float32)


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
                encoder_hidden_states_image=None, guidance=None) -> velocity[B,C,T,H,W]  (bare tensor)
    """

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        # CausalWanTransformer3DModel (self-forcing student) conditions across chunks via an internal
        # kv_cache, NOT a forward arg — so the chunk_rollout loop's latent `context` is ignored here;
        # with no kv_cache passed it dispatches to its full-attention _forward_train. (Faithful
        # kv_cache streaming is a follow-up; this runs the real causal weights per chunk.)
        self.causal = "Causal" in type(module).__name__

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None):
        hs = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0)      # add batch dim
        ehs = _to_torch(text_embed, device=self.device, dtype=self.dtype)
        if ehs is not None:
            ehs = ehs.unsqueeze(0)
        # BRINGUP risk B (confirmed): timestep = sigma * 1000. Standard Wan takes a scalar; the causal
        # model needs a per-latent-frame timestep [B, num_frames] (uniform here — the chunk_rollout loop
        # denoises a whole chunk at one sigma, not the staggered per-frame causal schedule).
        ts = float(sigma) * NUM_TRAIN_TIMESTEPS
        timestep = (torch.full((1, hs.shape[2]), ts, device=self.device) if self.causal
                    else torch.tensor([ts], device=self.device))
        # ``context`` is an i2v image embedding for standard Wan; the causal model takes none (see __init__).
        img = None if self.causal else _to_torch(context, device=self.device, dtype=self.dtype)
        if img is not None:
            img = img.unsqueeze(0)
        # The FastVideo attention layer reads attn_metadata via get_forward_context(), so the DiT forward
        # (like the text encoder) must run inside set_forward_context(...). attn_metadata=None selects the
        # dense path for the SDPA backend — matches the real denoise stages (BRINGUP, confirmed on box).
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=0, attn_metadata=None):
            velocity = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                                   encoder_hidden_states_image=img)
        # BRINGUP risk C (confirmed): the output is rectified-flow velocity, matching the mini sampler's
        # x_next = x_t + (sigma_next - sigma_t)*v. A sign/orientation flip would denoise backward.
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

    The DiT operates in NORMALIZED latent space, so encode applies ``(z - mean) / std`` and decode
    inverts it. BRINGUP risk D RESOLVED on-box: ``AutoencoderKLWan.encode/decode`` operate in RAW
    latent space (no internal shift/scale); ``vae.latents_std`` is the actual std and ``vae.latents_mean``
    the mean. (The model also exposes ``shift_factor == latents_mean`` and ``scaling_factor == 1/std`` —
    the canonical pipeline buffers — so denormalizing with mean/std here is the whole job. The earlier
    code additionally re-added ``shift_factor`` on decode: a double-mean that washed out / saturated
    the video.)
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
        z = (z.float() - mean) * inv_std              # -> the normalized latent the DiT expects
        return _to_numpy(z.squeeze(0))

    @torch.no_grad()
    def decode(self, latent):
        z = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0).float()
        mean, inv_std = self._mean_invstd(z)
        # Invert the encode-side normalization: raw = normalized / inv_std + mean = normalized*std + mean.
        # AutoencoderKLWan.decode then consumes the RAW latent. (No extra shift_factor — see class doc,
        # BRINGUP risk D.)
        z = z / inv_std + mean
        video = self.module.decode(z.to(self.dtype))  # -> video [B,3,T,H,W] in [-1,1] (wanvae.py:1240)
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
        # BRINGUP risk E: UMT5 resolved from config; tokenize to real tokens (max_length=512 cap).
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
        hidden = hidden.squeeze(0)                                  # [seq, dim]
        # Wan convention (configs/pipelines/wan.py:t5_postprocess_text): the DiT cross-attends over a
        # FIXED-length text sequence — the real-token rows followed by a ZERO-padded tail to text_len.
        # A short unpadded sequence mis-conditions the DiT (degraded / washed-out video).
        if hidden.shape[0] < self.max_length:
            pad = hidden.new_zeros(self.max_length - hidden.shape[0], hidden.shape[1])
            hidden = torch.cat([hidden, pad], dim=0)
        return _to_numpy(hidden)                                    # [text_len, dim]


# --------------------------------------------------------------------------- #
# Builders (called lazily by torch_cuda.py; receive (spec, instance, platform)) #
# --------------------------------------------------------------------------- #
def _device(platform) -> str:
    return "cuda" if platform.device == "cuda" else platform.device


def _native_dtype(module):
    """Keep each real component at the precision its loader produced (Wan: DiT bf16, VAE/text fp32 per
    the pipeline config) rather than upcasting to the card's uniform fp32. This is faithful to how
    fastvideo runs the model and ~2x faster/smaller for the DiT. The loop math stays numpy fp32; the
    adapter marshals fp32<->native at its boundary."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def build_torch_dit(spec, instance, platform):
    ckpt = _require_checkpoint(spec)          # fail fast on a mis-stamped card, before any dist init
    _ensure_fastvideo_runtime()
    module = _load_component("TransformerLoader", ckpt, _fastvideo_args(spec))
    device, dtype = _device(platform), _native_dtype(module)
    if "LTX2" in type(module).__name__:                          # LTX2Transformer3DModel
        from .torch_ltx2 import TorchLTX2DiT
        return TorchLTX2DiT(module, device=device, dtype=dtype)
    return TorchWanDiT(module, device=device, dtype=dtype)       # WanTransformer3DModel / Causal*


def build_torch_vae(spec, instance, platform):
    ckpt = _require_checkpoint(spec)
    _ensure_fastvideo_runtime()
    module = _load_component("VAELoader", ckpt, _fastvideo_args(spec))
    device, dtype = _device(platform), _native_dtype(module)
    if "LTX2" in type(module).__name__:                          # LTX2CausalVideoAutoencoder
        from .torch_ltx2 import TorchLTX2VAE
        return TorchLTX2VAE(module, device=device, dtype=dtype)
    return TorchWanVAE(module, device=device, dtype=dtype)


def build_torch_text_encoder(spec, instance, platform):
    ckpt = _require_checkpoint(spec)
    _ensure_fastvideo_runtime()
    args = _fastvideo_args(spec)
    module = _load_component("TextEncoderLoader", ckpt, args)        # text_encoder subfolder
    # The tokenizer is a SIBLING subfolder (<root>/tokenizer), not under text_encoder/. BRINGUP risk E.
    tokenizer = _load_component("TokenizerLoader", os.path.join(_model_root(spec), "tokenizer"), args)
    device, dtype = _device(platform), _native_dtype(module)
    if "Gemma" in type(module).__name__:                          # LTX2GemmaTextEncoderModel
        from .torch_ltx2 import TorchGemma
        return TorchGemma(module, tokenizer, device=device, dtype=dtype)
    return TorchT5Encoder(module, tokenizer, device=device, dtype=dtype)


def build_torch_upsampler(spec, instance, platform):
    ckpt = _require_checkpoint(spec)
    _ensure_fastvideo_runtime()
    module = _load_component("UpsamplerLoader", ckpt, _fastvideo_args(spec))   # LTX2LatentUpsampler
    device = _device(platform)
    # The LTX-2 spatial upsampler normalizes through the video VAE's per-channel statistics
    # (upsample_video: un_normalize -> learned upsample -> normalize), so it must share the VAE's
    # dtype/device. Fetch (build-if-needed) the VAE adapter and hand its real module to the upsampler.
    vae = instance.component("vae")
    dtype = getattr(vae, "dtype", _native_dtype(module))
    vae_module = getattr(vae, "module", None)
    # ``per_channel_statistics`` lives on the AE's decoder/encoder sub-modules, NOT the top-level
    # LTX2CausalVideoAutoencoder — upsample_video needs an object exposing it (same stats either side).
    stats_owner = getattr(vae_module, "decoder", None) or getattr(vae_module, "encoder", None) or vae_module
    from .torch_ltx2 import TorchLTX2Upsampler
    return TorchLTX2Upsampler(module, stats_owner, device=device, dtype=dtype)
