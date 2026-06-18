"""Real torch component adapters for the GPU backend (design_v3 §17; README "Running the real models").

ONE cohesive module (was the torch_adapters.py + torch_ltx2.py split): a ``TorchComponent`` base that
centralizes the shared mechanics — ``.to(device,dtype).eval()``, the numpy<->torch marshalling at the
loop boundary, the ``set_forward_context`` wrap every fastvideo forward needs, and the weight surface
(copy_from / blend_from / clone) — plus thin per-model subclasses that carry ONLY the forward semantics
(Wan ``sigma*1000`` -> velocity; LTX-2 per-token sigma, x0 -> ``(x_t-x0)/sigma``, joint A/V; VAE
normalization; T5 padding; Gemma dual-projection; upsampler; audio decode->vocoder). A single
``build_component(spec, instance, platform)`` dispatches by ``spec.kind`` through ``_MAKERS`` (replacing
the six ``build_torch_*`` builders + six lazy trampolines).

The loops / CFG / scheduler math stay numpy fp32 (lowest-risk bring-up); the boundary marshals
numpy<->torch in ONE place (``TorchComponent._t``/``_n``) so a torch-native path is a later swap.
Component construction goes through ``load_component`` — the v2-owned loader seam (currently delegates to
fastvideo's component_loader; a per-module vendored cutover replaces it later, no caller changes).

Imported ONLY lazily — inside ``torch_cuda.py``'s registered builder — so importing the platform package
on a CPU box never imports torch and the mini stays green. Wan2.1 + LTX-2 A/V GPU-verified.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# Flow-match timestep convention: the loop hands the raw sigma (1->0); the diffusers/FastVideo Wan
# convention embeds ``timestep = sigma * num_train_timesteps`` (BRINGUP risk B, confirmed on box).
NUM_TRAIN_TIMESTEPS = 1000

_RUNTIME_READY = False


# --------------------------------------------------------------------------- #
# Construction helpers (FastVideoArgs / loaders / dist-init / marshalling)      #
# --------------------------------------------------------------------------- #
def _ensure_fastvideo_runtime() -> None:
    """Single-process distributed env the real fastvideo loaders require (mirrors
    ``fastvideo/worker/gpu_worker.py:init_device``): the loaders call ``get_local_torch_device()`` and
    build a 1x1 device mesh, which needs an initialized process group. Idempotent. BRINGUP risk A."""
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
    from v2.distributed import maybe_init_distributed_environment_and_model_parallel
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
    the component subfolder (e.g. ``<root>/transformer``); its parent is the registry root. BRINGUP A."""
    return os.path.dirname(os.path.normpath(_require_checkpoint(spec)))


def _fastvideo_args(spec):
    """Build the FastVideoArgs the real loaders need (BRINGUP risk A). ``from_kwargs`` populates
    ``pipeline_config`` (dit/vae/text-encoder configs + precisions) from the model root. Single-GPU,
    all offload/FSDP OFF — weights resident on one device, the simplest correct bring-up."""
    from v2.fastvideo_args import FastVideoArgs
    return FastVideoArgs.from_kwargs(
        model_path=_model_root(spec), num_gpus=1, tp_size=1, sp_size=1,
        dit_cpu_offload=False, text_encoder_cpu_offload=False, vae_cpu_offload=False,
        image_encoder_cpu_offload=False, dit_layerwise_offload=False,
        use_fsdp_inference=False, pin_cpu_memory=False)


def load_component(loader_attr: str, path: str, args):
    """v2-owned loader seam (design: ``v2/loader``). Constructs a real component from a checkpoint via the
    component loaders. ``WanTransformer3DModel`` / ``AutoencoderKLWan`` have NO ``from_pretrained``; the
    loader reads the checkpoint config, resolves the class (UMT5 vs T5 from config), and loads weights.
    Currently delegates to fastvideo's loader; a vendored cutover swaps the body, not the callers."""
    from v2.loader import component_loader as _cl
    return getattr(_cl, loader_attr)().load(path, args)


def _device(platform) -> str:
    return "cuda" if platform.device == "cuda" else platform.device


def _native_dtype(module):
    """Keep each component at the precision its loader produced (Wan DiT bf16, VAE/text fp32) rather than
    the card's uniform fp32 — faithful to fastvideo and ~2x faster/smaller for the DiT."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _to_torch(a, *, device, dtype):
    return None if a is None else torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _to_numpy(t):
    if hasattr(t, "sample"):            # some heads wrap output in an object with .sample
        t = t.sample
    return t.detach().to("cpu", torch.float32).numpy()


# --------------------------------------------------------------------------- #
# TorchComponent — the shared base (eval/marshalling/forward-context/weights)  #
# --------------------------------------------------------------------------- #
class TorchComponent:
    """Wraps a real ``fastvideo.models.*`` module to the mini's numpy duck-typed surface. Subclasses
    override only the forward semantics (``__call__`` / ``encode`` / ``decode`` / ``upsample`` / ...)."""

    def __init__(self, module, *, device, dtype, eager: bool = True):
        self.device, self.dtype = device, dtype
        self.module = module.to(device=device, dtype=dtype).eval() if eager else module

    # numpy<->torch marshalling at the loop boundary (ONE place — torch-native is a later swap) ------- #
    def _t(self, a, *, batch: bool = True):
        t = _to_torch(a, device=self.device, dtype=self.dtype)
        return t.unsqueeze(0) if (t is not None and batch) else t

    @staticmethod
    def _n(t):
        return _to_numpy(t.squeeze(0))

    @staticmethod
    def _ctx(current_timestep=0):
        """The FastVideo attention layer reads attn_metadata via get_forward_context(), so every forward
        runs inside set_forward_context(...). attn_metadata=None selects the dense SDPA path."""
        from v2.forward_context import set_forward_context
        return set_forward_context(current_timestep=current_timestep, attn_metadata=None)

    # weight surface used by serving weight-sync + training (design_v3 §10) ------------------------- #
    def copy_from(self, other) -> None:
        self.module.load_state_dict(other.module.state_dict())

    def blend_from(self, other, decay: float) -> None:     # EMA / decayed-old-policy
        with torch.no_grad():
            for p, q in zip(self.module.parameters(), other.module.parameters()):
                p.mul_(decay).add_(q, alpha=1.0 - decay)

    def clone(self):
        import copy
        c = self.__class__.__new__(self.__class__)
        c.__dict__.update(self.__dict__)         # tokenizer / shapes / sibling refs shared
        c.module = copy.deepcopy(self.module)     # independent weights
        return c

    def mse_grad_step(self, *a, **k):
        raise NotImplementedError(
            "GPU training surface (mse_grad_step) is a separate workstream — see GPU_BRINGUP.md risk F")


# --------------------------------------------------------------------------- #
# DiT adapters                                                                  #
# --------------------------------------------------------------------------- #
class WanDiT(TorchComponent):
    """dit(latent[C,T,H,W], text_embed[seq,dim], sigma) -> velocity[C,T,H,W]. Real forward
    (wanvideo.py): forward(hidden_states[B,C,T,H,W], encoder_hidden_states, timestep,
    encoder_hidden_states_image=None) -> velocity (bare tensor)."""

    def __init__(self, module, *, device, dtype, offload_group=None, component_id="transformer"):
        # Wan2.2 MoE (A14B): two 14B experts don't both fit one 80GB GPU. With ``offload_group`` set, keep
        # this expert on CPU and bring only the *active* one onto the GPU on demand (single swap at the
        # boundary, not per-step thrash). Single-expert Wan stays resident (offload_group=None).
        self.offload_group = offload_group
        self.component_id = component_id
        super().__init__(module, device=device, dtype=dtype, eager=(offload_group is None))
        if offload_group is not None:
            self.module = self.module.to(device="cpu", dtype=dtype).eval()
            self._on_gpu = False
            offload_group[component_id] = self
        else:
            self._on_gpu = True
        # CausalWanTransformer3DModel (self-forcing student) conditions across chunks via an internal
        # kv_cache, not a forward arg, so the chunk_rollout loop's latent ``context`` is ignored here.
        self.causal = "Causal" in type(module).__name__

    def _ensure_resident(self) -> None:
        if self.offload_group is None or self._on_gpu:
            return
        for other in self.offload_group.values():
            if other is not self and other._on_gpu:
                other.module.to("cpu")
                other._on_gpu = False
        torch.cuda.empty_cache()
        self.module.to(self.device)
        self._on_gpu = True

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None):
        self._ensure_resident()
        hs = self._t(latent)
        ehs = self._t(text_embed)
        # timestep = sigma*1000 (BRINGUP risk B). Causal model needs per-latent-frame [B, num_frames].
        ts = float(sigma) * NUM_TRAIN_TIMESTEPS
        timestep = (torch.full((1, hs.shape[2]), ts, device=self.device) if self.causal
                    else torch.tensor([ts], device=self.device))
        img = None if self.causal else self._t(context)   # i2v image embedding for standard Wan
        with self._ctx():
            velocity = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                                   encoder_hidden_states_image=img)
        return self._n(velocity)                           # rectified-flow velocity (BRINGUP risk C)

    def clone(self):
        c = super().clone()
        c.offload_group, c._on_gpu = None, True            # standalone resident copy
        return c


class LTX2DiT(TorchComponent):
    """LTX-2 DiT: patchifies internally, takes a PER-TOKEN timestep ``ones(B, token_count, 1) * sigma``
    (sigma direct, 0..1) + a per-sample ``video_sigma``; predicts ``denoised`` (x0), so the adapter
    returns the flow-match velocity ``(x_t - x0)/sigma`` the loop integrates. Joint A/V (audio_latent
    given) -> (video_velocity, audio_velocity) in one forward."""

    def __init__(self, module, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        from v2.models.dits.ltx2 import VideoLatentShape
        self._VideoLatentShape = VideoLatentShape

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, audio_latent=None, audio_text=None):
        hs = self._t(latent)
        ehs = self._t(text_embed)
        s = float(sigma)
        token_count = self.module.patchifier.get_token_count(self._VideoLatentShape.from_torch_shape(tuple(hs.shape)))
        timestep = torch.full((1, token_count, 1), s, device=self.device, dtype=torch.float32)
        video_sigma = torch.tensor([s], device=self.device, dtype=torch.float32)
        av_kwargs: dict = {}
        au = None
        if audio_latent is not None:                       # joint A/V: audio latent [1,8,T,16] + audio text
            from v2.models.audio.ltx2_audio_vae import AudioLatentShape
            au = self._t(audio_latent)
            aeh = self._t(audio_text)
            atok = self.module.audio_patchifier.get_token_count(AudioLatentShape.from_torch_shape(tuple(au.shape)))
            av_kwargs = dict(
                audio_hidden_states=au, audio_encoder_hidden_states=aeh,
                audio_timestep=torch.full((1, atok, 1), s, device=self.device, dtype=torch.float32),
                audio_sigma=torch.tensor([s], device=self.device, dtype=torch.float32))
        with self._ctx(current_timestep=s):
            out = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                              video_sigma=video_sigma, encoder_attention_mask=None, **av_kwargs)
        if au is not None:                                 # LTX-2 DiT predicts x0; integrate velocity
            denoised_v, denoised_a = out
            vel_v = (hs.float() - denoised_v.float()) / max(s, 1e-6)
            vel_a = (au.float() - denoised_a.float()) / max(s, 1e-6)
            return self._n(vel_v), self._n(vel_a)
        denoised = out[0] if isinstance(out, tuple) else out
        return self._n((hs.float() - denoised.float()) / max(s, 1e-6))


# --------------------------------------------------------------------------- #
# VAE adapters                                                                  #
# --------------------------------------------------------------------------- #
class WanVAE(TorchComponent):
    """The DiT operates in NORMALIZED latent space, so encode applies ``(z - mean)/std`` and decode
    inverts it; ``AutoencoderKLWan.encode/decode`` operate in RAW latent space (BRINGUP risk D)."""

    def _mean_invstd(self, like):
        mean = torch.tensor(self.module.latents_mean, device=like.device,
                            dtype=torch.float32).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(self.module.latents_std, device=like.device,
                                      dtype=torch.float32)).view(1, -1, 1, 1, 1)
        return mean, inv_std

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        mean, inv_std = self._mean_invstd(z)
        return self._n((z.float() - mean) * inv_std)       # -> the normalized latent the DiT expects

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float()
        mean, inv_std = self._mean_invstd(z)
        z = z / inv_std + mean                             # invert encode normalization -> raw latent
        video = self.module.decode(z.to(self.dtype))       # -> video [B,3,T,H,W] in [-1,1]
        return self._n(video)


class LTX2VAE(TorchComponent):
    """LTX-2 VideoDecoder un-normalizes internally (per-channel stats); the adapter just marshals."""

    @torch.no_grad()
    def decode(self, latent):
        video = self.module.decode(self._t(latent))
        if hasattr(video, "sample"):
            video = video.sample
        return self._n(video)

    @torch.no_grad()
    def encode(self, video):
        dist = self.module.encode(self._t(video))
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z)


# --------------------------------------------------------------------------- #
# Text encoders                                                                 #
# --------------------------------------------------------------------------- #
class T5Encoder(TorchComponent):
    """(U)MT5 text_encoder.encode(text) -> embedding[text_len, dim] (numpy out)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 512):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        # Wan convention: the DiT cross-attends over a FIXED-length text sequence — real-token rows then
        # a ZERO-padded tail to text_len (a short unpadded sequence mis-conditions the DiT).
        if hidden.shape[0] < self.max_length:
            pad = hidden.new_zeros(self.max_length - hidden.shape[0], hidden.shape[1])
            hidden = torch.cat([hidden, pad], dim=0)
        return _to_numpy(hidden)


class Gemma(TorchComponent):
    """LTX-2 Gemma text encoder. ``encode`` -> video projection; ``encode_av`` -> (video, audio): the 2.3
    connector emits SEPARATE projections (audio in ``hidden_states[0]`` only when output_hidden_states)."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 256):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tok(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding="max_length")
        return toks.input_ids.to(self.device), toks.attention_mask.to(self.device)

    @torch.no_grad()
    def encode(self, text):
        ids, mask = self._tok(text)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return _to_numpy(hidden.squeeze(0))

    @torch.no_grad()
    def encode_av(self, text):
        ids, mask = self._tok(text)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        video = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hs = getattr(out, "hidden_states", None)
        audio = hs[0] if hs else video                     # separate audio connector projection
        return _to_numpy(video.squeeze(0)), _to_numpy(audio.squeeze(0))


# --------------------------------------------------------------------------- #
# Upsampler + audio (LTX-2)                                                     #
# --------------------------------------------------------------------------- #
class LTX2Upsampler(TorchComponent):
    """upsample(latent[C,T,H,W]) -> [C,T,2H,2W]. Applies the repo's ``upsample_video``: un_normalize (via
    the video VAE's per_channel_statistics) -> learned 2x upsample -> normalize."""

    def __init__(self, module, stats_owner, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.stats_owner = stats_owner          # exposes per_channel_statistics (the VAE decoder/encoder)

    @torch.no_grad()
    def upsample(self, latent):
        from v2.models.upsamplers.ltx2_upsampler import upsample_video
        out = upsample_video(self._t(latent), self.stats_owner, self.module)
        return self._n(out)


class LTX2Vocoder(TorchComponent):
    """Thin wrapper over the LTX-2 Vocoder (mel -> waveform @24kHz); chained by LTX2AudioVAE."""


class LTX2AudioVAE(TorchComponent):
    """audio_vae.decode(audio_latent[8,T,16]) -> waveform. Runs AudioDecoder (latent->mel) then the
    Vocoder (mel->waveform). ``sample_rate`` = the vocoder's output rate (so a saver writes the right wav)."""

    def __init__(self, module, vocoder, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.vocoder = vocoder                  # LTX2Vocoder | None
        self.sample_rate = int(getattr(getattr(vocoder, "module", None), "output_sample_rate", 24000))

    @torch.no_grad()
    def decode(self, audio_latent):
        mel = self.module(self._t(audio_latent))
        if hasattr(mel, "sample"):
            mel = mel.sample
        if self.vocoder is not None:
            wav = self.vocoder.module(mel)
            if hasattr(wav, "sample"):
                wav = wav.sample
        else:
            wav = mel
        return self._n(wav)


# --------------------------------------------------------------------------- #
# build_component — one dispatch (replaces the six build_torch_* + trampolines) #
# --------------------------------------------------------------------------- #
def _make_dit(spec, instance, platform, args):
    module = load_component("TransformerLoader", spec.checkpoint, args)
    device, dtype = _device(platform), _native_dtype(module)
    if "LTX2" in type(module).__name__:
        return LTX2DiT(module, device=device, dtype=dtype)
    # Wan2.2 MoE: >1 DiT expert -> CPU-offload all but the active one (swapped at the boundary).
    n_experts = sum(1 for c in instance.card.components.values() if getattr(c, "kind", None) == "dit")
    grp = None
    if n_experts > 1:
        grp = getattr(instance, "_dit_offload_group", None)
        if grp is None:
            grp = instance._dit_offload_group = {}
    return WanDiT(module, device=device, dtype=dtype, offload_group=grp, component_id=spec.component_id)


def _make_vae(spec, instance, platform, args):
    module = load_component("VAELoader", spec.checkpoint, args)
    device, dtype = _device(platform), _native_dtype(module)
    cls = LTX2VAE if "LTX2" in type(module).__name__ else WanVAE
    return cls(module, device=device, dtype=dtype)


def _make_text_encoder(spec, instance, platform, args):
    module = load_component("TextEncoderLoader", spec.checkpoint, args)
    tokenizer = load_component("TokenizerLoader", os.path.join(_model_root(spec), "tokenizer"), args)
    device, dtype = _device(platform), _native_dtype(module)
    cls = Gemma if "Gemma" in type(module).__name__ else T5Encoder
    return cls(module, tokenizer, device=device, dtype=dtype)


def _make_upsampler(spec, instance, platform, args):
    module = load_component("UpsamplerLoader", spec.checkpoint, args)   # LTX2LatentUpsampler
    vae = instance.component("vae")
    dtype = getattr(vae, "dtype", _native_dtype(module))
    vae_module = getattr(vae, "module", None)
    # per_channel_statistics lives on the AE's decoder/encoder sub-module, not the top-level autoencoder.
    stats_owner = getattr(vae_module, "decoder", None) or getattr(vae_module, "encoder", None) or vae_module
    return LTX2Upsampler(module, stats_owner, device=_device(platform), dtype=dtype)


def _make_audio_vae(spec, instance, platform, args):
    module = load_component("AudioDecoderLoader", spec.checkpoint, args)   # LTX2AudioDecoder
    voc = instance.component("vocoder")    # chained: decoder -> vocoder
    return LTX2AudioVAE(module, voc, device=_device(platform), dtype=_native_dtype(module))


def _make_vocoder(spec, instance, platform, args):
    module = load_component("VocoderLoader", spec.checkpoint, args)        # LTX2Vocoder
    return LTX2Vocoder(module, device=_device(platform), dtype=_native_dtype(module))


_MAKERS = {
    "dit": _make_dit, "vae": _make_vae, "text_encoder": _make_text_encoder,
    "upsampler": _make_upsampler, "audio_vae": _make_audio_vae, "vocoder": _make_vocoder,
}


def build_component(spec, instance, platform):
    """The single cuda component builder (registered for every kind in ``torch_cuda.py``). Shared prefix
    — checkpoint check, dist-init, FastVideoArgs — then dispatch by ``spec.kind`` to its maker."""
    _require_checkpoint(spec)             # fail fast on a mis-stamped card, before any dist init
    _ensure_fastvideo_runtime()
    args = _fastvideo_args(spec)
    try:
        maker = _MAKERS[spec.kind]
    except KeyError:
        raise RuntimeError(f"torch backend: no builder for component kind {spec.kind!r} "
                           f"(have {sorted(_MAKERS)})")
    return maker(spec, instance, platform, args)
