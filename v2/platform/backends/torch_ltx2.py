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
    def __call__(self, latent, text_embed, sigma, context=None, *, audio_latent=None, audio_text=None):
        """Video-only (audio_latent=None) -> velocity[C,T,H,W]; joint A/V (audio_latent given) -> a tuple
        (video_velocity, audio_velocity). LTX-2.3 cross-attends video<->audio in one forward, so the
        audio latent + audio text must be passed together (not a separate call)."""
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
        av_kwargs: dict = {}
        au = None
        if audio_latent is not None:                     # joint A/V: audio latent [1,8,T,16] + audio text
            from fastvideo.models.audio.ltx2_audio_vae import AudioLatentShape
            au = _to_torch(audio_latent, device=self.device, dtype=self.dtype).unsqueeze(0)
            aeh = _to_torch(audio_text, device=self.device, dtype=self.dtype)
            if aeh is not None:
                aeh = aeh.unsqueeze(0)
            atok = self.module.audio_patchifier.get_token_count(AudioLatentShape.from_torch_shape(tuple(au.shape)))
            av_kwargs = dict(
                audio_hidden_states=au, audio_encoder_hidden_states=aeh,
                audio_timestep=torch.full((1, atok, 1), s, device=self.device, dtype=torch.float32),
                audio_sigma=torch.tensor([s], device=self.device, dtype=torch.float32))
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=s, attn_metadata=None):
            out = self.module(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                              video_sigma=video_sigma, encoder_attention_mask=None, **av_kwargs)
        # LTX-2 DiT predicts x0; the v2 loop integrates velocity = (x_t - x0)/sigma.
        if au is not None:
            denoised_v, denoised_a = out
            vel_v = (hs.float() - denoised_v.float()) / max(s, 1e-6)
            vel_a = (au.float() - denoised_a.float()) / max(s, 1e-6)
            return _to_numpy(vel_v.squeeze(0)), _to_numpy(vel_a.squeeze(0))
        denoised = out[0] if isinstance(out, tuple) else out
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

    @torch.no_grad()
    def encode_av(self, text):
        """LTX-2.3's connector emits SEPARATE video + audio text projections; returns (video_text,
        audio_text) numpy arrays (video = last_hidden_state, audio = hidden_states[0]). For LTX-2.0 the
        two are the same shared projection, so both are returned equal."""
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding="max_length")
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(current_timestep=0, attn_metadata=None):
            # output_hidden_states=True populates hidden_states=(audio_encoding,) — the SEPARATE audio
            # connector projection (gemma.py:703); without it hidden_states is None and we'd wrongly
            # fall back to the video embedding (different dim -> cross-attn shape mismatch).
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        video = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hs = getattr(out, "hidden_states", None)
        audio = hs[0] if hs else video
        return _to_numpy(video.squeeze(0)), _to_numpy(audio.squeeze(0))


class TorchLTX2Vocoder:
    """Thin wrapper over the LTX-2 ``Vocoder`` module (mel-spectrogram -> waveform @24kHz). Built as its
    own component so ``TorchLTX2AudioVAE`` can chain AudioDecoder -> Vocoder."""

    def __init__(self, module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype


class TorchLTX2AudioVAE:
    """audio_vae.decode(audio_latent[C,T,F]) -> waveform (numpy). Runs the LTX-2 ``AudioDecoder``
    (latent -> mel spectrogram) then the ``Vocoder`` (mel -> waveform @24kHz)."""

    def __init__(self, module, vocoder, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()      # AudioDecoder
        self.vocoder = vocoder                                          # TorchLTX2Vocoder | None
        self.device, self.dtype = device, dtype
        # Final waveform rate = the Vocoder's output_sample_rate (24000 for LTX-2; AudioDecoder's own
        # sample_rate is the internal mel rate). Surfaced so a saver writes the wav at the right rate.
        self.sample_rate = int(getattr(getattr(vocoder, "module", None), "output_sample_rate", 24000))

    @torch.no_grad()
    def decode(self, audio_latent):
        z = _to_torch(audio_latent, device=self.device, dtype=self.dtype).unsqueeze(0)   # [1,8,T,16]
        mel = self.module(z)
        if hasattr(mel, "sample"):
            mel = mel.sample
        if self.vocoder is not None:
            wav = self.vocoder.module(mel)
            if hasattr(wav, "sample"):
                wav = wav.sample
        else:
            wav = mel
        return _to_numpy(wav.squeeze(0))


class TorchLTX2Upsampler:
    """upsample(latent[C,T,H,W]) -> latent[C,T,2H,2W] (numpy in/out). Wraps the real LTX2LatentUpsampler
    and applies the repo's ``upsample_video``: un_normalize (via the video VAE's per_channel_statistics)
    → learned 2× spatial upsample → normalize, so the output stays in the normalized latent space the
    refine stage integrates. This replaces the toy's nearest-neighbor np.repeat with the learned
    super-resolution between the base (half-res) and refine (full-res) stages."""

    def __init__(self, module, vae_module, *, device, dtype):
        self.module = module.to(device=device, dtype=dtype).eval()
        self.vae_module = vae_module          # CausalVideoAutoencoder: exposes per_channel_statistics
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def upsample(self, latent):
        from fastvideo.models.upsamplers.ltx2_upsampler import upsample_video
        z = _to_torch(latent, device=self.device, dtype=self.dtype).unsqueeze(0)   # [1,C,T,H,W]
        out = upsample_video(z, self.vae_module, self.module)
        return _to_numpy(out.squeeze(0))
