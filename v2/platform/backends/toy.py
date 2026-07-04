"""Toy numpy components — the CPU-testable backend.

There is no GPU/torch/weights in this environment, so the heavy Wan/LTX neural forwards are
represented by small, deterministic numpy components. They exercise the *real* loop control
flow, CFG/flow-shift policies, scheduler steps, cache reuse, and parity gates with real numbers -
just with a toy network instead of a 1.3B DiT.

The contract these toys honor is what matters:
  * deterministic given (weights, inputs) → bit-reproducible (the interleave/parity gates rely on this);
  * same prompt → same text embedding → feature-cache reuse is correct;
  * a velocity ("flow") prediction the flow-match sampler consumes (Wan/LTX prediction_type).

On a GPU box, ``ComponentSpec.factory`` swaps these for lazy torch adapters wrapping the real
``fastvideo.models`` modules + weights (see each model's ``components.py``); the loops, policies,
scheduler, caches, and parity are unchanged. That is the whole point of the (recipe, runtime)
separation.
"""
from __future__ import annotations

import hashlib

import numpy as np

LATENT_CHANNELS = 4
TEXT_SEQ = 4
TEXT_DIM = 8


def _seed_from(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class ToyTextEncoder:
    """Deterministic text→embedding (same text ⇒ same embedding ⇒ feature-cache reuse works)."""

    def __init__(self, seq: int = TEXT_SEQ, dim: int = TEXT_DIM):
        self.seq, self.dim = seq, dim

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(_seed_from("txt:" + (text or "<empty>")))
        return (rng.standard_normal((self.seq, self.dim)) * 0.1).astype("float32")

    def encode_av(self, text: str):
        """LTX-2.3's connector emits SEPARATE video + audio text projections; toy returns two distinct
        embeddings (the real Gemma.encode_av returns last_hidden_state + hidden_states[0])."""
        return self.encode(text), self.encode((text or "") + "\x00audio")


class ToyImageEncoder:
    """Deterministic image→embedding stand-in for the CLIP vision encoder (Wan i2v conditioning context)."""

    def __init__(self, seq: int = TEXT_SEQ, dim: int = TEXT_DIM):
        self.seq, self.dim = seq, dim

    def encode_image(self, image) -> np.ndarray:
        a = np.asarray(image, dtype="float32")
        rng = np.random.default_rng(_seed_from(f"img:{a.size}:{(float(a.mean()) if a.size else 0.0):.4f}"))
        return (rng.standard_normal((self.seq, self.dim)) * 0.1).astype("float32")


class ToyDiT:
    """A tiny deterministic velocity predictor (stands in for the 1.3B DiT).

    velocity = tanh( Wx·latent  +  Wt·σ  +  s·mean(text_embed)  +  c·mean(context) )
    Channel-mixing via ``Wx`` makes the trajectory non-trivial; ``context`` carries causal
    chunk history for the self-forcing chunk_rollout loop.
    """

    def __init__(self, channels: int = LATENT_CHANNELS, seed: int = 0):
        self.C = channels
        rng = np.random.default_rng(seed)
        self.w_x = (rng.standard_normal((channels, channels)) * 0.15).astype("float32")
        self.w_t = (rng.standard_normal(channels) * 0.1).astype("float32")
        self.s_text = 0.5
        self.s_ctx = 0.3

    def _pre_tanh(self, latent: np.ndarray, text_embed, sigma: float, context) -> np.ndarray:
        mixed = np.tensordot(self.w_x, latent, axes=([1], [0]))  # [C,...] channel mix
        mixed = mixed + self.w_t.reshape((self.C, ) + (1, ) * (latent.ndim - 1)) * float(sigma)
        cond = float(np.mean(text_embed)) if text_embed is not None else 0.0
        mixed = mixed + self.s_text * cond
        if context is not None:
            mixed = mixed + self.s_ctx * float(np.mean(context))
        return mixed

    def __call__(self,
                 latent: np.ndarray,
                 text_embed: np.ndarray | None,
                 sigma: float,
                 context: np.ndarray | None = None,
                 *,
                 audio_latent: np.ndarray | None = None,
                 audio_text: np.ndarray | None = None,
                 cond: np.ndarray | None = None):
        # ``cond`` (i2v mask+cond latent) is the real WanDiT's 36ch concat; the toy denoises the noise
        # channels only (image-conditioning is a GPU-path concern), so it's accepted and ignored here.
        latent = np.asarray(latent, dtype=np.float32)
        video = np.tanh(self._pre_tanh(latent, text_embed, sigma, context)).astype("float32")
        if audio_latent is None:
            return video
        # toy joint A/V (LTX-2.3): a simple element-wise audio velocity (the real DiT cross-attends
        # video<->audio in one forward and returns (video_vel, audio_vel)).
        au = np.asarray(audio_latent, dtype=np.float32)
        a_cond = float(np.mean(audio_text)) if audio_text is not None else 0.0
        audio = np.tanh(0.9 * au + 0.1 * float(sigma) + 0.3 * a_cond).astype("float32")
        return video, audio

    # --- weight/version surface for serving hot-reload tests ----------------------------- #
    def clone(self) -> ToyDiT:
        c = ToyDiT.__new__(ToyDiT)
        c.C, c.s_text, c.s_ctx = self.C, self.s_text, self.s_ctx
        c.w_x, c.w_t = self.w_x.copy(), self.w_t.copy()
        return c

    def blend_from(self, other: ToyDiT, decay: float) -> None:
        """Blend resident weights toward another component version."""
        self.w_x = (decay * self.w_x + (1.0 - decay) * other.w_x).astype("float32")
        self.w_t = (decay * self.w_t + (1.0 - decay) * other.w_t).astype("float32")

    def copy_from(self, other: ToyDiT) -> None:
        self.w_x, self.w_t = other.w_x.copy(), other.w_t.copy()


class ToyTokenizer:
    """Deterministic byte-ish tokenizer for the omni AR pathway (phase 2)."""
    EOS = 0
    VOCAB = 256

    def encode(self, text: str) -> list[int]:
        toks = [(ord(c) % (self.VOCAB - 1)) + 1 for c in (text or "")[:16]]
        return toks or [1]

    def decode(self, tokens) -> str:
        return "".join(chr(32 + (int(t) % 90)) for t in tokens)


class ToyMoTDiT(ToyDiT):
    """Mixture-of-Transformers stand-in: ONE resident module that runs BOTH an AR (understanding)
    pathway and a diffusion (generation) pathway on shared weights.

    ``ar_forward`` is the und pathway (next-token); ``__call__`` (inherited) is the gen pathway
    (velocity). Binding both the ar_decode and diffusion_denoise loops to one instance of this
    component is the MoT requirement no DAG-of-engines can express.
    """
    VOCAB = 256
    EOS = 0

    def ar_forward(self, tokens) -> int:
        # deterministic next-token from the context (greedy ⇒ trivially interleave-safe)
        ctx = sum(int(t) for t in tokens)
        nxt = (ctx * 7 + 13) % self.VOCAB
        return int(nxt)

    def reasoner_embed(self, tokens) -> np.ndarray:
        """Pack the und-pathway tokens into a conditioning embed the gen pathway consumes
        (the Cosmos3 'prompt upsampling before diffusion in the same request')."""
        rng = np.random.default_rng((sum(int(t) for t in tokens) + 1) % (1 << 31))
        return (rng.standard_normal((4, 8)) * 0.1).astype("float32")


class ToyTalker(ToyMoTDiT):
    """Toy Talker — Qwen-Omni stage 1 (AR over a speech-codec vocab, conditioned on the Thinker).

    A *separate expert* from the thinker (not weight-shared): its next-token depends on its OWN
    weights (a seed salt) AND the prefilled thinker payload (tokens + hidden state). So the cascade
    is real — change the thinker and the talker's tokens change; change the talker's weights and they
    change too. The vllm-omni Talker is likewise a distinct model conditioned on Thinker hidden states.
    """

    def __init__(self, channels: int = LATENT_CHANNELS, seed: int = 0):
        super().__init__(channels=channels, seed=seed)
        self.salt = (seed % 97) + 1

    def ar_forward(self, tokens) -> int:
        ctx = sum(int(t) for t in tokens)
        return int((ctx * 7 + 13 + self.salt) % self.VOCAB)  # weight-dependent (salt) + context


class ToyVocoder:
    """Toy streaming code2wav vocoder — Qwen-Omni stage 2 (BigVGAN/Code2Wav role).

    Speech codec tokens → audio waveform, synthesized in chunks. Deterministic given the tokens so
    the ``audio_decode`` loop is interleave-safe. Each token contributes a short, timbre-stamped
    waveform segment — enough to exercise chunked synthesis, streaming emits, and the AudioArtifact.
    """

    def __init__(self, samples_per_token: int = 16, vocab: int = 256, seed: int = 7):
        self.spt = int(samples_per_token)
        rng = np.random.default_rng(seed)
        self.bank = (rng.standard_normal(vocab) * 0.3).astype("float32")  # per-token timbre offset

    def synthesize(self, token_chunk) -> np.ndarray:
        """One chunk of codec tokens → a waveform segment in [-1, 1], length spt·len(chunk)."""
        segs = []
        for t in token_chunk:
            t = int(t) % self.bank.size
            phase = np.linspace(0.0, np.pi * (1 + (t % 8)), self.spt, dtype="float32")
            segs.append(np.tanh(self.bank[t] + np.sin(phase)).astype("float32"))
        return np.concatenate(segs) if segs else np.zeros(0, dtype="float32")


class ToyAudioVAE:
    """Toy audio VAE — mel-like audio latent → mono waveform (the LTX-2 / Cosmos3 audio branch).

    Decodes a small ``[C, A, 1, 1]`` audio latent (denoised jointly with the video latent) into a 1-D
    waveform: channel-collapse via a fixed projection, then upsample along time. Deterministic, so the
    joint A/V denoise loop stays interleave-safe. The video VAE and this share nothing — two decoders,
    one synchronized latent pair."""

    def __init__(self, samples_per_frame: int = 16, seed: int = 2):
        self.spf = int(samples_per_frame)
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal(LATENT_CHANNELS) * 0.3).astype("float32")

    def decode(self, audio_latent: np.ndarray) -> np.ndarray:
        a = np.asarray(audio_latent, dtype="float32")
        flat = a.reshape(a.shape[0], -1)  # [C, A·...]
        # project over the actual channel count (LTX-2.3 audio latent is 8ch; the 2-stage toy is
        # LATENT_CHANNELS) — np.resize is identity when they already match, so existing T2VS is unchanged.
        proj = np.resize(self.proj, a.shape[0]).astype("float32")
        mono = np.tensordot(proj, flat, axes=([0], [0]))  # [A·...]
        return np.tanh(np.repeat(mono, self.spf)).astype("float32")  # [A·spf] mono waveform


class ToyVAE:
    """Tiny deterministic VAE. encode: video→latent (mean-pool + channel proj); decode: latent→video."""

    def __init__(self, channels: int = LATENT_CHANNELS, spatial: int = 8, temporal: int = 4, seed: int = 1):
        self.C = channels
        self.spatial = spatial
        self.temporal = temporal
        rng = np.random.default_rng(seed)
        self.dec_proj = (rng.standard_normal((3, channels)) * 0.2).astype("float32")

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """latent [C,T,H,W] -> video [3, T, H*spatial, W*spatial] (deterministic upsample)."""
        latent = np.asarray(latent, dtype=np.float32)
        rgb = np.tensordot(self.dec_proj, latent, axes=([1], [0]))  # [3,T,H,W]
        up = np.repeat(np.repeat(rgb, self.spatial, axis=2), self.spatial, axis=3)
        return np.tanh(up).astype("float32")

    def encode(self, video: np.ndarray) -> np.ndarray:
        video = np.asarray(video, dtype=np.float32)
        pooled = video[:self.C]  # crude: take C channels
        return pooled.astype("float32")


class ToyUpsampler:
    """Toy latent upsampler — CPU stand-in for the learned LTX-2 spatial upsampler. Nearest-neighbor 2x
    spatial repeat on a ``[C,T,H,W]`` latent. The real ``LTX2LatentUpsampler`` learns this super-res; the
    GPU backend swaps in that module (torch_backend.LTX2Upsampler) via the ``upsampler`` component kind,
    so the program calls ``component('spatial_upsampler').upsample(...)`` on both backends (no device branch)."""

    def __init__(self, scale: int = 2):
        self.scale = int(scale)

    def upsample(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype="float32")
        return np.repeat(np.repeat(z, self.scale, axis=-2), self.scale, axis=-1).astype("float32")
