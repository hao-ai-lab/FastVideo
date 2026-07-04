"""Shared vocabulary enums.

These live in a leaf module so both ``card/`` (which references loop kinds and
consistency levels in its specs) and ``loop/``/``runtime/`` can import them
without a circular dependency. ``card/`` imports no runtime; this is pure vocabulary.
"""
from __future__ import annotations

from enum import Enum


class LoopKind(str, Enum):
    """The kind of iterative computation a LoopSpec describes."""
    DIFFUSION_DENOISE = "diffusion_denoise"  # N solver steps over full clip (Wan, LTX-2)
    CHUNK_ROLLOUT = "chunk_rollout"  # causal chunk loop × inner denoise (self-forcing, world models)
    AR_DECODE = "ar_decode"  # token loop until EOS (reasoner/thinker/talker — phase 2)
    VAE_TILE = "vae_tile"  # tiled VAE encode/decode
    ENCODER = "encoder"  # one-shot encoder (text/vision) — degenerate single-step loop
    AUDIO_DECODE = "audio_decode"  # vocoder / codec decode


class WorkUnitKind(str, Enum):
    """The smallest schedulable action — the scheduler's currency unit.

    Tokens are one kind among many, not the scheduler itself: this generalizes
    vLLM's token scheduler to the work units diffusion needs.
    """
    AR_PREFILL = "ar_prefill"
    AR_TOKEN = "ar_token"
    DIFFUSION_STEP = "diffusion_step"
    DIFFUSION_WINDOW = "diffusion_window"
    CHUNK_STEP = "chunk_step"
    ENCODER_CHUNK = "encoder_chunk"
    VAE_TILE = "vae_tile"
    AUDIO_CHUNK = "audio_chunk"
    TRANSFER = "transfer"
    CACHE_IO = "cache_io"
    GRAPH_CAPTURE = "graph_capture"


class ConsistencyLevel(str, Enum):
    """The inference parity ladder."""
    C0 = "C0"  # component parity (VAE/encoder/block/scheduler-step in isolation)
    C1 = "C1"  # loop parity (full denoise trajectory / AR logits, fixed seed)
    C2 = "C2"  # behavioral identity for inference-visible trajectories
    C3 = "C3"  # distribution parity (rollout distribution under allowed nondeterminism)
    C4 = "C4"  # artifact quality (SSIM / reward agreement / human preference)

    @property
    def rank(self) -> int:
        return {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4}[self.value]


class ExecutionProfile(str, Enum):
    """Inference execution profiles."""
    SERVE = "serve"  # no-grad, graphed, cached, possibly quantized
    ROLLOUT = "rollout"  # serve profile + trajectory capture / stochastic rollout


class Capability(str, Enum):
    """CapabilityMatrix entries (model plane)."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_VIDEO_SOUND = "text_to_video_sound"
    AUDIO_TO_VIDEO = "audio_to_video"
    TEXT_TO_SPEECH = "text_to_speech"  # thinker→talker→vocoder (Qwen-Omni): reason + speak
    REASONING_TEXT = "reasoning_text"
    ACTION_CONDITIONING = "action_conditioning"
    STREAMING_VIDEO_CONTINUATION = "streaming_video_continuation"
    VAE_ENCODE = "vae_encode"
    VAE_DECODE = "vae_decode"
