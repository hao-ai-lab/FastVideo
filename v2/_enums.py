"""Shared vocabulary enums (design_v3 §6.1, §9.2, §9.4).

These live in a leaf module so both ``card/`` (which references loop kinds and
consistency levels in its specs) and ``loop/``/``runtime/`` can import them
without a circular dependency. design_v3 §18 boundary: ``card/`` imports no
runtime; this is pure vocabulary.
"""
from __future__ import annotations

from enum import Enum


class LoopKind(str, Enum):
    """The kind of iterative computation a LoopSpec describes (design_v3 §4.3)."""
    DIFFUSION_DENOISE = "diffusion_denoise"   # N solver steps over full clip (Wan, LTX-2)
    CHUNK_ROLLOUT = "chunk_rollout"           # causal chunk loop × inner denoise (self-forcing, world models)
    AR_DECODE = "ar_decode"                   # token loop until EOS (reasoner/thinker/talker — phase 2)
    VAE_TILE = "vae_tile"                     # tiled VAE encode/decode
    ENCODER = "encoder"                       # one-shot encoder (text/vision) — degenerate single-step loop
    AUDIO_DECODE = "audio_decode"             # vocoder / codec decode
    TRAIN_FORWARD = "train_forward"           # grad-enabled forward for a training method


class WorkUnitKind(str, Enum):
    """The smallest schedulable action — the scheduler's currency unit (design_v3 §6.1).

    Tokens are *one kind*, not the scheduler — the generalization of vLLM's token
    scheduler that diffusion forces.
    """
    AR_PREFILL = "ar_prefill"
    AR_TOKEN = "ar_token"
    DIFFUSION_STEP = "diffusion_step"
    DIFFUSION_WINDOW = "diffusion_window"
    CHUNK_STEP = "chunk_step"
    ENCODER_CHUNK = "encoder_chunk"
    VAE_TILE = "vae_tile"
    AUDIO_CHUNK = "audio_chunk"
    REWARD_BATCH = "reward_batch"     # RL
    LOGPROB_BATCH = "logprob_batch"   # RL (likelihood-based)
    TRANSFER = "transfer"
    CACHE_IO = "cache_io"
    GRAPH_CAPTURE = "graph_capture"


class ConsistencyLevel(str, Enum):
    """The consistency ladder (design_v3 §9.2). RL methods declare their required rung."""
    C0 = "C0"   # component parity (VAE/encoder/block/scheduler-step in isolation)
    C1 = "C1"   # loop parity (full denoise trajectory / AR logits, fixed seed)
    C2 = "C2"   # behavioral identity (train-forward ≡ serve-forward on the RL objective's quantity)
    C3 = "C3"   # distribution parity (rollout distribution under allowed nondeterminism)
    C4 = "C4"   # artifact quality (SSIM / reward agreement / human preference)

    @property
    def rank(self) -> int:
        return {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4}[self.value]


class ExecutionProfile(str, Enum):
    """Three forwards, one loop definition — differ only in grad mode + capture (design_v3 §9.4)."""
    SERVE = "serve"       # no-grad, graphed, cached, possibly quantized
    ROLLOUT = "rollout"   # serve profile + behavior capture
    TRAIN = "train"       # grad, checkpointed, FSDP-gathered


class Capability(str, Enum):
    """CapabilityMatrix entries (designv2 model plane)."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_VIDEO_SOUND = "text_to_video_sound"
    AUDIO_TO_VIDEO = "audio_to_video"
    TEXT_TO_SPEECH = "text_to_speech"      # thinker→talker→vocoder (Qwen-Omni): reason + speak
    REASONING_TEXT = "reasoning_text"
    ACTION_CONDITIONING = "action_conditioning"
    STREAMING_VIDEO_CONTINUATION = "streaming_video_continuation"
    VAE_ENCODE = "vae_encode"
    VAE_DECODE = "vae_decode"
    POLICY_ROLLOUT = "policy_rollout"      # can serve as an RL rollout engine
    LOGPROB_RECOMPUTE = "logprob_recompute"
