# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline smoke test (CPU / float32, tiny components).

Exercises the native pipeline's runtime path end-to-end on tiny stub-or-real
components, with NO real weights and NO cosmos_framework dependency:

  * a stub transformer implementing the native DiT's packed-input ->
    ``{"preds_vision": [...]}`` contract with bounded, deterministic output
    (the REAL DiT forward / unpatchify is exhaustively bit-identical-tested in
    ``test_cosmos3_dit_parity*`` and ``test_cosmos3_denoise_cfg_parity``; an
    untrained real DiT emits unbounded velocities that overflow UniPC, so the
    smoke uses a stub to keep finiteness deterministic);
  * a stub VAE exposing the ``AutoencoderKLWan`` surface
    (``config.latents_mean/std/scale_factor_spatial``, ``encode().mode()``,
    ``decode()``) used by the encode/normalize + decode/denormalize bridges;
  * a stub Qwen2-shaped tokenizer (chat template + special tokens).

Two paths are covered:

  1. ``Cosmos3DenoiseEngine.denoise`` for >= 2 UniPC steps over a tiny T2V
     latent, asserting a finite final latent of the right shape, plus the VAE
     decode + ``(1 + x)/2`` clamp producing a finite ``[B, 3, T, H, W]`` video;
  2. the real ``Cosmos3DenoisingStage.forward`` (full tokenize -> noise ->
     denoise -> decode wiring) driven through a ``__new__``-built pipeline +
     ``ForwardBatch``, for both T2V and I2V.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_pipeline_smoke.py -q
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)

pytestmark = [pytest.mark.local]

_LATENT_CHANNEL = 16
_LATENT_PATCH_SIZE = 2
_SPATIAL_FACTOR = 8
_TEMPORAL_FACTOR = 4


# ---------------------------------------------------------------------------
# Stub transformer implementing the native DiT packed-input contract.
# ---------------------------------------------------------------------------
class StubCosmos3Transformer(torch.nn.Module):
    """Bounded stand-in for ``Cosmos3VFMTransformer.forward``.

    Consumes the same packed kwargs (``vision_token_shapes`` /
    ``vision_noisy_frame_indexes`` / ``vision_tokens`` / ``text_ids``) and
    returns ``{"preds_vision": [[1, C, T, H, W], ...]}`` with predictions only on
    noisy frames (zeros on conditioning frames), matching the real DiT's
    ``_unpatchify_and_unpack`` output structure. The prediction is a small
    ``tanh`` of the input latent, scaled by the first text id so the cond and
    uncond passes differ (exercising the CFG combination).
    """

    def __init__(self, latent_channel: int = _LATENT_CHANNEL) -> None:
        super().__init__()
        self.latent_channel = latent_channel
        # A real attribute the stage reads for device/dtype.
        self.embed_tokens = torch.nn.Embedding(64, 8)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        token_ids = kwargs["text_ids"]
        token = float(token_ids.reshape(-1)[0].item()) if token_ids.numel() else 1.0
        scale = 0.01 * (1.0 + (token % 7))
        token_shapes = kwargs["vision_token_shapes"]
        noisy = kwargs["vision_noisy_frame_indexes"]
        tokens = kwargs["vision_tokens"]
        preds: list[torch.Tensor] = []
        for latent, (t, _h, _w), nfi in zip(tokens, token_shapes, noisy):
            lat = latent.squeeze(0) if latent.dim() == 5 else latent  # [C, T, H, W]
            out = torch.zeros_like(lat)
            if nfi.numel() > 0:
                out[:, nfi] = scale * torch.tanh(lat[:, nfi])
            preds.append(out.unsqueeze(0))  # [1, C, T, H, W]
        return {"preds_vision": preds}


# ---------------------------------------------------------------------------
# Stub VAE matching the AutoencoderKLWan surface used by the bridges.
# ---------------------------------------------------------------------------
class _StubLatentDist:

    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    """Minimal VAE: deterministic encode/decode shaped by scale factors."""

    def __init__(self, z_dim: int = _LATENT_CHANNEL) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=_TEMPORAL_FACTOR,
            scale_factor_spatial=_SPATIAL_FACTOR,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        b, _c, t, h, w = video.shape
        lt = (t - 1) // self.config.scale_factor_temporal + 1
        lh = h // self.config.scale_factor_spatial
        lw = w // self.config.scale_factor_spatial
        latents = torch.ones(b, self.config.z_dim, lt, lh, lw, dtype=video.dtype, device=video.device)
        return _StubLatentDist(latents)

    def decode(self, z: torch.Tensor):
        # Upsample latents back to pixel dims; clamp like AutoencoderKLWan.
        b, _c, lt, lh, lw = z.shape
        t = (lt - 1) * self.config.scale_factor_temporal + 1
        h = lh * self.config.scale_factor_spatial
        w = lw * self.config.scale_factor_spatial
        # Bounded function of z so the output reflects (and stays finite with)
        # the latent: tanh maps any finite z to [-1, 1]; nan_to_num guards
        # against non-finite latents from an untrained denoise.
        z_signal = torch.nan_to_num(torch.tanh(z[:, :1, :1, :1, :1]))
        out = torch.zeros(b, 3, t, h, w, dtype=z.dtype, device=z.device) + z_signal.reshape(b, 1, 1, 1, 1)
        return torch.clamp(out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Stub Qwen2-shaped tokenizer (chat template + special tokens).
# ---------------------------------------------------------------------------
class StubQwen2Tokenizer:
    eos_token_id = 62

    _SPECIAL = {"<|vision_start|>": 60, "<|vision_end|>": 61}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL[token]

    def apply_chat_template(self, conversations, *, tokenize=True, add_generation_prompt=True, add_vision_id=False):
        # Deterministic small token ids from the user message length.
        user = next((c["content"] for c in conversations if c["role"] == "user"), "")
        n = max(1, min(8, len(user) % 8 + 1))
        return [10 + (i % 40) for i in range(n)]


def _make_scheduler() -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=10.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCosmos3PipelineSmoke:

    def test_denoise_engine_and_decode_finite(self):
        """Engine.denoise (>= 2 steps) + VAE decode -> finite output, right shape."""
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
            Cosmos3DenoiseEngine,
            Cosmos3VisionSpec,
            _VaeNorm,
            cosmos3_vae_decode,
        )

        dit = StubCosmos3Transformer()
        vae = StubCosmos3VAE()
        scheduler = _make_scheduler()
        scheduler.set_timesteps(2, device=torch.device("cpu"))

        latent_shape = (_LATENT_CHANNEL, 2, 2, 2)  # [C, T, H, W]
        torch.manual_seed(1)
        flat = torch.randn(int(torch.tensor(latent_shape).prod()))

        engine = Cosmos3DenoiseEngine(
            transformer=dit,
            scheduler=scheduler,
            special_tokens={"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62},
            latent_patch_size=_LATENT_PATCH_SIZE,
            temporal_modality_margin=15_000,
            reset_spatial_ids=True,
            enable_fps_modulation=False,
            base_fps=24.0,
            temporal_compression_factor=_TEMPORAL_FACTOR,
        )
        spec = Cosmos3VisionSpec(shape=latent_shape, condition_frame_indexes=[])
        out_flat = engine.denoise(
            flat_latent=flat,
            timesteps=scheduler.timesteps,
            guidance=6.0,
            specs=[spec],
            cond_token_ids=[10, 11, 12],
            uncond_token_ids=[13, 14],
        )
        assert out_flat.shape == flat.shape
        assert torch.isfinite(out_flat).all()

        norm = _VaeNorm.from_vae(vae, torch.float32)
        result_latent = out_flat.reshape(latent_shape).unsqueeze(0)
        decoded = cosmos3_vae_decode(vae, result_latent, norm)
        video = ((1.0 + decoded) / 2.0).clamp(0.0, 1.0)
        assert video.dim() == 5 and video.shape[1] == 3
        assert torch.isfinite(video).all()
        assert float(video.min()) >= 0.0 and float(video.max()) <= 1.0

    def _make_pipeline(self):
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import Cosmos3OmniDiffusersPipeline

        pipe = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
        scheduler = _make_scheduler()
        pipe.modules = {
            "transformer": StubCosmos3Transformer(),
            "vae": StubCosmos3VAE(),
            "scheduler": scheduler,
            "text_tokenizer": StubQwen2Tokenizer(),
        }
        pipe.scheduler = scheduler
        pipe._base_scheduler_config = scheduler.config
        pipe._current_flow_shift = float(scheduler.config.flow_shift)
        pipe._engine_init_flow_shift = 10.0
        return pipe

    def _make_args(self):
        from fastvideo.configs.pipelines.cosmos3 import Cosmos3Config

        cfg = Cosmos3Config()
        # Shrink the DiT arch config to the tiny smoke geometry.
        arch = cfg.dit_config.arch_config
        arch.latent_channel = _LATENT_CHANNEL
        arch.latent_patch_size = _LATENT_PATCH_SIZE
        arch.temporal_compression_factor = _TEMPORAL_FACTOR
        arch.enable_fps_modulation = False
        return SimpleNamespace(pipeline_config=cfg)

    def _make_batch(self, *, num_frames, height, width, image=None):
        from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

        return ForwardBatch(
            data_type="video",
            prompt="a calm ocean at sunrise",
            negative_prompt="",
            height=height,
            width=width,
            num_frames=num_frames,
            fps=24,
            num_inference_steps=2,
            guidance_scale=6.0,
            generator=torch.Generator("cpu").manual_seed(0),
            preprocessed_image=image,
        )

    def test_stage_forward_t2v_finite(self):
        """The real Cosmos3DenoisingStage.forward runs T2V end-to-end."""
        from fastvideo.pipelines.stages.cosmos3_stages import Cosmos3DenoisingStage

        pipe = self._make_pipeline()
        args = self._make_args()
        stage = Cosmos3DenoisingStage(
            transformer=pipe.modules["transformer"],
            scheduler=pipe.modules["scheduler"],
            vae=pipe.modules["vae"],
            tokenizer=pipe.modules["text_tokenizer"],
            pipeline=pipe,
        )
        # 5 frames -> latent_t = (5-1)//4 + 1 = 2; 16x16 px -> 2x2 latent.
        batch = self._make_batch(num_frames=5, height=16, width=16)
        out = stage.forward(batch, args)
        assert out.output is not None
        assert out.output.dim() == 5 and out.output.shape[1] == 3
        assert torch.isfinite(out.output).all()
        assert torch.isfinite(out.latents).all()

    def test_stage_forward_i2v_keeps_condition_frame(self):
        """I2V: a conditioning image is VAE-encoded and frame 0 stays clean."""
        from fastvideo.pipelines.stages.cosmos3_stages import Cosmos3DenoisingStage

        pipe = self._make_pipeline()
        args = self._make_args()
        stage = Cosmos3DenoisingStage(
            transformer=pipe.modules["transformer"],
            scheduler=pipe.modules["scheduler"],
            vae=pipe.modules["vae"],
            tokenizer=pipe.modules["text_tokenizer"],
            pipeline=pipe,
        )
        # Conditioning image as a [3, H, W] tensor in [-1, 1].
        image = torch.zeros(3, 16, 16)
        batch = self._make_batch(num_frames=5, height=16, width=16, image=image)
        out = stage.forward(batch, args)
        assert out.output is not None
        assert out.output.dim() == 5 and out.output.shape[1] == 3
        assert torch.isfinite(out.output).all()

    def test_tokenize_caption_special_tokens(self):
        """Pipeline.tokenize_caption uses the Qwen2 chat template + ids."""
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import cosmos3_special_tokens

        pipe = self._make_pipeline()
        ids = pipe.tokenize_caption("hello world", is_video=True, use_system_prompt=False)
        assert isinstance(ids, list) and len(ids) > 0
        special = cosmos3_special_tokens(pipe.get_module("text_tokenizer"))
        assert special == {"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62}
