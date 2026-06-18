# SPDX-License-Identifier: Apache-2.0
"""GEN3C torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so the GEN3C
recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

GEN3C is Cosmos-Predict2's EDM denoiser **extended with camera/3D-cache conditioning**. The deltas vs
the ``CosmosDiT`` adapter (see ``torch_cosmos.py``) are concentrated here:

* ``Gen3CDiT`` — the EDM denoiser ``F_θ``. The loop hands the *already EDM-input-scaled* model input
  (``x·c_in``) plus the **raw** conditioning bundle as kwargs; the DiT internally concats
  ``[latent(16) | input_mask(1) | pose_buffer(frame_buffer_max·32) | padding_mask(1)] = 82`` along the
  channel dim before ``patch_embed``. The conditioning is NOT pre-concatenated into the latent (the Wan
  i2v 16→36 concat would be wrong here): the loop passes ``condition_video_pose`` /
  ``condition_video_input_mask`` / ``condition_video_augment_sigma`` / ``padding_mask`` separately and
  the model assembles the 82-ch input. The model timestep is the EDM **noise preconditioning**
  ``c_noise = 0.25·log(σ)`` (NOT ``σ·1000`` like Wan, NOT raw σ) — this is exactly the value the
  diffusers ``EDMEulerScheduler.precondition_noise`` puts in ``scheduler.timesteps``, which the
  fastvideo ``Gen3CDenoisingStage`` feeds to the transformer. Returns the **raw** transformer output
  (the EDM ``c_skip``/``c_out`` → x0 reconstruction + x0-space CFG + frame-replace live in
  ``Gen3CDenoiseLoop``, NOT here). Faithful to ``Gen3CDenoisingStage.forward``
  (fastvideo/pipelines/stages/gen3c_stages.py).
* ``Gen3CVAE`` — wraps ``AutoencoderKLGen3CTokenizer`` (a JIT-backed tokenizer, NOT a diffusers
  ``AutoencoderKL``). Normalization is **internal** (the tokenizer applies its own mean/std buffers
  inside ``encode``/``decode``), so the adapter does NOT subtract/divide an external Wan-style
  ``(z-mean)/std`` (BRINGUP: contrast with ``WanVAE``). ``encode`` returns the tokenizer's ``.mode()``.
* ``Gen3CT5Encoder`` — T5-Large (1024-dim). GEN3C uses the **raw** last_hidden_state (NaN→0, no
  fixed-length zero-pad), identical to Cosmos's ``CosmosT5Encoder`` (reused by subclassing it).

BRINGUP (all GPU-only — written-not-run, verify on the box):
  * The ``AutoencoderKLGen3CTokenizer`` JIT loader expects ``encoder.jit`` / ``decoder.jit`` /
    ``mean_std.pt`` under the tokenizer dir (``from_jit_tokenizer``); the v2 loader seam must route the
    ``vae`` component to it rather than ``VAELoader``'s diffusers path — confirm the converted HF repo
    layout on the GPU box.
  * The camera / MoGe-depth / 3D-cache-render conditioning that fills ``condition_video_pose`` (and
    ``conditioning_latents`` for the first-frame anchor) is a separate pre-loop stage (see
    ``v2/recipes/gen3c/program.py``); without it the loop runs the degenerate **t2v** path (all-zeros
    conditioning → the pose-buffer concat is effectively zeroed inside the DiT).
"""
from __future__ import annotations

import math

import torch

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy
from v2.platform.backends.torch_cosmos import CosmosT5Encoder


def _c_noise(sigma: float) -> float:
    """EDM noise preconditioning ``c_noise = 0.25·log(σ)`` — exactly ``EDMEulerScheduler.timesteps[i]``
    for ``σ = sigmas[i]``. The GEN3C DiT embeds this directly via its sinusoidal ``Timesteps`` module
    (NO ``·1000``)."""
    return 0.25 * math.log(max(float(sigma), 1e-12))


class Gen3CDiT(TorchComponent):
    """``dit(model_input[16,T,h,w], text_embed[seq,1024], sigma, *, cond=<bundle>) -> raw output[16,T,h,w]``.

    ``model_input`` is pre-scaled by the loop (``x·c_in``); ``sigma`` is the *raw* EDM sigma (the adapter
    converts it to the model timestep ``c_noise = 0.25·log(σ)`` internally). ``cond`` is the GEN3C
    conditioning bundle dict (or None for t2v): ``condition_video_pose`` [buffer_channels,T,h,w],
    ``condition_video_input_mask`` [1,T,h,w], ``condition_video_augment_sigma`` (scalar), and the pixel
    ``height``/``width`` for the padding mask. None → the DiT builds zeros internally (degenerate t2v)."""

    def __init__(self, module, *, device, dtype, fps: int = 24):
        super().__init__(module, device=device, dtype=dtype)
        self.fps = int(fps)
        # add_augment_sigma_embedding is FALSE in the shipped 7B arch config → no augment_sigma weights;
        # the model only adds an augment-sigma embedding when that module exists (BRINGUP blocker 8).
        self.has_augment_embed = bool(getattr(module, "add_augment_sigma_embedding", False))

    @torch.no_grad()
    def __call__(self, model_input, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(model_input)  # [1, 16, T, h, w]
        ehs = self._t(text_embed)  # [1, seq, 1024]
        b, _c, t, h, w = hs.shape
        ts = torch.full((b, ), _c_noise(sigma), device=self.device, dtype=self.dtype)  # c_noise = 0.25·log σ

        # Conditioning bundle: real tensors for the camera/3D-cache path, zeros (degenerate t2v) otherwise.
        # The DiT concats input_mask + pose + padding_mask INTERNALLY → 82-ch patch_embed input.
        input_mask = pose = augment_sigma = None
        pix_h, pix_w = h, w  # padding_mask is at PIXEL h,w then resized inside the DiT
        if cond is not None:
            input_mask = self._t(cond.get("condition_video_input_mask")) \
                if cond.get("condition_video_input_mask") is not None else None
            pose = self._t(cond.get("condition_video_pose")) \
                if cond.get("condition_video_pose") is not None else None
            aug = cond.get("condition_video_augment_sigma")
            augment_sigma = torch.full((b, ), float(aug), device=self.device, dtype=self.dtype) \
                if (aug is not None and self.has_augment_embed) else None
            pix_h = int(cond.get("height", h))
            pix_w = int(cond.get("width", w))
        padding_mask = torch.zeros(b, 1, pix_h, pix_w, device=self.device, dtype=self.dtype)  # resized→[h,w]

        with self._ctx():
            out = self.module(hidden_states=hs,
                              timestep=ts,
                              encoder_hidden_states=ehs,
                              fps=self.fps,
                              condition_video_input_mask=input_mask,
                              condition_video_pose=pose,
                              condition_video_augment_sigma=augment_sigma,
                              padding_mask=padding_mask)
        if isinstance(out, tuple):
            out = out[0]
        return self._n(out)  # RAW EDM output (the loop reconstructs x0)


class Gen3CVAE(TorchComponent):
    """``AutoencoderKLGen3CTokenizer`` adapter. The tokenizer is JIT-backed and handles its OWN latent
    normalization internally (mean_std buffers) — so, unlike ``WanVAE``, this adapter does NOT apply an
    external ``(z-mean)/std``. ``encode`` returns the tokenizer's distribution ``.mode()`` (the
    ``_retrieve_latents`` contract the fastvideo stage uses); ``decode`` is a passthrough."""

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z)  # already in the DiT's (internally-normalized) latent space

    @torch.no_grad()
    def decode(self, latent):
        video = self.module.decode(self._t(latent))  # tokenizer un-normalizes internally → video [B,3,T,H,W]
        if hasattr(video, "sample"):
            video = video.sample
        return self._n(video)


class Gen3CT5Encoder(CosmosT5Encoder):
    """GEN3C T5-Large (1024-dim): raw last_hidden_state (NaN→0), no Wan zero-pad — identical to Cosmos.
    Reuses ``CosmosT5Encoder.encode`` verbatim. (BRINGUP: GEN3C pads with ``max_length`` and zeros rows
    past the attention-mask length, which ``CosmosT5Encoder`` already does — verify the max_length /
    mask-zeroing matches the converted T5-Large tokenizer on the GPU box.)"""


# Re-export so a reader can ``from v2.platform.backends.torch_gen3c import _to_numpy`` if needed.
__all__ = ["Gen3CDiT", "Gen3CVAE", "Gen3CT5Encoder", "_to_numpy"]
