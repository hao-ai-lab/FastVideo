"""Stable Diffusion 3.5 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the SD3.5 recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``
dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

Architecture deltas vs Wan/Cosmos (all faithful to ``fastvideo/pipelines/stages/sd35_conditioning.py``
+ ``fastvideo/models/dits/sd3.py:SD3Transformer2DModel``):

* ``SD3DiT`` — a FLOW-MATCH MMDiT. The forward needs TWO text conditioners: ``encoder_hidden_states``
  (the assembled triple-encoder joint embed [seq, 4096]) AND a separate ``pooled_projections`` vector
  [2048]. The v2 loop threads only a single ``text_embed`` + a positional ``context`` arg (the same
  surface ToyDiT exposes), so this adapter takes ``text_embed`` = the joint embed and ``context`` = the
  pooled vector. Latents are 4D image latents [1, 16, h, w] (num_frames=1) — NOT 5D video. The model
  output is the flow-match velocity directly (no x0 conversion, unlike LTX2DiT). ``timestep = sigma*1000``
  (diffusers FlowMatchEuler convention, same as WanDiT). The fastvideo forward self-manages
  ``set_forward_context`` (sd3.py lines 1011-1021) but wrapping in ``self._ctx()`` is idempotent (it
  detects the existing context), so we wrap for uniformity with the other adapters.

* ``SD3VAE`` — ``AutoencoderKL`` operates in RAW latent space; the DiT operates in a shift/scale
  NORMALIZED space. SD3.5: ``z_dit = (z - shift_factor) * scaling_factor`` on encode; decode inverts
  ``z_raw = z_dit / scaling_factor + shift_factor`` then ``vae.decode -> image in [-1,1]``. CRITICAL:
  the factors come from the LOADED checkpoint ``vae.config`` (scaling_factor=1.5305, shift_factor=0.0609
  for SD3.5), NOT the generic AutoencoderKLArchConfig defaults (0.18215 / None). 4D (image), not 5D.
  The decoded image is mapped to [0,1] by the program (``(img/2+0.5).clamp(0,1)``), matching
  ``SD35DecodingStage``.

* ``SD3ClipEncoder`` / ``SD3T5Encoder`` — the joint embed is assembled in the program (see
  ``sd35.program``), so each encoder adapter returns BOTH the per-encoder pieces the assembler needs:
  CLIP returns ``(penultimate_hidden[-2], pooler_output)`` and T5 returns ``last_hidden_state``.
  CLIP runs with ``output_hidden_states=True`` at max_length=77 padded; T5 at max_length=256 padded.
  Faithful to ``SD35ConditioningStage._clip_pooled`` + the TextEncodingStage CLIP penultimate extraction.
"""
from __future__ import annotations

import numpy as np
import torch

from v2.platform.backends.torch_backend import NUM_TRAIN_TIMESTEPS, TorchComponent, _to_numpy

# SD3.5 encoder lengths (SD35Pipeline.initialize_pipeline / SD35ConditioningStage).
CLIP_MAX_LENGTH = 77
T5_MAX_LENGTH = 256
# Generic SD1.5-era AutoencoderKL defaults — used ONLY as a last-resort fallback if the checkpoint
# config is missing the SD3.5 values (BRINGUP: the real checkpoint vae config carries 1.5305 / 0.0609).
_DEFAULT_SCALING = 1.5305
_DEFAULT_SHIFT = 0.0609


class SD3DiT(TorchComponent):
    """``dit(latent[16,h,w], encoder_hidden_states[seq,4096], sigma, context=pooled[2048]) -> velocity``.

    Faithful to ``SD35DenoisingStage`` + ``SD3Transformer2DModel.forward``: 4D image latents, the dual
    text conditioning (joint embed + pooled), ``timestep = sigma*1000`` broadcast to [B], flow-match
    velocity output. ``context`` carries the pooled_projections vector the loop threads alongside the
    joint embed (so the loop's dit-call stays the cosmos2-style ``dit(latent, text_embed, sigma, context)``)."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        # 4D image latent: ``self._t`` prepends the batch dim -> [1, 16, h, w]. (No temporal dim.)
        hs = self._t(latent)
        ehs = self._t(text_embed)  # [1, seq, 4096] joint embed
        if context is None:
            raise RuntimeError("SD3DiT requires pooled_projections (threaded as `context`); got None.")
        pp = self._t(context)  # [1, 2048] dual-CLIP pooled vector
        # diffusers FlowMatchEuler timestep = sigma * num_train_timesteps (1000); 1D [B] long-ish float.
        ts = torch.tensor([float(sigma) * NUM_TRAIN_TIMESTEPS], device=self.device, dtype=self.dtype)
        with self._ctx():  # idempotent (sd3.forward self-manages it)
            out = self.module(hidden_states=hs,
                              encoder_hidden_states=ehs,
                              pooled_projections=pp,
                              timestep=ts,
                              return_dict=False)[0]  # [1, 16, h, w] flow-match velocity
        return self._n(out)  # -> [16, h, w] numpy (no x0 conversion)


class SD3VAE(TorchComponent):
    """``AutoencoderKL`` with the SD3.5 shift/scale normalization the DiT latent space requires.

    encode: ``z_dit = (z - shift_factor) * scaling_factor``; decode: ``z_raw = z_dit / scaling_factor +
    shift_factor`` then ``vae.decode -> image[3,h,w] in [-1,1]``. The factors are read from the loaded
    ``module.config`` (BRINGUP risk D: the generic dataclass defaults would mis-scale). Marshals 4D
    (image), not 5D video. The [-1,1] -> [0,1] remap is done by the program, matching ``SD35DecodingStage``."""

    @staticmethod
    def _as_float(v: object) -> float:
        return float(v.item()) if isinstance(v, torch.Tensor) else float(v)  # type: ignore[arg-type]

    def _factors(self) -> tuple[float, float | None]:
        cfg = getattr(self.module, "config", None)
        sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
        sh = getattr(cfg, "shift_factor", None) if cfg is not None else None
        if sf is None:
            sf = getattr(self.module, "scaling_factor", _DEFAULT_SCALING)
        if sh is None:
            sh = getattr(self.module, "shift_factor", _DEFAULT_SHIFT)
        return self._as_float(sf), (None if sh is None else self._as_float(sh))

    @torch.no_grad()
    def encode(self, image):
        # image [3, h, w] -> [1, 3, h, w] (4D); AutoencoderKL.encode -> a DiagonalGaussian dist.
        x = self._t(image)
        dist = self.module.encode(x)
        dist = dist.latent_dist if hasattr(dist, "latent_dist") else dist
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        sf, sh = self._factors()
        z = z.float()
        if sh is not None:
            z = z - sh
        return self._n(z * sf)  # -> the normalized latent the DiT expects

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float()  # [1, 16, h, w]
        sf, sh = self._factors()
        z = z / sf
        if sh is not None:
            z = z + sh
        dec = self.module.decode(z.to(self.dtype))  # -> image [1, 3, H, W] in [-1, 1]
        image = dec.sample if hasattr(dec, "sample") else (dec[0] if isinstance(dec, tuple) else dec)
        return self._n(image)  # [3, H, W]; program remaps [-1,1]->[0,1]


class SD3ClipEncoder(TorchComponent):
    """``CLIPTextModelWithProjection`` (clip_l or clip_g). ``encode(text) -> {hidden, pooled}`` where
    ``hidden`` is the penultimate hidden state ([-2], the SD3 joint-embed source — NOT last_hidden_state)
    and ``pooled`` is ``pooler_output`` (the text-projection, the pooled-projection source). max_length=77
    padded. Faithful to ``SD35ConditioningStage._clip_pooled`` + the TextEncodingStage penultimate extraction."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = CLIP_MAX_LENGTH):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "",
                              return_tensors="pt",
                              max_length=self.max_length,
                              padding="max_length",
                              truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # ``hidden_states`` is the per-layer pool; [-2] is the penultimate layer (the SD3 joint-embed source).
        hs = out.hidden_states
        penultimate = hs[-2] if isinstance(hs, list | tuple) else hs
        penultimate = penultimate.squeeze(0)  # [seq=77, dim]
        pooled = out.pooler_output.squeeze(0)  # [dim] (text_projection of the EOS token)
        return {"hidden": _to_numpy(penultimate), "pooled": _to_numpy(pooled)}


class SD3T5Encoder(TorchComponent):
    """``T5EncoderModel`` (the third SD3.5 encoder). ``encode(text) -> last_hidden_state[seq=256, 4096]``
    (max_length=256 padded). NaN->0 for safety. Faithful to the T5 branch of ``SD35ConditioningStage``."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = T5_MAX_LENGTH):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "",
                              return_tensors="pt",
                              max_length=self.max_length,
                              padding="max_length",
                              truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        return _to_numpy(torch.nan_to_num(hidden, nan=0.0))  # [256, 4096]


def assemble_sd3_conditioning(clip_l: dict, clip_g: dict, t5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the SD3.5 dual-text conditioning from the three encoder outputs — the EXACT
    ``SD35ConditioningStage`` algebra, kept as a pure-numpy helper so both the GPU and CPU-toy program
    paths share one assembler (BRINGUP risk: the pad/concat order + hidden index are easy to get wrong):

      * joint embed:  ``cat([ F.pad(cat([clip_l_h, clip_g_h], dim=-1), (0, 4096-2048)) , t5_h ], dim=seq)``
      * pooled:       ``cat([clip_l_pool, clip_g_pool], dim=-1)``  -> [2048]
    """
    clip_h = np.concatenate([np.asarray(clip_l["hidden"]), np.asarray(clip_g["hidden"])], axis=-1)  # [77, 2048]
    t5_h = np.asarray(t5)  # [256, 4096]
    pad = t5_h.shape[-1] - clip_h.shape[-1]  # 4096 - 2048 = 2048
    if pad < 0:
        raise ValueError(f"CLIP joint dim {clip_h.shape[-1]} exceeds T5 dim {t5_h.shape[-1]}")
    clip_h = np.pad(clip_h, [(0, 0)] * (clip_h.ndim - 1) + [(0, pad)])  # [77, 4096]
    joint = np.concatenate([clip_h, t5_h], axis=-2).astype("float32")  # [77+256, 4096] along seq
    pooled = np.concatenate([np.asarray(clip_l["pooled"]), np.asarray(clip_g["pooled"])],
                            axis=-1).astype("float32")  # [2048]
    return joint, pooled
