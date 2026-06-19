"""FLUX.2 torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so the FLUX.2
recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/``_make_text_encoder`` dispatch
in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only on a GPU box.

These three pieces CANNOT reuse the Wan/T5 adapters (BRINGUP — GPU-verify each against a real FLUX.2 ckpt):

* ``Flux2DiT`` — the dual-stream MMDiT. The loop hands ``dit(latent[64,T,h,w], text_embed[seq,4096], σ)``;
  this adapter builds everything FLUX.2-specific INTERNALLY so the loop stays toy-compatible:
    - σ is passed DIRECTLY as the timestep (the pipeline's ``timestep/1000``); the DiT re-multiplies by
      1000. Guidance is passed RAW (``embedded_guidance`` scale, ×1000 internally). Passing σ·1000 (the
      Wan convention) would 1e6-blow-up the timestep embedding.
    - ``encoder_hidden_states`` is wrapped in a 1-element list (the forward indexes [0]).
    - ``img_ids`` = ``cartesian_prod(arange(T), arange(h), arange(w), arange(1))`` and ``txt_ids`` =
      ``cartesian_prod(1,1,1,arange(seq))`` — the N-D RoPE position tensors threaded into every forward.
    - run WITHOUT autocast (Gate 2: FLUX.2 long-seq attention breaks under autocast); module + inputs bf16.
  Returns the bare velocity. Faithful to ``fastvideo/pipelines/stages/denoising.py`` (the ``_is_flux``
  gates) + ``flux_2.py`` forward + ``flux_2_latent_preparation.py`` / ``flux_2_text_encoding.py`` id builders.

* ``Flux2VAE`` — decode does the FLUX.2 BatchNorm whitening + 2×2 unpatchify (NOT Wan mean/std):
  ``z·bn_std + bn_mean`` (the complete inverse normalization; no scaling_factor step) then unpack 64→16ch
  full-spatial, then ``module.decode``. Faithful to ``DecodingStage._flux2_bn_denorm_and_unpatchify`` +
  ``_unpatchify_latents`` (fastvideo/pipelines/stages/decoding.py). Image-only: T==1 squeezed.

* ``Flux2Mistral3Encoder`` / ``Flux2Qwen3Encoder`` — the chat-template multi-layer hidden-state stack.
  Apply the FLUX2 system+user chat template (dev only), tokenize padded to max_length (512),
  ``output_hidden_states=True``, stack the chosen decoder layers (dev 10/20/30, klein 9/18/27),
  permute+reshape to [seq, n_layers·hidden] (= 4096). Faithful to ``Flux2TextEncodingStage`` +
  ``flux2_klein_postprocess_text`` (fastvideo/configs/pipelines/flux_2.py).
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import TorchComponent, _to_numpy

# FLUX.2 system message prepended to the user prompt via the chat template (dev). Faithful copy of
# ``fastvideo/pipelines/basic/flux_2/flux_2_text_encoding.py:FLUX2_SYSTEM_MESSAGE``.
FLUX2_SYSTEM_MESSAGE = ("You are an AI that reasons about image descriptions. You give structured "
                        "responses focusing on object relationships, object\nattribution and actions "
                        "without speculation.")

NUM_TRAIN_TIMESTEPS = 1000  # the DiT re-multiplies timestep AND guidance by this internally


class Flux2DiT(TorchComponent):
    """``dit(latent[64,T,h,w], text_embed[seq,4096], sigma) -> velocity[64,T,h,w]``.

    Real forward (flux_2.py): ``forward(hidden_states[B,64,T,h,w], encoder_hidden_states=[ehs], timestep,
    img_ids, txt_ids, guidance) -> velocity (bare 5D tensor)``. The pipeline passes ``timestep == sigma``
    (DiT ×1000 internally) and ``guidance == embedded_cfg_scale`` raw (DiT ×1000 internally)."""

    def __init__(self, module, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        # FLUX.2-dev uses guidance_embeds=True (a guidance arg); klein distilled has no guidance embedder.
        self.guidance_embeds = bool(getattr(module, "guidance_embeds", True))

    def _ids(self, t: int, h: int, w: int, txt_len: int):
        # img_ids over (T, h, w, 1); txt_ids over (1,1,1,seq) — the N-D RoPE position tensors. Faithful to
        # flux_2_latent_preparation.py (latent_ids) + flux_2_text_encoding.py (_prepare_flux2_text_ids).
        img_ids = torch.cartesian_prod(torch.arange(t, device=self.device), torch.arange(h, device=self.device),
                                       torch.arange(w, device=self.device),
                                       torch.arange(1, device=self.device)).unsqueeze(0)
        txt_ids = torch.cartesian_prod(torch.arange(1, device=self.device), torch.arange(1, device=self.device),
                                       torch.arange(1, device=self.device),
                                       torch.arange(txt_len, device=self.device)).unsqueeze(0)
        return img_ids, txt_ids

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, 64, T, h, w]
        ehs = self._t(text_embed)  # [1, seq, 4096]
        b, _c, t, h, w = hs.shape
        s = float(sigma)
        # timestep == sigma directly (NOT sigma*1000) — the DiT multiplies by 1000 internally.
        timestep = torch.tensor([s] * b, device=self.device, dtype=self.dtype)
        guidance = None
        if self.guidance_embeds:  # dev: embedded guidance, raw scale
            guidance = torch.tensor([float(self._guidance(context))] * b, device=self.device, dtype=self.dtype)
        img_ids, txt_ids = self._ids(t, h, w, ehs.shape[1])
        with self._ctx(current_timestep=s):  # NO autocast (Gate 2) — module+inputs bf16
            velocity = self.module(
                hidden_states=hs,
                encoder_hidden_states=[ehs],  # forward indexes [0]
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance)
        return self._n(velocity)  # rectified-flow velocity

    @staticmethod
    def _guidance(context) -> float:
        # The loop threads the embedded-guidance scale through the ``context`` slot of the dit-call (the
        # toy-compatible signature). BRINGUP: the request-API extension that carries the scale per-request
        # is pending; until then the card's sampling default (dev 4.0) is the value, set on the adapter by
        # the loop wiring. Defaults to 4.0 if unset.
        return 4.0 if context is None else float(context)


class Flux2VAE(TorchComponent):
    """FLUX.2 image VAE. decode: BN-denorm (``z·bn_std + bn_mean``) → 2×2 unpatchify (64→16ch full-spatial)
    → ``module.decode`` → image[-1,1]. encode: ``module.encode`` (mode) → packed latent (BRINGUP: pack +
    BN-whiten on encode is GPU-only; t2i never encodes). NOT the Wan (z−mean)/std normalization."""

    def _bn_stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bn = self.module.bn
        eps = float(getattr(bn, "eps", 1e-5))
        running_mean = bn.running_mean.view(1, -1, 1, 1).to(like.device, like.dtype)
        running_var = bn.running_var.view(1, -1, 1, 1).to(like.device, like.dtype)
        bn_std = torch.sqrt(torch.clamp(running_var + eps, min=1e-6))
        return running_mean, bn_std

    @staticmethod
    def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
        # (B, C*4, h, w) -> (B, C, 2h, 2w): inverse of the 2×2 patch packing (DecodingStage._unpatchify_latents).
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float()  # [1, 64, T, h, w]
        if z.ndim == 5:  # image VAE expects 4D; T==1 squeezed
            z = z.squeeze(2)
        running_mean, bn_std = self._bn_stats(z)
        z = z * bn_std + running_mean  # BN-denorm: complete inverse (no scaling_factor)
        z = self._unpatchify(z)  # 64 -> 16ch full-spatial
        image = self.module.decode(z.to(self.dtype))
        if hasattr(image, "sample"):
            image = image.sample
        elif isinstance(image, tuple | list):
            image = image[0]
        image = (image / 2 + 0.5).clamp(0, 1)  # -> [0,1]
        return self._n(image.unsqueeze(2))  # restore the singleton T for a uniform [C,T,H,W]

    @torch.no_grad()
    def encode(self, image):
        # BRINGUP: t2i never encodes; the 2×2 pack + BN whiten on the encode path is GPU-only and unverified.
        x = self._t(image)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return _to_numpy(z)


class _Flux2TextEncoderBase(TorchComponent):
    """Shared FLUX.2 chat-template multi-layer hidden-state stack. Subclasses set ``layers`` and whether
    the FLUX2 system-message chat template is applied (dev=Mistral3) or a plain user template (klein=Qwen3)."""

    layers: tuple[int, ...] = (10, 20, 30)
    use_system_message: bool = True

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 512):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _messages(self, prompt: str):
        user = {"role": "user", "content": [{"type": "text", "text": (prompt or "").replace("[IMG]", "")}]}
        if self.use_system_message:
            system = {"role": "system", "content": [{"type": "text", "text": FLUX2_SYSTEM_MESSAGE}]}
            return [[system, user]]
        return [[user]]

    @torch.no_grad()
    def encode(self, text):
        inputs = self.tokenizer.apply_chat_template(self._messages(text),
                                                    add_generation_prompt=False,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with self._ctx():
            out = self.module(input_ids=input_ids,
                              attention_mask=attention_mask,
                              output_hidden_states=True,
                              use_cache=False)
        if out.hidden_states is None:
            raise ValueError("FLUX.2 requires output_hidden_states=True from the text encoder")
        # stack the chosen decoder layers -> [B, n_layers, seq, hidden] -> permute/reshape [B, seq, n·hidden]
        stacked = torch.stack([out.hidden_states[k] for k in self.layers], dim=1)
        b, n_layers, seq_len, hidden = stacked.shape
        embed = stacked.permute(0, 2, 1, 3).reshape(b, seq_len, n_layers * hidden)
        return _to_numpy(embed.squeeze(0))  # [seq, n_layers*hidden] (= 4096)


class Flux2Mistral3Encoder(_Flux2TextEncoderBase):
    """FLUX.2-dev text encoder: Mistral3, hidden-state layers 10/20/30, FLUX2 system-message chat template."""
    layers = (10, 20, 30)
    use_system_message = True


class Flux2Qwen3Encoder(_Flux2TextEncoderBase):
    """FLUX.2-klein text encoder: Qwen3, hidden-state layers 9/18/27, plain user chat template (no system msg)."""
    layers = (9, 18, 27)
    use_system_message = False
