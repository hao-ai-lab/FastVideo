"""HY-WorldPlay torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so the
recipe is self-contained (no edit to the shared ``_make_dit`` / ``_make_vae`` / ``_make_text_encoder`` /
``_make_image_encoder`` dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only
on a GPU box.

Faithful to ``fastvideo/pipelines/stages/hyworld_denoising.py:HYWorldDenoisingStage`` and
``fastvideo/models/dits/hyworld/hyworld.py:HYWorldTransformer3DModel.forward``:

  * ``HYWorldDiT`` — the rectified-flow velocity predictor. Its ``__call__`` keeps the
    cosmos2/ToyDiT-compatible ``dit(latent, text_embed, sigma, context=...)`` signature and marshals ALL
    the HY-WorldPlay specifics INTERNALLY: the 65ch ``[noise(32) | cond(32) + mask(1)]`` channel-concat,
    the LIST ``encoder_hidden_states=[qwen, byt5]`` + LIST ``encoder_hidden_states_image=[siglip]``, the
    per-latent-frame heterogeneous ``timestep`` (context frames pinned at ``stabilization_level - 1``) +
    the scalar ``timestep_txt``, the per-frame ``action`` (flattened B*T) + ``viewmats[B,T,4,4]`` /
    ``Ks[B,T,3,3]``, and the ``encoder_attention_mask=[mask, mask2]`` swapped per pos/neg branch. Returns
    the bare velocity tensor [B,32,T,H,W]. BRINGUP: written-not-run; needs the GPU dense+ProPE flash-attn
    backend + a pose-carrying request.
  * ``HYWorldVAE`` — encode -> ``.mode() * scaling_factor`` (1.03682, no shift/mean); decode inverts.
  * ``HYWorldQwenEncoder`` / ``HYWorldByT5Encoder`` / ``HYWorldSiglipEncoder`` — the 3 conditioning
    streams. The Qwen adapter runs the video chat template, takes ``hidden_states[-3]``, and crops the
    first 108 template tokens (BRINGUP). For plain t2v the byt5/siglip streams are zeroed.
"""
from __future__ import annotations

from typing import Any

import torch

from v2.platform.backends.torch_backend import T5Encoder, TorchComponent, _to_numpy

# Flow-match timestep convention: timestep = sigma * num_train_timesteps (matches WanDiT / the
# HYWorldDenoisingStage which feeds the scheduler's discrete timesteps in 0..1000).
NUM_TRAIN_TIMESTEPS = 1000
# SigLIP image stream geometry (HunyuanVideo1.5 / HY-WorldPlay): 729 tokens x 1152 dim.
SIGLIP_TOKENS = 729
SIGLIP_DIM = 1152
# HY-WorldPlay latent geometry (AutoencoderKLHYWorld): z=32 + 1 mask channel for the cond latent.
HYWORLD_LATENT_CHANNELS = 32
HYWORLD_VAE_SCALING_FACTOR = 1.03682


class HYWorldDiT(TorchComponent):
    """``dit(latent[C,T,h,w], text_embed[seq,dim], sigma, context=chunk_ctx) -> velocity[32,T,h,w]``.

    ``latent`` is the current chunk's NOISE latent (32ch). ``text_embed`` is the Qwen (mllm) embed for
    the active branch; the byt5 + siglip streams and the camera/action tensors are resolved from the
    sibling components + ``context`` here. ``sigma`` is the live step's sigma (-> ``timestep = sigma*1000``
    broadcast per latent frame; context/history frames in ``context`` are pinned at the stabilization
    level). The 65ch ``[noise | cond + mask]`` concat is done HERE (faithful to the stage), so the loop
    never assembles arch-specifics. BRINGUP: the chunk>0 camera-aligned memory path + ProPE attention +
    action need a pose-carrying request; absent, zero camera/action are built (degenerate t2v)."""

    def __init__(self, module, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)

    def _zero_image_embeds(self) -> torch.Tensor:
        # T2V: zero SigLIP embeds -> the DiT's ``torch.all(image_embeds == 0)`` masks the image stream.
        return torch.zeros(1, SIGLIP_TOKENS, SIGLIP_DIM, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context: Any = None, *, cond=None):
        ctx = context if isinstance(context, dict) else {}
        hs = self._t(latent)  # [1, 32, T, h, w]
        _b, _c, t, _h, _w = hs.shape
        # 65ch in-channel concat: [noise(32) | cond_latent(32) + mask(1)]. The cond latent is the
        # VAE-encoded first frame (i2v) or zeros (t2v) + a 1ch mask -> 33ch. BRINGUP: i2v_cond plumbing.
        i2v_cond = ctx.get("i2v_cond")
        if i2v_cond is not None:
            cond_latents = self._t(i2v_cond)[:, :, :t]  # [1, 33, T, h, w]
        else:
            cond_latents = torch.zeros(1, HYWORLD_LATENT_CHANNELS + 1, t, _h, _w, device=self.device, dtype=self.dtype)
        hs = torch.cat([hs, cond_latents], dim=1)  # -> 65ch (faithful to the stage's torch.concat)

        # 3-stream conditioning: encoder_hidden_states must be a LIST [qwen, byt5]; image a LIST [siglip].
        qwen = self._t(text_embed)  # [1, seq, dim]
        byt5 = self._t(ctx.get("byt5_embeds")) if ctx.get("byt5_embeds") is not None else \
            torch.zeros(1, 1, qwen.shape[-1], device=self.device, dtype=self.dtype)
        img = self._t(ctx.get("i2v_img_embeds")) if ctx.get("i2v_img_embeds") is not None else \
            self._zero_image_embeds()
        # Per-pos/neg attention masks (BRINGUP: the stage swaps prompt_attention_mask <-> negative). The
        # loop runs each branch separately, so the matching mask is selected per call by the caller's text.
        mask = torch.ones(1, qwen.shape[1], device=self.device, dtype=self.dtype)
        mask2 = torch.ones(1, byt5.shape[1], device=self.device, dtype=self.dtype)

        # Per-latent-frame timestep (NOT scalar): live ``t`` for the chunk frames; context/history frames
        # (chunk>0, BRINGUP) pinned at ``stabilization_level - 1``. timestep_txt is a separate scalar.
        ts_val = float(sigma) * NUM_TRAIN_TIMESTEPS
        timestep = torch.full((t, ), ts_val, device=self.device, dtype=torch.long)
        timestep_txt = torch.tensor([ts_val], device=self.device, dtype=torch.long)

        # Camera + action conditioning (BRINGUP): per-frame viewmats/Ks/action. Absent -> identity camera
        # + zero action (the action_in / prope projections are zero-init, so this is a faithful no-op).
        cam = ctx.get("camera")
        if cam is not None:
            start, end = ctx.get("chunk", (0, t))
            viewmats = self._t(cam["viewmats"])[:, start:end]
            Ks = self._t(cam["Ks"])[:, start:end]
            action = self._t(cam["action"])[:, start:end].reshape(-1)
        else:
            viewmats = torch.eye(4, device=self.device, dtype=self.dtype).expand(1, t, 4, 4).contiguous()
            Ks = torch.eye(3, device=self.device, dtype=self.dtype).expand(1, t, 3, 3).contiguous()
            action = torch.zeros(t, device=self.device, dtype=self.dtype)

        with self._ctx():
            velocity = self.module(
                hs,
                [qwen, byt5],
                timestep=timestep,
                encoder_hidden_states_image=[img],
                encoder_attention_mask=[mask, mask2],
                action=action.to(self.dtype),
                viewmats=viewmats.to(self.dtype),
                Ks=Ks.to(self.dtype),
                timestep_txt=timestep_txt,
            )
        return self._n(velocity)  # bare rectified-flow velocity [32, T, h, w]


class HYWorldVAE(TorchComponent):
    """AutoencoderKLHYWorld: encode -> ``.mode() * scaling_factor`` (1.03682, no shift/mean);
    decode -> ``/ scaling_factor`` then decode. Used for the first-frame cond latent + the final decode."""

    def __init__(self, module, *, device, dtype, scaling_factor: float = HYWORLD_VAE_SCALING_FACTOR):
        super().__init__(module, device=device, dtype=dtype)
        self.scaling_factor = float(getattr(getattr(module, "config", None), "scaling_factor", scaling_factor))

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        dist = self.module.encode(x)
        z = dist.mode() if hasattr(dist, "mode") else (dist.sample() if hasattr(dist, "sample") else dist)
        return self._n(z.float() * self.scaling_factor)

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float() / self.scaling_factor
        video = self.module.decode(z.to(self.dtype))  # -> video [B,3,T,H,W] in [-1,1]
        return self._n(video)


class HYWorldQwenEncoder(TorchComponent):
    """Qwen2.5-VL mllm (primary text). BRINGUP: faithful path runs the video chat template
    (PROMPT_TEMPLATE_ENCODE_VIDEO system msg), takes ``hidden_states[-3]``, and crops the first 108
    template tokens. Returns the cropped real-token embedding (numpy). Encoder weights are fp32 here; on
    the box mirror the fastvideo TextEncodingStage marshalling exactly."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 256, crop_start: int = 108):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.crop_start = crop_start  # PROMPT_TEMPLATE_ENCODE_VIDEO prepends 108 template tokens

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        # hidden_states[-3] is the layer the HY-WorldPlay mllm conditioning uses (BRINGUP: confirm on box).
        hidden = hs[-3] if hs is not None and len(hs) >= 3 else \
            (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])
        hidden = hidden.squeeze(0)
        if self.crop_start and hidden.shape[0] > self.crop_start:  # crop the prompt-template tokens
            hidden = hidden[self.crop_start:]
        return _to_numpy(torch.nan_to_num(hidden, nan=0.0))


class HYWorldByT5Encoder(T5Encoder):
    """ByT5 glyph encoder (rendered text; text_states_dim_2=1472). BRINGUP: the faithful path extracts
    quoted glyph text (``extract_glyph_texts`` regex) before encoding and returns ``last_hidden_state``;
    for plain t2v there is no glyph text -> a single zero token. Subclasses ``T5Encoder`` (raw, NO Wan
    fixed-length zero-pad)."""

    @torch.no_grad()
    def encode(self, text):
        glyph = self._extract_glyph(text)
        if not glyph:  # no quoted glyph text -> a 1-token zero embed (the DiT cond_type_embed still adds)
            dim = int(getattr(getattr(self.module, "config", None), "d_model", 1472))
            return _to_numpy(torch.zeros(1, dim, device=self.device, dtype=torch.float32))
        toks = self.tokenizer(glyph, return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        return _to_numpy(torch.nan_to_num(hidden, nan=0.0))

    @staticmethod
    def _extract_glyph(text) -> str:
        import re
        if not text:
            return ""
        return " ".join(re.findall(r'["“”‘’\'](.+?)["“”‘’\']', text))


class HYWorldSiglipEncoder(TorchComponent):
    """SigLIP image encoder (729 tokens x 1152 dim) for the i2v/first-frame stream. BRINGUP: the faithful
    path resizes+center-crops to the target res, runs ``processor.preprocess`` (fp16), and returns
    ``last_hidden_state``. For t2v there is no image -> the program zeroes this stream (the DiT then masks
    it via ``torch.all(image_embeds == 0)``)."""

    def __init__(self, module, processor=None, *, device, dtype):
        super().__init__(module, device=device, dtype=dtype)
        self.processor = processor

    @torch.no_grad()
    def encode_image(self, image):
        if image is None or self.processor is None:  # t2v: zero embeds -> DiT masks the image stream
            return _to_numpy(torch.zeros(SIGLIP_TOKENS, SIGLIP_DIM, device=self.device, dtype=torch.float32))
        inputs = self.processor.preprocess(images=image, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        with self._ctx():
            out = self.module(pixel_values=inputs["pixel_values"])
        embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return _to_numpy(embeds.squeeze(0))
