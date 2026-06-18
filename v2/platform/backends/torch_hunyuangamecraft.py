"""HunyuanGameCraft torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter``
so the GameCraft recipe is self-contained (no edit to the shared ``_make_dit``/``_make_vae``/
``_make_text_encoder`` dispatch in ``torch_backend.py``). Imported lazily by ``_explicit_adapter`` only
on a GPU box.

GameCraft is an *interactive* world model (camera/action-conditioned i2v). The deltas vs Wan/Cosmos that
force bespoke adapters (the BRINGUP/blocker list in the port spec):

* ``GameCraftDiT`` — the DiT forward expects a **33-channel** input ``x = cat([latent16 | gt_latent16 |
  mask1], dim=1)`` that the *denoise loop* (not the DiT) assembles, plus a **list** of text states
  ``encoder_hidden_states = [llama_states(4096d), clip_pooled(768d)]``, an optional ``camera_states``
  (CameraNet Plücker), ``guidance=None`` (NO embedded guidance — plain ClassicCFG), and the model
  timestep ``σ·1000``. The DiT returns a 16-channel rectified-flow velocity. The loop hands this adapter
  the bare 16ch latent + the cond slots; the adapter does the 33ch concat + dual-text-list packing
  INTERNALLY so the loop's dit-call stays ``dit(latent, text_embed, sigma, ...)`` (toy-compatible).
  Cannot reuse ``WanDiT`` (it concats a 20ch i2v cond and passes ``encoder_hidden_states_image``).
* ``GameCraftVAE`` — SCALAR ``scaling_factor=0.476986`` normalization (``latent = sample()·sf``;
  ``decode(latent/sf)``), NOT Wan's per-channel ``latents_mean``/``latents_std``. Reusing ``WanVAE``
  (which reads ``module.latents_mean``) would crash / corrupt latents.
* ``GameCraftLlamaEncoder`` / ``GameCraftClipEncoder`` — the two text encoders are LLaVA-LLaMA-3 (4096d
  hidden states, the DiT's ``text_states``) and CLIP ViT-L/14 (768d pooled, the DiT's ``text_states_2``).
  The toy CPU path uses ``ToyTextEncoder`` for both; these GPU subclasses are written-not-run.

Everything below is faithful to ``fastvideo/pipelines/stages/gamecraft_denoising.py`` (the DiT call) and
``fastvideo/pipelines/stages/gamecraft_image_encoding.py`` (the scalar VAE scaling). The camera / action
(CameraNet) conditioning needs a request-API camera-input channel v2 lacks today, so the t2v/degenerate
path passes ``camera_states=None`` (the DiT's ``if camera_states is not None`` branch is skipped) — see
the BRINGUP notes in ``loop.py``/``program.py``.
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import T5Encoder, TorchComponent, _to_numpy

# Flow-match timestep convention (same as Wan): the loop hands the raw sigma (1->0); the GameCraft DiT
# (like diffusers/FastVideo) embeds ``timestep = sigma * num_train_timesteps``. BRINGUP risk B.
NUM_TRAIN_TIMESTEPS = 1000
GAMECRAFT_SCALING_FACTOR = 0.476986  # GameCraftVAE.config.scaling_factor (scalar, NOT per-channel stats)


class GameCraftDiT(TorchComponent):
    """``dit(latent[16,T,h,w], text_embed, sigma, context=clip_pooled, *, cond=cond_pack) -> velocity[16,T,h,w]``.

    The loop calls this with the bare 16-channel noise latent. The adapter assembles the 33-channel DiT
    input + the dual-text list + the camera states INTERNALLY (so the loop's dit-call signature matches
    the toy ``ToyDiT.__call__``). Faithful to ``GameCraftDenoisingStage.forward``:

      latent_model_input = cat([latents, gt_latents, conditioning_mask], dim=1)   # [B,33,T,h,w]
      noise_pred = transformer(latent_model_input, [llama, clip], t·1000,
                               camera_states=cam, guidance=None)                   # [B,16,T,h,w]

    Conventions threaded through the call boundary (all numpy at the seam):
      * ``text_embed``  -> ``text_states`` (LLaMA hidden states), ``encoder_hidden_states[0]``.
      * ``context``     -> ``text_states_2`` (CLIP pooled), ``encoder_hidden_states[1]``. The loop reuses
        the WanDenoiseLoop ``context=`` kwarg slot to carry the per-branch CLIP pooled embed (cond vs
        uncond), so CFG combines correctly. ``None`` -> single-text fallback (still valid for the toy).
      * ``cond``        -> a dict ``{"gt_latents", "mask", "camera_states"}`` the loop builds once and
        passes every step. ``None`` (pure t2v) -> zero gt_latents + zero mask + no camera (the
        ``GameCraftDenoisingStage`` fallback), i.e. the degenerate standard-denoise path."""

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        hs = self._t(latent)  # [1, 16, T, h, w]
        b, _c, t, h, w = hs.shape
        # --- 33-channel concat [latent16 | gt_latent16 | mask1] (assembled HERE, not in the DiT) ------ #
        cond = cond or {}
        gt = self._t(cond.get("gt_latents")) if cond.get("gt_latents") is not None else torch.zeros_like(hs)
        mask = (self._t(cond.get("mask"))
                if cond.get("mask") is not None else torch.zeros(b, 1, t, h, w, device=self.device, dtype=self.dtype))
        model_input = torch.cat([hs, gt, mask], dim=1)  # [1, 33, T, h, w]
        # --- dual text states list: [LLaMA(4096d), CLIP-pooled(768d)] -------------------------------- #
        ehs = [self._t(text_embed)]  # text_states -> [1, seq, 4096]
        if context is not None:  # CLIP pooled (per CFG branch) -> text_states_2
            ehs.append(self._t(context))
        # --- camera_states (CameraNet Plücker): None for the t2v/degenerate path (BRINGUP) ----------- #
        camera = self._t(cond.get("camera_states")) if cond.get("camera_states") is not None else None
        # timestep = sigma*1000, shape [B] (the stage does ``t.repeat(B)``). BRINGUP risk B.
        ts = torch.tensor([float(sigma) * NUM_TRAIN_TIMESTEPS], device=self.device, dtype=self.dtype).expand(b)
        with self._ctx():
            velocity = self.module(
                model_input,
                ehs,
                ts,
                camera_states=camera,
                guidance=None,  # NO embedded guidance (guidance_embeds=False); plain CFG
                return_dict=False)
        return self._n(velocity)  # rectified-flow velocity (loop integrates via FLOW_MATCH_STEP)


class GameCraftVAE(TorchComponent):
    """Scalar-scaling VAE adapter (GameCraft uses a single ``scaling_factor``, NOT per-channel stats).

    Faithful to ``GameCraftImageVAEEncodingStage`` (encode: ``sample()·sf``) and ``DecodingStage`` (decode:
    ``decode(latent/sf)``). The reference-image preprocessing (resize/center-crop/normalize) lives in the
    program's image-cond node, not here — this adapter only marshals + applies the scalar factor."""

    def __init__(self, module, *, device, dtype, scaling_factor: float = GAMECRAFT_SCALING_FACTOR):
        super().__init__(module, device=device, dtype=dtype)
        # Prefer the loaded module's own config when present; fall back to the published GameCraft value.
        self.scaling_factor = float(getattr(getattr(module, "config", None), "scaling_factor", scaling_factor))

    @torch.no_grad()
    def encode(self, video):
        x = self._t(video)
        out = self.module.encode(x)
        dist = getattr(out, "latent_dist", out)
        z = dist.sample() if hasattr(dist, "sample") else (dist.mode() if hasattr(dist, "mode") else dist)
        return self._n(z.float() * self.scaling_factor)  # scaled latent the DiT expects

    @torch.no_grad()
    def decode(self, latent):
        z = self._t(latent).float() / self.scaling_factor  # invert the scalar scaling -> raw latent
        video = self.module.decode(z.to(self.dtype))
        if hasattr(video, "sample"):
            video = video.sample
        return self._n(video)  # video [B,3,T,H,W] in [-1,1]


class GameCraftClipEncoder(TorchComponent):
    """CLIP ViT-L/14 text encoder -> POOLED embedding (768d), the DiT's ``text_states_2``.

    BRINGUP: written-not-run. The program writes this into the ``clip_text_embeds`` slot; the loop threads
    it through the ``context=`` slot per CFG branch. Returns the pooled output (``[768]`` or ``[1,768]``),
    matching ``batch.clip_embedding_pos``/``clip_embedding_neg`` in the fastvideo stage."""

    def __init__(self, module, tokenizer, *, device, dtype, max_length: int = 77):
        super().__init__(module, device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "",
                              return_tensors="pt",
                              max_length=self.max_length,
                              truncation=True,
                              padding="max_length")
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            out = self.module(input_ids=ids, attention_mask=mask)
        # CLIPTextModel exposes ``pooler_output`` (the [EOS]-token projection) — the GameCraft text_states_2.
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[:, -1]
        return _to_numpy(pooled.squeeze(0))


class GameCraftLlamaEncoder(T5Encoder):
    """LLaVA-LLaMA-3-8B text encoder -> hidden states (4096d), the DiT's ``text_states``.

    BRINGUP: written-not-run. GameCraft consumes an intermediate hidden-state layer via the Hunyuan
    ``llama_postprocess`` (``output_hidden_states=True``) plus the LLaVA prompt template — those
    preprocess/postprocess functions are not yet ported to v2 (blocker 4). This subclass currently reuses
    the (U)MT5 ``encode`` contract (real-token rows, no fixed zero-pad surprises) as a structural
    placeholder; the real LLaVA template + hidden-layer pick must be wired during GPU bring-up."""

    @torch.no_grad()
    def encode(self, text):
        toks = self.tokenizer(text or "", return_tensors="pt", max_length=self.max_length, truncation=True)
        ids = toks.input_ids.to(self.device)
        mask = toks.attention_mask.to(self.device)
        with self._ctx():
            # BRINGUP: real GameCraft passes output_hidden_states=True and picks an intermediate layer via
            # llama_postprocess; here we take last_hidden_state as the structural stand-in.
            out = self.module(input_ids=ids, attention_mask=mask)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).squeeze(0)
        return _to_numpy(torch.nan_to_num(hidden, nan=0.0))


# Re-export the scalar scaling constant for the loop/program to keep the value single-sourced.
__all__ = ["GameCraftDiT", "GameCraftVAE", "GameCraftClipEncoder", "GameCraftLlamaEncoder", "GAMECRAFT_SCALING_FACTOR"]
