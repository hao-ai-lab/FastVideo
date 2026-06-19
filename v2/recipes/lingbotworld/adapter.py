"""LingBot-World torch adapters (GPU backend) — declared on the card via ``ComponentSpec.adapter`` so
the recipe is self-contained (no edit to the shared ``_make_dit`` dispatch in ``torch_backend.py``).
Imported lazily by ``_explicit_adapter`` only on a GPU box.

LingBot-World-Base-Cam is Wan2.2-I2V-A14B (a 2x14B boundary-routed MoE with CLIP + first-frame i2v
conditioning) PLUS a NEW camera/Plucker conditioning input. The DiT forward gains a ``c2ws_plucker_emb``
argument (``LingBotWorldTransformer3DModel.forward(..., c2ws_plucker_emb=None)``) which a vanilla WanDiT
reuse would silently drop -> the generation would degrade to a generic Wan i2v with no camera control
(BRINGUP blocker #1). ``LingBotWorldDiT`` threads that tensor.

Key design constraint (faithful + conflict-free): the v2 dit-call surface (``ToyDiT.__call__`` and the
loop's ``dit(latent, text_embed, sigma, context=, cond=)``) has NO ``c2ws_plucker_emb`` kwarg, and we may
not edit the toy/loop sampler. So the loop publishes the per-step Plucker tensor onto the adapter as a
plain attribute (``self.c2ws_plucker_emb``) right before the call (the cosmos2 pattern of building
arch-specifics adapter-side); ``__call__`` reads it and forwards it to the real module. The ToyDiT stand-in
simply never has the attribute, so the CPU path is the no-camera (degenerate t2v/i2v) path unchanged.

MoE CPU-offload (BRINGUP blocker #3): 2x14B bf16 won't co-reside on an 80GB GPU. The card declares this
adapter for BOTH ``transformer`` (high-noise) and ``transformer_2`` (low-noise) experts; we keep the
WanDiT offload-group swap so only the boundary-active expert is GPU-resident. ``_make_dit`` does not pass
an ``offload_group`` to an explicit adapter, so the two LingBotWorld experts share a process-wide group
keyed by the wrapped module's ``id`` would be fragile across instances — instead we share a single
class-level group; for the registered single-instance serving path that is exactly the WanDiT MoE
behavior. BRINGUP: revisit if multiple LingBotWorld instances must co-reside.
"""
from __future__ import annotations

import torch

from v2.platform.backends.torch_backend import NUM_TRAIN_TIMESTEPS, WanDiT


class LingBotWorldDiT(WanDiT):
    """``dit(latent[C,T,H,W], text_embed[seq,dim], sigma, context=clip, cond=[mask|cond]) -> velocity``,
    plus an out-of-band ``self.c2ws_plucker_emb`` (numpy ``[6*s^2, F_lat, H_lat, W_lat]``) the loop sets
    per step. Identical i2v conditioning to Wan2.2-I2V-A14B — the 36ch ``[noise|mask+cond]`` concat and
    CLIP ``encoder_hidden_states_image`` — with the extra Plucker camera tensor forwarded to the model.

    Faithful to ``LingBotWorldImageToVideoPipeline`` (== ``WanImageToVideoPipeline`` denoise) + the
    ``c2ws_plucker_emb=batch.c2ws_plucker_emb`` kwarg threaded by ``DenoisingStage`` (denoising.py:182).
    """

    # All LingBotWorld DiT experts share one CPU-offload group (only the boundary-active one is resident).
    _LBW_OFFLOAD_GROUP: dict[str, LingBotWorldDiT] = {}

    def __init__(self, module, *, device, dtype):
        # ``_make_dit`` builds an explicit adapter as ``cls(module, device=, dtype=)`` (no offload_group),
        # so wire the shared MoE offload group here ourselves (2x14B can't co-reside on 80GB). BOTH experts
        # carry the SAME ``config.prefix`` ("Wan"), so we must NOT key the offload group on the prefix — the
        # two adapters would collide on one dict slot and only one expert would ever be offloaded (both end
        # up GPU-resident -> 2x37GB). Key on the wrapped module's identity instead so each expert gets a
        # distinct slot and ``_ensure_resident`` correctly evicts the inactive one at the boundary swap.
        component_id = f"lingbotworld-{id(module):x}"
        super().__init__(module,
                         device=device,
                         dtype=dtype,
                         offload_group=LingBotWorldDiT._LBW_OFFLOAD_GROUP,
                         component_id=str(component_id))
        # The per-step camera/Plucker embedding the loop publishes before each call (None == no camera).
        self.c2ws_plucker_emb = None

    @torch.no_grad()
    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        self._ensure_resident()
        hs = self._t(latent)
        if cond is not None:  # i2v: concat [noise (16ch) ; mask+cond_latent (20ch)] -> 36ch input
            hs = torch.cat([hs, self._t(cond)], dim=1)
        ehs = self._t(text_embed)
        # LingBotWorld uses the SCALAR-timestep path (no expand_timesteps); timestep = sigma*1000.
        ts = float(sigma) * NUM_TRAIN_TIMESTEPS
        timestep = torch.tensor([ts], device=self.device)
        img = self._t(context)  # CLIP image embeds -> encoder_hidden_states_image
        plucker = None
        if self.c2ws_plucker_emb is not None:  # camera conditioning (BRINGUP: needs a per-request input)
            plucker = self._t(self.c2ws_plucker_emb)  # [1, 6*s^2, F_lat, H_lat, W_lat]
        with self._ctx():
            velocity = self.module(hidden_states=hs,
                                   encoder_hidden_states=ehs,
                                   timestep=timestep,
                                   encoder_hidden_states_image=img,
                                   c2ws_plucker_emb=plucker)
        return self._n(velocity)  # rectified-flow velocity (BRINGUP risk C)
