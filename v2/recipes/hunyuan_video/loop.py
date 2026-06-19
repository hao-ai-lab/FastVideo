"""HunyuanDenoiseLoop — a thin subclass of ``WanDenoiseLoop`` (flow-match Euler).

HunyuanVideo is a rectified flow-match model: the DiT predicts a velocity in unpatchified
``[B, 16, T, H, W]`` latent space, the model timestep is the scalar ``sigma * 1000`` (a per-batch
``LongTensor``), and the Euler update is ``x_next = x + (sigma_next - sigma)*v`` — exactly the math
``WanDenoiseLoop`` already runs. The latent geometry (16 channels, 8x spatial, 4x temporal) is the Wan
default too, so we reuse the loop verbatim and only add the dual-text-encoder marshalling delta below.

The one delta vs Wan is conditioning shape: the Hunyuan DiT consumes TWO text signals —
``encoder_hidden_states = [llama_hidden[B, L, 4096], clip_pooled[B, 768]]`` (a per-token LLaMA sequence
plus a single CLIP-pooled global vector). To keep the loop's ``dit(latent, text_embed, sigma)`` call
identical on the CPU toy and GPU backends (the adapter assembles arch-specifics internally), the program
writes the LLaMA sequence into the ``text_embeds`` slot and the CLIP-pooled vector into ``text_pooled``.
This loop reuses Wan's existing ``context=`` channel (Wan uses it for i2v CLIP-vision embeds) to carry the
pooled vector to the adapter — so the ``dit(x, text_embed, sigma, context=...)`` call is unchanged,
``ToyDiT`` (which means over both ``text_embed`` and ``context``) runs on CPU, and the real
``HunyuanVideoDiT`` adapter packs ``[text_embed, context]`` into the 2-element ``encoder_hidden_states``.

guidance_scale defaults to 1.0 for base HunyuanVideo → ``ClassicCFG.combine`` collapses to the single
``cond`` branch (CFG effectively off), matching the fastvideo pipeline. (``embedded_cfg_scale``/``guidance``
is a no-op on the base model whose ``guidance_in`` is ``None``; see the BRINGUP note in
``torch_hunyuan_video.HunyuanVideoDiT``.)

Faithful to ``fastvideo/pipelines/basic/hunyuan/hunyuan_pipeline.py`` (flow-match DenoisingStage) and
``fastvideo/models/dits/hunyuanvideo.py:HunyuanVideoTransformer3DModel.forward`` (the list-split of
``encoder_hidden_states`` into ``txt`` / ``text_states_2``).
"""
from __future__ import annotations

from v2.loop.contracts import LoopState
from v2.recipes.wan21.loop import WanDenoiseLoop

# AutoencoderKLHunyuanVideo compression matches Wan2.1: z=16, 4x temporal, 8x spatial.
HUNYUAN_LATENT_CHANNELS = 16
HUNYUAN_TEMPORAL_RATIO = 4
HUNYUAN_SPATIAL_RATIO = 8


class HunyuanDenoiseLoop(WanDenoiseLoop):
    """Wan flow-match loop with HunyuanVideo's dual-text-encoder marshalling.

    Reuses ``WanDenoiseLoop.next``/``advance``/``finalize`` unchanged. Overrides only ``init`` to thread the
    CLIP-pooled global vector (slot ``text_pooled``) through Wan's existing ``context=`` channel so the DiT
    adapter can reassemble the 2-element ``encoder_hidden_states`` the Hunyuan forward expects.
    """

    def init(self, req, model, ctx) -> LoopState:
        st = super().init(req, model, ctx)
        # Wan's ``context`` channel (``i2v_img_embeds`` -> dit ``context=``) carries the CLIP-pooled vector
        # for Hunyuan. None when the dual-encode node did not run -> the DiT adapter falls back to a
        # single-tensor encoder_hidden_states (still valid: the Hunyuan forward also accepts a bare tensor
        # whose token 0 is the pooled vector — but the dual-slot path is the faithful one).
        st.scratch["i2v_img_embeds"] = ctx.slots.get("text_pooled")
        return st
