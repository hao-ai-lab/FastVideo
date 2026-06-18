"""Lucy-Edit denoise loop — the Wan flow-match denoise, with a Lucy-specific conditioning fill.

Lucy-Edit is the Wan2.2-5B T2V network run as a video-to-video editor: the only change vs T2V is
that the model input is the per-step noise latent **channel-concatenated** with a frozen,
VAE-encoded latent of the INPUT video (the thing being edited). The flow-match denoise math —
sigma schedule, CFG combine, Euler step — is byte-identical to ``WanDenoiseLoop``.

The conditioning is threaded through the SAME hook the shared ``WanDenoiseLoop`` already exposes for
i2v: the program writes the conditioning latent into the ``i2v_cond`` slot, the loop reads it into
``st.scratch["i2v_cond"]`` and passes it to the DiT as ``cond=``, and the Wan torch adapter does
``torch.cat([noise, cond], dim=1)`` (its existing i2v concat path). For Lucy-Edit the cond is the
**full** video latent (48ch) appended to the 48ch noise -> the 96ch DiT input
(``LucyEditDevConfig.dit_config.in_channels == 96``); there is NO 4-channel binary mask and NO CLIP
image embedding (``i2v_img_embeds`` stays None), which is exactly how Lucy differs from Wan i2v /
generic Wan v2v (the generic v2v path additionally appends a zero-pad block — see
``fastvideo/pipelines/stages/denoising.py``; Lucy's ``is_lucy_edit`` branch concatenates only
``[latent, video_latent]``).

CRITICAL (GPU bring-up): the Lucy DiT is ALWAYS 96-channel-in (in_channels=96), even in the t2v
degrade where there is no input video. The 48ch noise alone would mis-shape the patch-embedding
conv. So this loop's ``init`` fills ``i2v_cond`` with a ZERO latent (48ch, matching the noise
latent's [C,T,h,w]) when the program supplied none -> the Wan adapter still concats to 96 channels.
This is the faithful v2v-degrades-to-t2v behavior: "no edit conditioning" == a zero video latent.

GPU BRINGUP note: the 5B DiT uses ``expand_timesteps`` (a per-token timestep vector) on the real
forward; the shared Wan torch adapter passes a scalar (1D) timestep, which ``WanTransformer3DModel``
accepts via its ``ts_seq_len is None`` path (the standard Wan timestep embedding). Per-token timestep
expansion lives inside the adapter, not this loop's flow-match arithmetic. See NOTES in ``card.py``.
"""
from __future__ import annotations

import numpy as np

from v2.recipes.wan21.loop import WanDenoiseLoop as _WanDenoiseLoop
from v2.recipes.wan21.loop import latent_shape

# Lucy-Edit rides the Wan2.2-5B (TI2V) VAE geometry: z_dim=48, 16x spatial, 4x temporal.
LUCY_LATENT_CHANNELS = 48
LUCY_SPATIAL_RATIO = 16
LUCY_TEMPORAL_RATIO = 4


class WanDenoiseLoop(_WanDenoiseLoop):
    """Wan flow-match denoise with the Lucy 96ch-in conditioning fill.

    Identical to the shared ``WanDenoiseLoop`` except ``init`` guarantees ``i2v_cond`` is a 48ch
    latent (the VAE-encoded input video when present, else zeros) so the Wan adapter's concat always
    produces the 96ch input ``WanTransformer3DModel(in_channels=96)`` requires."""

    def init(self, req, model, ctx):
        st = super().init(req, model, ctx)
        if st.scratch.get("i2v_cond") is None:
            # t2v degrade: no input video -> a zero 48ch conditioning latent matching the noise shape.
            # The Wan adapter channel-concats [noise(48) | cond(48)] -> the 96ch Lucy DiT input.
            x = st.latents["video"]
            st.scratch["i2v_cond"] = np.zeros_like(np.asarray(x, dtype="float32"))
        return st


__all__ = ["WanDenoiseLoop", "latent_shape", "LUCY_LATENT_CHANNELS", "LUCY_SPATIAL_RATIO", "LUCY_TEMPORAL_RATIO"]
