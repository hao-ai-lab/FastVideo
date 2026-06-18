"""Lucy-Edit denoise loop — the Wan flow-match denoise, REUSED unchanged.

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

Because ``i2v_cond`` is non-None whenever the input video is present, the shared loop already sets
``capturable=False`` for those steps (the conditioning latent is threaded outside the CUDA-graph
workspace) and leaves the deterministic ODE step capturable when it degrades to t2v (no input
video). So this module is a pure re-export with a Lucy-tuned ``latent_shape`` default; there is no
new sampler and no new loop subclass.

GPU BRINGUP note: the 5B DiT uses ``expand_timesteps`` (a per-token timestep vector) on the real
forward; that lives inside the Wan torch adapter / fastvideo denoising stage, NOT in this loop's
flow-match arithmetic, so the loop is unchanged. See NOTES in ``card.py``.
"""
from __future__ import annotations

from v2.recipes.wan21.loop import WanDenoiseLoop, latent_shape

# Lucy-Edit rides the Wan2.2-5B (TI2V) VAE geometry: z_dim=48, 16x spatial, 4x temporal.
LUCY_LATENT_CHANNELS = 48
LUCY_SPATIAL_RATIO = 16
LUCY_TEMPORAL_RATIO = 4

__all__ = ["WanDenoiseLoop", "latent_shape", "LUCY_LATENT_CHANNELS", "LUCY_SPATIAL_RATIO", "LUCY_TEMPORAL_RATIO"]
