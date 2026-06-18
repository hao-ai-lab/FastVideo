"""HunyuanVideo15DenoiseLoop — the canonical flow-match denoise loop, parameterized for HunyuanVideo 1.5.

The loop *math* is identical to Wan's (``WanDenoiseLoop``): N deterministic flow-match Euler steps over
the full latent, ``ClassicCFG`` combine, ``timestep = sigma*1000``, ``x_next = x + (σ_next-σ)·v``. This is
faithful to ``fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py`` — ``scale_model_input``
is the identity for flow-match, ``step`` is ``prev_sample = sample + (sigma_next - sigma)*model_output``,
and HunyuanVideo 1.5's DiT predicts velocity directly (no x0 reconstruction). The latent is initialized at
``randn·σ[0]`` with ``σ[0]=1.0`` (the preset's ``linspace(1,0,n+1)[:-1]`` schedule).

The ONLY architectural delta lives in the ``HunyuanVideo15DiT`` torch adapter (two text embeds + an
image-embed list + the 33-channel i2v cond concat). Because ``WanDenoiseLoop`` already threads
``context=`` (image embeds) and ``cond=`` (the conditioning latent) into the adapter, the cleanest port
is to subclass it: change only the latent geometry (z=32, 16× spatial, 4× temporal) and the slot names
the loop reads for the i2v signal. For t2v those slots are absent (``None``) → the dit-call degenerates
to ``dit(latent, text_embed, sigma)`` exactly like Wan t2v, and the toy backend's ``ToyDiT`` runs it.

BRINGUP: i2v conditioning (``context``=zero image embeds, ``cond``=33ch first-frame latent) is threaded
but the program currently writes neither slot for the registered t2v preset, so the loop is pure t2v on
CPU; the i2v VAE-encode node + image-latent slot are documented hooks (see ``program.py``).
"""
from __future__ import annotations

from v2.recipes.wan21.loop import WanDenoiseLoop

# HunyuanVideo 1.5 causal-3D VAE compression (configs/models/vaes/hunyuan15vae.py): z=32, 16× spatial,
# 4× temporal. The DiT's noisy-latent in_channels is 32 (out_channels); i2v reaches 65 by concatenating
# the 33ch first-frame cond latent inside the adapter, so the loop's latent stays 32ch.
HUNYUAN15_LATENT_CHANNELS = 32
HUNYUAN15_SPATIAL_RATIO = 16
HUNYUAN15_TEMPORAL_RATIO = 4


class HunyuanVideo15DenoiseLoop(WanDenoiseLoop):
    """Wan flow-match loop with HunyuanVideo 1.5 latent geometry. The i2v conditioning the base loop reads
    from ``ctx.slots`` is keyed the same way (``i2v_cond``/``i2v_img_embeds``), so a future i2v program can
    write them with no loop change; the t2v program leaves them unset (pure t2v)."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = HUNYUAN15_LATENT_CHANNELS,
                 spatial_ratio: int = HUNYUAN15_SPATIAL_RATIO,
                 temporal_ratio: int = HUNYUAN15_TEMPORAL_RATIO):
        super().__init__(loop_id=loop_id,
                         cfg=cfg,
                         flow_shift=flow_shift,
                         precision=precision,
                         expert=expert,
                         cost=cost,
                         latent_channels=latent_channels,
                         spatial_ratio=spatial_ratio,
                         temporal_ratio=temporal_ratio)
