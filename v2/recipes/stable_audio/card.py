"""Stable Audio Open ModelCard(s) — text→audio (the registered preset). AUDIO modality.

Two variants from the converted Diffusers repos:
  * ``FastVideo/stable-audio-open-1.0-Diffusers``     — depth 24, embed_dim 1536, qk_norm None, 3 cond
    ids (prompt + seconds_start + seconds_total), 2,097,152-frame window (~47.55s @ 44.1kHz).
  * ``FastVideo/stable-audio-open-small-Diffusers``   — depth 16, embed_dim 1024, qk_norm 'ln', 2 cond
    ids (prompt + seconds_total), 524,288-frame window (~11.89s). (depth/dim/qk_norm are resolved by the
    loader from the checkpoint config; the card carries the per-variant sample_size + audio defaults.)

Architecture deltas vs the video recipes (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.stable_audio:StableAudioDiT`` — a v-PREDICTION network (EDM-v / VDenoiser),
    NOT flow-match and NOT EDM-Karras. The adapter (``ComponentSpec.adapter`` -> ``StableAudioDiT``) packs
    the conditioner payload into (cross_attn_cond, global_embed), passes the RAW continuous sigma as the
    timestep, and returns the raw v output; ``StableAudioDenoiseLoop`` does the VDenoiser v->x0 + x0-space
    CFG + the polyexponential schedule (sigma_min 0.3, sigma_max 500, rho 1) and a DPM++ multistep step.
  * VAE  ``fastvideo.models.vaes.oobleck:OobleckVAE`` — 1-D conv audio VAE (hop_length 2048). The
    ``OobleckVAE`` adapter is RAW latent space (NO mean/std normalization, unlike WanVAE) and exposes
    ``sampling_rate`` for the decode-slice.
  * Conditioner ``fastvideo.models.encoders.stable_audio_conditioner:StableAudioMultiConditioner`` — T5-base
    (max_length 128) + duration NumberConditioners. Declared with ``kind="text_encoder"`` (kind reuse —
    the shared ``_MAKERS`` has no ``conditioner`` kind and editing it is out of scope) + the explicit
    ``StableAudioConditioner`` adapter. BRINGUP: the text-encoder maker calls ``TextEncoderLoader`` (a
    generic T5), so wiring the real SA conditioner on a GPU box needs the SA ``ConditionerLoader`` path.

``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder subfolder layout — the
conditioner lives in the ``text_encoder`` subfolder of the converted repo). The negative prompt is a LOCAL
module constant (Stable Audio's default negative prompt is empty).
"""
from __future__ import annotations

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    CostModel,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.loop.policies import ClassicCFG, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyTextEncoder, _seed_from
from v2.recipes.stable_audio.loop import StableAudioDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Stable Audio's default negative prompt is empty (no shared _prompts entry; kept LOCAL per the port spec).
STABLE_AUDIO_NEG = ""

# Oobleck VAE io_channels — the 1-D audio latent's channel dim (shared by the toy DiT + toy VAE below).
SA_IO_CHANNELS = 64

_SA_DIT = "v2.platform.backends.torch_stable_audio:StableAudioDiT"
_SA_VAE = "v2.platform.backends.torch_stable_audio:OobleckVAE"
_SA_CONDITIONER = "v2.platform.backends.torch_stable_audio:StableAudioConditioner"


def build_stable_audio_card(model_id: str = "stable-audio-open-1.0",
                            *,
                            checkpoint_root: str | None = None,
                            sigma_min: float = 0.3,
                            sigma_max: float = 500.0,
                            rho: float = 1.0,
                            sample_size: int = 2097152,
                            sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # fp16 throughout on the GPU path (DiT/VAE/T5); the loop math is numpy fp32 (lowest-risk bring-up).
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return StableAudioDenoiseLoop(loop_id="stable_audio_denoise",
                                      cfg=cfg,
                                      precision=precision,
                                      expert=expert,
                                      cost=cost,
                                      sigma_min=sigma_min,
                                      sigma_max=sigma_max,
                                      rho=rho,
                                      sample_size=sample_size)

    components = {
        # The SA multi-conditioner, declared under the ``text_encoder`` kind (kind reuse). On CPU this is
        # the ToyTextEncoder; on GPU the StableAudioConditioner adapter (BRINGUP: needs the SA loader).
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.stable_audio_conditioner:StableAudioMultiConditioner",
                      adapter=_SA_CONDITIONER,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2a"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.oobleck:OobleckVAE",
                      adapter=_SA_VAE,
                      factory=lambda inst: _ToyOobleckVAE(),
                      required_for={"t2a"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.stable_audio:StableAudioDiT",
                      adapter=_SA_DIT,
                      factory=lambda inst: _ToyAudioDiT(seed=seed),
                      resident_for=["stable_audio_denoise"],
                      required_for={"t2a"}),
    }
    loops = {
        "stable_audio_denoise":
        LoopSpec(
            loop_id="stable_audio_denoise",
            kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
            step_cost_model=cost,
            shared_weight_components=["transformer"],
            cache_policy=["feature"],
            graph_capture="eager",  # v->x0 + x0-space CFG + multistep host state -> eager
            loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="stable_audio",
        components=components,
        loops=loops,
        # No TEXT_TO_AUDIO capability on the v2 enum (audio output surface is the request-API extension —
        # BRINGUP); TEXT_TO_SPEECH is the nearest audio-output capability. VAE_DECODE + POLICY_ROLLOUT as usual.
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_SPEECH, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base",
                          assumes_loop="stable_audio_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        # Audio-shaped defaults. height/width/num_frames/fps are video-shaped and unused for audio (BRINGUP:
        # SamplingDefaults has no audio fields — the duration knobs ride node_params; see program.py).
        # num_steps=100, guidance_scale=7.0 are the published SA-1.0 defaults.
        sampling_defaults=sampling_defaults
        or SamplingDefaults(num_steps=100, guidance_scale=7.0, negative_prompt=STABLE_AUDIO_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_stable_audio_small_card(model_id: str = "stable-audio-open-small",
                                  *,
                                  checkpoint_root: str | None = None) -> ModelCard:
    """stable-audio-open-small: a 16-layer/1024-dim DiT with qk_norm='ln' and a shorter 524,288-frame
    window (~11.89s); 2 cond ids (prompt + seconds_total). Shares the Oobleck VAE + the v-prediction
    dpmpp-3m-sde loop with SA-1.0 — only the sample_size + the (loader-resolved) DiT geometry differ. The
    published small-variant default is a shorter clip (audio_end_in_s 6s) at fewer steps."""
    return build_stable_audio_card(model_id=model_id,
                                   checkpoint_root=checkpoint_root,
                                   sample_size=524288,
                                   sampling_defaults=SamplingDefaults(num_steps=8,
                                                                      guidance_scale=7.0,
                                                                      negative_prompt=STABLE_AUDIO_NEG))


class _ToyAudioDiT:
    """CPU toy v-prediction stand-in for ``StableAudioDiT`` (1-D audio latent [64, L]).

    The shared ``ToyDiT`` is channel-fixed (4 channels) and shaped for video; SA's latent is 64-channel
    1-D. This tiny deterministic toy honors the same contract the loop needs — ``__call__(latent,
    text_embed, sigma) -> v`` with the loop-supplied VDenoiser-scaled input — so the loop runs end-to-end
    on CPU. ``text_embed`` is the packed conditioning payload (the toy just means-pools it as a scalar
    conditioning signal, matching ToyDiT's text handling). Deterministic given (seed, inputs)."""

    def __init__(self, seed: int = 0):
        import numpy as np
        rng = np.random.default_rng(seed)
        self.w_t = float(rng.standard_normal() * 0.1)
        self.s_text = 0.5

    def __call__(self, latent, text_embed, sigma, context=None, *, cond=None):
        import numpy as np
        lat = np.asarray(latent, dtype="float32")
        cond_signal = float(np.mean(text_embed)) if text_embed is not None else 0.0
        # A simple bounded v-prediction: channel-local nonlinearity + timestep + conditioning bias.
        pre = 0.5 * lat + self.w_t * float(sigma) + self.s_text * cond_signal
        return np.tanh(pre).astype("float32")


class _ToyOobleckVAE:
    """CPU toy 1-D audio VAE stand-in for ``OobleckVAE`` (the shared ``ToyVAE`` is a video VAE that decodes
    a 4-channel ``[C,T,H,W]`` latent). Decodes the 64-channel ``[64, L]`` audio latent to a
    ``[channels, samples]`` waveform (channel-collapse projection + hop_length upsample), deterministic so
    the program node is interleave-safe. Exposes ``hop_length`` / ``sampling_rate`` like the real adapter."""

    def __init__(self, channels: int = 2, hop_length: int = 16, sampling_rate: int = 44100, seed: int = 4):
        import numpy as np
        self.hop_length = int(hop_length)  # tiny toy hop (real Oobleck is 2048); just for upsample shape
        self.sampling_rate = int(sampling_rate)
        self.channels = int(channels)
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal((channels, SA_IO_CHANNELS)) * 0.2).astype("float32")

    def decode(self, latent):
        import numpy as np
        z = np.asarray(latent, dtype="float32")  # [64, L]
        per_ch = np.tensordot(self.proj, z, axes=([1], [0]))  # [channels, L]
        wav = np.repeat(per_ch, self.hop_length, axis=-1)  # [channels, L*hop]
        return np.tanh(wav).astype("float32")

    def encode(self, audio):
        import numpy as np
        a = np.asarray(audio, dtype="float32")  # [channels, samples]
        flat = a.reshape(a.shape[0], -1)
        # crude inverse: project back to 64 channels at the downsampled rate (passthrough latent space)
        z = np.tensordot(self.proj.T, flat[:, ::self.hop_length], axes=([1], [0]))  # [64, ~L]
        return z.astype("float32")
