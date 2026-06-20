"""LTX-2 two-stage distilled program:

    text_encode → ltx2_base (8-step) → upsample(+noise mix) → ltx2_refine (3-step) → vae_decode

The two loop nodes bind the same ``transformer``; the upsample component sits between them and
mixes noise at σ₀=0.909375 before the refine stage (the repo's stage-2 noise injection).
"""
from __future__ import annotations

import numpy as np

from v2.core.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode
from v2.recipes.ltx2.loop import REFINE_SIGMAS


def _upsample(instance, slots, request, ctx) -> None:
    base = np.asarray(slots["ltx_base_out"]["latents"], dtype="float32")
    # Learned 2× spatial latent super-resolution between the base and refine stages. The
    # ``spatial_upsampler`` component is the real LTX2LatentUpsampler on the GPU backend (un_normalize →
    # learned upsample → normalize, via the VAE's per-channel stats) and a nearest-neighbor toy on CPU —
    # same call on both backends, no device branch here.
    up = np.asarray(instance.component("spatial_upsampler").upsample(base), dtype="float32")
    seed = (request.diffusion.seed if request.diffusion.seed is not None else 0) + 7
    noise = np.random.default_rng(seed).standard_normal(up.shape).astype("float32")
    sigma0 = float(REFINE_SIGMAS[0])  # 0.909375
    slots["ltx_upsampled"] = (noise * sigma0 + up * (1.0 - sigma0)).astype("float32")


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["ltx_refine_out"]["latents"])


def build_ltx2_program() -> Program:
    return Program(
        program_id="ltx2.t2v.2stage",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("base",
                          loop_id="ltx2_base",
                          output_slot="ltx_base_out",
                          reads=("text_embeds", ),
                          writes=("ltx_base_out", )),
            ComponentNode("upsample", fn=_upsample, reads=("ltx_base_out", ), writes=("ltx_upsampled", )),
            ModelLoopNode("refine",
                          loop_id="ltx2_refine",
                          output_slot="ltx_refine_out",
                          reads=("ltx_upsampled", ),
                          writes=("ltx_refine_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("ltx_refine_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "ltx_refine_out"
        },
    ).validate()


# --- single-stage LTX-2 base model (no distill, no upsampler) ---------------------- #
def _vae_decode_single(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["ltx_out"]["latents"])


def build_ltx2_base_program() -> Program:
    """Single-stage LTX-2 base: text_encode → denoise (request-driven many-step, full-res) → vae_decode.
    No upsample/refine (those are the distilled two-stage program above)."""
    return Program(
        program_id="ltx2.t2v.base",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="ltx2_single",
                          output_slot="ltx_out",
                          reads=("text_embeds", ),
                          writes=("ltx_out", )),
            ComponentNode("vae_decode", fn=_vae_decode_single, reads=("ltx_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "ltx_out"
        },
    ).validate()


# --- single-stage LTX-2.3 joint A/V (T2VS) ----------------------------------------- #
def _text_encode_av(instance, slots, request, ctx) -> None:
    """LTX-2.3 connector emits SEPARATE video + audio text projections (one Gemma call)."""
    video_text, audio_text = instance.component("text_encoder").encode_av(request.prompt())
    slots["text_embeds"] = video_text
    slots["audio_text_embeds"] = audio_text


def _audio_decode_single(instance, slots, request, ctx) -> None:
    out = slots.get("ltx_out", {})
    au = out.get("audio_latents") if isinstance(out, dict) else None
    slots["audio"] = instance.component("audio_vae").decode(au) if au is not None else None


def build_ltx2_3_program() -> Program:
    """LTX-2.3 single-stage T2VS: text_encode (separate video+audio connectors) → JOINT A/V denoise
    (one DiT forward per step cross-attends video<->audio) → video decode + audio decode
    (AudioDecoder→Vocoder). A plain T2V request leaves the audio latent unproduced, so audio_decode
    writes None and the audio components stay unbuilt."""
    return Program(
        program_id="ltx2.3.t2vs",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode_av, writes=("text_embeds", "audio_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="ltx2_3",
                          output_slot="ltx_out",
                          reads=("text_embeds", "audio_text_embeds"),
                          writes=("ltx_out", )),
            ComponentNode("vae_decode", fn=_vae_decode_single, reads=("ltx_out", ), writes=("video", )),
            ComponentNode("audio_decode", fn=_audio_decode_single, reads=("ltx_out", ), writes=("audio", )),
        ],
        output_artifacts={
            "video": "video",
            "audio": "audio",
            "latents": "ltx_out"
        },
    ).validate()


# --- joint audio+video (T2VS) program ---------------------------------------------- #
def _upsample_av(instance, slots, request, ctx) -> None:
    _upsample(instance, slots, request, ctx)  # video latent upsample (reuse)
    base = slots.get("ltx_base_out", {})
    slots["ltx_audio"] = base.get("audio_latents") if isinstance(base, dict) else None  # thread audio


def _audio_decode(instance, slots, request, ctx) -> None:
    ro = slots.get("ltx_refine_out", {})
    au = ro.get("audio_latents") if isinstance(ro, dict) else None
    slots["audio"] = instance.component("audio_vae").decode(au) if au is not None else None


def build_ltx2_av_program() -> Program:
    """T2VS: the two-stage denoise carries a synchronized audio latent through both stages; the video
    VAE and the audio VAE decode the two modalities → video + audio artifacts (per-modality guidance)."""
    return Program(
        program_id="ltx2.t2vs.2stage",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("base",
                          loop_id="ltx2_base",
                          output_slot="ltx_base_out",
                          reads=("text_embeds", ),
                          writes=("ltx_base_out", )),
            ComponentNode("upsample", fn=_upsample_av, reads=("ltx_base_out", ), writes=("ltx_upsampled", "ltx_audio")),
            ModelLoopNode("refine",
                          loop_id="ltx2_refine",
                          output_slot="ltx_refine_out",
                          reads=("ltx_upsampled", "ltx_audio"),
                          writes=("ltx_refine_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("ltx_refine_out", ), writes=("video", )),
            ComponentNode("audio_decode", fn=_audio_decode, reads=("ltx_refine_out", ), writes=("audio", )),
        ],
        output_artifacts={
            "video": "video",
            "audio": "audio",
            "latents": "ltx_refine_out"
        },
    ).validate()
