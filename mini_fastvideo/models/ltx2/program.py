"""LTX-2 two-stage distilled program (design_v3 §13, §15f):

    text_encode → ltx2_base (8-step) → upsample(+noise mix) → ltx2_refine (3-step) → vae_decode

The two loop nodes bind the same ``transformer``; the upsample component sits between them and
mixes noise at σ₀=0.909375 before the refine stage (the repo's stage-2 noise injection).
"""
from __future__ import annotations

import numpy as np

from ...program import ComponentNode, ModelLoopNode, Program, ProgramKind
from ..common import text_encode_node_fn as _text_encode
from .loop import REFINE_SIGMAS


def _upsample(instance, slots, request, ctx) -> None:
    base = np.asarray(slots["ltx_base_out"]["latents"], dtype="float32")
    up = np.repeat(np.repeat(base, 2, axis=2), 2, axis=3)          # 2× spatial (learned upsampler stand-in)
    seed = (request.diffusion.seed if request.diffusion.seed is not None else 0) + 7
    noise = np.random.default_rng(seed).standard_normal(up.shape).astype("float32")
    sigma0 = float(REFINE_SIGMAS[0])                                # 0.909375
    slots["ltx_upsampled"] = (noise * sigma0 + up * (1.0 - sigma0)).astype("float32")


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["ltx_refine_out"]["latents"])


def build_ltx2_program() -> Program:
    return Program(
        program_id="ltx2.t2v.2stage", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("base", loop_id="ltx2_base", output_slot="ltx_base_out",
                          reads=("text_embeds",), writes=("ltx_base_out",)),
            ComponentNode("upsample", fn=_upsample, reads=("ltx_base_out",), writes=("ltx_upsampled",)),
            ModelLoopNode("refine", loop_id="ltx2_refine", output_slot="ltx_refine_out",
                          reads=("ltx_upsampled",), writes=("ltx_refine_out",)),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("ltx_refine_out",), writes=("video",)),
        ],
        output_artifacts={"video": "video", "latents": "ltx_refine_out"},
    ).validate()
