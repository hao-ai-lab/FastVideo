"""Adapter-serving program: control_signal → text_encode → diffusion_denoise(+adapters) → vae_decode."""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.models.common import text_encode_node_fn as _text_encode


def _control_signal(instance, slots, request, ctx) -> None:
    """A ControlNet's conditioning image (pose/depth/edge) from the request → a slot the loop reads."""
    img = request.image()
    slots["control_signal"] = None if img is None else np.asarray(img.pixels, dtype="float32")


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_adapter_program() -> Program:
    return Program(
        program_id="wan.adapters", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("control_signal", fn=_control_signal, writes=("control_signal",)),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise", loop_id="diffusion_denoise", output_slot="denoise_out",
                          reads=("text_embeds", "control_signal"), writes=("denoise_out",)),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out",), writes=("video",)),
        ],
        output_artifacts={"video": "video", "latents": "denoise_out"},
    ).validate()
