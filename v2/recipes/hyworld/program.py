"""HY-WorldPlay t2v program: text_encode -> hyworld_denoise (chunk rollout) -> vae_decode.

Mirrors the cosmos2/wan inline node graph. The world-model specifics (3-stream conditioning, per-frame
camera/action, camera-aligned memory retrieval) live entirely in ``HYWorldDenoiseLoop`` + the
``HYWorldDiT`` adapter, so the registered (CPU-verified) degenerate t2v path needs only the standard
three nodes.

BRINGUP hooks (left as no-op slot writers + documented, not wired, because v2's request API has no
pose-string / first-frame-image fields yet):
  * ``camera_ctx`` — a pose string (``"w-31"``) expanded via ``pose.pose_to_input`` + ``compute_latent_num``
    to ``viewmats[T,4,4] / Ks[T,3,3] / action[T]`` plus a 50k-point sphere point cloud
    (``generate_points_in_sphere``). A future ``_camera_encode`` node would write it into ``slots``; the
    loop already reads ``ctx.slots.get("camera_ctx")``.
  * ``i2v_img_embeds`` / ``i2v_cond`` — SigLIP image embeds + the VAE-encoded first-frame ``[cond|mask]``
    (33ch) latent. A future ``_image_encode`` node (SigLIP + VAE-encode of the first frame) would write
    them; the loop already reads them. Absent -> the DiT sees zero image -> masks the image stream (t2v).
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_hyworld_program() -> Program:
    return Program(
        program_id="hyworld.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="hyworld_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
