"""GEN3C program: text_encode → diffusion_denoise (EDM) → vae_decode.

Same inline shape as the Cosmos t2v program — the EDM specifics + the camera/3D-cache frame-replace
conditioning live entirely in ``Gen3CDenoiseLoop`` and the ``Gen3CDiT`` adapter, so the registered
(degenerate t2v) node graph is unchanged.

BRINGUP — the camera-conditioning pre-stage (the GEN3C innovation, no v2 equivalent yet): a node placed
BEFORE ``denoise`` would (porting ``Gen3CConditioningStage`` + ``Gen3CLatentPreparationStage``):
  1. load the input image; predict metric depth via MoGe (``Ruicheng/moge-vitl``);
  2. build a ``Cache3DBuffer`` point cloud; ``generate_camera_trajectory(trajectory_type,
     movement_distance, camera_rotation)``; ``cache.render_cache(...)`` -> warped frames + masks;
  3. VAE-encode the warps into ``condition_video_pose`` [frame_buffer_max·32, T_lat, H_lat, W_lat], and
     VAE-encode the input image into ``conditioning_latents`` (frame-0 anchor); build
     ``condition_video_input_mask`` / ``cond_indicator`` (frame-0 = 1).
The loop already READS those slots (``conditioning_latents`` / ``cond_indicator`` / ``condition_video_pose``
/ ``condition_video_input_mask``); writing them needs (a) a CUDA point-cloud rasterizer dependency and
(b) a request-API extension carrying ``image_path`` + the camera-trajectory fields. Until then the slots
are unset (None) and the loop runs pure t2v (the DiT's internal pose concat is zeroed).
"""
from __future__ import annotations

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_gen3c_program() -> Program:
    return Program(
        program_id="gen3c.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
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
