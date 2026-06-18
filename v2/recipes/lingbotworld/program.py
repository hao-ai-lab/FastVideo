"""LingBot-World i2v program: image_encode (CLIP) -> cond_encode (first-frame VAE [mask|cond]) ->
camera_encode (Plucker) -> text_encode -> i2v_denoise (dual-guidance MoE) -> vae_decode.

Same inline shape as the Wan i2v program (the i2v conditioning is identical to Wan2.2-I2V-A14B), with one
NEW node: ``camera_encode`` builds the camera/Plucker tensor ``c2ws_plucker_emb`` the LingBot-World DiT
FiLM-injects per block. The ``LingBotWorldDenoiseLoop`` reads that slot and publishes it onto the active
expert adapter; the dual-guidance / MoE specifics live in the loop + adapter, so the node graph is a thin
superset of Wan i2v.

BRINGUP (blocker #2 — needs a request-API extension): v2's ``Request`` has no camera-trajectory input
slot. The camera node therefore looks for an ``action_path`` in the node override
(``request.node_override("camera_encode")["action_path"]``, pointing at a dir with ``poses.npy`` +
``intrinsics.npy``) and, if present, builds the Plucker tensor via the fastvideo
``prepare_camera_embedding`` (``spatial_scale=8`` to match the VAE 8x). Absent (the default, and the CPU
toy path) -> it writes ``None`` and the loop runs the degenerate no-camera i2v step. ``prepare_camera_embedding``
also re-clamps ``num_frames`` to the trajectory length (4k+1); the v2 frame count is taken from the
request, so on GPU the camera node and the request num_frames must agree (verify at bring-up).
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode

_WAN_TEMPORAL_RATIO = 4  # VAE temporal compression -> the i2v mask has this many channels
_SPATIAL_SCALE = 8  # Plucker spatial downsample == VAE 8x spatial compression (WanCamCtrl in_chans=6*64)


def _i2v_image_encode(instance, slots, request, ctx) -> None:
    # LingBot-World-Base-Cam has NO CLIP image encoder (image_dim=null) — the first-frame conditioning is
    # carried entirely by the 36ch ``[noise|mask+cond]`` latent concat. So there is no image-embed branch;
    # the slot is always None (the DiT forward and the adapter accept ``context=None`` / ``img=None``).
    slots["i2v_img_embeds"] = None


def _i2v_cond_encode(instance, slots, request, ctx) -> None:
    """First-frame VAE conditioning + mask (mirrors fastvideo's ImageVAEEncodingStage): encode the
    conditioning image as frame 0 (rest zeros), prepend a 4-channel mask (first latent frame = 1).

    This model's DiT has ``in_channels=36`` (16 noise + 4 mask + 16 cond latent), so the i2v cond slot is
    REQUIRED for a well-formed forward — a None cond would feed only 16ch into a 36ch patch embedding. The
    v2 convenience API (``generate_video(prompt=...)``) supplies no image, so for the bring-up we synthesize
    a BLANK first frame (zeros): the latent concat shape is correct and the run produces a finite output
    (degenerate no-image i2v, the goal being loads+runs+finite, not quality). A real conditioning image
    (when ``request.image()`` is present) takes the normal path."""
    img = request.image()
    nf = int(request.diffusion.num_frames)
    h_px, w_px = int(request.diffusion.height), int(request.diffusion.width)
    if img is not None and getattr(img, "pixels", None) is not None:
        px = np.asarray(img.pixels, dtype="float32")  # [3, H, W] in [-1, 1]
        cond_video = np.zeros((px.shape[0], nf) + px.shape[1:], dtype="float32")
        cond_video[:, 0] = px  # frame 0 = conditioning image, rest zeros
    else:
        # BRINGUP no-image fallback: a blank (zeros) conditioning video so the 36ch concat is well-formed.
        cond_video = np.zeros((3, nf, h_px, w_px), dtype="float32")
    cond_latent = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")  # [C, T, h, w]
    t, h, w = cond_latent.shape[1:]
    mask = np.zeros((_WAN_TEMPORAL_RATIO, t, h, w), dtype="float32")
    mask[:, 0] = 1.0  # first latent frame is the known (conditional) frame
    slots["i2v_cond"] = np.concatenate([mask, cond_latent], axis=0)  # [mask | cond] -> DiT concats with noise


def _camera_encode(instance, slots, request, ctx) -> None:
    """Build the camera/Plucker tensor ``c2ws_plucker_emb`` from a per-request camera trajectory.

    BRINGUP: until ``Request`` carries a camera-trajectory input, the trajectory dir is read from the
    node override ``action_path``. Absent -> ``None`` (the no-camera path the CPU toy backend takes)."""
    action_path = request.node_override("camera_encode").get("action_path")
    if not action_path:
        slots["c2ws_plucker_emb"] = None
        return
    from v2.models.dits.lingbotworld import prepare_camera_embedding
    d = request.diffusion
    emb, _num_frames = prepare_camera_embedding(action_path,
                                                int(d.num_frames),
                                                int(d.height),
                                                int(d.width),
                                                spatial_scale=_SPATIAL_SCALE)
    # Drop the leading batch dim -> the loop/adapter add it back ([6*s^2, F_lat, H_lat, W_lat]).
    arr = np.asarray(emb.squeeze(0).cpu().numpy() if hasattr(emb, "cpu") else emb, dtype="float32")
    slots["c2ws_plucker_emb"] = arr


def _i2v_vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["i2v_out"]["latents"])


def build_lingbotworld_program() -> Program:
    return Program(
        program_id="lingbotworld.i2v",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("image_encode", fn=_i2v_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("cond_encode", fn=_i2v_cond_encode, writes=("i2v_cond", )),
            ComponentNode("camera_encode", fn=_camera_encode, writes=("c2ws_plucker_emb", )),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="i2v_denoise",
                          output_slot="i2v_out",
                          reads=("text_embeds", "i2v_cond", "i2v_img_embeds", "c2ws_plucker_emb"),
                          writes=("i2v_out", )),
            ComponentNode("vae_decode", fn=_i2v_vae_decode, reads=("i2v_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "i2v_out"
        },
    ).validate()
