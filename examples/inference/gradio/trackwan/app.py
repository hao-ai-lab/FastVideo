# SPDX-License-Identifier: Apache-2.0
"""Interactive track-control demo for TrackWan (MotionStream-style).

Pick a preprocessed clip (gives the first frame + text + the option of its own
tracks), author a motion control, preview the tracks over the first frame, then
generate and watch whether the model follows them.

Control modes (mirroring MotionStream):
  - Trajectory / drag : click the first frame to set a drag handle, choose a
    direction + radius; optionally make it SPARSE (only the handle points active).
  - Camera-like       : pan / zoom / rotate / swirl presets.
  - Motion transfer   : reuse another clip's extracted tracks.
  - GT                : the clip's own tracks (reconstruction sanity check).

Launch (single GPU, on the cluster)::

    srun --jobid=<job> --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=1 \
      HF_HUB_CACHE=/.../hf_cache_clean PYTHONPATH=$PWD \
      .venv/bin/python examples/inference/gradio/trackwan/app.py \
      --export /.../trackwan_1.3b_overfit4k --port 7860

Then SSH-forward the port and open http://localhost:7860 .

NOTE: sparse drag control needs a model trained with point-subsampling aug
(WANTRACK_AUG=1). A clean-overfit (aug-off) checkpoint mostly ignores sparse
controls -- use dense presets to test it.
"""
from __future__ import annotations

import argparse
import os
import sys

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "data_pipeline"))
import synthetic_tracks as st  # noqa: E402
import trackwan_infer as twi  # noqa: E402

STATE: dict = {}  # model, tc, samples, Tpx, H, W, out_dir


def _overlay(frames_thwc, tracks_norm, vis, stride=3):
    from fastvideo.train.callbacks.track_validation import (_draw_overlay, _grid_colors, _subsample)
    T, H, W, _ = frames_thwc.shape
    tt = min(T, tracks_norm.shape[0])
    fr, tr, vs = frames_thwc[:tt], tracks_norm[:tt], vis[:tt]
    grid = int(round(tr.shape[1] ** 0.5))
    tr, vs = _subsample(tr, vs, grid, stride)
    colors = _grid_colors(grid, stride)
    if colors.shape[0] != tr.shape[1]:
        colors = _grid_colors(int(round(tr.shape[1] ** 0.5)) or 1, 1)[:tr.shape[1]]
    trpx = tr.copy()
    trpx[..., 0] *= W
    trpx[..., 1] *= H
    return _draw_overlay(fr, trpx, vs, colors, 12, 2, 0.5)


def _first_frame_rgb(clip_idx: int) -> np.ndarray:
    """Decode the clip's GT first frame for display/clicking."""
    s = STATE["samples"][clip_idx]
    ref = twi.decode_reference(STATE["model"], s["vae_latent"])  # [T,H,W,3]
    return ref[0]


def _build_tracks(clip_idx, mode, preset_name, strength, cx, cy, radius,
                  sparsity, drag_dx, drag_dy, transfer_idx):
    """Return (tracks_norm [T,N,2], vis [T,N]) or (None, None) for 'none'."""
    Tpx = STATE["Tpx"]
    g = st.make_grid(50)
    if mode == "gt":
        s = STATE["samples"][clip_idx]
        return s["track_points"][0].numpy()[:Tpx], s["track_visibility"][0].numpy()[:Tpx]
    if mode == "none":
        return None, None
    if mode == "transfer":
        s = STATE["samples"][int(transfer_idx)]
        return s["track_points"][0].numpy()[:Tpx], s["track_visibility"][0].numpy()[:Tpx]
    if mode == "preset":
        tr, vs = st.preset(preset_name, Tpx, 50, strength=float(strength))
    elif mode == "drag":
        tr, vs = st.drag(g, Tpx, center=(float(cx), float(cy)), dx=float(drag_dx),
                         dy=float(drag_dy), radius=float(radius))
    else:
        tr, vs = st.static(g, Tpx)
    # sparsity
    if sparsity == "radius (handle only)":
        vs = st.select_radius(vs, g, center=(float(cx), float(cy)), radius=float(radius))
    elif sparsity == "coarse grid":
        vs = st.select_stride(vs, 50, 4)
    return tr, vs


def on_select_clip(clip_idx):
    img = _first_frame_rgb(int(clip_idx))
    cap = STATE["samples"][int(clip_idx)]["caption"][:200]
    return img, cap


def on_image_click(clip_idx, evt: gr.SelectData):
    """Set drag center (normalized) from a click on the first frame."""
    img = _first_frame_rgb(int(clip_idx))
    H, W, _ = img.shape
    x, y = evt.index[0] / W, evt.index[1] / H
    return round(float(x), 3), round(float(y), 3)


def on_preview(clip_idx, mode, preset_name, strength, cx, cy, radius, sparsity,
               drag_dx, drag_dy, transfer_idx):
    img = _first_frame_rgb(int(clip_idx))
    tr, vs = _build_tracks(int(clip_idx), mode, preset_name, strength, cx, cy, radius,
                           sparsity, drag_dx, drag_dy, transfer_idx)
    if tr is None:
        return img, "no tracks (mode=none)"
    frames = np.repeat(img[None], tr.shape[0], 0)
    ov = _overlay(frames, tr, vs)[0]
    active = int((vs[0] > 0.5).sum())
    return ov, f"{active} active points (of {vs.shape[1]})"


def on_generate(clip_idx, mode, preset_name, strength, cx, cy, radius, sparsity,
                drag_dx, drag_dy, transfer_idx, steps, seed):
    clip_idx = int(clip_idx)
    s = STATE["samples"][clip_idx]
    tr, vs = _build_tracks(clip_idx, mode, preset_name, strength, cx, cy, radius,
                           sparsity, drag_dx, drag_dy, transfer_idx)
    tp = torch.from_numpy(tr)[None].float() if tr is not None else None
    tv = torch.from_numpy(vs)[None].float() if vs is not None else None
    lat = twi.generate(STATE["model"], first_frame_latent=s["first_frame_latent"],
                       text_embedding=s["text_embedding"], text_attention_mask=s["text_attention_mask"],
                       track_points=tp, track_visibility=tv, num_steps=int(steps), seed=int(seed))
    frames = twi.decode_to_pixels(STATE["model"], lat)
    ov_tr = tr if tr is not None else np.zeros((frames.shape[0], 1, 2), np.float32)
    ov_vs = vs if vs is not None else np.zeros((frames.shape[0], 1), np.float32)
    out_frames = _overlay(frames, ov_tr, ov_vs)
    path = os.path.join(STATE["out_dir"], f"gen_clip{clip_idx}_{mode}_{preset_name}.mp4")
    imageio.mimsave(path, out_frames, fps=24, macro_block_size=1)

    epe_txt = "EPE: n/a (no control tracks)"
    if tr is not None:
        from fastvideo.eval.metrics.motion.cotracker_epe.metric import compute_epe
        H, W = frames.shape[1], frames.shape[2]
        trpx = tr.copy()
        trpx[..., 0] *= W
        trpx[..., 1] *= H
        res = compute_epe(frames, trpx, vs, STATE["ct"], STATE["model"].device)
        if res["epe"] is not None:
            epe_txt = f"EPE: {res['epe']:.2f} px  ({res['n_points']} pts)  — lower = follows control"
    return path, epe_txt


def build_ui(num_clips):
    with gr.Blocks(title="TrackWan — interactive motion control") as demo:
        gr.Markdown("## TrackWan — interactive point-track control\n"
                    "Pick a clip, author a motion control, **Preview tracks**, then **Generate**. "
                    "Click the first frame to set the drag handle center.")
        with gr.Row():
            with gr.Column(scale=1):
                clip = gr.Dropdown(list(range(num_clips)), value=0, label="Clip")
                caption = gr.Textbox(label="Prompt", interactive=False, lines=2)
                mode = gr.Radio(["preset", "drag", "transfer", "gt", "none"], value="preset", label="Control mode")
                preset_name = gr.Dropdown(st.PRESETS, value="pan_right", label="Preset (preset mode)")
                strength = gr.Slider(0.0, 0.6, value=0.25, step=0.01, label="Strength (pan/zoom)")
                with gr.Row():
                    cx = gr.Number(value=0.5, label="drag center x (click frame)")
                    cy = gr.Number(value=0.5, label="drag center y")
                radius = gr.Slider(0.05, 0.6, value=0.2, step=0.01, label="Radius (drag/handle)")
                with gr.Row():
                    drag_dx = gr.Slider(-0.5, 0.5, value=0.3, step=0.01, label="drag dx")
                    drag_dy = gr.Slider(-0.5, 0.5, value=0.0, step=0.01, label="drag dy")
                sparsity = gr.Radio(["full", "radius (handle only)", "coarse grid"], value="full",
                                    label="Sparsity (sparse needs aug-trained ckpt)")
                transfer_idx = gr.Dropdown(list(range(num_clips)), value=min(1, num_clips - 1),
                                           label="Transfer from clip (transfer mode)")
                with gr.Row():
                    steps = gr.Slider(10, 60, value=30, step=5, label="Denoise steps")
                    seed = gr.Number(value=1000, label="Seed")
                with gr.Row():
                    preview_btn = gr.Button("Preview tracks")
                    gen_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                frame_img = gr.Image(label="First frame (click to set drag center) / track preview", type="numpy")
                status = gr.Textbox(label="Status", interactive=False)
                out_video = gr.Video(label="Generated (tracks overlaid)")
                epe_box = gr.Textbox(label="Motion fidelity", interactive=False)

        ctrl = [clip, mode, preset_name, strength, cx, cy, radius, sparsity, drag_dx, drag_dy, transfer_idx]
        clip.change(on_select_clip, [clip], [frame_img, caption])
        frame_img.select(on_image_click, [clip], [cx, cy])
        preview_btn.click(on_preview, ctrl, [frame_img, status])
        gen_btn.click(on_generate, ctrl + [steps, seed], [out_video, epe_box])
        demo.load(on_select_clip, [clip], [frame_img, caption])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, help="dcp_to_diffusers export dir")
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track/combined_parquet_dataset")
    p.add_argument("--num-clips", type=int, default=10)
    p.add_argument("--out-dir", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/gradio_out")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, tc = twi.load_trackwan(args.export, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(args.data, list(range(args.num_clips)), text_len)
    num_lat_t = samples[0]["first_frame_latent"].shape[2]
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import load_cotracker
    STATE.update(model=model, tc=tc, samples=samples, Tpx=(num_lat_t - 1) * ratio + 1,
                 out_dir=args.out_dir, ct=load_cotracker(model.device))

    demo = build_ui(len(samples))
    demo.queue().launch(server_name=args.host, server_port=args.port, allowed_paths=[args.out_dir])


if __name__ == "__main__":
    main()
