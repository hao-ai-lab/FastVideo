# SPDX-License-Identifier: Apache-2.0
"""TrackWan action-recording demo — draw a motion, watch the model follow it.

Designed to answer one question: *does this checkpoint actually follow the point
tracks (action-following), and how does that change across checkpoints?*

Interaction (the part the old demo lacked):
  1. Pick a clip (gives the first frame + prompt) and a checkpoint.
  2. Move the mouse over the first frame and press **Space** -> records your cursor
     path for 5 seconds (121 frames). The green dot is where the action starts,
     the red dot where it ends.
  3. The 50x50 CoTracker training grid is used directly: the grid points within the
     action radius of your start are dragged along your recorded path, the rest stay
     static. This is grid-snapped / in-distribution (NOT an off-grid patch added on
     top of a static grid). Preview shows the full field (moving=green, static=gray).
  4. Generate -> original + generation side by side, with the full field overlaid,
     plus the moving-point CoTracker-EPE of how well the arm followed your path.

Checkpoints hot-swap (only the transformer weights reload, ~3 s) so you can flip
between e.g. step-2000 / 3000 / 4000 on the same clip and action.

Visibility = "dense field" keeps the whole 50x50 grid visible (the trained coverage;
recommended for aug-off checkpoints); "sparse" makes only the moved points visible.

Launch (single GPU on the cluster)::

    srun --jobid=<job> --overlap --ntasks=1 env CUDA_VISIBLE_DEVICES=1 \
      PYTHONPATH=$PWD .venv/bin/python \
      examples/inference/gradio/trackwan/app_action.py \
      --exports-root /.../control_now_exports --share
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import sys

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "data_pipeline"))
import synthetic_tracks as st  # noqa: E402
import trackwan_infer as twi  # noqa: E402

STATE: dict = {}


# ----------------------------------------------------------------------------- utils
def _img_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _first_frame_rgb(clip_idx: int) -> np.ndarray:
    s = STATE["samples"][clip_idx]
    return twi.decode_reference(STATE["model"], s["vae_latent"])[0]


def _original_video(clip_idx: int) -> str:
    """Decode + cache the GT clip as an mp4 (checkpoint-independent)."""
    cache = STATE.setdefault("orig_cache", {})
    if clip_idx in cache:
        return cache[clip_idx]
    s = STATE["samples"][clip_idx]
    frames = twi.decode_reference(STATE["model"], s["vae_latent"])  # [T,H,W,3]
    path = os.path.join(STATE["out_dir"], f"orig_clip{clip_idx}.mp4")
    imageio.mimsave(path, frames, fps=24, macro_block_size=1)
    cache[clip_idx] = path
    return path


def swap_checkpoint(name: str) -> str:
    """Load a checkpoint's transformer weights in place (fast; no text/vae reload)."""
    if STATE.get("cur_ckpt") == name:
        return f"checkpoint: {name} (loaded)"
    from safetensors.torch import load_file
    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # older torch
        from torch.distributed._tensor import DTensor
    path = STATE["exports"][name]
    sd = load_file(os.path.join(path, "transformer", "model.safetensors"))
    tgt = STATE["model"].transformer
    # Params are FSDP DTensors (even on 1 GPU); copy into each param's local shard
    # instead of load_state_dict (which errors mixing Tensor and DTensor).
    params = dict(tgt.named_parameters())
    n_ok = 0
    with torch.no_grad():
        for k, p in params.items():
            if k not in sd:
                continue
            loc = p.to_local() if isinstance(p, DTensor) else p.data
            src = sd[k].to(dtype=loc.dtype, device=loc.device)
            if tuple(loc.shape) == tuple(src.shape):
                loc.copy_(src)
                n_ok += 1
    tgt.eval()
    STATE["cur_ckpt"] = name
    warn = "" if n_ok == len(params) else f"  [WARN copied {n_ok}/{len(params)} params]"
    return f"checkpoint: {name} (swapped){warn}"


def _grid_action_tracks(traj, Tpx: int, radius: float, falloff: str, sparse: bool):
    """Snap the drawn action to the TRAINING 50x50 grid.

    The grid points within ``radius`` of the drawn start follow the recorded path;
    the rest stay static. This matches the CoTracker training format exactly (2500
    grid points, a local region moves) -- unlike an off-grid patch added on top of a
    static grid, which is out-of-distribution. Returns
    (tracks[Tpx,2500,2], vis[Tpx,2500], moving[2500]) or (None, None, None).
    """
    traj = np.asarray(traj, np.float32)
    if traj.ndim != 2 or len(traj) < 2:
        return None, None, None
    idx = np.linspace(0, len(traj) - 1, Tpx)
    path = np.stack([
        np.interp(idx, np.arange(len(traj)), traj[:, 0]),
        np.interp(idx, np.arange(len(traj)), traj[:, 1]),
    ], 1).astype(np.float32)  # [Tpx,2]
    g = st.make_grid(50)  # [2500,2] frame-0 grid (x,y) in [0,1]
    start = path[0]
    d = np.linalg.norm(g - start[None], axis=-1)  # [2500]
    if falloff == "smooth":
        w = np.clip(1.0 - d / max(radius, 1e-6), 0.0, 1.0)
        w = (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.float32)  # handle-like weighting
    else:  # hard
        w = (d <= radius).astype(np.float32)
    disp = path - start[None]  # [Tpx,2] displacement from start each frame
    tracks = (g[None] + w[None, :, None] * disp[:, None, :]).astype(np.float32)  # [Tpx,2500,2]
    tracks = np.clip(tracks, 0.0, 1.0)
    moving = w > 1e-3
    if sparse:  # only the moved handle is visible (rest occluded)
        vis = np.tile(moving[None].astype(np.float32), (Tpx, 1))
    else:  # dense field: whole grid visible (a few static + the moving region)
        vis = np.ones(tracks.shape[:2], np.float32)
    return tracks, vis, moving


def _overlay_grid(frames: np.ndarray, tracks: np.ndarray, vis: np.ndarray, moving, stride: int = 3) -> np.ndarray:
    """Draw the WHOLE (subsampled) 50x50 field: static background points (gray dots) +
    the moving region (green, with motion tails) -- so what's fed is what you see."""
    from fastvideo.train.callbacks.track_validation import _draw_overlay, _subsample
    T, H, W, _ = frames.shape
    tt = min(T, tracks.shape[0])
    tr, vs = _subsample(tracks[:tt], vis[:tt], 50, stride)
    mv, _ = _subsample(np.broadcast_to(moving[None, :, None].astype(np.float32),
                                       (tt, moving.shape[0], 2)).copy(), vis[:tt], 50, stride)
    is_mv = mv[0, :, 0] > 0.5  # [Nsub]
    colors = np.where(is_mv[:, None], np.array([0, 255, 100], np.uint8),
                      np.array([120, 120, 120], np.uint8)).astype(np.uint8)
    trpx = tr.copy()
    trpx[..., 0] *= W
    trpx[..., 1] *= H
    return _draw_overlay(frames[:tt], trpx, vs, colors, 12, 2, 0.5)


# ----------------------------------------------------------------------------- handlers
def on_clip(clip_idx):
    clip_idx = int(clip_idx)
    img = _first_frame_rgb(clip_idx)
    cap = STATE["samples"][clip_idx]["caption"][:240]
    return cap, _img_b64(img), _original_video(clip_idx)


def on_ckpt(name):
    return swap_checkpoint(name)


def _build_tracks(traj_json, radius, falloff, sparse):
    """Parse the drawn trajectory -> (tracks[Tpx,2500,2], vis, moving[2500]) or (None, msg).

    Grid-snapped: moves the 50x50 grid points near the draw-start along the path,
    keeping the rest static (in-distribution with the CoTracker training tracks).
    """
    import json
    if not traj_json or traj_json.strip() in ("", "[]"):
        return None, "Record an action first: move the mouse over the frame and press Space (5 s)."
    try:
        traj = json.loads(traj_json)
    except Exception as e:  # noqa: BLE001
        return None, f"bad trajectory json: {e}"
    is_sparse = str(sparse).startswith("sparse")
    tracks, vis, moving = _grid_action_tracks(traj, STATE["Tpx"], float(radius), str(falloff), is_sparse)
    if tracks is None:
        return None, "trajectory too short — hold Space and move the mouse for the full 5 s."
    return (tracks, vis, moving), None


def _first_frame_cached(clip_idx):
    cache = STATE.setdefault("_ff_cache", {})
    if clip_idx not in cache:
        cache[clip_idx] = _first_frame_rgb(clip_idx)
    return cache[clip_idx]


def on_preview(clip_idx, traj_json, radius, falloff, sparse):
    """Show the full grid-snapped control overlaid on the (frozen) first frame -- BEFORE generating."""
    clip_idx = int(clip_idx)
    built, err = _build_tracks(traj_json, radius, falloff, sparse)
    if built is None:
        return None, err
    tracks, vis, moving = built
    ff = _first_frame_cached(clip_idx)  # [H,W,3]
    frames = np.repeat(ff[None], tracks.shape[0], axis=0)  # freeze frame 0, T copies
    out = _overlay_grid(frames, tracks, vis, moving)
    path = os.path.join(STATE["out_dir"], f"preview_clip{clip_idx}.mp4")
    imageio.mimsave(path, out, fps=24, macro_block_size=1)
    return path, (f"PREVIEW (snapped to the 50x50 training grid): {int(moving.sum())} points near your start "
                  f"move (green, with tails), the other {int((~moving).sum())} stay static (gray). "
                  f"This matches the training track format. Press Generate.")


def on_generate(clip_idx, ckpt_name, traj_json, radius, falloff, sparse, steps, seed):
    """Generator: streams per-step denoise progress to the log, yields the video at the end."""
    import time
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler, )
    clip_idx = int(clip_idx)
    built, err = _build_tracks(traj_json, radius, falloff, sparse)
    if built is None:
        yield gr.update(), f"[error] {err}", gr.update()
        return
    tracks, vis, moving = built
    steps = int(steps)
    t0 = time.time()

    yield gr.update(), f"[1/4] swapping to {ckpt_name} ...", gr.update()
    swap_checkpoint(ckpt_name)

    model = STATE["model"]
    device = model.device
    dtype = torch.bfloat16
    s = STATE["samples"][clip_idx]
    ff = s["first_frame_latent"].to(device, dtype)
    cond20 = model._build_i2v_cond_concat(ff)
    txt = s["text_embedding"].to(device, dtype)
    mask = s["text_attention_mask"].to(device, dtype)
    img = s["clip_feature"].to(device, dtype)
    tp = torch.from_numpy(tracks)[None].to(device, dtype)
    tv = torch.from_numpy(vis)[None].to(device, dtype)
    _, _, T, H, W = ff.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)
    sched = FlowMatchEulerDiscreteScheduler(shift=float(model.timestep_shift))
    sched.set_timesteps(steps, device=device)

    for i, t in enumerate(sched.timesteps):
        model_in = torch.cat([latents.to(dtype), cond20], dim=1)
        ts = t.reshape(1).to(device, dtype)
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype), \
                set_forward_context(current_timestep=ts, attn_metadata=None):
            v = model.transformer(hidden_states=model_in, encoder_hidden_states=txt,
                                  encoder_attention_mask=mask, timestep=ts,
                                  encoder_hidden_states_image=img, track_points=tp,
                                  track_visibility=tv, return_dict=False)
        latents = sched.step(v.float(), t, latents.float(), return_dict=False)[0]
        yield gr.update(), f"[2/4] denoising {i + 1}/{steps}  ({time.time() - t0:.1f}s)", gr.update()

    yield gr.update(), f"[3/4] decoding latents ...  ({time.time() - t0:.1f}s)", gr.update()
    frames = twi.decode_to_pixels(model, latents)
    out = _overlay_grid(frames, tracks, vis, moving)
    tag = ckpt_name.replace(" ", "")
    path = os.path.join(STATE["out_dir"], f"action_clip{clip_idx}_{tag}.mp4")
    imageio.mimsave(path, out, fps=24, macro_block_size=1)

    yield gr.update(), f"[4/4] computing CoTracker EPE ...  ({time.time() - t0:.1f}s)", gr.update()
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import compute_epe
    # EPE on the MOVING points only (did the arm actually follow your path)
    mv_idx = np.where(moving)[0]
    ppx = tracks[:, mv_idx].copy()
    ppx[..., 0] *= W
    ppx[..., 1] *= H
    res = compute_epe(frames, ppx, vis[:, mv_idx], STATE["ct"], device)
    epe = res["epe"]
    epe_msg = (f"moving-point EPE = {epe:.2f}px ({res['n_points']} moved pts) — "
               f"lower = the arm region followed your drawn path."
               if epe is not None else "EPE n/a (no points re-tracked)")
    yield path, f"[done] generated in {time.time() - t0:.1f}s", epe_msg


# ----------------------------------------------------------------------------- ui
def _canvas_js(Tpx: int) -> str:
    return ("() => {\n"
            "  const cvs = document.getElementById('cvs'); if(!cvs||cvs._init) return; cvs._init=true;\n"
            "  const ctx = cvs.getContext('2d');\n"
            f"  const st = {{img:new Image(), mouse:[0.5,0.5], rec:false, traj:[], N:{Tpx}}};\n"
            "  cvs.width=512; cvs.height=288;\n"
            "  function draw(){ ctx.clearRect(0,0,cvs.width,cvs.height);\n"
            "    if(st.img.complete && st.img.width) ctx.drawImage(st.img,0,0,cvs.width,cvs.height);\n"
            "    if(st.traj.length){ ctx.strokeStyle='#00ff66'; ctx.lineWidth=3; ctx.beginPath();\n"
            "      st.traj.forEach((p,i)=>{const x=p[0]*cvs.width,y=p[1]*cvs.height; i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();\n"
            "      const s=st.traj[0], e=st.traj[st.traj.length-1];\n"
            "      ctx.fillStyle='#00ff66'; ctx.beginPath(); ctx.arc(s[0]*cvs.width,s[1]*cvs.height,6,0,7); ctx.fill();\n"
            "      ctx.fillStyle='#ff3333'; ctx.beginPath(); ctx.arc(e[0]*cvs.width,e[1]*cvs.height,6,0,7); ctx.fill(); } }\n"
            "  cvs.addEventListener('mousemove', e=>{const r=cvs.getBoundingClientRect(); st.mouse=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];});\n"
            "  window._setbg=(b64)=>{ if(!b64) return; const im=new Image(); im.onload=()=>{ const ar=im.width/im.height; cvs.width=512; cvs.height=Math.round(512/ar); st.img=im; st.traj=[]; draw(); }; im.src=b64; };\n"
            "  window._clear=()=>{ st.traj=[]; const tb=document.querySelector('#traj_json textarea'); if(tb){tb.value=''; tb.dispatchEvent(new Event('input',{bubbles:true}));} draw(); };\n"
            "  window._startRec=()=>{ if(st.rec) return; st.rec=true; st.traj=[]; let n=0; const stt=document.getElementById('recstatus');\n"
            "    const iv=setInterval(()=>{ st.traj.push([st.mouse[0],st.mouse[1]]); n++; draw(); if(stt) stt.textContent='\\u25CF REC '+n+'/'+st.N;\n"
            "      if(n>=st.N){ clearInterval(iv); st.rec=false; if(stt) stt.textContent='recorded '+st.N+' frames'; \n"
            "        const tb=document.querySelector('#traj_json textarea'); tb.value=JSON.stringify(st.traj); tb.dispatchEvent(new Event('input',{bubbles:true})); } }, 1000/24); };\n"
            "  document.addEventListener('keydown', e=>{ if(e.code==='Space'){ e.preventDefault(); window._startRec(); } });\n"
            "  draw();\n"
            "}")


def build_ui(num_clips: int, ckpt_names: list[str], Tpx: int):
    canvas_html = ("<div style='user-select:none'>"
                   "<canvas id='cvs' style='border:1px solid #888;cursor:crosshair;max-width:100%'></canvas>"
                   "<div id='recstatus' style='font-weight:bold;color:#0a0;height:1.4em'></div>"
                   "<div style='font-size:0.85em;color:#888'>Move the mouse over the frame, press "
                   "<b>Space</b> to record your action for 5&nbsp;s. Green=start, red=end.</div></div>")
    with gr.Blocks(title="TrackWan — action recorder") as demo:
        gr.Markdown("## TrackWan — record an action, test action-following across checkpoints\n"
                    "Pick a clip + checkpoint, **press Space** and move the mouse over the frame to draw a "
                    "5&nbsp;second motion, then **Generate**. The 50x50 grid points near your start are dragged along your path (rest stay static).")
        with gr.Row():
            with gr.Column(scale=1):
                ckpt = gr.Dropdown(ckpt_names, value=ckpt_names[-1], label="Checkpoint (hot-swap)")
                clip = gr.Dropdown(list(range(num_clips)), value=0, label="Clip (first frame + prompt)")
                caption = gr.Textbox(label="Prompt", interactive=False, lines=2)
                gr.HTML(canvas_html)
                with gr.Row():
                    rec_btn = gr.Button("● Record (Space)")
                    clr_btn = gr.Button("Clear")
                with gr.Row():
                    radius = gr.Slider(0.02, 0.5, value=0.15, step=0.01,
                                       label="Action radius (grid pts within this move)")
                    falloff = gr.Radio(["hard", "smooth"], value="hard", label="Falloff")
                background = gr.Radio(["dense field (all visible)", "sparse (only moved visible)"],
                                      value="dense field (all visible)", label="Visibility")
                with gr.Row():
                    steps = gr.Slider(10, 60, value=30, step=5, label="Denoise steps")
                    seed = gr.Number(value=1000, label="Seed")
                with gr.Row():
                    preview_btn = gr.Button("Preview traces")
                    gen_btn = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status / generation log", interactive=False, lines=2)
            with gr.Column(scale=1):
                orig_vid = gr.Video(label="Original clip")
                gen_vid = gr.Video(label="Preview traces / Generated (your action overlaid)")
                epe_box = gr.Textbox(label="Action-following (moving-point CoTracker EPE)", interactive=False)

        ff_b64 = gr.Textbox(elem_id="ff_b64", visible=False)
        traj_json = gr.Textbox(elem_id="traj_json", visible=False)

        clip.change(on_clip, [clip], [caption, ff_b64, orig_vid])
        ckpt.change(on_ckpt, [ckpt], [status])
        ff_b64.change(None, [ff_b64], None, js="(b64)=>{ window._setbg(b64); }")
        rec_btn.click(None, None, None, js="()=>{ window._startRec(); }")
        clr_btn.click(None, None, None, js="()=>{ window._clear(); }")
        preview_btn.click(on_preview, [clip, traj_json, radius, falloff, background], [gen_vid, status])
        gen_btn.click(on_generate, [clip, ckpt, traj_json, radius, falloff, background, steps, seed],
                      [gen_vid, status, epe_box])
        demo.load(on_clip, [clip], [caption, ff_b64, orig_vid])
        demo.load(None, None, None, js=_canvas_js(Tpx))
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports-root", required=True,
                   help="dir containing export_* checkpoint dirs (each a dcp_to_diffusers export)")
    p.add_argument("--yaml", default="examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml")
    p.add_argument("--data", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/"
                   "wan22_a14b_720p_24fps/preprocessed_i2v_track_funinp_now/combined_parquet_dataset")
    p.add_argument("--num-clips", type=int, default=10)
    p.add_argument("--out-dir", default="/mnt/weka/home/hao.zhang/shao/data/motion_pipeline/gradio_action_out")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    roots = sorted(glob.glob(os.path.join(args.exports_root, "export_*")))
    if not roots:
        raise SystemExit(f"no export_* dirs under {args.exports_root}")
    exports = {f"step {os.path.basename(r).split('_')[-1]}": r for r in roots}
    names = list(exports.keys())

    model, tc = twi.load_trackwan(exports[names[-1]], args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    samples = twi.load_conditioning_from_parquet(args.data, list(range(args.num_clips)), text_len)
    num_lat_t = samples[0]["first_frame_latent"].shape[2]
    ratio = int(tc.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import load_cotracker
    STATE.update(model=model, tc=tc, samples=samples, Tpx=(num_lat_t - 1) * ratio + 1,
                 out_dir=args.out_dir, ct=load_cotracker(model.device), exports=exports,
                 cur_ckpt=names[-1])

    demo = build_ui(len(samples), names, STATE["Tpx"])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        allowed_paths=[args.out_dir])


if __name__ == "__main__":
    main()
