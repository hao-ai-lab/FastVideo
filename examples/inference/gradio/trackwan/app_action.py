# SPDX-License-Identifier: Apache-2.0
"""TrackWan action-recording demo — draw a motion, watch the model follow it.

Designed to answer one question: *does this checkpoint actually follow the point
tracks (action-following), and how does that change across checkpoints?*

Interaction (the part the old demo lacked):
  1. Pick a clip (gives the first frame + prompt) and a checkpoint.
  2. Move the mouse over the first frame and press **Space** -> records your cursor
     path for 5 seconds (121 frames). The green dot is where the action starts,
     the red dot where it ends.
  3. An ``N x N`` patch of CoTracker points is placed around the start and rigidly
     dragged along your recorded path. Tune the grid size and patch extent.
  4. Generate -> see the original clip and the generation side by side, with the
     controlled patch overlaid, plus the CoTracker-EPE of how well the patch
     followed your drawn action.

Checkpoints hot-swap (only the transformer weights reload, ~3 s) so you can flip
between e.g. step-2000 / 3000 / 4000 on the same clip and action.

Background = "static (dense)" fills the rest of the frame with held-still tracks
so the model gets the dense coverage it was trained on (recommended for aug-off
checkpoints); "none" sends only the sparse patch.

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


def _patch_tracks(traj: np.ndarray, Tpx: int, grid_n: int, extent: float):
    """N x N grid around the start point, rigidly translated along the recorded path."""
    traj = np.asarray(traj, np.float32)
    if traj.ndim != 2 or len(traj) < 2:
        return None, None
    idx = np.linspace(0, len(traj) - 1, Tpx)
    px = np.interp(idx, np.arange(len(traj)), traj[:, 0])
    py = np.interp(idx, np.arange(len(traj)), traj[:, 1])
    path = np.stack([px, py], 1)  # [Tpx,2]
    start = path[0]
    offs = np.linspace(-extent / 2, extent / 2, grid_n) if grid_n > 1 else np.array([0.0], np.float32)
    gx, gy = np.meshgrid(offs, offs)
    base = np.stack([start[0] + gx.ravel(), start[1] + gy.ravel()], 1)  # [N^2,2]
    disp = path - start  # [Tpx,2]
    pts = (base[None] + disp[:, None, :]).astype(np.float32)  # [Tpx,N^2,2]
    pts = np.clip(pts, 0.0, 1.0)
    vis = np.ones(pts.shape[:2], np.float32)
    return pts, vis


def _overlay_patch(frames: np.ndarray, patch_norm: np.ndarray, patch_vis: np.ndarray) -> np.ndarray:
    from fastvideo.train.callbacks.track_validation import _draw_overlay
    T, H, W, _ = frames.shape
    tt = min(T, patch_norm.shape[0])
    trpx = patch_norm[:tt].copy()
    trpx[..., 0] *= W
    trpx[..., 1] *= H
    colors = np.tile(np.array([[0, 255, 100]], np.uint8), (patch_norm.shape[1], 1))
    return _draw_overlay(frames[:tt], trpx, patch_vis[:tt], colors, 6, 2, 0.6)


# ----------------------------------------------------------------------------- handlers
def on_clip(clip_idx):
    clip_idx = int(clip_idx)
    img = _first_frame_rgb(clip_idx)
    cap = STATE["samples"][clip_idx]["caption"][:240]
    return cap, _img_b64(img), _original_video(clip_idx)


def on_ckpt(name):
    return swap_checkpoint(name)


def _build_tracks(traj_json, grid_n, extent, background):
    """Parse the drawn trajectory -> (patch, pvis, full_tr, full_vs) or (None, msg)."""
    import json
    if not traj_json or traj_json.strip() in ("", "[]"):
        return None, "Record an action first: move the mouse over the frame and press Space (5 s)."
    try:
        traj = json.loads(traj_json)
    except Exception as e:  # noqa: BLE001
        return None, f"bad trajectory json: {e}"
    Tpx = STATE["Tpx"]
    patch, pvis = _patch_tracks(traj, Tpx, int(grid_n), float(extent))
    if patch is None:
        return None, "trajectory too short — hold Space and move the mouse for the full 5 s."
    if background == "static (dense)":
        g = st.make_grid(50)
        bt, bv = st.static(g, Tpx)
        full_tr = np.concatenate([bt, patch], axis=1)
        full_vs = np.concatenate([bv, pvis], axis=1)
    else:
        full_tr, full_vs = patch, pvis
    return (patch, pvis, full_tr, full_vs), None


def _first_frame_cached(clip_idx):
    cache = STATE.setdefault("_ff_cache", {})
    if clip_idx not in cache:
        cache[clip_idx] = _first_frame_rgb(clip_idx)
    return cache[clip_idx]


def on_preview(clip_idx, traj_json, grid_n, extent, background):
    """Show the synthetic control tracks overlaid on the (frozen) first frame -- BEFORE generating."""
    clip_idx = int(clip_idx)
    built, err = _build_tracks(traj_json, grid_n, extent, background)
    if built is None:
        return None, err
    patch, pvis, full_tr, full_vs = built
    ff = _first_frame_cached(clip_idx)  # [H,W,3]
    frames = np.repeat(ff[None], patch.shape[0], axis=0)  # freeze frame 0, T copies
    out = _overlay_patch(frames, patch, pvis)
    path = os.path.join(STATE["out_dir"], f"preview_clip{clip_idx}.mp4")
    imageio.mimsave(path, out, fps=24, macro_block_size=1)
    nbg = (full_tr.shape[1] - patch.shape[1])
    return path, (f"PREVIEW of the traces to be fed (green = your {int(grid_n)}x{int(grid_n)} arm patch, "
                  f"{patch.shape[1]} pts + {nbg} static background pts). Press Generate when it looks right.")


def on_generate(clip_idx, ckpt_name, traj_json, grid_n, extent, background, steps, seed):
    """Generator: streams per-step denoise progress to the log, yields the video at the end."""
    import time
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler, )
    clip_idx = int(clip_idx)
    built, err = _build_tracks(traj_json, grid_n, extent, background)
    if built is None:
        yield gr.update(), f"[error] {err}", gr.update()
        return
    patch, pvis, full_tr, full_vs = built
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
    tp = torch.from_numpy(full_tr)[None].to(device, dtype)
    tv = torch.from_numpy(full_vs)[None].to(device, dtype)
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
    out = _overlay_patch(frames, patch, pvis)
    tag = ckpt_name.replace(" ", "")
    path = os.path.join(STATE["out_dir"], f"action_clip{clip_idx}_{tag}_g{int(grid_n)}.mp4")
    imageio.mimsave(path, out, fps=24, macro_block_size=1)

    yield gr.update(), f"[4/4] computing CoTracker EPE ...  ({time.time() - t0:.1f}s)", gr.update()
    from fastvideo.eval.metrics.motion.cotracker_epe.metric import compute_epe
    ppx = patch.copy()
    ppx[..., 0] *= W
    ppx[..., 1] *= H
    res = compute_epe(frames, ppx, pvis, STATE["ct"], device)
    epe = res["epe"]
    epe_msg = (f"patch EPE = {epe:.2f}px ({res['n_points']} pts) — lower = the arm patch followed your path."
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
                    "5&nbsp;second motion, then **Generate**. An `N×N` patch of points is dragged along your path.")
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
                    grid_n = gr.Slider(1, 9, value=5, step=1, label="Patch grid N (N×N points)")
                    extent = gr.Slider(0.02, 0.6, value=0.18, step=0.01, label="Patch extent (frac of frame)")
                background = gr.Radio(["static (dense)", "none (sparse patch)"], value="static (dense)",
                                      label="Background tracks")
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
                epe_box = gr.Textbox(label="Action-following (CoTracker EPE on the patch)", interactive=False)

        ff_b64 = gr.Textbox(elem_id="ff_b64", visible=False)
        traj_json = gr.Textbox(elem_id="traj_json", visible=False)

        clip.change(on_clip, [clip], [caption, ff_b64, orig_vid])
        ckpt.change(on_ckpt, [ckpt], [status])
        ff_b64.change(None, [ff_b64], None, js="(b64)=>{ window._setbg(b64); }")
        rec_btn.click(None, None, None, js="()=>{ window._startRec(); }")
        clr_btn.click(None, None, None, js="()=>{ window._clear(); }")
        preview_btn.click(on_preview, [clip, traj_json, grid_n, extent, background], [gen_vid, status])
        gen_btn.click(on_generate, [clip, ckpt, traj_json, grid_n, extent, background, steps, seed],
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
