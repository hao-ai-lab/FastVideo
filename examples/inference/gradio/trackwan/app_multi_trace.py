# SPDX-License-Identifier: Apache-2.0
"""TrackWan multi-trace demo — pick a preprocessed openvid clip, draw multiple traces
with mouse (Space to record), generate.

Design (matches training-time validation exactly):
  - Loads conditioning (text_embedding, clip_feature, first_frame_latent) DIRECTLY from
    the preprocessed openvid parquet — no on-the-fly encoding. Same path as
    ``fastvideo/train/callbacks/track_validation.py``.
  - First frame image shown to the user is decoded from the VAE latent (``decode_reference``).
  - Traces: press Space to record a 5s (121-frame) trajectory. Multi-trace stacks.
  - Background points: sampled from SAM foreground-avoiding pixels of frame-0. Click a dot
    to delete just it; buttons for resample / clear all.
  - Generation: uses the sample's precomputed conditioning + user's track_points, then runs
    the SAME motion-CFG denoise loop as ``TrackValidationCallback._sample``.

Launch::

    srun --overlap --jobid=<job> --ntasks=1 -w <node> --chdir=$PWD bash -lc "
      source .venv/bin/activate
      export ... TRACKWAN_TRACK_BIAS=1 CUDA_VISIBLE_DEVICES=0
      python examples/inference/gradio/trackwan/app_multi_trace.py \\
        --model-dir /path/to/diffusers_export --yaml <yaml> \\
        --data-path /mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset
    "

share=True doesn't work on aarch64 (no frpc); use SSH port-forward on the compute node.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
SAM_WEIGHT_CANDIDATES = [
    "/mnt/lustre/vlm-s4duan/models/seg/sam2.1_b.pt",
    "/mnt/lustre/vlm-s4duan/FastVideo/sam2.1_b.pt",
    "/mnt/lustre/vlm-s4duan/models/seg/sam2_b.pt",
]
NUM_FRAMES = 121
FPS = 24


# =============================================================================
# SAM
# =============================================================================
def load_sam(weight: str | None = None):
    from ultralytics import SAM
    if weight is None:
        for w in SAM_WEIGHT_CANDIDATES:
            if os.path.exists(w):
                weight = w
                break
    if weight is None:
        raise FileNotFoundError(f"No SAM weight found in {SAM_WEIGHT_CANDIDATES}")
    return SAM(weight)


def sam_foreground_mask(sam, img: np.ndarray, imgsz: int = 640) -> np.ndarray:
    res = sam(img, imgsz=imgsz, conf=0.4, iou=0.9, retina_masks=True, verbose=False)
    if not res or res[0].masks is None:
        return np.zeros(img.shape[:2], bool)
    m = res[0].masks.data.cpu().numpy().astype(bool)
    if m.shape[0] and m.shape[1:] != img.shape[:2]:
        import torch
        t = torch.from_numpy(m.astype(np.uint8))[None].float()
        t = torch.nn.functional.interpolate(t, size=img.shape[:2], mode="nearest")[0]
        m = t.numpy().astype(bool)
    return m.any(0) if m.shape[0] > 0 else np.zeros(img.shape[:2], bool)


def sample_bg_points_normalized(fg_mask: np.ndarray, n: int, seed: int = 0) -> list:
    ys, xs = np.where(~fg_mask)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    H, W = fg_mask.shape
    return [[float(xs[i]) / W, float(ys[i]) / H] for i in idx]


# =============================================================================
# frame b64 + track construction
# =============================================================================
def img_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_track_points(traces_json: str, bg_pts: list) -> tuple[np.ndarray, np.ndarray, int, int]:
    """List-of-normalized-trajectories → (track_points[T,N,2], track_visibility[T,N], n_tr, n_bg)."""
    T = NUM_FRAMES
    try:
        traces = json.loads(traces_json) if traces_json else []
    except Exception:
        traces = []
    n_tr, n_bg = len(traces), len(bg_pts)
    N = n_tr + n_bg
    if N == 0:
        return np.zeros((T, 0, 2), np.float32), np.zeros((T, 0), np.float32), 0, 0
    tp = np.zeros((T, N, 2), np.float32)
    tv = np.ones((T, N), np.float32)
    for i, traj in enumerate(traces):
        if not traj:
            continue
        arr = np.asarray(traj, np.float32)
        if arr.shape[0] < 2:
            tp[:, i] = arr[0]
        else:
            src = np.arange(arr.shape[0])
            tgt = np.linspace(0, arr.shape[0] - 1, T)
            tp[:, i, 0] = np.interp(tgt, src, arr[:, 0])
            tp[:, i, 1] = np.interp(tgt, src, arr[:, 1])
    for j, (bx, by) in enumerate(bg_pts):
        tp[:, n_tr + j, 0] = bx
        tp[:, n_tr + j, 1] = by
    return tp, tv, n_tr, n_bg


# =============================================================================
# canvas JS (multi-trace + click-to-delete bg points)
# =============================================================================
def canvas_js(num_frames: int, fps: int) -> str:
    palette = ["#ff3c3c", "#3cd23c", "#3c78ff", "#ffc83c", "#c83cc8", "#3cdcdc", "#ff821e", "#9696ff"]
    return ("() => {\n"
            "  const cvs = document.getElementById('cvs'); if (!cvs || cvs._init) return; cvs._init=true;\n"
            "  const ctx = cvs.getContext('2d');\n"
            f"  const PAL = {json.dumps(palette)};\n"
            f"  const N = {num_frames}; const DT = {int(1000 / fps)};\n"
            "  const st = { img:new Image(), mouse:[0.5,0.5], rec:false, current:[], traces:[], bg:[] };\n"
            "  cvs.width = 832; cvs.height = 480;\n"
            "  function commit_traces(){\n"
            "    const tb = document.querySelector('#traces_json textarea');\n"
            "    if (tb) { tb.value = JSON.stringify(st.traces); tb.dispatchEvent(new Event('input',{bubbles:true})); }\n"
            "  }\n"
            "  function commit_bg(){\n"
            "    const tb = document.querySelector('#bg_pts_json textarea');\n"
            "    if (tb) { tb.value = JSON.stringify(st.bg); tb.dispatchEvent(new Event('input',{bubbles:true})); }\n"
            "  }\n"
            "  function draw(){\n"
            "    ctx.clearRect(0,0,cvs.width,cvs.height);\n"
            "    if (st.img.complete && st.img.width) ctx.drawImage(st.img, 0, 0, cvs.width, cvs.height);\n"
            "    st.bg.forEach(p => { const x=p[0]*cvs.width, y=p[1]*cvs.height;\n"
            "      ctx.beginPath(); ctx.arc(x,y,3,0,7); ctx.fillStyle='#c8c8c8'; ctx.fill();\n"
            "      ctx.strokeStyle='#000'; ctx.stroke(); });\n"
            "    st.traces.forEach((traj, i) => {\n"
            "      const color = PAL[i % PAL.length];\n"
            "      ctx.strokeStyle=color; ctx.lineWidth=3; ctx.beginPath();\n"
            "      traj.forEach((p, j) => { const x=p[0]*cvs.width, y=p[1]*cvs.height; j?ctx.lineTo(x,y):ctx.moveTo(x,y); });\n"
            "      ctx.stroke();\n"
            "      const s = traj[0], e = traj[traj.length-1];\n"
            "      ctx.fillStyle=color; ctx.beginPath(); ctx.arc(s[0]*cvs.width, s[1]*cvs.height, 6, 0, 7); ctx.fill(); ctx.strokeStyle='#000'; ctx.stroke();\n"
            "      ctx.fillStyle='#ff3333'; ctx.beginPath(); ctx.arc(e[0]*cvs.width, e[1]*cvs.height, 6, 0, 7); ctx.fill(); ctx.strokeStyle='#000'; ctx.stroke();\n"
            "      ctx.fillStyle='#000'; ctx.font='13px sans-serif'; ctx.fillText('#'+(i+1), s[0]*cvs.width+8, s[1]*cvs.height-8);\n"
            "    });\n"
            "    if (st.rec && st.current.length) {\n"
            "      const color = PAL[st.traces.length % PAL.length];\n"
            "      ctx.strokeStyle=color; ctx.lineWidth=3; ctx.beginPath();\n"
            "      st.current.forEach((p, j) => { const x=p[0]*cvs.width, y=p[1]*cvs.height; j?ctx.lineTo(x,y):ctx.moveTo(x,y); });\n"
            "      ctx.stroke();\n"
            "    }\n"
            "  }\n"
            "  cvs.addEventListener('mousemove', e => {\n"
            "    const r = cvs.getBoundingClientRect();\n"
            "    st.mouse = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];\n"
            "  });\n"
            "  cvs.addEventListener('click', e => {\n"
            "    if (st.rec) return;\n"
            "    const r = cvs.getBoundingClientRect();\n"
            "    const cx = (e.clientX - r.left) * (cvs.width / r.width);\n"
            "    const cy = (e.clientY - r.top) * (cvs.height / r.height);\n"
            "    let idx = -1, best = 12*12;\n"
            "    st.bg.forEach((p, i) => { const dx = p[0]*cvs.width - cx, dy = p[1]*cvs.height - cy;\n"
            "      const d2 = dx*dx + dy*dy; if (d2 < best) { best = d2; idx = i; } });\n"
            "    if (idx >= 0) { st.bg.splice(idx, 1); commit_bg(); draw(); }\n"
            "  });\n"
            "  window._setbg = (b64) => {\n"
            "    if (!b64) return;\n"
            "    const im = new Image();\n"
            "    im.onload = () => { const ar = im.width / im.height; cvs.width = 832; cvs.height = Math.round(832 / ar); st.img = im; draw(); };\n"
            "    im.src = b64;\n"
            "  };\n"
            "  window._setbg_pts = (j) => { try { st.bg = JSON.parse(j) || []; } catch(e) { st.bg = []; } draw(); };\n"
            "  window._clear = () => {\n"
            "    st.traces = []; st.current = []; st.rec = false; commit_traces(); draw();\n"
            "    const stt = document.getElementById('recstatus'); if (stt) stt.textContent = 'cleared. Space for trace #1';\n"
            "  };\n"
            "  window._undo = () => {\n"
            "    if (st.traces.length) st.traces.pop();\n"
            "    commit_traces(); draw();\n"
            "    const stt = document.getElementById('recstatus'); if (stt) stt.textContent = st.traces.length + ' committed. Space for another';\n"
            "  };\n"
            "  window._clearbg = () => { st.bg = []; commit_bg(); draw(); };\n"
            "  window._startRec = () => {\n"
            "    if (st.rec) return;\n"
            "    st.rec = true; st.current = []; let n = 0;\n"
            "    const stt = document.getElementById('recstatus');\n"
            "    const iv = setInterval(() => {\n"
            "      st.current.push([st.mouse[0], st.mouse[1]]); n++; draw();\n"
            "      if (stt) stt.textContent = 'REC trace #' + (st.traces.length + 1) + ' — ' + n + '/' + N;\n"
            "      if (n >= N) {\n"
            "        clearInterval(iv); st.rec = false;\n"
            "        st.traces.push(st.current); st.current = [];\n"
            "        if (stt) stt.textContent = 'committed trace #' + st.traces.length + ' (Space for another, or Generate)';\n"
            "        commit_traces(); draw();\n"
            "      }\n"
            "    }, DT);\n"
            "  };\n"
            "  document.addEventListener('keydown', e => {\n"
            "    if (e.code !== 'Space' || e.repeat) return;\n"
            "    const tag = document.activeElement && document.activeElement.tagName;\n"
            "    if (tag === 'INPUT' || tag === 'TEXTAREA') return;\n"
            "    e.preventDefault(); window._startRec();\n"
            "  });\n"
            "  draw();\n"
            "}")


# =============================================================================
# Gradio UI
# =============================================================================
def build_ui(state):
    canvas_html = ("<div style='user-select:none'>"
                   "<canvas id='cvs' style='border:1px solid #888;cursor:crosshair;max-width:100%'></canvas>"
                   "<div id='recstatus' style='font-weight:bold;color:#0a0;height:1.4em;margin-top:6px'>"
                   "pick a clip, click <b>Load frame</b>, then hover and press <b>Space</b> to record trace #1</div>"
                   "</div>")
    labels = state["labels"]
    with gr.Blocks(title="TrackWan multi-trace demo") as demo:
        gr.Markdown(
            "## TrackWan — multi-trace generation (parquet-conditioned)\n"
            "1. Pick a preprocessed openvid clip → **Load frame**.\n"
            "2. Hover over the frame and press **Space** to record a 5-second trace.\n"
            "3. Repeat Space for more traces. Optionally include background points. **Generate**.")

        with gr.Row():
            with gr.Column(scale=1):
                clip = gr.Dropdown(labels, value=labels[0] if labels else None, label="Preprocessed clip")
                caption = gr.Textbox(label="Caption (from parquet)", interactive=False, lines=2)
                load_btn = gr.Button("Load frame (runs SAM)", variant="primary")
                sam_on = gr.Checkbox(True, label="Run SAM (needed for background points)")
                gr.Markdown("**Background points (SAM foreground-avoiding)**")
                bg_on = gr.Checkbox(True, label="Include background points")
                bg_n = gr.Slider(0, 50, 20, step=1, label="Num background points")
                with gr.Row():
                    resample_bg_btn = gr.Button("Resample")
                    clear_bg_btn = gr.Button("Clear all bg pts")
                gr.Markdown("_Tip: **click a bg dot** on the frame to delete just that one._")
                gr.Markdown("**Trace drawing**")
                with gr.Row():
                    rec_btn = gr.Button("● Record (Space)")
                    undo_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Clear")
                trace_info = gr.Markdown("_no traces yet._")
                gr.Markdown("**Generation** (denoise formula matches training's validation exactly)")
                seed_in = gr.Slider(0, 9999, 1000, step=1, label="Seed")
                steps_in = gr.Slider(10, 60, 30, step=1, label="Denoise steps")
                w_text = gr.Slider(1.0, 8.0, 3.0, step=0.25, label="Text CFG (w_t)")
                w_motion = gr.Slider(1.0, 5.0, 1.5, step=0.25, label="Motion CFG (w_m)")
                gen_btn = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=2):
                gr.HTML(canvas_html)
                out_video = gr.Video(label="Generated video", height=380)
                status = gr.Markdown("_ready_")
                log_box = gr.Textbox(label="Generation log", interactive=False, lines=8,
                                     max_lines=16, value="_no runs yet._")

        # hidden bridges
        ff_b64 = gr.Textbox(elem_id="ff_b64", visible=False)
        bg_pts_json = gr.Textbox(elem_id="bg_pts_json", visible=False, value="[]")
        traces_json = gr.Textbox(elem_id="traces_json", visible=False, value="[]")

        # server-side per-page state
        st_clip_idx = gr.State(0)
        st_frame = gr.State(None)
        st_fg = gr.State(None)

        # ---------------- caption on clip change ----------------
        def _cap(clip_name):
            i = state["label_to_idx"].get(clip_name, 0)
            return i, state["samples"][i]["caption"][:400]
        clip.change(_cap, [clip], [st_clip_idx, caption])

        # ---------------- load frame (decode ref frame from VAE latent + SAM) ----------------
        sam_cache: dict[int, np.ndarray] = {}
        first_frame_cache: dict[int, np.ndarray] = {}

        def _load(clip_name, sam_on_, bg_n_, bg_on_):
            i = state["label_to_idx"].get(clip_name, 0)
            if i not in first_frame_cache:
                first_frame_cache[i] = state["decode_first_frame"](i)
            frame = first_frame_cache[i]
            if sam_on_:
                if i in sam_cache:
                    fg = sam_cache[i]
                    msg = f"_clip loaded ({state['samples'][i]['caption'][:60]}...) — SAM cached, {int(fg.sum())} fg px_"
                else:
                    t0 = time.time()
                    try:
                        fg = sam_foreground_mask(state["sam"], frame)
                        sam_cache[i] = fg
                    except Exception as e:
                        fg = np.zeros(frame.shape[:2], bool)
                    msg = f"_clip loaded — SAM {time.time()-t0:.1f}s, {int(fg.sum())} fg px_"
            else:
                fg = np.zeros(frame.shape[:2], bool)
                msg = "_clip loaded (SAM skipped)_"
            bg = sample_bg_points_normalized(fg, int(bg_n_)) if (bg_on_ and fg.any()) else []
            return i, frame, fg, img_to_b64(frame), json.dumps(bg), msg

        load_btn.click(_load, [clip, sam_on, bg_n, bg_on],
                       [st_clip_idx, st_frame, st_fg, ff_b64, bg_pts_json, status])

        # preload first clip on startup
        def _preload():
            i = 0
            if i not in first_frame_cache:
                first_frame_cache[i] = state["decode_first_frame"](i)
            frame = first_frame_cache[i]
            fg = np.zeros(frame.shape[:2], bool)
            return (i, state["samples"][i]["caption"][:400], frame, fg, img_to_b64(frame),
                    "[]", "_default clip preloaded (SAM off). Click **Load frame** to enable bg points._")
        demo.load(_preload, None,
                  [st_clip_idx, caption, st_frame, st_fg, ff_b64, bg_pts_json, status])

        # ---------------- bg controls ----------------
        state.setdefault("bg_seed", [0])
        def _update_bg(fg, on_, n_, fresh: bool = False):
            if fg is None or not on_ or not fg.any():
                return "[]"
            if fresh:
                state["bg_seed"][0] += 1
            return json.dumps(sample_bg_points_normalized(fg, int(n_), seed=state["bg_seed"][0]))
        for w in (bg_on, bg_n):
            w.change(_update_bg, [st_fg, bg_on, bg_n], bg_pts_json)
        resample_bg_btn.click(lambda fg, on_, n_: _update_bg(fg, on_, n_, fresh=True),
                              [st_fg, bg_on, bg_n], bg_pts_json)
        clear_bg_btn.click(None, None, None, js="() => window._clearbg && window._clearbg()")

        # ---------------- JS bridges ----------------
        ff_b64.change(None, ff_b64, None, js="(b) => window._setbg && window._setbg(b)")
        bg_pts_json.change(None, bg_pts_json, None, js="(j) => window._setbg_pts && window._setbg_pts(j)")
        rec_btn.click(None, None, None, js="() => window._startRec && window._startRec()")
        undo_btn.click(None, None, None, js="() => window._undo && window._undo()")
        clear_btn.click(None, None, None, js="() => window._clear && window._clear()")

        def _tinfo(t):
            try:
                n = len(json.loads(t) if t else [])
            except Exception:
                n = 0
            return "_no traces yet. Press **Space** to record trace #1._" if n == 0 \
                else f"_{n} trace(s) committed. **Space** for another, or **Generate**._"
        traces_json.change(_tinfo, traces_json, trace_info)

        # ---------------- generate (MATCHES TrackValidationCallback._sample exactly) ----------------
        def _generate(clip_idx, traces_str, bg_str, seed, steps, w_t, w_m):
            def logln(log, msg):
                line = time.strftime("%H:%M:%S ") + msg
                print(line, flush=True)
                return (log + "\n" + line) if log else line

            log = ""
            try:
                bg = json.loads(bg_str) if bg_str else []
            except Exception:
                bg = []
            tp_np, tv_np, n_tr, n_bg = build_track_points(traces_str, bg)
            log = logln(log, f"clip idx={clip_idx}: {n_tr} trace(s) + {n_bg} bg point(s) → N={tp_np.shape[1]}")
            if tp_np.shape[1] == 0:
                return None, "_record at least one trace, or enable bg points_", log

            ts = time.strftime("%Y%m%d_%H%M%S")
            req_dir = Path(state["out_dir"]) / f"req_{ts}"
            req_dir.mkdir(parents=True, exist_ok=True)
            np.savez(req_dir / "tracks.npz", track_points=tp_np, track_visibility=tv_np)
            (req_dir / "meta.json").write_text(json.dumps({
                "clip_idx": int(clip_idx), "seed": int(seed), "steps": int(steps),
                "w_text": float(w_t), "w_motion": float(w_m),
                "num_traces": n_tr, "num_bg": n_bg,
            }, indent=2))
            log = logln(log, f"req -> {req_dir.name}, denoising ...")

            try:
                t0 = time.time()
                mp4 = state["run_generate"](int(clip_idx), tp_np, tv_np,
                                             int(seed), int(steps), float(w_t), float(w_m),
                                             req_dir)
                elapsed = time.time() - t0
                log = logln(log, f"done in {elapsed:.1f}s → {mp4}")
                return str(mp4), f"_generated in {elapsed:.1f}s_", log
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"_generation failed: {e}_", logln(log, f"EXC: {e}")

        gen_btn.click(_generate,
                      [st_clip_idx, traces_json, bg_pts_json, seed_in, steps_in, w_text, w_motion],
                      [out_video, status, log_box])

        demo.load(None, None, None, js=canvas_js(NUM_FRAMES, FPS))

    return demo


# =============================================================================
# Generation (MATCHES TrackValidationCallback._sample)
# =============================================================================
def make_generator(model, tc, samples):
    """Return a callable (clip_idx, tp_np, tv_np, seed, steps, w_t, w_m, out_dir) -> mp4 path.

    The denoise formula is copied verbatim from
    ``fastvideo/train/callbacks/track_validation.py::_sample`` (MotionStream Eq. 2).
    """
    import torch
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler, )

    device = model.device
    dtype = torch.bfloat16
    flow_shift = float(model.timestep_shift)
    transformer = model.transformer

    @torch.no_grad()
    def _run(clip_idx: int, tp_np: np.ndarray, tv_np: np.ndarray,
             seed: int, steps: int, w_t: float, w_m: float, out_dir: Path) -> str:
        s = samples[clip_idx]
        ff = s["first_frame_latent"].to(device, dtype)
        cond20 = model._build_i2v_cond_concat(ff)
        txt = s["text_embedding"].to(device, dtype)
        mask = s["text_attention_mask"].to(device, dtype)
        img = s["clip_feature"].to(device, dtype)
        # user tracks (override the sample's tracks)
        tp = torch.from_numpy(tp_np)[None].to(device, dtype)  # [1,T,N,2]
        tv = torch.from_numpy(tv_np)[None].to(device, dtype)  # [1,T,N]

        _, _, T, H, W = ff.shape
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn((1, 16, T, H, W), generator=gen, dtype=torch.float32).to(device)
        sched = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        sched.set_timesteps(int(steps), device=device)
        cfg_on = (w_t != 1.0) or (w_m != 1.0)
        txt_null = torch.zeros_like(txt) if cfg_on else None

        def _fwd(text_e, tp_e, tv_e, mi, tsv):
            with torch.autocast(device.type, dtype=dtype), set_forward_context(current_timestep=tsv,
                                                                                attn_metadata=None):
                return transformer(hidden_states=mi, encoder_hidden_states=text_e,
                                   encoder_attention_mask=mask, timestep=tsv,
                                   encoder_hidden_states_image=img, track_points=tp_e,
                                   track_visibility=tv_e, return_dict=False)

        for tt in sched.timesteps:
            model_in = torch.cat([latents.to(dtype), cond20], dim=1)
            tss = tt.reshape(1).to(device, dtype)
            v_full = _fwd(txt, tp, tv, model_in, tss)
            if not cfg_on:
                v = v_full
            else:
                v_no_text = _fwd(txt_null, tp, tv, model_in, tss)
                v_no_motion = _fwd(txt, None, None, model_in, tss)
                alpha = w_t / (w_t + w_m) if (w_t + w_m) > 0 else 0.5
                v_base = alpha * v_no_text + (1.0 - alpha) * v_no_motion
                v = v_base + w_t * (v_full - v_no_text) + w_m * (v_full - v_no_motion)
            latents = sched.step(v.float(), tt, latents.float(), return_dict=False)[0]

        px = model.decode_latents(latents.permute(0, 2, 1, 3, 4))[0]  # [3,T,H,W] in [0,1]
        video = (px.clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
        frames = np.transpose(video, (1, 2, 3, 0))
        mp4 = out_dir / "generation.mp4"
        imageio.mimsave(mp4, frames, fps=FPS, macro_block_size=1)
        return str(mp4)

    return _run


# =============================================================================
# Efficient parquet loader (reads only enough files to get N rows, NOT all 4494)
# =============================================================================
def _load_first_n_from_parquet(data_path: str, n: int, text_len: int) -> list:
    """Grab the first N rows from a preprocessed parquet dataset (like
    ``trackwan_infer.load_conditioning_from_parquet``, but stops after N rows so
    259k-row datasets don't OOM the loader).
    """
    import glob
    import pyarrow.parquet as pq
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema

    files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *.parquet under {data_path}")
    rows: list = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
        if len(rows) >= n:
            break
    sel = rows[:n]
    print(f"[app]   loaded {len(sel)} rows from {min(len(files), 1 + (n // max(1, (len(sel)//1))))} parquet file(s)", flush=True)
    batch = collate_rows_from_parquet_schema(sel, pyarrow_schema_i2v_track,
                                             text_padding_length=int(text_len), cfg_rate=0.0)
    infos = batch.get("info_list") or [{} for _ in sel]
    out = []
    for i in range(len(sel)):
        out.append({
            "text_embedding": batch["text_embedding"][i:i + 1].clone(),
            "text_attention_mask": batch["text_attention_mask"][i:i + 1].clone(),
            "vae_latent": batch["vae_latent"][i:i + 1].clone(),
            "first_frame_latent": batch["first_frame_latent"][i:i + 1].clone(),
            "clip_feature": batch["clip_feature"][i:i + 1].clone(),
            "track_points": batch["track_points"][i:i + 1].clone(),
            "track_visibility": batch["track_visibility"][i:i + 1].clone(),
            "caption": str(infos[i].get("caption", "") if i < len(infos) else ""),
        })
    return out


# =============================================================================
# main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="diffusers-exported ckpt dir")
    ap.add_argument("--yaml", required=True, help="training yaml")
    ap.add_argument("--data-path", required=True,
                    help="preprocessed openvid parquet root (combined_parquet_dataset)")
    ap.add_argument("--num-clips", type=int, default=30, help="how many parquet clips to preload")
    ap.add_argument("--out-dir", default="/mnt/lustre/vlm-s4duan/multi_trace_out")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7864)
    ap.add_argument("--sam-weight", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state: dict = {"out_dir": args.out_dir}

    # SAM (works on GPU when CUDA_VISIBLE_DEVICES set)
    print(f"[app] loading SAM ...", flush=True)
    state["sam"] = load_sam(args.sam_weight)

    # Load model
    print(f"[app] loading model from {args.model_dir} ...", flush=True)
    sys.path.insert(0, str(REPO / "data_pipeline"))
    import trackwan_infer as twi
    model, tc = twi.load_trackwan(args.model_dir, args.yaml)
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    state["model"] = model
    state["tc"] = tc

    # Load parquet conditioning EFFICIENTLY: stop as soon as we have enough rows.
    # (trackwan_infer.load_conditioning_from_parquet reads ALL 4494 files → OOM on 259k rows.)
    print(f"[app] loading {args.num_clips} parquet clips ...", flush=True)
    samples = _load_first_n_from_parquet(args.data_path, args.num_clips, text_len)
    labels = []
    label_to_idx = {}
    for i, s in enumerate(samples):
        # Trim caption to a short unique label for the dropdown
        cap = (s["caption"] or f"clip_{i}").strip().replace("\n", " ")
        lab = f"{i:03d}  {cap[:60]}"
        labels.append(lab)
        label_to_idx[lab] = i
    state["samples"] = samples
    state["labels"] = labels
    state["label_to_idx"] = label_to_idx

    # First-frame decoder (from vae_latent) — used as the drawing background
    def _decode_first_frame(i: int) -> np.ndarray:
        return twi.decode_reference(model, samples[i]["vae_latent"])[0]  # [H,W,3] uint8
    state["decode_first_frame"] = _decode_first_frame

    # Generator (validation callback's exact denoise loop)
    state["run_generate"] = make_generator(model, tc, samples)

    print(f"[app] ready. {len(samples)} clips loaded.", flush=True)
    demo = build_ui(state)
    demo.queue().launch(server_name=args.host, server_port=args.port,
                        share=False, allowed_paths=[args.out_dir])


if __name__ == "__main__":
    main()
