# SPDX-License-Identifier: Apache-2.0
"""TrackWan eval-artifact VIEWER — browse pre-generated controllability videos.

No GPU / no model: just serves the artifacts written by
``data_pipeline/gen_eval_artifacts.py``. Pick a checkpoint, a clip, and a
counterfactual control; see, side by side:

  - the original clip (and the original with its GT tracks),
  - the generation (raw),
  - the generation with the INPUT control tracks (cyan = what we asked for),
  - the generation with the tracks RE-EXTRACTED from it (yellow = what it did),
  - the EPE heatmap (per-point error: green = followed, red = ignored),

plus the CoTracker EPE for that control.

Launch (no GPU needed)::

    .venv/bin/python examples/inference/gradio/trackwan/app_viewer.py \
      --artifacts-root /.../eval_artifacts --share
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import gradio as gr

CONTROL_ORDER = ["gt", "none", "pan_right", "zoom_in", "drag_dense", "drag_sparse", "swap"]
STEPS: dict = {}  # name -> {"dir": path, "m": manifest}


def _p(ckpt, fname):
    if not fname:
        return None
    path = os.path.join(STEPS[ckpt]["dir"], fname)
    return path if os.path.exists(path) else None


def update(ckpt, clip, control):
    if ckpt not in STEPS:
        return [None] * 7 + ["", ""]
    clips = STEPS[ckpt]["m"]["clips"]
    e = clips.get(str(clip))
    if e is None:
        return [None] * 7 + ["(clip not in this checkpoint)", ""]
    c = e["controls"].get(control, {})
    parts = []
    if c.get("epe") is not None:
        parts.append(f"EPE_all={c['epe']:.1f}px")
    if c.get("epe_moving") is not None:
        parts.append(f"EPE_moving={c['epe_moving']:.1f}px ({c.get('n_moving', 0)} moving pts)")
    if c.get("sensitivity_roi") is not None:
        parts.append(f"sensitivity_roi={c['sensitivity_roi']:.3f}")
    if c.get("localization_iou") is not None:
        parts.append(f"localization_IoU={c['localization_iou']:.3f}")
    if c.get("bg_leakage") is not None:
        parts.append(f"bg_leakage={c['bg_leakage']:.3f}")
    txt = "   |   ".join(parts) if parts else "n/a (GT baseline / no control tracks)"
    return (_p(ckpt, e.get("original")), _p(ckpt, e.get("original_tracks")),
            _p(ckpt, c.get("gen")), _p(ckpt, c.get("input")),
            _p(ckpt, c.get("tracked")), _p(ckpt, c.get("heat")), _p(ckpt, c.get("diff")),
            e.get("caption", ""), txt)


def build_ui(ckpt_names, clip_ids):
    with gr.Blocks(title="TrackWan — eval viewer") as demo:
        gr.Markdown("## TrackWan — controllability eval viewer\n"
                    "Browse pre-generated videos per **checkpoint × clip × control**. "
                    "Cyan = input tracks (asked), yellow = re-tracked from the generation (did), "
                    "EPE heatmap green→red = per-point EPE (followed→ignored). "
                    "**Diff** = |this control − GT-track generation|: bright where the control changed the "
                    "output (the real test — dark everywhere = the model ignored the control).")
        with gr.Row():
            ckpt = gr.Dropdown(ckpt_names, value=ckpt_names[0], label="Checkpoint")
            clip = gr.Dropdown(clip_ids, value=clip_ids[0], label="Clip")
            control = gr.Dropdown(CONTROL_ORDER, value="drag_dense", label="Control")
        caption = gr.Textbox(label="Prompt", interactive=False, lines=2)
        epe_box = gr.Textbox(label="Metrics (EPE_moving + counterfactual sensitivity/IoU = the honest ones)",
                             interactive=False)
        with gr.Row():
            v_orig = gr.Video(label="Original clip")
            v_orig_tr = gr.Video(label="Original + GT tracks")
        with gr.Row():
            v_gen = gr.Video(label="Generated (raw)")
            v_input = gr.Video(label="Generated + INPUT tracks (cyan)")
        with gr.Row():
            v_tracked = gr.Video(label="Generated + RE-TRACKED (yellow)")
            v_heat = gr.Video(label="EPE heatmap (green=followed, red=ignored)")
        with gr.Row():
            v_diff = gr.Video(label="DIFF vs GT-track gen (bright = control changed the output)")
            gr.Markdown("")

        outs = [v_orig, v_orig_tr, v_gen, v_input, v_tracked, v_heat, v_diff, caption, epe_box]
        for comp in (ckpt, clip, control):
            comp.change(update, [ckpt, clip, control], outs)
        demo.load(update, [ckpt, clip, control], outs)
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-root", required=True, help="dir containing step*/manifest.json")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    for d in sorted(glob.glob(os.path.join(args.artifacts_root, "*"))):
        mf = os.path.join(d, "manifest.json")
        if os.path.isfile(mf):
            with open(mf) as f:
                m = json.load(f)
            STEPS[m["name"]] = {"dir": d, "m": m}
    if not STEPS:
        raise SystemExit(f"no */manifest.json under {args.artifacts_root}")

    names = list(STEPS.keys())
    clip_ids = sorted(int(k) for k in STEPS[names[0]]["m"]["clips"])
    demo = build_ui(names, clip_ids)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        allowed_paths=[args.artifacts_root])


if __name__ == "__main__":
    main()
