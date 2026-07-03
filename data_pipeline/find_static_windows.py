# SPDX-License-Identifier: Apache-2.0
"""Find relatively-static WINDOWS inside (long) videos, before tracking.

The camera in egocentric video turns and the whole view sweeps off-frame; a filter that
seeds points once and tracks the whole clip would die at the first turn and mis-score
everything after. Instead we compute a PER-FRAME global-motion signal that RE-SEEDS every
frame (so it recovers after a turn: motion spikes during the turn, drops when static), then
slide a window and locate the calm stretches.

Per-frame motion m[t] = median Lucas-Kanade optical-flow magnitude between frames t-1 and t
(fresh goodFeaturesToTrack each step), normalized by the frame diagonal. Camera pan/turn ->
large; static -> small. A window is static iff m[t] stays low across ALL its frames (we score
by a high percentile so one calm-but-not-perfect frame is fine but a turn is not). We also
report per-window SURVIVAL: seed a grid at the window START, LK-track to the window END, and
measure the fraction still tracked & in-frame -> "would CoTracker keep traces here?".

    .venv/bin/python data_pipeline/find_static_windows.py scan  --videos-dir <dir> --out <dir> --window-sec 5 --limit 12
    .venv/bin/python data_pipeline/find_static_windows.py serve --viz-dir <dir> --share
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def read_gray_lowres(path: str, max_frames: int = 900, target_w: int = 320) -> tuple[list, float, int]:
    """Decode DIRECTLY at low res (fast + tiny memory) -> gray frames + fps + total_frames.
    Only used for the motion analysis; full-res frames are read on demand for extraction."""
    import cv2
    from decord import VideoReader, cpu
    cap = cv2.VideoCapture(path)  # fast metadata read (no full decode)
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    tw = min(target_w, W0)
    th = int(round(H0 * tw / max(W0, 1)))
    try:
        vr = VideoReader(path, ctx=cpu(0), width=tw, height=th)
    except Exception:  # noqa: BLE001  (older decord without width/height)
        vr = VideoReader(path, ctx=cpu(0))
    n_total = len(vr)
    n = min(n_total, max_frames)
    batch = vr.get_batch(list(range(n))).asnumpy()
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in batch]
    return gray, fps, n_total


def read_rgb_frames(path: str, indices: Any) -> np.ndarray:
    """Full-res RGB for a specific set of frame indices (decord random access)."""
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    return vr.get_batch(list(indices)).asnumpy()


def motion_series(gray: list) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-frame normalized global camera motion via phase correlation (FFT-based global
    shift between consecutive frames, ~1ms/frame). Recovers after turns: spikes during a
    turn, drops when static. Returns (m, shifts, diag): m[t] = |shift|/diag, shifts[t] =
    signed (dx,dy)/diag (used to accumulate net view drift over a window)."""
    import cv2
    H, W = gray[0].shape
    diag = float(np.hypot(H, W))
    hann = cv2.createHanningWindow((W, H), cv2.CV_32F)
    m = np.zeros(len(gray), np.float32)
    shifts = np.zeros((len(gray), 2), np.float32)
    prev = gray[0].astype(np.float32)
    for t in range(1, len(gray)):
        cur = gray[t].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, hann)
        shifts[t] = (dx / diag, dy / diag)
        m[t] = float(np.hypot(dx, dy)) / diag
        prev = cur
    m[0] = m[1] if len(m) > 1 else 0.0
    return m, shifts, diag


def window_drift(shifts: np.ndarray, s: int, L: int) -> float:
    """Net view drift over a window: max cumulative excursion of the accumulated per-frame
    shifts, normalized by diagonal. Large => the camera slowly panned the view off-screen
    even if no single frame moved much. Free (from the phase-correlation shifts)."""
    c = np.cumsum(shifts[s:s + L], axis=0)
    return float(np.max(np.hypot(c[:, 0], c[:, 1]))) if len(c) else 0.0


def window_survival(gray: list, s: int, L: int, grid: int = 24) -> float:
    """Seed a grid at frame s, LK-track to s+L-1; fraction still tracked & in-frame."""
    import cv2
    H, W = gray[0].shape
    xs = np.linspace(W * 0.05, W * 0.95, grid)
    ys = np.linspace(H * 0.05, H * 0.95, grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    alive = np.ones(pts.shape[0], bool)
    cur = pts.copy()
    lk = dict(winSize=(21, 21), maxLevel=3)
    for t in range(s + 1, min(s + L, len(gray))):
        nxt, stt, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], cur, None, **lk)
        stt = stt.ravel().astype(bool)
        x, y = nxt[:, 0, 0], nxt[:, 0, 1]
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        alive &= stt & inb
        cur = nxt
    return float(alive.mean())


def best_windows(m: np.ndarray, L: int, pct: float = 95, top: int = 1, min_gap: int | None = None) -> list:
    """Return up to `top` low-motion windows (start index) minimizing the pct-percentile of
    per-frame motion, non-overlapping by min_gap (default L)."""
    if len(m) < L:
        return []
    min_gap = min_gap or L
    scores = np.array([np.percentile(m[s:s + L], pct) for s in range(len(m) - L + 1)])
    order = np.argsort(scores)
    picks: list[int] = []
    for s in order:
        if all(abs(s - p) >= min_gap for p in picks):
            picks.append(int(s))
            if len(picks) >= top:
                break
    return picks


def greedy_valid_windows(m: np.ndarray,
                         shifts: np.ndarray,
                         L: int,
                         pct: float,
                         p95_thresh: float,
                         drift_thresh: float,
                         cand_step: int | None = None) -> list:
    """Greedily take ALL non-overlapping windows passing p95_motion<=p95_thresh AND
    net-drift<=drift_thresh, calmest-first. Returns list of (start, p95, drift). All metrics
    come from the phase-correlation signal (no LK), so this is cheap."""
    if len(m) < L:
        return []
    cand_step = cand_step or max(1, L // 6)
    cand = [(s, float(np.percentile(m[s:s + L], pct))) for s in range(0, len(m) - L + 1, cand_step)]
    cand.sort(key=lambda x: x[1])  # calmest first
    accepted: list[tuple[int, float, float]] = []
    for s, p in cand:
        if p > p95_thresh:
            break  # sorted -> everything after is worse
        if any(abs(s - a[0]) < L for a in accepted):
            continue  # overlaps an accepted window
        drift = window_drift(shifts, s, L)
        if drift > drift_thresh:
            continue
        accepted.append((int(s), round(p, 4), round(drift, 3)))
    return sorted(accepted)


def cmd_scan(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    vids = sorted(Path(args.videos_dir).glob("**/*.mp4"))[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, vp in enumerate(vids, 1):
        gray, fps, _ = read_gray_lowres(str(vp), args.max_frames)
        L = int(round(args.window_sec * fps))
        if len(gray) < L:
            print(f"[win] [{i}/{len(vids)}] {vp.name}: only {len(gray)}f < {L}, skip", flush=True)
            continue
        m, shifts, diag = motion_series(gray)
        picks = best_windows(m, L, args.pct, top=args.top)
        stem = vp.stem
        # motion curve with chosen windows shaded
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(np.arange(len(m)) / fps, m, lw=0.8, color="steelblue")
        ax.axhline(args.calm_thresh, color="red", ls="--", lw=0.8, label=f"calm thresh {args.calm_thresh}")
        wstats = []
        for j, s in enumerate(picks):
            drift = window_drift(shifts, s, L)
            p95 = float(np.percentile(m[s:s + L], args.pct))
            ax.axvspan(s / fps, (s + L) / fps, color="limegreen", alpha=0.3)
            ax.text((s + L / 2) / fps, m.max() * 0.9, f"drift {drift:.3f}", ha="center", fontsize=8)
            wstats.append({
                "start_frame": s,
                "start_sec": round(s / fps, 2),
                "len_frames": L,
                "p95_motion": round(p95, 4),
                "drift": round(drift, 3)
            })
        ax.set_xlabel("time (s)")
        ax.set_ylabel("per-frame motion (norm)")
        ax.set_title(f"{stem}  ({len(m)}f @ {fps:.0f}fps)  green = static {args.window_sec}s window")
        ax.legend(fontsize=8)
        fig.tight_layout()
        curve = f"{stem}_motion.png"
        fig.savefig(str(out / curve), dpi=80)
        plt.close(fig)
        # preview: the best window as a short clip (full-res, read on demand)
        prev = ""
        if picks:
            s = picks[0]
            rgbw = read_rgb_frames(str(vp), range(s, s + L))
            imageio.mimsave(str(out / f"{stem}_win.mp4"), rgbw, fps=int(round(fps)), macro_block_size=1)
            prev = f"{stem}_win.mp4"
        entries.append({
            "id": stem,
            "src": str(vp),
            "fps": round(fps, 2),
            "n_frames": len(m),
            "curve": curve,
            "preview": prev,
            "windows": wstats,
            "median_motion": round(float(np.median(m)), 4)
        })
        best = wstats[0] if wstats else {}
        print(
            f"[win] [{i}/{len(vids)}] {stem}: {len(m)}f, best window @{best.get('start_sec')}s "
            f"p95={best.get('p95_motion')} surv={best.get('survival')}",
            flush=True)
    (out / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"[win] scanned {len(entries)} videos -> {out}", flush=True)


def cmd_build(args: argparse.Namespace) -> None:
    """Scan untrimmed source clips, greedily extract all non-overlapping static windows, and
    write them as target_frames-frame clips (subsampled to span the whole window) + manifest."""
    import imageio.v2 as imageio
    vids = sorted(Path(args.videos_dir).glob("**/*.mp4"))
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    GENERIC = "a first-person egocentric view of a person performing a kitchen task with their hands"
    man = []
    idx = 0
    for vi, vp in enumerate(vids, 1):
        if idx >= args.target_count:
            break
        try:
            gray, fps, _ = read_gray_lowres(str(vp), args.max_frames)
        except Exception as e:  # noqa: BLE001
            print(f"[build] {vp.name}: read fail {e}", flush=True)
            continue
        L = int(round(args.window_sec * fps))
        if len(gray) < L:
            continue
        m, shifts, _ = motion_series(gray)
        wins = greedy_valid_windows(m, shifts, L, args.pct, args.p95_thresh, args.drift_thresh)
        stride = max(1, round(L / args.target_frames))
        out_fps = round(fps / stride, 3)
        part = vp.stem.split("_")[0]
        n_from_clip = 0
        for (s, p95, drift) in wins:
            if idx >= args.target_count:
                break
            sel = np.arange(s, s + L, stride)[:args.target_frames]
            if sel.size < args.target_frames:
                continue
            f = read_rgb_frames(str(vp), sel)  # full-res, only the window frames
            f = f[:, :f.shape[1] // 2 * 2, :f.shape[2] // 2 * 2]  # even dims for libx264
            name = f"vid_{idx:06d}.mp4"
            imageio.mimsave(str(out / "videos" / name), f, fps=int(round(out_fps)), macro_block_size=1)
            man.append({
                "idx": idx,
                "path": name,
                "cap": [GENERIC],
                "fps": out_fps,
                "num_frames": int(args.target_frames),
                "duration": round(args.target_frames / out_fps, 3),
                "resolution": [int(f.shape[1]), int(f.shape[2])],
                "participant": part,
                "orig": vp.stem,
                "win_start_sec": round(s / fps, 2),
                "p95_motion": p95,
                "drift": drift
            })
            idx += 1
            n_from_clip += 1
        if n_from_clip:
            print(f"[build] [{vi}/{len(vids)}] {vp.stem}: +{n_from_clip} windows (total {idx})", flush=True)
    (out / "videos2caption.json").write_text(json.dumps(man, indent=2))
    (out / "merge.txt").write_text(f"{out}/videos,{out}/videos2caption.json\n")
    from collections import Counter
    print(
        f"[build] wrote {len(man)} static clips -> {out} | per-participant {dict(Counter(x['participant'] for x in man))}",
        flush=True)


def cmd_serve(args: argparse.Namespace) -> None:
    import gradio as gr
    viz = Path(args.viz_dir).resolve()
    entries = json.loads((viz / "manifest.json").read_text())
    gal = [(str(viz / e["curve"]), e["id"]) for e in entries]

    def show(evt: gr.SelectData):
        e = entries[evt.index]
        vp = str(viz / e["preview"]) if e.get("preview") else None
        return str(viz / e["curve"]), vp, json.dumps(e["windows"], indent=2)

    with gr.Blocks(title="Static-window finder") as demo:
        gr.Markdown("### Per-frame camera motion + static-window finder\n"
                    "The curve is per-frame global motion (re-seeded each frame, so it **recovers after a turn** "
                    "— spikes = turns, valleys = static). Green span = the chosen static window; `surv` = grid "
                    "survival seeded at the window start. Click a clip -> motion curve + a preview of the best "
                    "window + per-window stats. We keep windows with low `p95_motion` and high `survival`.")
        with gr.Row():
            g = gr.Gallery(value=gal, columns=2, height=560, label="videos (click)")
            with gr.Column():
                curve = gr.Image(label="per-frame motion (green = static window)")
                vid = gr.Video(label="preview of best static window")
                meta = gr.Code(label="windows", language="json")
        g.select(show, None, [curve, vid, meta])
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(viz)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("scan")
    r.add_argument("--videos-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--window-sec", type=float, default=5.0)
    r.add_argument("--pct", type=float, default=95.0, help="percentile of per-frame motion used to score a window")
    r.add_argument("--calm-thresh", type=float, default=0.01, help="reference 'calm' motion line for the plot")
    r.add_argument("--top", type=int, default=2, help="static windows to find per video")
    r.add_argument("--max-frames", type=int, default=900, help="cap frames scanned per video (bounds cost)")
    r.add_argument("--limit", type=int, default=12)
    r.set_defaults(func=cmd_scan)
    b = sub.add_parser("build")
    b.add_argument("--videos-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--window-sec", type=float, default=5.0)
    b.add_argument("--target-frames", type=int, default=121, help="frames per output clip (window subsampled to span)")
    b.add_argument("--target-count", type=int, default=200)
    b.add_argument("--pct", type=float, default=95.0)
    b.add_argument("--p95-thresh", type=float, default=0.004, help="max p95 per-frame motion for a valid window")
    b.add_argument("--drift-thresh", type=float, default=0.35, help="max net view drift (frac of diagonal)")
    b.add_argument("--max-frames", type=int, default=1500, help="cap frames scanned per source clip")
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("serve")
    s.add_argument("--viz-dir", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7890)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_serve)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
