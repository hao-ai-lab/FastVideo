# 05 — Evaluation Methodology (MotionStream reference)

Notes pulled from **MotionStream** (arXiv 2511.01266) + project page
(joonghyuk.com/motionstream-web) on how they evaluate and how motion control is
specified. This is the recipe we mirror.

## Motion controllability — End-Point Error (EPE)

> "L2 distance between visible input tracks and the tracks extracted from the
> generated videos."

Re-track the generated video with CoTracker3 at the input query points, compare the
extracted trajectories to the **input control tracks**, masked by input visibility.
Lower = better motion following. Reported EPE ranges ~**2.71 → 91.64** across methods.

This is the decisive controllability metric and what we implemented as
`motion.cotracker_epe` (see [06](06-tooling.md)). It is independent of whether the
content matches any reference — it measures adherence to the *control*.

## Visual quality

PSNR, SSIM, LPIPS, VBench-I2V. For motion transfer they compare to the GT video frames
(e.g. Wan2.1-1.3B 480p: teacher PSNR 16.61 / SSIM 0.477 / LPIPS 0.427).

## Datasets

| dataset | role | size |
|---|---|---|
| OpenVid-1M | teacher training | ~0.6M filtered |
| synthetic (Wan2.1) | teacher/distill | 70K, 480p, 81f |
| synthetic (Wan2.2) | teacher/distill | 30K, 720p, 121f |
| DAVIS val | motion eval | 30 videos |
| Sora demo subset | motion/quality eval | 20 videos |
| LLFF | novel-view / camera | standard |

Tracks for training are extracted with **CoTracker3 from a 50×50 uniform grid** on all
real and synthetic videos (matches our `extract_tracks.py`).

## Control input modes (how tracks are specified at test time)

1. **Trajectory painting / drag** — interactive click-and-drag; the user draws point
   paths in real time. Can pause/resume and add static points or multiple moving tracks.
2. **Camera control** — lift the image to 3D with a monocular depth network, then derive
   2D trajectories by projecting through camera params (dolly zoom, arcing, etc.).
3. **Motion transfer** — extract tracks from a source video (CoTracker3, or real-time
   facial/pose trackers) and apply them to a new first frame.

## Interactive / real-time

Click-and-drag demo at **~29 FPS @ 480p** (Tiny VAE) / 16.7 FPS @ 480p on a single
H100, sub-second latency. Real-time is the *post-distillation + self-forcing* payoff
(our Stages 3–4); our current bidir stage is tested **offline** with authored tracks
+ EPE.

## How we map onto this

| MotionStream | Our implementation |
|---|---|
| EPE (re-track vs input) | `motion.cotracker_epe` + `data_pipeline/test_controllability.py` |
| trajectory/drag | `synthetic_tracks.drag` + `select_radius` (sparse) + Gradio click-to-set-handle |
| camera-like | `synthetic_tracks.pan/zoom/rotate/swirl` (parametric; depth-based camera = TODO) |
| motion transfer | `synthetic_tracks.from_npz` / `swap` control |
| interactive demo | `examples/inference/gradio/trackwan/app.py` (offline, bidir) |
| quality (PSNR/SSIM/LPIPS) | already in `fastvideo/eval/metrics/common/` |
