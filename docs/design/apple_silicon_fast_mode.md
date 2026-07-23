# Fast mode (RIFE) — Apple Silicon

`--fast` makes local generation ~2.7× faster by **generating fewer frames and
interpolating the rest** with an Apple-Silicon-native RIFE model, instead of
denoising every frame. Video-diffusion denoise is dominated by self-attention,
which is O(tokens²); halving the frames cuts the token count ~2× and the denoise
compute ~3.7×, so the wall-clock drops far more than 2×. RIFE (which estimates
its own optical flow — no motion vectors needed) fills the dropped frames back
in for ~1.4 s, and a light unsharp pass counters its softening.

Measured on the 1.3B INT8 QAD model (fox, 480×832×81, M4): generate 41 + RIFE→81
runs in ~35 s of denoise vs ~90 s full, at reconstruction MS-SSIM **0.97**.
Reproduce with `python -m fastvideo.benchmarks.eval_metalfx_rife --mode int8`.

> **Note:** Apple's *MetalFX* frame interpolation is **not** usable here — it
> requires game-engine motion vectors + depth, which diffusion output lacks. We
> use the video-native **`rife-mlx`** model instead (Metal-backed, torch-free).

## Install

```bash
uv pip install -e ".[mlx]"   # pulls in rife-mlx (git dependency)
```

## Use

```bash
python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root <FastWan2.1-T2V-1.3B-INT8-QAD> \
  --prompt "A red fox trotting through a snowy pine forest at golden hour, cinematic" \
  --num-frames 81 --fast \
  --output-path video_samples/fox_fast.mp4
```

`--num-frames` stays the *target* length; fast mode generates `num_frames /
fast_factor` frames and interpolates up.

| Flag | Default | Meaning |
|---|---|---|
| `--fast` / `--no-fast` | off | enable fast mode |
| `--fast-factor` | 2 | generate 1/factor of the frames (2 = half) |
| `--fast-sharpen` | 0.6 | light unsharp strength to counter RIFE softness (0 disables) |

Fast mode composes with everything else (`--mlx-quantization int8`,
`--mlx-compile`, TAEHV vs `--decode-backend wan-vae`). Keep `--fast-factor` at 2
for quality — larger temporal gaps are where RIFE starts inventing motion.
