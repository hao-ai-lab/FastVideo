# SPDX-License-Identifier: Apache-2.0
"""VAE round-trip test: encode -> decode with/without tiling.

Loads a video, encodes it through the HunyuanVideo VAE, decodes it back,
and saves both original and reconstructed videos for visual comparison.
"""

import os
import sys
import glob

import cv2
import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29503")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")


def load_video_opencv(path: str, max_frames: int = 0) -> torch.Tensor:
    """Load video as tensor [1, C, T, H, W] in [-1, 1]."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).float()
    video = video / 127.5 - 1.0
    video = video.permute(3, 0, 1, 2).unsqueeze(0)
    return video


def save_video(tensor: torch.Tensor, path: str,
               fps: float = 16.0) -> None:
    """Save tensor [1, C, T, H, W] in [-1, 1] to mp4 via ffmpeg."""
    import subprocess
    import imageio_ffmpeg
    video = tensor.squeeze(0).permute(1, 2, 3, 0)
    video = ((video + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    video = video.cpu().numpy()
    h, w = video.shape[1], video.shape[2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for i in range(video.shape[0]):
        proc.stdin.write(video[i].tobytes())
    proc.stdin.close()
    proc.wait()


def compute_psnr(original: torch.Tensor,
                 reconstructed: torch.Tensor) -> float:
    mse = ((original - reconstructed) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(4.0 / mse)


def main():
    device = torch.device("cuda:0")

    model_base = "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/e8c2aaa66fe3742a32c11a6766aecbf07c56e773"
    vae_path = os.path.join(model_base, "vae")

    video_dir = "data/hunyuan_overfit/videos"
    output_dir = "data/hunyuan_overfit/vae_roundtrip"
    os.makedirs(output_dir, exist_ok=True)

    video_file = "yYcK4nANZz4-Scene-027.mp4"
    video_path = os.path.join(video_dir, video_file)

    # 77 frames -> 20 latent frames (forces temporal tiling with default
    # tile_sample_min_num_frames=16)
    max_frames = 77

    print(f"Loading video: {video_path}")
    video = load_video_opencv(video_path, max_frames=max_frames)
    print(f"Video shape: {video.shape}")

    # Load VAE directly
    print("Loading HunyuanVideo VAE...")
    from fastvideo.configs.models.vaes import HunyuanVAEConfig
    from fastvideo.models.vaes.hunyuanvae import AutoencoderKLHunyuanVideo

    vae_config = HunyuanVAEConfig()
    vae_config.load_encoder = True
    vae_config.load_decoder = True

    vae = AutoencoderKLHunyuanVideo(vae_config)
    print("Loading weights...")

    sf_files = glob.glob(os.path.join(vae_path, "*.safetensors"))
    loaded = {}
    for sf_file in sf_files:
        loaded.update(safetensors_load_file(sf_file))
    vae.load_state_dict(loaded, strict=False)
    vae = vae.to(device=device, dtype=torch.float16).eval()
    vae.use_parallel_tiling = False  # no SP group in single-GPU test

    print(f"Default blend_num_frames={vae.blend_num_frames} "
          f"(pixel space, should be latent-aware)")
    print(f"Correct blend_num_frames="
          f"{(vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames) // vae.temporal_compression_ratio}"
          f" (latent space)")
    print(f"VAE loaded. Parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")

    video_gpu = video.to(device=device, dtype=torch.float16)

    save_video(video, os.path.join(output_dir, "original.mp4"))

    def check_nans(t: torch.Tensor, label: str) -> None:
        n = t.isnan().sum().item()
        inf = t.isinf().sum().item()
        if n > 0 or inf > 0:
            print(f"  WARNING: {label} has {n} NaN, {inf} Inf "
                  f"out of {t.numel()} elements")

    # === Test 1: No tiling ===
    print("\n=== Test 1: No tiling ===")
    vae.use_tiling = False
    vae.use_temporal_tiling = False

    with torch.no_grad():
        latents_no_tile = vae.encode(video_gpu)
        latent_mean = latents_no_tile.mean
        check_nans(latent_mean, "latent (no tile)")
        print(f"Latent shape (no tile): {latent_mean.shape}")
        decoded_no_tile = vae.decode(latent_mean)
        check_nans(decoded_no_tile, "decoded (no tile)")
        print(f"Decoded shape (no tile): {decoded_no_tile.shape}")

    decoded_no_tile = decoded_no_tile[:, :, :max_frames].float()
    psnr = compute_psnr(video_gpu.float(), decoded_no_tile)
    print(f"PSNR (no tiling): {psnr:.2f} dB")
    save_video(
        decoded_no_tile.cpu(),
        os.path.join(output_dir, "reconstructed_no_tiling.mp4"),
    )

    # === Test 2: Spatial tiling only ===
    print("\n=== Test 2: Spatial tiling only ===")
    vae.use_tiling = True
    vae.use_temporal_tiling = False

    with torch.no_grad():
        latents_spatial = vae.encode(video_gpu)
        latent_mean_s = latents_spatial.mean
        print(f"Latent shape (spatial tile): {latent_mean_s.shape}")
        decoded_spatial = vae.decode(latent_mean_s)
        print(f"Decoded shape (spatial tile): {decoded_spatial.shape}")

    decoded_spatial = decoded_spatial[:, :, :max_frames].float()
    psnr_s = compute_psnr(video_gpu.float(), decoded_spatial)
    print(f"PSNR (spatial tiling): {psnr_s:.2f} dB")
    save_video(
        decoded_spatial.cpu(),
        os.path.join(output_dir, "reconstructed_spatial_tiling.mp4"),
    )

    diff_latent = (latent_mean - latent_mean_s).abs()
    print(f"Latent diff (no tile vs spatial): "
          f"max={diff_latent.max():.6f}, mean={diff_latent.mean():.6f}")

    # === Test 3: Spatial + temporal tiling (default) ===
    print("\n=== Test 3: Spatial + temporal tiling ===")
    vae.use_tiling = True
    vae.use_temporal_tiling = True

    with torch.no_grad():
        latents_tiled = vae.encode(video_gpu)
        latent_mean_t = latents_tiled.mean
        print(f"Latent shape (full tile): {latent_mean_t.shape}")
        decoded_tiled = vae.decode(latent_mean_t)
        print(f"Decoded shape (full tile): {decoded_tiled.shape}")

    decoded_tiled = decoded_tiled[:, :, :max_frames].float()
    psnr_t = compute_psnr(video_gpu.float(), decoded_tiled)
    print(f"PSNR (full tiling): {psnr_t:.2f} dB")
    save_video(
        decoded_tiled.cpu(),
        os.path.join(output_dir, "reconstructed_full_tiling.mp4"),
    )

    diff_latent_t = (latent_mean - latent_mean_t).abs()
    print(f"Latent diff (no tile vs full tile): "
          f"max={diff_latent_t.max():.6f}, mean={diff_latent_t.mean():.6f}")

    # === Test 4: Temporal tiling re-test (after code fix) ===
    print("\n=== Test 4: Re-test temporal tiling (code fix applied) ===")
    vae.use_tiling = True
    vae.use_temporal_tiling = True
    # blend_num_frames stays at default (4) — the fix is in tiled_encode
    vae.blend_num_frames = (
        vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames
    )
    print(f"blend_num_frames={vae.blend_num_frames} (pixel space, "
          f"tiled_encode now converts to latent internally)")

    with torch.no_grad():
        latents_fixed = vae.encode(video_gpu)
        latent_mean_f = latents_fixed.mean
        print(f"Latent shape (fixed tile): {latent_mean_f.shape}")
        decoded_fixed = vae.decode(latent_mean_f)
        print(f"Decoded shape (fixed tile): {decoded_fixed.shape}")

    decoded_fixed = decoded_fixed[:, :, :max_frames].float()
    psnr_f = compute_psnr(video_gpu.float(), decoded_fixed)
    print(f"PSNR (fixed tiling): {psnr_f:.2f} dB")
    save_video(
        decoded_fixed.cpu(),
        os.path.join(output_dir, "reconstructed_fixed_tiling.mp4"),
    )

    diff_latent_f = (latent_mean_s - latent_mean_f).abs()
    print(f"Latent diff (spatial vs fixed tile): "
          f"max={diff_latent_f.max():.6f}, mean={diff_latent_f.mean():.6f}")

    # === Per-frame PSNR analysis ===
    print("\n=== Per-frame PSNR: spatial vs full tiling ===")
    for t in range(max_frames):
        orig_frame = video_gpu[:, :, t:t+1].float()
        spatial_frame = decoded_spatial[:, :, t:t+1]
        tiled_frame = decoded_tiled[:, :, t:t+1]
        psnr_s_f = compute_psnr(orig_frame, spatial_frame)
        psnr_t_f = compute_psnr(orig_frame, tiled_frame)
        marker = " <<<" if psnr_t_f < 30 else ""
        print(f"  frame {t:3d}: spatial={psnr_s_f:.1f}  "
              f"full={psnr_t_f:.1f}{marker}")

    # === Per-latent-frame analysis ===
    print("\n=== Per-latent-frame diff (no tile vs full tile) ===")
    for lt in range(latent_mean.shape[2]):
        frame_diff = (latent_mean[:, :, lt] - latent_mean_t[:, :, lt]).abs()
        print(f"  latent frame {lt:3d}: "
              f"max={frame_diff.max():.4f}, mean={frame_diff.mean():.4f}")

    # === Summary ===
    print("\n=== Summary ===")
    print(f"Video: {video_file}, frames={max_frames}, shape={video.shape}")
    print(f"Tiling params: min_frames={vae.tile_sample_min_num_frames}, "
          f"stride={vae.tile_sample_stride_num_frames}, "
          f"blend={vae.blend_num_frames}")
    print(f"PSNR no_tile={psnr:.2f}  spatial={psnr_s:.2f}  "
          f"full={psnr_t:.2f}")
    print(f"Output saved to: {output_dir}/")


if __name__ == "__main__":
    main()
