# Waypoint: FastVideo vs Diffusers Comparison on RunPod

This guide walks through running the Waypoint comparison (FastVideo + Diffusers, same prompt/seed) on a RunPod GPU instance, and how to get files to/from the server.

---

## 1. Connect to RunPod

### SSH (recommended)

```bash
# From your local machine
ssh root@194.68.245.130 -p 22100 -i ~/.ssh/id_ed25519
```

If RunPod gives you a different host/port, use:

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

### Web terminal

Enable "Web terminal" in RunPod → Connect → open a browser-based terminal.

---

## 2. Clone repo on the server (without your local changes)

On the **server** (after SSH):

```bash
cd /workspace  # or wherever you want to work
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
pip install -e .  # or: pip install -e . --ignore-installed blinker
```

---

## 3. Send your new files to the server

Because the server clone doesn’t have your local edits, copy the new scripts.

### Option A: SCP (copy single files)

From your **local** machine:

```bash
# Replace PORT and IP with your RunPod values (e.g. 22100, 194.68.245.130)
scp -P 22100 -i ~/.ssh/id_ed25519 \
  "c:/Users/satvi/Hao ai lab/FastVideo/scripts/waypoint_compare_fastvideo_vs_diffusers.py" \
  root@194.68.245.130:/workspace/FastVideo/scripts/

scp -P 22100 -i ~/.ssh/id_ed25519 \
  "c:/Users/satvi/Hao ai lab/FastVideo/scripts/run_diffusers_waypoint.py" \
  root@194.68.245.130:/workspace/FastVideo/scripts/
```

### Option B: SCP entire scripts directory

```bash
scp -P 22100 -i ~/.ssh/id_ed25519 -r \
  "c:/Users/satvi/Hao ai lab/FastVideo/scripts/"* \
  root@194.68.245.130:/workspace/FastVideo/scripts/
```

### Option C: Paste content (no SCP)

1. SSH into the server.
2. Create or edit files with `nano` or `vim`.
3. Paste the contents of `waypoint_compare_fastvideo_vs_diffusers.py` and `run_diffusers_waypoint.py`.

---

## 4. Install Diffusers (if not already installed)

On the **server**:

```bash
cd /workspace/FastVideo
pip install "diffusers>=0.36.0" "transformers>=4.57.1" einops tensordict imageio imageio-ffmpeg tqdm pytorch-msssim
```

---

## 5. (Optional) Pre-download model to avoid disk quota

If you hit disk quota:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/workspace/tmp
mkdir -p $HF_HOME/hub $TMPDIR

huggingface-cli download Overworld/Waypoint-1-Small --local-dir /workspace/models/Waypoint-1-Small --resume-download
export WAYPOINT_DIFFUSERS_MODEL_PATH=/workspace/models/Waypoint-1-Small
```

---

## 6. Run the comparison

On the **server**:

```bash
cd /workspace/FastVideo

# Full run (FastVideo + Diffusers + compare), 16 frames
python scripts/waypoint_compare_fastvideo_vs_diffusers.py

# With more frames and low-memory mode (16GB VRAM)
python scripts/waypoint_compare_fastvideo_vs_diffusers.py --num-frames 60 --low-memory

# Custom prompt and seed
python scripts/waypoint_compare_fastvideo_vs_diffusers.py \
  --prompt "A first-person view of walking through a grassy field." \
  --seed 1024 --num-frames 16
```

Outputs go to `waypoint_compare_output/`:

- `fastvideo_waypoint.mp4`
- `diffusers_waypoint.mp4`

Plus a printed comparison (pixel diff + SSIM).

---

## 7. Download results to your local machine

From your **local** machine:

```bash
scp -P 22100 -i ~/.ssh/id_ed25519 -r \
  root@194.68.245.130:/workspace/FastVideo/waypoint_compare_output \
  ./waypoint_compare_output
```

Or download only the videos:

```bash
scp -P 22100 -i ~/.ssh/id_ed25519 \
  root@194.68.245.130:/workspace/FastVideo/waypoint_compare_output/fastvideo_waypoint.mp4 \
  root@194.68.245.130:/workspace/FastVideo/waypoint_compare_output/diffusers_waypoint.mp4 \
  ./
```

---

## 8. Compare-only mode (already have both videos)

```bash
python scripts/waypoint_compare_fastvideo_vs_diffusers.py \
  --compare-only \
  --fastvideo /path/to/fastvideo.mp4 \
  --diffusers /path/to/diffusers.mp4
```

---

## Quick reference

| Action | Command |
|--------|---------|
| SSH | `ssh root@194.68.245.130 -p 22100 -i ~/.ssh/id_ed25519` |
| Upload script | `scp -P 22100 -i ~/.ssh/id_ed25519 script.py root@194.68.245.130:/workspace/FastVideo/scripts/` |
| Download folder | `scp -P 22100 -i ~/.ssh/id_ed25519 -r root@194.68.245.130:/workspace/FastVideo/waypoint_compare_output ./` |
| Run comparison | `python scripts/waypoint_compare_fastvideo_vs_diffusers.py` |

---

## Jupyter Lab

If Jupyter is running on port 8888:

1. Open `https://<your-runpod-url>:8888` in a browser.
2. Create a notebook and run:

```python
!cd /workspace/FastVideo && python scripts/waypoint_compare_fastvideo_vs_diffusers.py --num-frames 16
```

3. Use the file browser to download the MP4s from `waypoint_compare_output/`.
