# Dreamverse on Slurm

Run Full H3 inside a one-node, four-GPU allocation. The maintained H3 examples
default to four GPUs; this is a starting configuration, not a measured minimum.
The full checkpoint supports T2VA, FL2VA, and Ref2VA. The FastH3 Preview profile
is a separate T2VA configuration.

`launch_backend.sh` checks that it is inside an `srun` step, preserves
`CUDA_VISIBLE_DEVICES`, and replaces itself with the backend process. It does
not allocate GPUs, kill existing processes, or source a personal credentials
file. The local `dreamverse-deploy` helper is not suitable for a shared Slurm
cluster because it kills processes by physical GPU and port.

## Prepare and allocate

Keep the checkout, weights, outputs, and logs on storage visible to the compute
node. Source installation is documented in the [GPU guide](../../../../docs/getting_started/installation/gpu.md).
On ARM64 GB200 use CUDA 13, a matching PyTorch build, and kernels built for
`sm_100`; the DGX Spark `sm_121` kernel image is not the GB200 image.

The repository's image workflow publishes an ARM64 GB200 variant under
`ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-sm100-latest`.
Resolve that tag to a digest for reproducible runs. If your compute nodes use
Pyxis/Enroot, pass the approved image or a prepared SquashFS file to
`srun --container-image`, with explicit mounts for your checkout and model cache.
The Dreamverse-specific Docker images are currently AMD64-only.

For the Slinky customer partition, a bounded allocation is:

```bash
salloc --account=customer --qos=normal --partition=hpc-rack-1 \
  --nodes=1 --ntasks=1 --cpus-per-task=72 --gres=gpu:nvidia_gb200:4 \
  --mem=800G --time=02:00:00 --job-name=dreamverse
srun --ntasks=1 --pty bash
```

Wait for Slurm to grant the allocation before entering the compute step. A
successful SSH login does not grant GPU resources. Inspect pending capacity
with `squeue -u "$USER" --start`; do not attach to another user's job.

The checkpoint includes duplicate release layouts. Download the diffusers
components needed by both base and reference pipelines, rather than the whole
repository (about 210 GB versus about 498 GB at revision
`42ed227ee7df40d41602854ae760620d6eb651fe`):

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include model_index.json --include modular_model_index.json \
  --include 'audio_scheduler/*' --include 'audio_vae/*' \
  --include 'processor/*' --include 'scheduler/*' \
  --include 'text_encoder/*' --include 'tokenizer/*' \
  --include 'transformer/*' --include 'transformer_ref/*' --include 'vae/*' \
  --local-dir /path/to/models/MiniMax-H3
```

The GPU environment needs `fastvideo[dreamverse]`, the Dreamverse workspace
package, and FFmpeg with H.264/AAC encoders. In a prepared FastVideo image,
install the checked-out code and its Dreamverse dependencies in that image's
Python environment. Keep its matching CUDA/PyTorch/kernel stack intact.

## Start and connect

From the checked-out repository inside the allocated step:

```bash
export DREAMVERSE_PYTHON=/path/to/environment/bin/python
export DREAMVERSE_MODEL_PATH=/path/to/models/MiniMax-H3
export FASTVIDEO_DREAMVERSE_HOME=/path/to/persistent/dreamverse-state
bash apps/dreamverse/scripts/slurm/launch_backend.sh
```

The default backend binds port 8009 on the private compute node. Connect through
the login node from your laptop, replacing `COMPUTE_NODE_IP` with the allocated
node's `NodeAddr` from `scontrol show node`:

```bash
ssh -N -L 8009:COMPUTE_NODE_IP:8009 USER@LOGIN_NODE
```

In another laptop terminal, run the frontend from your local checkout:

```bash
cd apps/dreamverse/web
BACKEND_HOST=127.0.0.1 BACKEND_PORT=8009 npm run dev
```

Open `http://localhost:5299`. `/healthz` reports the server process; `/readyz`
reports model readiness. Full H3 loads and generates more slowly than the
Preview adapter. Keep prompt enhancement disabled in the UI unless the
runtime has the selected provider's credentials.

## Verify and stop

Check all three modes with small, valid user-owned assets. Capture the selected
mode and assets, WebSocket errors or completion events, the generated video and
audio, and GPU memory usage. Also verify actionable validation errors and
backward compatibility with clients that omit `generation_mode`.

Use the frontend Playwright instructions in the
[Dreamverse development guide](../../../../docs/contributing/dreamverse-development.md)
against the forwarded backend. A mock-server demo validates UI and protocol
behavior; it is not evidence of GPU generation.

Stop the backend with Ctrl-C, exit the compute step, and release your allocation.
For a detached allocation, use `scancel YOUR_JOB_ID`. Cancel a pending demo job
when it is no longer needed; do not leave an unattended reservation queued.
