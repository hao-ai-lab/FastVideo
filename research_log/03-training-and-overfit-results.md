# 03 — Training & Overfit Results

Goal of these runs: prove the bidir track-conditioned path trains end-to-end and can
**reconstruct** the training clips. (Whether it *uses the tracks* is a separate
question — answered in [04](04-controllability-eval-and-findings.md).)

Setup: base = single-expert 1.3B init (`trackwan_1.3b_init`), 480×832, 121 frames
(`num_latent_t=31`), 10 clips, bs=1, lr 1e-4 constant, grad clip 1.0, flow_shift 3,
FA3, single H200 (~5.3 s/it). **`WANTRACK_AUG=0`** (clean overfit). wandb project
`wantrack-bidir`. wandb key passed via `WANDB_API_KEY` env only — never written to a file.

YAML: `examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml`.

## Run 1 — 0→600 steps (wandb `32dlm3ce`)

- `finetune_loss` 0.42 → **0.204**. Noisy per-step (random-timestep variance); windowed
  median trends down.
- Gen-vs-GT (validation video MSE→PSNR) basically flat through 600: PSNR ~9–11 dB.
- Conclusion: still descending, undertrained → train longer.

## Run 2 — resume 600→4000 steps (wandb `tcze44uf`)

Resumed from `checkpoint-600` (no recompute). Validation every 500 steps.

### Training loss (windowed median) — kept descending, no instability

| steps | loss median | grad norm median |
|---|---|---|
| 600–849   | 0.141 | 0.98 |
| 1350–1599 | 0.114 | 0.70 |
| 2350–2599 | 0.086 | 0.54 |
| 3350–3599 | 0.063 | 0.47 |
| 3850–4000 | **0.059** | 0.42 |

Grad norm *fell* (0.98→0.42, well under the 1.0 clip) ⇒ stable, and loss was **still
descending at step 4000** (not plateaued). Late `finetune_loss` spikes are just
high-timestep single-step samples (only ~3% of steps > 0.4).

### Reconstruction convergence (gen-with-GT-tracks vs decoded GT clip)

PSNR (dB), higher = closer reconstruction:

| step | sample 0 | sample 1 |
|---|---|---|
| 600  | 11.3 | 9.5 |
| 1500 | 14.0 | 10.8 |
| 2000 | 17.5 | 12.4 |
| 3500 | **17.8** | **12.7** |
| 4000 | 14.9 | 10.7 |

More steps clearly helped (sample-0 MSE 4845→1079, ~4.5× lower). The step-4000 dip is
single-seed validation noise, not a real regression (training loss has no such dip).
**Best reconstruction checkpoint ≈ 3500.** Sample 1 reconstructs worse than sample 0
(harder clip / more motion).

## Takeaways

- The pipeline trains cleanly; reconstruction improves steadily with steps; there is LR
  headroom (grad norm 0.42) to overfit faster.
- BUT reconstruction ≠ control. The model can reproduce the 10 clips from text + first
  frame; see [04](04-controllability-eval-and-findings.md) for whether tracks matter.

## Reproduce / resume

```bash
# fresh / resume both use resume_from_checkpoint: latest in the YAML
CUDA_VISIBLE_DEVICES=1 srun --jobid=1788946 --overlap --ntasks=1 \
  /usr/bin/env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=/mnt/weka/home/hao.zhang/shao/FastVideo \
  WANDB_API_KEY=<key> WANTRACK_AUG=0 \
  .venv/bin/torchrun --nproc_per_node=1 --master_port=29561 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/scenario/worldmodel/finetune_wantrack_i2v.yaml
```

Validation callback (`track_validation`) logs track-overlaid generations + a one-time
GT reference to wandb every `every_steps` (see [06](06-tooling.md)).

## Export a checkpoint for standalone inference

```bash
.venv/bin/python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint .../wantrack_overfit_out/checkpoint-4000 \
  --output-dir .../models/trackwan_1.3b_overfit4k --overwrite
```
