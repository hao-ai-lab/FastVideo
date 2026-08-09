# Handoff: VSA-H3 DMD2 overfit — run directly on tray 1795370

State as of 2026-08-09. Everything is staged on the tray; only the launch is
left (my login-node srun path was blocked by per-user thread exhaustion —
running inside the tray's container sidesteps it entirely).

## The experiment

Single-sample DMD2 overfit on `c17_v2` (your prompt-rewrite v2 generation,
1344×768 / 5.2 s / with audio), parity-first: **student on the VSA-H3 backend
at sparsity 0.0** (parity-proven ≤2e-4 vs dense in unit tests) vs **teacher +
critic on FLASH_ATTN with FA4**. 3-step schedule `[1000, 757, 522]`,
`generator_update_interval: 5` (TTUR: critic every step, student every 5th),
4 GPUs, sp=4, pure-FSDP sharding (hsdp 1×4), all three models
gradient-checkpointed, batch 1, 2000 steps, ckpt+validation every 250.

## What is already staged

- **Code**: `/mnt/fv-h3ssim` on branch `h3-dmd2-vsa` (= `feat/h3-dmd2-vsa`
  @ d81f1f166 = DMD2-for-H3 + PR #1695 VSA backend + per-role backend wiring
  + wandb credential guard). Also pushed to origin as `feat/h3-dmd2-vsa`.
- **Data**: `/mnt/h3-dmd2-overfit/data/data_00000.parquet` — one preprocessed
  t2va record (video+audio VAE latents + text embedding 225×5120).
- **Validation prompts**: `/mnt/h3-dmd2-overfit/c17_validation.json` (the c17
  v2 prompt, 3-step sampling) — the YAML may still point at the Wan
  placeholder; step 2 below fixes it idempotently.
- **Config**: `examples/train/configs/distribution_matching/minimax_h3/dmd2_vsa0_overfit.yaml`
- Raw inputs (for reference): `/mnt/h3-dmd2-overfit/c17_v2.mp4`, `c17_v2_prompt.txt`.

## Run it (on the tray)

```bash
# 0. enter the persistent container (created by my earlier runs)
docker exec -it fv-dev-1795370-0 bash
# (if it's gone: ~/docker.sh style run with /home/scratch.willlin_ent -> /mnt)

# 1. clean any leftover trainer from my attempts + confirm GPUs are free
ps -eo pid,args | grep entrypoint.train | grep -v grep   # kill any PIDs listed
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ~0 MiB x4

# 2. env + config fix (idempotent)
cd /mnt/fv-h3ssim
export PYTHONPATH=/mnt/studio8/pyoverride   # routes 'import fastvideo' to THIS checkout
export FASTVIDEO_FA4=1                      # teacher/critic FLASH_ATTN -> FA4 path
export WANDB_API_KEY=8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde
sed -i "s|dataset_file: examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json|dataset_file: /mnt/h3-dmd2-overfit/c17_validation.json|" \
  examples/train/configs/distribution_matching/minimax_h3/dmd2_vsa0_overfit.yaml

# 3. launch (interactive; or wrap in setsid nohup ... & for detach)
torchrun --nproc_per_node 4 -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/distribution_matching/minimax_h3/dmd2_vsa0_overfit.yaml \
  2>&1 | tee /mnt/h3-dmd2-overfit/logs/train.log
```

wandb: project `h3-dmd2-vsa`, run `dmd2_vsa0_overfit` — the URL prints within
the first seconds of trainer init (`https://wandb.ai/...`). Logged per step:
total/generator/fake-score loss, update_student flag, step_time_sec,
vsa_sparsity.

## What to expect / what parity means

- Trio load: ~3×(33B DiT) FSDP-sharded + text encoder; first load on this
  tray is warm for the text encoder (preprocess touched it) but cold for the
  3 transformer copies — expect ~10-20 min before step 1.
- Healthy signals: both losses finite from step 1; fake-score loss drops
  fast (critic fitting one sample); generator loss moves every 5th step.
- Parity claim to check: with sparsity 0.0 the student's VSA-H3 path should
  train indistinguishably from a dense student (the sparsity-0 mask selects
  every tile; kernel parity ≤2e-4). If it diverges/NaNs where a dense run
  wouldn't, that's a VSA-H3-under-autograd bug — grab
  `/mnt/h3-dmd2-overfit/logs/train.log` and the wandb run.
- Next knob after parity holds: `training.vsa.sparsity` (e.g. 0.5 → 0.875
  → 0.9) in the YAML — student-only; teacher/critic stay dense by
  construction (`models.*.attention_backend`).

## Known traps (all pre-paid)

- `PYTHONPATH=/mnt/studio8/pyoverride` is mandatory — without it,
  `import fastvideo` resolves to `/mnt/FastVideo` (the maintainer's checkout)
  via the venv's editable-install finders, and cwd `/mnt/FastVideo` shadows too.
- Don't set `FASTVIDEO_ATTENTION_BACKEND` globally — per-role backends come
  from the YAML (`models.student/teacher/critic.attention_backend`).
- The tile-buffer autograd fix is in this branch (grad paths get fresh
  buffers); if you see "modified by an inplace operation" at backward, you're
  on a stale checkout — `git -C /mnt/fv-h3ssim log --oneline -1` should show
  d81f1f16.
- Checkpoints land in `outputs/minimax_h3_dmd2_vsa0_overfit/` under the
  checkout (NFS); validation videos under the same tree per callback config.
