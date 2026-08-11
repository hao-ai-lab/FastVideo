# Handoff: MiniMax-H3 data-free DMD2 on 8x 4xGB200 (Slurm)

State as of 2026-08-11 on branch `feat/h3-dmd2-vsa`. Everything needed for
the 32-GPU run is committed; what remains is cluster-side: fill in the
`CLUSTER:` knobs in the sbatch, swap in the real dataset, and submit.

## What this run is

Data-free DMD2 distillation (`rollout_mode: simulate` — the student builds
its own trajectory from pure noise through `[1000, 757, 522]`; no video/audio
latents consumed, text conditioning only) of the 33B dual-modality H3 DiT.
All three roles (student/teacher/critic) on dense FLASH_ATTN with FA4.
**Global batch 64** = 8 data-parallel groups (one per tray) x
`gradient_accumulation_steps: 8`.

- Config: `examples/train/configs/distribution_matching/minimax_h3/dmd2_simulate_fa4_32gpu.yaml`
- Launcher: `examples/train/slurm/dmd2_32xgb200.sbatch`
- Docs: `examples/train/configs/distribution_matching/minimax_h3/README.md`

## Everything below is already validated on a single 4xGB200 node

- The exact simulate recipe trained stably (wandb `h3-dmd2-vsa/z2z91ruf`):
  TTUR cadence correct, generator loss settles into the expected adversarial
  0.2–1.0 band, no NaN/OOM. ~10 s/it critic steps, ~20 s/it generator steps
  (every 5th) at 1344x768x124f, sp=4.
- Effective-batch-2 via gradient accumulation over two distinct videos
  (wandb `58dr571y`, 1800+ steps): both modalities' losses decline
  monotonically (video 0.128->0.092, audio 0.179->0.053 by step 1000).
  Accumulation is the batching mechanism — H3's packed pipeline is batch-1
  per forward by design (variable caption lengths cannot stack).
- Memory: three 33B roles fit at sp=4 / HSDP shard=4 with full gradient
  checkpointing and `betas: [0.0, 0.999]` (buffer-free AdamW). Do NOT try
  sp=2 or fewer than 4 GPUs per model replica: the backward working set
  alone OOMs 184 GiB cards (measured 178 GiB before optimizer states exist).
- Triton VSA kernel backward fix, bf16 autocast boundaries, validation
  offload fix, sigma-exact validation — all on the branch (see git log).

## Cluster pre-flight

1. **Code**: checkout `feat/h3-dmd2-vsa` on the shared FS; venv with the
   repo's CUDA install. `run.sh` sets PYTHONPATH to the repo root and cd's
   into it — on trays that carry `/mnt/studio8/pyoverride`, make sure it is
   NOT in PYTHONPATH or `import fastvideo` resolves to the wrong checkout.
2. **Weights**: `/mnt/models/MiniMax-H3` must be visible from compute nodes
   (all three roles `init_from` it).
3. **Data**: the config ships pointing at the 2-row smoke-test set
   (`/mnt/h3-dmd2-overfit/data_bs2`) — fine for the first submission, swap
   `training.data.data_path` (and the validation `dataset_file`) for the
   real preprocessed t2va parquet dir before a production run. Simulate mode
   only reads text conditioning from the rows.
4. **sbatch knobs**: partition/account (marked `CLUSTER:`), and if trays
   require containers, wrap the srun payload (pyxis
   `--container-image/--container-mounts=/mnt:/mnt,/dev/shm`) — hooks are
   marked in the script.
5. **W&B**: `~/.netrc` on a home visible from compute nodes, or export
   `WANDB_MODE=offline` at submit time.
6. **Login-node limits**: a previous srun attempt died to per-user thread
   exhaustion on the login node — submit with `sbatch` (batch daemon runs
   the script on the first compute node), don't launch interactive `srun`
   pipelines from the login shell.

## Launch

```bash
sbatch examples/train/slurm/dmd2_32xgb200.sbatch                 # 8 trays / 32 GPUs
sbatch --nodes=2 examples/train/slurm/dmd2_32xgb200.sbatch       # scaled smoke test
CONFIG=<other.yaml> sbatch examples/train/slurm/dmd2_32xgb200.sbatch
```

The script derives world size / rendezvous / `hsdp_replicate_dim` from the
actual allocation, so a smaller `--nodes` shrinks the DP width (and global
batch) consistently. Recommended sequence: `--nodes=1` (reproduces the
validated local run), then `--nodes=2`, then the full 8.

## What to expect

- **First boot per node**: ~10-13 min per 33B role from cold NFS (~35 min
  worst case for the trio, plus text encoder/VAEs) — then each node writes
  its weight cache to `/dev/shm/fastvideo-wcache`
  (`FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE=1` is set by the sbatch so every
  tray caches, not just tray 0). **Relaunches: ~2-4 s per role.** Note
  `/dev/shm` may be purged by node-local cleaners — a purged cache degrades
  to a full load and rewrites itself, never fails the run.
- **Step time**: per-tray compute is identical to the validated 4-GPU run;
  with accum 8, expect roughly 8 x (10 s + 2 s amortized generator surcharge)
  ≈ 90-100 s per optimizer step, ~4000 steps ≈ 4.5 days. Cut
  `max_train_steps` or validation frequency to taste.
- **Logs**: one dir per node under `examples/train/logs/<jobid>-node<k>/`;
  wandb project `h3-dmd2-vsa`, run `dmd2_simulate_fa4_32gpu` (rank 0 only).
- **Checkpointing** is ON every 500 steps (DCP; large — ~120 GiB+ per save
  for two trainable roles). Point `output_dir` somewhere with headroom or
  set `training_state_checkpointing_steps: 0` to disable.
- **Validation** every 250 steps samples every prompt in the validation
  json at the training sigmas; with the smoke-test json that is 2 videos,
  ~3 min.

## Known traps (all pre-paid on the branch)

- Don't set `FASTVIDEO_ATTENTION_BACKEND` globally — per-role backends come
  from the YAML.
- `FASTVIDEO_FA4=1` is required (sbatch sets it); without it FLASH_ATTN
  falls back to a slower path.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required (sbatch
  sets it); without it fragmentation OOMs appear after a few steps.
- If a VSA student is used later (`attention_backend: VIDEO_SPARSE_ATTN_H3`
  + `training.vsa.sparsity`): the Triton backward bugfix is on this branch —
  do not run VSA training from an older checkout. Sparsity ~0.9+ matches
  dense wall-clock in training and ~2x in validation sampling; the win is
  inference-side.
- NCCL fabric env (IB HCA / socket ifname) is cluster-specific — hook in the
  sbatch.

## If something breaks

Grab the failing node's `examples/train/logs/<jobid>-node<k>/` log and the
wandb run. The historically likely failure classes and their signatures:
OOM ledger lines (memory — check betas are [0.0, 0.999] and grad ckpt full),
`attempting to assign a gradient with device type` (a module was CPU-offloaded
mid-training — should be fixed, see validation.py), rendezvous timeout at
boot (MASTER_ADDR/port reachability between trays).
