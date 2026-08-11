# MiniMax-H3 training configs (DMD2 distillation + SFT overfit)

Configs for training the 33B dual-modality (video+audio) MiniMax-H3 DiT with
the modular trainer (`fastvideo/train/`), validated on 4x GB200 (192 GB,
aarch64). SFT overfit configs live in
[`../../fine_tuning/minimax_h3/`](../../fine_tuning/minimax_h3/); the 8-tray
Slurm launch is [`../../../slurm/dmd2_32xgb200.sbatch`](../../../slurm/dmd2_32xgb200.sbatch).

## Configs

| Config | What it is |
|---|---|
| `dmd2_vsa0_overfit.yaml` | Single-sample DMD2, student on VIDEO_SPARSE_ATTN_H3 at sparsity 0 (parity vs dense), teacher/critic on FLASH_ATTN/FA4, `rollout_mode: data_latent` |
| `dmd2_simulate_fa4_overfit.yaml` | **Data-free** DMD2 (`rollout_mode: simulate`): student rolls out from pure noise through `dmd_denoising_steps`; parquet VAE latents ignored, only text conditioning used. All roles dense FA4 |
| `dmd2_simulate_fa4_32gpu.yaml` | The simulate recipe scaled to 8 trays x 4 GPUs, global batch 64 |
| `../../fine_tuning/minimax_h3/sft_*.yaml` | SFT overfit family: FA4 dense control, VSA at sparsity 0 / 0.8 / 0.9 / 0.95 / 0.97, and `sft_fa4_bs2_overfit.yaml` (effective batch 2 over two videos) |

## Launch

```bash
# single node (4x GB200)
FASTVIDEO_FA4=1 \
FASTVIDEO_WEIGHT_SHARD_CACHE=/dev/shm/fastvideo-wcache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash examples/train/run.sh examples/train/configs/distribution_matching/minimax_h3/dmd2_simulate_fa4_overfit.yaml

# 8 trays / 32 GPUs from a Slurm login node (global batch 64)
sbatch examples/train/slurm/dmd2_32xgb200.sbatch
```

`FASTVIDEO_FA4=1` selects the FA4 path inside the FLASH_ATTN backend and is
required for the dense roles. Per-role backends come from
`models.<role>.attention_backend` — do not set `FASTVIDEO_ATTENTION_BACKEND`
globally.

## Topology and batch semantics

- `sp_size: 4` — one packed `[text|cond|audio|video]` document is
  sequence-parallel across the 4 GPUs of a node; HSDP mesh
  `(replicate=nodes, shard=4)` keeps parameters sharded only inside a node
  (~145 GiB/rank for DMD2's three 33B roles with full gradient checkpointing).
- **The H3 packed pipeline is batch-1 per forward by design** (documents with
  different caption lengths cannot stack). Batching composes through
  `training.loop.gradient_accumulation_steps`, which the trainer loops
  natively — global batch = DP groups x accumulation rounds. The 32-GPU
  config's global 64 = 8 DP groups x 8 rounds; validated locally as
  effective-batch-2 (`sft_fa4_bs2_overfit.yaml`, two videos, both losses
  declining, ~10 s/step = 2 micro fwd+bwd).
- DMD2 TTUR: critic every step, generator every
  `generator_update_interval`-th. `rollout_mode: simulate` costs ~4 dense
  forwards per critic step and ~7 + 2 backwards per generator step
  (~10/20 s at 1344x768x124f on GB200).

## Measured reference points (1344x768, 124 frames, sp=4)

- SFT step time: FA4 dense 4.9 s/it; VSA 90% 4.8; 95% 4.49; 97% 4.33;
  VSA-at-density-1 11.2 (VSA machinery overhead — attention is only ~1/3 of
  step FLOPs, so sparsity is mostly an inference win: 3-step validation
  sample ~120 s dense vs ~59 s at 95%).
- 2x GB200 does NOT fit this model/sequence without CPU offload: the first
  backward alone allocates ~178 GiB at sp=2 (params 31 + grads 31 + ~116
  working set) before optimizer states exist.
- `betas: [0.0, 0.999]` auto-selects a buffer-free AdamW
  (`fastvideo/train/utils/optimizer.py`), saving one full set of optimizer
  state (~15.4 GiB/rank/model) — required for DMD2's two trainable roles.

## Weight shard cache

`FASTVIDEO_WEIGHT_SHARD_CACHE=<tmpfs dir>` caches each rank's post-shard
DTensor chunks after the first full load
(`fastvideo/models/loader/shard_cache.py`); relaunches rebuild each 33B role
in ~2-4 s instead of ~10 min of cold-NFS reads (DMD2's three roles share one
entry). On multi-node runs with node-local cache dirs set
`FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE=1` (the sbatch does) so every node
writes its own copy. Any validation failure degrades to the normal full load.

## Data

Single/two-sample overfit parquets are produced by
`fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py` (one mp4
with soundtrack + prompt -> one t2va row; >=124 frames at 24 fps and an audio
track are required). The preprocessor writes exactly one
`data_00000.parquet` per output dir — build multi-sample sets by copying rows
into one dir as `data_00000.parquet`, `data_00001.parquet`, ... The
validation callback samples every prompt in `callbacks.validation.dataset_file`
at the method's exact training sigmas (`dmd_denoising_steps` is injected into
the validation pipeline).
