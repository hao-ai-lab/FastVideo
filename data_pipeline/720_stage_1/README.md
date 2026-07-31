# Stage 1 — Bidirectional 14B/720p track teacher (full pipeline)

The complete bidirectional teacher: overfit the track pathway → merge it into a pristine 14B
base → OpenVid stage-1 (frozen head) → synth stage-2 (robustness). The final teacher then seeds
the causal (Self-Forcing) student. Full rationale: `../notes/trackwan_14b_720p_teacher_merge_plan.md`.

Recipe: **d64 + bias, sparse conditioning, flow_shift 6, 720p.** Hardware: 2×4 GB200, 400G IB,
manual 2-node `torchrun` (rack0 = `hpc-rack-1-6` = NODE_RANK 0; rack1 = `hpc-rack-1-8` = NODE_RANK 1;
`MASTER_ADDR` = rack0 on both).

## Files / run order

| Step | Script | What | Where |
|---|---|---|---|
| 0 | `00_make_overfit_subset.sh` | carve a small overfit set (symlinks ~2 parquet chunks) | login/compute |
| 1 | `01_build_init.sh` | build `trackwan_14b_i2v_d64_bias_init` | compute (CPU, ~62GB RAM) |
| 2 | `02_run_overfit.sh` | overfit track pathway, head trainable (Stage A→B) | **both racks** |
| 3 | `03_export.sh` | overfit DCP → diffusers | compute (1 GPU) |
| 4 | `04_merge.sh` | graft pathway into pristine 14B base → merged init | compute (CPU, ~62GB) |
| 5 | `05_run_openvid_stage1.sh` | OpenVid stage-1, **head frozen** (~10 days) | **both racks** |
| 6 | `06_export_stage1.sh` | stage-1 DCP → diffusers | compute (1 GPU) |
| 7 | `07_run_synth_stage2.sh` | stage-2 robustness, head frozen (~1 day) → **final teacher** | **both racks** |

Configs (referenced by the scripts): `finetune_wantrack_overfit_14b_720p_d64_bias.yaml`,
`finetune_wantrack_openvid_stage1_14b_720p_d64_bias.yaml`,
`finetune_wantrack_synth_stage2_14b_720p_d64_bias.yaml`.

## Commands

```bash
# 0-1: data + init
bash data_pipeline/720_stage_1/00_make_overfit_subset.sh
bash data_pipeline/720_stage_1/01_build_init.sh                    # compute node

# 2: overfit — both racks, Stage A (fixed IDs) then Stage B (random IDs, resumes A)
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 STAGE=A bash data_pipeline/720_stage_1/02_run_overfit.sh   # rack0
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 STAGE=A bash data_pipeline/720_stage_1/02_run_overfit.sh   # rack1
#   ...when track-following is clear in validation, stop and run Stage B (bump steps as needed):
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 STAGE=B bash data_pipeline/720_stage_1/02_run_overfit.sh --training.loop.max_train_steps 1600
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 STAGE=B bash data_pipeline/720_stage_1/02_run_overfit.sh --training.loop.max_train_steps 1600

# 3-4: export + merge
bash data_pipeline/720_stage_1/03_export.sh                        # compute node
bash data_pipeline/720_stage_1/04_merge.sh                         # compute node

# 5: OpenVid stage-1 — both racks (~10 days; resume by relaunching)
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 bash data_pipeline/720_stage_1/05_run_openvid_stage1.sh   # rack0
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 bash data_pipeline/720_stage_1/05_run_openvid_stage1.sh   # rack1

# 6-7: export stage-1 + synth stage-2 -> final teacher
bash data_pipeline/720_stage_1/06_export_stage1.sh                 # compute node
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 bash data_pipeline/720_stage_1/07_run_synth_stage2.sh      # rack0
MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 bash data_pipeline/720_stage_1/07_run_synth_stage2.sh      # rack1
```

## Notes & decisions baked in

- **Topology / batch:** stage-1 and stage-2 target the **upstream recipe on 8 nodes × 4 GB200 = 32
  GPUs** — `num_gpus 32, hsdp_replicate_dim 8, hsdp_shard_dim 4, grad_accum 4 → global bs 128`.
  Stage-1 = 4,800 steps = 2.4 epochs over 259k ≈ **~12 days**; stage-2 = 600 steps ≈ **~1.5 days**.
  (On only 8 GPUs / 2 nodes this recipe is ~49 days — if you drop back, set `NNODES=2`,
  `hsdp_replicate_dim 2`, `grad_accum 2` → global bs 16, and use ~8000 stage-1 steps for a ~10-day
  half-epoch.) The overfit (steps 0–3) stays on **2 nodes** — it's ~64 clips, so 32 GPUs / bs 128
  would exceed the dataset; leave it at `NNODES=2` there.
- **Launching the 8-node stages (05, 07):** the scripts take `NNODES` (default 8) and pass a matching
  `--training.distributed.num_gpus`. But manually running one command on each of 8 nodes (NODE_RANK
  0…7) is impractical — **use a SLURM launcher** (`srun` spans all nodes with `--node-rank=$SLURM_PROCID`,
  as in `examples/train/run_slurm.sh` / `run_wan14b_held.sh`). The overfit (02) is fine to launch
  manually on 2 nodes.
- **Env knobs** live in the launch scripts: overfit = `FREEZE_HEAD=0`; stage-1/2 = `FREEZE_HEAD=1`;
  stage-2 adds `TRACK_DROP=0.5 MOTION_DROP=0.3 PMASK=0.2 MASK_CHUNK=8`. `TRACKWAN_TRACK_BIAS=1` and
  `WANTRACK_SPARSE=1 EXTRA_RANDOM=20` throughout.
- **Stage-2 data** defaults to the openvid parquets + masking (you have no synth generated). Swap
  `data_path` to a synth set if you build one; the robustness comes from the masking either way.
- **`dit_precision`:** fp32 master (default) for the real teacher stages; upstream uses bf16 — flip
  only if memory-blocked.
- **Monitor track-following, not loss** — the `track_validation` with-track vs no-track/adversarial
  deltas. Gate the overfit (step 2) on these before merging, and watch stage-1's first validation
  (step 2000) before committing the full ~10 days. Check efficiency with `python scripts/mfu_estimate.py`.
- **Not runnable end-to-end yet:** steps 4–7 chain via default paths (04 reads 03's export, 05 reads
  04's merge, etc.), so they're correct now but only *run* once the prior step's output exists.
- **Two things to smoke-test once:** `dcp_to_diffusers` on a real checkpoint (steps 3 & 6), and that
  the overfit actually learns track-following on ~64 clips (if weak: `N_CHUNKS=4` in step 0, or more steps).
