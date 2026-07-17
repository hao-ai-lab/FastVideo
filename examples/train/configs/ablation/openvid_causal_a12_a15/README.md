# OpenVid causal A12-A15 plan (NOT STARTED)

Prepared from commit `5ce164972619c408245267d11d4495145ba2dbfe`. This directory contains configuration only;
no training command was launched.

Data source: `/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset` (4494 parquet files, about 5.5 TiB). It is owned by
`vlm-s4duan`; filesystem read permission exists, but obtain the owner's consent
before launching and coordinate I/O. Do not write into that directory.

All stages use 4 GPUs, microbatch 2/rank, gradient accumulation 8, hence global
batch = 2 * 4 * 8 = 64. `dataloader_num_workers=0` limits shared-memory and
Lustre prefetch pressure.

A15 "framewise" means `num_frames_per_block=1` on every causal role, plus
`method.chunk_size=1` in TF and SF. Causal CD has no independent-frame timestep
option: it samples one t/t_next pair and broadcasts it over T, so A15 CD is
framewise causal attention but not framewise diffusion-time sampling.

The requested LR/betas apply to each stage's main optimizer: 2e-6 and
(0.0, 0.999). SF critic keeps the proven DMD value 4e-7 with (0.0, 0.999).

To launch one condition later:

    export WANDB_API_KEY=...
    bash /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/prepared_20260717/scripts/run_condition.sh A12 /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/prepared_20260717/A12 29820

To run all four sequentially on one 4-GPU node:

    export WANDB_API_KEY=...
    bash /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/prepared_20260717/scripts/run_all_sequential.sh /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/prepared_20260717
