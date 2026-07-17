# OpenVid causal A12-A15 plan (NOT STARTED)

Prepared from commit `30ada30e4c6b05aa68cd1eb8940a34d149457147`. This directory contains configuration only;
no training command was launched.

Data source: `/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset` (4494 parquet files, about 5.5 TiB). It is owned by
`vlm-s4duan`; filesystem read permission exists, but obtain the owner's consent
before launching and coordinate I/O. Do not write into that directory.

The opt-in streaming loader projects only the 15 T2V columns, reads each
assigned row group sequentially, and stores its JSON manifest at
`/mnt/lustre/vlm-k1kong/dataset-index/openvid/streaming-t2v-v2.json` in user-owned Lustre. It uses zero DataLoader workers and never
writes a cache or index into the shared source tree.

All stages use 4 GPUs, microbatch 2/rank, gradient accumulation 8, hence global
batch = 2 * 4 * 8 = 64. `dataloader_num_workers=0` limits shared-memory and
Lustre prefetch pressure.

All four conditions use exactly 21 latent frames / 81 raw frames in TF, CD,
SF, and validation. Chunk-3 conditions therefore use seven identical
three-latent blocks. A15 is length-matched and uses framewise blocks.

A15 "framewise" means `num_frames_per_block=1` on every causal role, plus
`method.chunk_size=1` in TF and SF. Causal CD has no independent-frame timestep
option: it samples one t/t_next pair and broadcasts it over T, so A15 CD is
framewise causal attention but not framewise diffusion-time sampling.

The requested LR/betas apply to each stage's main optimizer: 2e-6 and
(0.0, 0.999). SF critic keeps the proven DMD value 4e-7 with (0.0, 0.999).

To launch one condition later:

    export WANDB_API_KEY=...
    bash /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15/scripts/run_condition.sh A12 /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15/A12 29820

To run all four sequentially on one 4-GPU node:

    export WANDB_API_KEY=...
    bash /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15/scripts/run_all_sequential.sh /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15

Online W&B is the default. For an intentionally offline launch, omit the key
and set `WANDB_MODE=offline`. Training checkpoints still resume from `latest`,
but W&B does not merge separate offline process restarts into one run; sync the
resulting offline runs individually later.

To validate all three configs for one condition without starting training,
creating W&B state, or requiring a key:

    PREFLIGHT_ONLY=1 bash /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15/scripts/run_condition.sh A12 /mnt/lustre/vlm-k1kong/experiments/openvid_causal_a12_a15/20260717_014659_openvid_a12_a15/A12 29820

`PREFLIGHT_ONLY=1` is also supported by `run_all_sequential.sh`; it validates
all twelve configs and exits before queue state or checkpoint checks.
