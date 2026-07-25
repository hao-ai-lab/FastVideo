# WanTrack causal synth stage-2 run

This directory archives the exact configuration and launch artifacts used by
the completed WanTrack causal run on 2026-07-23/24.

- Run root:
  `/mnt/lustre/vlm-k1kong/experiments/wantrack_causal/20260723_070728/wantrack-ckpt600-block3-local6-sink1-relative-gbs32`
- Initial checkpoint:
  `/mnt/lustre/vlm-k1kong/models/wantrack-synth-stage2-ckpt600-bias`
- Dataset:
  `/mnt/lustre/vlm-s4duan/data/combined_synth_parquets`
- Validation dataset:
  `/mnt/lustre/vlm-s4duan/val_examples_mixed/combined_parquet_dataset`
- Topology: 4 rack-3 nodes, 4 GPUs per node, micro-batch 2, global batch 32
- Attention: 3 latent frames per block, sink 1, local window 6,
  relativistic RoPE
- Schedule: TF 3000, CD 2000, SF 1000 optimizer steps
- Validation: every 250 steps; TF uses 30 denoising steps while CD and SF use
  4 denoising steps
- Checkpoints: every 500 optimizer steps

The TF job resumed from checkpoint 2000 after its validation configuration was
corrected to multi-step sampling. The YAML files in this directory are the
final files used by the run, including their absolute cluster paths.

`cluster/run_pipeline_node.sh` launches the resumable TF -> CD -> SF pipeline
and exports TF `student`, CD `ema`, and SF `student_ema`. It requires
`WANDB_API_KEY` to be injected through the process environment. The gallery
upload script similarly requires an in-memory `HF_TOKEN`; no credentials are
stored here.

The original Kubernetes manifest retains the historical workload label
`wantrack-causal-framewise-gbs32`. That label is stale metadata: the training
authority is the three YAML files, all of which use
`num_frames_per_block: 3`.

The SF EMA export intentionally contains only the trainable checkpoint role.
For standalone inference, four frozen track-encoder parameters were restored
from the initialization checkpoint. See `receipts/sf-full-export.json` for the
full bundle provenance and hashes.

The `gallery/` scripts produced 64 generated examples plus their 64 matching
ground-truth videos from the full SF EMA bundle.
