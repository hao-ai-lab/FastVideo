# FastH3 RVM post-training

This directory contains the executable experiment package for post-training the
released four-step FastH3 VSA checkpoint with **Reward-based Velocity Matching
(RVM)**. The production path follows the published RVM video recipe rather than
introducing a new diffusion-RL objective.

> **Production method:** use `RVMFaithfulMethod` or
> `RVMWithLocalMetricsMethod`. The older `RVMMethod` is retained only to
> reproduce the first runtime pilots; it used per-group reward standard
> deviations and deployment-grid regression times and is not the default for
> new scientific runs.

## Method

For prompt group `g` and candidate `i`, FastH3 generates an endpoint `x0[g,i]`
with its exact four-forward VSA sampler. The five public rewards are combined,
centered within each prompt group, and divided by one standard deviation
computed over the whole rollout collection:

```text
centered[g,i] = reward[g,i] - mean_i reward[g,i]
global_std = std_{g,i}(reward[g,i])
advantage[g,i] = 0.1 * clip(centered[g,i] / (global_std + 1e-4), -5, 5)
```

One continuous RVM regression time is then sampled:

```text
t ~ Uniform(0, 1)
x_t = (1 - sigma(t)) * x0 + sigma(t) * epsilon
velocity_target = epsilon - x0
```

The video velocity receives the signed RVM update. Audio receives no visual
reward term. Optional audio/video reference anchors are isolated ablations and
use the released FastH3 field obtained by temporarily disabling the new quality
LoRA; no second 35B model is loaded.

The signed gradient is implemented through a detached nonnegative MSE target.
This produces the exact RVM gradient while avoiding an unbounded negative scalar
loss for negative advantages.

FastH3-specific details:

- behavior rollouts always use base steps `1000,750,500,250` plus terminal zero;
- video and audio use their native scheduler shifts, 12 and 3;
- continuous time applies to the analytic RVM regression state, not the rollout;
- VSA metadata uses the nearest released deployment bin for a continuous state;
- video and audio reductions are computed independently;
- only LoRA parameters on `to_q`, `to_k`, `to_v`, and `to_out` are trained;
- the VSA compression gate remains frozen.

## Reward recipe

```text
1.5 * VideoAlign text alignment
1.0 * VideoAlign motion quality
0.1 * HPSv3 general quality
0.1 * HPSv3 prompt-conditioned top-30%-frame score
0.7 * RAFT dynamic tracking
```

Dynamic tracking is the published clipped reward. The same RAFT pass also logs
non-optimized diagnostics:

```text
dynamic_tracking_raw
dynamic_tracking_saturation
```

These distinguish useful motion variation from a reward already pinned at its
ceiling. All component values and global standard deviations are logged.

## Fixed model contract

| Contract | Value |
|---|---|
| Checkpoint | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` |
| Rollout forwards | 4 |
| Base steps | `1000,750,500,250` |
| CFG | guidance `1.0`; conditioning dropout `0.0` |
| Attention | `VIDEO_SPARSE_ATTN_H3` |
| VSA sparsity | `0.90` |
| Production geometry | `480x832`, 124 frames, 24 FPS |
| Full LoRA | rank 128, alpha 64 |
| Trainable modules | `to_q,to_k,to_v,to_out` |
| Training time | continuous `Uniform(0,1)` |
| Reward normalization | per-prompt center, batch-global std |

A five-second H3 chunk requires 124 frames. The one-H100 topology smoke uses a
clearly labeled compact config only to exercise the runtime path.

## Portable execution architecture

All reusable experiment logic lives outside provider-specific launchers:

| Layer | Source of truth |
|---|---|
| Python/runtime dependencies | `00_install_current_env.sh` |
| Conda bootstrap | `00_create_conda_env.sh` |
| Model/reward downloads | `01_download_models.sh` |
| Prompt download, split, and H3 encoding | `02_prepare_dataset.sh` |
| Hyperparameters and topology | `examples/train/configs/rl/minimax_h3/*.yaml` |
| One-/four-GPU smoke orchestration | `12_run_portable_smoke.sh` |
| 8/16-GPU production campaign | numbered node scripts plus `common.sh` |
| Modal transport | `modal_h3_rvm.py` |

`modal_h3_rvm.py` is deliberately a thin test-only wrapper. It only allocates
one or four GPUs, mounts persistent volumes, clones an exact Git ref, and calls
`12_run_portable_smoke.sh`. It contains no reward definitions, training
hyperparameters, dataset processing, runtime source patching, or training-loop
implementation. The Modal file can be removed before final merge without
affecting the custom-node workflow.

The portable smoke runner can also be called directly on another cloud service:

```bash
RVM_SKIP_CONDA=1 \
RVM_ARTIFACT_ROOT=/persistent/rvm_h3 \
RVM_SMOKE_RUN_ROOT=/persistent/runs \
RVM_SMOKE_GPUS=4 \
RVM_SMOKE_MODE=pilot \
RVM_SMOKE_MAX_STEPS=10 \
RVM_SMOKE_EVAL_PROMPTS=8 \
RVM_SMOKE_MAX_TRAIN_PROMPTS=64 \
  bash examples/train/rvm_h3/12_run_portable_smoke.sh
```

## Setup

```bash
git clone --branch adam/h3-rvm-posttraining \
  https://github.com/Abecid/FastVideo.git
cd FastVideo

bash examples/train/rvm_h3/00_create_conda_env.sh
bash examples/train/rvm_h3/01_download_models.sh
RVM_MAX_TRAIN_PROMPTS=4096 \
  bash examples/train/rvm_h3/02_prepare_dataset.sh
bash examples/train/rvm_h3/03_public_inference_smoke.sh
bash examples/train/rvm_h3/03_preflight_1gpu.sh
bash examples/train/rvm_h3/04_run_1gpu_smoke.sh
```

For a container that already owns its Python environment:

```bash
RVM_SKIP_CONDA=1 \
  bash examples/train/rvm_h3/00_install_current_env.sh
```

The Qwen3-VL layer-50 prompt embeddings are large. Use 4,096 prompts for the
medium campaign, then encode the complete prompt bank before the final run.
Preparation downloads the pinned DanceGRPO/VidProM prompt file, creates a
deterministic held-out split, and wraps prompts as H3 documents with audio fields
set to `N/A`.

## Optional Modal testing

Modal is used only for early one-/four-GPU correctness and short reward pilots:

```bash
# One strict H100 compact optimizer smoke.
H3_RVM_MODAL_GPU_1='H100!' \
H3_RVM_MODAL_GPU_4='H100!:4' \
H3_RVM_MODAL_SECRETS='hf-adamlee00,wandb-adamlee00' \
  modal run examples/train/rvm_h3/modal_h3_rvm.py \
    --gpus 1 \
    --mode smoke \
    --max-steps 1 \
    --eval-prompts 1

# Four strict H100 production-geometry pilot.
H3_RVM_MODAL_GPU_1='H100!' \
H3_RVM_MODAL_GPU_4='H100!:4' \
H3_RVM_MODAL_SECRETS='hf-adamlee00,wandb-adamlee00' \
  modal run examples/train/rvm_h3/modal_h3_rvm.py \
    --gpus 4 \
    --mode pilot \
    --max-steps 10 \
    --eval-prompts 8
```

Do not use Modal for the 8/16-H100 production campaign.

## Custom-node 8/16-GPU campaign

The production campaign runs on the custom node. `common.sh` derives the
data-parallel replica count from `NUM_GPUS / RVM_SP_SIZE`.

Eight H100s:

```bash
export NUM_GPUS=8
export RVM_SP_SIZE=4

bash examples/train/rvm_h3/05_run_8gpu_topology_smoke.sh
bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
RVM_SELECTED_LR=<winner> \
  bash examples/train/rvm_h3/06_run_8gpu_anchor_sweep.sh
RVM_SELECTED_LR=<winner> \
RVM_SCALEUP_CONFIG=<winning-config> \
  bash examples/train/rvm_h3/07_run_8gpu_scaleup_pilot.sh
```

Sixteen H100s use the same provider-agnostic scripts:

```bash
export NUM_GPUS=16
export RVM_SP_SIZE=4
```

This resolves to `SP4 x DP4`. The 16-GPU path is supported by the topology
arguments but must pass its own two-update smoke before a long run; it has not
yet been validated by the completed four-GPU pilots.

The full 180-update/23,040-endpoint run is deliberately gated:

```bash
RVM_FULL_APPROVED=1 \
RVM_SELECTED_LR=<winner> \
RVM_VIDEO_ANCHOR_BETA=<winner> \
RVM_AUDIO_ANCHOR_BETA=<winner> \
  bash examples/train/rvm_h3/07_run_8gpu_full.sh
```

Do not set `RVM_FULL_APPROVED=1` until topology, LR, anchor, held-out reward,
qualitative, audio, resume, and export gates pass.

## Evaluation

Production evaluation runs every:

```text
ceil(0.05 * max_optimizer_steps)
```

The full 180-step run therefore evaluates every nine steps on up to 100 fixed
prompts and seeds. A separate encoded eval split is used when present; otherwise
the method deterministically samples at most 100 training prompts and records
the fallback.

Track:

- aggregate and every component reward;
- global aggregate/component reward standard deviations;
- prompt-group standard deviation and zero-variance ratio;
- advantage magnitude and clipping ratio;
- gradient norm and clipping indicator;
- continuous training-time statistics;
- DT raw value and saturation fraction;
- generated validation videos and checkpoints.

Training rollout reward and held-out reward are not interchangeable. Select
checkpoints from paired fixed-prompt evaluation plus inspected full videos, not
from the last training batch.

## Source reproducibility

Every numbered training script runs `verify_clean_source.py` before launching.
It fails when tracked files differ from the checked-out Git commit and records
`HEAD` and `HEAD^{tree}`. Never use `RVM_ALLOW_DIRTY_SOURCE=1` for a reported
experiment.

## Documentation

- [`00_RVM_FIDELITY_AND_SCALEUP.md`](00_RVM_FIDELITY_AND_SCALEUP.md): exact
  method and scale-up contract.
- [`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md): operational debugging and tuning
  policy.
- [`PR3_MODAL_PROGRESS_REPORT.md`](PR3_MODAL_PROGRESS_REPORT.md): completed
  pre-fidelity runtime pilots and their limitations.

## Validation boundary

Earlier 1×H100 and 4×H100 runs validated model loading, full-geometry VSA
rollouts, all reward models, LoRA backward, Adam, checkpointing, and validation.
The batch-global normalization and continuous-time RVM path still requires a
fresh one-GPU smoke and exact 8-GPU topology gate. The refactored provider-
agnostic smoke runner and thin Modal wrapper have been syntax-checked, but have
not yet been executed on Modal or the custom node.
