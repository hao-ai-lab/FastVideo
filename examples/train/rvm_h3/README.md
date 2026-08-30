# FastH3 RVM post-training

This directory is the executable experiment package for reward-aligning the
released four-step FastH3 VSA checkpoint with **Reward-based Velocity Matching
(RVM)**. It intentionally follows the published RVM video recipe instead of
introducing a new diffusion-RL objective.

## What this trains

The behavior policy is the current FastH3 quality LoRA. For every prompt it
samples `K` complete FastH3 endpoints with the exact deployed four-step VSA
sampler, scores the decoded videos, and standardizes reward within the prompt
group:

```text
advantage = 0.1 * clip((reward - group_mean) / (group_std + 1e-4), -5, 5)
```

A generated endpoint `x0` is analytically forward-noised once. The model then
receives the ordinary rectified-flow target `epsilon - x0`; the coefficient on
that regression is the signed group-relative advantage. The signed gradient is
implemented through a detached MSE target, so the logged scalar remains finite
and nonnegative while the gradient is exactly the intended RVM gradient.

H3 uses one transformer for video and audio. The initial production recipe:

- applies reward only to the **video** slice;
- computes video and audio means independently;
- optionally anchors the audio velocity to the released FastH3 field;
- obtains the reference prediction by disabling the quality LoRA in-place,
  rather than loading a second 35B model.

## Fixed model contract

Do not change these during debugging unless the run is explicitly labeled as an
ablation:

| Contract | Value |
|---|---|
| Checkpoint | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` |
| Inference jumps | base timesteps `[1000, 750, 500, 250]`, then terminal zero |
| DiT forwards | 4 |
| CFG | disabled; guidance `1.0`, conditioning dropout `0.0` |
| Attention | `VIDEO_SPARSE_ATTN_H3` |
| VSA sparsity / tile | `0.90` / `64` |
| Training geometry | `480 x 832`, `124` frames at H3's fixed 24 FPS |
| Trainable layers | LoRA on `to_q`, `to_k`, `to_v`, `to_out` |
| Frozen layer | VSA `to_gate_compress` |
| Full-run LoRA | rank `128`, alpha `64` |

H3's released input contract has a five-second minimum. Therefore the training
recipe uses 124 frames, not the 53-frame Wan setup from the RVM paper.

## Reward mixture

The production config reproduces the RVM paper's video reward family:

```text
1.5 * VideoAlign text alignment
1.0 * VideoAlign motion quality
0.1 * HPSv3 general quality
0.1 * HPSv3 prompt-conditioned top-frame percentile
0.7 * RAFT dynamic tracking
```

Dynamic tracking averages the largest five percent of optical-flow magnitudes
across four evenly spaced frame pairs. It is included because preference and
visual-quality rewards alone can converge toward clean but nearly static
videos.

All component scores and the weighted sum are logged separately. Do not judge a
run from the weighted reward alone.

## Complete workflow

Clone this branch and run the numbered scripts in order:

```bash
git clone --branch adam/h3-rvm-posttraining \
  https://github.com/Abecid/FastVideo.git
cd FastVideo

bash examples/train/rvm_h3/00_create_conda_env.sh
bash examples/train/rvm_h3/01_download_models.sh
bash examples/train/rvm_h3/02_prepare_dataset.sh
bash examples/train/rvm_h3/03_public_inference_smoke.sh
bash examples/train/rvm_h3/03_preflight_1gpu.sh
bash examples/train/rvm_h3/04_run_1gpu_smoke.sh
bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
bash examples/train/rvm_h3/06_run_8gpu_anchor_sweep.sh
bash examples/train/rvm_h3/07_run_8gpu_full.sh
```

Recovery and deployment:

```bash
bash examples/train/rvm_h3/08_resume_8gpu.sh \
  examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_full.yaml \
  outputs/rvm_h3/8gpu_full/checkpoint-90

bash examples/train/rvm_h3/09_export_lora.sh \
  examples/train/configs/rl/minimax_h3/rvm_h3_8gpu_full.yaml \
  outputs/rvm_h3/8gpu_full/checkpoint-180 \
  outputs/rvm_h3/fasth3_rvm_lora.safetensors

bash examples/train/rvm_h3/10_infer_lora.sh \
  outputs/rvm_h3/fasth3_rvm_lora.safetensors
```

Every script supports environment overrides. The shared defaults are defined in
`common.sh`.

## Dataset preparation

`prepare_prompts.py` downloads the pinned DanceGRPO/VidProM video prompt list,
deduplicates it, creates deterministic train/evaluation splits, and wraps every
prompt in H3's document format:

```text
integrated_multimodal_description: ...
overall_soundscape: N/A
non_diegetic_music: N/A
```

Audio is deliberately not optimized by the first RVM runs. `N/A` avoids asking
the model to generate arbitrary audio content while the audio field is
preserved by the anchor.

The Qwen3-VL layer-50 prompt embeddings are FP32 and large—approximately six
megabytes per prompt. Encoding the full prompt bank can require hundreds of
gigabytes. For the first node test:

```bash
RVM_MAX_TRAIN_PROMPTS=256 RVM_PREPROCESS_GPUS=1 \
  bash examples/train/rvm_h3/02_prepare_dataset.sh
```

After the smoke and sweep gates pass, unset `RVM_MAX_TRAIN_PROMPTS` and encode
the complete bank.

## Evaluation every five percent

When `method.validation.every_steps: 0`, the method computes:

```text
ceil(0.05 * training.loop.max_train_steps)
```

For the 180-update run this is every nine optimizer updates. Evaluation uses up
to 100 deterministic prompts and fixed seeds. If `data_path` points at the
encoded evaluation split, it uses that split. If no separate split is present,
it deterministically samples at most 100 training prompts and logs the fallback.

All selected videos are generated and scored. Their MP4 files are saved under:

```text
<output_dir>/validation/step-XXXXXX/
```

To avoid flooding W&B, only `log_sample_limit` videos are uploaded to the
tracker, while metrics still cover the full set of at most 100 videos.

## Configurations

| Config | Purpose |
|---|---|
| `rvm_h3_1gpu_smoke.yaml` | Four-update correctness gate, K=2, rank-16 LoRA |
| `rvm_h3_8gpu_exact.yaml` | Unanchored RVM comparison |
| `rvm_h3_8gpu_audio_anchor.yaml` | Recommended video-RVM + audio-preservation run |
| `rvm_h3_8gpu_full_anchor.yaml` | Conservative video+audio function-space anchor |
| `rvm_h3_8gpu_full.yaml` | 180-update, 23,040-rollout production candidate |

The full sample budget is:

```text
90 rollout collections
x 32 prompt groups
x 8 candidates
= 23,040 generated training videos

2 optimizer updates per collection
= 180 optimizer updates
```

## Eight-GPU topology

The default is `SP4 x DP2` on memory-rich B200/GB200 nodes. On eight 80-GB
H100/H200 GPUs, use one sequence-parallel replica:

```bash
NUM_GPUS=8 RVM_SP_SIZE=8 bash examples/train/rvm_h3/05_run_8gpu_lr_sweep.sh
```

The reward models are loaded only on the first rank of each sequence-parallel
replica. Those leaders decode and score videos, then broadcast scalar rewards
to the remaining SP ranks.

## Required experiment order

1. **Strict inference parity:** public FastH3 runner, four forwards, CFG-free.
2. **One-GPU preflight:** imports, unit tests, all reward models, config build.
3. **One-GPU training smoke:** finite forward/backward/checkpoint/export.
4. **Learning-rate bracket:** `5e-6`, `1e-5`, `2e-5`.
5. **Anchor comparison:** exact, audio anchor, full anchor.
6. **Full run:** only after a sweep variant improves held-out videos without
   audio/static-collapse regressions.

The full operational and debugging policy is in
[`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md).

## Honest validation status

The repository implementation has CPU/static/unit-test coverage. The actual H3
model, VSA CUDA kernels, VAEs, reward checkpoints, and distributed optimizer
must still pass the numbered one-GPU and eight-GPU gates on the target node.
Do not describe the training run as successful until generated validation media
has been inspected and the held-out metrics/human comparisons improve.
