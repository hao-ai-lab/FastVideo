# FastH3 RVM GPU-agent runbook

This document is the operating contract for the agent responsible for running,
debugging, and tuning FastH3 RVM on the GPU node. The objective is not novelty;
it is a reproducible quality improvement over the released four-step FastH3
checkpoint.

## 1. Method and intuition

RVM is endpoint-only reward-weighted self-distillation. The current behavior
policy generates a clean endpoint `x0`. A black-box reward assigns a scalar.
The endpoint is analytically noised at one training time and the model is
updated with its original flow-matching regression target.

For one generated sample:

```text
x_t = (1 - sigma) * x0 + sigma * epsilon
velocity_target = epsilon - x0
```

For prompt group `g`, candidate `i`:

```text
A_i = 0.1 * clip((R_i - mean_g R) / (std_g R + 1e-4), -5, 5)
```

The desired video-output gradient is:

```text
g_video = A_i * (v_video - velocity_target_video)
          + beta_video * (v_video - v_reference_video)
```

Audio receives no reward coefficient in the initial runs:

```text
g_audio = beta_audio * (v_audio - v_reference_audio)
```

The code constructs a detached target `v - g` and minimizes an ordinary
nonnegative MSE against it. Because the target is detached, the derivative with
respect to `v` is exactly `g`, including signed negative advantages. Do not
replace this with a literal negative-weighted MSE: the scalar would be
unbounded below and harder to monitor safely.

The reference field is the same released FastH3 backbone with the new quality
LoRA temporarily disabled. It does not require a second H3 checkpoint in GPU
memory.

### Why this method was chosen

The RVM paper's controlled video experiments found that endpoint velocity
matching can outperform substantially more expensive trajectory-policy methods
while preserving the original pretraining objective. It also found that reward
choice—especially an explicit motion term—matters more than small differences
among closely related velocity losses. Therefore this implementation follows
its public data/reward/loss setup as closely as H3 permits.

H3 differs from the paper's Wan model in three important ways:

1. it is already a native four-forward distilled model;
2. it jointly generates video and audio with different scheduler shifts;
3. the public checkpoint uses 90% VSA sparsity.

The code handles those differences without changing the core RVM algorithm.

## 2. Non-negotiable invariants

Never silently modify these to get a run to start:

| Invariant | Required value | Why |
|---|---:|---|
| Behavior sampling | current FastH3 policy before its attached updates | RVM is on-policy/recent-policy |
| Base steps | `1000,750,500,250` | released four-step student schedule |
| CFG | guidance `1.0`; dropout `0.0`; no negative prompt | H3 is guidance-distilled |
| Attention | `VIDEO_SPARSE_ATTN_H3` | dense is not the deployed checkpoint |
| VSA | sparsity `0.90`, tile `64` | trained deployment policy |
| Video shift | `12` | H3 scheduler contract |
| Audio shift | `3` | H3 scheduler contract |
| FPS | `24` | H3 fixed FPS |
| Frames | `124` | valid five-second H3 chunk geometry |
| Modality reductions | independent video/audio means | packed video otherwise erases audio |
| LoRA target | `to_q,to_k,to_v,to_out` | avoids changing sparse gate |
| VSA gate | frozen | prevents architecture/reward confound |
| LoRA master dtype | FP32 | prevents 1e-5 updates rounding away |
| Reward prompt | visual description only | sound fields must not enter visual TA/HPS |

Any experiment that changes one must receive a distinct run name and a written
rationale in the experiment log.

## 3. Reward stack

Production weights:

```text
videoalign_ta       1.5
videoalign_mq       1.0
hpsv3_general       0.1
hpsv3_percentile    0.1
dynamic_tracking    0.7
```

### Meaning

- **VideoAlign TA:** semantic adherence to the visual prompt.
- **VideoAlign MQ:** temporal stability and motion plausibility.
- **HPSv3 general:** frame-level broad human preference using the generic
  prompt `A high-quality image`.
- **HPSv3 percentile:** prompt-conditioned mean over the best 30% frames,
  matching the public RVM/GenRL recipe.
- **Dynamic tracking:** RAFT flow magnitude over four frame pairs, taking the
  largest 5% spatial values before averaging.

### Reward preflight

`03_preflight_1gpu.sh` must finish before training. It verifies all models load,
all outputs are finite, and the moving synthetic clip receives more dynamic
tracking reward than a matched static clip.

Before the full run, perform a 200-pair human audit containing:

- current FastH3 versus Base H3;
- two FastH3 seeds;
- moving subject versus frozen subject;
- subject motion versus pure camera pan;
- clean static shot versus flicker;
- prompt-correct versus pretty-but-wrong;
- good video with weak audio versus weaker video with good audio.

Target at least about 65% pairwise agreement for the aggregate reward. If it is
lower, fix preprocessing/weights before spending the full compute budget.

### Static-collapse diagnosis

Symptoms:

- aggregate reward rises;
- dynamic tracking, dynamic degree, or foreground flow falls;
- outputs repeat frames or become still images.

Actions, in order:

1. verify `dynamic_tracking` is actually nonzero and included with weight `0.7`;
2. verify reward media has temporal dimension `[B,C,T,H,W]`, not first frame;
3. verify RAFT is pretrained and did not silently fall back to random weights;
4. inspect VideoAlign MQ preprocessing/FPS;
5. reduce LR one bracket;
6. stop the run if held-out motion worsens despite reward growth.

Do not simply increase the motion weight without inspecting whether camera
motion is gaming it. Report foreground and global flow separately in analysis.

## 4. Data

Primary source: the pinned DanceGRPO/VidProM prompt file. Preparation is
deterministic and records a SHA256 digest in `artifacts/rvm_h3/prompts/metadata.json`.

H3 documents use:

```text
integrated_multimodal_description: <visual prompt>
overall_soundscape: N/A
non_diegetic_music: N/A
```

The training reward receives only the integrated visual description. H3 still
receives the full structured document.

### Prompt embedding storage

The preprocessor writes Qwen3-VL layer-50 embeddings as FP32. One prompt may be
roughly six megabytes, so a 50K-prompt bank may consume around 300 GB before
filesystem overhead. Never begin the full preprocessing job without checking:

```bash
df -h "$RVM_ARTIFACT_ROOT"
du -sh "$RVM_ARTIFACT_ROOT"
```

For debugging use `RVM_MAX_TRAIN_PROMPTS=256` or `1024`. Do not accidentally
publish results from the tiny smoke bank as the full run.

## 5. Execution gates

### Gate A: strict inference

Run `03_public_inference_smoke.sh` and verify:

- exactly five sigma points / four DiT forwards;
- guidance scale is 1.0;
- VSA sparsity is 0.9, tile 64;
- output video and audio are present;
- no unrequested fusion/compile profile changed numerics.

Archive the prompt, seed, environment, output, and stage timing.

### Gate B: static/reward/config preflight

Run `03_preflight_1gpu.sh`. It must pass:

- Python compilation;
- RVM gradient-sign tests;
- scheduler/video-audio shift tests;
- LoRA disable/restore tests;
- config invariants;
- reward checkpoint inference;
- actual H3 config build.

### Gate C: one-GPU optimizer smoke

Run four optimizer updates. Required checks:

1. validation at step zero generates valid media;
2. `total_loss`, component losses, rewards, advantages, and grad norm are finite;
3. zero-initialized LoRA reproduces the released checkpoint before the first
   update;
4. both positive and negative advantage paths produce nonzero gradients;
5. audio anchor loss starts at or very near zero;
6. a checkpoint saves, reloads, and exports;
7. exported LoRA loads into inference and produces the same output as the
   unexported checkpoint within numerical tolerance.

A one-GPU run is a correctness test, not evidence of a quality improvement.

### Gate D: eight-GPU distributed smoke

Before any sweep, override the selected topology and run two optimizer updates.
Verify:

- all ranks use the same prompt/noise inside each SP group;
- only SP leaders load reward models;
- scalar rewards are identical across the ranks of each SP group;
- DP groups receive different prompt groups;
- no rank enters a mismatched collective;
- checkpoint metadata contains all trainable LoRA and optimizer state.

## 6. Hyperparameter policy

### Learning rate

Run exactly this first bracket:

```text
5e-6
1e-5
2e-5
```

Use the same prompts, evaluation set, seeds, reward stack, anchor, and update
count. Select the **largest** LR satisfying all of:

- finite gradients;
- fewer than roughly 10% persistently clipped steps;
- held-out aggregate and component rewards improve;
- videos do not become static/repetitive;
- audio metrics and listening checks do not regress;
- validation quality is not merely oversaturated/sharpened.

The default is `1e-5`. Do not jump to the Wan paper's larger LR merely because
its model was smaller and less aggressively distilled.

### Anchor

Compare:

```text
exact:        beta_video=0,    beta_audio=0
video reward: beta_video=0,    beta_audio=1e-3  (recommended)
full anchor:  beta_video=1e-3, beta_audio=1e-3
```

Only increase audio beta to `1e-2` when measured audio drift persists. A larger
anchor reduces learning capacity and should not be the first response to an
unrelated reward or sampler bug.

### Group size

Production `K=8`. For systems debugging only, use `K=2`. Do not compare reward
curves across different K without noting that group-normalized advantage
statistics changed.

Zero group variance means RVM has no learning signal for that prompt. Monitor
`rvm/zero_std_group_ratio`. If it exceeds about 30%:

1. verify seeds differ among candidates;
2. verify reward batching returns one scalar per candidate;
3. inspect scorer saturation;
4. increase K or prompt diversity only after fixing implementation errors.

### LoRA rank

- smoke: 16;
- production: 128, alpha 64;
- memory fallback: 64, alpha 64.

Changing rank changes capacity and should be isolated from LR/reward changes.

### Gradient clipping

Default `1.0`. If repeated severe clipping and quality collapse occur, lower LR
first. DanceGRPO reports that reducing max grad norm can help reward collapse,
but clipping should not conceal exploding gradients caused by a sign, CFG, or
normalization bug.

## 7. Validation and checkpoint selection

Automatic interval:

```text
ceil(0.05 * max_train_steps)
```

For 180 steps: every 9 steps. Up to 100 fixed prompts and seeds are scored at
each checkpoint. The method saves all MP4s and uploads only a bounded subset.

Track at minimum:

- weighted reward;
- every reward component;
- group reward standard deviation;
- zero-variance group ratio;
- advantage mean/min/max and clipping ratio;
- video RVM loss;
- audio anchor loss;
- gradient norm;
- LR;
- repeated-frame/static-video rate;
- independent VideoAlign VQ or another non-training metric;
- audio CLAP/ASR/AV-sync on a fixed audio subset.

Do not select the last checkpoint automatically. Select the checkpoint with the
best held-out/human tradeoff before any deterioration.

Stop immediately when:

- training reward rises while held-out human preference falls;
- repeated/static rate rises by more than five percentage points;
- foreground motion falls while global camera motion rises;
- audio becomes missing, corrupted, or desynchronized;
- loss/gradients become non-finite;
- reward models disagree increasingly and no human audit supports the aggregate.

## 8. Runtime recovery

### CUDA OOM during behavior sampling

Try, in this order:

1. keep four-step/VSA/geometry fixed and set
   `FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE=1`;
2. confirm only SP leaders hold reward models;
3. run reward inference after moving the VAE back to CPU;
4. use `SP8 x DP1` instead of `SP4 x DP2` on eight 80-GB GPUs;
5. reduce LoRA rank 128 -> 64;
6. reduce prompt groups per collection while keeping K=8;
7. only for the one-GPU correctness smoke, use CPU/offloaded reward inference.

Do not reduce the four FastH3 steps, VSA policy, or clip below five seconds to
solve memory errors; that changes the actual model contract.

### OOM during backward

1. verify only LoRA parameters require gradients;
2. verify the base transformer is frozen;
3. verify full activation checkpointing is active;
4. set `vsa_cache_tile_buf: false`;
5. switch SP4 -> SP8;
6. lower LoRA rank;
7. inspect for retained decoded video/reward-model tensors.

### VSA backend errors

Check:

```bash
echo "$FASTVIDEO_ATTENTION_BACKEND"
echo "$FASTVIDEO_VSA_SM100A"
python -c 'import fastvideo_kernel; print(fastvideo_kernel)'
```

Use the Triton tile-64 path first for correctness. Enable the `sm100a` kernel
only on a compatible Blackwell build after strict output parity is established.
Never silently fall back to dense attention.

### VideoAlign failure

Verify:

```bash
test -f "$VIDEOALIGN_RUNTIME_PATH/inference.py"
test -d "$VIDEOALIGN_CHECKPOINT_PATH"
```

Common causes:

- old Qwen2-VL key names versus Transformers 5;
- missing `qwen-vl-utils`, `trl`, or `liger-kernel`;
- unavailable torchvision video reader;
- temporary MP4 codec failure;
- reward model left on a different CUDA device.

The FastVideo adapter includes key-remapping and OpenCV fallback patches. Do not
downgrade FastVideo's entire transformers/torch environment to VideoAlign's old
standalone environment.

### HPSv3 failure

Confirm `hpsv3==1.0.0`, HF access, sufficient cache space, and that frames are
uint8 RGB. HPSv3 is frame-based; a video score is aggregated across frames by
the adapter. Verify the expected general versus percentile path rather than
replacing both with a first-frame score.

### RAFT download/failure

The dynamic scorer uses `ptlflow>=0.4` and a pretrained RAFT checkpoint. The
first launch may download weights through the library. Cache them before a
multi-node job. A random/uninitialized RAFT model invalidates the reward stack
and must fail preflight.

### NaN or Inf

1. stop optimizer updates;
2. print each reward component before normalization;
3. verify group std and epsilon;
4. inspect endpoint/noise/prediction ranges per modality;
5. verify signed gradient is implemented via detached target;
6. verify FP32 LoRA masters and optimizer state;
7. reduce LR after implementation checks pass;
8. restart from the last clean checkpoint—do not continue a contaminated run.

### Audio degradation

1. confirm audio coefficient is exactly zero in the reward term;
2. confirm audio uses its own packed slice mean;
3. confirm LoRA-disabled reference prediction is evaluated at the same noisy
   audio state and timestep;
4. increase `audio_anchor_beta` from `1e-3` to `1e-2` only after those checks;
5. do not tune audio from VideoAlign/HPS/RAFT, which are visual rewards.

### Resume mismatch

Resume with the same config, topology, prompt bank, reward versions, and output
path. Check `metadata.json` and RNG snapshots. The rollout buffer is regenerated
if interruption occurred between attached optimizer updates. Validate the first
resumed checkpoint before continuing.

### LoRA export mismatch

Use `09_export_lora.sh` with the topology used to save the checkpoint. Then:

1. inspect the generated JSON manifest;
2. ensure every expected LoRA layer has A, B, and alpha tensors;
3. generate a fixed prompt/seed from the training checkpoint and exported
   adapter;
4. compare video/audio latents or decoded output;
5. do not publish an adapter before this parity test passes.

## 9. Required experiment record

For every run, save:

- git branch and exact commit SHA;
- full resolved YAML and CLI overrides;
- container/conda package lock or `pip freeze`;
- GPU model/count, topology, CUDA/driver, kernel versions;
- FastH3 and reward checkpoint revisions;
- prompt-source SHA and split metadata;
- all random seeds;
- W&B/local logs;
- evaluation MP4 directories;
- failure reason and exact fix for every restart;
- checkpoint selected and why.

Never alter a configuration after failure and reuse the same run name. A fix is
a new run with a recorded diff.

## 10. Go/no-go criteria for the full run

Proceed from sweeps to the full run only when the best candidate:

- wins at least 55% of a blinded pilot against released FastH3, or shows a
  clearly positive held-out metric delta corroborated by inspection;
- improves prompt adherence/aesthetics without a static-video increase;
- keeps audio metrics/listening within roughly 2% of baseline;
- preserves four-step inference latency after adapter merge;
- exports and reloads reproducibly.

A blog post should report the released baseline, exact compute/sample budget,
all reward/evaluation methods, both positive and negative dimensions, and the
actual selected checkpoint—not only the maximum training reward.
