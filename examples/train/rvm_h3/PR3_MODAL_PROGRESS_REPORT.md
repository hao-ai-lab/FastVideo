# PR #3 Modal H3 RVM Progress Report

- Date: 2026-08-31
- PR: [Abecid/FastVideo#3](https://github.com/Abecid/FastVideo/pull/3)
- Tested PR commit: `74c0d0d5c68fdbfbd3f80ab2ca369dc7f9019679`
- PR branch: `adam/h3-rvm-posttraining`

## Outcome

The H3 RVM Modal workflow now completes its preflight, public inference,
strict one-H100 training smoke, production-shaped four-H100 pilot, and final
four-H100 training run.

The final run used four strict NVIDIA H100 80GB GPUs, completed 34 optimizer
steps over 17 fresh rollout collections, consumed 68 prompt groups from a
64-row dataset (1.0625 epochs), saved checkpoints 17 and 34, ran the final
eight-prompt evaluation, synchronized its W&B media and metrics, and exited
cleanly. The 34-step loop took 3h18m20s; total wall time including image setup,
35B model load, reward-model load, and baseline validation was approximately
3h38m, safely below the 12-hour limit.

No traceback, CUDA OOM, NCCL error, reward-worker exception, NaN, or Inf was
found in the completed run log.

## Runtime fixes

### Current dependency compatibility

- Added Transformers 5 compatibility bridges for VideoAlign and HPSv3 Qwen2-VL
  reward models without downgrading Transformers.
- Corrected HPSv3 reward-model invocation and preserved its trained reward-head
  checkpoint loading.
- Added the H3 text-only preprocessor/model-loader compatibility needed by the
  current component loading and validation APIs.
- Scoped component-directory validation so transformer and VAE subdirectories
  are not incorrectly treated as complete pipelines.

### Reward correctness and memory

- Merged top-level per-reward options with inline reward specifications, with
  inline values taking precedence. This activates the configured device and
  dynamic-tracking settings instead of silently discarding them.
- Added compatibility for the H3 dynamic-tracking option names and maintained
  resolution-aware RAFT scoring.
- Bounded HPSv3 frame inference to batches of 16. The scorer still evaluates
  all 53 selected frames per video and preserves the original mean/top-30%
  aggregation, while avoiding a single 212-frame K=4 activation batch.
- Released reward/VAE CUDA cache before metric collectives so NCCL can allocate
  its communication buffers.
- Added bounded reward VAE decoding and the one-GPU layerwise-offload path.

### Training correctness

- Kept trainable LoRA master parameters in FP32 while explicitly casting
  forward operands to the BF16 compute dtype. This fixes the DTensor Adam
  parameter/gradient mismatch without changing frozen 35B base weights.
- Generalized H3 packed/unpacked batch handling for the RVM K-sample layout.
- Preserved the intended four-step DMD schedule, VSA configuration, prompt
  grouping, and buffer-reuse update cadence.
- Added focused coverage for config parsing, H3 batch behavior, attention
  backend loading, reward-option precedence, cache ordering, and HPS frame
  chunking/aggregation.

## Hardware gate

Modal may upgrade a plain `H100` request to an H200. The launcher therefore
supports strict `H100!` and `H100!:4` requests, and the manifests record the
actual device names and memory.

The accepted runs reported:

```text
NVIDIA H100 80GB HBM3, 81559 MiB
```

for every allocated GPU.

The full 35B, 480x832x124, rank-128 configuration does not fit on a single
80GB H100. The one-GPU gate is consequently a clearly labeled compact topology
smoke that exercises rollout, reward, sparse-attention forward/backward, Adam,
checkpointing, and validation. The exact production geometry and all learned
rewards are gated on four strict H100s.

## Validation sequence

| Gate | Configuration | Result |
|---|---|---|
| Preflight | Imports, focused unit tests, component/reward loading, synthetic scoring | Passed; 29 tests and all configured scorers |
| Public inference | 480x832, 124 frames, audio/video output | Passed; repeated runs were byte-identical |
| Strict one-H100 smoke | Compact 320x576x39, K=2, one update | Passed rollout, backward, Adam, checkpoint, and validation |
| Strict four-H100 pilot | Full 480x832x124, SP=4, K=4, rank/alpha 128/64, all five rewards, two updates | Passed with checkpoints 1 and 2 |
| Strict four-H100 final | Same production topology, 34 steps, eight validation prompts | Passed 1.0625 epochs with checkpoints 17 and 34 |

### Strict one-H100 topology smoke

- Modal: [ap-EVLM5yJYyr3LwZ0yHtKqx4](https://modal.com/apps/hao-ai-lab/main/ap-EVLM5yJYyr3LwZ0yHtKqx4)
- W&B: [642389c31d43e7d2](https://wandb.ai/attentionx2023/fasth3-rvm/runs/642389c31d43e7d2)
- Training loop: 2m04s
- Final metrics: reward `0.48306`, advantage `0.04981`, grad norm `0.04966`
- Checkpoint: `checkpoint-1`

### Strict four-H100 production pilot

- Modal: [ap-RRbxw7yDlSLRYQZndTVkIp](https://modal.com/apps/hao-ai-lab/main/ap-RRbxw7yDlSLRYQZndTVkIp)
- W&B: [3ebd45dc2c088630](https://wandb.ai/attentionx2023/fasth3-rvm/runs/3ebd45dc2c088630)
- Training loop: 12m22s
- Checkpoints: `checkpoint-1`, `checkpoint-2`
- The pilot verified a fresh K=4 rollout and the buffer-reuse update, all five
  GPU rewards, sparse-attention backward, optimizer updates, and final media
  evaluation on the exact production topology.

## Final four-H100 run

- Modal: [ap-TqlPuMGQNSIkzsTH56LldY](https://modal.com/apps/hao-ai-lab/main/ap-TqlPuMGQNSIkzsTH56LldY)
- W&B: [b20dc07781ecdd4b](https://wandb.ai/attentionx2023/fasth3-rvm/runs/b20dc07781ecdd4b)
- GPUs: 4x strict H100 80GB
- Base model: 35.05B parameters
- LoRA: rank 128, alpha 64, 208 adapted layers, FP32 masters
- Geometry: 480x832, 124 frames
- Distributed topology: SP=4, TP=1, HSDP shard=4
- Sampling: K=4, four global prompt groups per collection
- Optimization: two updates per rollout collection, LR `1e-5`
- Steps/collections: 34/17
- Dataset consumption: 68 prompt groups / 64 rows = 1.0625 epochs
- Checkpoints: `checkpoint-17`, `checkpoint-34`
- Step-loop runtime: 3h18m20s
- W&B total runtime: approximately 3h30m

Final training summary:

| Metric | Value |
|---|---:|
| `reward/avg` | 2.26628 |
| `reward/dynamic_tracking` | 1.00000 |
| `reward/hpsv3_general` | 4.12118 |
| `reward/hpsv3_percentile` | 7.58411 |
| `reward/videoalign_mq` | 0.74428 |
| `reward/videoalign_ta` | -0.23235 |
| `total_loss` | 0.00299136 |
| `rvm/grad_norm` | 4.48528 |
| `rvm/zero_std_group_ratio` | 0.00000 |

All summary values were finite. A zero `zero_std_group_ratio` confirms that
the K=4 groups retained reward variance rather than collapsing to identical
scores.

## Eight-prompt evaluation

| Metric | Baseline | Step 34 | Delta |
|---|---:|---:|---:|
| Aggregate reward | 1.51039 | 1.46332 | -0.04707 |
| Dynamic tracking | 0.97165 | 1.00000 | +0.02835 |
| HPSv3 general | 3.69434 | 3.79748 | +0.10314 |
| HPSv3 percentile | 7.20946 | 7.43185 | +0.22240 |
| VideoAlign MQ | 0.16878 | 0.15438 | -0.01441 |
| VideoAlign TA | -0.28595 | -0.34266 | -0.05671 |

Three evaluation components improved. The aggregate changed by only -0.04707
because the two VideoAlign components decreased slightly. This short run should
therefore be treated as a runtime/correctness and non-collapse result, not as
evidence that every learned quality metric improves after 34 steps.

## Qualitative review

All eight baseline/final prompt pairs were downloaded and reviewed at four
timestamps per video. They cover a video-editing scene, a family memorial
scene, a talking cat, an animated living room, a stickman energy sphere, a
1930s Greek orchestra, a mountain valley, and an animated Toy Land scene.

Observed results:

- Prompt intent remained recognizable in every final sample.
- Subject identity, scene layout, and object geometry remained coherent across
  time.
- Motion progressed sensibly; no static-output collapse was observed.
- No severe flicker burst, duplicated face/limb, frame corruption, or temporal
  discontinuity was found.
- Framing changed in some pairs, most visibly in the editor/camera prompt, but
  the semantic intent was preserved.
- The cat, family, stickman action, orchestra, valley, and Toy Land samples were
  particularly stable between baseline and step 34.

Every validation artifact was valid H.264 video at 832x480, 124 frames, 24 fps,
and 5.1667 seconds. Paired baseline-to-final SSIM averaged `0.63073` with a
range of `0.55405` to `0.70125`, indicating meaningful learned variation
without destructive scene drift. SSIM here is a drift diagnostic, not a claim
that the final sample should reproduce the baseline pixel-for-pixel.

Training validation artifacts are visual-only because the configured rewards
evaluate video frames. The separate public inference smoke verified the full
H3 media path, including a valid H.264 video stream and AAC stereo audio; its
repeat runs were byte-identical with SHA-256
`ba60579ae9e0deb09300cbd58f2df101d73a679a2500b805b0a03613b6ae288b`.

## Reproduction

Strict one-H100 topology smoke:

```bash
H3_RVM_MODAL_GPU_1='H100!' \
H3_RVM_MODAL_GPU_4='H100!:4' \
H3_RVM_MODAL_SECRETS='hf-adamlee00,wandb-adamlee00' \
conda run --no-capture-output -n fastvideo modal run \
  examples/train/rvm_h3/modal_h3_rvm.py \
  --gpus 1 \
  --mode smoke \
  --commit 74c0d0d5c68fdbfbd3f80ab2ca369dc7f9019679 \
  --max-steps 1 \
  --eval-prompts 1
```

Final four-H100 run shape:

```bash
H3_RVM_MODAL_GPU_1='H100!' \
H3_RVM_MODAL_GPU_4='H100!:4' \
H3_RVM_MODAL_SECRETS='hf-adamlee00,wandb-adamlee00' \
conda run --no-capture-output -n fastvideo modal run \
  examples/train/rvm_h3/modal_h3_rvm.py \
  --gpus 4 \
  --mode all \
  --skip-preflight \
  --commit 74c0d0d5c68fdbfbd3f80ab2ca369dc7f9019679 \
  --run-name h3-rvm-pr3-h100-strict-4gpu-1p0625epoch-20260831-r46 \
  --max-steps 34 \
  --eval-prompts 8 \
  --max-train-prompts 64 \
  --smoke-prompts 16
```

The launcher functions have an 11h56m remote timeout so a stuck run cannot
exceed the requested 12-hour training window.

## Remaining caveats

- The 34-step run establishes runtime correctness, finite learning signals,
  checkpointing, and qualitative non-collapse. It is too short to claim a
  statistically reliable quality improvement across all reward models.
- The one-GPU test is a compact topology smoke by necessity; exact production
  capacity is established by the four-H100 pilot and final run.
- FastVideo package tests are excluded from repository pre-commit by design.
  The Modal preflight exercised the focused test set, while the completed pilot
  and final run exercised the current HPS chunking and cache-ordering paths at
  production scale.

## Post-run reward interpretation

The r45 and r46 validation values should not be read as one continuous learning
curve. r45 evaluated one fixed prompt, whereas r46 averaged eight fixed prompts.
The apparently stronger r45 baseline is therefore primarily a prompt-selection
effect. Within each run, validation is paired and deterministic: the prompt
indices are selected with seed `4242`, and each prompt is sampled with seed
`4242 + prompt_index`. Baseline-to-final changes are model changes on those
fixed prompt/noise pairs, not fresh validation sampling noise.

r45's one-prompt aggregate drop (`-0.67567`) was dominated by VideoAlign MQ
(`-0.77497` before weighting). A single fixed sample cannot establish a
population regression. In r46, the weighted contributions to the aggregate
delta were approximately `+0.01985` dynamic tracking, `+0.01031` HPSv3 general,
`+0.02224` HPSv3 percentile, `-0.01441` VideoAlign MQ, and `-0.08506`
VideoAlign TA. Thus the small net change (`-0.04707`) was mostly the
`1.5`-weighted TA decrease, while three of five components improved.

The completed run was intentionally a runtime/non-collapse gate, not a
published-scale reward-optimization run. It used 17 fresh rollout collections,
four prompts per collection, K=4, and 272 rewarded endpoints total. The RVM
paper's Wan video recipe uses 90 rollout collections, 32 prompts per collection,
K=8, two optimizer updates per collection, and 23,040 rewarded endpoints. The
repository's full H3 config mirrors that 180-update/23,040-endpoint budget. r46
therefore covered about 19% of the intended optimizer-update horizon but only
about 1.2% of the published rewarded-sample budget. Its LR (`1e-5`) is also
five times lower than the paper's Wan recipe (`5e-5`). A weak or mixed held-out
reward change after r46 is consequently expected and is not evidence that RVM
has converged.

Training longer is warranted, but additional passes over the same 64 prompts
are less informative than increasing rollout diversity. The next quality run
should prioritize more prompt groups per collection and a larger held-out set,
retain paired seeds, report per-prompt deltas with uncertainty, and evaluate an
external benchmark such as VBench in addition to the optimized rewards. This
matters because the RVM paper trained on 48,998 prompts and evaluated 946 VBench
prompts, while DanceGRPO used more than 10,000 training prompts and 1,000 video
evaluation prompts. Earlier video reward-fine-tuning work also found that some
RL/RWR objectives plateaued or deteriorated with more wall time, so longer runs
should be checkpointed and selected by held-out metrics and qualitative review
rather than duration alone.

Primary references:

- [Scaling Reinforcement Learning for Diffusion Models via Velocity Matching](https://arxiv.org/abs/2608.23664)
- [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
- [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://arxiv.org/abs/2312.12490)
