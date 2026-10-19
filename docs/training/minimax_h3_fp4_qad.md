# FastH3 NVFP4 QAD

This page describes the shared quantization-aware distillation (QAD) recipe for
the FastH3 (MiniMax-H3) family. One recipe covers two model profiles:

| Profile | Model plugin | Task | Sampling | VSA tile |
|---|---|---|---|---|
| OmniRef | `omniref_pdd.MiniMaxH3OmniRefPDDModel` | Ref2VA | 8-forward PDD ladder | 128 |
| V2 / V2.1 Consumer | `v2_t2v.MiniMaxH3V2T2VModel` | T2VA | 8-step DMD | 64 |

Both profiles ship the same deployed numerics: NVFP4 W4A4 linears in all 50
blocks plus FP4 sparse attention. QAD trains the student against a frozen bf16
teacher so that quantization error is absorbed by the weights instead of
appearing as drift, ghosting or audio loss at inference.

The runnable configs are:

- `examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml`
- `examples/train/configs/knowledge_distillation/minimax_h3/v2_nvfp4_qad.yaml`

## Target numerics

The student runs exactly the deployed inference numerics from step 0, so the
starting point is the released NVFP4 model rather than a bf16 approximation.

- **Linears:** NVFP4 W4A4 for `attn.to_q`, `attn.to_k`, `attn.to_v`,
  `attn.to_out`, `ff.fc_in` and `ff.fc_out` in every block.
  - FFN inputs use static activation scales from the calibrated `amax` table.
  - Attention and gate linears use the unit activation scale; their `amax`
    entries are optional.
  - `skip_blocks` is empty: no block falls back to bf16 linears.
- **Attention:** the FP4 sparse-attention emulator for VSA tile 128 (OmniRef)
  or tile 64 (V2/V2.1). Two-level P quantization and K smoothing are on. The
  emulator reproduces the deployed kernel's block order and rounding on any
  CUDA GPU and provides straight-through gradients.
- **Kept frozen:** the VSA compression gates and every norm stay at their
  released values. Training the gate weights drifts the sparse-attention tile
  selection and produces ghosting; the gate linears still execute the deployed
  NVFP4 format, they are just not updated. `proj_in`, the time embedder,
  `proj_out` and the audio projections stay fp32 and frozen, matching
  inference.

The calibrated `amax` table is produced once per released checkpoint and
shipped with it. `qad.linears.amax_sha256` pins the table; the run copies it
into the output directory and refuses to start if the bytes differ.

## Training objective

For each row the frozen bf16 teacher walks the profile's sampling ladder. At
the same inputs and timesteps, the student minimizes:

- **Output regression:** per-forward x0 error, normalized per forward and
  modality by its step-0 value.
- **Layer-wise reconstruction:** each attention and FFN sub-layer receives the
  teacher's input (stop-gradient) and matches the teacher sub-layer's output.
  The loss is weighted per row group (target video and audio 1.0,
  conditioning 0.25) because audio is a small fraction of the packed tokens.
  `recon_normalize: step0` divides it by its step-0 value so `recon_lambda`
  balances the two terms.

From `rollout_start_step`, `rollout_fraction` of the rows continue from the
student's own trajectory instead of the teacher's, which reduces exposure bias
in the sampled output.

## Evaluation and guards

- **Teacher-forced eval** every `eval_every` steps on held-out rows.
- **Full eval** every `full_eval_every` steps: the student's own sampled output
  against the teacher at the same seed, keyframe alignment, and saved audio and
  video latents.
- **Noise floor:** `noise_floor_eval` re-runs the step-0 eval at seed + 1 so a
  metric move can be compared with teacher seed noise.
- **Best checkpoint:** selected by the teacher-forced `score`. A checkpoint is
  eligible only if the keys in `eligibility_keys` (audio error and endpoint)
  stay within their step-0 guardrails.
- **Auto-stop:** after `patience` consecutive bad evals.
- **Step-0 cache:** `step0_cache` fingerprints the teacher, student,
  calibration, quantization plan, eval set and loss weights, so reruns skip the
  baseline evaluation.
- **Resume:** checkpoints carry the step-0 baseline and guard state; do not
  delete `method.qad_baseline` if you edit a checkpoint by hand.

## Running

Fill in the placeholder paths in the config (starting checkpoint, `amax`
table, latent manifests, output directory), then launch with `torchrun`.

OmniRef, 4 nodes x 4 GPUs (SP 4):

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml
```

V2/V2.1, 4 GPUs:

```bash
torchrun --nproc_per_node=4 \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/v2_nvfp4_qad.yaml
```

The V2/V2.1 recipe has an optional lower-learning-rate continuation stage. It
keeps the same numerics and objective and resumes from the first stage's `best`
DCP. The knobs are listed at the bottom of the config:

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/v2_nvfp4_qad.yaml \
  --method.init_student_dcp outputs/v2_1_nvfp4_qad/best \
  --training.optimizer.learning_rate 1.5e-6 \
  --training.distributed.num_gpus 16 \
  --training.distributed.hsdp_shard_dim 16 \
  --training.checkpoint.output_dir outputs/v2_1_nvfp4_qad_continuation
```

## Checkpoints

Runs write regular `checkpoint-<step>/` DCP checkpoints plus a `best/`
directory:

- `best/`: the best eligible state (weights only), refreshed at every eval and
  never pruned.
- `checkpoint-<step>/`: weights, optimizer, RNG state and the method baseline;
  the last step is always saved.

The saved state is the bf16 master plus the quantization plan, not packed NVFP4
weights; inference applies the same converter used for the released model. Use
`fastvideo/train/entrypoint/dcp_to_diffusers.py` to export a checkpoint if your
deployment path expects Diffusers weights. It detects the weights-only marker
on a tagged checkpoint, so `best/` can be exported directly:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint outputs/v2_1_nvfp4_qad/best \
  --output-dir outputs/v2_1_nvfp4_qad/export
```

## Code map

| Path | Purpose |
|---|---|
| `fastvideo/train/methods/knowledge_distillation/pdd_qad.py` | Method: ladder, losses, evaluation, guards |
| `fastvideo/train/methods/knowledge_distillation/pdd_qad_metrics.py` | Rung aggregation and score |
| `fastvideo/train/methods/knowledge_distillation/pdd_qad_recon.py` | Layer-wise reconstruction hooks |
| `fastvideo/train/methods/knowledge_distillation/pdd_qad_step0.py` | Step-0 baseline cache |
| `fastvideo/train/methods/knowledge_distillation/pdd_qad_audio_stats.py` | Audio guard statistics |
| `fastvideo/train/models/minimax_h3/omniref_pdd.py` | OmniRef Ref2VA profile |
| `fastvideo/train/models/minimax_h3/v2_t2v.py` | V2/V2.1 T2VA profile |
| `fastvideo/layers/quantization/nvfp4_qad.py` | NVFP4 linear emulation and STE |
| `fastvideo/attention/backends/fp4_vsa_qat.py` | VSA fine-attention override |
| `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_vsa_train.py` | Tile 64/128 FP4 block-sparse Triton kernel |
| `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_vsa_reference.py` | PyTorch numeric reference |

## Expected behavior

Teacher-forced video and audio error both fall over the first tens of steps,
and the student's own video samples improve beyond the teacher's seed noise.
Audio samples improve more slowly and can remain within noise even when
teacher-forced audio error clearly drops. Gains slow after roughly 60-100
steps; the remaining error is dominated by FP4 attention. Evaluate a checkpoint
with the full-eval samples, not the teacher-forced score alone, before
promoting it.
