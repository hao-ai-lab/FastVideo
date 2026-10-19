# FastH3 NVFP4 QAD

This page describes the shared quantization-aware distillation (QAD) recipe for
the FastH3 (MiniMax-H3) family. One recipe covers two model profiles:

| Profile | Model plugin | Task | Sampling | VSA tile |
|---|---|---|---|---|
| OmniRef | `omniref_pdd.MiniMaxH3OmniRefPDDModel` | Ref2VA | 8-forward PDD ladder | 128 |
| V2 / V2.1 Consumer | `v2_t2v.MiniMaxH3V2T2VModel` | T2VA | 8-step DMD | 64 |

Both profiles use the same shared recipe: NVFP4 W4A4 linears in all 50 blocks,
the FP4 sparse-attention emulator, and the same objective and guards. QAD
trains the student against a frozen bf16 teacher so that quantization error is
absorbed by the weights instead of appearing as drift, ghosting or audio loss
at inference. OmniRef's published v2 checkpoint includes FP4 attention; the
published V2.1 checkpoint predates it and keeps BF16 VSA attention (see
[Released checkpoints](#released-checkpoints)).

The runnable configs are:

- `examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml`
- `examples/train/configs/knowledge_distillation/minimax_h3/v2_nvfp4_qad.yaml`

## Released checkpoints

| Role | Checkpoint |
|---|---|
| OmniRef QAD v2 result | `FastVideo/FastH3-Omni-NVFP4-QAD-v2` |
| V2.1 result | `FastVideo/FastVideo-FastH3-8-Step-V2.1-NVFP4-Consumer` |
| V2 bf16 teacher / starting weights | `FastVideo/FastVideo-FastH3-8-Step-V2` |
| NVFP4 consumer baseline | `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer` |

The OmniRef config starts from the **calibrated OmniRef checkpoint used as QAD
step 0**, not from the released `-v2` result; point `init_from` at that
checkpoint. The V2 config starts from the bf16 V2 snapshot.

The published V2.1 checkpoint is the **linears-only variant** of this
schedule: NVFP4 W4A4 linears with BF16 VSA attention, 300 steps. The shared
FP4 recipe on this page additionally runs the FP4 attention emulator from step
0; it is the same training code and objective, but it does not reproduce the
published V2.1 checkpoint bit for bit. Set
`models.student.qad.fp4_attention.enabled: false` to get the published
linears-only variant.

## Target numerics

The student starts from the bf16 masters, and the deployed numerics are active
from the first forward, so every update sees the quantization error it is
trained to absorb.

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

The calibrated `amax` tables for both profiles are vendored under
`examples/train/calibration/fasth3/`:

| Table | sha256 |
|---|---|
| `amax-omniref-mse-bf16-top8.json` | `14b93ec3…df083d` |
| `amax-consumer.json` | `0800516c…d39e52` |

`qad.linears.amax_sha256` pins the table; the run copies it into the output
directory and refuses to start if the bytes differ.

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

## Data preparation

V2/V2.1 is data-free: it needs prompts and their Qwen3-VL text embeddings, not
VAE latents.

1. Build the split manifests from prompt sources (parquet shards with a
   `prompt` column, JSONL with a `prompt` field):

   ```bash
   python scripts/distill/minimax_h3_v2_qad/build_v2_prompts.py \
     --parquet 'data/prompts/*.parquet' --out-dir data/fasth3-v2 --eval-count 30
   ```

   This writes `train.jsonl` and `eval.jsonl` plus an `embeddings/` directory.
2. Fill the embeddings (safe to rerun and shard across GPUs):

   ```bash
   python scripts/distill/minimax_h3_v2_qad/precompute_v2_text.py \
     --config examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml \
     --model-path data/models/FastH3-V2 \
     --manifests data/fasth3-v2/train.jsonl data/fasth3-v2/eval.jsonl \
     --shard 0 --num-shards 8
   ```

   The V2 config's `train_manifest` and `eval_manifest` point at these files.

OmniRef training rows are preprocessed Ref2VA parquet rows that carry the
reference latents, text embeddings and geometry; the encoded dataset itself is
an input to this recipe and is not produced by the training code. Provide a
`MANIFEST.jsonl` with `case`, `id`, `parquet` and `resolution` fields and point
`models.student.data.manifests` at it.

`scripts/distill/minimax_h3_omniref_qad/generate_omniref_latents.py` builds the
held-out eval plan and runs the PDD ladder once per row: use `--eval-dir` to
write the held-out `manifest.json` consumed by `models.student.data.eval_manifest`
(and the raw latents for decoding).

## Running

Fill in the placeholder paths in the config (starting checkpoint, data
manifests, output directory), then launch with `torchrun`.

OmniRef, 4 nodes x 4 GPUs (sequence parallel 4):

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/omniref_nvfp4_qad.yaml
```

V2/V2.1, 4 nodes x 4 GPUs (16-way HSDP, lr 5e-6, 300 steps):

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/minimax_h3/v2_nvfp4_qad.yaml
```

To reproduce the published V2.1 linears-only checkpoint, disable FP4 attention
for the same schedule:

```bash
  --models.student.qad.fp4_attention.enabled false
```

## FP4-attention adaptation

The internal follow-up that this recipe tracks started from a linears-only
300-step DCP, enabled FP4 attention, and trained 300 more steps at lr 1.5e-6
with no warmup and a constant LR on 16 GPUs. It is an **adaptation of the
linears-only checkpoint to FP4 attention**, not a second epoch of the FP4
stage 1 above: the first stage's attention numerics were BF16, and the
continuation needs a DCP of that linears-only model (the packed HF release is
not one). It is therefore not part of the public reproduction path, and the
published V2.1 checkpoint remains the linears-only result. The exact overrides
are listed at the bottom of the V2 config.

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
| `fastvideo/train/models/minimax_h3/omniref_pdd.py` | OmniRef Ref2VA profile |
| `fastvideo/train/models/minimax_h3/v2_t2v.py` | V2/V2.1 T2VA profile |
| `fastvideo/layers/quantization/nvfp4_qad.py` | NVFP4 linear emulation and STE |
| `fastvideo/attention/backends/fp4_vsa_qat.py` | VSA fine-attention override |
| `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_vsa_train.py` | Tile 64/128 FP4 block-sparse Triton kernel |
| `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_vsa_reference.py` | PyTorch numeric reference |
| `scripts/distill/minimax_h3_v2_qad/build_v2_prompts.py` | V2 prompt split manifests |
| `scripts/distill/minimax_h3_v2_qad/precompute_v2_text.py` | V2 Qwen3-VL text embeddings |
| `scripts/distill/minimax_h3_omniref_qad/generate_omniref_latents.py` | OmniRef held-out eval plan and latents |
| `examples/train/calibration/fasth3/` | Pinned FFN activation `amax` tables |

## Expected behavior

Teacher-forced video and audio error both fall over the first tens of steps,
and the student's own video samples improve beyond the teacher's seed noise.
Audio samples improve more slowly and can remain within noise even when
teacher-forced audio error clearly drops. Gains slow after roughly 60-100
steps; the remaining error is dominated by FP4 attention. Evaluate a checkpoint
with the full-eval samples, not the teacher-forced score alone, before
promoting it.

Released reference points (lower is better):

| Run | Metric | Baseline | Selected |
|---|---|---|---|
| V2.1 (linears-only) | mean per-forward x0 rel-L2 | 0.431 | 0.351 (video 0.382, audio 0.319) |
| OmniRef QAD v2 | teacher-forced score | 0.362 | 0.303 at step 120 (video 0.198, audio 0.409) |

The V2.1 baseline is the NVFP4 Consumer model before QAD; the OmniRef baseline
is the calibrated checkpoint at QAD step 0.
