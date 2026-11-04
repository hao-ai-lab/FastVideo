# Pika H3 Flex NVFP4 QAD

This page describes how the NVFP4 version of
[Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) is trained: a
calibration pass followed by quantization-aware distillation (QAD) against
the bf16 Flex model, for both Flex adapters. It applies the shared FastH3 QAD
recipe ([FastH3 NVFP4 QAD](minimax_h3_fp4_qad.md)) to Flex's six-forward
re-noise sampler.

| Adapter | Trunk | Tasks | Model plugin |
|---|---|---|---|
| Ref2VA | `transformer_ref` | image / video / audio references | `flex.MiniMaxH3FlexRef2VAModel` |
| FL2VA | `transformer` | T2VA, I2VA (first frame), L2VA (last frame), FL2VA (both) | `flex.MiniMaxH3FlexFL2VAModel` |

Training runs in two stages per adapter:

1. **NVFP4 linears** (`*_nvfp4_qad.yaml`): W4A4 linears with BF16 dense
   attention.
2. **Dense FP4 attention** (`*_fp4attn_qad.yaml`): continues from stage 1's
   best checkpoint with FlashAttention-4 NVFP4 attention numerics.

The stage-1 result is the released checkpoint below; stage 2 is optional and
trades a small quality cost for faster long clips (see
[Dense FP4 attention](#stage-2-dense-fp4-attention)).

## Released checkpoint

| Role | Checkpoint |
|---|---|
| Teacher (bf16 adapters) | `Pika-Labs/Pika-H3-Flex` on `MiniMaxAI/MiniMax-H3` @ `42ed227` |
| QAD result (packed NVFP4, both trunks) | `FastVideo/Pika-H3-Flex-NVFP4-QAD` |

The release contains packed NVFP4 `transformer_ref` (Ref2VA) and `transformer`
(FL2VA) trunks with the adapters already merged, an NVFP4 Qwen3-VL text
encoder, an NVFP4 VAE decoder and the Pika schedulers. It targets Blackwell
GPUs and loads without a conversion step (validated on GB200):

```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1 \
python examples/inference/basic/basic_pika_h3_flex.py \
  --model-path FastVideo/Pika-H3-Flex-NVFP4-QAD --nvfp4-dit h3_dit \
  --nvfp4-decoder <snapshot>/vae_decoder_nvfp4/decoder-nvfp4.safetensors \
  --task ref2va --reference image:character.png --reference video:motion.mp4 \
  --prompt "$(cat prompt.txt)" --resolution 480p --duration 5
```

## Target numerics

- **Linears:** NVFP4 W4A4 for `attn.to_q`, `attn.to_k`, `attn.to_v`,
  `attn.to_out`, `ff.fc_in` and `ff.fc_out` in all 50 blocks, active from
  step 0. FFN inputs use static activation scales (`448 * 6 / amax`) from the
  calibrated table; attention linears use the unit activation scale.
- **Attention:** BF16 dense attention in stage 1. Flex has no VSA gate, so
  there is no sparse-attention stage and no gate to freeze. Stage 2 switches
  to the dense FP4 emulator (below).
- **Trained:** only the six quantized linear families. Norms, AdaLN, the token
  refiner, embeddings and the fp32 boundary modules stay frozen.
- **Teacher:** the merged bf16 Flex trunk, frozen. The student starts from the
  same weights with fp32 masters.

### Merged adapters

Flex ships each adapter as an fp32 PEFT LoRA (rank and alpha 128) over the
MiniMax-H3 trunk. QAD needs the teacher as plain weights, so
`merge_pika_adapter.py` merges `W + (alpha / rank) * B @ A` for every adapted
module in fp32 and rounds once to bf16. The merged directory also carries
`pika_flex_qad.json`, the sampling contract the model plugin reads (six
forwards at knots 1000/750/500/250/100/40, video shift 12, audio shift 3,
stochastic re-noise between forwards).

Exported trunks that already contain the adapter are marked by
`pika_h3_flex_merged.json` at the model root; the Pika pipeline then skips
loading the adapter again.

## Training objective

Each optimizer step takes one row. The frozen teacher walks the six-forward
ladder; three of the six forwards are trained per row. At each trained forward
the student predicts from the teacher's state:

- **Output:** relative L2 of the student's x0 prediction against the
  teacher's, for video and audio.
- **Reconstruction:** relative L2 of every block's output against the
  teacher's block output (`objective: output+recon`, `recon_lambda: 1.0`).

Both terms are normalized by their step-0 values. From step 20, 25% of rows
roll out from the student's own states instead of the teacher's. The learning
rate is 5e-6 (cosine to 0.3x after a 10-step warmup); gradients are clipped at
1.0. The re-noise draws are seeded per row and per forward, so teacher and
student see identical noise.

## Evaluation and guards

Every `eval_every` steps the run scores held-out rows (teacher-forced x0 error
per forward, averaged into `eval/score`); every `full_eval_every` steps it also
samples each row with the student's own six forwards and compares the result
with the teacher's (`endpoint/video`, `endpoint/audio`) and, for keyframe
tasks, the first and last frames.

- **Eligibility:** a checkpoint can become `best/` only if its full-eval
  audio, endpoint and keyframe metrics stay within `step0 * (1 + tolerance)`.
  With `noise_floor_eval: true` the tolerance per metric is twice its seed
  noise at step 0 (re-evaluated at seed + 1), at least 2%.
- **Patience:** the run stops after `patience` consecutive evaluations that are
  worse than step 0 on the score, or (full evaluations only) misaligned on the
  endpoint and keyframe metrics.

Keyframe metrics matter most for FL2VA: the last-frame match is the metric most
sensitive to the learning rate (see [Expected behavior](#expected-behavior)).

## Data preparation

QAD regresses on the teacher, so rows need conditioning (text embeddings and
reference latents) and target geometry, not curated targets. The starting
point is a set of encoded Ref2VA rows in the packed schema
(`fastvideo/dataset/dataloader/schema.py`, `pyarrow_schema_ref2va`) with a
`MANIFEST.jsonl` listing `case`, `id`, `parquet` and `resolution`, plus a
held-out manifest, as in the FastH3 OmniRef recipe (see
[Data preparation](minimax_h3_fp4_qad.md#data-preparation) there). Raw clips can
be encoded into this format with
`fastvideo/pipelines/preprocess/preprocess_minimax_h3_packed_ref2va.py`. Row
ids must contain the source video's id as `asset-<32 hex digits>`; clips of the
same video share it. Below, the source rows are `data/omniref/MANIFEST.jsonl`
and their held-out manifest is `data/omniref/eval-manifest.json`.

1. **Merge each adapter** into its trunk:

   ```bash
   python scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py \
     --base <MiniMax-H3 snapshot @42ed227> --adapter Pika-Labs/Pika-H3-Flex \
     --task ref2va --output data/models/pika-flex-ref2va-merged
   python scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py \
     --base <MiniMax-H3 snapshot @42ed227> --adapter Pika-Labs/Pika-H3-Flex \
     --task fl2va --output data/models/pika-flex-fl2va-merged
   ```

2. **Build Ref2VA rows** with `build_flex_ref2va_rows.py` (`plan` on CPU,
   `encode` sharded over GPUs, then `finalize`). Each reference is decoded
   from its stored latents and re-encoded at Flex's reference sizing, so the
   source media is not needed. The plan re-tiers the rows the QAD config will
   train on and adds input mixes the source set does not cover: a standalone
   audio reference, mixed image and video references, 6-9 image references,
   and two video references.

   `plan` reads the QAD plan from `--plan-config`. The Flex rows do not exist
   yet at this point, so plan with a copy of `ref2va_nvfp4_qad.yaml`
   (`plan-ref2va.yaml` below) whose `manifests` lists only
   `data/omniref/MANIFEST.jsonl` and whose `eval_manifest` is
   `data/omniref/eval-manifest.json`. `--eval-manifest` is the source held-out
   manifest; its clips are kept out of the new rows.

   ```bash
   python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py plan --mode ref2va \
     --source-manifest data/omniref/MANIFEST.jsonl --plan-config plan-ref2va.yaml \
     --eval-manifest data/omniref/eval-manifest.json --output-dir data/flex-ref2va
   CUDA_VISIBLE_DEVICES=$i python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py encode \
     --model-path data/models/pika-flex-ref2va-merged --output-dir data/flex-ref2va \
     --shard $i --num-shards 8
   python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py finalize \
     --output-dir data/flex-ref2va --eval-manifest data/omniref/eval-manifest.json
   ```

   `finalize` merges the shard manifests into `data/flex-ref2va/MANIFEST.jsonl`
   and writes `data/flex-ref2va/eval-manifest.json`: the source held-out rows,
   with each row that was re-tiered pointing at its Flex-sized parquet. The
   Ref2VA configs list the Flex manifest ahead of the source manifest, so
   Flex-sized rows replace source rows with the same id and resolution.

3. **Build FL2VA rows** from the same source rows. `--mode fl2va` builds T2VA,
   I2VA, L2VA and FL2VA rows whose keyframes are the clip's exact first/last
   frames, with the caption's reference paragraphs rewritten as keyframe
   definitions. `--fl2va-counts` and `--resolution-split` size the set
   (defaults: 1,100 rows, 80% 480p). FL2VA holds out its eval rows from its own
   groups, so `finalize` takes no `--eval-manifest` and writes an empty one.

   ```bash
   python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py plan --mode fl2va \
     --source-manifest data/omniref/MANIFEST.jsonl \
     --eval-manifest data/omniref/eval-manifest.json --output-dir data/flex-fl2va
   CUDA_VISIBLE_DEVICES=$i python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py encode \
     --model-path data/models/pika-flex-fl2va-merged --output-dir data/flex-fl2va \
     --shard $i --num-shards 8
   python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py finalize --output-dir data/flex-fl2va
   ```

Rows that fail to encode (for example a reference video longer than Flex's
15-second limit) are recorded per shard; `encode --retry-skipped` retries only
those. The configs train on rows up to 345 frames at 480p and 768p
(`per_group: {480p: 160, 768p: 40}` per case) and hold out 3 rows per case.

## Calibration

`calibrate_flex_nvfp4.py` picks the static FFN activation scales on rows walked
by the bf16 teacher over its six forwards:

1. `collect` (torchrun): forward pre-hooks record each row's per-layer input
   amax, split into target video, target audio and conditioning tokens.
   `--max-rows` takes the first rows of the plan; leave rows uncollected for
   `mse`, which stops if every planned row was collected.
2. `candidates` (CPU): per layer, the max over rows plus clipped alternatives
   (per-row p99/p90, and 0.7/0.5/0.35/0.25 of the max).
3. `mse` (torchrun): on disjoint rows plus the rows that set the maxima, each
   candidate's NVFP4 GEMM output is compared with the bf16 output.
4. `choose` (CPU): per layer, the candidate with the lowest mean of video and
   audio relative error (`amax-mse-balanced.json`), next to the all-token
   criterion (`amax-mse.json`) and the plain max.

```bash
torchrun --nproc_per_node 4 scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py collect \
  --config examples/train/configs/knowledge_distillation/pika_h3_flex/ref2va_nvfp4_qad.yaml \
  --output-dir calib/ref2va --shard $NODE --num-shards $NODES --max-rows $COLLECT_ROWS \
  --training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4
python scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py candidates --output-dir calib/ref2va
torchrun --nproc_per_node 4 scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py mse \
  --config <same config> --output-dir calib/ref2va --shard $NODE --num-shards $NODES \
  --training.distributed.num_gpus 4 --training.distributed.hsdp_shard_dim 4
python scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py choose --output-dir calib/ref2va
```

Compare the candidate tables by the QAD harness's step-0 score (run the QAD
config with each table and read `eval/score` at step 0) and pin the best one.
For both adapters the video/audio-balanced table scored best:

| Adapter | Calibration rows | Step-0 score: max / all-token MSE / balanced |
|---|---|---|
| Ref2VA | 1,133 | 0.3311 / 0.3299 / **0.3260** |
| FL2VA | 546 | 0.2730 / 0.2741 / **0.2724** |

The selected tables are vendored and pinned by sha256:

| Table | sha256 |
|---|---|
| `examples/train/calibration/pika_h3_flex/amax-ref2va-balanced.json` | `ad60b45f…da90b373` |
| `examples/train/calibration/pika_h3_flex/amax-fl2va-balanced.json` | `b690d117…553e0daa` |

## Running

Point the placeholder paths in the config at the merged model, the row
manifests and an output directory, then launch with `torchrun`. The stage-1
configs assume 16 GPUs and the stage-2 configs 12 (sequence parallel 4,
HSDP over all GPUs); the released FL2VA stage-1 run used 24. Rows reach about
58k tokens; on GB200 the runs use about 90-100 GB per GPU.

```bash
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/knowledge_distillation/pika_h3_flex/ref2va_nvfp4_qad.yaml
```

Loading and the step-0 evaluation of a multi-node run can exceed the default
10-minute process-group timeout; set `FASTVIDEO_PG_TIMEOUT_MIN=120`. Any config
key can be overridden on the command line (`--training.distributed.num_gpus 24
--training.distributed.hsdp_shard_dim 24` for 6 nodes, for example). Stage 2
launches the same way with `ref2va_fp4attn_qad.yaml` or `fl2va_fp4attn_qad.yaml`
once stage 1 has written `best/`.

## Stage 2: dense FP4 attention

Deployment can additionally run attention through FlashAttention-4's NVFP4
path: Q and K are quantized to NVFP4 per 16 channels with a unit global scale,
P and V stay BF16. Stage 2 adapts the stage-1 student to these numerics with
`fp4_dense_qat.py`, which routes every dense attention layer through the FP4
attention training kernel with full block lists and straight-through
gradients. `scripts/distill/pika_h3_flex_qad/check_fp4_attention_parity.py`
checks that the emulator matches the FA4 kernel (on a Blackwell GPU): the
emulator-to-kernel distance is under 1% of the FP4 quantization error.

Plain FP4 attention costs 13-17% on the step-0 score. Two exact Q/K transforms
and two BF16 blocks recover most of it before any training:

- **smooth-K:** subtract each head's token-mean of K before quantizing
  (softmax is invariant to a per-query shift).
- **Hadamard:** rotate Q and K by the same orthonormal 128-point Hadamard
  matrix (`Q H (K H)^T = Q K^T`), spreading outlier channels across the
  16-channel scale blocks.
- **BF16 blocks 45 and 49:** the two attention layers with the largest FP4
  error keep BF16 attention.

| Per-layer FP4 attention error vs BF16 (real activations) | FL2VA | Ref2VA |
|---|---|---|
| Plain FP4 | 0.115 | 0.113 |
| smooth-K + Hadamard | 0.095 | 0.095 |
| + blocks 45 and 49 in BF16 | 0.083 | 0.080 |

The stage-2 configs set `qk_transform: smooth_k+hadamard` and
`bf16_blocks: [45, 49]`, start from stage 1's `best/`
(`method.init_student_dcp`) and train at 5e-6 (Ref2VA) or 3e-6 (FL2VA). Deploy
a stage-2 checkpoint with the same settings:

```bash
python examples/inference/basic/basic_pika_h3_flex.py ... \
  --nvfp4-attention smooth_k+hadamard --nvfp4-attention-bf16-blocks 45,49
```

(equivalently `FASTVIDEO_NVFP4_FA4=1`,
`FASTVIDEO_NVFP4_FA4_QK_TRANSFORM=smooth_k+hadamard`,
`FASTVIDEO_NVFP4_FA4_BF16_BLOCKS=45,49`).

Measured on one GB200 (FL2VA, NVFP4 DiT and decoder, warm), FP4 attention
helps long clips and is neutral on short ones:

| Clip | BF16 attention | FP4 attention | FP4 + transforms |
|---|---|---|---|
| 480p, 5 s | 5.87 s | 5.66 s | 6.08 s |
| 768p, 15 s | 99.7 s | 84.7 s | 87.4 s |

The transforms currently run as separate PyTorch ops before the FP4 quantizer;
fusing them into the quantizer removes most of their cost.

## Export

Checkpoints hold fp32 masters in DCP format. Export a checkpoint to the packed
NVFP4 format the inference loader reads, then verify it:

```bash
# 1. bf16 trunk from the student's masters (CPU)
python scripts/distill/pika_h3_flex_qad/export_student_bf16.py \
  --dcp outputs/pika_flex_ref2va_nvfp4_qad/best/dcp \
  --base data/models/pika-flex-ref2va-merged --trunk transformer_ref \
  --out exports/ref2va/transformer_bf16
# 2. pack NVFP4 linears with the pinned activation scales
python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py \
  --src exports/ref2va/transformer_bf16 --dst exports/ref2va/transformer_ref \
  --quantize-ffn --quantize-attention \
  --act-amax examples/train/calibration/pika_h3_flex/amax-ref2va-balanced.json
# 3. byte-level parity with the QAD numerics (1 GPU)
python scripts/distill/pika_h3_flex_qad/check_nvfp4_export_parity.py \
  --export exports/ref2va/transformer_ref --bf16 exports/ref2va/transformer_bf16 \
  --amax examples/train/calibration/pika_h3_flex/amax-ref2va-balanced.json
```

The parity check confirms that packed weights and block scales are
byte-identical to what training quantized and that the deployed NVFP4 GEMM
matches the training GEMM bit for bit. Repeat for FL2VA with
`--trunk transformer` and its table, then assemble a model directory:

```bash
python scripts/distill/pika_h3_flex_qad/assemble_flex_release.py --output release/Pika-H3-Flex-NVFP4 \
  --base <MiniMax-H3 snapshot> --pika <Pika-H3-Flex snapshot> \
  --transformer-ref exports/ref2va/transformer_ref --ref2va-merged data/models/pika-flex-ref2va-merged \
  --transformer exports/fl2va/transformer --fl2va-merged data/models/pika-flex-fl2va-merged
```

`--text-encoder` and `--vae-decoder-nvfp4` add NVFP4 text-encoder and decoder
exports.

## Code map

| Path | Purpose |
|---|---|
| `fastvideo/train/models/minimax_h3/flex.py` | Flex model plugins: sampling contract, re-noise ladder, FL2VA keyframe rows, QAD install |
| `fastvideo/train/methods/knowledge_distillation/pdd_qad.py` | Shared QAD method: losses, evaluation, guards |
| `fastvideo/attention/backends/fp4_dense_qat.py` | Dense FP4 attention emulator and install hook |
| `fastvideo/attention/utils/nvfp4_qk_precondition.py` | Exact Q/K transforms shared by training and inference |
| `fastvideo/attention/backends/flash_attn.py` | Deployed FA4 NVFP4 attention (`nvfp4_fa4`, `nvfp4_qk_transform`) |
| `fastvideo/pipelines/basic/pika_h3_flex/adapter.py` | Adapter loading and the merged-adapter marker |
| `scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py` | Merge an adapter into its trunk |
| `scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py` | Ref2VA and FL2VA training rows |
| `scripts/distill/pika_h3_flex_qad/calibrate_flex_nvfp4.py` | FFN activation-scale calibration |
| `scripts/distill/pika_h3_flex_qad/check_fp4_attention_parity.py` | Emulator vs FA4 NVFP4 kernel parity |
| `scripts/distill/pika_h3_flex_qad/export_student_bf16.py` | DCP masters to a bf16 trunk |
| `scripts/distill/pika_h3_flex_qad/check_nvfp4_export_parity.py` | Packed export vs QAD numerics parity |
| `scripts/distill/pika_h3_flex_qad/assemble_flex_release.py` | Self-contained NVFP4 model directory |
| `examples/train/configs/knowledge_distillation/pika_h3_flex/` | Stage-1 and stage-2 configs for both adapters |
| `examples/train/calibration/pika_h3_flex/` | Pinned FFN activation `amax` tables |

## Expected behavior

Teacher-forced error falls steadily for the first 30-60 steps and flattens by
step 90-120. Video improves faster than audio. Results of the released runs
(held-out rows, lower is better, distances to the bf16 Flex teacher):

| Run | Step 0 (PTQ) | Selected | Notes |
|---|---|---|---|
| Ref2VA stage 1, 16 GPUs | 0.326 | **0.270** at step 120 (-17.1%) | video x0 0.185, audio x0 0.356 |
| FL2VA stage 1, 24 GPUs | 0.272 | **0.234** at step 60 (-14.1%) | first-frame 0.072, last-frame 0.247 |

- FL2VA's last-frame match can drift above its tolerance while the score keeps
  improving; at 5e-6 this stopped stage 2 at step 60 with no eligible
  checkpoint, so its stage-2 config uses 3e-6.
- A lower teacher-forced score with a failed guard is not a better model:
  promote only eligible checkpoints, and review the full-eval samples
  (`eval_save_dir`) before release.
