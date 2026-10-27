# MiniMax-H3 reference-conditioned PDD and DMPDD training

The modular trainer supports Parallel Decoding Distillation (PDD) and Distribution
Matching with Parallel Decoding Distillation (DMPDD) for MiniMax-H3 reference-conditioned
video and audio. The methods adapt FastGen's PDD objectives to FastVideo's model,
sequence-parallel and checkpoint interfaces.

## Prepare model weights and data

Use a converted FastVideo MiniMax-H3 Ref2VA model directory containing `transformer_ref`,
text encoder and video/audio VAEs. The student, frozen teacher and trainable critic can
start from the same base checkpoint. A fresh PDD student expands the base output heads
into the configured fine time grid; this is a new initialization, not a pretrained
distillation checkpoint. A compatible PDD checkpoint can also initialize the student.

Follow [packed Ref2VA preprocessing](minimax_h3_ref2va_preprocess.md) to encode your own
records. That preprocessing path stores clean reference video/audio rows separately
from target latents and keeps ordered presentation metadata. Its output is the
`preprocessed_data_type: ref2va` contract consumed by these adapters. The existing
Ref2VA finetuning preprocessor uses a separate schema.

Native shape bucketing preserves each record's spatial and temporal dimensions.
References remain conditioning throughout student, teacher and critic forwards;
only target video/audio contribute to noising and losses. Set
`max_presentation_tokens` to a suitable rejection bound for your inputs.

## Train

Start from [dmpdd_ref2va.yaml](../../examples/train/configs/distribution_matching/minimax_h3/dmpdd_ref2va.yaml).
The example uses eight GPUs with SP8 and FSDP8. Memory requirements depend on joint
sequence length, reference count and checkpointing. The minimum GPU memory for this
public configuration has not been established.

Tile-128 sparse training requires the native Blackwell kernel and its backward operator.
Build/install the matching `fastvideo-kernel` package and use FlashAttention 4 for the
dense roles. Launch inside your allocated GPU environment:

```bash
FASTVIDEO_FA4=1 FASTVIDEO_VSA_PTX_BLACKWELL=1 \
  torchrun --standalone --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
  --config examples/train/configs/distribution_matching/minimax_h3/dmpdd_ref2va.yaml \
  --models.student.init_from /models/MiniMax-H3-ref2va \
  --models.teacher.init_from /models/MiniMax-H3-ref2va \
  --models.critic.init_from /models/MiniMax-H3-ref2va \
  --training.data.data_path /data/packed-ref2va \
  --training.checkpoint.output_dir /outputs/h3-dmpdd
```

Adjust `num_gpus`, `sp_size`, `hsdp_shard_dim`, and the torchrun process count together.
The effective batch size is `(world_size / sp_size) × train_batch_size ×
gradient_accumulation_steps`. Keep FP32 master weights; normal mixed-precision groups
compute in BF16 while declared precision-sensitive groups retain FP32 computation.

The example alternates four critic updates with one student update. Each iteration
updates exactly one role, so `generator_update_interval` must be at least 2; unlike
DMD2, an interval of 1 would never train the critic. The student uses
a 32-head fine grid, integrates the remaining tail for the distribution-matching
objective and applies a teacher-supervised PDD head loss with weight 0.3. Sampling
uses eight local blocks of four heads. Teacher and critic guidance are 2; critic
conditioning dropout is 0.1. The unconditional text/presentation branch is zeroed
while reference latents remain available. Gradient clipping applies to the student;
the critic's norm is measured without clipping. Non-finite gradient sanitation is
explicitly enabled in the example and reports corrections.

For pure PDD, use `fastvideo.train.methods.trajectory_matching.pdd.PDDMethod` with
student and teacher roles. DMPDD also supports `rollout_mode: simulate`; `data_latent`
requires real target latents. Existing DMD2 configurations retain their update cadence.

## Compilation and capacity padding

The example starts with compilation disabled. After verifying eager execution, enable
`training.model.enable_torch_compile` for dense regional compilation and set
`training.model.minimax_h3_student_compile: grad_only_full_block` for the sparse student.
The latter compiles gradient-enabled block forwards and keeps no-gradient rollouts
eager. It requires full non-reentrant activation checkpointing, FP32 masters and
`dynamic: true`, `fullgraph: true`, and a positive `recompile_limit`.

Optional `minimax_h3_compile_limit_fallback: true` keeps cached graphs active when
Dynamo reaches its limit. Checkpoint recomputation retains the original eager or
compiled execution mode. Compiler/kernel failures other than limit exhaustion propagate.
These controls require a PyTorch version exposing per-call `recompile_limit` and
`torch.compiler.config.dynamic_sources`.

`training.data.fixed_sequence_capacities` optionally pads each complete joint stream
to the smallest fitting capacity. Choose increasing multiples of 256 from your own
sequence lengths; records larger than the largest capacity are rejected. This requires
native shape bucketing and microbatch size 1. Padding is masked out of attention,
losses and normalization. It does not combine independent records. Cross-sample
packing is unsupported for this reference-conditioned recipe.

## Resume and export

Training checkpoints include optimizer/scheduler state, dataloader position and per-rank
RNG state. With `require_complete_training_checkpoint: true`, resume accepts only a
complete DCP checkpoint and all rank RNG snapshots. Use
`--training.checkpoint.resume_from_checkpoint latest` to continue the same output directory.
`reset_lr_on_resume: true` explicitly reapplies the YAML learning rates after restoration.

Export a deployable student from a saved checkpoint using the existing converter:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint /outputs/h3-dmpdd/checkpoint-100 \
  --output-dir /models/h3-dmpdd-student \
  --role student --verify \
  --base-model-revision hf://MiniMaxAI/MiniMax-H3@<revision>
```

The converter loads/reshards the role on a GPU; ensure it has enough memory for the
full model. The export records the widened PDD architecture in the component config.
The CPU streaming export utilities also support model-declared shapes and independent
inference checkpoint retention. Model-only exports cannot resume optimizer training.

Both export paths write `fastvideo_inference.json`, the sampling contract the FastH3
Ref2VA pipeline reads: the balanced partition of the student's finest trained grid
(`pdd_steps / block_size_min` fused blocks, eight for the example), its scheduler
shifts and its VSA settings. The contract names the base snapshot whose text encoder
and VAEs the export reuses. It comes from `training.checkpoint.inference_base_model_revision`
(or `--base-model-revision`), from a PDD initialization checkpoint's own contract, or
from an `init_from` inside the Hugging Face cache; set it explicitly for a local copy.
That pipeline samples only sparse `VIDEO_SPARSE_ATTN_H3` students, so other students
export without a contract. A student trained with `video_scheduler_shift` or
`audio_scheduler_shift` values that differ from its `init_from` scheduler configs is
refused at export, because inference samples with those configs.

In-training validation samples a PDD student with the balanced partition for each
`sampling_steps` value and, for sparse Ref2VA runs, the training reference keep rate.

No validation dataset is bundled. Evaluate your own held-out records and generated
media before relying on a newly initialized or distilled checkpoint. CPU regression
tests cover objectives, reference/padding behavior, configuration and checkpoint logic;
GPU correctness, memory and throughput must be checked on the intended topology.
