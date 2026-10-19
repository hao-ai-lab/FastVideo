# FastH3 T2AV DMD training

This recipe exposes the data-free DMD2 training method used for
[FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2),
with the current modular trainer and optional runtime optimizations. It trains
text-to-audio-video generation from prompt embeddings and requested geometry.
It does not require training video or audio latents.

The public recipe preserves the release's algorithm and schedule. It does not
include its training corpus or reproduce the released weights from a new corpus.
See [prompt preprocessing](minimax_h3_t2av_preprocess.md) to prepare your inputs.

## Release recipe

Use [`t2av_dmd.yaml`](../../examples/distill/minimax_h3/t2av_dmd.yaml).
The student uses VSA-H3; the frozen teacher and trainable critic use dense
attention. Each optimizer step updates one role: four critic steps followed by
one student step in each five-step cycle. Both active roles use gradient norm clipping at 1.0.
The existing `DMD2Method` remains available with its established update cadence;
this recipe selects `CarriedDMDMethod` explicitly.

| Setting | Release value |
| --- | --- |
| Student transformer forwards | 8 |
| Unshifted DMD rungs | `[999, 874, 749, 624, 500, 375, 250, 125]` |
| Video / audio scheduler shifts | 10 / 3 |
| Student VSA sparsity / tile size | 0.8 / 64 |
| Rollout | Simulated ODE, carried between optimizer steps |
| Carry slots / gradient accumulation | 4 / 4 |
| Score sampling | Continuous; shift 2.4; maximum warp 0.999 |
| Critic regression | Clean sample (`x0`) |
| Teacher guidance / conditioning dropout | 1.0 / 0.0 |
| Student / critic learning rate | `2e-6` / `2e-6` |
| Master weights | FP32 |

Carried rollouts retain their own geometry and conditioning. One independent
slot per accumulation microbatch prevents a batch fetched at a different shape
from changing an unfinished trajectory. Carry state is transient: a resumed run starts fresh staggered trajectories.
It does not continue an in-flight trajectory bit for bit.

The recipe uses 64 GPUs, sequence parallelism 4 and accumulation 4, giving a
global batch of 64. Changing that topology or accumulation changes the effective
batch unless adjusted together. The minimum GPU memory for this rebased trainer
has not been measured.

Launch on your allocated GPU workers after setting the model, data and output
paths. With native kernels installed on data-center Blackwell, for example,
use eight workers and eight GPUs per worker:

```bash
FASTVIDEO_FA4=1 FASTVIDEO_VSA_PTX_BLACKWELL=1 torchrun --nnodes=8 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  -m fastvideo.train.entrypoint.train \
  --config examples/distill/minimax_h3/t2av_dmd.yaml
```

The YAML uses local placeholders for a complete converted MiniMax-H3 checkpoint
and your preprocessed dataset. All three model roles must reference a compatible
base checkpoint. Training and validation use separate video and audio shifts;
do not replace them with a shared `flow_shift`.

## Runtime optimizations

Dense teacher/critic regional compilation is enabled in the release recipe.
The port also supports native Blackwell VSA backward, BF16 H3 QKV communication,
sharded FP32-sensitive modules, and fresh training tile buffers. Native shape
bucketing groups equal geometries across data-parallel ranks.

These options are disabled by default and should be measured on your shapes:

- `training.data.fixed_sequence_capacities`: ascending multiples of 256. Pad
  each joint stream to its smallest fitting capacity while preserving real-row
  attention masks and losses. Oversize examples fail rather than truncate.
- `training.model.minimax_h3_activation_offload_min_tokens`: save activations in
  pinned CPU memory above a token threshold. This trades memory for transfers.
- `training.loop.fsdp_defer_backward_finalize`: defer FSDP finalization until
  the last accumulation backward while retaining gradient reductions.

The newer sparse student compilation policy is a separate experimental path:

```yaml
training:
  vsa:
    sparsity: 0.8
    tile_size: 128
    cache_tile_buf: false
  model:
    minimax_h3_student_compile: grad_only_full_block
    minimax_h3_compile_limit_fallback: true
    minimax_h3_accumulated_recompile_limit: 1024
    torch_compile_kwargs:
      dynamic: true
      fullgraph: true
      recompile_limit: 32
```

Apply those fields to a copy of the release YAML. This policy requires native
Blackwell kernels, FP32 masters, full non-reentrant checkpointing and full-weight
FSDP roles. It compiles gradient-enabled student blocks, keeps no-grad rollouts
eager, and preserves checkpoint replay mode when the compile cache fills.
Tile 128 changes sparse region selection, so it is not the release's tile-64
recipe. Validate outputs and gradients before using it for a training run.

No new MFU or speedup claim is made for this port. CPU tests cover control flow,
losses, carry state, configuration, masks and checkpoint metadata. Distributed
training, resume, native CUDA parity, compiled backward and generated audio/video
still require GPU validation on this branch.

## Checkpoints and inference

Resumable DCP checkpoints retain model state, both optimizers, schedulers and
RNG. Carried trajectories restart after resume. Strict completion checking excludes partially written checkpoints.
Inference exports select the student and include its DMD ladder and scheduler
shifts. Export writes replace linked configuration files without modifying the
base model.

For a manual export from resumable state:

```bash
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint outputs/h3_t2av_dmd/checkpoint-1000 \
  --output-dir outputs/h3_t2av_dmd/export-1000
```

See [FastH3 distilled inference](../inference/fasth3-distilled.md). Eight model
forwards require nine sigma-grid points, including the terminal zero. Configure
80% sparsity and tile 64 for the release recipe; the schedule sidecar does not
silently select attention settings.
