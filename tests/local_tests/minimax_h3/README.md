# MiniMax H3 validation

Local tests keep checks that need the pinned Diffusers source, published weights, or the public registry surface.
FastVideo-owned unit contracts belong under `fastvideo/tests/`.

## Reference

- Diffusers implementation: `https://github.com/huggingface/diffusers/pull/14355`
- Source checkout: `${MINIMAX_H3_OFFICIAL_REF_DIR:-$PWD/DiffusersMiniMaxH3}`
- Checkpoint: `MiniMaxAI/MiniMax-H3`

The reference helper verifies the pinned source and import origin. A missing checkout may skip a source-parity module;
that skip is not parity evidence.

## FastVideo unit contracts

```bash
pytest \
  fastvideo/tests/train/models/test_minimax_h3_lora.py \
  fastvideo/tests/train/models/test_minimax_h3_ref2va.py \
  fastvideo/tests/train/utils/test_dcp_to_diffusers_h3.py \
  fastvideo/tests/train/utils/test_lora_fsdp.py \
  fastvideo/tests/workflow/test_minimax_h3_ref2va_preprocess.py \
  fastvideo/tests/vaes/test_minimax_h3_video_vae_streaming.py \
  fastvideo/tests/stages/test_minimax_h3_vae_streaming.py -q
```

The LoRA ownership test starts two CPU/Gloo ranks, shards the adapter with
composable FSDP/DTensor, compares two synchronized optimization steps against
the exact unsharded mean-loss reference, and round-trips adapter state through
DCP. Export tests prove that Ref2VA replaces `transformer_ref/`, that LoRA is
merged back to native keys even inside a real PyTorch `CheckpointWrapper`, and
that strict native reload preserves the forward result. They also exercise
canonical Diffusers single-file and sharded safetensors layouts while retaining
the legacy filename for model plugins that have not opted in.

The complete local checkpoint may be used for a later explicit real-model
gate, but not on a 121 GiB unified-memory GB10: export gathers the roughly
62 GiB H3 transformer on CPU while the live model is still resident. Use a
machine with comfortably more than the combined model, gathered state, and
runtime working set. `--verify` releases the training graph before reload but
does not make the initial full-state gather streaming.

The narrower real-weight initialization gate has been validated on one 121 GiB
GB10. It loads only `transformer_ref`, keeps checkpointing disabled, and never
gathers a state dict. The 61.73 GiB checkpoint is first staged as a CPU mapping;
the loader then consumes and releases each source tensor while its CUDA/FSDP
state grows, avoiding two simultaneous complete copies. Rank-32 adapters and
runtime overhead bring the expected isolated unified-memory working set to
approximately 70--80 GiB. Run it in isolation after other GPU work has stopped:

```bash
cd /path/to/FastVideo
CUDA_VISIBLE_DEVICES=0 \
MINIMAX_H3_MODEL_ROOT=/path/to/MiniMax-H3 \
MINIMAX_H3_RUN_LORA_REAL_INIT=1 \
torchrun --standalone --nproc-per-node=1 \
  -m pytest tests/local_tests/minimax_h3/test_minimax_h3_lora_real_init.py -q -s
```

This asserts strict base loading, 312 LoRA wrappers and 624 trainable adapter
parameters, shared FSDP/DTensor placement with each base weight, CUDA residency,
and absence of per-layer CPU snapshots. The recorded GB10 run loaded 33.30B
parameters in 279.56 seconds with 62.08 GiB peak CUDA allocation and 62.50 GiB
peak reservation. It performs no forward or export.

## Registry smoke

```bash
pytest tests/local_tests/pipelines/test_minimax_h3_pipeline_smoke.py -q
```

## Pinned implementation parity

```bash
PYTHONPATH="${MINIMAX_H3_OFFICIAL_REF_DIR:-$PWD/DiffusersMiniMaxH3}/src:$PWD" pytest \
  tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py \
  tests/local_tests/minimax_h3/test_minimax_h3_packing.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_packing.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_media.py -v -s
```

## Checkpoint component parity

```bash
export MINIMAX_H3_MODEL_ROOT=/path/to/MiniMax-H3
export MINIMAX_H3_OFFICIAL_REF_DIR=/path/to/DiffusersMiniMaxH3

PYTHONPATH="$MINIMAX_H3_OFFICIAL_REF_DIR/src:$PWD" \
MINIMAX_H3_RUN_ENCODER_PARITY=1 \
pytest tests/local_tests/encoders/test_minimax_h3_qwen3_vl_parity.py -v -s

PYTHONPATH="$MINIMAX_H3_OFFICIAL_REF_DIR/src:$PWD" \
MINIMAX_H3_RUN_DIT_PARITY=1 \
MINIMAX_H3_RUN_VIDEO_VAE_PARITY=1 \
MINIMAX_H3_RUN_AUDIO_VAE_PARITY=1 \
pytest \
  tests/local_tests/transformers/test_minimax_h3_transformer_parity.py \
  tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py \
  tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py -v -s
```

With a gate enabled, missing CUDA, source, or weights is a failure. Recorded component evidence is exact for both DiT
partitions and the video VAE; audio decode has maximum absolute drift `2.4e-7`. The encoder gate compares the slim
forward's selected layer-50 hidden state bit-exactly against the same state from the official full stack across text,
image, and video inputs.

The video VAE test verifies the reference checkout at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` and compares the production CPU `uint8` `encode_pixels()` path against
the official posterior element by element.

## Video VAE memory benchmark

The benchmark uses one warmup and three measured runs with `vae_cpu_offload=True`. It reports absolute and
stage-incremental allocated/reserved CUDA peaks for every rank. For SP runs, the reported aggregate is explicitly the
sum of rank-local maxima, not a simultaneous node peak.

```bash
python tests/local_tests/vaes/benchmark_minimax_h3_video_vae_memory.py \
  --source-root "$PWD" --model-root "$MINIMAX_H3_MODEL_ROOT" \
  --revision-label candidate --operation encode

python -m torch.distributed.run --nproc_per_node=4 \
  tests/local_tests/vaes/benchmark_minimax_h3_video_vae_memory.py \
  --source-root "$PWD" --model-root "$MINIMAX_H3_MODEL_ROOT" \
  --revision-label candidate-sp4 --operation decode
```

Run the same script with `--source-root` pointed at the base checkout for a comparable baseline. The default workload
is deterministic `124 x 768 x 1344` video geometry with seed `20260803`; the JSON record includes source/model
revisions, software/allocator metadata, exact measurement boundaries, per-repetition values, and output shapes.

FastVideo joint audio/video generation and SP=1/SP=4 latent consistency have been validated. T2VA, FL2VA, and
Ref2VA video/audio latents match the pinned Diffusers pipeline exactly.
