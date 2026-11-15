# FastH3 distilled checkpoint schedules

Base MiniMax-H3 still uses the scheduler shifts in its checkpoint (video 12,
audio 3), BF16 text encoding, and the existing uniform schedule. The default
`basic_fasth3.py` example still targets the four-forward preview. Selecting a
shift-10 eight-forward checkpoint is an explicit choice of model and recipe;
it does not change either default or enable NVFP4.

## Eight-forward T2AV recipe

The public checkpoint is
[`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)
(MiniMax H3 Community License), trained with video/audio shifts 10/3, VSA
sparsity 0.8, 64-token tiles, and the DMD rungs
`[999, 874, 749, 624, 500, 375, 250, 125]`. `basic_fasth3_8step.py` pins that
checkpoint and recipe as defaults; it shares the preview example's CLI, so every
other flag works unchanged:

```bash
python examples/inference/basic/basic_fasth3_8step.py \
  --prompt 'A slow cinematic drone shot glides over a coastal town; gulls call over the harbor.' \
  --num-gpus 4 --vsa-kernel ptx_blackwell \
  --profile strict --no-inference-torch-compile --no-compile-vae \
  --height 768 --width 1344 --num-frames 124 \
  --output outputs/fasth3-8step
```

Pass `--model-path` to use a local snapshot of the full export (not just its
`transformer` subdirectory). `--steps` is the number of sigma-grid points,
including the terminal zero; nine points run exactly eight transformer
forwards, and the script rejects any other value because the checkpoint's
ladder has eight rungs. The rungs are unshifted noise levels on the 1000-step training
clock; each scheduler applies its own shift once, and the transformer receives
H3 clean-time values (`1 - sigma`). A uniform nine-point grid is not a substitute
for those rungs.

Compilation and H3 fusions are disabled above to establish an eager reference;
they can be evaluated separately. Where the `ptx_blackwell` kernel is unavailable
(non-Blackwell GPUs, or a build without it), use `--vsa-kernel triton`; compare outputs and performance before adopting that
backend. This recipe is T2AV-only, not a distilled `transformer_ref` model.

## Export metadata and validation

The export's `fastvideo_inference.json` supplies the trained ladder. The
schedule fields of `fasth3-inference-contract-v1` are:

```json
{
  "schema_version": "fasth3-inference-contract-v1",
  "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
  "num_inference_steps": 9,
  "transformer_forwards": 8,
  "video_scheduler_shift": 10.0,
  "audio_scheduler_shift": 3.0
}
```

The loader keeps this file when downloading the selected H3 components from
Hugging Face. It checks that the two declared shifts agree with
`scheduler/scheduler_config.json` and `audio_scheduler/scheduler_config.json`.
Missing/invalid rungs, inconsistent counts, or an explicit conflicting ladder
are errors. The denoiser rejects a request with the wrong number of grid points.
The metadata does not silently change request dimensions, step count, attention
backend, sparsity, precision, or offload/compile settings: set those explicitly
as above.

For exports without this sidecar, an explicit ladder is supported via
`MiniMaxH3PipelineConfig.dmd_denoising_steps`, or through the typed API's
`PipelineSelection(experimental={"dmd_denoising_steps": [...]})`. The shifts
still come from the checkpoint scheduler configs. Keep generic `flow_shift`
unset: H3 has separate video and audio shifts, not one shared shift.

This documents execution support for the published checkpoint. It is not a
quality claim: compare video/audio output against base MiniMax-H3 on your own
prompts before adopting it.

## Ref2VA PDD students

A Parallel Decoding Distillation (PDD) student widens the transformer's two
output projections to `pdd_steps` heads, one per interval of a fixed fine
time grid on `[0, 0.999]`. Each transformer forward fuses a block of
consecutive heads into their integration-weighted mean, and an ordinary Euler
step over the block's two node sigmas applies it. The FastH3 OmniRef PDD-8
student is a Ref2VA (`transformer_ref`) student with 32 heads, sampled in
eight blocks of four.

Its export carries only what distillation changed:

| Path | Contents |
| --- | --- |
| `transformer_ref/` | Widened `proj_out` and `audio_proj_out`, trained VSA compression gates; `config.json` records `pdd_steps` |
| `scheduler/`, `audio_scheduler/` | Video and audio shifts (12 and 3) |
| `fastvideo_inference.json` | The sampling contract below |
| `modular_model_index.json` | The Diffusers manifest |

The text encoder, tokenizer, processor, and both VAEs are base MiniMax-H3's.
`basic_fasth3_omniref_pdd.py` composes the two into one local directory of
symlinks and runs the recipe the contract records. `--model-path` is the
export, as a local directory or a Hugging Face repo id. The base comes from
the revision the contract pins in `base_model_revision`, or from
`--base-model-path`:

```bash
python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path <local export directory or Hugging Face repo id> \
  --video reference.mp4 --image character.png \
  --prompt 'The dancer from the video performs the routine in the pictured outfit.' \
  --height 480 --width 832 --num-frames 124 \
  --output outputs/fasth3-omniref-pdd
```

References are ordered: pass `--image`, `--video`, and `--audio` in the order
the prompt refers to them. At least one image or video is required.

A PDD export uses `fasth3-inference-contract-v1` with PDD fields in place of
the DMD ladder:

```json
{
  "schema_version": "fasth3-inference-contract-v1",
  "model_type": "ref2va",
  "transformer_component": "transformer_ref",
  "pdd_steps": 32,
  "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
  "num_inference_steps": 8,
  "transformer_forwards": 8,
  "grid_max_t": 0.999,
  "video_scheduler_shift": 12.0,
  "audio_scheduler_shift": 3.0,
  "guidance_scale": 1.0,
  "attention_backend": "VIDEO_SPARSE_ATTN_H3",
  "vsa_sparsity": 0.9,
  "vsa_tile_size": 128,
  "vsa_ref_policy": "p2_multi_region",
  "vsa_ref_keep_rate": 0.1
}
```

The export also records `schema`, `conditioning`, and `base_model_revision`,
the base snapshot it was distilled against, as `hf://<repo id>@<revision>`
(for example `hf://MiniMaxAI/MiniMax-H3@<commit>`); any other form is an
error. FastVideo reads the file once, when the run's arguments are built and
before any weights load. The file must carry exactly these fields: a missing
or unknown field is an error. Fields that repeat a value stored elsewhere must
equal it: `pdd_steps` equals `transformer_ref/config.json`, the shifts equal
the scheduler configs, and `num_inference_steps` and `transformer_forwards`
equal the block count of `pdd_step_indices`, a strictly increasing partition
of the grid from 0 to `pdd_steps`. Only `MiniMaxH3Ref2VAModularPipeline` runs
the export. A `transformer_ref` whose `config.json` sets `pdd_steps` without
this file is an error.

Each sampling setting has one source, the file:

| Setting                         | Run leaves it unset    | Run sets a different value                     |
| ------------------------------- | ---------------------- | ---------------------------------------------- |
| `pdd_step_indices`              | Taken from the file    | Error                                          |
| `num_inference_steps` (request) | Set to the block count | Error                                          |
| `attention_backend`             | Taken from the file    | Error, including `FASTVIDEO_ATTENTION_BACKEND` |
| `VSA_tile_size`                 | Taken from the file    | Error                                          |
| `VSA_sparsity`                  | Taken from the file    | The run's value, with a warning                |
| `vsa_ref_keep_rate`             | Taken from the file    | The run's value, with a warning                |

A request leaves `num_inference_steps` unset only when it is parsed from a
mapping or a config file. A `GenerationRequest` built in Python counts every
field as set, so it must pass the block count. The trained compression gates
of `transformer_ref` load only under `VIDEO_SPARSE_ATTN_H3`, so the attention
backend cannot change. `dmd_denoising_steps` must stay unset.

A MiniMax-H3 checkpoint without PDD fields, such as a DMD export, whose
transformer carries VSA compression gates (`to_gate_compress` weights) also
runs only with `VIDEO_SPARSE_ATTN_H3`. The pipeline reads the shard index (or
the safetensors headers) and rejects any other backend, including automatic
selection, before it loads any component.

### Reference-video sparsity

Every PDD contract sets `"vsa_ref_policy": "p2_multi_region"`, the only value FastVideo accepts, which means:
`VIDEO_SPARSE_ATTN_H3` tiles every reference video as its own sparse region,
in place in the packed sequence. Each video query keeps `vsa_ref_keep_rate` of
every reference video's tiles and `1 - VSA_sparsity` of the target video's
tiles. Text, audio, and image references stay dense. With `VSA_sparsity` 0,
every region is dense and the reference keep rate has no effect. Checkpoints
other than PDD students keep every conditioning row dense.

### Hardware

The contract's 128-token tiles, `(4, 4, 8)`, run only on the sm_100a/sm_103a
CUDA block-sparse kernel (B200, B300, GB200, GB300) of a fastvideo-kernel
build with the Blackwell VSA extension. There is no Triton fallback for tile
128: the backend raises instead. Tile 128 runs eagerly. Regional compile
requires 64-token tiles. With `--num-gpus` above 1 the example shards the DiT
across the GPUs (FSDP) and splits the sequence across them.

On GB200, a 480x832, 124-frame request fits on one GPU: the eight forwards
take about 54 s, and device memory in use peaks near 103 GiB. A 768x1344,
345-frame request with `--num-gpus 4` takes about 36 s for the eight
forwards. With the DiT sharded, peak allocated memory is about 66 GiB per GPU
(about 95 GiB in use), against 83 GiB (103 GiB) with a full DiT copy on each
GPU; the output is identical. Model loading and decoding come on top of this.

## Token reduction

`--token-reduction` is an opt-in, training-free speed mode that follows the
`--token-reduction` mode of [antirez/h3.c](https://github.com/antirez/h3.c).
Inside DiT blocks 4 to 29, every two horizontally adjacent tokens of the
generated video are averaged into one, so those blocks see the video at half
width. During the first 4 of the 8 forwards, where the coarse layout forms,
the interval extends to block 39. Before the first full-width block, each
token gets back its own value plus the update its pair received. Text, audio,
reference and condition tokens are never paired.

FastH3 8-Step V2 and the OmniRef PDD export both support it:

```bash
python examples/inference/basic/basic_fasth3_8step.py \
  --prompt 'Two people walk along a beach at sunset, talking and laughing' \
  --num-gpus 4 --token-reduction

python examples/inference/basic/basic_fasth3_omniref_pdd.py \
  --model-path <OmniRef export> --image reference.png \
  --prompt '<Picture 1> ...' --num-gpus 4 --token-reduction
```

In a recipe, set `pipeline.experimental.h3_token_reduction` to
`{enabled: true}`. The optional keys `blocks`, `early` and `update_scale`
match the `--token-reduction-blocks`, `--token-reduction-early` and
`--token-reduction-scale` flags. Malformed values, and intervals past the
checkpoint's DiT depth, fail before any weights load.

Measured on one GB200 tray with 4 GPUs, 1344x768, 124 frames and BF16
weights, after warm-up:

| Checkpoint | Denoising, 8 forwards | Whole request |
|---|---|---|
| FastH3 8-Step V2, default profile with regional compile | 5.08 s to 3.53 s, 30.6% less | 9.13 s to 7.58 s, 17.0% less |
| FastH3 8-Step V2, regional compile off | 6.10 s to 4.26 s, 30.2% less | 10.18 s to 8.41 s, 17.4% less |
| OmniRef PDD with one image reference, eager | 12.07 s to 9.33 s, 22.7% less | 24.67 s to 21.62 s, 12.3% less |

The saving follows how many rows the middle blocks drop. A V2 request has a
short text and audio prefix, so the reduced blocks see about half the rows.
An OmniRef request also carries the reference tokens, which stay unpaired, so
it saves less. Decoding takes the same time either way.

The reduced video keeps the quality of the unreduced one, but it is not the
same video. Most scenes keep their overall layout, while details can change,
such as one of two people in a scene or the venue of a concert. The
checkpoints were not trained with token reduction, and fine-tuning with it
enabled is the expected way to keep the original shot. Only the GB200 BF16
path has been validated so far; the NVFP4 and FP8 consumer profiles have not
been tested with it.

## Apple Silicon

`mlx_fasth3.py` stays on FastH3 V1 and its uniform AdaLN cache.
`mlx_fasth3_8step.py` is the eight-forward MLX recipe for FastH3 V2. It
reads the same `fastvideo_inference.json` rungs and shifts, and it expects an
MLX DiT whose AdaLN cache was converted from that contract. Reuse the V1
snapshot's VAE, audio VAE, text encoder, and tokenizer; only the DiT and the
sidecar change. Rank-reduced AdaLN checkpoints are unchanged and are not
produced by the MLX converter.

Install is in the
[MLX install guide](../getting_started/installation/mlx.md). Generation and
serving are in the [MiniMax H3 cookbook](../cookbook/minimax-h3.md) and the
[H3 server guide](../cookbook/openai-api.md).
