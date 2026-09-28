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
  --num-gpus 4 --vsa-kernel sm100a \
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
they can be evaluated separately. On hardware without the sm100a extension,
use `--vsa-kernel triton`; compare outputs and performance before adopting that
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

## Apple Silicon

`examples/inference/basic/mlx_fasth3.py` stays on the four-step preview and its
uniform AdaLN cache. `mlx_fasth3_8step.py` is the eight-forward MLX recipe. It
reads the same `fastvideo_inference.json` rungs and shifts, and it expects an
MLX DiT whose AdaLN cache was converted from that contract. Reuse the preview
snapshot's VAE, audio VAE, text encoder, and tokenizer; only the DiT and the
sidecar change. Rank-reduced AdaLN checkpoints are unchanged and are not
produced by the MLX converter.
