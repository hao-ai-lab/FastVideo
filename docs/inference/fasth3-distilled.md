# FastH3 distilled checkpoint schedules

Base MiniMax-H3 still uses the scheduler shifts in its checkpoint (video 12,
audio 3), BF16 text encoding, and the existing uniform schedule. The default
`basic_fasth3.py` example still targets the four-forward preview. Selecting a
shift-10 eight-forward checkpoint is an explicit choice of model and recipe;
it does not change either default or enable NVFP4.

## Eight-forward T2AV recipe

Use the full exported model directory, not just its `transformer` subdirectory.
For a checkpoint trained with video/audio shifts 10/3, VSA sparsity 0.8,
64-token tiles, and the DMD rungs
`[999, 874, 749, 624, 500, 375, 250, 125]`:

```bash
python examples/inference/basic/basic_fasth3.py \
  --model-path /path/to/FastH3-8step \
  --prompt 'A slow cinematic drone shot glides over a coastal town; gulls call over the harbor.' \
  --steps 9 --num-gpus 4 \
  --vsa-sparsity 0.8 --vsa-tile-size 64 --vsa-kernel sm100a \
  --profile strict --no-inference-torch-compile --no-compile-vae \
  --height 768 --width 1344 --num-frames 124 \
  --output outputs/fasth3-8step
```

`--steps` is the number of sigma-grid points, including the terminal zero.
Nine points run exactly eight transformer forwards. `--steps 8` is not the
same recipe. The rungs are unshifted noise levels on the 1000-step training
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

This documents execution support, not checkpoint publication or quality approval.
Release requires verification of the final checkpoint revision, video/audio
comparison against the baseline, and an explicit release decision.
