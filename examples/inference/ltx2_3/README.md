# LTX-2.3 distilled inference configs

Ready-to-run [`fastvideo generate`](../../..) run configs for the LTX-2.3
distilled model (`FastVideo/LTX-2.3-Distilled-Diffusers`), covering both
workloads (t2v / i2v), both two-stage step schedules (`5+2`, `8+3` = denoise
+ refine), and four resolutions.

```bash
fastvideo generate --config examples/inference/ltx2_3/t2v_8s3_1280x832.yaml
```

Each config is self-contained (no preset registry needed): the two-stage
refine is wired via `generator.pipeline.preset_overrides.refine`, and the
base sampling knobs live under `request.sampling`. The refine upsampler
auto-resolves from the model's `spatial_upscaler`.

## Configs

| workload | schedule | resolution (HxW) | file |
|---|---|---|---|
| t2v | 5+2 | 1280x832 | `t2v_5s2_1280x832.yaml` |
| t2v | 5+2 | 1024x1536 | `t2v_5s2_1024x1536.yaml` |
| t2v | 5+2 | 768x1280 | `t2v_5s2_768x1280.yaml` |
| t2v | 5+2 | 512x768 | `t2v_5s2_512x768.yaml` |
| t2v | 8+3 | 1280x832 | `t2v_8s3_1280x832.yaml` |
| t2v | 8+3 | 1024x1536 | `t2v_8s3_1024x1536.yaml` |
| t2v | 8+3 | 768x1280 | `t2v_8s3_768x1280.yaml` |
| t2v | 8+3 | 512x768 | `t2v_8s3_512x768.yaml` |
| i2v | 5+2 | 1280x832 | `i2v_5s2_1280x832.yaml` |
| i2v | 5+2 | 1024x1536 | `i2v_5s2_1024x1536.yaml` |
| i2v | 5+2 | 768x1280 | `i2v_5s2_768x1280.yaml` |
| i2v | 5+2 | 512x768 | `i2v_5s2_512x768.yaml` |
| i2v | 8+3 | 1280x832 | `i2v_8s3_1280x832.yaml` |
| i2v | 8+3 | 1024x1536 | `i2v_8s3_1024x1536.yaml` |
| i2v | 8+3 | 768x1280 | `i2v_8s3_768x1280.yaml` |
| i2v | 8+3 | 512x768 | `i2v_8s3_512x768.yaml` |

## Overriding without editing a file

Dotted overrides (prefixes `generator.` / `request.`) let you tweak any field:

```bash
# swap prompt
fastvideo generate --config examples/inference/ltx2_3/t2v_8s3_1280x832.yaml \
  --request.prompt "a red fox running through fresh snow"

# change output path / gpu count
fastvideo generate --config examples/inference/ltx2_3/t2v_5s2_512x768.yaml \
  --request.output.output_path outputs/preview.mp4 \
  --generator.engine.num_gpus 4
```

## i2v

The `i2v_*` configs take a first-frame image via
`request.extensions.ltx2_images` (`[[path, frame_offset, weight]]`). Edit the
path in the file, or override it:

```bash
fastvideo generate --config examples/inference/ltx2_3/i2v_8s3_1280x832.yaml \
  --request.extensions.ltx2_images '[["/data/portrait.jpg", 0, 1.0]]'
```

## Schedules

`5+2` is the fast preview schedule; `8+3` is the higher-quality distilled
recipe. Refine (`preset_overrides.refine.num_inference_steps`) only accepts 2
or 3 steps.

