# MiniMax-H3-Image recipes

<RecipeFamily family="minimax_h3_image" />

Six-step text-to-image and reference editing for `MiniMaxAI/MiniMax-H3-Image`.
The DiT, Qwen3-VL encoder, and causal video VAE encoder are the MiniMax-H3
stack. The decoder is the image 2D ViT. Both VAE halves stay in FP32.

The checkpoint is guidance-distilled: only 6 steps and `guidance_scale` 1 are
valid, and a negative prompt is rejected. Sizes below 2048×2048 are generated
on a larger canvas and resized back. Weights load from `transformer_diffusers/`
and `vae_diffusers/`.

DiT feed-forward names still match the NVFP4 and MXFP8 filters already on main.
The text encoder is the H3 Qwen3-VL conditioner, so serialized NVFP4/FP8
checkpoints and `override_text_encoder_quant` apply unchanged. The six PDD
output heads stay full precision.

Sequential load is on by default, so the text encoder is released before the
DiT and VAE load. Pass `--no-h3-sequential-load` to keep them resident together.

## Text to image

```bash
python examples/inference/basic/basic_minimax_h3_image_t2i.py
```

## Reference edit

The prompt names references as `<Picture 1>`. The encoder label is `Picture 1: `
without angle brackets. Up to 16 references.

```bash
python examples/inference/basic/basic_minimax_h3_image_edit.py --image path/to/reference.png
```

## Serving

`examples/serving/openai_minimax_h3_image.yaml` serves
`POST /v1/images/generations` and `POST /v1/images/edits`. Both reject anything
other than 6 steps, guidance 1, and an empty negative prompt.

MiniMax-H3-Image is released under the MiniMax-H3-Image Community License,
which is not an OSI-approved open-source license. Output quality against the
official sampler was not measured.
