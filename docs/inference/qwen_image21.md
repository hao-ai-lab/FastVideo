# Qwen-Image-2.1

The native FastVideo implementation includes text-to-image generation, image
editing, reference-to-image generation with 1–10 ordered images, and RGBA
generation and editing. Annotations and separate mask images use the same
ordered-reference path. Describe their meaning in the prompt.

This port is implemented and has CPU component and pipeline contract checks.
Released-weight CUDA parity, output quality and the 24 GB memory target are
pending validation. The repository runbook is
`tests/local_tests/qwen_image21/README.md`, with evidence and remaining gates in
`tests/local_tests/qwen_image21/PORT_STATUS.md`.

## Run on a GPU

Use one CUDA GPU, with `tp_size=sp_size=1`. The initial hardware target is a
24 GB GPU and at least 96 GB host RAM. Text layers and DiT blocks offload to
CPU, the VAE tiles, and prefix keys/values stay on CPU by default. These settings
reduce GPU residency; the memory limit has not yet been measured.

A desktop RTX 5090 has [32 GB of VRAM](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
and is a reasonable validation target with these settings. Plan 96–128 GB host
RAM for the reference-heavy cases. Use a PyTorch CUDA build with Blackwell
support. CPU transfers can limit throughput; no 5090 timing has been measured.

Provide a complete local Diffusers-format `Qwen/Qwen-Image-2.1` checkpoint, or
let the Hugging Face loader fetch it on the GPU machine. Model weights are
subject to Qwen's checkpoint license. The base checkpoint contains the processor,
Qwen3-VL encoder, DiT, VAE and scheduler. No conversion is needed.

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --model-path /path/to/Qwen-Image-2.1 \
  --prompt 'A red ceramic teapot, isolated on a transparent background.' \
  --output outputs/teapot.png
```

For a single-image edit:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --model-path /path/to/Qwen-Image-2.1 \
  --reference source.png \
  --prompt 'Change the shirt to blue; retain the subject and composition.'
```

For reference-to-image or an annotation-guided edit, repeat `--reference` in the
order referred to by the prompt:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --reference subject.png --reference style.png \
  --prompt 'Paint the subject from image 1 in the style of image 2.'
```

```bash
fastvideo generate --config examples/inference/basic/qwen_image21_edit.yaml \
  --request.inputs.references '["source.png", "mask.png"]' \
  --request.prompt 'In image 1, replace the area highlighted by image 2 with flowers.'
```

Guided mask edits follow the released model workflow. Pixels outside the mask
are not guaranteed to remain identical. The mask is supplied as a reference
image, alongside the source, with instructions in the prompt.

## Python API

The example above uses `VideoGenerator.from_config` and `GenerationRequest`.
`InputConfig.references` accepts local paths, PIL images, or uint8 image arrays.
The legacy API also supports `generate_video(prompt=..., references=[...])`.
For one reference, `image_path` or `pil_image` is a shortcut. Supply only one
of these input sources.

Outputs are four-channel PNGs. Reference alpha remains intact for VAE encoding;
only the vision encoder's copy is composited over white. Ask for a transparent
background in the prompt when you want the model to generate transparency.

Defaults are 1024×1024, 40 steps, one image and guidance 1. Output dimensions and
`reference_resolution` must be positive multiples of 32. References are resized
to approximately `reference_resolution²` pixels while retaining their aspect
ratio. A 2K output can keep the reference resolution at 1024.

Set `true_cfg_scale > 1` with an explicit `negative_prompt` for two-branch CFG.
An empty string is an explicit negative prompt. `guidance_scale` stays at 1;
this checkpoint does not use embedded guidance.

Set `request.sampling.use_kv_cache: false` to recompute prefixes each step.
The default CPU cache avoids holding every reference prefix on the GPU; it
falls back to uncached execution if a host cache allocation fails. Request
caches are cleared after success or failure. GPU cache placement is available
through `QwenImage21PipelineConfig(kv_cache_device="cuda")` in a pipeline config
file, for hardware with enough VRAM.

## Scope

The first port is single GPU, batch one, single frame and unquantized. SP/TP,
custom attention processors, LoRA, prompt rewriting and HTTP serving are not
included. The prompt rewriter's separate 9B checkpoints can be integrated later.
The first GPU gates cover all modes at 1024, followed by 2K text-to-image;
2K with ten references is outside the initial memory gate.

The numerical source is [Diffusers at revision
578c9b2](https://github.com/huggingface/diffusers/tree/578c9b2c6636ab2424a0e56186268b83623656b2).
Runtime numerical models are native FastVideo classes. Transformers supplies
the processor/tokenizer; reference Diffusers model classes are used only by
parity tests.
