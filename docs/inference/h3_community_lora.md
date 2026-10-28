# Community LoRAs for MiniMax-H3 and FastH3

FastVideo loads LoRAs published for MiniMax-H3 in the layouts that ComfyUI, fal, ai-toolkit, and musubi-tuner
trainers write. The loader detects the layout, converts it to FastVideo's module names, and applies the adapter to
the base or FastH3 transformer. It works for dense bf16 checkpoints and for NVFP4 and FP8 checkpoints.

!!! warning "Third-party weights"
    Community LoRAs are third-party weights. FastVideo does not ship or endorse any of them. Each adapter has its
    own licence, and it also inherits the MiniMax H3 Community License of the model it is applied to. Read both before
    you use the output.

## What is supported

| Aspect | Supported |
| --- | --- |
| Key prefixes | `diffusion_model.`, `model.diffusion_model.`, `transformer.`, `base_model.model.`, kohya `lora_unet_` (names with `.` flattened to `_`) |
| Factor names | `lora_A`/`lora_B` (PEFT, with or without an adapter-name infix) and `lora_down`/`lora_up` (kohya, ComfyUI) |
| Scaling | Per-module `alpha` (or `lora_alpha`) scaled by `alpha / rank`; no `alpha` key means scale 1 |
| Fused attention | `attn.qkv_proj` is split by output rows into `to_q`, `to_k`, `to_v`, which share `lora_A` |
| Attention and MLP | `attn.out_proj` to `attn.to_out`; `mlp.fc1`/`mlp.fc2` to `ff.fc_in`/`ff.fc_out` (the two SwiGLU halves of `fc1` are swapped to match FastVideo's order) |
| Blocks | `blocks.N` and `token_refiner.blocks.N` to `transformer_blocks.N` and `token_refiner.refiner_blocks.N` |
| Native layout | Adapters already in FastVideo's diffusers-style names load unchanged |
| Boundary projections | `audio_patch_proj`, `condition_proj`, `final_layer.*` are recognized by name; see [Limitations](#limitations) |

The loader fails instead of running an unadapted model. If an adapter reaches no layer, or fewer than half of its
modules, loading raises an error that lists the unmatched modules. Keys that match no known H3 layer are logged.

## Use a LoRA

Pass the adapter when you construct the generator. This applies to both Python and the CLI configuration.

```python
from fastvideo import VideoGenerator
from fastvideo.api import ComponentConfig, EngineConfig, GeneratorConfig, PipelineSelection, QuantizationConfig

generator = VideoGenerator.from_config(
    GeneratorConfig(
        model_path="FastVideo/FastVideo-FastH3-8-Step-V2",
        pipeline=PipelineSelection(
            components=ComponentConfig(
                lora_path="/path/to/community_h3_lora.safetensors",  # file, directory, or Hugging Face repo id
                lora_strength=0.8,
            ),
        ),
    ))
```

The same works for packed NVFP4 and FP8 checkpoints, for example FastH3-QAD-v2:

```python
generator = VideoGenerator.from_config(
    GeneratorConfig(
        model_path="FastVideo/FastH3-QAD-v2",
        engine=EngineConfig(
            quantization=QuantizationConfig(transformer_quant="NVFP4", layer_profile="h3_dit_vsa"),
        ),
        pipeline=PipelineSelection(
            components=ComponentConfig(lora_path="fal/MiniMax-H3-Realism-People-LoRA", lora_strength=0.8),
        ),
    ))
```

Packed NVFP4 exports require an explicit quantization profile. FastH3-QAD-v2 uses `h3_dit_vsa` for attention, MLP, and
compression-gate weights; the FFN-only V2 export uses `h3_dit_ffn`.

The FastH3 example scripts take the same options. `basic_fasth3_lora_preview.py` accepts `--lora-path` and
`--lora-strength`, and `examples/serving/openai_fasth3_lora.yaml` shows the YAML form.

`lora_strength` multiplies the adapter delta. `1.0` is the published scale and `0` is exactly the model without the
adapter. Values above `1.0` exaggerate the adapter and usually degrade quality. After construction, you can change
the strength with `generator.set_lora_adapter(nickname, strength=...)` only when the adapter runs as a side branch, described
next. Otherwise, create a new generator.

## LoRAs trained on base H3

Most community LoRAs are trained on base MiniMax-H3, which samples in 50 steps with classifier-free guidance. FastH3
is distilled to 8 steps without guidance. Style, character and camera-motion adapters usually carry over, but the
best strength can differ from the one recommended for base H3, so start at the published strength and adjust.

Step-reduction adapters (for example "turbo" or "acceleration" LoRAs) are not needed on FastH3, which already
samples in 8 steps. Stacking them on a distilled model is not supported.

## Quantized checkpoints

An NVFP4 or FP8 linear loaded from a packed export has no original dense weight to merge into. FastVideo keeps
the packed weight and its calibration untouched and runs the adapter next to it:

```
y = base(x) + strength * (alpha / rank) * B(A x)
```

`A` and `B` stay in bf16. Nothing is re-quantized, and strength `0` returns the base layer's output unchanged.
The extra cost is two small matmuls per adapted linear.

A dense bf16 checkpoint that FastVideo quantizes at load time (for example `transformer_quant: NVFP4` on a bf16
export) still merges the adapter first and quantizes afterwards, so the quantized weight includes the adapter.
If quantization retains the original dense weight, adapters continue to merge into that weight and FastVideo
rebuilds the packed buffers after each adapter change.

To make the adapter strength changeable at runtime on a dense checkpoint, set `FASTVIDEO_LORA_SIDE_BRANCH=1`. The
adapter then runs as a side branch for every layer, at the cost of the extra matmuls. See
[Environment variables](../contributing/env_vars.md).

## When the whole model does not fit

On a 32 GB card, and on any GPU where the text encoder and the DiT do not fit together, FastVideo loads the
conditioner first and the transformer afterward (`h3_sequential_load`). That is the same one-module-at-a-time path
the consumer profiles already use, for packed NVFP4 and for bf16. Pass `lora_path` at construction as usual. The
adapter is attached when the transformer arrives: as a bf16 side branch beside packed weights, or merged into a
dense bf16 transformer. You do not load the LoRA as a separate resident copy of the model.

## Per-request adapters on the OpenAI API

`fastvideo serve` can hold an allow-list of adapters and apply one per request on `POST /v1/videos/sync` (the whole
MP4 comes back in the response) and on the async `POST /v1/videos` job route. Name each adapter and its strength
range in the serve config; see `examples/serving/openai_fasth3_nvfp4_lora_allowlist.yaml`:

```yaml
generator:
  model_path: /models/FastVideo-FastH3-8-Step-V2.1-NVFP4-Consumer
  engine:
    quantization:
      transformer_quant: NVFP4  # required: per-request adapters need the packed NVFP4 side-branch layers
      layer_profile: h3_dit_vsa
lora:
  adapters:
    - id: fal/MiniMax-H3-Realism-People-LoRA
      path: fal/MiniMax-H3-Realism-People-LoRA
      default_strength: 0.8
      min_strength: 0.0
      max_strength: 1.5
      license: Apache-2.0
      source_url: https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA
```

An allow-list cannot be combined with a startup adapter (`generator.pipeline.components.lora_path`); the server
refuses to start with both. Without an allow-list, the startup-adapter behavior is unchanged.

`GET /v1/model_info` reports the allow-list in its `lora` block, never the load paths. A request selects one
adapter with the vLLM-style `lora` object:

```bash
curl -s http://127.0.0.1:8000/v1/videos/sync \
  -H 'content-type: application/json' \
  -d '{"prompt": "...", "size": "1344x768", "num_frames": 124, "seed": 7,
       "lora": {"name": "fal/MiniMax-H3-Realism-People-LoRA", "scale": 0.8}}' \
  -o out.mp4
```

`name` (or `lora_name`, `adapter`) must be an allow-listed `id`. `scale` (or `lora_scale`) is the strength; it
defaults to the adapter's `default_strength` and must lie in `[min_strength, max_strength]`. A client-supplied `path`
is rejected. A bad selection gets HTTP 400 before any job is stored.

Strength `0` is exactly the base model, with output identical to a request without `lora`. A request without `lora`
sets the last-used adapter to strength 0 instead of unloading it, so one request never inherits another's
adapter. The server applies the adapter under the same lock as the generation, so requests never interleave. If
a change fails, that request fails with the loader's message, and the server sets an adapter it already holds back
to strength 0 before the next request; it never renders with an adapter the request did not select.

Per-request adapters need side-branch LoRA layers, so the server refuses an allow-list unless the model is a packed
NVFP4 export (a `transformer/nvfp4_weights.safetensors` file with `transformer_quant: NVFP4`) or
`FASTVIDEO_LORA_SIDE_BRANCH=1` is set. At startup it also checks that every adapter resolves to a safetensors file
and is purely low-rank: adapters with dense replacement payloads (`.diff` and `.set_weight` tensors, as in the
[FastH3 adapter](fasth3-distilled.md)) can only be applied when the model loads, and the server refuses to start with
one. A side-branch layer holds one adapter at a time. Linears that a packed export leaves unquantized still merge the
adapter into their weights; switching to another adapter first restores each of them, including those the new
adapter does not cover.

## Limitations

- Block-pruned models (FastH3 Trim, 42 of 50 blocks) are not supported yet. Community LoRAs index the full
  50-block model, so their block numbers do not line up with a renumbered pruned transformer. The Trim
  `config.json` records `kept_block_indices`, but the loader does not read it, so do not load a community LoRA on a
  pruned model.
- Only low-rank factors are converted. Dense `.diff` and `.set_weight` payloads follow the existing
  [FastH3 adapter](fasth3-distilled.md) path and must be given at construction.
- LoRA targets the attention, feed-forward, and AdaLN projections of the transformer and token refiner, plus
  the final AdaLN projection (`final_layer.adaln_proj.linear`). The input/output projections (`audio_patch_proj`,
  `condition_proj`, `final_layer.video_out`, `final_layer.audio_out`) are mapped by name but are not adapted;
  they appear in the log as unmatched keys.
- A side-branch layer holds one adapter at a time; accumulating a second adapter on a packed layer raises an error.
- Adapters that require additional conditioning need that input path in FastVideo. For example, character-swap
  adapters trained with aligned guide latents cannot reproduce their full workflow using native references alone;
  the ComfyUI Add Guide path is not supported. Loading an adapter successfully does not establish task quality.
- Adapters trained against another base model are not checked for semantic compatibility. FastVideo verifies layer
  names and tensor shapes only.
