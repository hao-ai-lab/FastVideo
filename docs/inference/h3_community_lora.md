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
from fastvideo.api import ComponentConfig, GeneratorConfig, PipelineSelection

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

The FastH3 example scripts take the same options. `basic_fasth3_lora_preview.py` accepts `--lora-path` and
`--lora-strength`, and `examples/serving/openai_fasth3_lora.yaml` shows the YAML form.

`lora_strength` multiplies the adapter delta. `1.0` is the published scale and `0` is exactly the model without the
adapter. Values above `1.0` exaggerate the adapter and usually degrade quality. After construction, you can change
the strength with `generator.set_lora_adapter(nickname, strength=...)` only when the adapter runs as a side branch, described
next. Otherwise, create a new generator.

## Quantized checkpoints

An NVFP4 or FP8 linear stores a packed weight whose scales were calibrated for the original model. Merging a LoRA
into it would mean quantizing the weight again, and the calibration would be lost. FastVideo therefore keeps the
packed weight untouched and runs the adapter next to it:

```
y = base(x) + strength * (alpha / rank) * B(A x)
```

`A` and `B` stay in bf16. Nothing is re-quantized, and strength `0` returns the base layer's output unchanged.
The extra cost is two small matmuls per adapted linear.

A dense bf16 checkpoint that FastVideo quantizes at load time (for example `transformer_quant: NVFP4` on a bf16
export) still merges the adapter first and quantizes afterwards, so the quantized weight includes the adapter.

To make the adapter strength changeable at runtime on a dense checkpoint, set `FASTVIDEO_LORA_SIDE_BRANCH=1`. The
adapter then runs as a side branch for every layer, at the cost of the extra matmuls. See
[Environment variables](../contributing/env_vars.md).

## Limitations

- Block-pruned models (FastH3 Trim, 42 of 50 blocks) are not supported yet. Community LoRAs index the full
  50-block model, so their block numbers do not line up with a renumbered pruned transformer. The Trim
  `config.json` records `kept_block_indices`, but the loader does not read it, so do not load a community LoRA on a
  pruned model.
- Only low-rank factors are converted. Dense `.diff` and `.set_weight` payloads follow the existing
  [FastH3 adapter](fasth3-distilled.md) path and must be given at construction.
- LoRA targets the attention projections, the feed-forward projections, and the AdaLN projections of the
  transformer and token refiner. Boundary projections (`audio_patch_proj`, `condition_proj`, `final_layer.*`) are
  mapped by name but are not adapted; they appear in the log as unmatched keys.
- A side-branch layer holds one adapter at a time; accumulating a second adapter on a packed layer raises an error.
- Adapters trained against another base model are not checked for semantic compatibility. FastVideo verifies layer
  names and tensor shapes only.
