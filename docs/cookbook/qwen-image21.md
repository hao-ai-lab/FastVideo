# Qwen-Image-2.1 recipes

<RecipeFamily family="qwen_image21" />

Generate images, edit an image, or combine up to ten ordered references with
`Qwen/Qwen-Image-2.1`. Outputs are RGBA PNGs. Request a transparent background
in the prompt when needed.

## Run on a GPU

Follow the [CUDA installation guide](../getting_started/installation/gpu.md), then
run from the repository root on one CUDA GPU (`tp_size=sp_size=1`). The examples
use CPU offload and VAE tiling. The loader downloads the complete checkpoint;
use `--model-path /path/to/Qwen-Image-2.1` for local weights. No conversion is needed.

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --prompt 'A red ceramic teapot, isolated on a transparent background.' \
  --output outputs/teapot.png
```

This saves a 1024×1024 PNG after 40 denoising steps. For instruction editing,
replace `source.png` with your image:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --reference source.png \
  --prompt 'Change the shirt to blue; retain the subject and composition.' \
  --output outputs/edit.png
```

Repeat `--reference` for reference-to-image generation, annotations or masks:

```bash
python examples/inference/basic/basic_qwen_image21.py \
  --reference subject.png --reference style.png \
  --prompt 'Paint the subject from image 1 in the style of image 2.'
```

A mask is another reference interpreted through the prompt; pixels outside it
are not guaranteed to remain identical.

## Configuration

- Set `--height`, `--width`, `--steps` and `--seed` as needed. Dimensions must be
  positive multiples of 32; use `--height 2048 --width 2048` for 2K output.
- `--reference-resolution` defaults to 1024 and must be a positive multiple of
  32. References retain their aspect ratio and are resized to approximately
  that resolution squared in pixels, independently of output dimensions.
- Prefix KV caching avoids recomputing text/reference tokens each step. The
  Python example caches on CPU; `--no-use-kv-cache` disables reuse.
- For two-branch CFG, set `--true-cfg-scale` above 1 and supply
  `--negative-prompt` (an empty string is valid). Embedded guidance is not used.

For the CLI, use the checked-in [T2I YAML](../../examples/inference/basic/qwen_image21_t2i.yaml)
or [editing YAML](../../examples/inference/basic/qwen_image21_edit.yaml) with
`fastvideo generate --config <file>`. The T2I YAML keeps prefix KV caches on CUDA;
the Python example and editing YAML default to CPU caches. See
[Configuration](../inference/configuration.md) for the Python API and YAML options.

## Attention

Dense image attention uses FastVideo's shared automatic backend selector, which
prefers FlashAttention on supported CUDA hardware. Masked text attention uses
SDPA. To request FA4 with its dependencies installed on supported hardware:

```bash
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export FASTVIDEO_FA4=1
```

Use `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` and `FASTVIDEO_FA4=0` for upstream
parity comparisons. Different attention kernels can produce different pixels.
See [Optimizations](../inference/optimizations.md) for backend and offload settings.

## Measured memory

Peaks below are in **GiB**, measured on one **NVIDIA GB200** with BF16,
1024×1024 output, seed 42, compilation disabled and two 40-step requests per
process. Image cases use one reference. These measurements do not establish
minimum hardware capacity; an actual 24 GB GPU has not been verified.

| Configuration | KV cache | GPU allocated | GPU reserved | Device VRAM including driver | Host process-tree RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| T2I YAML, SDPA, offloaded, tiled | CUDA | 13.28 | 22.54 | 24.19 | 43.87 |
| Plain I2I Python, SDPA, offloaded, tiled | CPU | 13.29 | 16.11 | 17.76 | 48.29 |
| Editing Python, SDPA, offloaded, tiled | CPU | 13.29 | 16.11 | 17.76 | 49.25 |
| T2I, FA4, resident models, untiled VAE | CUDA | 32.77 | 35.50 | 37.09 | 13.12 |

Measurements cover loading, cold/warm requests, PNG export and shutdown.
Allocated/reserved values use PyTorch counters; device VRAM and simultaneous
process-tree RSS were sampled every 20 ms. Device VRAM includes the GB200's
0.72 GiB driver reservation. Allow host RAM headroom beyond process RSS.
Offloaded rows explicitly use SDPA; the resident FA4 row disables offload and
tiling and is a separate configuration, not a controlled latency comparison.

## Validation and scope

On GB200, 40-step upstream SDPA parity passed for eleven 1024×1024 cases:
T2I, plain I2I, editing, reference-to-image, RGBA generation/editing, annotations,
mask images, true CFG, uncached execution and ten references. Separate 2048×2048
T2I also passed. Denoising trajectories and output pixels matched exactly.
FA4 block and T2I/edit image regressions passed against their matching references.
See the [validation runbook](../../tests/local_tests/qwen_image21/README.md)
for reproducible commands, pinned revisions and numerical diagnostics.

The supported scope is single GPU, batch one and unquantized inference.
SP/TP, LoRA, prompt rewriting and HTTP serving are not included.
