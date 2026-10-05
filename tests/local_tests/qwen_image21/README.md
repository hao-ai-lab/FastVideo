# Qwen-Image-2.1 validation

Released-weight CUDA validation completed on an RTX 5090 using the local
fork installed editable in the existing environment. The production encoder,
DiT and VAE component gates pass without skips. Plain I2I and editing both
match upstream exactly at 1024 x 1024 and 40 steps and have been visually
confirmed. T2I and two-reference generation match exactly at four steps.
The requested non-layerwise-offload benchmark also passes, with a 42.44 s
generation request and 16.85 GiB sampled peak GPU usage for 40-step plain I2I.
With preloaded models, no DiT offload and CUDA prefix KV storage, two warm
40-step T2I requests averaged 15.38 s for encoder + DiT + decoder and 15.64 s
including PNG saving. Their four-step warmup exactly matched the validated
upstream baseline; the forty-step outputs matched each other.
The completed runs are recorded in [GPU_VALIDATION.md](GPU_VALIDATION.md)
for confirmed runs, commands, numerical metrics and persistent log links, and
[PORT_STATUS.md](PORT_STATUS.md) for remaining gates.

Text evidence is archived separately under `outputs/qwen_image21/`. Images,
tensor dumps, weights and the upstream checkout are not included in this
pod-deletion archive. Reproduction requires downloading the pinned assets,
updating historical absolute paths and regenerating baseline tensors.

## CPU checks on the Mac

Run from the repository root, in a Python environment with torch, pytest,
NumPy, Pillow, SciPy and PyYAML:

```bash
python -m pytest \
  tests/local_tests/qwen_image21 \
  tests/local_tests/transformers/test_qwen_image21_transformer_contracts.py \
  tests/local_tests/transformers/test_qwen_image21_transformer_parity.py \
  tests/local_tests/encoders/test_qwen_image21_conditioner.py \
  tests/local_tests/encoders/test_qwen_image21_conditioner_parity.py \
  tests/local_tests/vaes/test_qwen_image21_vae_cpu.py \
  tests/local_tests/vaes/test_qwen_image21_vae_parity.py -q
```

The component tests isolate imports from the CUDA runtime. DiT and encoder
checks substitute ordinary Torch linear/embedding dispatch; their numerical
graphs remain native. Pipeline checks substitute small deterministic components.
The CLI checks run the actual typed parser with startup imports isolated.
These are useful implementation checks, not production-loader parity.

In the original Mac environment, reference, tokenizer and GPU tests skipped.
A skip is not a passing parity result. An explicitly enabled GPU gate fails
when CUDA, dependencies or assets are missing.

The CPU cases are included in `.buildkite/scripts/unit_test.sh`.

## Prepare an allocated CUDA machine

The initial target is one 24 GB CUDA GPU and at least 96 GB host RAM. Use
`tp_size=sp_size=1`, BF16 components, CPU text and layerwise DiT offload, VAE
tiling, deferred module loading and CPU prefix caches. The 24 GB bound is a
target until measured by the GPU tests.

Install FastVideo following the repository's CUDA installation guide. The
dependency update requires Transformers 5.17+, tokenizers 0.23.1–0.23.x and
Accelerate 1.1+. Run the CLIP tokenizer regression after installing the resolved
environment; it requires no network or checkpoint download.

Prepare the reference checkout and a complete local checkpoint on that machine:

```bash
git clone https://github.com/huggingface/diffusers.git /path/to/diffusers-reference
git -C /path/to/diffusers-reference checkout 578c9b2c6636ab2424a0e56186268b83623656b2
export QWEN_IMAGE21_DIFFUSERS_DIR=/path/to/diffusers-reference
export QWEN_IMAGE21_MODEL_ROOT=/path/to/Qwen-Image-2.1
export PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
```

The local model must contain `model_index.json`, `processor`, `text_encoder`,
`transformer`, `vae` and `scheduler`, including all weight shards. Acquire
the checkpoint under Qwen's license on the GPU machine. There is no conversion
script because native keys preserve the released checkpoint layout; the encoder
excludes only the unused language-model head.

## Component parity gates

Run these individually to keep failure diagnosis clear:

```bash
QWEN_IMAGE21_RUN_DIT_PARITY=1 python -m pytest \
  tests/local_tests/transformers/test_qwen_image21_transformer_parity.py -q

QWEN_IMAGE21_RUN_VAE_PARITY=1 python -m pytest \
  tests/local_tests/vaes/test_qwen_image21_vae_parity.py -q

QWEN_IMAGE21_RUN_ENCODER_PARITY=1 python -m pytest \
  tests/local_tests/encoders/test_qwen_image21_conditioner_parity.py -q
```

These use the production FastVideo component loaders and the official reference.
The DiT test checks every block, the final prediction and cached decode. The VAE
test checks posterior statistics and RGBA decode; random-weight reference tests
cover tiling. The encoder test checks text, long text, 1/2/10 images and padding,
with the reference final norm neutralized to expose the required pre-norm output.
The existing component tolerances have passed on the recorded RTX 5090 runs;
no tolerance was widened. Set `QWEN_IMAGE21_ENCODER_PARITY_REPORT` and
`QWEN_IMAGE21_DIT_PARITY_REPORT` to save numerical JSON diagnostics.

## Pipeline and output gates

```bash
QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 python -m pytest \
  tests/local_tests/qwen_image21/test_pipeline_parity.py -q

QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 QWEN_IMAGE21_RUN_2K_PARITY=1 \
  python -m pytest tests/local_tests/qwen_image21/test_pipeline_parity.py \
  -k t2i_2k -q
```

Each test runs native and reference pipelines in separate processes using
identical explicit BF16 initial latents, sigmas and cache settings. Synthetic
references are generated by default; `QWEN_IMAGE21_PARITY_CASES_JSON` can
override prompts and ordered paths with original-source images. Native
execution uses CPU KV caches; the reference uses its canonical cache behavior
with the same `use_kv_cache` value. Tests compare four-channel pixel
tensors, verify the exported PNG retains alpha, and check the native reported
memory peak when available. The 1024 cases cover T2I, plain I2I, edit, Ref2Img, ten refs,
RGBA generation/edit, annotations, separate mask images, true CFG and no cache.
The separate 2K case covers T2I only. `QWEN_IMAGE21_PARITY_DIAGNOSTICS=1`
additionally saves observational conditioning, scheduler and denoising tensors
to local run artifacts; it does not change model arithmetic or assertions.

Pipeline tests default to four steps for debugging. After these pass, repeat
with `QWEN_IMAGE21_PARITY_STEPS=40` and inspect generated images for text
rendering, subject identity, edit instructions and alpha edges. Capture peak
GPU memory and host RSS on the actual 24 GB machine. Do not claim mask exterior
pixel preservation: masks are semantic references in the released workflow.

For a CLI smoke run:

```bash
fastvideo generate --config examples/inference/basic/qwen_image21_t2i.yaml \
  --generator.model_path "$QWEN_IMAGE21_MODEL_ROOT"
```

Before release, run the changed-file pre-commit checks in the installed
development environment and calibrate parity tolerances from actual GPU results.
Add approved image/latent quality references only after inspecting real outputs.
