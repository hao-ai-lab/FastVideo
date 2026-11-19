# Pyramid-JiT image inference

FastVideo loads Linum's original `config.json` and `model.safetensors` together with the separate
`Qwen/Qwen3.5-4B` tokenizer and text encoder. Supply a detailed caption directly. The pipeline uses the
published 50-step Euler sampler with adaptive projected guidance and exports a single RGB PNG.

```bash
fastvideo generate --config examples/inference/basic/pjit_21k_t2i.yaml
```

The [Python example](../../../examples/inference/basic/basic_pjit_21k.py) accepts local paths through
`--model-path` and `--qwen-model-path`. Use one device with Torch SDPA for reference comparisons.
Lazy loading releases Qwen after text encoding and loads the FP32 image transformer for denoising.
On Apple silicon, initial noise comes from a seeded CPU generator. On NVIDIA GPUs, the default noise
generator follows Linum's H100 SXM convention. A given seed therefore produces different images across these paths.
The seed-2 portrait was verified on one Apple M5 Pro with Torch SDPA and 48 GB unified memory.
CUDA image parity, FlashAttention, and minimum device capacity have not been measured.

## Compare with Linum

Use Linum's code at `95403933f3727abe947977c75915a53292e1d841`, with the PJiT checkpoint at
`251725ad9a0618f0a4ba3cdf4a488d35312be352` and Qwen at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Set `PJIT_REFERENCE_ROOT` to the reference checkout, `PJIT_MODEL_DIR` to the PJiT weights, and
`PJIT_QWEN_MODEL_DIR` to the Qwen weights. Run tests from the FastVideo repository root.

```bash
pytest -q tests/local_tests/transformers/test_pjit_dit_parity.py
PJIT_ENCODER_DEVICE=mps PJIT_REAL_ENCODER_PARITY=1 \
  pytest -s tests/local_tests/encoders/test_pjit_21k_qwen_parity.py
PJIT_RUN_MPS_PARITY=1 pytest -s tests/local_tests/pjit/test_pipeline_parity.py
PJIT_RUN_CUDA_PARITY=1 pytest -s tests/local_tests/pjit/test_pipeline_parity.py
```

The MPS comparison uses the same CPU noise on both sides, runs Linum's FP64 operations on CPU,
and applies BF16 autocast on Metal. The CUDA comparison runs Linum's unchanged generator and the FastVideo example sequentially at 864×1216
with seed 2. It reports the PNGs' MS-SSIM, absolute pixel differences and exact-match fraction.
The CI test in `fastvideo/tests/ssim/test_pjit_similarity.py` separately compares with a reviewed PNG
for the selected GPU, attention backend and quality tier.
