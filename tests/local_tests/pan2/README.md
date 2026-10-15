# PAN2 Local Tests

Runbook for validating the PAN2 port (text-to-video and first-frame image-to-video). These tests run manually and are
not part of the default CI suite.

## Reference

PAN2's diffusers integration (`PAN2Transformer3DModel`, `PAN2ModularPipeline`) is not public yet, and the full
`IFM/PAN2` weights are not released. The reference is the vLLM-Omni port,
[vllm-project/vllm-omni#8602](https://github.com/vllm-project/vllm-omni/pull/8602) (head `59dd8c9a`), which reports
parity with the diffusers implementation. `reference.py` rebuilds it from diffusers and transformers building blocks:
the DiT from `AdaLayerNormZero`, `AdaLayerNormContinuous`, `FeedForward` (gelu-approximate and swiglu), `Timesteps`,
`TimestepEmbedding` and the diffusers RoPE helpers, using the checkpoint's module names so the checkpoint loads
strictly. The pipeline is `reference_generate`, a step-by-step copy of the vLLM-Omni `PAN2Pipeline.forward`
using `Qwen3_5TextModel`, `AutoencoderKLWan` and `FlowMatchEulerDiscreteScheduler`.

The checkpoint is the public random-weight [`wuqing157/tiny-pan2-modular-pipe`](https://huggingface.co/wuqing157/tiny-pan2-modular-pipe)
(revision `3ff3130b`), the same one vLLM-Omni tests against. It has two DiT blocks, a four-layer Qwen3.5 text model
(three Gated DeltaNet layers and one full-attention layer) and a small Wan2.2-layout VAE.

## Setup

```bash
huggingface-cli download wuqing157/tiny-pan2-modular-pipe --revision 3ff3130b4527f61aaab7bde01465c9884dc84cdd
# or point the tests at a local copy:
export PAN2_TINY_MODEL_DIR=/path/to/tiny-pan2-modular-pipe
```

The tests need `diffusers` with `AutoencoderKLWan` and `transformers` with `Qwen3_5TextModel` (5.x). They run on CPU
when no GPU is present. On a GPU-less host, `import fastvideo` also needs the `fastvideo_kernel` stub described in
`fastvideo/tests/AGENTS.md`.

## Tests

| Area | Command | Result (CPU, fp32 unless noted) |
|---|---|---|
| DiT: tiny checkpoint, strict load + forward | `pytest tests/local_tests/transformers/test_pan2_transformer_parity.py -k tiny_checkpoint` | max abs 6.0e-7 |
| DiT: random weights, batch 1 and 2 | `pytest tests/local_tests/transformers/test_pan2_transformer_parity.py -k random_weights` | max abs 0 and 1.9e-5 (outputs ~16) |
| DiT: Ulysses SP=2 vs one rank, padded tokens | `pytest tests/local_tests/transformers/test_pan2_transformer_parity.py -k sp2` | max abs 5.7e-6 (gloo on CPU; NCCL on 2 GPUs) |
| Qwen3.5 text encoder | `pytest tests/local_tests/encoders/test_pan2_qwen3_5_parity.py` | all hidden states exact (fp32, bf16, padded batch) |
| Wan2.2 VAE encode/decode + normalization | `pytest tests/local_tests/vaes/test_pan2_vae_parity.py` | exact; normalization 2.4e-7 |
| Flow-match scheduler, PAN2 sigmas + shift | `pytest tests/local_tests/schedulers/test_pan2_scheduler_parity.py` | exact |
| Pipeline, T2V and I2V (bf16 DiT) | `pytest tests/local_tests/pipelines/test_pan2_pipeline_parity.py` | final latents and video exact |

The pipeline test builds the FastVideo pipeline through the registry and component loaders and compares the final
latents (relative L2 < 2e-2, cosine > 0.9995) and the decoded video (PSNR > 40 dB) with `reference_generate`. To
confirm the threshold catches real deviations, three injected changes were each run against it: conditioning on
`hidden_states[-2]`, letting the latents fall to bf16 between steps, and shifting the first-frame crop by one pixel.
Each moved the latents by about 8e-2 relative L2.

## Not covered

- Real-weight parity and an SSIM regression: `IFM/PAN2` is not public.
- GPU runs: everything above ran on CPU. The bf16 GPU path, FlashAttention, and SP over NCCL still need a GPU node.
