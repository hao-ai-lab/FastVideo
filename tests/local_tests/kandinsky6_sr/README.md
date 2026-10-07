# Kandinsky-6 Video Super-Resolution (SR) Local Tests

Local-only parity checks for the Kandinsky6 SR port. CI runs the CPU unit tests in `fastvideo/tests/stages/kandinsky6_sr/`
(tiny random models, both bundle kinds through the full pipeline); the tests here need a GPU, the released weights, a
checkout of the previous port or the official reference, and skip without them.

| Field | Value |
|---|---|
| Official reference | `k6_video/src/kandinsky_sr/` (`KANDINSKY_SR_SRC`, default `../k6_video/src`), HEAD `473ea253603c98e23130da8f67c7f00504f33d4e` |
| HF weights | `kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers`, `kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` |
| Previous port | PR head `12be8fdd` (own sampler loops, vendored upscaler / KVAE packages) |

## Tests

| Test | Compares | Needs |
|---|---|---|
| `../pipelines/test_kandinsky6_sr_pipeline_parity.py` | full pipeline vs the previous port, both bundles, x2 / x2.25 / x4 (PSNR >= 48 dB) | CUDA, `KANDINSKY6_SR_MODELS`, `KANDINSKY6_SR_VIDEO`, `KANDINSKY6_SR_PREVIOUS_ROOT` |
| `test_kandinsky6_sr_latent_upscaler_parity.py` | latent upscaler vs the previous port, bitwise, x2 / x4, fp32 / bf16 | CUDA, `KANDINSKY6_SR_BUNDLE`, `KANDINSKY6_SR_PREVIOUS_ROOT` |
| `test_kandinsky6_sr_vae_parity.py` | KVAE encode / decode vs the previous port, bitwise | CUDA, `KANDINSKY6_SR_VAE_DIR`, `KANDINSKY6_SR_VAE_REFERENCE` |
| `test_kandinsky6_sr_dit_parity.py` | SR DiT vs the reference DiT, tiny random weights | `KANDINSKY_SR_SRC` |
| `test_kandinsky6_sr_real_weights.py` | real-weights smoke; optional PSNR vs a `kandy-sr` reference output | CUDA, `KANDINSKY6_SR_BUNDLE`, `KANDINSKY6_SR_TEST_VIDEO` |

The previous-port comparisons run each implementation in its own subprocess, because both are the package `fastvideo`.

## Commands

```bash
# CPU unit tests (the CI set)
pytest fastvideo/tests/stages/kandinsky6_sr -q

# pipeline parity on a GPU
KANDINSKY6_SR_MODELS=/path/to/dir/with/both/bundles KANDINSKY6_SR_VIDEO=/path/to/clip_17f.mp4 \
KANDINSKY6_SR_PREVIOUS_ROOT=/path/to/previous/checkout \
pytest tests/local_tests/pipelines/test_kandinsky6_sr_pipeline_parity.py -v -s

# component parity on a GPU
KANDINSKY6_SR_BUNDLE=/path/to/Kandinsky-6.0-VSR-5s-Diffusers KANDINSKY6_SR_PREVIOUS_ROOT=/path/to/previous/checkout \
pytest tests/local_tests/kandinsky6_sr/test_kandinsky6_sr_latent_upscaler_parity.py -v -s
KANDINSKY6_SR_VAE_DIR=/path/to/Kandinsky-6.0-VSR-5s-Diffusers/vae KANDINSKY6_SR_VAE_REFERENCE=/path/to/previous/checkout \
pytest tests/local_tests/kandinsky6_sr/test_kandinsky6_sr_vae_parity.py -v -s
```
