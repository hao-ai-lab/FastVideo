# Local Stable Audio Tests

End-to-end parity tests for the Stable Audio Open 1.0 pipeline + the
Oobleck VAE component. They compare FastVideo against the published
weights and the official `Stability-AI/stable-audio-tools` reference,
so they're skipped in CI and run locally on a single GPU.

## Setup

### 1. Hugging Face access

Stable Audio Open 1.0 is a gated repo. Accept the terms once at
https://huggingface.co/stabilityai/stable-audio-open-1.0 and export
your token in the shell:

```bash
export HF_TOKEN=hf_...
# any of HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY works
```

The tests skip cleanly with a helpful message if no token is set.

### 2. Optional inference dependencies

`k_diffusion` and friends are not part of FastVideo's base install.
One-shot:

```bash
uv pip install k_diffusion einops_exts alias_free_torch torchsde
```

### 3. Clone the upstream reference repo

The pipeline-level parity tests (`test_stable_audio_pipeline_parity.py`,
`test_stable_audio_a2a_parity.py`) and the VAE-component parity tests
(`test_oobleck_vae_parity.py`, `test_oobleck_vae_official_parity.py`)
import directly from `stable_audio_tools`. Clone it under the repo
root and `uv pip install` editable, then add it to `.gitignore`:

```bash
cd <FastVideo repo root>
git clone --depth 1 https://github.com/Stability-AI/stable-audio-tools.git
uv pip install --no-deps -e ./stable-audio-tools
echo "/stable-audio-tools/" >> .git/info/exclude   # personal ignore
```

The smoke test and the inpainting self-consistency test do not need the
upstream clone.

### 4. (Optional) Pre-warm the model cache

The first parity-test run downloads the full Stable Audio Open 1.0
checkpoint (~3 GB). To avoid the download blocking your first test
run, fetch it ahead of time:

```bash
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('stabilityai/stable-audio-open-1.0')"
```

## Running the tests

All Stable Audio tests in one shot (~80 s on a single B200):

```bash
pytest tests/local_tests/stable_audio/test_stable_audio_pipeline_smoke.py \
       tests/local_tests/stable_audio/test_stable_audio_pipeline_parity.py \
       tests/local_tests/stable_audio/test_stable_audio_a2a_parity.py \
       tests/local_tests/stable_audio/test_stable_audio_inpaint_parity.py \
       tests/local_tests/stable_audio/test_oobleck_vae_parity.py \
       tests/local_tests/stable_audio/test_oobleck_vae_official_parity.py \
       -v -s
```

Or, to run the entire family:

```bash
pytest tests/local_tests/stable_audio/ -v -s
```

Add `-s` to print the per-test diff numbers (shape / abs_mean / max
diff / drift / RMS ratios).

### What each test covers

| Test | Compares against | Notes |
|---|---|---|
| `test_stable_audio_pipeline_smoke.py` | — (no GPU) | Imports + registry + preset wiring. CPU-only. |
| `test_stable_audio_pipeline_parity.py` | `stable_audio_tools.inference.generate_diffusion_cond` | T2A end-to-end. Drift bound: < 1% / 0.05 element-wise. |
| `test_stable_audio_a2a_parity.py` | `generate_diffusion_cond(init_audio=...)` | Audio-to-audio variation. Drift bound: < 2% / 0.05 mean. |
| `test_stable_audio_inpaint_parity.py` | self-consistency | RePaint blending — kept-region RMS ratio in [0.5, 2.0], unkept region non-silent + bounded. SA Open 1.0 isn't inpaint-trained so there's no upstream reference to compare against. |
| `test_oobleck_vae_parity.py` | `stable_audio_tools.models.AudioAutoencoder` (real weights) | Decode + encode + round-trip; should be bit-identical (fp32, diff = 0). |
| `test_oobleck_vae_official_parity.py` | `OobleckEncoder` / `OobleckDecoder` (random init) | Architectural parity — verifies the FastVideo port matches upstream's structure. Bit-identical (diff = 0). |

### Reproducing a single test

Each test file is independent. Run one:

```bash
pytest tests/local_tests/stable_audio/test_stable_audio_a2a_parity.py -v -s
```

Expected output (approximate, on a B200, fp32, default seed = 0):

```
off shape=(1, 2, 66150) abs_mean=0.181810
fv  shape=(1, 2, 66150) abs_mean=0.180913
diff max=0.030 mean=0.003 median=0.002  abs_mean rel drift=0.49%
PASSED
```

## Troubleshooting

- **All tests skip with "gated repo" reason.** Check `echo $HF_TOKEN`
  and that you've accepted the model terms at
  https://huggingface.co/stabilityai/stable-audio-open-1.0.
- **`ModuleNotFoundError: stable_audio_tools` / `k_diffusion`.** Re-run
  the setup steps; the parity tests skip cleanly if these aren't
  importable but only after detecting both, so a partial install can
  produce an unhelpful error mid-test.
- **A2A or inpaint test passes individually but fails in a batch
  run.** The deterministic-math toggles (TF32 / cuDNN benchmark off)
  are set in `StableAudioPipeline.load_modules`. If a test earlier in
  the batch flipped them on, drift can spike — re-run the failing
  test alone to confirm.
- **Wall-clock blew up past 5 min.** First call downloads the
  checkpoint and the T5-base weights (~4 GB total) — see the
  pre-warm step above.

## Adding new parity tests for this family

Use any of the existing `test_stable_audio_*` files as a template — the
HF-token detection (`_hf_token` / `_can_access` / `_setup_hf_env`) and
the loader for the official model (`_load_official`) are duplicated
across the parity-test files; copy them as a starting point and call
out the comparison reference (upstream class + invocation) in the file
docstring.
