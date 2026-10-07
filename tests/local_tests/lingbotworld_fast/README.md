# LingBot-World-Fast local verification

Local-only numerical parity and port evidence for
`FastVideo/LingBot-World-Fast-Diffusers`.

## Scope

- Official implementation: `Robbyant/lingbot-world`, `wan/modules/model_fast.py`
- Released fast checkpoint: `robbyant/lingbot-world-fast`
- FastVideo bundle: `FastVideo/LingBot-World-Fast-Diffusers`
- Workload: causal image-to-video with camera poses
- Released sampling contract:
  - global self-attention (`local_attn_size=-1`)
  - three latent frames per chunk
  - four fixed scheduler indices: `(0, 179, 358, 679)`
  - flow shift `10.0`

## Setup

Use the current FastVideo environment. The official source needs `easydict` in
addition to FastVideo's dependencies.

```bash
git clone https://github.com/Robbyant/lingbot-world.git \
  /tmp/lingbot-world-reference
uv pip install --python .venv/bin/python easydict
```

Resolve a local snapshot of the FastVideo bundle:

```bash
LINGBOTWORLD_FAST_MODEL_DIR=$(
  .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("FastVideo/LingBot-World-Fast-Diffusers"))
PY
)
export LINGBOTWORLD_FAST_MODEL_DIR
export LINGBOTWORLD_FAST_REFERENCE_DIR=/tmp/lingbot-world-reference
```

Do not store Hugging Face tokens in this directory.

## Full transformer parity

The parity test loads the same released transformer weights into both:

1. the official `WanModelFast` implementation; and
2. FastVideo's production `TransformerLoader`.

It runs the models sequentially to stay within memory limits. Both sides use
the mathematical SDPA backend. The official cross-attention helper is routed
through the official source's own SDPA-capable dispatcher because the direct
helper only supports FlashAttention.

The deterministic input is one real three-frame causal chunk with:

- 16 noisy latent channels;
- 20 image-conditioning channels;
- 4096-wide text context;
- packed 384-channel camera rays;
- real per-layer self- and cross-attention caches.

Run:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
LINGBOTWORLD_FAST_MODEL_DIR="$LINGBOTWORLD_FAST_MODEL_DIR" \
LINGBOTWORLD_FAST_REFERENCE_DIR="$LINGBOTWORLD_FAST_REFERENCE_DIR" \
.venv/bin/python -m pytest \
  tests/local_tests/lingbotworld_fast/test_lingbotworld_fast_transformer_parity.py \
  -v -s
```

This is valid evidence only when the full-transformer test reports `PASSED`,
not `SKIPPED`.

## Lightweight runtime-contract tests

These tests cover backend dispatch, the fixed-step preset contract, and exact
registry/pipeline resolution without loading the 18.5B transformer:

```bash
.venv/bin/python -m pytest \
  fastvideo/tests/attention/test_lingbotworld_fast_backend.py \
  fastvideo/tests/api/test_presets.py::TestLingBotWorldFastPresets \
  -v
```

## SSIM

The package test is:

```bash
pytest fastvideo/tests/ssim/test_lingbot_fast_similarity.py -v -s
```

It requires two GPUs and device-specific references. New L40S references must
be generated with the repository's `seed-ssim-references` workflow, reviewed
visually, and uploaded to `FastVideo/ssim-reference-videos`. A GB10 output is
not a valid replacement for the L40S CI reference.

See `PORT_STATUS.md` for the latest measured results and remaining blockers.
