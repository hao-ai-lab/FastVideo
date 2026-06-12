use

```
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="$CONDA_ROOT/bin/conda"
ENV_NAME="${ENV_NAME:-FastVideo_kaiqin}"

if [ -x "$CONDA_BIN" ]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "conda activated"
else
    echo "conda not found at $CONDA_BIN" >&2
    exit 1
fi
```

to get python environment, and then export PYTHONPATH
since I dont want to re-install whole environment and build torch / flashattention.

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"   # run from the repo root (FastVideo-PR3/)
```

---

## Branch: `feat-contribution`

5 commits ahead of `main`:

```
8a9e5a1b Add Ovis-Image-7B text-to-image pipeline
f6ffce3e Add Ovis-Image-7B tests
156d68e8 Update fastvideo/tests/ssim/test_ovis_image_similarity.py
94fa3ce6 Update examples/inference/basic/basic_ovis_image.py
10e19b9c Update __init__.py
```

### What it introduces

Native **Ovis-Image-7B** text-to-image (T2I) support in FastVideo. Ovis-Image
is a FLUX-style DiT optimized for **text rendering in images** (posters,
banners, logos, UI mockups, bilingual text). The text encoder is **Qwen3**
(from Ovis2.5-2B) rather than the usual CLIP/T5.

New files (~2.6k LOC added, per `git diff --stat main...HEAD`):

| Path | Purpose |
| --- | --- |
| `fastvideo/models/dits/ovisimage.py` | FastVideo-native Ovis-Image Transformer2D DiT (FLUX-style 3D RoPE, AdaLN-Zero, GEGLU FFN) |
| `fastvideo/configs/models/dits/ovisimage.py` | `OvisImageTransformer2DModelConfig` |
| `fastvideo/models/encoders/qwen3.py` | Qwen3 text encoder |
| `fastvideo/configs/models/encoders/qwen3.py` | `Qwen3Config` |
| `fastvideo/configs/models/vaes/base.py` | VAE config additions |
| `fastvideo/pipelines/basic/ovis_image/ovis_image_pipeline.py` | the T2I pipeline |
| `fastvideo/configs/pipelines/ovis_image.py` | `OvisImageT2IConfig` + Qwen3 pre/post-process (system prompt, 28-token slice) |
| `fastvideo/configs/ovis_image_7b_t2i_pipeline.json` | JSON pipeline config (bf16 DiT, fp32 VAE decoder-only, tiling) |
| `examples/inference/basic/basic_ovis_image.py` | runnable T2I example |
| `fastvideo/tests/transformers/test_ovisimage.py` | DiT unit test |
| `fastvideo/tests/encoders/test_qwen3_encoder.py` | encoder unit test |
| `fastvideo/tests/ssim/test_ovis_image_similarity.py` | SSIM regression (256×256, seed 42, MS-SSIM ≥ 0.98) |
| `tests/.../test_ovis_image_pipeline_smoke.py` | smoke test |

Also touches the registries (`fastvideo/models/registry.py`,
`fastvideo/pipelines/pipeline_registry.py`, `fastvideo/registry.py`,
`fastvideo/training/__init__.py`) and refactors `denoising.py` to register the
new model. A T2I training pipeline (`ovis_image_training_pipeline.py`) is added
too.

### Model download (done)

The branch loads the model by HF id **`AIDC-AI/Ovis-Image-7B`** (see
`examples/inference/basic/basic_ovis_image.py:23`,
`fastvideo/tests/ssim/test_ovis_image_similarity.py:33`). `from_pretrained`
takes the single repo id, so it is a self-contained diffusers-layout repo
(DiT transformer + VAE + Qwen3 text_encoder + tokenizer + `model_index.json`).

Downloaded into a subfolder of this repo:

```bash
# from repo root, env activated (see top of this doc)
mkdir -p models
huggingface-cli download AIDC-AI/Ovis-Image-7B \
    --local-dir models/Ovis-Image-7B
```

Weights now live at **`models/Ovis-Image-7B/`** (53 GB, all 44 files, verified
complete — no `.incomplete` leftovers).

**Layout** — the HF repo ships two redundant forms; only the diffusers layout is
needed by FastVideo:

| Component | Size | Needed by FastVideo? |
| --- | --- | --- |
| `model_index.json` | tiny | ✅ `OvisImagePipeline` entrypoint |
| `transformer/` (2 shards) | 14 GB | ✅ `OvisImageTransformer2DModel` DiT |
| `vae/` | 161 MB | ✅ `AutoencoderKL` decoder |
| `text_encoder/` (2 shards) | 6.5 GB | ✅ `Qwen3Model` encoder |
| `tokenizer/`, `scheduler/` | 17 MB | ✅ |
| `ovis_image.safetensors` | 28 GB | ❌ original single-file DiT blob |
| `ae.safetensors` | 320 MB | ❌ original single-file VAE |
| `Ovis2.5-2B/` | 4.9 GB | ❌ original combined encoder bundle |

The diffusers layout (transformer + vae + text_encoder + tokenizer +
model_index) is ~21 GB. The `ovis_image.safetensors` / `ae.safetensors` /
`Ovis2.5-2B/` files are the upstream single-file release and are **not** loaded
by FastVideo's `from_pretrained`. To reclaim ~33 GB:

```bash
cd models/Ovis-Image-7B && rm -rf ovis_image.safetensors ae.safetensors Ovis2.5-2B
```

### Run

Inference example (edit the model path to the local dir, or pass the HF id):

```bash
# point the example / generator at the local download:
python examples/inference/basic/basic_ovis_image.py
# (the script uses "AIDC-AI/Ovis-Image-7B"; swap to "models/Ovis-Image-7B"
#  to load the local copy and avoid re-downloading)
```

SSIM regression test (uses the local weights via env var):

```bash
OVIS_WEIGHTS=models/Ovis-Image-7B \
    pytest fastvideo/tests/ssim/test_ovis_image_similarity.py -vs
# First run prints the exact `cp` command to "bless" the reference image.
```

Unit tests (no full weights needed):

```bash
pytest fastvideo/tests/transformers/test_ovisimage.py \
       fastvideo/tests/encoders/test_qwen3_encoder.py -v
```
