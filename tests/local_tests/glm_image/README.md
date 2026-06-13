# GLM-Image Local Tests

Local-only parity and smoke tests for the `glm_image` port: FastVideo vs the
official HF `transformers` + `diffusers` GLM-Image implementation. These are
**not run in CI** — they need the full weights, CUDA, and (for the reference
side) the diffusers GLM-Image classes; each test skips with an actionable
message when a prerequisite is missing.

## What you need

| | |
|---|---|
| Weights | `zai-org/GLM-Image` → `official_weights/glm_image` (~30 GB: AR encoder 4 shards, transformer 3 shards, vae + text_encoder) |
| Runtime deps | `transformers>=5.0.0` (first release with the AR encoder `GlmImageForConditionalGeneration`) and `diffusers>=0.38.0` — both committed in `pyproject.toml` |
| Test-only deps | the diffusers `GlmImageTransformer2DModel` / `GlmImagePipeline` reference classes need `diffusers>=0.37.0.dev0` (used **only** by the parity tests) |

Download weights (from the repo root; cache via `HF_HOME` if `/` is small):

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "zai-org/GLM-Image" "official_weights/glm_image"
```

Optionally install the diffusers reference classes to run the parity side:

```bash
uv pip install -U "git+https://github.com/huggingface/diffusers.git@ff3b86b4"
```

Parity was checked against transformers `a65bf6c9` + diffusers `ff3b86b4`.

## Parity results (GB200)

| Component | Test | Result |
|---|---|---|
| `transformer` (DiT) | `test_transformer_parity.py` | fp32 cosine `1.000000` (100% within 5e-3); bf16 cosine `0.999969` (element-wise rounding over 30 layers vs a different SDPA kernel — not a bug; RoPE proven exactly equivalent). fp32 + bf16 + strict-load |
| `vae` (`AutoencoderKL`) | `test_vae_parity.py` | decode + encode + config, 3/3 |
| `text_encoder` (T5) + **ByT5** tokenizer | `test_t5_parity.py` | tokenizer + glyph extract + embeds, 3/3 |
| `vision_language_encoder` (GLM-4-9B AR) | `test_ar_parity.py` | surface + to/eval + greedy determinism, 3/3 |
| `pipeline` (T2I) | `test_pipeline_parity.py` | injected-input determinism: latent cosine `0.999997`, decoded-image MAE `0.159/255`. validity + deterministic parity, 2/2 |
| `pipeline` (I2I / edit) | `test_edit_pipeline_parity.py` | validity / wiring gate (condition image enters via a KV-cache write pass, not noise-the-latent). diffusers distribution parity deferred (stochastic AR) |

`processor` (`GlmImageProcessor`) and `scheduler` (`FlowMatchEulerDiscreteScheduler`,
shared flow-matching) are exercised via the AR and pipeline tests.

Run:

```bash
pytest tests/local_tests/glm_image/test_transformer_parity.py -v -s
pytest tests/local_tests/glm_image/test_vae_parity.py -v -s
pytest tests/local_tests/glm_image/test_t5_parity.py -v -s
pytest tests/local_tests/glm_image/test_ar_parity.py -v -s
pytest tests/local_tests/glm_image/test_pipeline_parity.py -v -s
pytest tests/local_tests/glm_image/test_edit_pipeline_parity.py -v -s
```

## Design notes (load-bearing — don't regress)

- **Pipeline parity is by injection.** Full-pipeline pixel parity is ill-posed
  (stochastic AR + independent latent RNG), so the pipeline tests inject matched
  `(prompt_embeds, prior_token_ids, latents)` into both the diffusers pipeline
  and the real `GlmImageDenoisingStage`, then compare denoised latents + decoded
  image — not raw end-to-end pixels.
- **HF imports are confined** to the lazy-wrapper loader
  `fastvideo/models/encoders/glm_image_ar_loader.py`, never the production
  loaders (production-boundary rule).
- **Tokenizer is ByT5**, matching the diffusers reference (not plain T5);
  `compute_glyph_embeds` mirrors diffusers `_get_glyph_embeds`.
- **Model-specific stages** live under
  `fastvideo/pipelines/basic/glm_image/stages/` per the `add-model` Files Map.
- **Param mapping** is handled in-place at load via `param_names_mapping`
  (`fastvideo/configs/models/dits/glm_image.py`) + the native VAE/encoder
  configs — no conversion script; the loader raises on any unmatched param and
  `test_transformer_strict_load.py` asserts completeness with `strict=True`.
