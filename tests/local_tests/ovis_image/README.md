# Ovis-Image Local Tests

Local-only component parity tests for the `ovis_image` FastVideo port
(`AIDC-AI/Ovis-Image-7B`, a FLUX-style MM-DiT text-to-image model with a Qwen3
text encoder). These are skipped in CI and meant to run locally on a single GPU.
They skip cleanly when weights or the official reference are absent; a skip is
**not** a verified pass (see `../../../.agents/skills/add-model/shared/common_rules.md`).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `ovis_image` |
| Workload types | `T2I` |
| Official reference | Diffusers `OvisImageTransformer2DModel` / `AutoencoderKL`; HF `transformers.Qwen3Model` |
| HF weights | `AIDC-AI/Ovis-Image-7B` |
| Local weights dir | env `OVIS_WEIGHTS` (default `official_weights/ovis_image`); must contain `transformer/`, `text_encoder/`, `tokenizer/`, `vae/` |
| Source layout | Diffusers (`model_index.json` + per-component dirs) |
| Needs conversion | `no` — loaders consume the Diffusers layout directly |

> Use only the env-var **name** for tokens (e.g. `HF_TOKEN`). Never paste a token value.

## Environment Setup

Run from the repo root in the **`FastVideo_kaiqin`** conda env (the env this port
was developed/verified in — do not use other envs):

```bash
conda activate FastVideo_kaiqin
export PYTHONPATH="$PWD:$PYTHONPATH"
# Point at local weights (fastest) or let it resolve from the Hub.
export OVIS_WEIGHTS=models/Ovis-Image-7B   # or official_weights/ovis_image
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval. The official
references (`diffusers`, `transformers`) are imported only inside the tests, never
in production code.

## Official Environment Status

```text
dependency_changes: none
official_env_status: imports_ok (transformers.Qwen3Model confirmed importable);
                     diffusers Ovis support is release-dependent — the transformer
                     and VAE parity tests skip if the diffusers symbols are absent.
private_dep_stubs: none
blocked_on: none
```

## Tests in this directory

```bash
pytest tests/local_tests/ovis_image/ -v -s
```

| Component | Test | Scope | Concerns | Status |
|---|---|---|---|---|
| `transformer/DiT` | [`test_ovis_transformer_parity.py`](./test_ovis_transformer_parity.py) | both | cross-impl DiT parity vs Diffusers `OvisImageTransformer2DModel` | **PASS** (max_diff ≈6e-2, drift ≈1%) |
| `text encoder` (Qwen3) | [`test_ovis_qwen3_parity.py`](./test_ovis_qwen3_parity.py) | both | Ovis chat-template + system-prompt path vs HF `Qwen3Model`; 28-token slice | **PASS** (max_diff < 1e-3) |
| `vae` (reused AutoencoderKL) | [`test_ovis_vae_parity.py`](./test_ovis_vae_parity.py) | both | decoder-only `decode` parity vs Diffusers `AutoencoderKL` | **PASS** |

All three **PASS** (non-skip): `pytest tests/local_tests/ovis_image/ -q` → `3 passed`.

Verified in env `FastVideo_kaiqin` (torch 2.11+cu128, diffusers 0.37.1 — which ships
`OvisImageTransformer2DModel` — transformers 4.57) against local weights at
`models/Ovis-Image-7B`, `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.

Two production bugs were found and fixed while activating these tests (see
PORT_STATUS R003/R005): `ovisimage.py` imported the removed `CachableDiT`
(now `BaseDiT`), and its joint-attention call used the pre-merge positional
signature of `DistributedAttention.forward`.

Related (not in this directory):

- `tests/local_tests/ovis_image/test_ovis_image_pipeline_smoke.py` — end-to-end
  VideoGenerator smoke (loadability only).
- `tests/local_tests/ovis_image/test_ovis_image_pipeline_parity.py` — end-to-end
  **pipeline parity vs official Diffusers `OvisImagePipeline`** (3 tests, PASS):
  bit-exact timestep schedule, RNG-aligned denoised-latent parity (drift 3.3%),
  and a full multiprocess image-health check.
- `fastvideo/tests/ssim/test_ovis_image_similarity.py` — CI-side output-quality
  regression (the only Ovis test kept under `fastvideo/tests/`; the DiT/Qwen3
  component parity now lives entirely in this directory).
- `fastvideo/tests/ssim/test_ovis_image_similarity.py` — output quality
  regression (GPU-heavy); PASS (MS-SSIM ≥ 0.98) against a GB200 reference.

## Activation Notes (per add-model Phase 9)

On the porter's machine, each scaffold above must be turned into a non-skip
PASS before handoff:

1. Component parity: transformer, Qwen3 encoder, VAE.
2. Pipeline smoke (`tests/local_tests/ovis_image/test_ovis_image_pipeline_smoke.py`).
3. Pipeline parity vs the official Diffusers `OvisImagePipeline` (denoised
   latents / decoded image) — **not yet authored**; tracked as `I001` in
   `PORT_STATUS.md`.
4. Basic example.

If transformer parity is numerically red, first confirm the two open
conventions on the FastVideo side: the timestep scale (FastVideo expects
`[0, 1000]`; Diffusers' input scale must match) and the `img_ids`/`txt_ids`
construction (this test reuses the production helpers from
`fastvideo.models.dits.ovisimage`, so any drift is on the reference side).

## Review Notes

- Component parity: non-skip PASS for DiT, Qwen3, and the reused AutoencoderKL VAE.
- Pipeline parity vs official `OvisImagePipeline`: **PASS** (3/3) — see
  `tests/local_tests/ovis_image/test_ovis_image_pipeline_parity.py`.
- Activating these tests surfaced and fixed 4 real bugs that blocked end-to-end
  inference (see `PORT_STATUS.md` I008–I011): the dynamic-shift timestep schedule
  (Ovis-specific), plus three shared-stage bugs in text encoding, VAE denorm, and
  2D VAE decode.
