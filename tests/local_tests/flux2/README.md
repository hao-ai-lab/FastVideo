# Flux2 Local Tests

Local-only component parity tests for the `flux2` FastVideo port. Compares
FastVideo's Flux2 components (DiT transformer, VAE, Qwen3 text encoder) against
the Diffusers reference and the published `black-forest-labs/FLUX.2-klein-4B`
checkpoint. Skipped in CI; CUDA required.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `flux2` |
| Workload types | `T2I` |
| Official reference | `diffusers.FluxTransformer2DModel`, `diffusers.AutoencoderKL`, `transformers.Qwen3ForCausalLM` |
| Local reference dir | `none` (Diffusers + transformers reference, no clone) |
| Official commit/version | diffusers >= 0.32, transformers >= 4.52 |
| HF weights | `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` |
| HF revision | latest |
| Local weights dir | `official_weights/black-forest-labs__FLUX.2-klein-4B` (env: `FLUX2_MODEL_DIR`) |
| Source layout | `diffusers` (native HF Diffusers format, no conversion needed) |
| Needs conversion | No |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. The
reference is the published Diffusers + transformers classes — no clone or
upstream install is required beyond the FastVideo pins.

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: none
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: none
```

## Weight Setup

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "black-forest-labs/FLUX.2-klein-4B" \
    "official_weights/black-forest-labs__FLUX.2-klein-4B"
```

## Tests in this directory

```bash
pytest tests/local_tests/flux2/ -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| DiT transformer | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Forward-pass numerical parity vs Diffusers | `scaffold` |
| VAE | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Encode/decode parity vs Diffusers | `scaffold` |
| Qwen3 text encoder | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Token embedding parity vs transformers | `scaffold` |

## Scope Notes

- **Validated**: Flux2 Klein (distilled, 4-step, Qwen3 text encoder, no guidance).
  End-to-end inference produces coherent images matching Diffusers pipeline output.
- **Scaffold only**: Full Flux2 (non-Klein, with guidance). Registry/config wiring
  present but text encoder config + conditioning path not validated.
- **Deferred**: SSIM regression tests; will be added once reference videos are seeded.

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
