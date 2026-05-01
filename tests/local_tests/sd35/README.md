# Stable Diffusion 3.5 Local Tests

Local-only component parity tests for the `sd35` FastVideo port. Compares
FastVideo's SD3.5 components (text encoders, transformer, VAE, scheduler)
against the Diffusers reference and the published `stabilityai/stable-diffusion-3.5-medium`
checkpoint. Skipped in CI; CUDA required.

Port progress, open questions, issues, and handoff notes live in
[`PORT_STATUS.md`](./PORT_STATUS.md).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `sd35` |
| Workload types | `T2I` |
| Official reference | `diffusers.SD3Transformer2DModel`, `diffusers.AutoencoderKL`, `transformers` text encoders |
| Local reference dir | `none` (Diffusers + transformers reference, no clone) |
| Official commit/version | `<TODO: diffusers + transformers versions>` |
| HF weights | `stabilityai/stable-diffusion-3.5-medium` |
| HF revision | `<TODO>` |
| Local weights dir | `official_weights/stabilityai__stable-diffusion-3.5-medium` (env: `SD35_MODEL_DIR`) |
| Source layout | `diffusers` |
| Needs conversion | `<TODO>` |

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
official_env_status: <TODO: imports_ok | private_deps_need_stubs | blocked>
private_dep_stubs: none
blocked_on: <TODO>
```

## Weight Setup

```bash
python "$HOME/.config/opencode/skill/add-model-prep/scripts/download_hf_weights.py" \
    "stabilityai/stable-diffusion-3.5-medium" \
    "official_weights/stabilityai__stable-diffusion-3.5-medium"
```

## Tests in this directory

```bash
pytest tests/local_tests/sd35/ -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| all components (text encoders, transformer, VAE, scheduler) | [`test_sd35_component_parity.py`](./test_sd35_component_parity.py) | single-file omnibus parity test | `<TODO>` |

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
- See [`PORT_STATUS.md`](./PORT_STATUS.md) for open issues, escape hatches, and
  the component matrix.
