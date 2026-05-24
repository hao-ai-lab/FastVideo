# Flux2 Local Tests

Local-only component and pipeline parity tests for the `flux2` FastVideo port.
Compares FastVideo's Flux2 components (DiT transformer, VAE, Qwen3 text encoder)
and Klein pipeline against the Diffusers reference and the published
`black-forest-labs/FLUX.2-klein-4B` checkpoint. Skipped in CI; CUDA required for
activated parity runs.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `flux2` |
| Workload types | `T2I` |
| Official reference | `diffusers.Flux2KleinPipeline`, `diffusers.Flux2Transformer2DModel`, `diffusers.AutoencoderKLFlux2`, `transformers.Qwen3ForCausalLM` |
| Local reference dir | `none` (Diffusers + transformers reference, no clone) |
| Official commit/version | diffusers >= 0.38.0, transformers >= 4.52 |
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

Do not change core dependency versions (`torch`, `transformers`, `flash-attn`,
`triton`, CUDA packages) without explicit approval. The required Diffusers floor
bump is recorded below.

## Official Environment Status

```text
dependency_changes: diffusers>=0.38.0
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

Port state is tracked in [`PORT_STATUS.md`](./PORT_STATUS.md).

```bash
pytest tests/local_tests/flux2/ -v -s

FLUX2_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-klein-4B \
pytest tests/local_tests/pipelines/test_flux2_pipeline_smoke.py \
       tests/local_tests/pipelines/test_flux2_pipeline_parity.py \
       tests/local_tests/flux2/test_flux2_component_parity.py -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| DiT transformer | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Strict weight load + numerical output parity vs Diffusers, including 5D pipeline path | `PASSED on Modal L40S` |
| VAE | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Encode/decode exact parity vs Diffusers | `PASSED on Modal L40S` |
| Qwen3 text encoder | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | HF passthrough load + exact hidden-state parity | `PASSED on Modal L40S` |
| Pipeline smoke | [`../pipelines/test_flux2_pipeline_smoke.py`](../pipelines/test_flux2_pipeline_smoke.py) | Import, registry, preset, config wiring; four-step latent generate | `PASSED on Modal L40S` |
| Pipeline parity | [`../pipelines/test_flux2_pipeline_parity.py`](../pipelines/test_flux2_pipeline_parity.py) | Four-step denoised latent parity vs `diffusers.Flux2KleinPipeline` | `PASSED on Modal L40S` |
| Pipeline TP2 parity | [`../pipelines/test_flux2_pipeline_parity.py`](../pipelines/test_flux2_pipeline_parity.py) | Two-worker tensor-parallel load/generate with `num_gpus=2`, `tp_size=2`, `sp_size=1` | `PASSED on Modal L40S:2` |
| Pixel image comparison | Modal image runner | Same-prompt Diffusers vs FastVideo PNG generation and pixel metrics | `PASSED on Modal L40S` |

## Latest Remote Evidence

Modal L40S run `ap-5Zha6ev4NhKsIahsjdWiEb` applied patch
`patches/flux2-local-cea4ed4d.patch` to commit
`c23820e93d7b77d4113ca8fceac8ef3a19f572d3` with
`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`.

```text
tests/local_tests/flux2/test_flux2_component_parity.py -v -s: 3 passed
tests/local_tests/pipelines/test_flux2_pipeline_smoke.py -v -s: 2 passed
tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s: 1 passed
Flux2 modal final statuses: component=0 smoke=0 pipeline=0
```

The parity tests print `assert_close` input means before every strict tensor
comparison. The pipeline parity run reported zero max/mean/median diff for all
four trajectory steps and final latents.

Additional Modal `L40S:2` allocation run `ap-sg5G52nqwBMDh1YN0IsoKd` applied
patch `patches/flux2-local-l40s2-f708504d.patch` to the same commit and
confirmed `torch.cuda.device_count() == 2`.

```text
tests/local_tests/flux2/test_flux2_component_parity.py -v -s: 3 passed
tests/local_tests/pipelines/test_flux2_pipeline_smoke.py -v -s: 2 passed
tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s: 1 passed
Flux2 modal L40S:2 statuses: component=0 smoke=0 pipeline=0
```

The `L40S:2` run verifies the same parity suite in a two-GPU Modal allocation.
These tests currently instantiate FastVideo with `num_gpus=1`, so this is not a
tensor-parallel two-GPU parity test.

Additional Modal `L40S:4` allocation run `ap-tQEUnFr00uOvpMZdOngebQ` applied
patch `patches/flux2-local-multil40s-cb92dbaa.patch` to the same commit and
confirmed `torch.cuda.device_count() == 4`.

```text
tests/local_tests/flux2/test_flux2_component_parity.py -v -s: 3 passed
tests/local_tests/pipelines/test_flux2_pipeline_smoke.py -v -s: 2 passed
tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s: 1 passed
Flux2 modal L40S:4 statuses: component=0 smoke=0 pipeline=0
```

Modal `L40S:2` tensor-parallel run `ap-szNgcJRiUv11lmmNFvjjPT` applied patch
`patches/flux2-local-tp2-c850fcad.patch`, confirmed two visible L40S devices,
and instantiated FastVideo with `num_gpus=2`, `tp_size=2`, `sp_size=1`,
`executor_world_size=2`.

```text
tests/local_tests/pipelines/test_flux2_pipeline_parity.py::test_flux2_klein_pipeline_tensor_parallel_latent_parity -v -s: 1 passed
TP2 final diff max=2.107544 mean=0.020110 median=0.015625
TP2 abs-mean drift diffusers=1.319441 fastvideo=1.319235
```

The TP2 test uses relaxed TP-specific bounds because BF16 tensor-parallel
matmuls are not bit-exact with the single-GPU Diffusers reference. The strict
single-GPU pipeline parity remains `atol=rtol=1e-4`.

Modal `L40S:1` image comparison run `ap-t0t6x3k15OoRhP56DYFcnj` generated a
same-prompt Diffusers reference PNG and FastVideo PNG for prompt
`a photo of a banana on a wooden table, studio lighting`, seed `0`,
1024x1024, four steps, guidance `1.0`. Artifacts were downloaded locally under
`outputs/flux2_image_compare/flux2_klein_seed0_files/`.

```text
official_diffusers.png: 1024x1024 RGB
fastvideo.png: 1024x1024 RGB
pixel max_abs_diff=5 mean_abs_diff=0.480136 median_abs_diff=0.0 rmse=0.694312
```

## Scope Notes

- **Validated**: Flux2 Klein (distilled, 4-step, Qwen3 text encoder, no guidance).
  End-to-end latent inference matches Diffusers exactly for the four-step Klein
  pipeline parity prompt.
- **Scaffold only**: Full Flux2 (non-Klein, with guidance). Registry/config wiring
  present but text encoder config + conditioning path not validated.

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
