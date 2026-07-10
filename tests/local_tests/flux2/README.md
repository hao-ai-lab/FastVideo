# Flux2 Local Tests

Local-only component and pipeline parity tests for the `flux2` FastVideo port.
Compares FastVideo's Flux2 components and pipelines against the Diffusers
reference. Flux2 Klein uses Qwen3 and the published
`black-forest-labs/FLUX.2-klein-4B` checkpoint. Full Flux2 uses the Mistral3 /
AutoProcessor text path and is activated with `FLUX2_FULL_MODEL_DIR`. Skipped in
CI; CUDA required for activated parity runs.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `flux2` |
| Workload types | `T2I` |
| Official reference | `diffusers.Flux2Pipeline`, `diffusers.Flux2KleinPipeline`, `diffusers.Flux2Transformer2DModel`, `diffusers.AutoencoderKLFlux2`, `transformers.Mistral3ForConditionalGeneration`, `transformers.Qwen3ForCausalLM` |
| Local reference dir | `none` (Diffusers + transformers reference, no clone) |
| Official commit/version | diffusers >= 0.38.0, transformers >= 4.52 |
| HF weights | `black-forest-labs/FLUX.2-dev`, `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B` |
| HF revision | latest |
| Local weights dir | Klein: `official_weights/black-forest-labs__FLUX.2-klein-4B` (env: `FLUX2_MODEL_DIR`); full: env `FLUX2_FULL_MODEL_DIR` |
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

FLUX2_FULL_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-dev \
pytest tests/local_tests/flux2/test_flux2_component_parity.py::test_flux2_mistral3_text_encoder_parity \
       tests/local_tests/flux2/test_flux2_component_parity.py::test_flux2_full_transformer_guidance_parity \
       tests/local_tests/flux2/test_flux2_component_parity.py::test_flux2_full_vae_encode_decode_parity \
       tests/local_tests/pipelines/test_flux2_pipeline_smoke.py::test_flux2_full_pipeline_load_generate_smoke \
       tests/local_tests/pipelines/test_flux2_pipeline_parity.py::test_flux2_full_pipeline_latent_parity -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| DiT transformer | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Strict weight load + numerical output parity vs Diffusers, including 5D pipeline path | `PASSED on Modal L40S` |
| Full DiT transformer | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Strict full-weight load + embedded-guidance numerical output parity vs Diffusers | `PASSED on Modal L40S` |
| VAE | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Encode/decode exact parity vs Diffusers | `PASSED on Modal L40S` |
| Full VAE | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Full-weight encode/decode exact parity vs Diffusers | `PASSED on Modal L40S` |
| Qwen3 text encoder | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | HF passthrough load + exact hidden-state parity | `PASSED on Modal L40S` |
| Mistral3 text encoder | [`test_flux2_component_parity.py`](./test_flux2_component_parity.py) | Full Flux2 HF passthrough load + exact hidden-state parity | `PASSED on Modal L40S` |
| Pipeline smoke | [`../pipelines/test_flux2_pipeline_smoke.py`](../pipelines/test_flux2_pipeline_smoke.py) | Import, registry, preset, config wiring; four-step latent generate | `PASSED on Modal L40S` |
| Full pipeline smoke | [`../pipelines/test_flux2_pipeline_smoke.py`](../pipelines/test_flux2_pipeline_smoke.py) | Full Flux2 Mistral3/AutoProcessor wiring; short latent generate | `PASSED on Modal L40S:2` |
| Pipeline parity | [`../pipelines/test_flux2_pipeline_parity.py`](../pipelines/test_flux2_pipeline_parity.py) | Four-step denoised latent parity vs `diffusers.Flux2KleinPipeline` | `PASSED on Modal L40S` |
| Full pipeline parity | [`../pipelines/test_flux2_pipeline_parity.py`](../pipelines/test_flux2_pipeline_parity.py) | Short full Flux2 latent parity vs `diffusers.Flux2Pipeline` | `PASSED on Modal L40S:2, L40S:4, and H100:1` |
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

Modal `L40S:2` current-changes rerun `ap-CtccuhEHUwQy4Zv08nmrom` applied
`/root/data/flux2_l40s2_current/runner/flux2-current.patch` to commit
`69d22881a266306ad3bdbe820508ac17c13d2798` with
`FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`, `FLUX2_TP_SIZE=2`, and two visible
L40S devices.

```text
tests/local_tests/flux2/test_flux2_component_parity.py -v -s: 3 passed
tests/local_tests/pipelines/test_flux2_pipeline_smoke.py -v -s: 2 passed
tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s: 2 passed
```

The pipeline parity file covered both strict single-GPU latent parity and true
TP2 latent parity. The strict path reported zero max/mean/median diff for all
four trajectory steps and final latents. The TP2 path instantiated FastVideo
with `num_gpus=2`, `tp_size=2`, `sp_size=1`, and `executor_world_size=2`.

```text
TP2 final diff max=2.107544 mean=0.020110 median=0.015625
TP2 abs-mean drift diffusers=1.319441 fastvideo=1.319235
```

The same Modal run also regenerated the same-prompt Diffusers and FastVideo PNGs
with the current changes. Artifacts were downloaded locally under
`flux2_image_compare_results_l40s2_current/`.

```text
official_diffusers.png: 1024x1024 RGB
fastvideo.png: 1024x1024 RGB
comparison_grid.png: 3072x1058 RGB
pixel max_abs_diff=5 mean_abs_diff=0.480136 median_abs_diff=0.0 rmse=0.694312
```

Full Flux2 weight probe:

- Modal run `ap-70hWVB2eZAE1JWw0Im4E2T` checked
  `/root/data/official_weights` and found only
  `/root/data/official_weights/black-forest-labs__FLUX.2-klein-4B`.
- Modal run `ap-pp2v3Iu3NvJrkiMVdsIk0x` confirmed the same with
  `ls -la /root/data/official_weights`.
- Modal run `ap-erDuzyLLnsd3OXKaVAhlUW` rechecked the current volume and still
  found only `/root/data/official_weights/black-forest-labs__FLUX.2-klein-4B`.
- Local env-var name probe found `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and
  `HF_API_KEY` absent, so the launcher cannot pass a gated HF token through to
  Modal.

The full `FLUX.2-dev` checkpoint was later staged and committed to the Modal
`hf-model-weights` volume by app `ap-0SFRuDEG24nHlGKWhZwCcj`; the staged path is
`/root/data/official_weights/black-forest-labs__FLUX.2-dev`.

Current-session Modal evidence:

- Modal run `ap-tIeZbOsqsjMfN5qXJoz0WG` applied the current local patch after
  the launcher venv fix and passed:
  `test_flux2_full_typed_surface_preflight` and
  `test_flux2_full_text_stage_uses_mistral3_format_and_embedded_guidance`.
- Modal run `ap-oetobVLKHRX7uBkZ1ZTo7X` applied the current local patch, ran
  `examples/inference/basic/basic_flux2_klein.py`, and verified
  `outputs/flux2/flux2_klein_example.png` as `1024x1024 RGB`.
- Modal run `ap-d3yQLJ2jwAYewcCr5pyBYJ` passed full Mistral3 hidden-state
  parity with exact zero diff.
- Modal app logs for `ap-i6ZLmxsptBW49OP1DC38ZE` confirmed full transformer
  CPU BF16 parity passed with exact zero diff.
- Modal run `ap-5Nm5rPHf0Dph5M6En9azVT` passed full VAE encode/decode parity.
- Modal `L40S:2` run `ap-XPH4aM4LZxKhHJz7IDSMRg` passed full pipeline smoke at
  `128x128`, one step, `max_sequence_length=64`, `tp_size=2`, `sp_size=1`.
- Modal `L40S:2` run `ap-nVr8tDtQ0lVwviZDm6rIjH` passed full pipeline latent
  parity. Final latent diff max `0.062500`, mean `0.008581`, median
  `0.007812`.
- Modal `L40S:2` diagnostic run `ap-nhB228LmgVjGP6oRuCYSPT` reproduced the
  full pipeline latent parity diff and printed quantization details: max
  `0.062500`, mean `0.008581`, median `0.007812`; only `9/8192` latent entries
  hit the max bucket.
- Modal `L40S:4` run `ap-KKOzv9THDmTYt3gevhQkVR` passed full pipeline latent
  parity with true TP4 (`num_gpus=4`, `tp_size=4`, `sp_size=1`). Final latent
  diff max stayed `0.062500`, mean was `0.007946`, and median was `0.007812`.
- Modal `L40S:4` diagnostic run `ap-SlhykTTLxzsYxnsCUbCH7S` reproduced the TP4
  result with max `0.062500`, mean `0.007946`, median `0.007812`; only `6/8192`
  latent entries hit the max bucket.
- Modal `L40S:2` input-variant diagnostic run `ap-5N1yHy8udeZKslrQJTFHYX` used
  `FLUX2_FULL_RUN_INPUT_VARIANTS=1` to change prompt/seed. Changed prompt with
  seed `0` produced max `0.062500`, mean `0.006670`, median `0.003906`, and
  `5/8192` max-bucket entries. Default prompt with seed `123` produced max
  `0.062500`, mean `0.008709`, median `0.007812`, and `22/8192` max-bucket
  entries.
- Modal `L40S:4` input-variant diagnostic run `ap-TvJQ5cvNculTxJbTJAPNzx`
  passed the same cases. Changed prompt with seed `0` produced max `0.062500`,
  mean `0.006709`, median `0.003906`, and `3/8192` max-bucket entries. Default
  prompt with seed `123` produced max `0.062500`, mean `0.008820`, median
  `0.007812`, and `8/8192` max-bucket entries.
- Modal `L40S:1` run `ap-A8mRqCmPjUEs3kZJ3j8twH` attempted the same current
  full pipeline latent parity command with TP1. Diffusers produced reference
  latents, but FastVideo OOMed while loading the full transformer before parity
  comparison.
- Modal `L40S:1` setup probe `ap-FfM5ggOjtmc1oYK93T9Bvd` passed
  `test_flux2_full_pipeline_setup_matches_diffusers` with exact zero diff for
  Mistral3 prompt embeddings, text ids, raw/packed latents, image ids,
  timesteps, guidance scaling, and packed-vs-5D scheduler stepping.
- Modal `L40S:1` direct CUDA full-transformer component attempt
  `ap-LJm3FpKIJJzf1mosv7qfmd` OOMed before forward while moving the Diffusers
  transformer to GPU, so TP1/single-GPU CUDA full-transformer evidence remains
  infeasible on one L40S.
- Modal `H100:1` run `ap-HOpne8l9NbpWwU9qhUXYxT` used the updated
  `fastvideo/tests/modal/launch_l40s_job.py --gpu-type H100` path and passed
  `test_flux2_full_pipeline_latent_parity` with `FLUX2_FULL_NUM_GPUS=1`,
  `FLUX2_FULL_TP_SIZE=1`, and `FLUX2_FULL_SP_SIZE=1`. The trajectory step 0 and
  final packed latent diffs were exactly zero: max `0.000000`, mean `0.000000`,
  median `0.000000`.
- Modal `H100:1` run `ap-Gy6MRyZduxTHihgxiWWL13` generated full Flux2
  Diffusers/FastVideo image comparison artifacts at `1024x1024`, four steps,
  guidance `4.0`, max sequence length `64`. Local artifacts are in
  `flux2_full_image_compare_20260526_h100_full_t2i_files/`; pixel metrics:
  max absolute diff `14`, mean absolute diff `0.658212`, median `1.0`, RMSE
  `0.848854`.
- Modal `L40S:2` run `ap-xJ1MnjIQz79KPD4X9EOTLY` ran
  `examples/inference/basic/basic_flux2.py` and verified the generated PNG as
  `(128, 128) RGB`.

## Scope Notes

- **Validated**: Flux2 Klein (distilled, 4-step, Qwen3 text encoder, no guidance).
  End-to-end latent inference matches Diffusers exactly for the four-step Klein
  pipeline parity prompt.
- **Validated**: Full Flux2 T2I (`Flux2Pipeline`) uses
  Mistral3/AutoProcessor text conditioning and treats `guidance_scale` as
  embedded transformer guidance. Full component parity, pipeline smoke, latent
  parity, and example generation passed on Modal with `FLUX2_FULL_MODEL_DIR`.
- **Investigated**: The full TP latent max diff `0.062500` is not caused by
  prompt setup, latent packing, ids, timesteps, guidance scaling, or scheduler
  layout. Dedicated setup parity is bit-exact, and prompt/seed changes preserve
  the same worst-case bucket; the remaining diff is rare BF16-grid TP denoiser
  drift.
- **Validated**: Full Flux2 single-GPU pipeline parity is exact on Modal H100:1.
  The nonzero full pipeline diff is therefore specific to tensor-parallel
  execution, not the full model wiring.
- **Deferred**: Full Flux2 image conditioning and caption upsampling are not
  claimed by this port.

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as pending coverage; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
