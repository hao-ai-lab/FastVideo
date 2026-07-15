# Z-Image Local Tests

Local component coverage and quality handoff for the `zimage` FastVideo port
(`T2I`, Z-Image-Turbo only). The same PR is authorized to carry the components,
FastVideo-native `ZImagePipeline`, example, pipeline parity, SSIM test, and L40S
reference seed. The native pipeline surface and 8-step 1024x1024 PNG SSIM test
are implemented. Native one-step GPU parity is bit-exact, and the L40S candidate
has been generated; only its author-approved HF upload remains.

> **Status:** Implementation complete, Phase 10 quality regression at the
> required reference-review/upload gate. The
> Phase 6 exact-head closure is complete: the full pinned component suite passed
> 34/34 with no skips on GB200 at `f310aacb1`. The required production text
> encoder is the official-loader-compatible Transformers `AutoModel` path, which
> matches an independently materialized reference exactly. Scheduler, tokenizer,
> VAE, native Qwen fp32/bf16, and the 6.15B real-weight transformer all pass.
> Native one-step pipeline parity is bit-exact on RTX 5090, and the 1024x1024
> L40S candidate generated at inference head `9a6dbcadc` is staged for author review. See
> [`PORT_STATUS.md`](./PORT_STATUS.md) for the live state.

## Reference assets and scope

| Field | Value |
|---|---|
| Model family / variant | `zimage` / `Z-Image-Turbo` |
| Workload | `T2I` |
| PR scope | required components, native pipeline, example/parity, SSIM test, and L40S seed |
| Required component scope | production `AutoModel` text encoder, tokenizer, VAE, scheduler, native transformer |
| Additional quality scope | direct FastVideo-native Qwen parity (`model/r10`; not used by the pipeline loader) |
| Official reference | `https://github.com/Tongyi-MAI/Z-Image` |
| Local reference dir | `<repo_root>/Z-Image/src` |
| Official commit | `26f23eda626ffadda020b04ff79488e1d72004cd` |
| HF weights | `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6` |
| Local weights dir | `<repo_root>/official_weights/Z-Image/` (component subfolders; transformer is 24.6 GB) |
| Reference implementation layout | Official repo-native `src/zimage/` Python; Diffusers is not the parity oracle |
| Weight layout | HF per-component subfolders (`text_encoder/`, `vae/`, `scheduler/`, `transformer/`) |
| Conversion | not needed; the native and official production transformer expose the same 521 checkpoint keys/shapes |
| HF token env | `HF_TOKEN` (the pinned repository is public; never record a token value) |

The pinned text-encoder config declares `Qwen3ForCausalLM` with 36 decoder
blocks, hidden size 2560, intermediate size 9728, 32 attention heads, 8 KV
heads, and head dimension 128. FastVideo loads those values from
`text_encoder/config.json`; they are not hard-coded in the Z-Image integration.

## Shared environment setup

Run from the FastVideo repository root in the same environment used for
FastVideo. Keep the reference clone and weights inside the ignored paths shown
below.

Clone and pin the official reference:

```bash
python .agents/skills/add-model-01-prep/scripts/clone_reference_repo.py \
  https://github.com/Tongyi-MAI/Z-Image.git \
  Z-Image \
  --commit 26f23eda626ffadda020b04ff79488e1d72004cd \
  --update-gitignore
git -C Z-Image rev-parse HEAD
```

The final command must print
`26f23eda626ffadda020b04ff79488e1d72004cd`. The scheduler and VAE tests also
enforce that exact HEAD and verify that imported `zimage` modules resolve under
`<repo_root>/Z-Image/src`. A missing clone may skip local parity; a wrong SHA or
wrong import origin fails rather than silently testing another installation.

Download the immutable HF snapshot:

```bash
python .agents/skills/add-model-01-prep/scripts/download_hf_weights.py \
  Tongyi-MAI/Z-Image-Turbo \
  official_weights/Z-Image \
  --revision f332072aa78be7aecdf3ee76d5c247082da564a6 \
  --allow-pattern 'text_encoder/*' \
  --allow-pattern 'tokenizer/*' \
  --allow-pattern 'vae/*' \
  --allow-pattern 'scheduler/*'
```

The real-weight transformer test additionally needs `transformer/*` from the
same pinned revision (24.6 GB). Do not download it without approving that cost;
point `ZIMAGE_TRANSFORMER_DIR` at an existing pinned transformer subfolder.

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, or CUDA packages) without explicit approval.

```text
dependency_changes: none
official_env_status: official source pinned and imported; GB200 component validation run completed
private_dep_stubs: none
blocked_on: author approval in model/r12 before the staged L40S PNG is uploaded to HF
```

## Run the tests

```bash
pytest tests/local_tests/zimage/ -v -s
```

| Component | Coverage scope and contract | Status |
|---|---|---|
| Scheduler ([`test_zimage_scheduler_parity.py`](./test_zimage_scheduler_parity.py)) | `implementation_subcomponent`; exact official `sigma_min=0.0` plus `use_reference_discrete_timesteps=True`; positional/default-path regressions remain asset-free | PINNED GB200 PASS (5 passed) |
| Tokenizer ([`test_zimage_tokenizer_parity.py`](./test_zimage_tokenizer_parity.py)) | `production_loader`; exact `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=True)` and `max_length=512`; absence of `apply_chat_template` is a failure, not a skip | PINNED GB200 PASS (2 passed) |
| VAE ([`test_zimage_vae_parity.py`](./test_zimage_vae_parity.py)) | `both`; direct raw decode plus production `VAELoader`, config values, and raw decode | PINNED GB200 PASS (2 passed) |
| Production text encoder ([`test_zimage_encoder_parity.py`](./test_zimage_encoder_parity.py)) | `production_loader`; `TextEncoderLoader` materializes a body-only Transformers `AutoModel` matching official `src/utils/loader.py`; required by the pipeline | PINNED GB200 PASS (reference/production outputs exact) |
| Native Qwen quality ([`test_zimage_encoder_parity.py`](./test_zimage_encoder_parity.py)) | `implementation_subcomponent`; direct FastVideo-native construction, strict loading, and fp32/bf16 diagnostics; not loaded by the pipeline | PINNED GB200 PASS; fp32 `atol=2e-4`, `rtol=1e-4`; bf16 thresholds unchanged |
| Transformer ([`test_zimage_transformer_parity.py`](./test_zimage_transformer_parity.py)) | `both`; pinned production meta key/shape surface, additive-mask attention parity, deterministic tiny CPU full-forward parity, explicit `sp_size=1` guard, and real-weight production-loader/full-forward parity | PINNED GB200 PASS (5 passed; real-weight full forward exact) |

## Pinned official pipeline context

| Field | Contract |
|---|---|
| Oracle | `Tongyi-MAI/Z-Image@26f23eda626ffadda020b04ff79488e1d72004cd`, `src/zimage/pipeline.py::generate` and `src/config/inference.py`; never Diffusers |
| Pipeline/modules | `_class_name=ZImagePipeline`; `transformer`, `vae`, `text_encoder`, `tokenizer`, `scheduler` |
| Inputs/outputs | text-to-image prompt with optional negative prompt; PIL images by default or latents when requested |
| Official defaults | `1024x1024`, `num_inference_steps=8`, `guidance_scale=0.0`, `cfg_truncation=1.0`, `max_sequence_length=512` |
| Reproducible RNG | `torch.Generator("cuda").manual_seed(42)`; `generate()` accepts the generator rather than fixing a seed internally |
| Scheduler | `use_reference_discrete_timesteps=True`, runtime `sigma_min=0.0`, and the endpoint-preserving `num_steps + 1` schedule with the final zero-timestep DiT call skipped |
| Decode | apply `(latents / 0.3611) + 0.1159` before official VAE decode |
| FastVideo status | Native pipeline/config/preset/registry/example and PNG SSIM test are implemented. One-step RTX 5090 parity is bit-exact; the L40S candidate generated at `9a6dbcadc` awaits HF-upload approval. Later changes affect only API inventory/tests/docs. |

## Pipeline and quality verification

| Scope | Command | Status |
|---|---|---|
| Hardware-free pipeline surface | `pytest tests/local_tests/pipelines/test_zimage_pipeline_smoke.py::test_zimage_typed_surface_preflight -v` | PASS |
| Hardware-free native stage contract | `pytest tests/local_tests/pipelines/test_zimage_pipeline_parity.py::test_zimage_native_default_stage_math -v` | PASS |
| Pinned native full GPU pipeline parity | `ZIMAGE_REFERENCE_REPO=<pinned-clone> ZIMAGE_MODEL_DIR=<pinned-weights> DISABLE_SP=1 pytest tests/local_tests/pipelines/test_zimage_pipeline_parity.py::test_zimage_pipeline_latents_match_pinned_native_repo -v -s` | PASS on RTX 5090; one-step latents bit-exact (`max=0`, `mean=0`) |
| 8-step 1024x1024 PNG SSIM | `pytest fastvideo/tests/ssim/test_zimage_similarity.py -v -s` | L40S candidate generated at inference head `9a6dbcadc`; HF reference upload pending author review (`model/r12`) |

## Component contracts

### Text encoder and tokenizer

- The production `TextEncoderLoader` path deliberately loads Transformers
  `AutoModel`, even when the checkpoint advertises `Qwen3ForCausalLM`. FastVideo
  consumes hidden states only, so the production path is body-only and does not
  materialize or execute an LM head.
- The parity oracle is the pinned official Z-Image source and pipeline contract.
  A separate Transformers `AutoModel` instance materializes the official
  loader behavior independently; it is not the object returned by FastVideo's
  production loader. The native `Qwen3ForCausalLM` implementation is compared
  separately.
- CPU/MPS text-encoder offload remains on the requested device. The loader records
  `_fastvideo_input_device`, and `TextEncodingStage` sends token tensors there.
- Native loading must account for every destination parameter and every required
  fused source shard. Q/K/V and gate/up shards are all required unless the exact
  destination is supplied already fused; quantized auxiliary parameters such as
  `scale_weight` are not silently ignored.
- The pinned official prompt path enables thinking, tokenizes to 512, and consumes
  `hidden_states[-2]` at valid-token positions. A 36-block Qwen model exposes 37
  hidden-state entries in this contract (embedding/intermediate entries plus the
  final normalized state).
- `TextEncoderConfig.chat_template_enable_thinking` is keyword-only and defaults
  to `False` for existing model families. The Z-Image pipeline config opts in
  with `True`.

### VAE

The test covers both direct implementation parity and production `VAELoader`
resolution/strict loading, confirms the official config's scaling/shift values,
and compares raw decode outputs. The official pipeline's
`(latents / 0.3611) + 0.1159` transformation is pipeline behavior, so it remains
a Phase 7 Z-Image pipeline-stage and pipeline-parity requirement. This port
reuses the
existing shared `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` wrapper,
which subclasses Diffusers `AutoencoderKL`, as a narrowly scoped exception to
the native-component boundary. That runtime inheritance does not make Diffusers
the reference implementation: parity is against pinned
`Tongyi-MAI/Z-Image/src/zimage/autoencoder.py`. This is not precedent for the
transformer or pipeline implementation.

### Scheduler

The pinned official pipeline mutates `scheduler.sigma_min = 0.0` before building
its `num_steps + 1` schedule. The Z-Image pipeline applies both
`use_reference_discrete_timesteps=True` and `sigma_min=0.0`; the published
`scheduler_config.json` alone does not encode the complete runtime contract.

### Transformer

- The FastVideo-native transformer preserves all 521 published checkpoint keys.
  The sorted key digest from the pinned HF safetensors index is
  `3a9216f208c1873b2cf06394411a53e1e95e10fae3b01dca0f7223556e47c354`.
- Tiny CPU parity uses the pinned official class, identical deterministic state,
  variable-length image/text batches, and concrete full-forward output checks.
- Attention uses raw torch SDPA because the padded variable-length stream needs
  a key mask not exposed by FastVideo's distributed wrappers. Runtime rejects
  initialized sequence-parallel world sizes above one.
- The production-loader/full-forward test runs when the pinned transformer
  subfolder and CUDA are available. The pinned GB200 run strict-loaded the
  6.15B model and matched the official full-forward output exactly.

## Remaining work

- Obtain the author's `model/r12` visual approval before uploading only the
  staged `Tongyi-MAI__Z-Image-Turbo/TORCH_SDPA/*.png` reference subtree.
- Keep HF upload and merge as separate author-gated actions.
