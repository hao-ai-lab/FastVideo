# Z-Image Local Tests

Local-only component coverage for the `zimage` FastVideo port (`T2I`,
Z-Image-Turbo only). Coverage includes the Qwen3 text encoder, tokenizer,
AutoencoderKL VAE, FlowMatchEulerDiscreteScheduler, and a native
`ZImageTransformer2DModel`. Pipeline wiring, pipeline parity, examples, and
image-quality regression remain out of scope.

> **Status:** DRAFT port, in progress. The pinned official-source GB200 run now
> passes the scheduler, tokenizer, VAE, bf16 encoder, and real-weight transformer
> gates. The only unresolved component assertion is the fp32 native encoder's
> small accumulated tail at `hidden_states[-2]`; its tolerance decision is in
> maintainer packet `model/r10`. A complete non-skip rerun at the resulting exact
> PR head is still required. See
> [`PORT_STATUS.md`](./PORT_STATUS.md) for the live state.

## Reference assets and scope

| Field | Value |
|---|---|
| Model family / variant | `zimage` / `Z-Image-Turbo` |
| Workload | `T2I` |
| Component scope | text encoder, tokenizer, VAE, scheduler, native transformer |
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
blocked_on: model/r10 fp32 encoder tolerance decision, then one exact-head full-suite rerun
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
| Text encoder ([`test_zimage_encoder_parity.py`](./test_zimage_encoder_parity.py)) | `both`; independent Transformers `AutoModel` materialization of the official loader contract, body-only production loader, and FastVideo-native implementation; fp32/bf16 output checks plus fused/split/quant-scale strictness | PRODUCTION EXACT + BF16 PASS; FP32 PRE-NORM TOLERANCE DECISION PENDING (`model/r10`) |
| Transformer ([`test_zimage_transformer_parity.py`](./test_zimage_transformer_parity.py)) | `both`; pinned production meta key/shape surface, additive-mask attention parity, deterministic tiny CPU full-forward parity, explicit `sp_size=1` guard, and real-weight production-loader/full-forward parity | PINNED GB200 PASS (5 passed; real-weight full forward exact) |

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
  to `False` for existing model families. A future Z-Image pipeline config must
  opt in with `True`.

### VAE

The test covers both direct implementation parity and production `VAELoader`
resolution/strict loading, confirms the official config's scaling/shift values,
and compares raw decode outputs. The official pipeline's
`(latents / 0.3611) + 0.1159` transformation is pipeline behavior, so it remains
a Phase 7 Z-Image pipeline-stage and pipeline-parity requirement. This
component-only PR explicitly accepts the
existing shared `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` wrapper,
which subclasses Diffusers `AutoencoderKL`, as a narrowly scoped exception to
the native-component boundary. That runtime inheritance does not make Diffusers
the reference implementation: parity is against pinned
`Tongyi-MAI/Z-Image/src/zimage/autoencoder.py`. This is not precedent for the
future transformer or full Z-Image pipeline, and the exception must be revisited
if that scope grows.

### Scheduler

The pinned official pipeline mutates `scheduler.sigma_min = 0.0` before building
its `num_steps + 1` schedule. A future Z-Image pipeline config must set both
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

- Resolve `model/r10`, which asks whether the traced Z-Image-only native encoder
  fp32 `atol` may move from `1e-4` to `2e-4` while retaining `rtol=1e-4` and all
  other gates.
- Run the entire component suite non-skip in a clean worktree at the resulting
  exact PR head; the earlier full run preceded the loader/oracle corrections.
- Add the Z-Image pipeline/config/preset/registry/example, pipeline parity, and
  image-quality regression after all component gates pass.
