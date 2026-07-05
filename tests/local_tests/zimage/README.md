# Z-Image Local Tests

Local-only component coverage for the `zimage` FastVideo port (`T2I`,
Z-Image-Turbo only). This component-only PR covers the Qwen3 text encoder,
tokenizer, AutoencoderKL VAE, and FlowMatchEulerDiscreteScheduler. The native
`ZImageTransformer2DModel`, pipeline, pipeline parity, example, and image-quality
regression remain out of scope.

> **Status:** DRAFT port, in progress. Every test that consumes the reference
> clone or HF assets requires a fresh non-skip rerun against the immutable pins
> below. Earlier A40/L40S results used an unrecorded HF snapshot and are retained
> only as historical diagnostics, not current verification evidence. See
> [`PORT_STATUS.md`](./PORT_STATUS.md) for the live state.

## Reference assets and scope

| Field | Value |
|---|---|
| Model family / variant | `zimage` / `Z-Image-Turbo` |
| Workload | `T2I` |
| Component-only PR scope | text encoder, tokenizer, VAE, scheduler |
| Official reference | `https://github.com/Tongyi-MAI/Z-Image` |
| Local reference dir | `<repo_root>/Z-Image/src` |
| Official commit | `26f23eda626ffadda020b04ff79488e1d72004cd` |
| HF weights | `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6` |
| Local weights dir | `<repo_root>/official_weights/Z-Image/` (`text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`) |
| Source layout | Diffusers-style per-component subfolders |
| Conversion | not needed for the reused components in this PR; the future native transformer/full-pipeline decision will come from native and official key/shape prototypes |
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

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, or CUDA packages) without explicit approval.

```text
dependency_changes: none
official_env_status: not_verified
private_dep_stubs: none
blocked_on: pinned-snapshot non-skip component reruns
```

## Run the tests

```bash
pytest tests/local_tests/zimage/ -v -s
```

| Component | Coverage scope and contract | Status |
|---|---|---|
| Scheduler ([`test_zimage_scheduler_parity.py`](./test_zimage_scheduler_parity.py)) | `implementation_subcomponent`; exact official `sigma_min=0.0` plus `use_reference_discrete_timesteps=True`; positional/default-path regressions remain asset-free | REVALIDATION REQUIRED |
| Tokenizer ([`test_zimage_tokenizer_parity.py`](./test_zimage_tokenizer_parity.py)) | `production_loader`; exact `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=True)` and `max_length=512`; absence of `apply_chat_template` is a failure, not a skip | REVALIDATION REQUIRED (historical unpinned PASS) |
| VAE ([`test_zimage_vae_parity.py`](./test_zimage_vae_parity.py)) | `both`; direct implementation decode plus production `VAELoader`; production check applies `(latents / scaling_factor) + shift_factor` before decode | REVALIDATION REQUIRED (historical unpinned direct-decode PASS only) |
| Text encoder ([`test_zimage_encoder_parity.py`](./test_zimage_encoder_parity.py)) | `both`; independent Transformers `AutoModel` reference, body-only production loader, and FastVideo-native implementation; fp32/bf16 output checks plus fused/split/quant-scale strictness | REVALIDATION REQUIRED (historical unpinned native PASS only) |

## Component contracts

### Text encoder and tokenizer

- The production `TextEncoderLoader` path deliberately loads Transformers
  `AutoModel`, even when the checkpoint advertises `Qwen3ForCausalLM`. FastVideo
  consumes hidden states only, so the production path is body-only and does not
  materialize or execute an LM head.
- The parity oracle is a separate `AutoModel` instance; it is not the object
  returned by the production loader. The native `Qwen3ForCausalLM` implementation
  is compared separately.
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
resolution/strict loading. This component-only PR explicitly accepts the
existing shared `fastvideo.models.vaes.autoencoder_kl.AutoencoderKL` wrapper,
which subclasses Diffusers `AutoencoderKL`, as a narrowly scoped exception to
the native-component boundary. This is not precedent for the future transformer
or full Z-Image pipeline, and the exception must be revisited if that scope grows.

### Scheduler

The pinned official pipeline mutates `scheduler.sigma_min = 0.0` before building
its `num_steps + 1` schedule. A future Z-Image pipeline config must set both
`use_reference_discrete_timesteps=True` and `sigma_min=0.0`; the published
`scheduler_config.json` alone does not encode the complete runtime contract.

## Remaining work

- Run every asset-backed component test non-skip against the pinned clone and HF
  snapshot before claiming component parity.
- Port and validate `ZImageTransformer2DModel` in a separate PR.
- Add the Z-Image pipeline/config/preset/registry/example, pipeline parity, and
  image-quality regression after all component gates pass.
