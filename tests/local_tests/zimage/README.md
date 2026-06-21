# Z-Image Local Tests

Local-only component parity tests for the `zimage` FastVideo port (T2I).
Compares FastVideo's Qwen3 text encoder, AutoencoderKL VAE, and
FlowMatchEulerDiscreteScheduler (Z-Image reference-timestep branch) against
the local Z-Image reference implementation. Skipped in CI; CUDA recommended
(required for bf16 encoder parity).

> **Status:** DRAFT port, in-progress. The transformer (ZImageTransformer2DModel),
> pipeline, conversion script, and SSIM media regression are not yet
> implemented — see [`PORT_STATUS.md`](./PORT_STATUS.md) for the live state.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `zimage` |
| Workload types | `T2I` |
| Official reference | Tongyi-MAI / `Z-Image` (Qwen3-based T2I) |
| Local reference dir | `<repo_root>/Z-Image/src` (cloned manually) |
| Official commit/version | `<TODO: pin a Z-Image git SHA before handoff>` |
| HF weights | `<TODO: pin published HF id once Tongyi-MAI publishes weights>` |
| Local weights dir | `<repo_root>/official_weights/Z-Image/` (subfolders: `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`) |
| Source layout | Diffusers-style (per-component subfolders + `config.json`) |
| Needs conversion | `<TODO: confirm after transformer port — encoder loads raw safetensors today>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. The
reference is the Z-Image source clone under `<repo_root>/Z-Image/src/` — the
scheduler and VAE parity tests add this to `sys.path` at import time.

Clone the reference (one-time):

```bash
git clone <Z-Image repo URL> Z-Image
cd Z-Image && git checkout <TODO: pin SHA> && cd ..
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: none
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: <TODO: pin Z-Image reference clone SHA>
```

## Tests in this directory

```bash
pytest tests/local_tests/zimage/ -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| Scheduler (`FlowMatchEulerDiscreteScheduler` + `use_reference_discrete_timesteps`) | [`test_zimage_scheduler_parity.py`](./test_zimage_scheduler_parity.py) | full `scheduler_config.json` forwarded; pipeline must also pin the new flag at load time | PASS |
| Tokenizer (`TokenizerLoader` vs `AutoTokenizer`) | [`test_zimage_tokenizer_parity.py`](./test_zimage_tokenizer_parity.py) | `apply_chat_template` parity included | PASS |
| VAE decode (`AutoencoderKL`) | [`test_zimage_vae_parity.py`](./test_zimage_vae_parity.py) | decode-only; encode path deferred until pipeline | PASS |
| Text encoder (shared `Qwen3ForCausalLM`, reused) | [`test_zimage_encoder_parity.py`](./test_zimage_encoder_parity.py) | parametrized fp32 + bf16; bf16 uses calibrated distribution checks + diagnostic prints. Z-Image's `Qwen3Model` checkpoint routes to the shared encoder via the registry | PASS (fp32 bit-exact + bf16, L40S 2026-06-21) |

## Known Blockers / Open Items

See [`PORT_STATUS.md`](./PORT_STATUS.md) for the live tracker. Highlights:

- The text encoder reuses the shared `Qwen3ForCausalLM` (added for Flux2 Klein,
  #1349); Z-Image-Turbo's `Qwen3Model` architecture string routes to it via the
  model registry. Z-Image-Turbo's full Qwen3 checkpoint carries an `lm_head.weight`
  the body-only encoder does not own, so the encoder parity test allowlists exactly
  that key (`_ALLOWED_UNEXPECTED_KEYS`) and fails on any other unmatched key.
- `tests/local_tests/zimage/test_zimage_scheduler_parity.py` builds the
  FastVideo scheduler with `use_reference_discrete_timesteps=True` programmatically.
  When the pipeline lands, `<repo_root>/official_weights/Z-Image/scheduler/scheduler_config.json`
  must pin this flag, otherwise stock loaders will silently fall back to the
  default Diffusers timestep mode (numerically different).
- The shared `Qwen3TextArchConfig.text_len = 512` derives `tokenizer_kwargs.max_length = 512`
  via `TextEncoderArchConfig.__post_init__`, but the parity tests tokenize at
  `max_length=96..128`. Reconcile when the pipeline preset lands. (Also note the
  shared config defaults `is_chat_model=True`, vs Z-Image's removed bespoke `False`.)

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including the bf16 path for the encoder.
- Pipeline parity, conversion, transformer, and SSIM are deferred to future
  PRs; this directory will gain new tests as those land.
