# LTX-2 Local Tests

Local-only parity, registry, and pipeline-smoke tests for the `ltx2` FastVideo
port. Most tests depend on a checked-out `LTX-2/` directory at the repo root
(`FastVideo/LTX-2`) and converted Diffusers weights; without those they will
skip or fail. Skipped in CI; run locally on a single GPU.

Port progress, open questions, issues, and handoff notes live in
[`PORT_STATUS.md`](./PORT_STATUS.md).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `ltx2` |
| Workload types | `T2V` (video + optional audio) |
| Official reference | `Lightricks/LTX-2` (local clone) |
| Local reference dir | `LTX-2/` (with `LTX-2/packages/ltx-core/src` on `sys.path`) |
| Official commit/version | `<TODO>` |
| HF weights | `Lightricks/LTX-2`, `FastVideo/LTX2-base`, `FastVideo/LTX2-Distilled-Diffusers` |
| HF revision | `<TODO>` |
| Local weights dir | `<TODO>` (env: `LTX2_DIFFUSERS_PATH`, `LTX2_FASTVIDEO_GEMMA_LOG`, etc.) |
| Source layout | `<TODO: diffusers / raw_official / mixed>` |
| Needs conversion | `<TODO>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. See the
`add-model-prep` skill for canonical clone + install commands.

```bash
# Clone the official reference under the FastVideo repo root.
git clone <official_repo_url> LTX-2
uv pip install --no-deps -e ./LTX-2
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: <TODO>
official_env_status: <TODO: imports_ok | private_deps_need_stubs | blocked>
private_dep_stubs: <TODO>
blocked_on: <TODO>
```

## Tests in this directory

Run the whole family:

```bash
pytest tests/local_tests/ltx2/ -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `text encoder` (Gemma) | [`test_ltx2_gemma_encoder.py`](./test_ltx2_gemma_encoder.py) | encoder + connector | `<TODO>` |
| `text encoder` (Gemma, full parity) | [`test_ltx2_gemma_parity.py`](./test_ltx2_gemma_parity.py) | full Gemma parity vs official | `<TODO>` |
| `transformer/DiT` | [`test_ltx2.py`](./test_ltx2.py) | LTX2VideoConfig DiT parity | `<TODO>` |
| `transformer/DiT` (audio modality) | [`test_ltx2_audio.py`](./test_ltx2_audio.py) | audio-conditioning DiT parity (imports helpers from `test_ltx2.py`) | `<TODO>` |
| `vae` (video) | [`test_ltx2_vae.py`](./test_ltx2_vae.py) | LTX2VideoEncoder/Decoder parity | `<TODO>` |
| `vae` (video, official path) | [`test_ltx2_vae_official.py`](./test_ltx2_vae_official.py) | LTX2VAEConfig + VAELoader path | `<TODO>` |
| `vae` (audio) | [`test_ltx2_audio_vae.py`](./test_ltx2_audio_vae.py) | audio encoder/decoder/vocoder parity | `<TODO>` |
| `pipeline` (smoke) | [`test_ltx2_pipeline_smoke.py`](./test_ltx2_pipeline_smoke.py) | VideoGenerator end-to-end smoke | `<TODO>` |
| `registry` | [`test_ltx2_registry.py`](./test_ltx2_registry.py) | sampling/pipeline registry resolution | `<TODO>` |

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
- See [`PORT_STATUS.md`](./PORT_STATUS.md) for open issues, escape hatches, and
  the component matrix.
