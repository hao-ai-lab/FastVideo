# GEN3C Local Tests

Local-only parity and smoke tests for the `gen3c` FastVideo port. The pipeline
smoke test runs without weights; the transformer parity test compares against
the official `GEN3C` repo and converted weights, and is skipped in CI.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `gen3c` |
| Workload types | `T2V` |
| Official reference | `GEN3C` (local clone, see test files) |
| Local reference dir | `GEN3C/` |
| Official commit/version | `<TODO>` |
| HF weights | `<TODO>` |
| HF revision | `<TODO>` |
| Local weights dir | `<TODO>` (env: `GEN3C_FASTVIDEO_PATH`, `GEN3C_DIFFUSERS_PATH`) |
| Source layout | `<TODO>` |
| Needs conversion | `<TODO>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. See the
`add-model-prep` skill for canonical clone + install commands.

```bash
# Clone the official reference under the FastVideo repo root.
git clone <official_repo_url> GEN3C
uv pip install --no-deps -e ./GEN3C
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
pytest tests/local_tests/gen3c/ -v
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `transformer` (Gen3CTransformer3DModel) | [`test_gen3c.py`](./test_gen3c.py) | `<TODO>` | `<TODO>` |
| `pipeline` (smoke) | [`test_gen3c_pipeline_smoke.py`](./test_gen3c_pipeline_smoke.py) | random-weight smoke; full mode requires `GEN3C_DIFFUSERS_PATH` | `<TODO>` |

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
