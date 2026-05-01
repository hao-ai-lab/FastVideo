# Kandinsky-5 Local Tests

Local-only parity tests for the `kandinsky5` FastVideo port. Compares the
FastVideo Kandinsky-5 transformer against the Diffusers reference
(`diffusers.Kandinsky5Transformer3DModel`); skipped in CI.

Port progress, open questions, issues, and handoff notes live in
[`PORT_STATUS.md`](./PORT_STATUS.md).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `kandinsky5` |
| Workload types | `T2V` |
| Official reference | `diffusers.Kandinsky5Transformer3DModel` (Diffusers package) |
| Local reference dir | `none` (Diffusers reference, no clone) |
| Official commit/version | `<TODO: diffusers version>` |
| HF weights | `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` |
| HF revision | `<TODO>` |
| Local weights dir | `official_weights/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` (env: `KANDINSKY5_DIFFUSERS_PATH`, `KANDINSKY5_TRANSFORMER_PATH`) |
| Source layout | `diffusers` |
| Needs conversion | `<TODO>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. The
reference is the published Diffusers class — no clone or upstream install is
required beyond the FastVideo `diffusers` pin.

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
    "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers" \
    "official_weights/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
```

## Tests in this directory

```bash
pytest tests/local_tests/kandinsky5/ -v
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `transformer` (Kandinsky5Transformer3DModel) | [`test_kandinsky5_lite_transformer_parity.py`](./test_kandinsky5_lite_transformer_parity.py) | `<TODO>` | `<TODO>` |

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
- See [`PORT_STATUS.md`](./PORT_STATUS.md) for open issues, escape hatches, and
  the component matrix.
