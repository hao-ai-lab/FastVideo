# Kandinsky-6 Local Tests

Local-only parity tests for the `kandinsky6` FastVideo port (T2IVA and T2IVA distilled: text, optionally plus an
image, to video with audio). Compares the FastVideo Kandinsky6 transformer against the Diffusers reference
(`diffusers.Kandinsky6Transformer3DModel`); skipped in CI.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `kandinsky6` |
| Workload types | `T2V`, `I2V` (one merged TI2VA pipeline serves both) |
| Official reference | `diffusers.Kandinsky6Transformer3DModel` (used by `diffusers.Kandinsky6TI2VAPipeline`) |
| Local reference dir | set `KANDINSKY6_DIFFUSERS_REPO_PATH` to a `diffusers` checkout that provides Kandinsky6 (its `src/` is prepended to `sys.path`); only needed when the installed `diffusers` does not |
| Official commit/version | not pinned; the official repos are exported with `_diffusers_version` `0.41.0.dev0` |
| HF weights | `kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` (T2IVA), `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` (T2IVA distilled) |
| HF revision | `main` |
| Local weights dir | `official_weights/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` (env: `KANDINSKY6_DIFFUSERS_PATH`, `KANDINSKY6_TRANSFORMER_PATH`) |
| Source layout | `diffusers` |
| Needs conversion | no: the official repos are in Diffusers layout and load directly |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Why this differs from `tests/local_tests/kandinsky5/`

Kandinsky5's parity test imports its diffusers reference class directly from the pip-installed `diffusers` package.
`test_kandinsky6_transformer_parity.py` tries the same import first and, when the installed `diffusers` does not provide
`Kandinsky6Transformer3DModel`, falls back to a `diffusers` checkout at `KANDINSKY6_DIFFUSERS_REPO_PATH`. The test
skips when the transformer weights are missing, when CUDA is unavailable, or when the reference cannot be imported; a
skip is not a verified pass (see `.agents/skills/add-model/shared/common_rules.md`).

The reference's `forward()` also uses a packed/ragged `(sum_T, H, W, C)` layout (`cu_seqlens`-addressed) rather than
FastVideo's ordinary batched `(B, T, H, W, C)` tensor. The test converts between them (batch size 1, so it is a plain
squeeze/unsqueeze).

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo.

```bash
export KANDINSKY6_DIFFUSERS_REPO_PATH=/path/to/diffusers-checkout-with-kandinsky6
export KANDINSKY6_DIFFUSERS_PATH=official_weights/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Weight Setup

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers" \
    "official_weights/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
```

The base repo is about 90 GB (the transformer alone is 70 GB); set `KANDINSKY6_TRANSFORMER_PATH` to a different
`transformer/` directory to test another checkpoint.

## Tests in this directory

```bash
pytest tests/local_tests/kandinsky6/ -v
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `transformer` (Kandinsky6Transformer3DModel) | [`test_kandinsky6_transformer_parity.py`](./test_kandinsky6_transformer_parity.py) | Packed-vs-batched shape conversion; needs a CUDA GPU that holds the 60 GB transformer | `scaffold_skip` |

Audio codec (`MMAudioVAE` + `BigVGANV2`) has no dedicated parity test here:
Kandinsky6 reuses those classes unmodified from FastVideo's existing MMAudio
port (verified via byte-identical `DATA_MEAN_128D`/`DATA_STD_128D` constants
against the diffusers reference), so their existing parity coverage applies.

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
