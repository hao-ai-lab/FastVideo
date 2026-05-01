# Wan2.2 Local Tests

Local-only tests for the `wan22` (Wan2.2 Image-to-Video) FastVideo port. The
single test in this directory is a record-schema regression test that verifies
the I2V record creator handles missing CLIP embeddings — Wan2.2 I2V conditions
on the input image through the VAE only and skips `ImageEncodingStage`. Skipped
in CI; runs CPU-only.

Port progress, open questions, issues, and handoff notes live in
[`PORT_STATUS.md`](./PORT_STATUS.md).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `wan22` |
| Workload types | `I2V` |
| Official reference | `<TODO>` |
| Local reference dir | `<TODO>` |
| Official commit/version | `<TODO>` |
| HF weights | `<TODO>` |
| HF revision | `<TODO>` |
| Local weights dir | `<TODO>` |
| Source layout | `<TODO>` |
| Needs conversion | `<TODO>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

The current test in this directory is a CPU-only record-schema regression
(no weights, no upstream clone required). It loads
`fastvideo.dataset.dataloader.record_schema` and
`fastvideo.pipelines.pipeline_batch_info` directly from file via
`importlib.util` to bypass heavy package-level imports.

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: none
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: <TODO>
```

## Tests in this directory

```bash
pytest tests/local_tests/wan22/ -v
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `record schema` (i2v_record_creator) | [`test_i2v_record_no_clip.py`](./test_i2v_record_no_clip.py) | verifies record creator handles empty `image_embeds` (Wan2.2 I2V has no CLIP encoder) | `non_skip_pass` |

## Review Notes

- This directory currently covers only the I2V record-schema regression. Add
  encoder/transformer/VAE/pipeline parity tests here as the full Wan2.2 port
  proceeds.
- See [`PORT_STATUS.md`](./PORT_STATUS.md) for open issues, escape hatches, and
  the component matrix.
