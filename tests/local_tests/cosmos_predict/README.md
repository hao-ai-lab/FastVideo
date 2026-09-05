# cosmos_predict Local Tests

Local-only parity and smoke tests for the `cosmos_predict` FastVideo port. These
tests compare FastVideo against the official reference implementation and are
not expected to run in CI unless explicitly promoted later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/<model_family>/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `cosmos_predict` |
| Workload types | `Text2World` |
| Official reference | `https://github.com/NVIDIA/Cosmos` |
| Local reference dir | `Cosmos` |
| Official commit/version | `e7ad5e77eecd47acadf17db47d6eb56282a099cc` |
| HF weights | `nvidia/Cosmos-Predict2.5-2B` |
| HF revision | `main` |
| Local weights dir | `official_weights/cosmos_predict` |
| Source layout | `raw_official` |
| Needs conversion | `yes` |

Do not write token values in this file. Use only the token env var name:
`<HF_TOKEN or HUGGINGFACE_HUB_TOKEN or HF_API_KEY>`.

## Shared Environment Setup

Run from the FastVideo repo root in the same conda/env used for FastVideo.
Do not create a separate upstream environment for parity tests.

```bash
# Official reference source, if cloneable.
python ".agents/skills/add-model-01-prep/scripts/clone_reference_repo.py" \
    "https://github.com/NVIDIA/Cosmos" \
    "Cosmos" \
    --commit "e7ad5e77eecd47acadf17db47d6eb56282a099cc" \
    --update-gitignore

# Editable install without changing shared core pins.
uv pip install --no-deps -e ./Cosmos

# Additional official deps installed or required for imports:
uv pip install accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: installed official deps in current env
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: none
```

## Weight Setup

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "nvidia/Cosmos-Predict2.5-2B" \
    "official_weights/cosmos_predict" \
    --revision "main"
```

If weights are local-only, record the local path and do not copy large files into
the repository.

## Prototype And Conversion Artifacts

State-dict key/shape dumps are generated after FastVideo native prototypes exist
and are used to build the conversion mapping.

```text
official_key_dumps:
  <component>: converted_weights/cosmos_predict/_mapping/<component>_official_keys.json
fastvideo_key_dumps:
  <component>: converted_weights/cosmos_predict/_mapping/<component>_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/cosmos_predict_to_diffusers.py
conversion_source_layout: monolithic
converted_weights_dir: converted_weights/cosmos_predict
strict_load_status: not_run
```

For monolithic official checkpoints, record the component prefix split here. For
example, a single checkpoint may contain transformer, VAE/pretransform,
conditioner, and scheduler/vocoder keys that the conversion script writes into
separate FastVideo component subfolders.

## Expected Parity Tests

Planned local tests for this family:

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| `<component>` | `<definition path; instantiation path + args>` | `tests/local_tests/cosmos_predict/test_cosmos_predict_<component>_parity.py` | `<prototype or setup concerns>` | `planned` |
| `pipeline` | `<official pipeline call>` | `tests/local_tests/pipelines/test_cosmos_predict_pipeline_parity.py` | `<pipeline concerns>` | `planned` |

Include reused components in this table. Reuse is accepted only after the
FastVideo component definition and official instantiation arguments have both
been checked and the component parity test passes non-skip.

Run the relevant tests with:

```bash
pytest tests/local_tests/cosmos_predict/test_cosmos_predict_<component>_parity.py -v -s
pytest tests/local_tests/pipelines/test_cosmos_predict_pipeline_parity.py -v -s
```

## Review Notes

- Required before handoff: non-skip PASS for each required component parity
  test, including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold, but final handoff requires non-skip
  PASS or an explicit blocker accepted through the escape-hatch process.
- User decisions and pause points are tracked as `E###` rows in
  `PORT_STATUS.md`; do not rely on chat history for escape-hatch context.
- Review agents should verify this README's setup commands still match the PR,
  then run the listed parity tests or report the exact blocker.
