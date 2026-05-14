# <Model Family> Local Tests

Local-only parity and smoke tests for the `<model_family>` FastVideo port. These
tests compare FastVideo against the official reference implementation and are
not expected to run in CI unless explicitly promoted later.

Port progress, open questions, issues, and handoff notes live in
`tests/local_tests/<model_family>/PORT_STATUS.md`.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `<model_family>` |
| Workload types | `<T2V/I2V/V2V/T2I/or compatibility shim with rationale>` |
| Official reference | `<url or import path>` |
| Local reference dir | `<ReferenceDir or none>` |
| Official commit/version | `<sha, tag, package version, or unknown>` |
| HF weights | `<HF repo id/url or local path>` |
| HF revision | `<revision or default>` |
| Local weights dir | `<official_weights/model_family or local path>` |
| Source layout | `<diffusers/raw_official/monolithic/separate_components/mixed/custom/unknown>` |
| Needs conversion | `<yes/no/unknown>` |

Do not write token values in this file. Use only the token env var name:
`<HF_TOKEN or HUGGINGFACE_HUB_TOKEN or HF_API_KEY>`.

## Shared Environment Setup

Run from the FastVideo repo root in the same conda/env used for FastVideo.
Do not create a separate upstream environment for parity tests.

```bash
# Official reference source, if cloneable.
python ".agents/skills/add-model-01-prep/scripts/clone_reference_repo.py" \
    "<official_repo_url>" \
    "<ReferenceDir>" \
    --commit "<commit-sha>" \
    --update-gitignore

# Editable install without changing shared core pins.
uv pip install --no-deps -e ./<ReferenceDir>

# Additional official deps installed or required for imports:
# <package list or none>
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: <none | installed no-deps editable | installed official deps in current env | blocked on user>
official_env_status: <imports_ok | private_deps_need_stubs | blocked>
private_dep_stubs: <none or tests/local_tests/helpers/<model_family>_upstream.py>
blocked_on: <none or exact blocker>
```

## Weight Setup

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "<Org/Model>" \
    "official_weights/<model_family>" \
    --revision "<revision>"
```

If weights are local-only, record the local path and do not copy large files into
the repository.

## Prototype And Conversion Artifacts

State-dict key/shape dumps are generated after FastVideo native prototypes exist
and are used to build the conversion mapping.

```text
official_key_dumps:
  <component>: converted_weights/<model_family>/_mapping/<component>_official_keys.json
fastvideo_key_dumps:
  <component>: converted_weights/<model_family>/_mapping/<component>_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/<model_family>_to_diffusers.py
conversion_source_layout: <diffusers | separate_components | monolithic | mixed | custom>
converted_weights_dir: converted_weights/<model_family>
strict_load_status: <not_run | pass | pass_with_documented_exclusions | blocked>
```

For monolithic official checkpoints, record the component prefix split here. For
example, a single checkpoint may contain transformer, VAE/pretransform,
conditioner, and scheduler/vocoder keys that the conversion script writes into
separate FastVideo component subfolders.

## Expected Parity Tests

Planned local tests for this family:

| Component | Official files / args | Test | Concerns | Status |
|---|---|---|---|---|
| `<component>` | `<definition path; instantiation path + args>` | `tests/local_tests/<bucket>/test_<model_family>_<component>_parity.py` | `<prototype or setup concerns>` | `<planned/scaffold_skip/debug_red/non_skip_pass/blocked>` |
| `pipeline` | `<official pipeline call>` | `tests/local_tests/pipelines/test_<model_family>_pipeline_parity.py` | `<pipeline concerns>` | `<planned/scaffold_skip/debug_red/non_skip_pass/blocked>` |

Include reused components in this table. Reuse is accepted only after the
FastVideo component definition and official instantiation arguments have both
been checked and the component parity test passes non-skip.

Run the relevant tests with:

```bash
pytest tests/local_tests/<bucket>/test_<model_family>_<component>_parity.py -v -s
pytest tests/local_tests/pipelines/test_<model_family>_pipeline_parity.py -v -s
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
