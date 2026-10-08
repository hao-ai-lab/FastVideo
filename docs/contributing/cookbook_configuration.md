# Adding a model deployment

The cookbook combines public Python/Pydantic field metadata, a native serving
YAML as the recommended configuration, and a small manifest selecting the
options to expose. One shared builder generates YAML and launch instructions;
ordinary model additions require no custom JavaScript.

## 1. Reference a native baseline

Reuse or add a runnable YAML under `examples/serving/`. Keep recommended values
there, including required settings hidden from the form. All deployments in one
model manifest must have the same `generator.model_path`. Omitted fields stay
inherited; the builder does not insert every runtime default.

Create `docs/cookbook/recipes/<model-key>.yaml`, or add a deployment to an
existing model file. Filenames and deployment keys use lowercase hyphenated
names. Adding the entry publishes it; there is no separate model list.

```yaml
# docs/cookbook/recipes/fastwan21.yaml
title: FastWan2.1 1.3B
guide: docs/cookbook/guides/fastwan21-serving.md
deployments:
  cuda-rest:
    runtime: fastvideo-cuda-rest
    workload: t2v
    config: examples/serving/openai_fastwan21_1_3b.yaml
    env:
      FASTVIDEO_ATTENTION_BACKEND: VIDEO_SPARSE_ATTN
    controls:
      - server
      - generator.engine.compile.enabled
      - default_request.sampling.num_frames
```

| Level | Required | Optional |
| --- | --- | --- |
| Model | `title`, nonempty `deployments` | `summary`, `guide` |
| Deployment | `runtime`, `workload`, `config`, `controls` | `label`, `summary`, `guide`, `env`, `requirements` |

`config` and `guide` are repository-relative paths. `label` defaults to the
runtime label; use it to distinguish deployments using the same runtime.
`env` maps environment names to strings; quote numeric values. `requirements`
is a list of plain-text setup notes or documentation URLs. Unknown keys fail
validation. Only summary and guide fall back from model to deployment; there
is no configuration, control or environment inheritance.

For another supported runtime, add a sibling entry under `deployments` in the
same model file. For example, `fasth3-8step.yaml` can add:

```yaml
mlx-rest:
  runtime: fastvideo-mlx-rest
  workload: t2v
  config: examples/serving/mlx_fasth3_8step.yaml
  controls:
    - server
    - generator.model_root
    - generator.mlx_checkpoint
```

## 2. Choose runtime and controls

| Runtime | Native contract and example |
| --- | --- |
| `fastvideo-cuda-rest` | `ServeConfig` without an active streaming block; FastWan T2V, FastH3 T2V and Wan2.1 I2V |
| `fastvideo-mlx-rest` | `MLXServeConfig`; FastH3 T2V with prepared local checkpoints |
| `fastvideo-cuda-streaming` | `ServeConfig` with an active streaming block; LTX2 Distilled WebSocket serving |

`workload` is one demonstrated `t2v` or `i2v` route accepted by that runtime.
These examples are not every model/backend combination. A genuinely new runtime
needs one shared adapter and tests before contributors can select it.

Controls are an explicit ordered list; `[]` is valid. Select a typed leaf or
non-nullable declared namespace such as `server`. Namespaces expand all children
in declaration order; arrays stay leaves. Unsupported or opaque descendants,
unknown paths, duplicates and overlaps such as `server` plus `server.port` fail
rather than silently disappearing. Nullable objects and maps cannot be expanded.
There is no implicit list, `add` or `hide`. Namespaces opt into future public
children; use exact leaves for a stable surface.

Types and declared constraints come from the selected runtime. Existence in a
schema does not prove every combination is meaningful. Keep coupled topology
or sampling choices fixed unless reviewed: FastWan's three steps and DMD
schedule belong together. Hidden experimental values stay in the output. MLX's
untyped `default_request` stays hidden rather than borrowing an HTTP schema.

Put machine and installation prerequisites in `requirements`. Wan I2V uses two
CUDA GPUs and needs an image reference. The LTX2 NVFP4 baseline needs compute
capability >= 10.0, `flashinfer-python`, FFmpeg with `libx264` and the `streaming`
extra. Its hidden DreamVerse settings do not make the bare server activate
those integrations or require `CEREBRAS_API_KEY`.

## 3. Add an optional Markdown guide

A `guide` points to a page under `docs/`, shared at model level or overridden
for a deployment. Cover prerequisites, launch, client workflow and relevant
troubleshooting; link native YAML instead of copying it. See the
[FastWan guide](../cookbook/guides/fastwan21-serving.md) and
[Wan I2V guide](../cookbook/guides/wan21-i2v-serving.md).

MkDocs renders the page, and the demo provides a normal guide link. Guide text
describes the baseline, while generated commands track edits. Streaming
uses its [protocol guide](../design/server_contracts/streaming.md), a health/liveness
command and WebSocket URL, not a REST generation request.

## 4. Generate and validate

Use the CPU environment from the
[docs setup guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/README.md).
Run from the repository root:

```bash
python docs/cookbook_config.py
mkdocs serve
```

Open `/cookbook/config-builder/` and choose Model, then Deployment. Check the
unchanged download against the full baseline; edit controls and reset, ensuring
hidden settings survive. Inherited means omitted, not a resolved default;
explicit `false`, `0` and allowed `null` stay explicit. Check launch environment,
client port/alias, the I2V image reference and optional guide. Streaming's health
response alone does not establish generation readiness or output.

```bash
python -m pytest tests/local_tests/test_cookbook_config_metadata.py tests/local_tests/test_cookbook_config_roundtrip.py
node --test tests/local_tests/test_cookbook_config.mjs tests/local_tests/test_cookbook_demo.mjs
node --test tests/local_tests/test_cookbook_config_integration.mjs
mkdocs build
python docs/cookbook_config.py --check-site site
pre-commit run --files docs/cookbook/recipes/fastwan21.yaml
```

The first Node command runs fixture-based unit tests without Python. The Python
tests and the integration Node file require the CPU configuration environment.
Browser validation checks selected field constraints, not every native runtime
or cross-field rule. Include other edited files in pre-commit and add checks for unusual
requirements. Generated catalogs under `docs/assets/cookbook-config/` are ignored
build output: regenerate after source changes, never hand-edit or commit them.
Validation loads no weights or GPU inference. See the [design](../design/serving-cookbook.md)
for the data/API contract; this demo does not promise every runtime field is editable.
