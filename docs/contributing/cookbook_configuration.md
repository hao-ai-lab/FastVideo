# Adding a model deployment

The cookbook combines authored JSON Schemas in `docs/cookbook/options.yaml`, a
native serving YAML as the recommended configuration, and a small model manifest
selecting the options to expose. The Node builder generates static catalogs;
the shared JavaScript API generates YAML and launch instructions. Ordinary model
additions require no custom JavaScript or runtime imports.

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
deployments:
  cuda-rest:
    runtime: fastvideo-cuda-rest
    workload: t2v
    config: examples/serving/openai_fastwan21_1_3b.yaml
    env:
      FASTVIDEO_ATTENTION_BACKEND: VIDEO_SPARSE_ATTN
    controls:
      - server
      - generator.engine.num_gpus
      - generator.engine.parallelism
      - generator.engine.compile.enabled
      - default_request.sampling.num_frames
```

| Level | Required | Optional |
| --- | --- | --- |
| Model | `title`, nonempty `deployments` | `summary`, `guide` |
| Deployment | `runtime`, `workload`, `config`, `controls` | `label`, `summary`, `guide`, `env`, `requirements`, `overrides` |

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

`workload` is one authored `t2v` or `i2v` route listed for the runtime in
`options.yaml`. These examples are not every model/backend combination. A new
runtime needs shared runtime metadata, option definitions and tests; a new
launch protocol also needs support in the JavaScript API.

Controls are an explicit ordered list; `[]` is valid. Select a declared option
path or a namespace prefix such as `server`. A namespace expands matching
catalog entries in their declaration order; it does not expand `properties`
inside an option's schema. Unknown paths, duplicates and overlaps such as
`server` plus `server.port` fail rather than silently disappearing. There is no
implicit list, `add` or `hide`. Namespaces opt into future matching options;
use exact paths for a stable surface.

Types and constraints are curated by the option author, who must keep them
aligned with native configuration definitions and supported usage. The build
does not import Python schemas or perform registry/admission checks. Existence
in an authored schema does not prove runtime support. GPU count can be selected
with `generator.engine.num_gpus`; `generator.engine.parallelism` expands TP, SP,
HSDP dimensions and the distributed timeout into ordinary controls. The shared
JavaScript API limits parallel degrees to the selected GPU count and checks
divisibility. TP, SP and HSDP shard size retain the native `-1` automatic value.
Invalid combinations block output until corrected; the editor does not silently
change the recipe's topology or FSDP policy. Model-specific and other native
cross-field checks still apply when the server starts.

Keep coupled sampling choices fixed unless reviewed: FastWan's three steps and
DMD schedule belong together. Hidden experimental values stay in the output.
MLX's untyped `default_request` stays hidden rather than borrowing an HTTP schema.

### Shared options and model overrides

The root `options` map in `docs/cookbook/options.yaml` maps canonical paths, such
as `server.port`, to JSON Schemas. The root `runtimes` map gives each runtime its
`metadata`, allowed `workloads`, and an `options` map of schema overrides or
additions. Each option schema is self-contained; `$ref` is unsupported. Each
deployment can provide its own `overrides` map using those same canonical paths.
The order is:

```text
common options -> runtime options -> deployment.overrides
```

Objects, including schema `properties`, merge recursively. Arrays such as `enum`
replace the previous array; omitted keys inherit. Explicit `false`, `0` and
`null` remain explicit where the schema permits them. This is a Hydra-like merge,
with no Hydra dependency, interpolation, defaults list or sweeps.

An override may change annotations or constraints but must retain an existing
shared option's path and type. Keep constraints mutually consistent; enum values
and default annotations must satisfy the resulting schema. To add a model-local
option, give it a complete schema in `overrides` and select its path in
`controls`. Overrides never enable controls by themselves. The ordered controls
list remains the only selection mechanism, including namespace expansion.

Schema `default` is informational. The displayed and saved values always come
from the native baseline plus user edits; missing values remain inherited. An
override of `default` does not insert or replace a YAML value. Keep hidden
baseline settings intact, including experimental entries. An untyped map key
needs an explicitly authored schema before it can become an editable control.

Copy necessary installation or usage notes from the referenced serving example
or existing runbook into `requirements`, with a source comment. Do not add a
new hardware recommendation or claim that every selectable value is tested.

## 3. Add an optional Markdown guide

A `guide` points to a page under `docs/`, shared at model level or overridden
for a deployment. Reuse an existing runbook instead of adding a second copy of
its instructions. Omit the optional guide when no suitable page exists. The
FastH3 example uses [the existing server runbook](../cookbook/openai-api.md):

```yaml
guide: docs/cookbook/openai-api.md
```

MkDocs renders the page, and the demo provides a normal guide link. Guide text
describes the baseline, while generated commands track edits. Streaming
uses its [protocol guide](../design/server_contracts/streaming.md), a health/liveness
command and WebSocket URL, not a REST generation request.

## 4. Generate and validate

Use the Node dependencies from the
[docs setup guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/README.md).
Run from the repository root:

```bash
npm ci --prefix docs
node docs/build-cookbook-config.mjs
python -m http.server 8195 --bind 127.0.0.1
```

Open `http://127.0.0.1:8195/examples/cookbook/` and choose Model, then Deployment. Check the
unchanged download against the full baseline; edit controls and reset, ensuring
hidden settings survive. Inherited means omitted, not a resolved default;
explicit `false`, `0` and allowed `null` stay explicit. Check launch environment,
client port/alias, the I2V image reference and optional guide. Streaming's health
response alone does not establish generation readiness or output.

```bash
npm run test:cookbook --prefix docs
pre-commit run --files docs/cookbook/recipes/fastwan21.yaml
```

Catalog and browser tests run in Node and check the authored schema contract.
They do not establish native runtime validity, server startup or GPU compatibility.
Review the referenced serving example and its configuration documentation when
changing option definitions. Include other edited files in pre-commit and add
Node checks for unusual requirements.

Generated catalogs under `docs/assets/cookbook-config/` are ignored by Git:
never hand-edit or commit them. `mkdocs build` and `mkdocs serve` include generated
catalogs without regenerating them; rerun the Node build after source changes.
The normal `.github/workflows/infra-docs.yml` docs job installs only Node and
MkDocs dependencies, generates catalogs, and builds the static site. Invalid
metadata blocks deployment. `options.yaml` and recipe manifests are source-only
and excluded from the site.

The demo stays local under `examples/cookbook/`, outside MkDocs, until the final
UI replaces it. No weights or GPU inference are needed for these checks. See the
[design](../design/serving-cookbook.md) for the data/API contract.
