# Serving cookbook design

The cookbook helps a user choose a model, select one maintained deployment,
adjust reviewed settings, and download its native serving configuration. It is
not an automatic form for every runtime field or a matrix of arbitrary model,
hardware and backend combinations.

The model picker is followed by a deployment picker. FastWan currently offers
CUDA T2V REST; FastH3 8-Step V2 offers CUDA REST and native MLX REST. Wan2.1 I2V
uses its maintained two-GPU REST baseline, and LTX2 Distilled uses its native
CUDA streaming baseline. These are explicit deployment examples, not a complete
capability matrix. H3 is registered for I2V, but no maintained H3 I2V serving
YAML is presented here; FastWan's example is T2V. Choosing MLX for FastH3 does
not imply that FastWan supports it. The UI is a temporary demo;
the model manifests, runtime adapters, exporter and shared resolver are reusable.

## Ownership and authoring

| Information | Authoritative source |
| --- | --- |
| Explicit recommended settings | Native `examples/serving/*.yaml` baseline for each deployment |
| Model identity | `generator.model_path` in the native baselines |
| Model title and available deployments | `docs/cookbook/recipes/<model-key>.yaml` |
| Runtime, workload, controls, environment and requirements | Each deployment entry in that manifest |
| Field metadata and config validation | The selected runtime's public configuration contract |
| GPU specifications | Shared `docs/cookbook/hardware.yaml` inventory |
| Hardware status and evidence | Explicit hardware records in a deployment |
| Optional explanatory instructions | Markdown page referenced by `guide` |
| Widgets, labels, grouping and command formatting | Shared cookbook code |
| Static delivery data | Generated, gitignored `docs/assets/cookbook-config/` |

Adding a manifest publishes a model; adding an entry under `deployments`
publishes another supported way to serve that same model. There is no separate
model-selection list. All baselines in one manifest must identify the same
model. Configuration values are not duplicated in the manifest.

```yaml
# docs/cookbook/recipes/fastwan21.yaml
title: FastWan2.1 1.3B
summary: Distilled text-to-video serving with sparse attention.
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
      - generator.engine.offload.dit_layerwise
      - generator.engine.offload.text_encoder
      - generator.engine.offload.vae
      - generator.engine.compile.enabled
      - default_request.sampling.num_frames
      - default_request.sampling.height
      - default_request.sampling.width
      - default_request.sampling.fps
      - default_request.sampling.seed
```

Model-level keys are `title`, optional `summary` and `guide`, and a nonempty
`deployments` mapping. Every deployment requires `runtime`, `workload`, `config`
and `controls`; optional keys are `label`, `env`, `requirements`, `hardware`,
`summary` and `guide`. A deployment label defaults to its runtime's label;
provide one to distinguish multiple setups using the same runtime. Unknown
keys are errors. Only summary and guide fall back from
model to deployment. Controls, environment and requirements are never inherited
or merged across deployments.

Model filenames and deployment keys are stable lowercase hyphenated names.
The generated deployment ID is `<model-key>/<deployment-key>`. Config paths are
repository-relative references to native serving YAML files. Runtime and
workload are explicit: a recipe demonstrates one workload, not every workload
the model might support.

## Controls and baseline preservation

Each positive control selector names a typed editable leaf or a non-nullable
declared object namespace. A namespace expands to all descendant editable
fields in declaration order. Arrays remain leaves. Manifest order determines
order between selectors. If a descendant is protected, opaque or unsupported,
generation fails with its path rather than silently filtering it. Unknown
paths, duplicate selectors and overlaps such as `server` plus `server.port`
are errors. Nullable objects and free-form maps cannot be expanded.

There are no implicit controls, `add`, `hide` or per-model JavaScript branches.
A namespace opts into future public fields beneath it; an unsupported new child
fails generation for review. Exact leaves provide a stable surface. Native
availability does not establish that a field is useful for every deployment.
For example, FastWan's fixed DMD schedule makes the whole sampling namespace
an inappropriate selector.

```text
Displayed values = baseline + user edits; absent fields show Inherited
Saved YAML       = baseline + user edits
Reset            = baseline for the selected deployment
```

Keep the complete raw baseline mapping. Parsing validates it but must not
replace it with a dataclass/model dump populated with defaults. Hidden values,
including experimental settings, timesteps and negative prompts, survive.
Comments and formatting need not survive. Changing a deployment loads its own
baseline and discards the previous deployment's edits.

Schema defaults are declarations, not a simulation of checkpoint, hardware or
HTTP request resolution. An absent field stays omitted until edited. Explicit
`false`, `0`, empty strings and `null` remain values. Removing an override and
assigning null are different operations. Exact-path edits preserve siblings;
arrays replace as whole values. GPU topology stays fixed where its coupled
changes are not supported by the builder.

## Runtime adapters

A small shared adapter supplies configuration validation, selected-field
metadata, installation guidance and launch arguments. Runtime metadata also
identifies its backend and interface. The initial adapters are:

| Runtime | Configuration source | Launch |
| --- | --- | --- |
| `fastvideo-cuda-rest` | `ServeConfig`, native parser and serving compatibility adapter | `fastvideo serve --config config.yaml` |
| `fastvideo-mlx-rest` | `MLXServeConfig` from the native MLX server | `python -m fastvideo.entrypoints.openai.mlx_server --config config.yaml` |
| `fastvideo-cuda-streaming` | `ServeConfig` with an active `streaming` block, native parser and generator translation | `fastvideo serve --config config.yaml` |

CUDA exposes reviewed typed public fields. Experimental map entries remain
hidden and preserved. CUDA REST rejects an active streaming block and fields
that its compatibility adapter rejects.

The streaming adapter has backend `cuda` and interface `websocket`. It requires
an active streaming block, which makes the native CLI select its streaming
server. The LTX2 example references `examples/serving/streaming_demo.yaml` and
exposes only host, port, frames, height and width. Its NVFP4 baseline requires
compatible compute capability >= 10.0, `flashinfer-python`, FFmpeg with `libx264`
and FastVideo's `streaming` extra. Preserve its hidden warmup, prompt, safety
and pool settings, but do not imply that the bare serving entrypoint activates
those DreamVerse integrations or requires `CEREBRAS_API_KEY`.

Streaming metadata validation uses configuration parsing and generator
translation without importing the streaming execution modules. Client guidance
shows the health URL, `ws://<host>:<port>/v1/stream` and the
[streaming protocol contract](server_contracts/streaming.md). It does not
invent a REST video request or add a standalone streaming client.

MLX currently supports the FastH3 models declared by its native config.
Controls can expose its typed server fields and generator paths such as
`model_root`, `mlx_checkpoint`, `prompt_cache_dir`, `vae_dtype` and
`vsa_sparsity`. MLX's `default_request` is an untyped mapping: preserve it but
do not borrow types or constraints from an HTTP request schema to expose it.
The supplied baseline retains its required request settings. Weight conversion,
local paths and machine requirements are explained in installation guidance
and deployment requirements, not performed by the exporter.

A deployment for an existing adapter requires native YAML and manifest data,
not new frontend code. A genuinely new serving interface requires one shared
adapter and tests; adding a manifest alone cannot implement a new runtime.

## Hardware and optional guides

The shared hardware inventory stores exact GPU facts and their source URLs.
It is not automatically displayed for every model or deployment. A deployment
explicitly lists the inventory IDs relevant to its hardware table. An omitted
hardware map produces no rows and establishes no serving evidence. A listed
row is `unverified` unless evidence or a known incompatibility is declared.

`verified` requires an HTTPS evidence link to a successful serving run of that
baseline. `unsupported` requires a concrete reason. Evidence should record SKU,
count, configuration, code/software versions, environment, workload and successful
server output. Rated memory does not establish fit, and a direct inference
result is not REST or WebSocket serving evidence. Configuration edits are custom and
unverified; baseline evidence remains labeled with its original scope.
Hardware records never change configuration values. There is no hardware
selector or automatic VRAM estimate in this implementation.

A `guide` points to an existing Markdown file under `docs/`. Model-level guides
are useful for shared background; deployments can choose more specific pages.
MkDocs renders them normally. Generated metadata carries a compiled relative
URL, not Markdown contents or a browser Markdown renderer.

## Generated output and browser behavior

The docs build reads model manifests in filename order and deployments in
mapping order. It validates their baselines and writes:

```text
docs/assets/cookbook-config/index.json
docs/assets/cookbook-config/recipes/<model-key>/<deployment-key>.json
```

The index is grouped by model:

```text
models: [{
  id: model-key, title, model_id: native model identity,
  deployments: [{id: model-key/deployment-key, label, runtime, workload, catalog_url}]
}]
```

Each complete catalog contains the selected deployment's identity, source
configuration, model identity, runtime metadata, workload, environment,
requirements, hardware records, guide URL, raw `base_config` and a flat expanded
`controls: [{path, schema}]` list. The browser performs no manifest inheritance
or namespace expansion.

The browser fetches the index, shows Model then Deployment, and fetches only
the selected catalog. It offers declared deployments, not a backend/workload
cross-product. Loading or errors disable stale output; late responses cannot
replace the current choice. Only the active catalog and validator are retained.
The same pure resolver serves browser interactions and Node tests. Ajv checks
values without coercing types or inserting defaults.

The page presents runtime-specific installation guidance, requirements, source
YAML, controls, hardware evidence when provided, optional guide, complete YAML,
launch command and runtime-specific client guidance. Environment values are safely quoted.
REST samples use the effective port and model alias and distinguish T2V from
I2V; the Wan2.1 I2V request includes a required `input_reference` placeholder.
Streaming instead shows a health/liveness-check command, its WebSocket URL and protocol
guide. Both map wildcard bind addresses to loopback client addresses. They
describe a local client workflow, not network reachability or successful model
generation. Copying commands executes nothing.

## Validation and extension boundaries

CI checks manifest/reference consistency, same-model baselines, runtime config
validation, supported controls, deterministic delivery and compiled URLs. Tests
cover complete baseline preservation, edits/reset, explicit versus inherited
values, environment quoting, runtime/workload commands, nested model/deployment
selection and failed or out-of-order fetches. Generated files are temporary test
fixtures or ignored build output, not hand-maintained source files.

Generation uses CPU configuration dependencies without loading weights,
creating generators or running GPU/MLX inference. Structural checks do not
certify model output, hardware fit or an actual serving run. Docs CI generates
the assets before MkDocs builds and checks their URLs before deployment.

Add adapters only for maintained native serving interfaces. Avoid a generic
command-template language, deployment inheritance system or model-specific UI.
Memory estimation and automatic evidence matching remain separate future work.
See [Adding a model deployment](../contributing/cookbook_configuration.md) for
the contributor workflow.
