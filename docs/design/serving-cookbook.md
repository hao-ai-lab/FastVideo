# Serving cookbook design

Authored option schemas describe the cookbook's selectable fields; native
examples provide recommended values. The cookbook combines them with curated
model/deployment choices, selected controls and optional Markdown guidance.
A Node build prepares static JSON metadata without importing the model runtime;
a shared JavaScript API resolves edits locally and emits YAML and commands.
The temporary preview is a standalone local page to remove when the final UI
replaces it. The static catalogs and reusable JavaScript API are published with
the docs independently of that preview.

## Sources and delivery

| Source | Responsibility |
| --- | --- |
| `examples/serving/*.yaml` | Complete explicit baseline values, including hidden settings |
| `docs/cookbook/options.yaml` | Authored common schemas, runtime metadata, workloads and runtime option overrides |
| `docs/cookbook/recipes/<model-key>.yaml` | Model title and deployments: runtime, workload, baseline, controls, schema overrides, environment, requirements and optional guide |
| `docs/build-cookbook-config.mjs` | Validate authored metadata, merge schema overrides, expand selected paths and publish static catalogs |
| Existing serving examples and configuration documentation | Author review references for native behavior; no docs-build imports |
| `docs/assets/cookbook-config.js` | Load metadata, expose option values, resolve YAML and commands |
| `examples/cookbook/cookbook-demo.js` | Temporary form, download buttons and guide links |

See the [contributor guide](../contributing/cookbook_configuration.md) for manifest
syntax. Every deployment in one model file references the same native model ID.
Only summary and guide fall back from model to deployment; no controls or config
inheritance is performed. The UI offers authored deployments, not a cross-product
of models, hardware and backends.

The exporter writes ignored build output:

```text
docs/assets/cookbook-config/index.json
docs/assets/cookbook-config/recipes/<model-key>/<deployment-key>.json
```

The normal GitHub docs job installs `requirements-mkdocs.txt` and runs
`npm ci --prefix docs`, cookbook tests, `npm run build:catalog --prefix docs`,
and `mkdocs build`. Invalid metadata stops the build. No FastVideo configuration
imports, runtime packages or separate metadata job are required. MkDocs includes
the Git-ignored JSON in the published static site, while source-only
`options.yaml` and recipe manifests remain excluded. No runtime server is involved.

For local assets, run `npm ci --prefix docs` and
`npm run build:catalog --prefix docs` before building the docs or opening the
standalone demo. `mkdocs build` and `mkdocs serve` do not regenerate catalogs;
rerun the Node build when shared options, manifests or baseline YAML change.

The small index contains no configurations or schemas:

```text
models: [{
  id: model-key, title, model_id: native model identity,
  deployments: [{id: model-key/deployment-key, label, runtime, workload, catalog_url}]
}]
```

Each deployment catalog contains its identity, model, runtime metadata, workload,
source YAML path, environment, requirements, optional compiled guide URL, raw
`base_config` and `controls: [{path, schema}]`. Controls are already expanded to
flat leaf paths. Model files are ordered by filename; deployment and selector
order follow the manifest. The browser fetches only the selected catalog.
Relative catalog, guide and runtime client-guide URLs resolve against the index
URL. `loadDeployment` returns catalog metadata without rewriting those links.

## Authored schemas and composition

`options.yaml` has a root `options` map from flat canonical paths to
self-contained JSON Schemas. Its root `runtimes` map associates each runtime ID
with `metadata`, declared `workloads`, and an `options` map of schema overrides
or additions. A deployment's `overrides` map uses the same canonical paths.
Schema precedence is common options, then runtime options, then deployment
overrides.

Objects, including nested schema properties, merge recursively; arrays replace
whole arrays. Omitted keys inherit, and explicit `false`, `0` and allowed `null`
remain explicit. The merge is Hydra-like; there is no Hydra dependency,
interpolation, defaults list or sweep behavior. Existing shared paths retain
their type identity. The builder rejects inconsistent constraints, invalid enum
values and defaults, and schema `$ref` references. New model-local paths need
complete schemas in `overrides` and an explicit selection in `controls`.

The ordered `controls` list is independent of schema composition: an override
alone never enables an option. An exact path selects one declared option;
a namespace expands prefix-matching catalog entries in declaration order.
It does not expand properties inside one option schema. Duplicate or overlapping
selections and unknown paths fail. Namespace selection also includes future
matching options; exact paths keep the selected surface stable.

Types, ranges and annotations are an author-maintained contract. The Node build
checks that contract's internal consistency, without querying the runtime
registry or performing model admission checks. Authors must review native
definitions when adding or changing options. JSON Schema `default` remains an
annotation: schema merging never changes the baseline or inserts missing YAML
values.

Compared with [PR #1912](https://github.com/hao-ai-lab/FastVideo/pull/1912), this
alternative keeps the same UI, JavaScript API, native baselines and lazy
per-deployment JSON contract. The metadata authority changes from exported
Python/Pydantic definitions to authored YAML schemas, so docs builds need only
Node and MkDocs dependencies. That removes runtime imports from publication but
makes schema drift an explicit maintenance responsibility, checked separately
by reviewing existing serving examples and configuration documentation.

## Shared JavaScript API

The framework-independent core exposes four functions through
`FastVideoConfigCookbook` in the browser and CommonJS in Node:

| Function | Result |
| --- | --- |
| `loadIndex(url, {signal, fetcher} = {})` | `{models, url}`; URL context for resolving catalog links |
| `loadDeployment(indexContext, deploymentId, {signal, fetcher} = {})` | The selected complete catalog |
| `getOptions(catalog, config = catalog.base_config, edits = {})` | `[{path, schema, value}]` for the declared controls, with GPU-dependent degree limits |
| `resolveConfig(catalog, edits = {})` | `config`, `yaml`, `command`, `clientRequest`, `clientCommand`, and optional `websocketUrl` |

`fetcher` is optional dependency injection for tests; `signal` supports canceling
obsolete requests. `getOptions` returns `undefined` for an inherited value;
`null`, `false` and `0` remain explicit. `schema.default` is informational and
must not populate missing values. Edits map expanded field paths to new values.
The optional `edits` argument previews current values and dependent limits even
while a configuration is incomplete or invalid. `resolveConfig` remains the
validation boundary before displaying or downloading output.

For direct browser use, load the bundled validator before resolving edits:

```html
<script src="/assets/cookbook-validator.js"></script>
<script src="/assets/cookbook-config.js"></script>
```

Adjust asset URLs for the site's deployment prefix. Loading metadata and reading
options do not require the validator; `resolveConfig` does. The standalone demo HTML
loads these dependencies in order. CommonJS consumers load the bundled validator
through the core module's internal `require`.

```javascript
const api = FastVideoConfigCookbook;
const index = await api.loadIndex(indexUrl);
const catalog = await api.loadDeployment(index, "fastwan21/cuda-rest");
const fields = api.getOptions(catalog);
const result = api.resolveConfig(catalog, {"server.port": 9000});
// Render fields and result.yaml/result.command in the chosen UI framework.
```

The core does not mount UI, manipulate guide HTML or execute commands.
`examples/cookbook/index.html` loads the validator, core and demo scripts and
mounts the replaceable demo directly; it needs no MkDocs build or global loader.
The demo owns Model/Deployment selectors, cancellation and stale-response checks, error
states, copying/downloads and ordinary guide links. Loading a new catalog disables
old outputs and resets edits. Only active catalog/validator state is retained.

## Configuration invariants

```text
Displayed values = baseline + edits; missing fields show Inherited
Saved YAML       = complete explicit baseline + edits
Reset            = selected deployment's baseline
```

Validation must not replace the raw baseline with a default-filled model dump.
Hidden values survive; formatting/comments need not. Exact-path edits preserve
siblings; arrays replace whole values. Missing and explicit null differ. The
runtime still resolves checkpoint-dependent, hardware-dependent and request
fallback values; the cookbook does not simulate those decisions.

Browser validation checks selected fields against their exported constraints and
validates GPU count against the configured TP, SP and HSDP degrees. Positive
degrees must be at most the GPU count and divide it evenly; the native automatic
`-1` values remain available. `getOptions` returns GPU-dependent numeric limits
for future UIs, while the shared resolver rejects invalid combinations without
changing user values. It does not reproduce arbitrary model/FSDP validators or
hardware compatibility. Reset discards edits by resolving the baseline again.

Entries in `experimental` and other untyped maps can become controls only with
explicitly authored schemas and control selection. Existing hidden values still
survive every download. No implicit controls, add/hide rules or model-specific
JavaScript are introduced.

## Runtime boundaries and examples

| Runtime | Native contract and launch |
| --- | --- |
| `fastvideo-cuda-rest` | `ServeConfig` + serving translation; `fastvideo serve --config config.yaml`; rejects active streaming |
| `fastvideo-mlx-rest` | `MLXServeConfig`; `python -m fastvideo.entrypoints.openai.mlx_server --config config.yaml` |
| `fastvideo-cuda-streaming` | `ServeConfig` + generator translation, requires active streaming; same `fastvideo serve` command |

The four models demonstrate five deployments: FastWan CUDA T2V, FastH3 CUDA and
MLX T2V, Wan2.1 CUDA I2V, and LTX2 CUDA streaming. This is not a complete capability
matrix. MLX's curated options follow its native fields; its untyped
request-default map stays hidden. All catalog generation runs without importing
runtime modules. The table describes native contracts, not build-time validation.

REST client examples use effective host, port and model alias; I2V includes an
image reference. Wildcard hosts become loopback client addresses. Streaming
instead emits a health/liveness command, WebSocket URL and the
[protocol guide](server_contracts/streaming.md); it does not fake a REST client.
The baseline retains actual GPU topology and deployment requirements.

The demo reuses native serving examples and existing runbooks. It does not add
model runbooks or establish which hardware/option combinations have been tested.
Manifest setup notes are copied from those sources; optional guide links are
omitted where there is no suitable existing page.

## Guides and checks

Catalogs carry optional guide URLs, not Markdown. The local demo resolves these
links against the existing published documentation, without fetching or
transforming guide HTML. Guide text remains baseline reference material;
generated instructions reflect edits.

Catalog generation and browser checks need only Node. They cover authored schema
validation and composition, baseline preservation, explicit/inherited values,
environment quoting, public API behavior, launch commands and selection races.
These authored-schema and browser checks do not establish native runtime
validity, server startup or GPU compatibility. Authors review existing serving
examples and configuration documentation to keep option definitions aligned.
No weights or GPU inference are required. A new runtime
needs metadata, option schemas and tests, plus JavaScript support if it adds a
launch protocol; ordinary deployment additions remain data-only.
