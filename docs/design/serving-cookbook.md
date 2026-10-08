# Serving cookbook design

Existing configuration schemas describe accepted fields; native examples provide
recommended values. The cookbook combines them with curated model/deployment
choices, selected controls and optional Markdown guidance. It remains a static
site: Python prepares metadata at build time; a shared JavaScript API resolves
edits locally and emits YAML and commands.

## Sources and delivery

| Source | Responsibility |
| --- | --- |
| `examples/serving/*.yaml` | Complete explicit baseline values, including hidden settings |
| `docs/cookbook/recipes/<model-key>.yaml` | Model title and deployments: runtime, workload, baseline, controls, environment, requirements and optional guide |
| Runtime Python/Pydantic definitions | Public field metadata and native configuration validation |
| `docs/cookbook_config.py` | Validate, expand selected fields and publish static catalogs |
| `docs/assets/cookbook-config.js` | Load metadata, expose option values, resolve YAML and commands |
| `docs/assets/cookbook-demo.js` | Temporary form, download buttons and guide links |

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

## Shared JavaScript API

The framework-independent core exposes four functions through
`FastVideoConfigCookbook` in the browser and CommonJS in Node:

| Function | Result |
| --- | --- |
| `loadIndex(url, {signal, fetcher} = {})` | `{models, url}`; URL context for resolving catalog links |
| `loadDeployment(indexContext, deploymentId, {signal, fetcher} = {})` | The selected complete catalog |
| `getOptions(catalog, config = catalog.base_config)` | `[{path, schema, value}]` for the declared controls |
| `resolveConfig(catalog, edits = {})` | `config`, `yaml`, `command`, `clientRequest`, `clientCommand`, and optional `websocketUrl` |

`fetcher` is optional dependency injection for tests; `signal` supports canceling
obsolete requests. `getOptions` returns `undefined` for an inherited value;
`null`, `false` and `0` remain explicit. `schema.default` is informational and
must not populate missing values. Edits map expanded field paths to new values.

For direct browser use, load the bundled validator before resolving edits:

```html
<script src="/assets/cookbook-validator.js"></script>
<script src="/assets/cookbook-config.js"></script>
```

Adjust asset URLs for the site's deployment prefix. Loading metadata and reading
options do not require the validator; `resolveConfig` does. The demo bootstrap
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

The core does not mount UI, manipulate guide HTML or execute commands. A small
page-only bootstrap loads dependencies and mounts the replaceable demo. The demo
owns Model/Deployment selectors, cancellation and stale-response checks, error
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

Browser validation checks the selected fields against their exported constraints.
It does not reproduce arbitrary Python validators, cross-field runtime checks or
hardware compatibility. Reset discards edits by resolving the baseline again.

Selectors are positive leaves or non-nullable declared namespaces. Expansion
follows declaration order and rejects any protected/opaque/unsupported child,
duplicate or overlap. Arrays are leaves; maps and nullable objects are not
expandable. Namespace selection opts into future public fields. No implicit
controls, add/hide rules or model-specific JavaScript are introduced.

## Runtime boundaries and examples

| Runtime | Native validation and launch |
| --- | --- |
| `fastvideo-cuda-rest` | `ServeConfig` + serving translation; `fastvideo serve --config config.yaml`; rejects active streaming |
| `fastvideo-mlx-rest` | `MLXServeConfig`; `python -m fastvideo.entrypoints.openai.mlx_server --config config.yaml` |
| `fastvideo-cuda-streaming` | `ServeConfig` + generator translation, requires active streaming; same `fastvideo serve` command |

The four models demonstrate five deployments: FastWan CUDA T2V, FastH3 CUDA and
MLX T2V, Wan2.1 CUDA I2V, and LTX2 CUDA streaming. This is not a complete capability
matrix. MLX uses its native typed fields; its untyped request-default map stays
hidden. Streaming exports metadata without importing execution modules.

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

MkDocs renders optional guide pages. Catalogs carry their URLs, not Markdown.
The demo links to the guide without fetching or transforming its HTML. Guide
text remains baseline reference material; generated instructions reflect edits.

CI generates catalogs before MkDocs and checks catalog/guide URLs in the built
site. Focused Python/Node tests cover native validation, baseline preservation,
explicit/inherited values, environment quoting, public API behavior, runtime
commands and selection races. Fixture-based JS unit tests need only Node;
generated-catalog and native-parser integration tests use the CPU Python
environment and generate temporary assets. No weights, server startup or GPU
inference are required. A new runtime needs one shared
adapter and tests; ordinary deployment additions remain data-only.
