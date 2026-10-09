# Serving cookbook design

The cookbook publishes deployment catalogs and a reusable JavaScript API so
different UIs can share the same configuration logic. Native serving examples
provide the starting values; authored JSON Schemas describe editable fields.
The browser applies explicit edits and generates YAML and launch instructions.

Recipe syntax, schema overrides, supported runtimes, build commands and tests
belong in the [contributor guide](../contributing/cookbook_configuration.md).
This page defines the data flow, public API and configuration guarantees.

## Sources and delivery

```text
Native serving YAML + option schemas + deployment manifests
                         |
              build-cookbook-config.mjs
                         |
               Generated JSON catalogs
                         |
                 MkDocs / static hosting
                         |
            cookbook-config.js + UI edits
                         |
                 YAML and command strings
```

| Source | Responsibility |
| --- | --- |
| `examples/serving/*.yaml` | Complete explicit baseline, including hidden settings |
| `docs/cookbook/options.yaml` | Authored field schemas and runtime metadata |
| `docs/cookbook/recipes/*.yaml` | Deployments, baseline references, selected fields and schema overrides |
| `docs/build-cookbook-config.mjs` | Validate metadata and export static catalogs |
| `docs/assets/cookbook-config.js` | Load catalogs, preview fields, validate edits and produce output |

Catalog generation runs in Node without importing FastVideo or loading model
weights. The docs build generates assets before MkDocs copies them into the
published site. Generated JSON, the validator bundle and its license notices
are Git-ignored build output; the validator and notices are published together.
Website visitors need only the static assets. The schemas are author-maintained
and must stay aligned with native runtime behavior.

## Catalog contract

The builder writes a small index and one catalog per deployment:

```text
docs/assets/cookbook-config/index.json
docs/assets/cookbook-config/recipes/<model-key>/<deployment-key>.json
```

The index has no configurations or field schemas:

```text
models: [{
  id, title, model_id,
  deployments: [{id, label, workload, runtime, catalog_url}]
}]
```

Each deployment catalog contains `id`, `title`, `summary`, `model`, `deployment`,
`workload`, `runtime`, `source_config`, `env`, `requirements`, optional `guide`,
the complete `base_config`, and ordered `controls: [{path, schema}]`.
Controls use canonical paths such as `server.port`; their schemas are JSON
Schema fragments that a UI can render using its own widgets.

The browser fetches only the selected deployment. `loadDeployment()` resolves
relative catalog URLs against `index.url` but returns stored metadata links
without rewriting them. The UI resolves `guide.url` and
`runtime.client_guide_url` against the published index URL. Guides are links
to documentation, not embedded HTML.

## Shared JavaScript API

`docs/assets/cookbook-config.js` exposes these four functions through the browser
global `FastVideoConfigCookbook` and CommonJS in Node:

| Function | Result |
| --- | --- |
| `loadIndex(url, {signal, fetcher} = {})` | Promise of `{models, url}`; URL context for catalog links |
| `loadDeployment(index, id, {signal, fetcher} = {})` | Promise of the selected complete catalog |
| `getOptions(catalog, config = catalog.base_config, edits = {})` | `[{path, schema, value}]`, including GPU-dependent degree limits |
| `resolveConfig(catalog, edits = {})` | `config`, `yaml`, `argv`, `command`, `clientRequest`, `clientCommand`, optional `websocketUrl` |

For WebSocket deployments, `clientRequest` is `null` and `websocketUrl` is
present. REST deployments return a request object and omit `websocketUrl`.

`signal` supports canceling obsolete loads; `fetcher` accepts a fetch-compatible
replacement for tests or other callers. Loading failures reject the promise.
Validation failures throw an `Error` with a message; there is no structured
field-error or error-code contract.

Load the validator before resolving configuration in the browser. For this
site's deployment prefix:

```html
<script src="/FastVideo/assets/cookbook-validator.js"></script>
<script src="/FastVideo/assets/cookbook-config.js"></script>
```

```javascript
const api = FastVideoConfigCookbook;
const index = await api.loadIndex("/FastVideo/assets/cookbook-config/index.json");
const catalog = await api.loadDeployment(index, "fastwan21/cuda-rest");
const edits = {"server.port": 9000};
const fields = api.getOptions(catalog, catalog.base_config, edits);
const result = api.resolveConfig(catalog, edits);
// Render fields; display result.yaml/result.command and offer a YAML download.
```

Adjust asset URLs for other deployment prefixes. Loading catalogs and previewing
fields do not need the validator. `getOptions()` previews pending edits without
full schema/topology validation; `resolveConfig()` validates before generating
output. Edits are a flat path-to-value object with correctly typed JavaScript
values, and only declared controls are editable.

The core returns data and strings. The UI owns layout, widgets, edit state,
input parsing, loading/error states, cancellation, stale-response handling,
guide links, copying and downloads. Catch failures and clear invalid output.
The replaceable adapter in `examples/cookbook/cookbook-demo.js` demonstrates
these responsibilities without coupling them to the shared API.

## Configuration invariants

```text
Displayed values = baseline + edits; missing fields show Inherited
Saved YAML       = complete explicit baseline + edits
Reset            = resolveConfig(catalog, {})
```

- `schema.default` is informational. A field value of `undefined` remains
  omitted/inherited; validation never inserts schema defaults.
- Explicit `false`, `0` and `null` retain their meanings. Missing and null differ.
- Resolution works on a copy. Hidden settings and sibling fields survive edits;
  arrays replace whole values. YAML formatting and comments need not survive.
- Removing an edit restores baseline behavior; it does not delete a baseline
  setting. Changing deployments requires the UI to reset its edit state.
- Field validation uses authored schemas. Topology checks enforce positive GPU
  counts and parallel degrees that divide the count, with supported `-1`
  automatic values. Invalid combinations are rejected without changing edits.

The native runtime still resolves model-, hardware- and request-dependent
defaults. Cookbook validation does not establish complete native validity,
server readiness or GPU memory fit. Client instructions use same-machine
addresses: REST deployments return a sample request command, while streaming
deployments return a health command and WebSocket URL. The UI displays these
instructions; the API does not execute them.
