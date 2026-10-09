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
| `docs/js/build-cookbook-config.mjs` | Validate metadata and export static catalogs |
| `docs/js/cookbook-config.mjs` | Load catalogs, preview fields, validate edits and produce output |

Catalog generation runs in Node without importing FastVideo or loading model
weights. The docs build generates assets before MkDocs copies them into the
published site. Generated JSON, the combined API/validator bundle and its license notices
are Git-ignored build output published together under `assets/cookbook-config/`.
Website visitors need only the static assets. The schemas are author-maintained
and must stay aligned with native runtime behavior.

## Catalog contract

The builder writes a small index and one catalog per deployment:

```text
docs/assets/cookbook-config/cookbook-config.js
docs/assets/cookbook-config/cookbook-config.LICENSE.txt
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

`docs/js/cookbook-config.mjs` exports the four functions below. The generated
`assets/cookbook-config/cookbook-config.js` bundles them with Ajv and exposes
the browser global `FastVideoConfigCookbook` and CommonJS exports:

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
Loading checks the consumed index/catalog structure, including unique model,
deployment and control identities. Malformed data fails before reaching UI
helpers. Relative catalog URLs are supported; these checks do not validate
full native configurations or certify runtime compatibility.

Load one bundle before calling the API. Ajv is included; no separate validator
script or CDN is needed. For this site's deployment prefix:

```html
<script src="/FastVideo/assets/cookbook-config/cookbook-config.js"></script>
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

Adjust asset URLs for other deployment prefixes. Node tools can import the
handwritten ES module directly; they do not require a generated bundle.
`getOptions()` previews pending edits without
full schema/topology validation; `resolveConfig()` validates before generating
output. Edits are a flat path-to-value object with correctly typed JavaScript
values, and only declared controls are editable.

The core returns data and strings. The UI owns layout, widgets, edit state,
input parsing, loading/error states, cancellation, stale-response handling,
guide links, copying and downloads. Catch failures and clear invalid output.
Reuse the loaded catalog and treat it as read-only: validators are cached by
schema object identity. Returned field previews and configurations are copies.

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

## Validation errors

`CookbookValidationError` is an additional named export and browser-global member.
It extends `Error`, so existing `catch (error) { ...error.message }` handlers work.

| Property | Meaning |
| --- | --- |
| `message` | Human-readable explanation; do not parse it for field identity |
| `path` | Canonical selected field or topology path, such as `server.port`; `null` for configuration-wide failures |
| `errors` | Detached records containing `instancePath`, `schemaPath`, `keyword`, `params` and `message` |

For field-schema failures, `errors` preserves Ajv details. `instancePath` is a
JSON Pointer **relative to the selected field's value**: for an object control
at `generator.pipeline.experimental.settings`, `/tags/0` points inside that
object. Missing/additional property names remain in `params`. Non-Ajv failures
use `editable`, `path`, `topology` or `configuration` as their keyword;
`instancePath` and `schemaPath` are empty. A topology failure may include
`params.relatedPaths`. HTTP, JSON parsing and cancellation failures remain
ordinary loading errors. The error records are copied, so later validations
cannot overwrite details already shown by the UI.

```javascript
try {
  const result = api.resolveConfig(catalog, edits);
} catch (error) {
  if (error instanceof api.CookbookValidationError) {
    highlightField(error.path, error.errors);
  }
  showError(error.message);
}
```

## Validation boundaries and future work

The Node builder and browser validate individual selected fields with Ajv and
apply the existing GPU/parallel-degree checks. Hidden baseline fields are
preserved, not exhaustively validated. Native path/type drift checks run
separately in an existing FastVideo Python environment; they are not a Python
dependency of the docs build. See the contributor guide for the command and
its coverage limits.

TODO: consider reusable deployment-wide validation only when a maintained
recipe needs it. It is not implemented in this change:

- An offset frame grid such as `(frames - 1) % 4 === 0` is not `multipleOf: 4`.
  A bounded enum or reusable step/offset rule could express it. Some Wan paths
  align frames rather than reject them, so exact-output policy must be labeled.
- H3's `width * height <= 768 * 1344` requires a cross-field arithmetic check.
  Separate per-dimension bounds cannot enforce it.
- Refinement-dependent geometry could use JSON Schema `if`/`then`/`else` on a
  complete configuration object. The current field-only validator cannot see
  sibling fields. The current LTX2 baseline fixes refinement on and uses 64-pixel
  dimensions accordingly.

Memory estimates, measured hardware support, format versioning and HTTP cache
freshness remain separate follow-ups. No general arithmetic validator or CLI
override workflow is introduced here.

## Coexistence with the current cookbook

Family pages still load `docs/assets/cookbook-recipes.json` through `cookbook.js`
for generate commands, hardware evidence, and the H3 "Run a server" panel.
Those pages are unchanged. This catalog and API sit beside them: manifests under
`docs/cookbook/recipes/` point at the same `examples/serving/*.yaml` baselines
the family pages already advertise, plus authored editable-field metadata.

A later UI change can call `loadIndex` / `resolveConfig` from those pages so
users download a customized serving YAML instead of copying a static file. Until
then, keep recipe IDs aligned with the family catalog (`fasth3-8step`,
`fastwan21`, `wan21-i2v`, `ltx2-distilled`) and treat `cookbook-recipes.json`
`serving.source` as the native baseline, not a second options schema.
