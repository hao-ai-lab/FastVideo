# Cookbook configuration API — temporary review guide

> **TEMPORARY: remove this file after the API documentation has been reviewed.**
> Keep any agreed permanent guidance in the [serving cookbook design](../cookbook/design.md).

This guide describes the public API of [cookbook-config.js](cookbook-config/cookbook-config.js)
for someone replacing the current cookbook UI. The same catalogs and functions
can support React, Vue, or plain HTML. The library returns data and strings;
your UI owns components, layout, state, events, error display, copy buttons,
and downloads. It does not access the DOM, mount a UI, or execute commands.

## Load the library and data

Prepare local assets from the repository root:

```bash
npm ci --prefix docs
npm run build:catalog --prefix docs
```

The npm command bundles the API and Ajv together before generating the JSON catalogs.
Serve the files over HTTP using the docs server or the local demo setup.
Website visitors only need the published assets, not Node or the YAML sources.

Load one script, adjusting the prefix for your deployment:

```html
<script src="/FastVideo/assets/cookbook-config/cookbook-config.js"></script>
```

The browser API is `globalThis.FastVideoConfigCookbook`. The generated bundle
also supports CommonJS. Node tools can instead import named exports directly
from the handwritten `docs/js/cookbook-config.mjs`, without building assets.

```js
const api = require("./docs/assets/cookbook-config/cookbook-config.js"); // From the repository root in CommonJS.
```

`docs/js/cookbook-config.mjs` and `docs/js/cookbook-validator.mjs` are handwritten
source. The combined bundle, its license notices and JSON catalogs are generated
under `docs/assets/cookbook-config/`. Publish this generated directory together.

## Public methods

These four methods and the `CookbookValidationError` class are public. Other
functions are implementation details. The shapes below document the current JavaScript API;
they are not TypeScript declarations or a versioned compatibility guarantee.

| Method | Returns | Purpose |
| --- | --- | --- |
| `loadIndex(url, { signal, fetcher } = {})` | Promise of `{ models, url }` | Fetch the model/deployment menu and retain its URL context. |
| `loadDeployment(index, id, { signal, fetcher } = {})` | Promise of a catalog | Fetch the catalog for a deployment ID from the index. |
| `getOptions(catalog, config = catalog.base_config, edits = {})` | Array of `{ path, schema, value }` | Preview editable fields, values, and GPU-dependent constraints. |
| `resolveConfig(catalog, edits = {})` | Resolved output object | Apply edits, validate, and generate YAML and commands. |

### `loadIndex(url, options)`

Pass an index URL such as `/FastVideo/assets/cookbook-config/index.json`.
The returned `models` array contains entries shaped like:

```js
{
  id: "fastwan21",
  title: "FastWan2.1 1.3B",
  model_id: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
  deployments: [{
    id: "fastwan21/cuda-rest",
    label: "CUDA REST",
    workload: "t2v",
    runtime: "fastvideo-cuda-rest",
    catalog_url: "recipes/fastwan21/cuda-rest.json"
  }]
}
```

Keep the whole returned index context, including `url`, for `loadDeployment()`.
The URL is needed to resolve relative catalog links.

### `loadDeployment(index, id, options)`

Use a deployment ID from `index.models`, such as `fastwan21/cuda-rest`.
The method locates its catalog URL, fetches JSON, and checks that the returned
catalog ID matches, then validates consumed metadata and configuration/control
containers. Both loaders reject malformed entries with context; duplicate model,
deployment and control IDs are rejected. Catalogs are loaded on demand.

The returned catalog includes:

| Property | Meaning |
| --- | --- |
| `id`, `title`, `summary` | Deployment identity and descriptive text. |
| `model`, `deployment` | Model identity and deployment key/label. |
| `workload` | `t2v` or `i2v`. |
| `base_config` | Complete explicit native baseline, including hidden settings. |
| `controls` | Ordered editable fields, each with a `path` and JSON `schema`. |
| `runtime` | Backend, interface, launch arguments, defaults, and setup links. |
| `env` | Environment variables for the launch command. |
| `requirements` | Setup notes to display. |
| `guide` | Optional `{ url }`, or `null`. |
| `source_config` | Repository-relative source baseline path. |

The method returns stored link values without rewriting them. Resolve relative
`guide.url` and `runtime.client_guide_url` against the published index URL.
For a local demo using published documentation links, use the published index
URL as that documentation base instead of the local data index URL.

Both loading methods accept an `AbortSignal` through `signal`, so a UI can
cancel a previous request when selection changes. `fetcher` is an optional
fetch-compatible replacement, useful for tests. HTTP, JSON, lookup, and abort
failures reject the promise; the caller owns loading and error states.

### `getOptions(catalog, config, edits)`

Returns an ordered array of field descriptors:

```js
{
  path: "server.port",
  schema: {
    title: "Port",
    type: "integer",
    default: 8000,
    minimum: 1,
    maximum: 65535
  },
  value: 8000
}
```

Use `path` as the field identifier, `schema` for labels and constraints, and
`value` for the current value. Your renderer chooses the widget. Schemas can
include enums, nullable alternatives, arrays, and objects; they are JSON Schema
fragments, not component specifications.

To preview pending edits and dependent limits, call:

```js
const fields = api.getOptions(catalog, catalog.base_config, edits);
```

GPU-dependent parallelism limits reflect the preview configuration. This method
does not perform full schema/topology validation, though invalid edit paths or
edit-object shapes can throw. Use `resolveConfig()` before presenting output.

### `resolveConfig(catalog, edits)`

`edits` is a flat path-to-value object, not a nested configuration:

```js
const edits = {
  "server.port": 9000,
  "generator.engine.compile.enabled": false
};
const result = api.resolveConfig(catalog, edits);
```

Only fields declared in `catalog.controls` can be edited. Pass correctly typed
JavaScript values: `9000`, not `"9000"`; `false`, not `"false"`. The function
copies the native baseline, applies edits, validates present editable values,
checks topology, and returns:

| Property | Contract |
| --- | --- |
| `config` | Complete edited configuration object. |
| `yaml` | Serialized configuration string to display or save as `config.yaml`. |
| `argv` | Array of server launch arguments from runtime metadata. |
| `command` | Shell command string including recipe environment variables. |
| `clientRequest` | Sample REST request object; `null` for streaming deployments. |
| `clientCommand` | REST sample command or streaming health-check command. |
| `websocketUrl` | Present only for WebSocket deployments. |

Client instructions target the server machine: bind-all hosts become loopback
addresses. A UI describing remote access must explain how to use the server's
reachable address. Returned command strings are instructions, not executed work.

## Minimal UI integration flow

The following code illustrates the data flow inside an async UI initializer;
render the returned variables using your own components.

```js
const api = globalThis.FastVideoConfigCookbook;
const index = await api.loadIndex("/FastVideo/assets/cookbook-config/index.json");
// Populate your model/deployment selector from index.models.
const catalog = await api.loadDeployment(index, "fastwan21/cuda-rest");
let edits = {};
const initialFields = api.getOptions(catalog);

function preview() {
  const fields = api.getOptions(catalog, catalog.base_config, edits);
  try {
    return { fields, result: api.resolveConfig(catalog, edits), error: null };
  } catch (error) {
    return { fields, result: null, error: error.message };
  }
}

// After the UI parses a numeric input:
edits["server.port"] = 9000;
const updated = preview();
// Render updated.fields; display/download updated.result only when valid.

// Remove an edit to restore that field's baseline behavior:
delete edits["server.port"];
// Reset all fields:
edits = {};
```

Catch loading and field-preview failures in the surrounding UI as well.
When changing deployments, clear previous edits and output, cancel obsolete
loads, and render fields from the new catalog. The core does not retain the
UI's selected deployment or edit state.

## Value and error rules

- A field value of `undefined` means omitted/inherited. Do not replace it with
  `schema.default`; defaults are informational annotations.
- Preserve explicit `false`, `0`, and `null`. Do not use truthiness to decide
  whether a field has a value.
- Omitting an edit preserves the baseline. Deleting an edit restores the
  baseline; it does not delete the baseline setting from generated YAML.
- Hidden baseline fields remain in output. Do not reconstruct the configuration
  from visible controls alone.
- Treat the catalog as read-only and reuse it while editing so its cached
  validators can be reused. Preview and resolution return copies.
- Validation errors throw `CookbookValidationError`, which extends `Error`.
  `path` identifies the selected field (or is `null` for configuration-wide
  failures); `errors` contains copied Ajv-compatible records. Nested
  `instancePath` pointers are relative to that field value. Topology, unselected
  edit and path failures also provide structured records. See the
  [error contract](../cookbook/design.md#validation-errors) for keywords and
  related paths. Existing message-only handlers still work. Hide or disable
  stale output when validation fails; do not parse messages for field identity.
- These are authored-field and topology checks. Native runtime validation and
  GPU capacity remain outside this API's contract.

For the existing renderer, see `examples/cookbook/cookbook-demo.js`. For permanent
architecture and setup guidance, see the [design](../cookbook/design.md)
and [contributor guide](../contributing/cookbook_configuration.md).

<!-- REMOVE AFTER REVIEW: this file is temporary API review documentation. -->
