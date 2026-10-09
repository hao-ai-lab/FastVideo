/**
 * Reusable recipe data and configuration helpers. No DOM access or automatic initialization.
 * Native baselines and manifest-selected JSON Schema controls come from the authored YAML catalog.
 */
import { createValidator } from "./cookbook-validator.mjs";

const scope = globalThis;

const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const clone = (value) =>
  Array.isArray(value)
    ? value.map(clone)
    : value !== null && typeof value === "object"
      ? Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]))
      : value;

const pathParts = (path) => path.split(".");
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value) => typeof value === "string" && value.trim().length > 0;

/** Configuration failures retain a field path and detached Ajv-compatible details. */
export class CookbookValidationError extends Error {
  constructor(message, path = null, errors = []) {
    super(message);
    this.name = "CookbookValidationError";
    this.path = path;
    this.errors = clone(errors);
  }
}

function invalidValue(message, path = null, keyword = "configuration", params = {}) {
  return new CookbookValidationError(message, path, [
    {
      instancePath: "",
      schemaPath: "",
      keyword,
      params,
      message,
    },
  ]);
}

function requireShape(condition, context, expectation) {
  if (!condition) throw new Error(`${context}: expected ${expectation}`);
}

function validateIndex(index, baseUrl) {
  requireShape(object(index), "Recipe index", "an object");
  requireShape(Array.isArray(index.models), "Recipe index.models", "an array");
  const modelIds = new Set(),
    deploymentIds = new Set();
  for (const [position, model] of index.models.entries()) {
    const context = `Recipe index.models[${position}]`;
    requireShape(object(model), context, "a model object");
    for (const key of ["id", "title", "model_id"]) requireShape(text(model[key]), `${context}.${key}`, "nonempty text");
    requireShape(!modelIds.has(model.id), `${context}.id (${model.id})`, "a unique model ID");
    modelIds.add(model.id);
    requireShape(Array.isArray(model.deployments), `${context} (${model.id}).deployments`, "an array");
    for (const [offset, deployment] of model.deployments.entries()) {
      const entry = `${context} (${model.id}).deployments[${offset}]`;
      requireShape(object(deployment), entry, "a deployment object");
      for (const key of ["id", "label", "workload", "runtime", "catalog_url"]) {
        requireShape(text(deployment[key]), `${entry}.${key}`, "nonempty text");
      }
      requireShape(
        !deploymentIds.has(deployment.id),
        `${entry}.id (${deployment.id})`,
        "a globally unique deployment ID",
      );
      deploymentIds.add(deployment.id);
      try {
        new URL(deployment.catalog_url, baseUrl);
      } catch {
        throw new Error(`${entry}.catalog_url: expected a URL relative to ${baseUrl}`);
      }
    }
  }
}

function validateServer(server, context) {
  requireShape(object(server), context, "an object");
  requireShape(text(server.host), `${context}.host`, "nonempty text");
  requireShape(Number.isInteger(server.port), `${context}.port`, "an integer");
  if (own(server, "served_model_name"))
    requireShape(
      server.served_model_name === null || typeof server.served_model_name === "string",
      `${context}.served_model_name`,
      "a string or null",
    );
}

function validateCatalog(catalog, deployment) {
  const context = `Deployment ${deployment.id}`;
  requireShape(object(catalog), context, "a catalog object");
  if (catalog.id !== deployment.id) throw new Error(`Unexpected deployment: ${catalog.id}; expected ${deployment.id}`);
  requireShape(object(catalog.model) && text(catalog.model.id), `${context}.model.id`, "nonempty text");
  requireShape(object(catalog.base_config), `${context}.base_config`, "a configuration object");
  requireShape(Array.isArray(catalog.controls), `${context}.controls`, "an array");
  requireShape(
    object(catalog.env) &&
      Object.entries(catalog.env).every(
        ([key, value]) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && typeof value === "string" && !value.includes("\0"),
      ),
    `${context}.env`,
    "environment names mapped to strings",
  );
  const runtime = catalog.runtime;
  requireShape(object(runtime), `${context}.runtime`, "an object");
  requireShape(runtime.id === deployment.runtime, `${context}.runtime.id`, deployment.runtime);
  requireShape(["rest", "websocket"].includes(runtime.interface), `${context}.runtime.interface`, "rest or websocket");
  requireShape(
    Array.isArray(runtime.launch_argv) && runtime.launch_argv.length > 0 && runtime.launch_argv.every(text),
    `${context}.runtime.launch_argv`,
    "a nonempty array of arguments",
  );
  validateServer(runtime.server_defaults, `${context}.runtime.server_defaults`);
  if (own(catalog.base_config, "server"))
    requireShape(object(catalog.base_config.server), `${context}.base_config.server`, "an object");
  validateServer({ ...runtime.server_defaults, ...catalog.base_config.server }, `${context}.base_config.server`);
  if (runtime.topology_defaults != null) {
    requireShape(object(runtime.topology_defaults), `${context}.runtime.topology_defaults`, "an object");
    for (const key of ["num_gpus", ...topologyDegrees]) {
      requireShape(
        Number.isInteger(runtime.topology_defaults[key]),
        `${context}.runtime.topology_defaults.${key}`,
        "an integer",
      );
    }
  }
  requireShape(catalog.workload === deployment.workload, `${context}.workload`, deployment.workload);
  const paths = new Set();
  for (const [position, control] of catalog.controls.entries()) {
    const field = `${context}.controls[${position}]`;
    requireShape(object(control) && text(control.path), `${field}.path`, "a field path");
    requireShape(!paths.has(control.path), `${field}.path (${control.path})`, "a unique control path");
    paths.add(control.path);
    try {
      checkPath(control.path, runtime);
    } catch (error) {
      throw new Error(`${field}.path: ${error.message}`, { cause: error });
    }
    requireShape(object(control.schema), `${field}.schema (${control.path})`, "a schema object");
  }
  if (catalog.requirements !== undefined)
    requireShape(
      Array.isArray(catalog.requirements) && catalog.requirements.every(text),
      `${context}.requirements`,
      "an array of text notes",
    );
  if (catalog.guide != null)
    requireShape(object(catalog.guide) && text(catalog.guide.url), `${context}.guide.url`, "nonempty text");
}

async function loadIndex(url, { signal, fetcher = (...args) => scope.fetch(...args) } = {}) {
  const response = await fetcher(url, { signal });
  signal?.throwIfAborted();
  if (!response.ok) throw new Error(`Could not load recipe index: HTTP ${response.status}`);
  const index = await response.json();
  signal?.throwIfAborted();
  const baseUrl = response.url || new URL(url, scope.location?.href).href;
  validateIndex(index, baseUrl);
  return { models: index.models, url: baseUrl };
}

async function loadDeployment(index, id, { signal, fetcher = (...args) => scope.fetch(...args) } = {}) {
  requireShape(object(index) && text(index.url), "Recipe index.url", "an absolute URL");
  validateIndex(index, index.url);
  const deployment = index.models.flatMap((model) => model.deployments).find((item) => item.id === id);
  if (!deployment) throw new Error(`No exported deployment: ${id}`);
  const response = await fetcher(new URL(deployment.catalog_url, index.url).href, { signal });
  signal?.throwIfAborted();
  if (!response.ok) throw new Error(`Could not load deployment: HTTP ${response.status}`);
  const catalog = await response.json();
  signal?.throwIfAborted();
  validateCatalog(catalog, deployment);
  return catalog;
}

/** Quote a shell argument independently of YAML serialization. */
function shellQuote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(text) ? text : `'${text.replace(/'/g, `'"'"'`)}'`;
}

/** Assign a canonical field path into a private copy of the base configuration. */
function checkPath(path, runtime) {
  const parts = typeof path === "string" ? pathParts(path) : [];
  const roots = ["generator", "server", "default_request"];
  if (runtime.interface === "websocket") roots.push("streaming");
  if (
    parts.length < 2 ||
    !roots.includes(parts[0]) ||
    parts.some(
      (part) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(part) || ["__proto__", "constructor", "prototype"].includes(part),
    )
  ) {
    throw invalidValue(`Invalid configuration path: ${path}`, path, "path");
  }
  return parts;
}

function setPath(config, path, value, runtime) {
  const parts = checkPath(path, runtime);
  let cursor = config;
  for (const part of parts.slice(0, -1)) {
    if (!own(cursor, part)) cursor[part] = {};
    if (!cursor[part] || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) {
      throw invalidValue(`Cannot apply nested field ${path}`, path, "path");
    }
    cursor = cursor[part];
  }
  cursor[parts.at(-1)] = value;
}

const validatorCache = new WeakMap();

function getValue(config, path) {
  return pathParts(path).reduce((value, key) => value?.[key], config);
}

const topologyPrefix = "generator.engine.parallelism.";
const topologyDegrees = ["tp_size", "sp_size", "hsdp_replicate_dim", "hsdp_shard_dim"];
const autoDegrees = new Set(["tp_size", "sp_size", "hsdp_shard_dim"]);

function topologyValues(catalog, config) {
  const defaults = catalog.runtime.topology_defaults;
  if (!defaults) return null; // MLX has no distributed CUDA topology contract.
  const engine = config.generator?.engine;
  return {
    ...defaults,
    ...engine?.parallelism,
    num_gpus: engine?.num_gpus === undefined ? defaults.num_gpus : engine.num_gpus,
  };
}

/** Match inference auto sentinels and distributed group divisibility without changing user values. */
function validateTopology(catalog, config) {
  const topology = topologyValues(catalog, config);
  if (!topology) return;
  const gpuCount = topology.num_gpus;
  if (!Number.isInteger(gpuCount) || gpuCount < 1) {
    throw invalidValue("generator.engine.num_gpus must be a positive integer", "generator.engine.num_gpus", "topology");
  }
  for (const key of topologyDegrees) {
    const degree = topology[key],
      path = `${topologyPrefix}${key}`;
    if (degree === -1 && autoDegrees.has(key)) continue;
    if (!Number.isInteger(degree) || degree < 1) {
      throw invalidValue(
        `${path} must be a positive integer${autoDegrees.has(key) ? " or -1 (automatic)" : ""}`,
        path,
        "topology",
      );
    }
    if (degree > gpuCount || gpuCount % degree !== 0) {
      throw invalidValue(
        `${path} must be at most num_gpus (${gpuCount}) and divide num_gpus evenly`,
        path,
        "topology",
        { relatedPaths: ["generator.engine.num_gpus"] },
      );
    }
  }
}

function applySelections(catalog, config, selections, validate = false) {
  if (!selections || typeof selections !== "object" || Array.isArray(selections)) {
    throw invalidValue("Selections must be a field-to-value object");
  }
  const controls = new Map(catalog.controls.map((control) => [control.path, control]));
  for (const [path, value] of Object.entries(selections)) {
    const control = controls.get(path);
    if (!control) throw invalidValue(`Field is not an editable recipe control: ${path}`, path, "editable");
    if (validate) validateConfig(control.schema, value, path);
    setPath(config, path, clone(value), catalog.runtime);
  }
}

function validateConfig(schema, config, path) {
  let validate = validatorCache.get(schema);
  if (!validate) {
    validate = createValidator(schema);
    validatorCache.set(schema, validate);
  }
  if (!validate(config)) {
    const messages = validate.errors.map((error) => {
      const parts = error.instancePath
        .split("/")
        .filter(Boolean)
        .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"));
      const property = error.params.missingProperty ?? error.params.additionalProperty;
      if (property) parts.push(property);
      return `${parts.join(".") || "config"} ${error.message}`;
    });
    throw new CookbookValidationError(`${path}: ${messages.join("; ")}`, path, validate.errors);
  }
}

/**
 * Serialize JSON-shaped configuration data as YAML without a browser dependency.
 * Nested mappings use indentation; arrays use JSON syntax, which YAML accepts.
 * String values and mapping keys use JSON quoting so YAML 1.1 readers preserve their types.
 */
function yamlValue(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw invalidValue("Configuration numbers must be finite");
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw invalidValue("Configuration integer exceeds JavaScript's safe integer range");
    }
    // PyYAML needs a decimal point and signed exponent to recognize scientific notation as a float.
    return JSON.stringify(value).replace(
      /^(-?\d+(?:\.\d+)?)e([+-]?)(\d+)$/i,
      (_, mantissa, sign, exponent) =>
        `${mantissa.includes(".") ? mantissa : `${mantissa}.0`}e${sign || "+"}${exponent}`,
    );
  }
  if (Array.isArray(value)) return `[${value.map(yamlValue).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .map(([key, item]) => `${JSON.stringify(key)}:${yamlValue(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function toYaml(value, depth = 0) {
  const indent = "  ".repeat(depth);
  return Object.entries(value)
    .map(([key, item]) => {
      const yamlKey = JSON.stringify(key);
      const isMapping = item !== null && typeof item === "object" && !Array.isArray(item);
      if (isMapping && Object.keys(item).length) return `${indent}${yamlKey}:\n${toYaml(item, depth + 1)}`;
      return `${indent}${yamlKey}: ${yamlValue(item)}\n`;
    })
    .join("");
}

/** Build same-machine client instructions; a bind-all address is not a client destination. */
function sampleRequest(catalog, config) {
  const server = { ...catalog.runtime.server_defaults, ...config.server };
  let host = server.host;
  if (host === "0.0.0.0") host = "127.0.0.1";
  if (host === "::" || host === "[::]") host = "::1";
  if (host.includes(":") && !host.startsWith("[")) host = `[${host}]`;
  if (catalog.runtime.interface === "websocket") {
    return {
      body: null,
      command: ["curl", "--fail-with-body", `http://${host}:${server.port}/health`].map(shellQuote).join(" "),
      websocketUrl: `ws://${host}:${server.port}/v1/stream`,
    };
  }
  const components = config.generator?.pipeline?.components;
  const model = components?.lora_path ? components.lora_nickname : server.served_model_name || catalog.model.id;
  const body = { ...(model === undefined ? {} : { model }), prompt: "A river flowing through a peaceful forest" };
  if (catalog.workload === "i2v") body.input_reference = "/absolute/path/to/first-frame.png";
  const argv = [
    "curl",
    "--fail-with-body",
    `http://${host}:${server.port}/v1/videos/sync`,
    "-H",
    "Content-Type: application/json",
    "--data",
    JSON.stringify(body),
  ];
  argv.push("--output", "output.mp4");
  return { body, command: argv.map(shellQuote).join(" ") };
}

/**
 * Keep every explicit native recipe setting, then apply edits only to selected controls.
 * Omitted fields remain visibly inherited; declared defaults are not effective runtime values.
 * Reset is resolveConfig(catalog, {}); explicit false, zero, null and arrays survive.
 * Example: {"server.port": 9000} changes only the recipe port; hidden VSA settings remain.
 */
function resolveConfig(catalog, selections = {}) {
  const config = clone(catalog.base_config);
  applySelections(catalog, config, selections, true);
  for (const control of catalog.controls) {
    const value = getValue(config, control.path);
    if (value !== undefined) {
      validateConfig(control.schema, value, control.path);
    }
  }
  validateTopology(catalog, config);
  const argv = [...catalog.runtime.launch_argv];
  const environment = Object.entries(catalog.env).map(([key, value]) => `${key}=${shellQuote(value)}`);
  const sample = sampleRequest(catalog, config);
  return {
    config,
    yaml: toYaml(config),
    argv,
    command: [...environment, ...argv.map(shellQuote)].join(" "),
    clientCommand: sample.command,
    clientRequest: sample.body,
    ...(sample.websocketUrl ? { websocketUrl: sample.websocketUrl } : {}),
  };
}

/**
 * Return metadata/current values, including degree limits derived from the selected GPU count.
 * Optional edits preview incomplete configurations without validation; resolveConfig validates output.
 * Neither previews nor declared defaults insert inherited fields into saved configuration.
 */
function getOptions(catalog, config = catalog.base_config, selections = {}) {
  const preview = clone(config);
  applySelections(catalog, preview, selections);
  const topology = topologyValues(catalog, preview);
  return catalog.controls.map(({ path, schema }) => {
    const field = clone(schema),
      value = getValue(preview, path);
    const key = path.startsWith(topologyPrefix) ? path.slice(topologyPrefix.length) : null;
    if (topology && topologyDegrees.includes(key)) {
      field.minimum = Math.max(field.minimum ?? -Infinity, autoDegrees.has(key) ? -1 : 1);
      if (Number.isInteger(topology.num_gpus) && topology.num_gpus > 0) {
        field.maximum = Math.min(field.maximum ?? Infinity, topology.num_gpus);
      }
      const rule = { not: { const: 0 } };
      if (autoDegrees.has(key)) field.allOf = [...(field.allOf || []), rule];
      field.description = [
        field.description,
        `Positive values must divide num_gpus evenly${autoDegrees.has(key) ? "; -1 selects automatic sizing" : ""}.`,
      ]
        .filter(Boolean)
        .join(" ");
    } else if (topology && path === "generator.engine.num_gpus") {
      field.minimum = Math.max(field.minimum ?? -Infinity, 1);
    }
    return { path, schema: field, value: value === undefined ? undefined : clone(value) };
  });
}

export { loadIndex, loadDeployment, getOptions, resolveConfig };
