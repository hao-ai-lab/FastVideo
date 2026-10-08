/**
 * Reusable recipe data and configuration helpers. No DOM access or automatic initialization.
 * Native baselines and manifest-selected JSON Schema controls come from the Python exporter.
 */
((scope) => {
  "use strict";

  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));

  const pathParts = (path) => path.split(".");

  async function loadIndex(url, { signal, fetcher = (...args) => scope.fetch(...args) } = {}) {
    const response = await fetcher(url, { signal });
    signal?.throwIfAborted();
    if (!response.ok) throw new Error(`Could not load recipe index: HTTP ${response.status}`);
    const index = await response.json();
    signal?.throwIfAborted();
    if (!Array.isArray(index.models)) throw new Error("Invalid recipe index: models must be an array");
    return { models: index.models, url: response.url || new URL(url, scope.location?.href).href };
  }

  async function loadDeployment(index, id, { signal, fetcher = (...args) => scope.fetch(...args) } = {}) {
    const deployment = index.models.flatMap((model) => model.deployments).find((item) => item.id === id);
    if (!deployment) throw new Error(`No exported deployment: ${id}`);
    const response = await fetcher(new URL(deployment.catalog_url, index.url).href, { signal });
    signal?.throwIfAborted();
    if (!response.ok) throw new Error(`Could not load deployment: HTTP ${response.status}`);
    const catalog = await response.json();
    signal?.throwIfAborted();
    if (catalog.id !== id) throw new Error(`Unexpected deployment: ${catalog.id}`);
    return catalog;
  }

  /** Quote a shell argument independently of YAML serialization. */
  function shellQuote(value) {
    const text = String(value);
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(text) ? text : `'${text.replace(/'/g, `'"'"'`)}'`;
  }

  /** Assign a canonical field path into a private copy of the base configuration. */
  function setPath(config, path, value) {
    const parts = pathParts(path);
    if (!["generator", "server", "default_request"].includes(parts[0]) || parts.some((part) =>
      !/^[A-Za-z_][A-Za-z0-9_.]*$/.test(part) || ["__proto__", "constructor", "prototype"].includes(part))) {
      throw new Error(`Invalid configuration path: ${path}`);
    }
    let cursor = config;
    for (const part of parts.slice(0, -1)) {
      if (!own(cursor, part)) cursor[part] = {};
      if (!cursor[part] || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) {
        throw new Error(`Cannot apply nested field ${path}`);
      }
      cursor = cursor[part];
    }
    cursor[parts.at(-1)] = value;
  }

  const validatorCache = new WeakMap();

  function getValue(config, path) {
    return pathParts(path).reduce((value, key) => value?.[key], config);
  }

  function validateConfig(schema, config) {
    let validate = validatorCache.get(schema);
    if (!validate) {
      const library = typeof module !== "undefined" && module.exports ?
        require("./cookbook-validator.js") : scope.FastVideoSchema;
      if (!library) throw new Error("The JSON Schema validator has not loaded");
      validate = library.createValidator(schema);
      validatorCache.set(schema, validate);
    }
    if (!validate(config)) {
      const messages = validate.errors.map((error) => {
        const parts = error.instancePath.split("/").filter(Boolean)
          .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"));
        const property = error.params.missingProperty ?? error.params.additionalProperty;
        if (property) parts.push(property);
        return `${parts.join(".") || "config"} ${error.message}`;
      });
      throw new Error(messages.join("; "));
    }
  }

  /**
   * Serialize JSON-shaped configuration data as YAML without a browser dependency.
   * Nested mappings use indentation; arrays use JSON syntax, which YAML accepts.
   * All strings use JSON quoting so empty text and values such as "false" stay strings.
   */
  function yamlValue(value) {
    if (typeof value === "number") {
      // PyYAML needs a decimal point and signed exponent to recognize scientific notation as a float.
      return JSON.stringify(value).replace(/^(-?\d+(?:\.\d+)?)e([+-]?)(\d+)$/i,
        (_, mantissa, sign, exponent) => `${mantissa.includes(".") ? mantissa : `${mantissa}.0`}e${sign || "+"}${exponent}`);
    }
    if (Array.isArray(value)) return `[${value.map(yamlValue).join(",")}]`;
    if (value !== null && typeof value === "object") {
      return `{${Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}:${yamlValue(item)}`).join(",")}}`;
    }
    return JSON.stringify(value);
  }

  function toYaml(value, depth = 0) {
    const indent = "  ".repeat(depth);
    return Object.entries(value).map(([key, item]) => {
      const yamlKey = /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ? key : JSON.stringify(key);
      const isMapping = item !== null && typeof item === "object" && !Array.isArray(item);
      if (isMapping && Object.keys(item).length) return `${indent}${yamlKey}:\n${toYaml(item, depth + 1)}`;
      return `${indent}${yamlKey}: ${yamlValue(item)}\n`;
    }).join("");
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
    const body = { model: server.served_model_name || catalog.model.id, prompt: "A river flowing through a peaceful forest" };
    if (catalog.workload === "i2v") body.input_reference = "/absolute/path/to/first-frame.png";
    const argv = ["curl", "--fail-with-body", `http://${host}:${server.port}/v1/videos/sync`,
      "-H", "Content-Type: application/json", "--data", JSON.stringify(body)];
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
    if (!selections || typeof selections !== "object" || Array.isArray(selections)) {
      throw new Error("Selections must be a field-to-value object");
    }
    const config = clone(catalog.base_config);
    const controls = new Map(catalog.controls.map((control) => [control.path, control]));
    for (const [path, value] of Object.entries(selections)) {
      const control = controls.get(path);
      if (!control) throw new Error(`Field is not an editable recipe control: ${path}`);
      try { validateConfig(control.schema, value); }
      catch (failure) { throw new Error(`${path}: ${failure.message}`); }
      setPath(config, path, value);
    }
    for (const control of catalog.controls) {
      const value = getValue(config, control.path);
      if (value !== undefined) validateConfig(control.schema, value);
    }
    const argv = [...catalog.runtime.launch_argv];
    const environment = Object.entries(catalog.env).map(([key, value]) => `${key}=${shellQuote(value)}`);
    const sample = sampleRequest(catalog, config);
    return {
      config, yaml: toYaml(config), argv,
      command: [...environment, ...argv.map(shellQuote)].join(" "),
      clientCommand: sample.command, clientRequest: sample.body,
      ...(sample.websocketUrl ? { websocketUrl: sample.websocketUrl } : {}),
    };
  }

  /** Return every enabled control, preserving undefined (inherited) versus explicit null/false/zero. */
  function getOptions(catalog, config = catalog.base_config) {
    return catalog.controls.map(({ path, schema }) => {
      const value = getValue(config, path);
      return { path, schema: clone(schema), value: value === undefined ? undefined : clone(value) };
    });
  }

  const api = { loadIndex, loadDeployment, getOptions, resolveConfig };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  scope.FastVideoConfigCookbook = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
