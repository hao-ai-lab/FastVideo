/**
 * Schema-backed cookbook demo: exported Python defaults -> selections -> config.yaml.
 *
 * An explicit Python export writes the committed cookbook-config.json; the documentation
 * build verifies and ships it. resolveConfig() reads that data without a browser and returns
 * a complete configuration, YAML, and short launch
 * command. mount() fetches the static JSON once and connects the resolver to controls.
 * Standard JSON Schema defaults and constraints come from the exporter; Ajv validates them.
 * Labels and grouping live here.
 * Neither function starts a server, loads a model, or submits an inference request.
 */
((scope) => {
  "use strict";

  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));

  /** Quote a shell argument independently of YAML serialization. */
  function shellQuote(value) {
    const text = String(value);
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(text) ? text : `'${text.replace(/'/g, `'"'"'`)}'`;
  }

  /** Assign a canonical field path into a private copy of the base configuration. */
  function setPath(config, path, value) {
    const parts = path.split(".");
    if (!/^(generator|server|default_request)\./.test(path) || parts.some((part) =>
      !/^[A-Za-z_][A-Za-z0-9_]*$/.test(part) || ["__proto__", "constructor", "prototype"].includes(part))) {
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

  /** Resolve local JSON Schema references for default extraction and field presentation. */
  function resolveSchemaNode(node, root, seen = new Set()) {
    if (!node || typeof node !== "object" || !node.$ref) return node;
    if (!node.$ref.startsWith("#/") || seen.has(node.$ref)) {
      throw new Error(`Cannot materialize schema reference: ${node.$ref}`);
    }
    seen.add(node.$ref);
    const target = node.$ref.slice(2).split("/").reduce((value, part) =>
      value?.[part.replace(/~1/g, "/").replace(/~0/g, "~")], root);
    if (!target) throw new Error(`Missing schema reference: ${node.$ref}`);
    const { $ref, ...siblings } = node;
    return { ...resolveSchemaNode(target, root, seen), ...siblings };
  }

  /** Materialize only explicit JSON Schema defaults/const values, including false, zero and null. */
  function getDefaultConfig(schema, root = schema) {
    const node = resolveSchemaNode(schema, root);
    if (own(node, "default")) return clone(node.default);
    if (own(node, "const")) return clone(node.const);
    if (!node.properties) return undefined;
    const config = {};
    for (const [key, property] of Object.entries(node.properties)) {
      const value = getDefaultConfig(property, root);
      if (value !== undefined) config[key] = value;
    }
    return config;
  }

  /** Look up a canonical field through the schema's properties; constraints remain standard JSON Schema. */
  function getSchemaField(schema, path) {
    let node = schema;
    for (const part of path.split(".")) node = resolveSchemaNode(node, schema)?.properties?.[part];
    return resolveSchemaNode(node, schema);
  }

  /** Enumerate editable leaves so newly exported fields need no frontend registration. */
  function getEditableFields(schema, prefix = "", root = schema) {
    const node = resolveSchemaNode(schema, root);
    if (own(node, "const")) return [];
    if (!node.properties) return prefix ? [{ path: prefix, field: node }] : [];
    return Object.entries(node.properties).flatMap(([key, property]) =>
      getEditableFields(property, prefix ? `${prefix}.${key}` : key, root));
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

  /**
   * Resolve schema/model defaults plus flat canonical-path overrides into runnable YAML.
   * Inputs are not mutated. Ajv validates the complete result before any output is emitted.
   * Example: resolveConfig(catalog, {"default_request.sampling.num_frames": 81}).
   * Request defaults are saved in YAML; the launch command remains `--config config.yaml`.
   */
  function resolveConfig(catalog, selections = {}) {
    if (!selections || typeof selections !== "object" || Array.isArray(selections)) {
      throw new Error("Selections must be a field-to-value object");
    }
    const config = getDefaultConfig(catalog.schema);
    for (const [path, value] of Object.entries(selections)) setPath(config, path, value);
    validateConfig(catalog.schema, config);

    let host = config.server.host;
    if (["0.0.0.0", "::"].includes(host)) host = "127.0.0.1";
    if (host.includes(":")) host = `[${host}]`;
    const baseUrl = `http://${host}:${config.server.port}`;
    const argv = [...catalog.runtime.launch_argv];
    const clientRequest = {
      endpoint: `${baseUrl}/v1/videos/sync`,
      body: {
        model: config.server.served_model_name || catalog.model.id,
        prompt: "A cinematic view of a futuristic city at sunset, smooth camera pan",
      },
      outputFile: "output.mp4",
    };
    return {
      config,
      yaml: toYaml(config),
      argv,
      command: argv.map(shellQuote).join(" "),
      healthCommand: `curl --fail-with-body ${shellQuote(`${baseUrl}/health`)}`,
      clientRequest,
      clientCommand: `curl --fail-with-body --request POST ${shellQuote(clientRequest.endpoint)} \\\n  --header 'Content-Type: application/json' \\\n  --data ${shellQuote(JSON.stringify(clientRequest.body))} \\\n  --output output.mp4`,
    };
  }

  const api = { resolveConfig, getDefaultConfig, getSchemaField, getEditableFields, toYaml, shellQuote };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  scope.FastVideoConfigCookbook = api;
  if (typeof document === "undefined") return;

  // Optional presentation preferences. Every other exported field gets a generic label/group.
  const groups = {
    execution: [
      ["generator.engine.num_gpus", "GPU count"],
      ["generator.engine.offload.vae", "Offload VAE to CPU"],
    ],
    request: [
      ["default_request.sampling.num_frames", "Frames"],
      ["default_request.sampling.width", "Width"],
      ["default_request.sampling.height", "Height"],
      ["default_request.sampling.num_inference_steps", "Sampling steps"],
    ],
    advanced: [
      ["default_request.sampling.fps", "Frames per second"],
      ["default_request.sampling.guidance_scale", "Guidance scale"],
      ["default_request.sampling.seed", "Seed"],
      ["server.host", "Server host"],
      ["server.port", "Server port"],
    ],
  };

  /**
   * Load static metadata, render basic controls, and keep YAML/copy/download output synchronized.
   * TODO(cookbook-ui): Replace this temporary page adapter with the final cookbook UI;
   * retain the pure resolver, serializer, and their tests as the shared generation layer.
   */
  async function mount(root) {
    const status = root.querySelector("[data-config-status]");
    const error = root.querySelector("[data-config-error]");
    const output = root.querySelector("[data-config-output]");
    const outputButtons = root.querySelectorAll("[data-config-copy], [data-config-download]");
    let resolved = null;
    let selections = {};
    const inputs = new Map();

    function invalidate(message) {
      resolved = null;
      output.hidden = true;
      outputButtons.forEach((button) => { button.disabled = true; });
      error.textContent = message;
    }

    try {
      const response = await fetch(root.dataset.metadata);
      if (!response.ok) throw new Error(`Could not load configuration metadata: HTTP ${response.status}`);
      const catalog = await response.json();
      root.querySelector("[data-config-model]").textContent = `${catalog.model.label} · ${catalog.runtime.label}`;

      function render() {
        status.textContent = "";
        try {
          resolved = resolveConfig(catalog, selections);
          error.textContent = "";
          output.hidden = false;
          outputButtons.forEach((button) => { button.disabled = false; });
          for (const key of ["yaml", "command", "healthCommand", "clientCommand"]) {
            root.querySelector(`[data-config-code="${key}"]`).textContent = resolved[key];
          }
        } catch (failure) {
          invalidate(failure.message);
        }
      }

      const preferences = Object.entries(groups).flatMap(([group, definitions]) =>
        definitions.map(([path, label]) => ({ path, label, group })));
      const fields = getEditableFields(catalog.schema).sort((a, b) => {
        const rank = (path) => {
          const index = preferences.findIndex((item) => item.path === path);
          return index < 0 ? preferences.length : index;
        };
        return rank(a.path) - rank(b.path);
      });
      for (const { path, field: declared } of fields) {
        const preference = preferences.find((item) => item.path === path);
        const group = preference?.group ?? (path.startsWith("generator.") ? "execution" :
          path.startsWith("default_request.") ? "request" : "advanced");
        const fallback = path.split(".").at(-1).replace(/_/g, " ");
        const label = preference?.label ?? declared.title ?? `${fallback[0].toUpperCase()}${fallback.slice(1)}`;
        const container = root.querySelector(`[data-config-group="${group}"]`);
        // Nullable schemas still render the non-null type; Ajv validates the original whole schema.
        const branch = declared.anyOf?.find((item) => item.type && item.type !== "null");
        const field = { ...branch, ...declared };
        const type = Array.isArray(field.type) ? field.type.find((item) => item !== "null") : field.type;
        const defaultValue = getDefaultConfig(declared, catalog.schema);
        const row = document.createElement("label");
        row.className = "config-builder-field";
        const title = document.createElement("strong");
        title.textContent = label;
        let input;
        if (field.enum) {
          input = document.createElement("select");
          field.enum.forEach((value, index) => {
            const option = document.createElement("option");
            option.value = index;
            option.textContent = value;
            input.append(option);
          });
        } else {
          input = document.createElement("input");
          input.type = type === "boolean" ? "checkbox" : type === "string" ? "text" : "number";
          if (field.minimum !== undefined) input.min = field.minimum;
          if (field.maximum !== undefined) input.max = field.maximum;
          input.step = field.multipleOf ?? (type === "integer" ? "1" : "any");
        }
        input.dataset.configPath = path;
        input.setAttribute("aria-label", label);
        const setDefault = () => {
          if (field.enum) input.value = field.enum.indexOf(defaultValue);
          else if (type === "boolean") input.checked = defaultValue;
          else input.value = defaultValue;
        };
        setDefault();
        inputs.set(path, setDefault);
        input.addEventListener(field.enum ? "change" : "input", () => {
          selections[path] = field.enum ? field.enum[input.value] : type === "boolean" ? input.checked :
            type === "string" ? input.value : input.value === "" ? NaN : Number(input.value);
          render();
        });
        const origin = document.createElement("small");
        origin.textContent = `Default: ${defaultValue}`;
        row.append(title, input, origin);
        if (path === "default_request.sampling.num_frames" || (field.multipleOf !== undefined &&
          ["default_request.sampling.width", "default_request.sampling.height"].includes(path))) {
          const help = document.createElement("small");
          help.textContent = path.endsWith("num_frames") ?
            "Runtime may align the requested length to the model’s VAE." :
            `Multiples of ${field.multipleOf} pixels.`;
          row.append(help);
        }
        container.append(row);
      }

      root.querySelectorAll("[data-config-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          if (!resolved) return;
          try {
            await navigator.clipboard.writeText(resolved[button.dataset.configCopy]);
            status.textContent = `${button.dataset.copyLabel} copied.`;
          } catch (_) {
            status.textContent = "Clipboard access failed. Select and copy the text in the output block.";
          }
        });
      });
      root.querySelector("[data-config-download]").addEventListener("click", () => {
        if (!resolved) return;
        const url = URL.createObjectURL(new Blob([resolved.yaml], { type: "text/yaml;charset=utf-8" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = "config.yaml";
        link.hidden = true;
        document.body.append(link);
        link.click();
        link.remove();
        // Give browsers time to consume the Blob before releasing its URL.
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        status.textContent = "config.yaml download started.";
      });
      const reset = root.querySelector("[data-config-reset]");
      reset.disabled = false;
      reset.addEventListener("click", () => {
        selections = {};
        inputs.forEach((setDefault) => setDefault());
        render();
        status.textContent = "Defaults restored.";
      });
      render();
    } catch (failure) {
      status.textContent = "";
      invalidate(failure.message);
    }
  }

  const init = () => document.querySelectorAll("[data-config-builder]").forEach((root) => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = "true";
    mount(root);
  });
  if (scope.document$) scope.document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof globalThis !== "undefined" ? globalThis : window);
