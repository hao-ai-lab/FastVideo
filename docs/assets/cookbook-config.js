/**
 * Schema-backed cookbook demo: exported Python defaults -> selections -> config.yaml.
 *
 * Python exports a selected model index and complete per-model catalogs. The
 * browser loads one catalog and edits the discovered fields, without a field list.
 * Standard JSON Schema defaults and constraints come from the exporter; Ajv validates them.
 * Labels and grouping live here.
 * These helpers do not start a server, load a model, or submit an inference request.
 */
((scope) => {
  "use strict";

  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));

  const pathParts = (path) => path.startsWith("/") ? path.slice(1).split("/")
    .map((key) => key.replace(/~1/g, "/").replace(/~0/g, "~")) : path.split(".");

  /** Fetch only the selected catalog; obsolete responses and failures never reach the form. */
  function createCatalogLoader(index, indexUrl, onChange, fetcher = (...args) => scope.fetch(...args)) {
    let generation = 0, controller;
    return async (modelId) => {
      const token = ++generation;
      controller?.abort();
      controller = new AbortController();
      const model = index.models.find((item) => item.id === modelId);
      onChange({ state: "loading", model });
      try {
        if (!model) throw new Error(`No exported configuration for model: ${modelId}`);
        const response = await fetcher(new URL(model.catalog_url, indexUrl).href, { signal: controller.signal });
        if (token !== generation) return;
        if (!response.ok) throw new Error(`Could not load model configuration: HTTP ${response.status}`);
        const catalog = await response.json();
        if (token !== generation) return;
        if (catalog.model.id !== modelId) throw new Error(`Unexpected model configuration: ${catalog.model.id}`);
        onChange({ state: "ready", catalog });
      } catch (failure) {
        if (token === generation) onChange({ state: "error", message: failure.message });
      }
    };
  }

  /** Quote a shell argument independently of YAML serialization. */
  function shellQuote(value) {
    const text = String(value);
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(text) ? text : `'${text.replace(/'/g, `'"'"'`)}'`;
  }

  /** Assign a canonical field path into a private copy of the base configuration. */
  function setPath(config, path, value) {
    const parts = pathParts(path);
    if (!["generator", "server", "default_request", "streaming"].includes(parts[0]) || parts.some((part) =>
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
    for (const part of pathParts(path)) node = resolveSchemaNode(node, schema)?.properties?.[part];
    return resolveSchemaNode(node, schema);
  }

  /** Enumerate editable leaves so newly exported fields need no frontend registration. */
  function getEditableFields(schema, parts = [], root = schema) {
    const node = resolveSchemaNode(schema, root);
    if (own(node, "const")) return [];
    if (!node.properties) return parts.length ? [{
      path: parts.join("."), pointer: "/" + parts.map((key) => key.replace(/~/g, "~0").replace(/\//g, "~1")).join("/"), field: node,
    }] : [];
    return Object.entries(node.properties).flatMap(([key, property]) =>
      getEditableFields(property, [...parts, key], root));
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
   * Resolve declared defaults for the form and explicit edits for the YAML download.
   * Inputs are not mutated. Ajv validates both views before any output is emitted.
   * Example: resolveConfig(catalog, {"default_request.sampling.num_frames": 81}).
   * Only edited request defaults are pinned in YAML, preserving runtime inheritance.
   * `values` supplies the form's defaults; `config` and `yaml` contain the saved edits.
   * The launch command remains `fastvideo serve --config config.yaml`.
   */
  function resolveConfig(catalog, selections = {}) {
    if (!selections || typeof selections !== "object" || Array.isArray(selections)) {
      throw new Error("Selections must be a field-to-value object");
    }
    const values = getDefaultConfig(catalog.schema);
    const config = { generator: { model_path: catalog.model.id } };
    if (values.generator?.pipeline?.workload_type) {
      config.generator.pipeline = { workload_type: values.generator.pipeline.workload_type };
    }
    for (const [path, value] of Object.entries(selections)) {
      setPath(values, path, value);
      setPath(config, path, value);
    }
    validateConfig(catalog.schema, values);
    validateConfig(catalog.schema, config);

    const argv = [...catalog.runtime.launch_argv];
    return {
      config,
      values,
      yaml: toYaml(config),
      argv,
      command: argv.map(shellQuote).join(" "),
    };
  }

  const api = { resolveConfig, createCatalogLoader, getDefaultConfig, getSchemaField, getEditableFields, toYaml, shellQuote };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  scope.FastVideoConfigCookbook = api;
  if (typeof document === "undefined") return;

  // TODO(cookbook-ui): Replace this demo DOM adapter; keep schema selection and YAML generation.
  async function mount(root) {
    const status = root.querySelector("[data-config-status]");
    const error = root.querySelector("[data-config-error]");
    const output = root.querySelector("[data-config-output]");
    const buttons = root.querySelectorAll("[data-config-copy], [data-config-download]");
    const picker = root.querySelector("[data-config-model-picker]");
    const controls = root.querySelector("[data-config-controls]");
    const reset = root.querySelector("[data-config-reset]");
    let catalog, resolved, selections = {};
    const inputErrors = new Map();

    function invalidate(message) {
      resolved = null;
      output.hidden = true;
      buttons.forEach((button) => { button.disabled = true; });
      error.textContent = message;
    }
    function render() {
      status.textContent = "";
      try {
        if (inputErrors.size) throw new Error([...inputErrors.values()].join("; "));
        resolved = resolveConfig(catalog, selections);
        error.textContent = "";
        output.hidden = false;
        buttons.forEach((button) => { button.disabled = false; });
        for (const key of ["yaml", "command"]) {
          root.querySelector(`[data-config-code="${key}"]`).textContent = resolved[key];
        }
      } catch (failure) { invalidate(failure.message); }
    }

    function renderControls() {
      controls.replaceChildren();
      const sections = new Map();
      for (const { path, pointer, field } of getEditableFields(catalog.schema)) {
        const parts = pathParts(pointer);
        const sectionName = parts.length > 2 ? parts.slice(0, 2).join(".") : parts[0];
        if (!sections.has(sectionName)) {
          const section = document.createElement("details");
          section.className = "config-builder-section";
          const title = document.createElement("summary");
          title.textContent = sectionName;
          const content = document.createElement("div");
          content.className = "config-builder-fields";
          section.append(title, content);
          controls.append(section);
          sections.set(sectionName, content);
        }
        const row = document.createElement("label");
        row.className = "config-builder-field";
        const title = document.createElement("strong");
        title.textContent = field.title || parts.at(-1).replace(/_/g, " ");
        const address = document.createElement("small");
        address.textContent = path;
        const value = getDefaultConfig(field, catalog.schema);
        const branches = field.anyOf || field.oneOf || [field];
        const nonNull = branches.filter((item) => item.type !== "null");
        const basic = nonNull.length === 1 ? nonNull[0] : field;
        const type = basic.type;
        const nullable = branches.some((item) => item.type === "null");
        const choiceValues = field.enum || basic.enum || (type === "boolean" ? [false, true] : null);
        let input, automatic;
        if (choiceValues) {
          input = document.createElement("select");
          const choices = [...choiceValues];
          if (nullable && !choices.includes(null)) choices.unshift(null);
          choices.forEach((item, index) => {
            const option = document.createElement("option");
            option.value = index;
            option.textContent = item === null ? "Auto" : String(item);
            input.append(option);
          });
          input.value = choices.findIndex((item) => item === value);
          input.addEventListener("change", () => { selections[pointer] = choices[input.value]; render(); });
        } else {
          const scalar = ["string", "integer", "number"].includes(type);
          input = document.createElement(scalar ? "input" : "textarea");
          if (scalar) {
            input.type = type === "string" ? "text" : "number";
            if (basic.minimum !== undefined) input.min = basic.minimum;
            if (basic.maximum !== undefined) input.max = basic.maximum;
            input.step = basic.multipleOf || (type === "integer" ? "1" : "any");
            input.value = value == null ? "" : value;
          } else {
            input.rows = 3;
            input.placeholder = "JSON value";
            input.value = value === undefined ? "" : JSON.stringify(value, null, 2);
          }
          const update = () => {
            try {
              if (automatic?.checked) selections[pointer] = null;
              else if (type === "string" && scalar) selections[pointer] = input.value;
              else if (scalar) selections[pointer] = input.value === "" ? NaN : Number(input.value);
              else if (input.value.trim() === "" && value === undefined) delete selections[pointer];
              else selections[pointer] = JSON.parse(input.value);
              inputErrors.delete(pointer);
              render();
            } catch (failure) {
              inputErrors.set(pointer, `${path}: ${failure.message}`);
              invalidate(inputErrors.get(pointer));
            }
          };
          if (nullable && scalar) {
            automatic = document.createElement("input");
            automatic.type = "checkbox";
            automatic.checked = value === null;
            automatic.setAttribute("aria-label", `${path}: Auto`);
            input.disabled = automatic.checked;
            automatic.addEventListener("change", () => { input.disabled = automatic.checked; update(); });
            const autoLabel = document.createElement("span");
            autoLabel.append(automatic, " Auto");
            row.append(autoLabel);
          }
          input.addEventListener("input", update);
        }
        input.dataset.configPath = pointer;
        input.setAttribute("aria-label", path);
        const initial = document.createElement("small");
        initial.textContent = value === undefined ? "Default: unset" : `Default: ${JSON.stringify(value)}`;
        row.prepend(title, address);
        row.append(input, initial);
        if (field.description) {
          const help = document.createElement("small");
          help.textContent = field.description;
          row.append(help);
        }
        sections.get(sectionName).append(row);
      }
    }

    try {
      const response = await fetch(root.dataset.metadata);
      if (!response.ok) throw new Error(`Could not load registry metadata: HTTP ${response.status}`);
      const indexUrl = response.url;
      const index = await response.json();
      if (!index.models.length) throw new Error("No models are selected for this configuration builder");
      for (const model of index.models) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.id;
        picker.append(option);
      }
      const resetForm = () => {
        selections = {};
        inputErrors.clear();
        renderControls();
        render();
      };
      const load = createCatalogLoader(index, indexUrl, (update) => {
        if (update.state === "loading") {
          catalog = null;
          selections = {};
          inputErrors.clear();
          controls.replaceChildren();
          reset.disabled = true;
          invalidate("");
          root.querySelector("[data-config-model]").textContent = update.model?.label || picker.value;
          status.textContent = "Loading the selected model configuration…";
        } else if (update.state === "ready") {
          catalog = update.catalog;
          root.querySelector("[data-config-model]").textContent =
            `${catalog.model.label} · ${catalog.model.workload_types.join(", ")} · ${catalog.runtime.label}`;
          resetForm();
          reset.disabled = false;
        } else {
          status.textContent = "";
          invalidate(update.message);
        }
      });
      picker.value = index.models[0].id;
      picker.disabled = false;
      picker.addEventListener("change", () => load(picker.value));
      reset.addEventListener("click", resetForm);
      root.querySelectorAll("[data-config-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          if (!resolved) return;
          try {
            await navigator.clipboard.writeText(resolved[button.dataset.configCopy]);
            status.textContent = `${button.dataset.copyLabel} copied.`;
          } catch (_) { status.textContent = "Clipboard unavailable. Select and copy the output text."; }
        });
      });
      root.querySelector("[data-config-download]").addEventListener("click", () => {
        if (!resolved) return;
        const url = URL.createObjectURL(new Blob([resolved.yaml], {type:"text/yaml;charset=utf-8"}));
        const link = document.createElement("a");
        link.href = url;
        link.download = "config.yaml";
        link.hidden = true;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        status.textContent = "config.yaml download started.";
      });
      await load(picker.value);
    } catch (failure) { status.textContent = ""; invalidate(failure.message); }
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
