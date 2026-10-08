/**
 * Recipe-backed cookbook demo: native serving baseline -> selected controls -> config.yaml.
 *
 * Python exports a model/deployment index and one baseline-preserving catalog per deployment.
 * Only explicitly selected controls are editable. Their JSON Schema metadata
 * comes from public configuration declarations; Ajv validates edited values.
 * Labels and grouping live here.
 * These helpers do not start a server, load a model, or submit an inference request.
 */
((scope) => {
  "use strict";

  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));

  const pathParts = (path) => path.split(".");

  /** Fetch only the selected catalog; obsolete responses and failures never reach the form. */
  function createCatalogLoader(index, indexUrl, onChange, fetcher = (...args) => scope.fetch(...args)) {
    const entries = index.models.flatMap((model) => model.deployments.map((deployment) => ({ model, deployment })));
    let generation = 0, controller;
    return async (deploymentId) => {
      const token = ++generation;
      controller?.abort();
      controller = new AbortController();
      const entry = entries.find((item) => item.deployment.id === deploymentId);
      onChange({ state: "loading", ...entry });
      try {
        if (!entry) throw new Error(`No exported deployment: ${deploymentId}`);
        const response = await fetcher(new URL(entry.deployment.catalog_url, indexUrl).href, { signal: controller.signal });
        if (token !== generation) return;
        if (!response.ok) throw new Error(`Could not load deployment: HTTP ${response.status}`);
        const catalog = await response.json();
        if (token !== generation) return;
        if (catalog.id !== deploymentId) throw new Error(`Unexpected deployment: ${catalog.id}`);
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

  function controlPresentation(control) {
    const parts = pathParts(control.path);
    const labels = {
      "server.host": "Bind address", "server.port": "Port",
      "server.served_model_name": "Served model name", "server.output_dir": "Output directory",
      "generator.engine.compile.enabled": "Compile transformer", "generator.engine.offload.vae": "Offload VAE",
      "generator.engine.offload.dit_layerwise": "Offload transformer layers",
      "generator.engine.offload.text_encoder": "Offload text encoder",
      "default_request.sampling.num_frames": "Frames", "default_request.sampling.height": "Height",
      "default_request.sampling.width": "Width", "default_request.sampling.fps": "FPS",
      "default_request.sampling.seed": "Seed",
      "generator.model_root": "Model directory", "generator.mlx_checkpoint": "Converted MLX checkpoint",
      "generator.prompt_cache_dir": "Prompt cache directory", "generator.vae_dtype": "VAE precision",
      "generator.vsa_sparsity": "VSA sparsity",
    };
    const groups = { server: "Server", generator: "Resources", default_request: "Request defaults" };
    return { label: labels[control.path] || parts.at(-1).replace(/_/g, " "), group: groups[parts[0]] };
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

  /** Recorded hardware evidence describes only the complete original recipe baseline. */
  function hardwareView(catalog, config) {
    const isBaseline = JSON.stringify(config) === JSON.stringify(catalog.base_config);
    return {
      isBaseline,
      message: isBaseline ? "Recipe baseline — hardware records apply to this configuration." :
        "Custom configuration — unverified. Records below apply only to the original recipe baseline.",
      records: (catalog.hardware || []).map((item) => ({ ...item, status: item.status || "unverified" })),
    };
  }

  function guideUrl(catalog, indexUrl) {
    return catalog.guide?.url ? new URL(catalog.guide.url, indexUrl).href : null;
  }

  const api = { resolveConfig, createCatalogLoader, hardwareView, guideUrl, toYaml, shellQuote };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  scope.FastVideoConfigCookbook = api;
  if (typeof document === "undefined") return;

  // TODO(cookbook-ui): Replace this demo DOM adapter; keep schema selection and YAML generation.
  async function mount(root) {
    const status = root.querySelector("[data-config-status]");
    const error = root.querySelector("[data-config-error]");
    const output = root.querySelector("[data-config-output]");
    const buttons = root.querySelectorAll("[data-config-copy], [data-config-download]");
    const modelPicker = root.querySelector("[data-config-model-picker]");
    const deploymentPicker = root.querySelector("[data-config-deployment-picker]");
    const controls = root.querySelector("[data-config-controls]");
    const reset = root.querySelector("[data-config-reset]");
    const hardware = root.querySelector("[data-config-hardware]");
    const hardwareState = root.querySelector("[data-config-hardware-state]");
    const hardwareRows = root.querySelector("[data-config-hardware-rows]");
    const guide = root.querySelector("[data-config-guide]");
    const streamingClient = root.querySelector("[data-config-streaming-client]");
    const websocketUrl = root.querySelector("[data-config-websocket-url]");
    const clientGuide = root.querySelector("[data-config-client-guide]");
    let catalog, resolved, selections = {};
    const inputErrors = new Map();

    function clearMetadata() {
      hardware.hidden = true;
      hardwareState.textContent = "";
      hardwareRows.replaceChildren();
      guide.hidden = true;
      guide.removeAttribute("href");
      streamingClient.hidden = true;
      websocketUrl.textContent = "";
      clientGuide.removeAttribute("href");
      root.querySelector("[data-config-request-note]").hidden = true;
      root.querySelector("[data-config-summary]").textContent = "";
      root.querySelector("[data-config-topology]").textContent = "";
      root.querySelector("[data-config-source]").removeAttribute("href");
    }
    function renderHardware() {
      const view = hardwareView(catalog, resolved.config);
      hardware.hidden = false;
      hardwareState.textContent = !view.isBaseline || view.records.length ? view.message :
        "Unverified — no hardware records published.";
      root.querySelector("[data-config-hardware-table]").hidden = !view.records.length;
      root.querySelector("[data-config-hardware-note]").hidden = !view.records.length;
      hardwareRows.replaceChildren();
      for (const item of view.records) {
        const row = document.createElement("tr");
        const gpu = document.createElement("td");
        const source = document.createElement("a");
        source.textContent = item.label;
        source.href = item.source_url;
        gpu.append(source);
        const memory = document.createElement("td");
        memory.textContent = `${item.memory_gb} GB`;
        const recordedStatus = document.createElement("td");
        recordedStatus.textContent = { verified: "Verified", unverified: "Unverified", unsupported: "Unsupported" }[item.status];
        const record = document.createElement("td");
        if (item.evidence_url) {
          const evidence = document.createElement("a");
          evidence.textContent = "Baseline evidence";
          evidence.href = item.evidence_url;
          record.append(evidence);
        }
        if (item.reason) record.append(`${item.evidence_url ? " — " : ""}${item.reason}`);
        if (!item.evidence_url && !item.reason) record.textContent = "—";
        row.append(gpu, memory, recordedStatus, record);
        hardwareRows.append(row);
      }
    }
    function invalidate(message) {
      resolved = null;
      output.hidden = true;
      buttons.forEach((button) => { button.disabled = true; });
      error.textContent = message;
      streamingClient.hidden = true;
      websocketUrl.textContent = "";
      if (catalog) hardwareState.textContent = "Configuration incomplete — hardware status unavailable.";
    }
    function render() {
      status.textContent = "";
      try {
        if (inputErrors.size) throw new Error([...inputErrors.values()].join("; "));
        resolved = resolveConfig(catalog, selections);
        error.textContent = "";
        output.hidden = false;
        buttons.forEach((button) => { button.disabled = false; });
        for (const key of ["yaml", "command", "clientCommand"]) {
          root.querySelector(`[data-config-code="${key}"]`).textContent = resolved[key];
        }
        const streaming = Boolean(resolved.websocketUrl);
        root.querySelector("[data-config-client-title]").textContent = streaming ?
          "4. Check server and connect a streaming client" : "4. Send a sample request";
        const clientCopy = root.querySelector('[data-config-copy="clientCommand"]');
        clientCopy.textContent = streaming ? "Copy health check" : "Copy sample request";
        clientCopy.dataset.copyLabel = streaming ? "Health check" : "Sample request";
        root.querySelector("[data-config-client-note]").textContent = streaming ?
          "Run this health check, then connect a compatible WebSocket client using the contract below. Use the server machine’s address from another computer." :
          catalog.workload === "i2v" ? "Same-machine example: replace the image path with a file accessible to the server." :
            "Same-machine example. Use the server machine’s address when calling from another computer.";
        streamingClient.hidden = !streaming;
        websocketUrl.textContent = resolved.websocketUrl || "";
        root.querySelector("[data-config-request-note]").hidden = streaming;
        renderHardware();
      } catch (failure) { invalidate(failure.message); }
    }

    function renderControls() {
      controls.replaceChildren();
      const sections = new Map();
      for (const control of catalog.controls) {
        const { path, schema: field } = control;
        const pointer = path;
        const presentation = controlPresentation(control);
        const sectionName = presentation.group;
        if (!sections.has(sectionName)) {
          const section = document.createElement("details");
          section.className = "config-builder-section";
          section.open = sectionName === "Server";
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
        title.textContent = presentation.label;
        const address = document.createElement("small");
        address.textContent = path;
        const value = getValue(catalog.base_config, path);
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
            option.textContent = item === null ? "null (explicit)" : String(item);
            input.append(option);
          });
          if (value === undefined) {
            const inherited = document.createElement("option");
            inherited.value = ""; inherited.textContent = "Inherited"; input.prepend(inherited);
          }
          input.value = value === undefined ? "" : choices.findIndex((item) => item === value);
          input.addEventListener("change", () => {
            if (input.value === "") delete selections[pointer];
            else selections[pointer] = choices[input.value];
            render();
          });
        } else {
          const scalar = ["string", "integer", "number"].includes(type);
          input = document.createElement(scalar ? "input" : "textarea");
          if (scalar) {
            input.type = type === "string" ? "text" : "number";
            if (basic.minimum !== undefined) input.min = basic.minimum;
            if (basic.maximum !== undefined) input.max = basic.maximum;
            input.step = basic.multipleOf || (type === "integer" ? "1" : "any");
            input.value = value == null ? "" : value;
            if (value === undefined) input.placeholder = "Inherited";
          } else {
            input.rows = 3;
            input.placeholder = "JSON value";
            input.value = value === undefined ? "" : JSON.stringify(value, null, 2);
          }
          const update = () => {
            try {
              if (automatic?.checked) selections[pointer] = null;
              else if (scalar && input.value === "" && value === undefined) delete selections[pointer];
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
            automatic.setAttribute("aria-label", `${path}: Set null`);
            input.disabled = automatic.checked;
            automatic.addEventListener("change", () => { input.disabled = automatic.checked; update(); });
            const autoLabel = document.createElement("span");
            autoLabel.append(automatic, " Set null explicitly");
            row.append(autoLabel);
          }
          input.addEventListener("input", update);
        }
        input.dataset.configPath = pointer;
        input.setAttribute("aria-label", path);
        const initial = document.createElement("small");
        initial.textContent = value === undefined ? "Inherited / unset" : `Recipe value: ${JSON.stringify(value)}`;
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
      if (!response.ok) throw new Error(`Could not load recipe index: HTTP ${response.status}`);
      const indexUrl = response.url;
      const index = await response.json();
      if (!index.models.length) throw new Error("No models are published");
      for (const model of index.models) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.title;
        modelPicker.append(option);
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
          clearMetadata();
          reset.disabled = true;
          invalidate("");
          root.querySelector("[data-config-model]").textContent = update.model ?
            `${update.model.title} · ${update.deployment.label}` : modelPicker.value;
          status.textContent = "Loading the selected deployment…";
        } else if (update.state === "ready") {
          catalog = update.catalog;
          root.querySelector("[data-config-model]").textContent =
            `${catalog.model.title} · ${catalog.deployment.label} · ${catalog.workload}`;
          root.querySelector("[data-config-summary]").textContent = catalog.summary;
          root.querySelector("[data-config-source]").href = `https://github.com/hao-ai-lab/FastVideo/blob/main/${catalog.source_config}`;
          const guideHref = guideUrl(catalog, indexUrl);
          guide.hidden = !guideHref;
          if (guideHref) guide.href = guideHref;
          if (catalog.runtime.client_guide_url) {
            clientGuide.href = new URL(catalog.runtime.client_guide_url, indexUrl).href;
          }
          const requirements = root.querySelector("[data-config-requirements]");
          requirements.replaceChildren();
          for (const note of catalog.requirements) {
            const item = document.createElement("li"); item.textContent = note; requirements.append(item);
          }
          const installation = root.querySelector("[data-config-install]");
          installation.href = catalog.runtime.install_url;
          installation.textContent = `${catalog.runtime.label} installation guide`;
          root.querySelector("[data-config-topology]").textContent =
            catalog.runtime.hardware_label ? `Hardware family: ${catalog.runtime.hardware_label}.` :
              `Recipe baseline GPU count: ${catalog.base_config.generator.engine?.num_gpus ?? "inherited"}.`;
          resetForm();
          reset.disabled = false;
        } else {
          status.textContent = "";
          clearMetadata();
          invalidate(update.message);
        }
      });
      const selectModel = () => {
        const model = index.models.find((item) => item.id === modelPicker.value);
        deploymentPicker.replaceChildren();
        for (const deployment of model.deployments) {
          const option = document.createElement("option");
          option.value = deployment.id;
          option.textContent = deployment.label;
          deploymentPicker.append(option);
        }
        deploymentPicker.value = model.deployments[0]?.id || "";
        deploymentPicker.disabled = !model.deployments.length;
        return load(deploymentPicker.value);
      };
      modelPicker.value = index.models[0].id;
      modelPicker.disabled = false;
      modelPicker.addEventListener("change", selectModel);
      deploymentPicker.addEventListener("change", () => load(deploymentPicker.value));
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
      await selectModel();
    } catch (failure) { status.textContent = ""; clearMetadata(); invalidate(failure.message); }
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
