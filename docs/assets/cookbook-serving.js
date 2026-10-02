/**
 * Render cookbook controls and generate commands from shared and model-specific metadata.
 *
 * Data flow:
 *   defaults.yaml + hardware.yaml + models/*.yaml
 *     -> docs/cookbook_serving.py publishes cookbook-serving-example.json
 *     -> this script loads that JSON and applies the visitor's selections
 *     -> the page displays server, health-check, and client commands.
 *
 * Start reading at resolveRecipe(): it contains the metadata-to-command logic and
 * can run without a browser. mount() connects that function to the page: it loads
 * the JSON once, builds controls, and resolves again whenever an input changes.
 * The remaining helpers write nested settings, validate values, and format CLI text.
 *
 * This script only produces configuration objects and command strings. It never
 * starts an inference server, loads model weights, or sends a generation request.
 */
((scope) => {
  "use strict";

  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  /** Quote one shell argument so spaces, apostrophes, and shell operators remain literal text. */
  const shellQuote = (value) => {
    const text = String(value);
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(text) ? text : `'${text.replace(/'/g, `'"'"'`)}'`;
  };

  /**
   * Write a dotted configuration key into the mutable copy of the recipe settings.
   * For example, setPath(config, "server.port", 9000) writes config.server.port = 9000.
   * Missing parent objects are created; malformed paths and scalar parents are rejected.
   */
  function setPath(config, path, value) {
    const parts = path.split(".");
    if (!/^(generator|server)\./.test(path) ||
        parts.some((part) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(part) ||
          ["__proto__", "constructor", "prototype"].includes(part))) {
      throw new Error(`Invalid serving configuration path: ${path}`);
    }
    let cursor = config;
    for (const part of parts.slice(0, -1)) {
      if (!own(cursor, part)) cursor[part] = {};
      if (cursor[part] === null || typeof cursor[part] !== "object" || Array.isArray(cursor[part])) {
        throw new Error(`Cannot apply nested option to ${path}`);
      }
      cursor = cursor[part];
    }
    cursor[parts.at(-1)] = value;
  }

  /**
   * Check a value against the merged option's type, choices, and numeric bounds.
   * Throws on invalid input. This checks cookbook rules; the serving CLI still
   * validates the resulting configuration against its own runtime schema.
   */
  function validateValue(key, definition, value) {
    const validType = {
      integer: Number.isSafeInteger(value),
      number: typeof value === "number" && Number.isFinite(value),
      boolean: typeof value === "boolean",
      string: typeof value === "string",
    }[definition.type];
    if (!validType) throw new Error(`${key} must be ${definition.type}`);
    if (definition.choices && !definition.choices.includes(value)) {
      throw new Error(`${key} must be one of: ${definition.choices.join(", ")}`);
    }
    if (definition.min !== undefined && value < definition.min) {
      throw new Error(`${key} must be at least ${definition.min}`);
    }
    if (definition.max !== undefined && value > definition.max) {
      throw new Error(`${key} must be at most ${definition.max}`);
    }
  }

  /**
   * Flatten a nested configuration into alternating flag/value strings, without shell quoting.
   * Example: {server: {port: 9000}} becomes ["--server.port", "9000"].
   * Arrays and empty mappings stay single JSON values; false, null, and empty strings are kept.
   * This prototype emits every configured leaf, including values that match runtime defaults.
   */
  function configArguments(config, prefix = "") {
    return Object.entries(config).flatMap(([key, value]) => {
      const path = prefix ? `${prefix}.${key}` : key;
      if (value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length) {
        return configArguments(value, path);
      }
      return [`--${path}`, typeof value === "string" ? value : JSON.stringify(value)];
    });
  }

  /**
   * Resolve one model's metadata and current selections into everything the page displays.
   * Does not mutate the input metadata or access the DOM/network.
   *
   * @param {Object} bundle Generated JSON: shared defaults, hardware catalog, and model recipes.
   * @param {string} recipeId Model recipe ID, such as "fasth3-v2" or "zimage-turbo".
   * @param {Object} selections Optional hardware ID and deployment option values.
   *   Example: {hardware: "gb200", values: {port: 9000}}.
   * @returns {Object} options for rendering controls; config for inspection; argv and command
   *   for launching the server; installCommand, healthCommand, clientRequest, and clientCommand.
   * @throws {Error} When a selection, default, configuration path, or static example request is invalid.
   *
   * Short return-value examples (other fields and command flags omitted):
   *   options: [{key: "port", type: "integer", value: 9000, source: "user"}]
   *   config: {server: {port: 9000}}
   *   argv: ["fastvideo", "serve", "--server.port", "9000", ...]
   *   command: "fastvideo serve --server.port 9000 ..."
   *   installCommand: 'UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"'
   *   healthCommand: "curl --fail-with-body http://127.0.0.1:9000/health"
   *   clientRequest: {
   *     endpoint: "http://127.0.0.1:9000/v1/videos/sync",
   *     body: {model: "fasth3", prompt: "A fox in snow"}, outputFile: "output.mp4"
   *   }
   *   clientCommand: "curl --request POST http://127.0.0.1:9000/v1/videos/sync ... --output output.mp4"
   *
   * Resolution order:
   * 1. Select the recipe, compatible GPU type, runtime, and task.
   * 2. Merge common option definitions with model overrides; model choice lists replace common lists.
   * 3. Choose each value: common default -> model default -> explicit user selection.
   * 4. Validate and write values into a copy of recipe.settings. also_set updates linked paths,
   *    such as keeping sequence parallelism equal to the selected GPU count.
   * 5. Serialize the deployment config. Attach its endpoint/model alias to the static example request.
   */
  function resolveRecipe(bundle, recipeId, selections = {}) {
    const recipe = bundle.recipes.find((item) => item.id === recipeId);
    if (!recipe) throw new Error(`Unknown recipe: ${recipeId}`);
    if (Object.keys(selections).some((key) => !["hardware", "values"].includes(key))) {
      throw new Error("Selections must contain only hardware and values");
    }
    const hardwareId = selections.hardware ?? recipe.default_hardware;
    const hardware = bundle.hardware[hardwareId];
    if (!recipe.hardware.includes(hardwareId) || !hardware || hardware.runtime !== recipe.runtime) {
      throw new Error(`Unsupported hardware: ${hardwareId}`);
    }
    const runtime = bundle.defaults.runtimes[recipe.runtime];
    const task = bundle.defaults.tasks[recipe.task];
    if (!runtime || !task || !["video", "image"].includes(task.client)) {
      throw new Error("Unsupported runtime or task");
    }
    const overrides = recipe.options || {};
    const keys = [...new Set([...task.options, ...Object.keys(overrides)])]
      .filter((key) => overrides[key]?.supported !== false);
    const values = selections.values || {};
    for (const key of Object.keys(values)) {
      if (!keys.includes(key)) throw new Error(`Unknown or unsupported option: ${key}`);
    }

    const config = clone(recipe.settings || {});
    if (Object.keys(config).some((key) => !["generator", "server"].includes(key))) {
      throw new Error("Cookbook settings must contain only generator and server configuration");
    }
    setPath(config, "generator.model_path", recipe.model_id);
    const options = keys.map((key) => {
      const common = bundle.defaults.options[key] || {};
      const model = overrides[key] || {};
      // Override individual option properties; omitted properties stay shared, lists replace.
      // This is flat option metadata, not recursive inheritance of runtime configuration trees.
      const definition = { ...common, ...model };
      let value = common.default;
      let defaultSource = "shared";
      if (own(model, "default")) {
        value = model.default;
        defaultSource = "model";
      }
      // Validate defaults even when a user selection would otherwise conceal an invalid recipe.
      validateValue(key, definition, value);
      definition.default = value;
      const source = own(values, key) ? "user" : defaultSource;
      if (source === "user") value = values[key];
      validateValue(key, definition, value);
      setPath(config, definition.path, value);
      for (const path of definition.also_set || []) setPath(config, path, value);
      return { key, ...definition, value, source, defaultSource, sharedDefault: common.default };
    });

    const args = configArguments(config);
    const command = [runtime.command.map(shellQuote).join(" "),
      ...Array.from({ length: args.length / 2 }, (_, index) =>
        `${args[index * 2]} ${shellQuote(args[index * 2 + 1])}`)].join(" \\\n  ");
    let host = config.server.host;
    if (typeof host !== "string" || !/^[A-Za-z0-9_.:-]+$/.test(host)) {
      throw new Error("host must be a nonempty hostname or IP address, without spaces or URL paths");
    }
    // The CLI casts scalar-looking tokens even when the shell quotes them.
    if (/^(true|false|null|none|nan|[+-]?inf(inity)?)$/i.test(host) || Number.isFinite(Number(host))) {
      throw new Error("host must be a hostname or IP address that the CLI can preserve as text");
    }
    if (["0.0.0.0", "::"].includes(host)) host = "127.0.0.1";
    if (host.includes(":") && !host.startsWith("[")) host = `[${host}]`;
    const endpoint = `http://${host}:${config.server.port}`;
    // Request parameters belong to this optional example, never the deployment controls or flags.
    const body = clone(recipe.example_request || {});
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      throw new Error("example_request must be an object");
    }
    body.model = config.server.served_model_name || recipe.model_id;
    if (typeof body.prompt !== "string" || !body.prompt.trim()) {
      throw new Error("example_request.prompt must be a nonempty string");
    }
    if (task.requires_image && (typeof body.input_reference !== "string" || !body.input_reference.trim())) {
      throw new Error("example_request.input_reference must be a nonempty server-local path or URL");
    }
    const isImage = task.client === "image";
    if (isImage) Object.assign(body, { response_format: "b64_json", output_format: "png" });
    const clientRequest = {
      endpoint: `${endpoint}${isImage ? "/v1/images/generations" : "/v1/videos/sync"}`,
      body,
      outputFile: isImage ? "output.png" : "output.mp4",
    };
    let clientCommand = `curl --fail-with-body --request POST ${shellQuote(clientRequest.endpoint)} \\\n  --header 'Content-Type: application/json' \\\n  --data ${shellQuote(JSON.stringify(body))} \\\n  --output ${isImage ? "response.json" : "output.mp4"}`;
    if (isImage) {
      const decode = 'import base64,json,pathlib; response=json.loads(pathlib.Path("response.json").read_text()); ' +
        'pathlib.Path("output.png").write_bytes(base64.b64decode(response["data"][0]["b64_json"]))';
      clientCommand += ` && \\\npython -c ${shellQuote(decode)}`;
    }
    return {
      recipe, task, hardware, options, config, argv: [...runtime.command, ...args], command,
      installCommand: recipe.install,
      healthCommand: `curl --fail-with-body ${shellQuote(`${endpoint}/health`)}`,
      clientCommand, clientRequest,
    };
  }

  // Share the same resolver with browser code and Node tests; Node does not mount a page.
  const api = { resolveRecipe, shellQuote, configArguments };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  scope.FastVideoCookbook = api;
  if (typeof document === "undefined") return;

  /**
   * Initialize one cookbook element using its data-metadata URL and data-recipe default ID.
   * Fetch metadata once, then keep deployment option values in local UI state.
   * Input changes call render(); model changes rebuild controls with the new model's defaults.
   * No model-specific controls are hardcoded here: option definitions determine their widgets.
   *
   * TODO(cookbook-ui): Replace this demo UI adapter (mount, its nested handlers, and init)
   * with the new UI components. Retain or extract resolveRecipe and its validation/CLI
   * helpers, together with their tests, so the new UI reuses the same resolution logic.
   *
   * @param {HTMLElement} root The page element marked with data-serving-example.
   */
  async function mount(root) {
    const error = root.querySelector("[data-serving-error]");
    const output = root.querySelector("[data-serving-output]");
    try {
      const response = await fetch(root.dataset.metadata);
      if (!response.ok) throw new Error(`Could not load metadata: HTTP ${response.status}`);
      const bundle = await response.json();
      let recipeId = root.dataset.recipe;
      let values = {};
      let selectedHardware;
      const controls = root.querySelector("[data-serving-controls]");
      let origins = {};

      /** Resolve current UI state and update all command blocks together; hide output on invalid input. */
      function render() {
        try {
          const result = resolveRecipe(bundle, recipeId, { hardware: selectedHardware, values });
          error.textContent = "";
          output.hidden = false;
          root.querySelector("[data-serving-context]").textContent =
            `${result.task.label} · ${result.recipe.runtime.toUpperCase()} · ${result.hardware.label}`;
          for (const option of result.options) {
            const inherited = option.sharedDefault === undefined ? "Model-specific option; " :
              option.defaultSource !== "shared" ? `Shared default: ${option.sharedDefault} → ` : "";
            origins[option.key].textContent =
              `${inherited}default: ${option.default} (${option.defaultSource}); current: ${option.source}`;
          }
          for (const key of ["installCommand", "command", "healthCommand", "clientCommand"]) {
            root.querySelector(`[data-serving-code="${key}"]`).textContent = result[key];
          }
          root.querySelector("[data-serving-config]").textContent = JSON.stringify(result.config, null, 2);
          root.querySelector("[data-serving-client-note]").textContent =
            "Copy this static example after the server is ready; edit its prompt or request values in your terminal. " +
            (result.task.requires_image ?
              "Replace /path/to/first-frame.png with an image path on the server or a URL it can access. " : "") +
            (result.task.client === "image" ?
              "The command saves response.json and decodes its PNG to output.png." :
              "The synchronous endpoint returns MP4 bytes to output.mp4.");
        } catch (failure) {
          error.textContent = failure.message;
          output.hidden = true;
        }
      }

      /**
       * Reset selections when the model changes and build its controls from resolved definitions:
       * singleton choices -> fixed text; multiple choices -> dropdown; other types -> input widgets.
       */
      function buildControls() {
        values = {};
        origins = {};
        const initial = resolveRecipe(bundle, recipeId);
        selectedHardware = initial.recipe.default_hardware;
        controls.replaceChildren();
        root.querySelector("[data-serving-client-details]").removeAttribute("open");
        const hardwareRow = document.createElement("label");
        hardwareRow.className = "serving-example-option";
        const hardwareTitle = document.createElement("strong");
        hardwareTitle.textContent = "GPU type";
        const hardwareSelect = document.createElement("select");
        hardwareSelect.setAttribute("aria-label", "GPU type");
        hardwareSelect.dataset.servingHardware = "";
        initial.recipe.hardware.forEach((id) => {
          const item = document.createElement("option");
          item.value = id;
          item.textContent = bundle.hardware[id].label;
          item.selected = id === selectedHardware;
          hardwareSelect.append(item);
        });
        hardwareSelect.addEventListener("change", () => { selectedHardware = hardwareSelect.value; render(); });
        const hardwareHelp = document.createElement("small");
        hardwareHelp.textContent = "Device type; GPU count is a separate model option.";
        hardwareRow.append(hardwareTitle, hardwareSelect, hardwareHelp);

        for (const option of initial.options) {
          if (option.key === "num_gpus") controls.append(hardwareRow);
          const row = document.createElement("label");
          row.className = "serving-example-option";
          const title = document.createElement("strong");
          title.textContent = option.label;
          let control;
          if (option.choices?.length === 1) {
            control = document.createElement("output");
            control.textContent = `${option.value} (fixed for this recipe)`;
          } else if (option.choices) {
            control = document.createElement("select");
            option.choices.forEach((value, index) => {
              const item = document.createElement("option");
              item.value = index;
              item.textContent = value;
              item.selected = value === option.value;
              control.append(item);
            });
            control.addEventListener("change", () => { values[option.key] = option.choices[control.value]; render(); });
          } else {
            control = document.createElement("input");
            control.type = option.type === "boolean" ? "checkbox" : option.type === "string" ? "text" : "number";
            if (option.type === "boolean") control.checked = option.value;
            else control.value = option.value;
            if (option.min !== undefined) control.min = option.min;
            if (option.max !== undefined) control.max = option.max;
            if (option.type === "integer") control.step = "1";
            if (option.type === "number") control.step = "any";
            control.addEventListener("input", () => {
              values[option.key] = option.type === "boolean" ? control.checked : option.type === "string" ?
                control.value : control.value === "" ? NaN : Number(control.value);
              render();
            });
          }
          control.setAttribute("aria-label", option.label);
          control.dataset.option = option.key;
          const origin = document.createElement("small");
          origins[option.key] = origin;
          row.append(title, control, origin);
          if (option.help) {
            const help = document.createElement("small");
            help.textContent = option.help;
            row.append(help);
          }
          controls.append(row);
        }
        render();
      }

      const modelSelect = root.querySelector("[data-serving-model]");
      bundle.recipes.forEach((recipe) => {
        const item = document.createElement("option");
        item.value = recipe.id;
        item.textContent = recipe.label;
        item.selected = recipe.id === recipeId;
        modelSelect.append(item);
      });
      modelSelect.addEventListener("change", () => { recipeId = modelSelect.value; buildControls(); });
      root.querySelectorAll("[data-serving-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(root.querySelector(`[data-serving-code="${button.dataset.servingCopy}"]`).textContent);
            button.textContent = "Copied";
          } catch (_) {
            button.textContent = "Select and copy the text below";
          }
        });
      });
      buildControls();
    } catch (failure) {
      error.textContent = failure.message;
      output.hidden = true;
    }
  }

  // Run on the initial page load and MkDocs instant navigation, mounting each element only once.
  const init = () => document.querySelectorAll("[data-serving-example]").forEach((root) => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = "true";
    mount(root);
  });
  if (scope.document$) scope.document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof globalThis !== "undefined" ? globalThis : window);
