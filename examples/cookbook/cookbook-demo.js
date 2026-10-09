/** Temporary cookbook UI. The reusable data/configuration API lives in cookbook-config.js. */
((scope) => {
  "use strict";

  const cookbook = typeof module !== "undefined" && module.exports ?
    require("../../docs/assets/cookbook-config.js") : scope.FastVideoConfigCookbook;
  const pathParts = (path) => path.split(".");

  /** The demo discards obsolete catalog responses when its selection changes. */
  function createCatalogLoader(index, onChange, fetcher = (...args) => scope.fetch(...args)) {
    const entries = index.models.flatMap((model) => model.deployments.map((deployment) => ({ model, deployment })));
    let generation = 0, controller;
    return async (id) => {
      const token = ++generation;
      controller?.abort();
      controller = new AbortController();
      const entry = entries.find((item) => item.deployment.id === id);
      onChange({ state: "loading", ...entry });
      try {
        const catalog = await cookbook.loadDeployment(index, id, { signal: controller.signal, fetcher });
        if (token === generation) onChange({ state: "ready", catalog });
      } catch (failure) {
        if (token === generation) onChange({ state: "error", message: failure.message });
      }
    };
  }

  function controlPresentation(control) {
    const parts = pathParts(control.path);
    const groups = { server: "Server", generator: "Resources", default_request: "Request defaults" };
    return { label: control.schema?.title || parts.at(-1).replace(/_/g, " "), group: groups[parts[0]] };
  }

  function guideUrl(catalog, indexUrl) {
    return catalog.guide?.url ? new URL(catalog.guide.url, indexUrl).href : null;
  }

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
    const requirementsSection = root.querySelector("[data-config-requirements-section]");
    const guide = root.querySelector("[data-config-guide]");
    const streamingClient = root.querySelector("[data-config-streaming-client]");
    const websocketUrl = root.querySelector("[data-config-websocket-url]");
    const clientGuide = root.querySelector("[data-config-client-guide]");
    let catalog, resolved, selections = {};
    const inputErrors = new Map();
    const fieldInputs = new Map();

    function updateControlLimits() {
      for (const { path, schema } of cookbook.getOptions(catalog, catalog.base_config, selections)) {
        const input = fieldInputs.get(path);
        if (input?.type !== "number") continue;
        const branches = schema.anyOf || schema.oneOf || [schema];
        const nonNull = branches.filter((item) => item.type !== "null");
        const basic = nonNull.length === 1 ? nonNull[0] : schema;
        for (const [property, keyword] of [["min", "minimum"], ["max", "maximum"]]) {
          const bound = schema[keyword] ?? basic[keyword];
          if (bound === undefined) input.removeAttribute(property);
          else input[property] = bound;
        }
      }
    }

    function clearMetadata() {
      requirementsSection.hidden = true;
      root.querySelector("[data-config-requirements]").replaceChildren();
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
    function invalidate(message) {
      resolved = null;
      output.hidden = true;
      buttons.forEach((button) => { button.disabled = true; });
      error.textContent = message;
      streamingClient.hidden = true;
      websocketUrl.textContent = "";
    }
    function render() {
      status.textContent = "";
      try {
        updateControlLimits();
        if (inputErrors.size) throw new Error([...inputErrors.values()].join("; "));
        resolved = cookbook.resolveConfig(catalog, selections);
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
      } catch (failure) { invalidate(failure.message); }
    }

    function renderControls() {
      controls.replaceChildren();
      fieldInputs.clear();
      const sections = new Map();
      for (const control of cookbook.getOptions(catalog)) {
        const { path, schema: field, value } = control;
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
        fieldInputs.set(path, input);
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
      const index = await cookbook.loadIndex(root.dataset.metadata);
      const indexUrl = root.dataset.documentationIndex || index.url;
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
      const load = createCatalogLoader(index, (update) => {
        if (update.state === "loading") {
          catalog = null;
          selections = {};
          inputErrors.clear();
          controls.replaceChildren();
          fieldInputs.clear();
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
          requirementsSection.hidden = false;
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

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { mount, createCatalogLoader };
  }
  scope.FastVideoCookbookDemo = { mount };
})(typeof globalThis !== "undefined" ? globalThis : window);
