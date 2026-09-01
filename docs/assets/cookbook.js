(() => {
  const PATTERN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const PATTERN_LENGTH = 900;
  const SCRAMBLE_MS = 140;

  const loadRecipes = (url) =>
    fetch(url).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });

  const generatePattern = (length) => {
    const chars = new Array(length);
    for (let index = 0; index < length; index += 1) {
      chars[index] = PATTERN_CHARS.charAt((Math.random() * PATTERN_CHARS.length) | 0);
    }
    return chars.join("");
  };

  const motionQuery = () =>
    window.matchMedia("(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)");

  const initEvervault = (root) => {
    root.querySelectorAll("[data-evervault]").forEach((visual) => {
      if (visual.dataset.evervaultReady) return;
      visual.dataset.evervaultReady = "true";
      const noise = visual.querySelector("[data-cookbook-pattern]");
      if (!noise) return;

      const query = motionQuery();
      if (!query.matches) return;

      let rect = null;
      let pointerX = 0;
      let pointerY = 0;
      let frame = 0;
      let lastScramble = 0;
      let patternReady = false;

      const paint = (scramble) => {
        visual.style.setProperty("--mouse-x", `${pointerX}px`);
        visual.style.setProperty("--mouse-y", `${pointerY}px`);
        if (scramble) noise.textContent = generatePattern(PATTERN_LENGTH);
        frame = 0;
      };

      const onEnter = () => {
        rect = visual.getBoundingClientRect();
        if (!patternReady) {
          noise.textContent = generatePattern(PATTERN_LENGTH);
          patternReady = true;
        }
      };
      const onMove = (event) => {
        if (!rect) rect = visual.getBoundingClientRect();
        pointerX = event.clientX - rect.left;
        pointerY = event.clientY - rect.top;
        if (frame) return;
        frame = window.requestAnimationFrame(() => {
          const now = performance.now();
          const scramble = now - lastScramble >= SCRAMBLE_MS;
          if (scramble) lastScramble = now;
          paint(scramble);
        });
      };
      const onLeave = () => {
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        rect = null;
        pointerX = 0;
        pointerY = 0;
        visual.style.setProperty("--mouse-x", "50%");
        visual.style.setProperty("--mouse-y", "50%");
      };
      visual.addEventListener("mouseenter", onEnter);
      visual.addEventListener("mousemove", onMove);
      visual.addEventListener("mouseleave", onLeave);
    });
  };

  let familyPopstate = null;
  const bindFamilyPopstate = () => {
    if (bindFamilyPopstate.bound) return;
    bindFamilyPopstate.bound = true;
    window.addEventListener("popstate", () => {
      if (typeof familyPopstate === "function") familyPopstate();
    });
  };

  const groupIdFor = (recipe) => recipe.group || recipe.id;

  const gpuCountLabel = (hardware) => {
    const count = hardware?.gpu_count;
    return `${count} GPU${count === 1 ? "" : "s"}`;
  };

  const knobsFor = (recipe) => recipe.knobs || [];

  const knobOptions = (knob) =>
    knob.options.map((option) =>
      option !== null && typeof option === "object" ? option : { value: option, label: String(option) },
    );

  const knobDefaultLabel = (knob) => {
    const match = knobOptions(knob).find((option) => option.value === knob.default);
    return match ? match.label : String(knob.default);
  };

  // Knob flags are always shown explicitly in the displayed command, even at
  // their default value, so the command stays copy-pasteable and precise
  // about what it runs -- not just "trust the script's own default".
  const appendKnobFlags = (commandText, knobs, knobValues) => {
    const flags = knobs
      .filter((knob) => knobValues[knob.key] !== undefined)
      .map((knob) => `${knob.flag} ${knobValues[knob.key]}`);
    if (!flags.length) return commandText;
    const lines = commandText.split("\n");
    lines[lines.length - 1] = `${lines[lines.length - 1]} ${flags.join(" ")}`;
    return lines.join("\n");
  };

  const runtimeFor = (recipe) => {
    const platform = recipe.hardware?.platform || "cuda";
    if (platform === "mlx") {
      return {
        id: "mlx",
        label: "Apple Silicon · MLX",
        hint:
          recipe.hardware?.minimum_memory ||
          [recipe.hardware?.accelerator, recipe.hardware?.system_memory].filter(Boolean).join(" · ") ||
          "Memory not recorded",
      };
    }
    if (platform === "mps") {
      return {
        id: "mps",
        label: "Apple Silicon · MPS",
        hint: recipe.hardware?.minimum_memory || recipe.hardware?.system_memory || "Memory not recorded",
      };
    }
    const hardware = recipe.hardware || {};
    return {
      id: "cuda",
      label: "NVIDIA CUDA",
      hint: hardware.accelerator
        ? `${hardware.accelerator} · ${gpuCountLabel(hardware)}`
        : `${gpuCountLabel(hardware)} configured · GPU model not recorded`,
    };
  };

  const runtimeSummary = (recipe) => {
    const runtime = runtimeFor(recipe);
    const hardware = recipe.hardware || {};
    if (runtime.id === "mlx") {
      return [hardware.accelerator || "Apple Silicon", hardware.system_memory || hardware.minimum_memory, "MLX"]
        .filter(Boolean)
        .join(" · ");
    }
    if (runtime.id === "mps") {
      return [hardware.accelerator || "Apple Silicon", hardware.system_memory || hardware.minimum_memory, "PyTorch MPS"]
        .filter(Boolean)
        .join(" · ");
    }
    if (hardware.accelerator) return [hardware.accelerator, gpuCountLabel(hardware)].join(" · ");
    return `NVIDIA CUDA · ${gpuCountLabel(hardware)} configured · GPU model and VRAM not recorded`;
  };

  const renderHardwareEvidence = (container, badge, recipe) => {
    const hardware = recipe.hardware || {};
    const runtime = runtimeFor(recipe);
    const isValidated = hardware.evidence === "validated";
    container.classList.toggle("cookbook-hardware-state--verified", isValidated);
    container.classList.toggle("cookbook-hardware-state--source", !isValidated);
    badge.classList.toggle("cookbook-badge--verified", isValidated);
    badge.classList.toggle("cookbook-badge--configured", !isValidated);
    badge.textContent = isValidated ? "Recorded run" : "Source config";

    const heading = document.createElement("strong");
    heading.textContent = isValidated ? "Recorded hardware" : "Source configuration";
    const details = document.createElement("span");

    if (isValidated) {
      const recorded = [
        hardware.accelerator,
        runtime.id === "cuda" ? gpuCountLabel(hardware) : hardware.system_memory,
      ].filter(Boolean);
      const statements = [`${recorded.join(" · ")}.`];
      if (hardware.minimum_memory) statements.push(`Documented minimum: ${hardware.minimum_memory}.`);
      if (hardware.peak_memory) statements.push(`Measured: ${hardware.peak_memory}.`);
      if (!hardware.minimum_memory) statements.push("This recorded device is not a minimum requirement.");
      details.textContent = ` ${statements.join(" ")}`;
    } else if (runtime.id === "cuda") {
      details.textContent = ` NVIDIA CUDA · ${gpuCountLabel(hardware)}. The source does not record the GPU model or VRAM.`;
    } else {
      details.textContent = ` ${runtime.label}. The source does not record a device or memory requirement.`;
    }

    container.replaceChildren(heading, details);
    if (hardware.evidence_url) {
      const evidenceLink = document.createElement("a");
      evidenceLink.href = hardware.evidence_url;
      evidenceLink.textContent = "View run evidence";
      evidenceLink.setAttribute("aria-label", `View recorded hardware evidence for ${recipe.label}`);
      container.append(" ", evidenceLink);
    }
  };

  const compactLifecycle = (root) => {
    const lifecycle = root.querySelector(".cookbook-lifecycle");
    if (!lifecycle || lifecycle.dataset.compact) return;
    lifecycle.dataset.compact = "true";
    const stages = [...lifecycle.querySelectorAll(".cookbook-lifecycle__stage")];
    const active = stages.find((stage) => stage.classList.contains("cookbook-lifecycle__stage--active"));
    const planned = stages
      .filter((stage) => stage !== active)
      .map((stage) => stage.childNodes[0]?.textContent?.trim())
      .filter(Boolean);
    const summary = document.createElement("span");
    summary.className = "cookbook-lifecycle__summary";
    summary.textContent = `Next: ${planned.join(", ")}`;
    lifecycle.replaceChildren(...(active ? [active] : []), summary);
  };

  const initFamilyBuilder = async (root) => {
    const family = root.dataset.family;
    if (!family) return;

    compactLifecycle(root);

    const modelOptions = root.querySelector("[data-cookbook-model-options]");
    const hardwareOptions = root.querySelector("[data-cookbook-hardware-options]");
    const description = root.querySelector("[data-cookbook-description]");
    const label = root.querySelector("[data-cookbook-label]");
    const model = root.querySelector("[data-cookbook-model]");
    const task = root.querySelector("[data-cookbook-task]");
    const hardwareValue = root.querySelector("[data-cookbook-gpus]");
    const artifact = root.querySelector("[data-cookbook-artifact]");
    const evidenceCell = root.querySelector("[data-cookbook-evidence]");
    const source = root.querySelector("[data-cookbook-source]");
    const modelLink = root.querySelector("[data-cookbook-model-link]");
    const command = root.querySelector("[data-cookbook-command]");
    const status = root.querySelector("[data-cookbook-status]");
    const hardwareState = root.querySelector("[data-cookbook-hardware-state]");
    const hardwareBadge = root.querySelector("[data-cookbook-hardware-badge]");
    const count = root.querySelector("[data-cookbook-count]");
    const result = root.querySelector(".cookbook-result");
    const commandBlock = root.querySelector(".cookbook-command");
    const servingPanel = root.querySelector("[data-cookbook-serving]");
    const usage = root.querySelector("[data-cookbook-usage]");
    const servingAvailability = root.querySelector("[data-cookbook-serving-availability]");
    const knobsContainer = root.querySelector("[data-cookbook-knobs]");

    modelOptions.setAttribute("aria-label", "Recipe");
    hardwareOptions.setAttribute("aria-label", "Runtime");

    let recipes;
    try {
      ({ recipes } = await loadRecipes(root.dataset.recipes));
    } catch (error) {
      if (status) status.textContent = "Recipes could not be loaded. Use the maintained examples link below.";
      console.error("Failed to load FastVideo cookbook recipes", error);
      return;
    }
    if (!root.isConnected) return;

    const familyRecipes = recipes.filter((recipe) => recipe.family === family);
    if (!familyRecipes.length) return;

    let servingProfiles = {};
    let servingLoadFailed = false;
    if (servingPanel && familyRecipes.some((recipe) => recipe.serving)) {
      try {
        const dataUrl = new URL(root.dataset.recipes, document.baseURI);
        servingProfiles = await loadRecipes(new URL("cookbook-serving.json", dataUrl));
      } catch (error) {
        servingLoadFailed = true;
        console.error("Failed to load FastVideo serving profiles", error);
      }
    }

    const byId = new Map(familyRecipes.map((recipe) => [recipe.id, recipe]));
    const groups = new Map();
    familyRecipes.forEach((recipe) => {
      const groupId = groupIdFor(recipe);
      if (!groups.has(groupId)) groups.set(groupId, []);
      groups.get(groupId).push(recipe);
    });
    if (count) count.textContent = `${familyRecipes.length} maintained recipes`;

    modelOptions.replaceChildren();
    groups.forEach((groupRecipes, groupId) => {
      const representative = groupRecipes[0];
      const option = document.createElement("button");
      option.type = "button";
      option.dataset.recipeGroup = groupId;
      option.setAttribute("aria-pressed", "false");
      const optionLabel = document.createElement("strong");
      optionLabel.textContent = representative.group_label || representative.label;
      const optionTask = document.createElement("span");
      optionTask.textContent = representative.group_task || representative.task;
      option.append(optionLabel, optionTask);
      modelOptions.append(option);
    });

    const query = new URLSearchParams(window.location.search);
    const requestedRecipe = query.get("recipe");
    const defaultRecipeId = byId.has(root.dataset.defaultRecipe) ? root.dataset.defaultRecipe : familyRecipes[0].id;
    let selectedRecipeId = requestedRecipe && byId.has(requestedRecipe) ? requestedRecipe : defaultRecipeId;
    let selectedGroupId = groupIdFor(byId.get(selectedRecipeId));
    let renderedRuntimeGroup = null;
    // Keep previously shared local/openai links working after renaming workflows.
    const workflow = (value) => ["local", "python"].includes(value) ? "python" : "server";
    let usagePreference = workflow(query.get("use"));
    let selectedClient = ["python", "javascript", "curl"].includes(query.get("client")) ? query.get("client") : "curl";
    const clientDetails = servingPanel?.querySelector(".cookbook-serving__code");
    if (clientDetails && query.has("client")) clientDetails.open = true;

    const knobDefs = new Map();
    familyRecipes.forEach((recipe) => knobsFor(recipe).forEach((knob) => {
      if (!knobDefs.has(knob.key)) knobDefs.set(knob.key, knob);
    }));
    const knobValues = {};
    knobDefs.forEach((knob, key) => {
      const fromQuery = query.get(key);
      const validValues = knobOptions(knob).map((option) => String(option.value));
      const useQueryValue = fromQuery !== null && validValues.includes(fromQuery);
      const raw = useQueryValue ? fromQuery : knob.default;
      knobValues[key] = typeof knob.default === "number" ? Number(raw) : raw;
    });

    const renderKnobs = (recipe, hidden) => {
      if (!knobsContainer) return;
      const knobs = knobsFor(recipe);
      const renderedKeys = [...knobsContainer.querySelectorAll("[data-knob-row]")].map((row) => row.dataset.knobRow);
      if (renderedKeys.join(",") !== knobs.map((knob) => knob.key).join(",")) {
        knobsContainer.replaceChildren();
        knobs.forEach((knob) => {
          const row = document.createElement("div");
          row.className = "cookbook-selection-row";
          row.dataset.knobRow = knob.key;
          const labelWrap = document.createElement("div");
          labelWrap.className = "cookbook-selection-row__label";
          const strongLabel = document.createElement("strong");
          strongLabel.textContent = knob.label;
          const hintLabel = document.createElement("span");
          hintLabel.textContent = knob.hint || "";
          labelWrap.append(strongLabel, hintLabel);
          const grid = document.createElement("div");
          grid.className = "cookbook-option-grid cookbook-option-grid--hardware";
          grid.setAttribute("role", "group");
          grid.setAttribute("aria-label", knob.label);
          knobOptions(knob).forEach((option) => {
            const optionButton = document.createElement("button");
            optionButton.type = "button";
            optionButton.dataset.knobKey = knob.key;
            optionButton.dataset.knobValue = String(option.value);
            optionButton.setAttribute("aria-pressed", "false");
            const optionLabel = document.createElement("strong");
            optionLabel.textContent = option.label;
            optionButton.append(optionLabel);
            grid.append(optionButton);
          });
          row.append(labelWrap, grid);
          knobsContainer.append(row);
        });
      }
      knobsContainer.hidden = hidden || knobs.length === 0;
      knobsContainer.querySelectorAll("button[data-knob-key]").forEach((optionButton) => {
        const selected = String(knobValues[optionButton.dataset.knobKey]) === optionButton.dataset.knobValue;
        optionButton.classList.toggle("cookbook-option--selected", selected);
        optionButton.setAttribute("aria-pressed", String(selected));
      });
    };

    const renderRuntimeOptions = () => {
      const groupRecipes = groups.get(selectedGroupId) || [];
      const renderedIds = [...hardwareOptions.querySelectorAll("[data-recipe-id]")].map((option) => option.dataset.recipeId);
      if (renderedIds.join(",") === groupRecipes.map((recipe) => recipe.id).join(",")) return;
      hardwareOptions.replaceChildren();
      groupRecipes.forEach((recipe) => {
        const runtime = runtimeFor(recipe);
        const option = document.createElement("button");
        option.type = "button";
        option.dataset.recipeId = recipe.id;
        option.dataset.runtimeId = runtime.id;
        option.setAttribute("aria-pressed", "false");
        const optionLabel = document.createElement("strong");
        optionLabel.textContent = runtime.label;
        const optionHint = document.createElement("span");
        optionHint.textContent = runtime.hint;
        option.append(optionLabel, optionHint);
        hardwareOptions.append(option);
      });
    };

    let notes = root.querySelector("[data-cookbook-notes]");
    if (!notes) {
      notes = document.createElement("aside");
      notes.className = "cookbook-recipe-notes";
      notes.dataset.cookbookNotes = "";
      notes.hidden = true;
      result.insertBefore(notes, commandBlock);
    }

    const render = ({ groupChanged = false, historyMode = "replace" } = {}) => {
      if (!root.isConnected) return;
      if (!byId.has(selectedRecipeId)) selectedRecipeId = defaultRecipeId;
      let recipe = byId.get(selectedRecipeId);
      if (groupChanged || groupIdFor(recipe) !== selectedGroupId) {
        const currentRuntime = runtimeFor(recipe).id;
        const groupRecipes = groups.get(selectedGroupId) || [];
        recipe = groupRecipes.find((candidate) => runtimeFor(candidate).id === currentRuntime) || groupRecipes[0];
        selectedRecipeId = recipe.id;
      }

      selectedGroupId = groupIdFor(recipe);
      if (renderedRuntimeGroup !== selectedGroupId) {
        renderRuntimeOptions();
        renderedRuntimeGroup = selectedGroupId;
      }
      const runtime = runtimeFor(recipe);
      const profile = servingPanel && servingProfiles[recipe.id];
      const useServer = Boolean(profile && usagePreference === "server");
      // The measured local profile and the server config have separate evidence.
      const activeRecipe = useServer ? { ...recipe, hardware: profile.hardware, evidence: "Source-backed" } : recipe;
      const knobs = knobsFor(recipe);
      renderKnobs(recipe, useServer);

      if (usage) {
        usage.querySelectorAll("[data-cookbook-mode]").forEach((option) => {
          const selected = option.dataset.cookbookMode === (useServer ? "server" : "python");
          option.disabled = option.dataset.cookbookMode === "server" && !profile;
          option.classList.toggle("cookbook-option--selected", selected);
          option.setAttribute("aria-pressed", String(selected));
        });
        servingAvailability.textContent = profile
          ? "The playground and API clients share one server process. Both workflows can run on your own machine."
          : servingLoadFailed
            ? "Server examples could not be loaded. Open the H3 server guide below, or use Python directly."
            : "This recipe uses Python directly. For the playground and API clients, choose FastH3 Preview with CUDA or MLX.";
        servingPanel.hidden = !useServer;
        commandBlock.hidden = useServer;
        root.querySelector("[data-cookbook-python-note]").hidden = useServer;
      }

      if (useServer) {
        const isMLX = profile.runtime === "mlx";
        servingPanel.querySelector("[data-cookbook-server-lifetime]").textContent = isMLX
          ? "Start once, then change prompts in the playground or your app. MLX reuses its pipeline and prompt cache, but loads and releases model components between phases to limit unified-memory use. It does not keep all weights resident."
          : "Start once, then change prompts in the playground or your app. CUDA requests reuse the loaded model. The Python SDK can also reuse a generator within one process.";
        servingPanel.querySelector("[data-cookbook-install-guide]").href = isMLX
          ? "../../getting_started/installation/mps/#run-fasth3-preview" : "../../getting_started/installation/gpu/";
        servingPanel.querySelector("[data-cookbook-prepare]").hidden = !profile.prepare;
        servingPanel.querySelector("[data-cookbook-server-prepare]").textContent = profile.prepare;
        servingPanel.querySelector("[data-cookbook-server-install]").textContent = profile.install;
        servingPanel.querySelector("[data-cookbook-server-command]").textContent = profile.command;
        servingPanel.querySelector("[data-cookbook-health-command]").textContent = profile.health_command;
        servingPanel.querySelector("[data-cookbook-playground]").href = profile.playground_url;
        const client = profile.clients[selectedClient];
        const filename = client.source.split("/").pop();
        servingPanel.querySelector("[data-cookbook-client-install]").textContent = client.install;
        const clientCode = servingPanel.querySelector("[data-cookbook-client-code]");
        clientCode.className = `language-${selectedClient === "curl" ? "bash" : selectedClient}`;
        clientCode.textContent = client.code;
        servingPanel.querySelector("[data-cookbook-client-filename]").textContent = filename;
        servingPanel.querySelector("[data-cookbook-client-source]").href = `https://github.com/hao-ai-lab/FastVideo/blob/main/${client.source}`;
        const runner = { python: "python", javascript: "node", curl: "bash" }[selectedClient];
        servingPanel.querySelector("[data-cookbook-client-run]").textContent = `Save as ${filename} and run ${runner} ${filename}. The MP4 is saved with the job ID as its filename.`;
        servingPanel.querySelectorAll("[data-cookbook-client]").forEach((option) => {
          option.setAttribute("aria-pressed", String(option.dataset.cookbookClient === selectedClient));
        });
      }

      modelOptions.querySelectorAll("button").forEach((option) => {
        const selected = option.dataset.recipeGroup === selectedGroupId;
        option.classList.toggle("cookbook-option--selected", selected);
        option.setAttribute("aria-pressed", String(selected));
      });
      hardwareOptions.querySelectorAll("button").forEach((option) => {
        const selected = option.dataset.recipeId === recipe.id;
        option.classList.toggle("cookbook-option--selected", selected);
        option.setAttribute("aria-pressed", String(selected));
        const candidate = byId.get(option.dataset.recipeId);
        const candidateProfile = servingProfiles[candidate.id];
        const candidateRuntime = runtimeFor(candidateProfile && usagePreference === "server"
          ? { ...candidate, hardware: candidateProfile.hardware } : candidate);
        option.querySelector("span").textContent = candidateRuntime.hint;
      });

      description.textContent = useServer
        ? `FastH3 Preview generates video with audio. This server profile uses the checked-in ${profile.runtime.toUpperCase()} configuration.`
        : recipe.summary;
      label.textContent = useServer ? `${recipe.group_label || recipe.label} · Server` : recipe.label;
      model.textContent = recipe.model;
      task.textContent = recipe.task;
      hardwareValue.textContent = runtimeSummary(activeRecipe);
      if (artifact) artifact.textContent = useServer ? "MP4 with audio" : recipe.expected_artifact || "Not yet documented for this recipe.";
      if (evidenceCell) {
        evidenceCell.textContent = activeRecipe.evidence || "Source-backed";
        evidenceCell.classList.toggle("cookbook-badge--verified", activeRecipe.evidence === "Verified");
        evidenceCell.classList.toggle("cookbook-badge--source-backed", activeRecipe.evidence !== "Verified");
      }
      source.href = `https://github.com/hao-ai-lab/FastVideo/blob/main/${useServer ? profile.source : recipe.source}`;
      source.textContent = useServer ? "View server configuration" : "Open example source";
      modelLink.href = `https://huggingface.co/${recipe.model}`;
      command.textContent = useServer ? recipe.command : appendKnobFlags(recipe.command, knobs, knobValues);

      renderHardwareEvidence(hardwareState, hardwareBadge, activeRecipe);

      const knobCaveats = useServer ? [] : knobs
        .filter((knob) => String(knobValues[knob.key]) !== String(knob.default))
        .map((knob) => `${knob.label} is set away from its recorded default (${knobDefaultLabel(knob)}). ` +
          "The script accepts this value, but it has not been benchmarked here.");
      const limitations = [...(useServer ? [
        profile.runtime === "mlx"
          ? "This MLX server config has no recorded hardware run. Measurements from the Python recipe are not server memory requirements. Only text-to-video/audio is wired; reference inputs and fast modes are not exposed here."
          : "This server config has no recorded serving benchmark. Compilation is disabled, unlike the measured Python performance profile.",
        `${profile.sampling.width} × ${profile.sampling.height} · ${profile.sampling.num_frames} frames · ${profile.sampling.fps} fps. The server supplies these defaults; the client sends the model and prompt.`,
        "Generation is serialized. Job metadata is held in memory and is lost when the server restarts.",
      ] : recipe.limitations || []), ...knobCaveats];
      notes.replaceChildren();
      notes.hidden = limitations.length === 0;
      if (limitations.length) {
        const notesHeading = document.createElement("strong");
        notesHeading.textContent = "Know before you run";
        const notesList = document.createElement("ul");
        limitations.forEach((item) => {
          const listItem = document.createElement("li");
          listItem.textContent = item;
          notesList.append(listItem);
        });
        notes.append(notesHeading, notesList);
      }

      const nextQuery = new URLSearchParams(window.location.search);
      nextQuery.set("recipe", recipe.id);
      nextQuery.set("runtime", runtime.id);
      nextQuery.delete("gpus");
      knobDefs.forEach((knob, key) => {
        if (knobs.some((activeKnob) => activeKnob.key === key)) nextQuery.set(key, String(knobValues[key]));
        else nextQuery.delete(key);
      });
      if (usage) {
        nextQuery.set("use", useServer ? "server" : "python");
        if (useServer && clientDetails?.open) nextQuery.set("client", selectedClient);
        else nextQuery.delete("client");
      }
      const nextUrl = `${window.location.pathname}?${nextQuery.toString()}${window.location.hash}`;
      if (historyMode === "push") window.history.pushState({}, "", nextUrl);
      else if (historyMode === "replace") window.history.replaceState({}, "", nextUrl);
      const modeSummary = useServer ? " with a persistent server" : "";
      status.textContent = `${recipe.label} selected for ${runtime.label}${modeSummary}.`;
    };

    modelOptions.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-recipe-group]");
      if (!option) return;
      selectedGroupId = option.dataset.recipeGroup;
      render({ groupChanged: true, historyMode: "push" });
    });
    hardwareOptions.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-recipe-id]");
      if (!option) return;
      selectedRecipeId = option.dataset.recipeId;
      selectedGroupId = groupIdFor(byId.get(selectedRecipeId));
      render({ historyMode: "push" });
    });
    usage?.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-cookbook-mode]");
      if (!option || option.disabled) return;
      usagePreference = option.dataset.cookbookMode;
      render({ historyMode: "push" });
    });
    servingPanel?.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-cookbook-client]");
      if (!option) return;
      selectedClient = option.dataset.cookbookClient;
      render({ historyMode: "push" });
    });
    knobsContainer?.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-knob-key]");
      if (!option) return;
      const knob = knobDefs.get(option.dataset.knobKey);
      knobValues[option.dataset.knobKey] = typeof knob.default === "number"
        ? Number(option.dataset.knobValue) : option.dataset.knobValue;
      render({ historyMode: "push" });
    });

    render();

    familyPopstate = () => {
      if (!root.isConnected) return;
      const nextQuery = new URLSearchParams(window.location.search);
      const nextRecipe = nextQuery.get("recipe");
      selectedRecipeId = nextRecipe && byId.has(nextRecipe) ? nextRecipe : defaultRecipeId;
      selectedGroupId = groupIdFor(byId.get(selectedRecipeId));
      usagePreference = workflow(nextQuery.get("use"));
      selectedClient = ["python", "javascript", "curl"].includes(nextQuery.get("client")) ? nextQuery.get("client") : "curl";
      if (clientDetails) clientDetails.open = nextQuery.has("client");
      knobDefs.forEach((knob, key) => {
        const fromQuery = nextQuery.get(key);
        const validValues = knobOptions(knob).map((option) => String(option.value));
        if (fromQuery !== null && validValues.includes(fromQuery)) {
          knobValues[key] = typeof knob.default === "number" ? Number(fromQuery) : fromQuery;
        }
      });
      render({ historyMode: "none" });
    };
    bindFamilyPopstate();
  };

  const init = () => {
    initEvervault(document);
    document.querySelectorAll("[data-cookbook][data-family]").forEach((root) => {
      if (root.dataset.initialized) return;
      root.dataset.initialized = "true";
      initFamilyBuilder(root);
    });
  };

  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
