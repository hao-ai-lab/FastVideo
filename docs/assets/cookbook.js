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
    const defaultRecipeId = familyRecipes[0].id;
    let selectedRecipeId = requestedRecipe && byId.has(requestedRecipe) ? requestedRecipe : defaultRecipeId;
    let selectedGroupId = groupIdFor(byId.get(selectedRecipeId));
    let renderedRuntimeGroup = null;

    const renderRuntimeOptions = () => {
      const groupRecipes = groups.get(selectedGroupId) || [];
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

      modelOptions.querySelectorAll("button").forEach((option) => {
        const selected = option.dataset.recipeGroup === selectedGroupId;
        option.classList.toggle("cookbook-option--selected", selected);
        option.setAttribute("aria-pressed", String(selected));
      });
      hardwareOptions.querySelectorAll("button").forEach((option) => {
        const selected = option.dataset.recipeId === recipe.id;
        option.classList.toggle("cookbook-option--selected", selected);
        option.setAttribute("aria-pressed", String(selected));
      });

      description.textContent = recipe.summary;
      label.textContent = recipe.label;
      model.textContent = recipe.model;
      task.textContent = recipe.task;
      hardwareValue.textContent = runtimeSummary(recipe);
      if (artifact) artifact.textContent = recipe.expected_artifact || "Not yet documented for this recipe.";
      if (evidenceCell) {
        evidenceCell.textContent = recipe.evidence || "Source-backed";
        evidenceCell.classList.toggle("cookbook-badge--verified", recipe.evidence === "Verified");
        evidenceCell.classList.toggle("cookbook-badge--source-backed", recipe.evidence !== "Verified");
      }
      source.href = `https://github.com/hao-ai-lab/FastVideo/blob/main/${recipe.source}`;
      modelLink.href = `https://huggingface.co/${recipe.model}`;
      command.textContent = recipe.command;

      renderHardwareEvidence(hardwareState, hardwareBadge, recipe);

      const limitations = recipe.limitations || [];
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
      const nextUrl = `${window.location.pathname}?${nextQuery.toString()}${window.location.hash}`;
      if (historyMode === "push") window.history.pushState({}, "", nextUrl);
      else if (historyMode === "replace") window.history.replaceState({}, "", nextUrl);
      status.textContent = `${recipe.label} selected for ${runtime.label}.`;
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

    render();

    familyPopstate = () => {
      if (!root.isConnected) return;
      const nextQuery = new URLSearchParams(window.location.search);
      const nextRecipe = nextQuery.get("recipe");
      selectedRecipeId = nextRecipe && byId.has(nextRecipe) ? nextRecipe : defaultRecipeId;
      selectedGroupId = groupIdFor(byId.get(selectedRecipeId));
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
