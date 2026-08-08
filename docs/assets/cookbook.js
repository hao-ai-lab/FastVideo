(() => {
  let recipesPromise;

  const loadRecipes = (url) => {
    recipesPromise ||= fetch(url).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    });
    return recipesPromise;
  };

  const init = () => {
    document.querySelectorAll("[data-cookbook]").forEach(async (root) => {
      if (root.dataset.initialized) return;
      root.dataset.initialized = "true";

      const select = root.querySelector("[data-cookbook-recipe]");
      const model = root.querySelector("[data-cookbook-model]");
      const source = root.querySelector("[data-cookbook-source]");
      const command = root.querySelector("[data-cookbook-command]");
      const status = root.querySelector("[data-cookbook-status]");

      try {
        const { recipes } = await loadRecipes(root.dataset.recipes);
        const byId = new Map(recipes.map((recipe) => [recipe.id, recipe]));
        const groups = new Map();

        select.replaceChildren();
        recipes.forEach((recipe) => {
          if (!groups.has(recipe.task)) {
            const group = document.createElement("optgroup");
            group.label = recipe.task;
            groups.set(recipe.task, group);
            select.append(group);
          }
          groups.get(recipe.task).append(new Option(recipe.label, recipe.id));
        });

        const render = () => {
          const recipe = byId.get(select.value);
          model.textContent = recipe.model;
          source.textContent = recipe.source;
          source.href = `https://github.com/hao-ai-lab/FastVideo/blob/main/${recipe.source}`;
          command.textContent = recipe.command;
          status.textContent = `${recipe.label} selected.`;
        };

        select.addEventListener("change", render);
        select.disabled = false;
        render();
      } catch (error) {
        status.textContent = "Recipes could not be loaded. Use the examples link below.";
        console.error("Failed to load FastVideo cookbook recipes", error);
      }
    });
  };

  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
