/** TODO(cookbook-ui): Load the temporary demo only on pages containing its mount point. */
(() => {
  const assetBase = new URL(".", document.currentScript.src);
  let assets;

  function loadScript(name) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = new URL(name, assetBase).href;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Could not load ${name}. Reload the page to retry.`));
      document.head.append(script);
    });
  }

  function init() {
    for (const root of document.querySelectorAll("[data-config-builder]")) {
      if (root.dataset.initialized) continue;
      root.dataset.initialized = "true";
      assets ??= ["cookbook-validator.js", "cookbook-config.js", "cookbook-demo.js"]
        .reduce((ready, name) => ready.then(() => loadScript(name)), Promise.resolve());
      assets.then(() => {
        if (root.isConnected) return globalThis.FastVideoCookbookDemo.mount(root);
      }).catch((error) => {
        root.querySelector("[data-config-status]").textContent = "";
        root.querySelector("[data-config-error]").textContent = error.message;
      });
    }
  }

  if (globalThis.document$) globalThis.document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
