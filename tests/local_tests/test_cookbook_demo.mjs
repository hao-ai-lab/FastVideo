import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { runInNewContext } from "node:vm";

const require = createRequire(import.meta.url);
const { createCatalogLoader } = require("../../examples/cookbook/cookbook-demo.js");

function demoData() {
  const definitions = [
    ["alpha/cuda-rest", "cuda", "rest", "t2v"], ["alpha/mlx-rest", "mlx", "rest", "t2v"],
    ["image/cuda-rest", "cuda", "rest", "i2v"], ["stream/cuda-streaming", "cuda", "websocket", "t2v"],
  ];
  const catalogs = new Map(), models = new Map();
  for (const [id, backend, interfaceName, workload] of definitions) {
    const [key, deploymentId] = id.split("/"), modelId = `Example/${key}`;
    const label = backend === "mlx" ? "MLX REST" : interfaceName === "websocket" ? "CUDA streaming" : "CUDA REST";
    const controls = [
      { path: "server.host", schema: { type: "string" } },
      { path: "server.port", schema: { type: "integer", minimum: 1, maximum: 65535 } },
      ...(backend === "mlx" ? [
        { path: "generator.vsa_sparsity", schema: { type: "number", minimum: 0, exclusiveMaximum: 1 } },
        { path: "generator.vae_dtype", schema: { type: "string", enum: ["fp32", "fp16", "bf16"] } },
      ] : [{ path: "default_request.sampling.num_frames", schema: { type: "integer", minimum: 1 } }]),
    ];
    const data = {
      id, model: { id: modelId, key, title: key }, deployment: { id: deploymentId, label },
      workload, summary: "Example recipe", source_config: "examples/example.yaml", env: {}, requirements: [], controls,
      guide: backend === "mlx" ? null : { url: `../../cookbook/guides/${key}-serving/` },
      runtime: { backend, interface: interfaceName, label, install_url: "https://example.test/install/",
        server_defaults: { host: "0.0.0.0", port: 8000 },
        launch_argv: backend === "mlx" ? ["python", "-m", "fastvideo.entrypoints.openai.mlx_server", "--config", "config.yaml"] :
          ["fastvideo", "serve", "--config", "config.yaml"],
        ...(backend === "mlx" ? { hardware_label: "Apple Silicon" } : {}),
        ...(interfaceName === "websocket" ? { client_guide_url: "../../design/server_contracts/streaming/" } : {}),
      },
      base_config: {
        generator: { model_path: modelId, engine: { num_gpus: 1 }, vsa_sparsity: 0.8, vae_dtype: "fp32" },
        server: { host: "0.0.0.0", port: interfaceName === "websocket" ? 8009 : 8000 },
        default_request: { sampling: { num_frames: 81 } },
      },
    };
    catalogs.set(id, data);
    if (!models.has(key)) models.set(key, { id: key, title: key, model_id: modelId, deployments: [] });
    models.get(key).deployments.push({ id, label, catalog_url: `recipes/${id}.json` });
  }
  return { index: { models: [...models.values()] }, catalogs };
}

// Minimal DOM surface used by the adapter: exercise its actual event handlers without browser packages.
class Element {
  constructor(tag = "div") { this.tagName = tag; this.children = []; this.dataset = {}; this.listeners = new Map(); }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text || "") + this.children.map((child) => child.textContent ?? child).join(""); }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.text = ""; this.children = children; }
  setAttribute(key, value) { this[key] = value; }
  removeAttribute(key) { delete this[key]; }
  addEventListener(event, listener) { this.listeners.set(event, listener); }
  trigger(event) { return this.listeners.get(event)?.(); }
}

async function browserFixture(customize = () => {}, documentationIndex) {
  const { index, catalogs } = demoData();
  customize(catalogs);
  const nodes = new Map(), requests = [], urls = [], failures = new Set();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  };
  const root = new Element();
  root.dataset.metadata = "https://example.test/project/assets/cookbook-config/index.json";
  if (documentationIndex) root.dataset.documentationIndex = documentationIndex;
  root.querySelector = node;
  const copyButtons = ["yaml", "command", "clientCommand"].map((name) => {
    const button = new Element("button"); button.dataset.configCopy = name; return button;
  });
  root.querySelectorAll = (selector) => selector.includes("data-config-download") ?
    [...copyButtons, node("[data-config-download]")] : copyButtons;
  const document = { readyState: "complete", querySelectorAll: () => [root], createElement: (tag) => new Element(tag) };
  const fetch = async (url) => {
    urls.push(url);
    if (url === root.dataset.metadata) return { ok: true, url, json: async () => index };
    assert.ok(url.includes("/recipes/"), "Demo must not fetch guide pages");
    const id = url.split("/recipes/")[1]?.replace(/\.json$/, "");
    requests.push(id);
    if (failures.has(id)) throw new Error("Deployment unavailable");
    assert.ok(catalogs.has(id), `Unexpected catalog request: ${url}`);
    return { ok: true, json: async () => catalogs.get(id) };
  };
  const context = { document, fetch, URL, AbortController, setTimeout,
    FastVideoSchema: require("../../docs/assets/cookbook-validator.js") };
  for (const file of ["docs/assets/cookbook-config.js", "examples/cookbook/cookbook-demo.js"]) {
    runInNewContext(readFileSync(new URL(`../../${file}`, import.meta.url), "utf8"), context);
  }
  assert.deepEqual(Object.keys(context.FastVideoCookbookDemo), ["mount"]);
  assert.equal(requests.length, 0, "Demo import must not mount automatically");
  await context.FastVideoCookbookDemo.mount(root);
  await new Promise(setImmediate);
  const element = (name) => node(`[data-config-${name}]`);
  const descendants = (parent) => parent.children.flatMap((child) => typeof child === "string" ? [] :
    [child, ...descendants(child)]);
  const control = (path) => descendants(element("controls")).find((item) => item.dataset.configPath === path);
  return { element, control, requests, urls, failures, code: (name) => node(`[data-config-code="${name}"]`).textContent };
}

test("local demo keeps catalog requests local and links to published runbooks", async () => {
  const { element, urls } = await browserFixture(undefined,
    "https://docs.example.test/FastVideo/assets/cookbook-config/index.json");
  assert.equal(element("guide").href, "https://docs.example.test/FastVideo/cookbook/guides/alpha-serving/");
  element("model-picker").value = "stream";
  await element("model-picker").trigger("change");
  assert.equal(element("client-guide").href, "https://docs.example.test/FastVideo/design/server_contracts/streaming/");
  assert.ok(urls.every((url) => url.startsWith("https://example.test/project/assets/cookbook-config/")));
});

test("model and deployment selectors reset edits, update runtime guidance and clear failed deployment data", async () => {
  const { element, control, requests, urls, failures, code } = await browserFixture();
  assert.deepEqual(element("model-picker").children.map((item) => item.value),
    ["alpha", "image", "stream"]);
  assert.deepEqual(element("deployment-picker").children.map((item) => item.value),
    ["alpha/cuda-rest", "alpha/mlx-rest"]);
  assert.deepEqual(requests, ["alpha/cuda-rest"]);
  assert.equal(element("output").hidden, false);
  assert.equal(element("guide").hidden, false);
  assert.equal(element("guide").href, "https://example.test/project/cookbook/guides/alpha-serving/");
  assert.deepEqual(urls, [
    "https://example.test/project/assets/cookbook-config/index.json",
    "https://example.test/project/assets/cookbook-config/recipes/alpha/cuda-rest.json",
  ]);
  control("server.port").value = "9001";
  control("server.port").trigger("input");
  assert.match(code("yaml"), /port: 9001/);
  element("deployment-picker").value = "alpha/mlx-rest";
  await element("deployment-picker").trigger("change");
  assert.match(code("yaml"), /port: 8000/);
  assert.match(code("command"), /python -m fastvideo.entrypoints.openai.mlx_server/);
  assert.equal(element("install").textContent, "MLX REST installation guide");
  assert.equal(element("topology").textContent, "Hardware family: Apple Silicon.");
  assert.equal(element("guide").hidden, true);
  assert.equal(element("guide").href, undefined);
  assert.equal(control("default_request.sampling.num_frames"), undefined);
  control("generator.vsa_sparsity").value = "1";
  control("generator.vsa_sparsity").trigger("input");
  assert.equal(element("output").hidden, true);
  assert.match(element("error").textContent, /must be < 1/);
  control("generator.vsa_sparsity").value = "0.5";
  control("generator.vsa_sparsity").trigger("input");
  control("generator.vae_dtype").value = "2";
  control("generator.vae_dtype").trigger("change");
  assert.equal(element("output").hidden, false);
  assert.match(code("yaml"), /vae_dtype: "bf16"/);
  element("model-picker").value = "image";
  await element("model-picker").trigger("change");
  assert.deepEqual(element("deployment-picker").children.map((item) => item.value), ["image/cuda-rest"]);
  assert.equal(element("guide").href, "https://example.test/project/cookbook/guides/image-serving/");
  assert.equal(element("requirements-section").hidden, false);
  assert.equal(element("topology").textContent, "Recipe baseline GPU count: 1.");
  control("server.port").value = "9003";
  control("server.port").trigger("input");
  element("reset").trigger("click");
  assert.match(code("yaml"), /port: 8000/);
  failures.add("alpha/cuda-rest");
  element("model-picker").value = "alpha";
  await element("model-picker").trigger("change");
  assert.equal(element("output").hidden, true);
  assert.equal(element("requirements-section").hidden, true);
  assert.equal(element("requirements").children.length, 0);
  assert.equal(element("guide").hidden, true);
  assert.equal(element("guide").href, undefined);
  assert.equal(element("controls").children.length, 0);
  assert.match(element("error").textContent, /Deployment unavailable/);
  assert.ok(urls.every((url) => url.endsWith("/index.json") || url.includes("/recipes/")));
});

test("streaming UI updates its endpoint and clears protocol details when returning to REST or failing", async () => {
  const { element, control, code, failures } = await browserFixture();
  element("model-picker").value = "stream";
  await element("model-picker").trigger("change");
  assert.equal(element("client-title").textContent, "4. Check server and connect a streaming client");
  assert.equal(code("clientCommand"), "curl --fail-with-body http://127.0.0.1:8009/health");
  assert.equal(element("streaming-client").hidden, false);
  assert.equal(element("websocket-url").textContent, "ws://127.0.0.1:8009/v1/stream");
  assert.equal(element("client-guide").href, "https://example.test/project/design/server_contracts/streaming/");
  assert.equal(element("request-note").hidden, true);
  control("server.host").value = "::";
  control("server.host").trigger("input");
  control("server.port").value = "9012";
  control("server.port").trigger("input");
  assert.equal(element("websocket-url").textContent, "ws://[::1]:9012/v1/stream");
  assert.match(code("clientCommand"), /http:\/\/\[::1\]:9012\/health/);
  element("model-picker").value = "image";
  await element("model-picker").trigger("change");
  assert.equal(element("client-title").textContent, "4. Send a sample request");
  assert.match(code("clientCommand"), /input_reference.*first-frame\.png/);
  assert.equal(element("streaming-client").hidden, true);
  assert.equal(element("websocket-url").textContent, "");
  assert.equal(element("client-guide").href, undefined);
  assert.equal(element("request-note").hidden, false);
  element("model-picker").value = "stream";
  await element("model-picker").trigger("change");
  failures.add("image/cuda-rest");
  element("model-picker").value = "image";
  await element("model-picker").trigger("change");
  assert.equal(element("output").hidden, true);
  assert.equal(element("streaming-client").hidden, true);
  assert.equal(element("websocket-url").textContent, "");
  assert.equal(element("client-guide").href, undefined);
});

test("GPU edits update degree bounds in place while invalid topology waits for the user to fix it", async () => {
  const { control, element, code } = await browserFixture((catalogs) => {
    const data = catalogs.get("alpha/cuda-rest");
    data.runtime.topology_defaults = { num_gpus: 1, tp_size: -1, sp_size: -1,
      hsdp_replicate_dim: 1, hsdp_shard_dim: -1 };
    data.base_config.generator.engine = { num_gpus: 4, parallelism: { tp_size: 1, sp_size: 4 } };
    for (const path of ["generator.engine.num_gpus", "generator.engine.parallelism.tp_size",
      "generator.engine.parallelism.sp_size"]) data.controls.push({ path, schema: { type: "integer" } });
  });
  const gpu = control("generator.engine.num_gpus"), sp = control("generator.engine.parallelism.sp_size");
  assert.equal(sp.max, 4);
  assert.equal(sp.min, -1);
  gpu.value = "2";
  gpu.trigger("input");
  assert.equal(control("generator.engine.parallelism.sp_size"), sp, "An edit must retain existing DOM input/focus");
  assert.equal(sp.value, 4, "Invalid degrees must not be silently clamped");
  assert.equal(sp.max, 2);
  assert.equal(element("output").hidden, true);
  assert.match(element("error").textContent, /sp_size.*num_gpus \(2\)/);
  sp.value = "2";
  sp.trigger("input");
  assert.equal(element("output").hidden, false);
  assert.match(code("yaml"), /num_gpus: 2/);
  assert.match(code("yaml"), /sp_size: 2/);
  element("reset").trigger("click");
  assert.equal(control("generator.engine.parallelism.sp_size").max, 4);
  assert.equal(control("generator.engine.parallelism.sp_size").value, 4);
});


function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function loaderFixture() {
  const requests = [], updates = [];
  const models = [
    { id: "example", title: "Example model", model_id: "Example/Video", deployments: ["A", "B"].map((id) =>
      ({ id, label: id, workload: "t2v", catalog_url: `recipes/${id}.json` })) },
    { id: "other", title: "Other model", model_id: "Other/Video", deployments: [
      { id: "C", label: "CUDA REST", workload: "t2v", catalog_url: "recipes/C.json" },
    ] },
  ];
  const load = createCatalogLoader({ models, url: "https://example.test/project/assets/cookbook-config/index.json" },
    (update) => updates.push(update), (url, options) => {
      const request = { ...deferred(), url, signal: options.signal };
      requests.push(request);
      return request.promise;
    });
  return { load, requests, updates, models };
}

const modelResponse = (id) => ({ ok: true, json: async () => ({ id }) });

test("loader exposes only authored model deployments and isolates cross-model requests", async () => {
  const { load, requests, updates } = loaderFixture();
  await load("other/mlx-rest");
  assert.equal(requests.length, 0);
  assert.equal(updates.at(-1).state, "error");
  assert.match(updates.at(-1).message, /No exported deployment/);
  const first = load("A"), second = load("C");
  assert.equal(updates.at(-1).model.id, "other");
  assert.equal(updates.at(-1).deployment.id, "C");
  assert.equal(requests[0].signal.aborted, true);
  requests[1].resolve(modelResponse("C"));
  await second;
  requests[0].resolve(modelResponse("A"));
  await first;
  assert.equal(updates.at(-1).catalog.id, "C");
});

test("loader fetches only the selected catalog relative to the captured index URL and does not cache", async () => {
  const { load, requests, updates } = loaderFixture();
  assert.equal(requests.length, 0);
  const first = load("B");
  assert.deepEqual(updates.map((update) => update.state), ["loading"]);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "https://example.test/project/assets/cookbook-config/recipes/B.json");
  requests[0].resolve(modelResponse("B"));
  await first;
  assert.equal(updates.at(-1).catalog.id, "B");
  const second = load("B");
  assert.equal(requests.length, 2);
  assert.equal(updates.at(-1).state, "loading");
  assert.equal(updates.at(-1).catalog, undefined);
  requests[1].resolve(modelResponse("B"));
  await second;
  assert.deepEqual(updates.map((update) => update.state), ["loading", "ready", "loading", "ready"]);
});

test("loader aborts earlier fetches and ignores a late successful response", async () => {
  const { load, requests, updates } = loaderFixture();
  const first = load("A"), second = load("B");
  assert.equal(requests[0].signal.aborted, true);
  assert.equal(requests[1].signal.aborted, false);
  requests[1].resolve(modelResponse("B"));
  await second;
  requests[0].resolve({ ok: true, json: () => assert.fail("Obsolete response must not be parsed") });
  await first;
  assert.deepEqual(updates.map((update) => update.state), ["loading", "loading", "ready"]);
  assert.equal(updates.at(-1).catalog.id, "B");
});

test("loader ignores obsolete JSON completion and parsing failure after a new selection", async () => {
  for (const reject of [false, true]) {
    const { load, requests, updates } = loaderFixture();
    const body = deferred(), parsing = deferred();
    const first = load("A");
    requests[0].resolve({ ok: true, json: () => { parsing.resolve(); return body.promise; } });
    await parsing.promise;
    const second = load("B");
    requests[1].resolve(modelResponse("B"));
    await second;
    if (reject) body.reject(new Error("Obsolete JSON failure"));
    else body.resolve({ id: "A" });
    await first;
    assert.deepEqual(updates.map((update) => update.state), ["loading", "loading", "ready"]);
    assert.equal(updates.at(-1).catalog.id, "B");
  }
});

test("loader ignores obsolete fetch errors while the active selection is pending", async () => {
  const { load, requests, updates } = loaderFixture();
  const first = load("A"), second = load("B");
  requests[0].reject(new Error("Obsolete network failure"));
  await first;
  assert.equal(updates.at(-1).state, "loading");
  requests[1].resolve(modelResponse("B"));
  await second;
  assert.deepEqual(updates.map((update) => update.state), ["loading", "loading", "ready"]);
});

test("active HTTP, network, malformed JSON and model mismatch failures clear ready state and allow recovery", async () => {
  for (const fail of [
    (request) => request.resolve({ ok: false, status: 404 }),
    (request) => request.reject(new Error("Network unavailable")),
    (request) => request.resolve({ ok: true, json: async () => { throw new SyntaxError("Invalid JSON"); } }),
    (request) => request.resolve(modelResponse("Wrong model")),
  ]) {
    const { load, requests, updates } = loaderFixture();
    const first = load("A");
    requests[0].resolve(modelResponse("A"));
    await first;
    const second = load("B");
    assert.equal(updates.at(-1).state, "loading");
    assert.equal(updates.at(-1).catalog, undefined);
    fail(requests[1]);
    await second;
    assert.equal(updates.at(-1).state, "error");
    assert.equal(updates.at(-1).catalog, undefined);
    assert.ok(updates.at(-1).message);
    const retry = load("B");
    requests[2].resolve(modelResponse("B"));
    await retry;
    assert.equal(updates.at(-1).catalog.id, "B");
  }
});
