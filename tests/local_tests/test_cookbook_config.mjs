import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { runInNewContext } from "node:vm";

const require = createRequire(import.meta.url);
const { resolveConfig, createCatalogLoader, hardwareView, guideUrl, toYaml, shellQuote } =
  require("../../docs/assets/cookbook-config.js");

function catalog() {
  return {
    id: "example-rest", model: { id: "Example/Video" }, workload: "t2v",
    runtime: { id: "fastvideo-cuda-rest", launch_argv: ["fastvideo", "serve", "--config", "config.yaml"],
      server_defaults: { host: "0.0.0.0", port: 8000 } },
    env: { BACKEND: "special attention" },
    base_config: {
      generator: { model_path: "Example/Video", engine: { compile: { enabled: true } },
        pipeline: { experimental: { hidden: [1000, 757, 522] } } },
      server: { host: "0.0.0.0", port: 8000, served_model_name: "video" },
      default_request: { sampling: { num_frames: 81, seed: 5 }, output: { return_frames: false } },
    },
    controls: [
      { path: "server.port", schema: { type: "integer", minimum: 1, maximum: 65535, default: 8000 } },
      { path: "server.host", schema: { type: "string", default: "0.0.0.0" } },
      { path: "server.served_model_name", schema: { anyOf: [{ type: "string" }, { type: "null" }], default: null } },
      { path: "generator.engine.compile.enabled", schema: { type: "boolean", default: false } },
      { path: "default_request.sampling.num_frames", schema: { type: "integer", minimum: 1, default: 61 } },
      { path: "default_request.sampling.guidance_scale", schema: { type: "number", default: 5 } },
      { path: "default_request.sampling.seed", schema: { anyOf: [{ type: "integer" }, { type: "null" }], default: null } },
    ],
  };
}
const field = (data, path) => data.controls.find((control) => control.path === path).schema;

test("hardware defaults to unverified and optional metadata cannot change saved configuration", () => {
  const data = catalog(), original = resolveConfig(data);
  assert.deepEqual(hardwareView(data, original.config).records, []);
  data.hardware = [{ id: "example-gpu", label: "Example GPU", platform: "cuda", memory_gb: 80,
    source_url: "https://example.test/gpu-specification" }];
  data.guide = { url: "../../cookbook/guides/example/" };
  assert.equal(hardwareView(data, original.config).records[0].status, "unverified");
  assert.equal(data.hardware[0].status, undefined);
  assert.deepEqual(resolveConfig(data), original);
});

test("hardware evidence applies to baseline only; equivalent edits and reset restore baseline status", () => {
  const data = catalog();
  data.hardware = [{ id: "example-gpu", label: "Example GPU", platform: "cuda", memory_gb: 80,
    source_url: "https://example.test/gpu-specification", status: "verified",
    evidence_url: "https://example.test/baseline-run" }];
  const before = JSON.stringify(data);
  const baseline = hardwareView(data, resolveConfig(data).config);
  assert.equal(baseline.isBaseline, true);
  assert.equal(baseline.records[0].status, "verified");
  const custom = hardwareView(data, resolveConfig(data, { "default_request.sampling.num_frames": 121 }).config);
  assert.equal(custom.isBaseline, false);
  assert.match(custom.message, /^Custom configuration — unverified/);
  assert.match(custom.message, /original recipe baseline/);
  assert.equal(custom.records[0].evidence_url, "https://example.test/baseline-run");
  assert.deepEqual(hardwareView(data, resolveConfig(data, { "server.port": 8000 }).config), baseline);
  assert.deepEqual(hardwareView(data, resolveConfig(data).config), baseline);
  assert.equal(JSON.stringify(data), before);
});

test("unsupported baseline reasons remain visible without claiming custom configurations unsupported", () => {
  const data = catalog();
  data.hardware = [{ id: "example-gpu", label: "Example GPU", platform: "cuda", memory_gb: 16,
    source_url: "https://example.test/gpu-specification", status: "unsupported",
    reason: "The recorded baseline exceeds this GPU's memory." }];
  const baseline = hardwareView(data, resolveConfig(data).config);
  assert.equal(baseline.records[0].status, "unsupported");
  const custom = hardwareView(data, resolveConfig(data, { "default_request.sampling.num_frames": 17 }).config);
  assert.match(custom.message, /^Custom configuration — unverified/);
  assert.doesNotMatch(custom.message, /unsupported/i);
  assert.equal(custom.records[0].reason, data.hardware[0].reason);
});

test("guide links use the captured index URL and preserve a deployed project prefix", () => {
  const data = catalog();
  const indexUrl = "https://example.test/project/assets/cookbook-config/index.json";
  assert.equal(guideUrl(data, indexUrl), null);
  data.guide = null;
  assert.equal(guideUrl(data, indexUrl), null);
  data.guide = { url: "../../cookbook/guides/fastwan21-serving/" };
  assert.equal(guideUrl(data, indexUrl), "https://example.test/project/cookbook/guides/fastwan21-serving/");
});

test("unedited download preserves the entire explicit baseline and display defaults stay unsaved", () => {
  const data = catalog(), before = JSON.stringify(data);
  const result = resolveConfig(data);
  assert.deepEqual(result.config, data.base_config);
  assert.equal(result.config.default_request.sampling.num_frames, 81);
  assert.equal(result.config.default_request.sampling.guidance_scale, undefined);
  assert.equal(JSON.stringify(data), before);
  assert.deepEqual(result.config.generator.pipeline.experimental.hidden, [1000, 757, 522]);
});

test("edits preserve hidden settings; empty edits reset to baseline without mutation", () => {
  const data = catalog();
  const initial = resolveConfig(data);
  const edited = resolveConfig(data, { "server.port": 9000, "default_request.sampling.num_frames": 121,
    "generator.engine.compile.enabled": false });
  assert.equal(edited.config.server.port, 9000);
  assert.equal(edited.config.default_request.sampling.num_frames, 121);
  assert.equal(edited.config.generator.engine.compile.enabled, false);
  assert.deepEqual(edited.config.generator.pipeline, initial.config.generator.pipeline);
  assert.equal(edited.config.default_request.output.return_frames, false);
  assert.deepEqual(resolveConfig(data), initial);
});

test("only manifest-selected controls can be edited, including existing hidden paths", () => {
  for (const path of ["server.extra", "generator.model_path", "generator.pipeline.experimental.hidden",
    "generator.__proto__.polluted", "default_request.output.return_frames"]) {
    assert.throws(() => resolveConfig(catalog(), { [path]: 1 }), /not an editable recipe control/);
  }
  assert.equal({}.polluted, undefined);
});

test("types, bounds, enum and nullable choices use standard field schema validation", () => {
  const data = catalog();
  for (const value of [0, 65536, 1.5, "8000", NaN, Infinity]) {
    assert.throws(() => resolveConfig(data, { "server.port": value }), /server.port/);
  }
  assert.throws(() => resolveConfig(data, { "generator.engine.compile.enabled": "false" }));
  assert.throws(() => resolveConfig(data, []));
  assert.throws(() => resolveConfig(data, null));
  field(data, "default_request.sampling.num_frames").enum = [81, 121];
  assert.throws(() => resolveConfig(data, { "default_request.sampling.num_frames": 99 }), /allowed values/);
  assert.equal(resolveConfig(data, { "default_request.sampling.seed": null }).config.default_request.sampling.seed, null);
  assert.equal(resolveConfig(data, { "default_request.sampling.seed": 0 }).config.default_request.sampling.seed, 0);
});

test("arrays replace complete values and missing defaults remain inherited", () => {
  const data = catalog();
  data.controls.push({ path: "default_request.prompt", schema: { anyOf: [
    { type: "string" }, { type: "array", items: { type: "string" } }, { type: "null" } ] } });
  assert.equal(resolveConfig(data).config.default_request.prompt, undefined);
  assert.deepEqual(resolveConfig(data, { "default_request.prompt": ["One", "Two"] }).config.default_request.prompt, ["One", "Two"]);
  assert.throws(() => resolveConfig(data, { "default_request.prompt": [1] }));
});

test("sample requests follow effective alias, host and port without repinning request defaults", () => {
  const result = resolveConfig(catalog(), { "server.port": 9000, "server.served_model_name": "my-video" });
  assert.match(result.clientCommand, /http:\/\/127\.0\.0\.1:9000\/v1\/videos\/sync/);
  assert.deepEqual(result.clientRequest, { model: "my-video", prompt: "A river flowing through a peaceful forest" });
  assert.match(result.clientCommand, /--output output.mp4/);
  for (const host of ["::", "[::]"]) assert.match(resolveConfig(catalog(), { "server.host": host }).clientCommand, /\[::1\]:8000/);
  assert.equal(resolveConfig(catalog(), { "server.served_model_name": null }).clientRequest.model, "Example/Video");
  const image = catalog(); image.workload = "i2v";
  assert.equal(resolveConfig(image).clientRequest.input_reference, "/absolute/path/to/first-frame.png");
});

test("environment variables are present and shell quoted without evaluation", () => {
  const data = catalog();
  data.env = { BACKEND: "it's literal $(touch /never-run)" };
  const result = resolveConfig(data);
  assert.match(result.command, /^BACKEND=/);
  assert.ok(result.command.endsWith("fastvideo serve --config config.yaml"));
  const values = ["", "a b", "it's literal", "$(printf unexpected)", "`date`"];
  const probe = spawnSync("/bin/sh", ["-c", `printf '%s\\0' ${values.map(shellQuote).join(" ")}`], { encoding: "utf8" });
  assert.equal(probe.status, 0);
  assert.deepEqual(probe.stdout.split("\0").slice(0, -1), values);
});

test("YAML preserves literal types and scientific notation", () => {
  assert.equal(toYaml({ off: false, zero: 0, empty: "", auto: null, list: ["a", true] }),
    'off: false\nzero: 0\nempty: ""\nauto: null\nlist: ["a",true]\n');
  assert.equal(toYaml({ small: 1e-7, large: 1e21, text: "1e-7" }), 'small: 1.0e-7\nlarge: 1.0e+21\ntext: "1e-7"\n');
});

function exportedCatalogs() {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const output = mkdtempSync(join(tmpdir(), "fastvideo-cookbook-recipes-"));
  const localPython = join(root, ".venv/bin/python");
  const python = process.env.PYTHON || (existsSync(localPython) ? localPython : "python3");
  try {
    const result = spawnSync(python, ["docs/cookbook_config.py", "--output-dir", output], { cwd: root, encoding: "utf8" });
    assert.equal(result.status, 0, result.error?.message || result.stderr || result.stdout);
    const index = JSON.parse(readFileSync(join(output, "index.json"), "utf8"));
    const catalogs = new Map(index.models.flatMap((model) => model.deployments).map((recipe) => [recipe.id,
      JSON.parse(readFileSync(join(output, recipe.catalog_url), "utf8"))]));
    return { index, catalogs };
  } finally { rmSync(output, { recursive: true, force: true }); }
}

const { index, catalogs } = exportedCatalogs();
test("generated index groups only authored deployments and MLX preserves its opaque request defaults", () => {
  assert.deepEqual(index.models.map((model) => [model.id, model.deployments.map((item) => item.id)]), [
    ["fasth3-8step", ["fasth3-8step/cuda-rest", "fasth3-8step/mlx-rest"]],
    ["fastwan21", ["fastwan21/cuda-rest"]],
    ["ltx2-distilled", ["ltx2-distilled/cuda-streaming"]],
    ["wan21-i2v", ["wan21-i2v/cuda-rest"]],
  ]);
  const cuda = catalogs.get("fasth3-8step/cuda-rest"), mlx = catalogs.get("fasth3-8step/mlx-rest");
  assert.equal(cuda.model.id, "FastVideo/FastVideo-FastH3-8-Step-V2");
  assert.equal(mlx.model.id, cuda.model.id);
  assert.equal(mlx.runtime.backend, "mlx");
  assert.equal(mlx.runtime.hardware_label, "Apple Silicon");
  assert.deepEqual(mlx.hardware, []);
  assert.deepEqual(mlx.controls.map((control) => control.path), [
    "server.host", "server.port", "server.output_dir", "server.served_model_name",
    "generator.model_root", "generator.mlx_checkpoint", "generator.prompt_cache_dir",
    "generator.vae_dtype", "generator.vsa_sparsity",
  ]);
  assert.equal(resolveConfig(cuda).command, "fastvideo serve --config config.yaml");
  assert.equal(resolveConfig(mlx).command, "python -m fastvideo.entrypoints.openai.mlx_server --config config.yaml");
  const result = resolveConfig(mlx, { "generator.model_root": "/local/H3", "generator.vae_dtype": "bf16",
    "generator.vsa_sparsity": 0.5, "server.port": 9002 });
  assert.equal(result.config.generator.model_root, "/local/H3");
  assert.equal(result.config.generator.vae_dtype, "bf16");
  assert.equal(result.config.generator.vsa_sparsity, 0.5);
  assert.deepEqual(result.config.default_request, mlx.base_config.default_request);
  assert.equal(result.config.generator.vsa_tile_size, mlx.base_config.generator.vsa_tile_size);
  assert.match(result.clientCommand, /127\.0\.0\.1:9002/);
  assert.throws(() => resolveConfig(mlx, { "default_request.sampling.num_frames": 61 }), /not an editable recipe control/);
  assert.throws(() => resolveConfig(mlx, { "generator.vsa_sparsity": 1 }), /must be < 1/);
  assert.throws(() => resolveConfig(mlx, { "generator.vae_dtype": "int8" }), /allowed values/);
});

test("maintained I2V sample supplies a source image and preserves its two-GPU baseline", () => {
  const data = catalogs.get("wan21-i2v/cuda-rest"), result = resolveConfig(data);
  assert.equal(data.workload, "i2v");
  assert.equal(result.config.generator.engine.num_gpus, 2);
  assert.deepEqual(result.config, data.base_config);
  assert.equal(result.clientRequest.model, "wan21-i2v-14b");
  assert.equal(result.clientRequest.input_reference, "/absolute/path/to/first-frame.png");
  assert.match(result.clientCommand, /\/v1\/videos\/sync/);
  assert.equal(result.websocketUrl, undefined);
});

test("streaming offers a health check and effective WebSocket endpoint without a REST video request", () => {
  const data = catalogs.get("ltx2-distilled/cuda-streaming"), initial = resolveConfig(data);
  assert.equal(data.runtime.interface, "websocket");
  assert.equal(initial.command, "fastvideo serve --config config.yaml");
  assert.equal(initial.clientCommand, "curl --fail-with-body http://127.0.0.1:8009/health");
  assert.equal(initial.websocketUrl, "ws://127.0.0.1:8009/v1/stream");
  assert.equal(initial.clientRequest, null);
  assert.doesNotMatch(initial.clientCommand, /--data|--output|\/v1\/videos/);
  assert.deepEqual(initial.config, data.base_config);
  assert.equal(data.runtime.client_guide_url, "../../design/server_contracts/streaming/");
  assert.deepEqual(data.controls.map((item) => item.path), [
    "server.host", "server.port", "default_request.sampling.num_frames",
    "default_request.sampling.height", "default_request.sampling.width",
  ]);
  for (const [host, clientHost] of [["0.0.0.0", "127.0.0.1"], ["::", "[::1]"], ["[::]", "[::1]"],
    ["2001:db8::5", "[2001:db8::5]"], ["video.example.test", "video.example.test"]]) {
    const result = resolveConfig(data, { "server.host": host, "server.port": 9012,
      "default_request.sampling.num_frames": 65 });
    assert.equal(result.clientCommand, `curl --fail-with-body ${shellQuote(`http://${clientHost}:9012/health`)}`);
    assert.equal(result.websocketUrl, `ws://${clientHost}:9012/v1/stream`);
    assert.deepEqual(result.config.streaming, data.base_config.streaming);
    assert.deepEqual(result.config.generator, data.base_config.generator);
  }
  assert.throws(() => resolveConfig(data, { "server.served_model_name": "stream" }), /not an editable recipe control/);
  assert.deepEqual(resolveConfig(data), initial);
});

test("every generated recipe resolves without edits and baseline defaults win", () => {
  assert.deepEqual(Object.keys(index), ["models"]);
  for (const recipe of index.models.flatMap((model) => model.deployments)) {
    const data = catalogs.get(recipe.id);
    assert.equal(data.id, recipe.id);
    assert.deepEqual(resolveConfig(data).config, data.base_config);
  }
  const data = catalogs.get("fastwan21/cuda-rest");
  const result = resolveConfig(data);
  assert.equal(result.config.default_request.sampling.num_frames, 81);
  assert.equal(result.config.default_request.sampling.num_inference_steps, 3);
  assert.deepEqual(data.controls.map((control) => control.path), [
    "server.host", "server.port", "server.output_dir", "server.served_model_name",
    "generator.engine.offload.dit_layerwise", "generator.engine.offload.text_encoder", "generator.engine.offload.vae",
    "generator.engine.compile.enabled", "default_request.sampling.num_frames", "default_request.sampling.height",
    "default_request.sampling.width", "default_request.sampling.fps", "default_request.sampling.seed",
  ]);
  assert.match(result.command, /FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN/);
  assert.deepEqual(hardwareView(data, result.config).records.map(({ id, status }) => ({ id, status })), [
    { id: "nvidia-h100-sxm-80gb", status: "unverified" },
    { id: "nvidia-rtx-4090", status: "unverified" },
  ]);
  assert.equal(guideUrl(data, "https://example.test/project/assets/cookbook-config/index.json"),
    "https://example.test/project/cookbook/guides/fastwan21-serving/");
});

test("expanded server controls edit as ordinary leaves and reset preserves the native baseline", () => {
  const data = catalogs.get("fastwan21/cuda-rest");
  const initial = resolveConfig(data);
  const result = resolveConfig(data, {
    "server.host": "::", "server.port": 9001,
    "server.served_model_name": "reviewed-recipe", "server.output_dir": "outputs/custom",
  });
  assert.deepEqual(result.config.server, {
    host: "::", port: 9001, served_model_name: "reviewed-recipe", output_dir: "outputs/custom",
  });
  assert.match(result.clientCommand, /\[::1\]:9001/);
  assert.equal(result.clientRequest.model, "reviewed-recipe");
  assert.deepEqual(result.config.generator, initial.config.generator);
  assert.deepEqual(result.config.default_request, initial.config.default_request);
  assert.deepEqual(resolveConfig(data), initial);
});

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

async function browserFixture() {
  const nodes = new Map(), requests = [], failures = new Set();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element());
    return nodes.get(selector);
  };
  const root = new Element();
  root.dataset.metadata = "https://example.test/project/assets/cookbook-config/index.json";
  root.querySelector = node;
  const copyButtons = ["yaml", "command", "clientCommand"].map((name) => {
    const button = new Element("button"); button.dataset.configCopy = name; return button;
  });
  root.querySelectorAll = (selector) => selector.includes("data-config-download") ?
    [...copyButtons, node("[data-config-download]")] : copyButtons;
  const document = { readyState: "complete", querySelectorAll: () => [root], createElement: (tag) => new Element(tag) };
  const fetch = async (url) => {
    if (url === root.dataset.metadata) return { ok: true, url, json: async () => index };
    const id = url.split("/recipes/")[1]?.replace(/\.json$/, "");
    requests.push(id);
    if (failures.has(id)) throw new Error("Deployment unavailable");
    assert.ok(catalogs.has(id), `Unexpected catalog request: ${url}`);
    return { ok: true, json: async () => catalogs.get(id) };
  };
  runInNewContext(readFileSync(new URL("../../docs/assets/cookbook-config.js", import.meta.url), "utf8"),
    { document, fetch, URL, AbortController, setTimeout, FastVideoSchema: require("../../docs/assets/cookbook-validator.js") });
  await new Promise(setImmediate);
  const element = (name) => node(`[data-config-${name}]`);
  const descendants = (parent) => parent.children.flatMap((child) => typeof child === "string" ? [] :
    [child, ...descendants(child)]);
  const control = (path) => descendants(element("controls")).find((item) => item.dataset.configPath === path);
  return { element, control, requests, failures, code: (name) => node(`[data-config-code="${name}"]`).textContent };
}

test("model and deployment selectors reset edits, update runtime guidance and clear failed deployment data", async () => {
  const { element, control, requests, failures, code } = await browserFixture();
  assert.deepEqual(element("model-picker").children.map((item) => item.value),
    ["fasth3-8step", "fastwan21", "ltx2-distilled", "wan21-i2v"]);
  assert.deepEqual(element("deployment-picker").children.map((item) => item.value),
    ["fasth3-8step/cuda-rest", "fasth3-8step/mlx-rest"]);
  assert.deepEqual(requests, ["fasth3-8step/cuda-rest"]);
  control("server.port").value = "9001";
  control("server.port").trigger("input");
  assert.match(code("yaml"), /port: 9001/);
  element("deployment-picker").value = "fasth3-8step/mlx-rest";
  await element("deployment-picker").trigger("change");
  assert.match(code("yaml"), /port: 8000/);
  assert.match(code("command"), /python -m fastvideo.entrypoints.openai.mlx_server/);
  assert.equal(element("install").textContent, "MLX REST installation guide");
  assert.equal(element("topology").textContent, "Hardware family: Apple Silicon.");
  assert.equal(element("hardware-table").hidden, true);
  assert.equal(element("hardware-note").hidden, true);
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
  element("model-picker").value = "fastwan21";
  await element("model-picker").trigger("change");
  assert.deepEqual(element("deployment-picker").children.map((item) => item.value), ["fastwan21/cuda-rest"]);
  assert.equal(element("guide").href, "https://example.test/project/cookbook/guides/fastwan21-serving/");
  assert.equal(element("hardware-rows").children.length, 2);
  assert.equal(element("hardware-table").hidden, false);
  control("server.port").value = "9003";
  control("server.port").trigger("input");
  assert.match(element("hardware-state").textContent, /^Custom configuration — unverified/);
  element("reset").trigger("click");
  assert.match(code("yaml"), /port: 8000/);
  assert.match(element("hardware-state").textContent, /^Recipe baseline/);
  failures.add("fasth3-8step/cuda-rest");
  element("model-picker").value = "fasth3-8step";
  await element("model-picker").trigger("change");
  assert.equal(element("output").hidden, true);
  assert.equal(element("hardware").hidden, true);
  assert.equal(element("hardware-rows").children.length, 0);
  assert.equal(element("guide").hidden, true);
  assert.equal(element("guide").href, undefined);
  assert.equal(element("controls").children.length, 0);
  assert.match(element("error").textContent, /Deployment unavailable/);
});

test("streaming UI updates its endpoint and clears protocol details when returning to REST or failing", async () => {
  const { element, control, code, failures } = await browserFixture();
  element("model-picker").value = "ltx2-distilled";
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
  element("model-picker").value = "wan21-i2v";
  await element("model-picker").trigger("change");
  assert.equal(element("client-title").textContent, "4. Send a sample request");
  assert.match(code("clientCommand"), /input_reference.*first-frame\.png/);
  assert.equal(element("streaming-client").hidden, true);
  assert.equal(element("websocket-url").textContent, "");
  assert.equal(element("client-guide").href, undefined);
  assert.equal(element("request-note").hidden, false);
  element("model-picker").value = "ltx2-distilled";
  await element("model-picker").trigger("change");
  failures.add("fastwan21/cuda-rest");
  element("model-picker").value = "fastwan21";
  await element("model-picker").trigger("change");
  assert.equal(element("output").hidden, true);
  assert.equal(element("streaming-client").hidden, true);
  assert.equal(element("websocket-url").textContent, "");
  assert.equal(element("client-guide").href, undefined);
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
  const load = createCatalogLoader({ models }, "https://example.test/project/assets/cookbook-config/index.json",
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
