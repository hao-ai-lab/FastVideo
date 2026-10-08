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
const { loadIndex, loadDeployment, getOptions, resolveConfig } =
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

test("core exposes only reusable functions and importing it has no DOM or fetch side effects", () => {
  const context = {
    document: new Proxy({}, { get: () => assert.fail("Core must not read the DOM") }),
    document$: { subscribe: () => assert.fail("Core must not subscribe to page navigation") },
    fetch: () => assert.fail("Importing the core must not fetch"),
  };
  runInNewContext(readFileSync(new URL("../../docs/assets/cookbook-config.js", import.meta.url), "utf8"), context);
  assert.deepEqual(Object.keys(context.FastVideoConfigCookbook).sort(),
    ["getOptions", "loadDeployment", "loadIndex", "resolveConfig"]);
});

test("getOptions returns all enabled metadata and current values without substituting declared defaults", () => {
  const data = catalog();
  data.base_config.default_request.prompt = ["First prompt"];
  data.controls.push({ path: "default_request.prompt", schema: { type: "array", items: { type: "string" } } });
  const before = JSON.stringify(data);
  const options = getOptions(data);
  assert.deepEqual(options.map((item) => item.path), data.controls.map((item) => item.path));
  options.forEach((item, index) => assert.deepEqual(item.schema, data.controls[index].schema));
  const value = (items, path) => items.find((item) => item.path === path).value;
  assert.equal(value(options, "default_request.sampling.num_frames"), 81);
  assert.equal(value(options, "default_request.sampling.guidance_scale"), undefined);
  const updated = getOptions(data, resolveConfig(data, { "generator.engine.compile.enabled": false,
    "default_request.sampling.seed": null, "default_request.sampling.guidance_scale": 0 }).config);
  assert.equal(value(updated, "generator.engine.compile.enabled"), false);
  assert.equal(value(updated, "default_request.sampling.seed"), null);
  assert.equal(value(updated, "default_request.sampling.guidance_scale"), 0);
  const prompt = options.find((item) => item.path === "default_request.prompt");
  prompt.value.push("A UI edit");
  prompt.schema.items.type = "number";
  assert.equal(JSON.stringify(data), before);
});

test("loadIndex and loadDeployment preserve the response URL, metadata and caller abort signal", async () => {
  const model = { id: "example", deployments: [{ id: "example/rest", catalog_url: "recipes/example/rest.json" }] };
  const controller = new AbortController(), calls = [];
  const data = { id: "example/rest", controls: [], guide: { url: "../../cookbook/guide/" } };
  const context = await loadIndex("https://example.test/old/index.json", {
    signal: controller.signal,
    fetcher: async (url, options) => {
      calls.push({ url, signal: options.signal });
      return { ok: true, url: "https://example.test/project/assets/cookbook-config/index.json",
        json: async () => ({ models: [model] }) };
    },
  });
  assert.deepEqual(context, { models: [model], url: "https://example.test/project/assets/cookbook-config/index.json" });
  const result = await loadDeployment(context, "example/rest", {
    signal: controller.signal,
    fetcher: async (url, options) => {
      calls.push({ url, signal: options.signal });
      return { ok: true, json: async () => data };
    },
  });
  assert.equal(calls[1].url, "https://example.test/project/assets/cookbook-config/recipes/example/rest.json");
  assert.ok(calls.every((call) => call.signal === controller.signal));
  assert.deepEqual(result, data);
  await assert.rejects(loadDeployment(context, "example/mlx", { fetcher: () => assert.fail("Unlisted deployment fetched") }),
    /No exported deployment/);
  await assert.rejects(loadDeployment(context, "example/rest", { fetcher: async () =>
    ({ ok: true, json: async () => ({ id: "other/rest" }) }) }), /Unexpected deployment/);
  await assert.rejects(loadIndex(context.url, { fetcher: async () => ({ ok: false, status: 404 }) }), /HTTP 404/);
  await assert.rejects(loadDeployment(context, "example/rest", { fetcher: async () => ({ ok: false, status: 503 }) }), /HTTP 503/);
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
  data.runtime.launch_argv = ["printf", "%s\\0", ...values];
  const probe = spawnSync("/bin/sh", ["-c", resolveConfig(data).command], { encoding: "utf8" });
  assert.equal(probe.status, 0);
  assert.deepEqual(probe.stdout.split("\0").slice(0, -1), values);
});

test("YAML preserves literal types and scientific notation", () => {
  const yaml = (value) => {
    const data = catalog(); data.base_config = value; data.controls = [];
    return resolveConfig(data).yaml;
  };
  assert.equal(yaml({ off: false, zero: 0, empty: "", auto: null, list: ["a", true] }),
    'off: false\nzero: 0\nempty: ""\nauto: null\nlist: ["a",true]\n');
  assert.equal(yaml({ small: 1e-7, large: 1e21, text: "1e-7" }), 'small: 1.0e-7\nlarge: 1.0e+21\ntext: "1e-7"\n');
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
  const ids = index.models.flatMap((model) => model.deployments.map((deployment) => deployment.id));
  assert.equal(new Set(index.models.map((model) => model.id)).size, index.models.length);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual([...catalogs.keys()], ids);
  for (const model of index.models) {
    assert.ok(model.deployments.length);
    for (const deployment of model.deployments) {
      const data = catalogs.get(deployment.id);
      assert.equal(data.model.key, model.id);
      assert.equal(data.model.id, model.model_id);
      assert.equal(data.id, `${model.id}/${data.deployment.id}`);
      assert.equal(data.deployment.label, deployment.label);
      assert.equal(data.runtime.id, deployment.runtime);
    }
  }
  const cuda = catalogs.get("fasth3-8step/cuda-rest"), mlx = catalogs.get("fasth3-8step/mlx-rest");
  assert.equal(cuda.model.id, "FastVideo/FastVideo-FastH3-8-Step-V2");
  assert.equal(mlx.model.id, cuda.model.id);
  assert.equal(mlx.runtime.backend, "mlx");
  assert.equal(mlx.runtime.hardware_label, "Apple Silicon");
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
    assert.ok(result.clientCommand.includes(`http://${clientHost}:9012/health`));
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
