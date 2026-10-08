import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
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

test("caller edits, catalog and resolved configuration do not share nested values", () => {
  const data = catalog();
  data.base_config.default_request.prompt = [{ text: "Baseline", tags: ["baseline"] }];
  data.controls.push({ path: "default_request.prompt", schema: {
    type: "array", items: { type: "object", properties: {
      text: { type: "string" }, tags: { type: "array", items: { type: "string" } },
    }, required: ["text", "tags"], additionalProperties: false },
  } });
  const selections = { "default_request.prompt": [{ text: "Selected", tags: ["selected"] }] };
  const catalogBefore = JSON.stringify(data), selectionsBefore = JSON.stringify(selections);
  const result = resolveConfig(data, selections);
  result.config.default_request.prompt[0].tags.push("Result edit");
  result.config.default_request.prompt.push({ text: "Another result", tags: [] });
  result.config.generator.pipeline.experimental.hidden.push(1);
  assert.equal(JSON.stringify(selections), selectionsBefore);
  assert.equal(JSON.stringify(data), catalogBefore);
  const resultBefore = JSON.stringify(result.config);
  selections["default_request.prompt"][0].tags.push("Caller edit");
  selections["default_request.prompt"].push({ text: "Another caller", tags: [] });
  data.base_config.generator.pipeline.experimental.hidden.push(2);
  assert.equal(JSON.stringify(result.config), resultBefore);
  assert.deepEqual(data.base_config.default_request.prompt, [{ text: "Baseline", tags: ["baseline"] }]);
});
