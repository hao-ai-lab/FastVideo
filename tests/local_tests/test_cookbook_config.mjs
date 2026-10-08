import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const require = createRequire(import.meta.url);
const { resolveConfig, createCatalogLoader, getDefaultConfig, getSchemaField, getEditableFields, toYaml, shellQuote } = require("../../docs/assets/cookbook-config.js");
const object = (properties, required = []) => ({ type: "object", properties, required, additionalProperties: false });

function catalog() {
  return {
    model: { id: "Example/Video", label: "Video model", workload: "t2v", preset: "video" },
    runtime: { id: "cuda", label: "CUDA", launch_argv: ["fastvideo", "serve", "--config", "config.yaml"] },
    schema: {
      $schema: "https://json-schema.org/draft/2020-12/schema",
      ...object({
        generator: object({
          model_path: { type: "string", const: "Example/Video" },
          engine: object({
            num_gpus: { type: "integer", minimum: 1, default: 1 },
            offload: object({ vae: { type: "boolean", default: true } }),
          }),
        }, ["model_path"]),
        server: object({
          host: { type: "string", pattern: "^[A-Za-z0-9_.:-]+$", default: "127.0.0.1" },
          port: { type: "integer", minimum: 1, maximum: 65535, default: 8000 },
        }),
        default_request: object({
          output: object({ return_frames: { type: "boolean", const: false } }),
          sampling: object({
            num_frames: { type: "integer", minimum: 1, default: 121 },
            width: { type: "integer", minimum: 8, multipleOf: 8, default: 1280 },
            guidance_scale: { type: "number", minimum: 0, default: 5 },
            seed: { anyOf: [{ type: "integer", minimum: 0 }, { type: "null" }], default: null },
          }),
        }),
      }, ["generator"]),
    },
    sources: {},
  };
}

const field = (data, path) => getSchemaField(data.schema, path);

test("standard defaults and const values build config without hardcoded frame counts", () => {
  const data = catalog();
  const before = JSON.stringify(data);
  const result = resolveConfig(data);
  assert.equal(result.values.default_request.sampling.num_frames, 121);
  assert.equal(result.values.default_request.sampling.seed, null);
  assert.equal(result.values.default_request.output.return_frames, false);
  assert.deepEqual(result.config, { generator: { model_path: "Example/Video" } });
  assert.equal(result.config.generator.model_path, "Example/Video");
  assert.equal(JSON.stringify(data), before);
  const another = catalog();
  field(another, "default_request.sampling.num_frames").default = 125;
  field(another, "default_request.sampling.guidance_scale").default = 0;
  assert.equal(getDefaultConfig(another.schema).default_request.sampling.guidance_scale, 0);
  assert.equal(resolveConfig(another).values.default_request.sampling.num_frames, 125);
});

test("UI field discovery includes new exported leaves and omits fixed constants", () => {
  const data = catalog();
  data.schema.properties.generator.properties.engine.properties.compile = object({
    enabled: { type: "boolean", default: false, title: "Compile transformer" },
  });
  const fields = getEditableFields(data.schema);
  assert.equal(fields.some((item) => item.path === "generator.model_path"), false);
  assert.equal(fields.some((item) => item.path === "default_request.output.return_frames"), false);
  const added = fields.find((item) => item.path === "generator.engine.compile.enabled");
  assert.equal(added.field.title, "Compile transformer");
  assert.equal(added.field.type, "boolean");
  assert.equal(resolveConfig(data, { [added.path]: true }).config.generator.engine.compile.enabled, true);
});

test("numeric selections change YAML while the launch command stays short", () => {
  const result = resolveConfig(catalog(), {
    "default_request.sampling.num_frames": 82,
    "default_request.sampling.guidance_scale": 4.5,
    "generator.engine.num_gpus": 2,
    "generator.engine.offload.vae": false,
  });
  assert.equal(result.config.default_request.sampling.num_frames, 82);
  assert.match(result.yaml, /num_frames: 82/);
  assert.match(result.yaml, /guidance_scale: 4.5/);
  assert.match(result.yaml, /vae: false/);
  assert.equal(result.command, "fastvideo serve --config config.yaml");
  assert.deepEqual(result.argv, ["fastvideo", "serve", "--config", "config.yaml"]);
});

test("Ajv rejects unknown fields, wrong types, nonfinite numbers and declared bounds", () => {
  const data = catalog();
  for (const selections of [
    { "server.extra": 1 }, { "server.port": 0 }, { "server.port": 65536 }, { "server.port": 1.5 },
    { "server.port": "8000" }, { "server.port": NaN }, { "default_request.sampling.guidance_scale": Infinity },
    { "generator.engine.offload.vae": "false" }, { "default_request.sampling.num_frames": 0 },
    { "default_request.sampling.num_frames": "" }, null, [],
  ]) assert.throws(() => resolveConfig(data, selections));
  assert.throws(() => resolveConfig(data, { "server.extra": 1 }), /server.extra.*additional properties/);
  assert.throws(() => resolveConfig(data, { "server.port": 0 }), /server.port/);
});

test("Ajv evaluates enum, exclusive bounds, nullable fields and multipleOf", () => {
  const data = catalog();
  field(data, "generator.engine.num_gpus").enum = [1, 2, 4];
  Object.assign(field(data, "default_request.sampling.guidance_scale"), { exclusiveMinimum: 0, exclusiveMaximum: 10 });
  assert.throws(() => resolveConfig(data, { "generator.engine.num_gpus": 3 }), /allowed values/);
  assert.throws(() => resolveConfig(data, { "default_request.sampling.guidance_scale": 0 }), /> 0/);
  assert.throws(() => resolveConfig(data, { "default_request.sampling.guidance_scale": 10 }), /< 10/);
  assert.throws(() => resolveConfig(data, { "default_request.sampling.width": 1279 }), /multiple of 8/);
  assert.equal(resolveConfig(data, { "default_request.sampling.seed": null }).config.default_request.sampling.seed, null);
  assert.equal(resolveConfig(data, { "default_request.sampling.seed": 0 }).config.default_request.sampling.seed, 0);
  assert.throws(() => resolveConfig(data, { "default_request.sampling.seed": -1 }));
});

test("local $ref definitions provide defaults and are validated by Ajv", () => {
  const data = catalog();
  data.schema.$defs = { GpuCount: { type: "integer", enum: [1, 2, 4], default: 1 } };
  data.schema.properties.generator.properties.engine.properties.num_gpus = { $ref: "#/$defs/GpuCount", default: 2 };
  assert.deepEqual(field(data, "generator.engine.num_gpus").enum, [1, 2, 4]);
  assert.equal(resolveConfig(data).values.generator.engine.num_gpus, 2);
  assert.throws(() => resolveConfig(data, { "generator.engine.num_gpus": 3 }), /allowed values/);
});

test("Ajv enforces cross-field if/then rules without frontend-specific conditions", () => {
  const data = catalog();
  const engine = data.schema.properties.generator.properties.engine;
  engine.if = { type: "object", properties: { num_gpus: { type: "integer", minimum: 2 } }, required: ["num_gpus"] };
  engine.then = { type: "object", properties: {
    offload: { type: "object", properties: { vae: { type: "boolean", const: false } } },
  } };
  assert.throws(() => resolveConfig(data, { "generator.engine.num_gpus": 2 }), /constant/);
  assert.equal(resolveConfig(data, {
    "generator.engine.num_gpus": 2, "generator.engine.offload.vae": false,
  }).config.generator.engine.num_gpus, 2);
});

test("validators are reused during edits and separate catalogs may share a schema ID", () => {
  const library = require("../../docs/assets/cookbook-validator.js");
  const compile = library.createValidator;
  let compilations = 0;
  // The bundled export is read-only; count compilations through the module boundary.
  const modulePath = require.resolve("../../docs/assets/cookbook-validator.js");
  const original = require.cache[modulePath].exports;
  require.cache[modulePath].exports = { createValidator: (schema) => { compilations++; return compile(schema); } };
  try {
    const first = catalog(), second = catalog();
    first.schema.$id = second.schema.$id = "https://example.test/model-config";
    resolveConfig(first);
    resolveConfig(first, { "server.port": 9000 });
    assert.equal(compilations, 1);
    field(second, "server.port").maximum = 8500;
    resolveConfig(second);
    assert.throws(() => resolveConfig(second, { "server.port": 9000 }), /<= 8500/);
    assert.equal(compilations, 2);
    assert.equal(resolveConfig(first, { "server.port": 9000 }).config.server.port, 9000);
  } finally { require.cache[modulePath].exports = original; }
});

test("required fields and model/output constants remain schema constraints", () => {
  const missing = catalog();
  delete field(missing, "server.port").default;
  missing.schema.properties.server.required = ["port"];
  assert.throws(() => resolveConfig(missing), /server.port.*required property/);
  assert.throws(() => resolveConfig(catalog(), { "generator.model_path": "Other/Model" }), /constant/);
  assert.throws(() => resolveConfig(catalog(), { "default_request.output.return_frames": true }), /constant/);
});

test("YAML preserves string types, false, empty text, arrays and nested mappings", () => {
  assert.equal(toYaml({
    server: { host: "false", output_dir: "", text: "A: value\nwith 'quotes'", active: false },
    list: ["a", true], empty: {}, missing: null,
  }), 'server:\n  host: "false"\n  output_dir: ""\n  text: "A: value\\nwith \'quotes\'"\n  active: false\nlist: ["a",true]\nempty: {}\nmissing: null\n');
});

test("scientific-notation numbers stay YAML floats without rewriting quoted strings", () => {
  assert.equal(toYaml({ small: 1e-7, large: 1e21, fraction: 1.5e-7, text: "1e-7", list: [1e-7, { large: 1e21 }] }),
    'small: 1.0e-7\nlarge: 1.0e+21\nfraction: 1.5e-7\ntext: "1e-7"\nlist: [1.0e-7,{"large":1.0e+21}]\n');
  const result = resolveConfig(catalog(), { "default_request.sampling.guidance_scale": 0.0000001 });
  assert.match(result.yaml, /guidance_scale: 1.0e-7/);
});

test("host and port edits are saved; synthetic schema constraints still apply", () => {
  const result = resolveConfig(catalog(), { "server.host": "0.0.0.0", "server.port": 9000 });
  assert.equal(result.config.server.host, "0.0.0.0");
  assert.equal(result.config.server.port, 9000);
  assert.match(result.yaml, /host: "0.0.0.0"/);
  for (const host of ["", " ", "host/path", "http://localhost", "a\nb"]) {
    assert.throws(() => resolveConfig(catalog(), { "server.host": host }), /server.host/);
  }
  assert.match(resolveConfig(catalog(), { "server.host": "true" }).yaml, /host: "true"/);
});

test("shell quoting preserves spaces and executable-looking text in launch arguments", () => {
  const values = ["", "a b", "it's literal", "$(printf unexpected)", "`date`"];
  const probe = spawnSync("/bin/sh", ["-c", `printf '%s\\0' ${values.map(shellQuote).join(" ")}`], { encoding: "utf8" });
  assert.equal(probe.status, 0);
  assert.deepEqual(probe.stdout.split("\0").slice(0, -1), values);
  const data = catalog();
  data.runtime.launch_argv[3] = "demo config.yaml";
  assert.equal(resolveConfig(data).command, "fastvideo serve --config 'demo config.yaml'");
});

test("malformed canonical paths cannot overwrite object prototypes", () => {
  assert.throws(() => resolveConfig(catalog(), { "generator.__proto__.polluted": true }), /Invalid configuration path/);
  assert.equal({}.polluted, undefined);
});

function exportedCatalogs() {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const output = mkdtempSync(join(tmpdir(), "fastvideo-cookbook-config-"));
  const localPython = join(root, ".venv/bin/python");
  const python = process.env.PYTHON || (existsSync(localPython) ? localPython : "python3");
  try {
    const result = spawnSync(python, ["docs/cookbook_config.py", "--output-dir", output,
      "--models-file", "docs/cookbook/config-builder-models.yaml"], { cwd: root, encoding: "utf8" });
    assert.equal(result.status, 0, result.error?.message || result.stderr || result.stdout);
    const index = JSON.parse(readFileSync(join(output, "index.json"), "utf8"));
    const catalogs = new Map(index.models.map((model) => [model.id,
      JSON.parse(readFileSync(join(output, model.catalog_url), "utf8"))]));
    return { index, catalogs };
  } finally { rmSync(output, { recursive: true, force: true }); }
}

const { index, catalogs } = exportedCatalogs();
const wanId = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";

test("real catalog uses serving defaults without endpoint-derived limits", () => {
  const data = catalogs.get(wanId);
  const result = resolveConfig(data);
  assert.equal(Object.hasOwn(data, "fields"), false);
  assert.equal(Object.hasOwn(data, "base_config"), false);
  assert.equal(result.config.generator.model_path, data.model.id);
  assert.equal(result.values.default_request.sampling.num_frames, 121);
  assert.equal(result.config.default_request, undefined);
  assert.equal(field(data, "default_request.sampling.num_frames").default, 121);
  assert.equal(field(data, "default_request.sampling.num_frames").minimum, -Number.MAX_SAFE_INTEGER);
  assert.equal(field(data, "default_request.sampling.num_frames").multipleOf, undefined);
  assert.equal(result.values.default_request.output.return_frames, true);
  const updated = resolveConfig(data, { "default_request.sampling.num_frames": 82, "generator.engine.offload.vae": false });
  assert.match(updated.yaml, /num_frames: 82/);
  assert.match(updated.yaml, /vae: false/);
  assert.equal(updated.command, "fastvideo serve --config config.yaml");
  // These are structurally valid serving values; endpoint/runtime policy is not exported.
  const unbounded = resolveConfig(data, {
    "default_request.sampling.num_frames": 0,
    "default_request.sampling.num_inference_steps": 201,
    "default_request.sampling.guidance_scale": 21,
  });
  assert.equal(unbounded.config.default_request.sampling.num_inference_steps, 201);
  assert.throws(() => resolveConfig(data, { "default_request.sampling.seed": Number.MAX_SAFE_INTEGER + 1 }));
});

test("all selected catalogs validate independently without mutating the model index", () => {
  const before = JSON.stringify(index);
  assert.deepEqual(Object.keys(index), ["models"]);
  for (const model of index.models) {
    assert.deepEqual(Object.keys(model).sort(), ["catalog_url", "id", "label", "workload_types"]);
    assert.match(model.catalog_url, /^models\/[a-f0-9]{64}\.json$/);
    const data = catalogs.get(model.id);
    assert.deepEqual(Object.keys(data).sort(), ["model", "runtime", "schema"]);
    assert.deepEqual(Object.keys(data.model).sort(), ["id", "label", "workload_types"]);
    assert.equal(resolveConfig(data).config.generator.model_path, model.id);
  }
  assert.equal(JSON.stringify(index), before);
});

test("model switching selects different defaults and does not reuse edits", () => {
  const wan = catalogs.get(wanId);
  const other = catalogs.get(index.models.find((model) => model.id !== wanId).id);
  const initialOther = resolveConfig(other);
  const edited = resolveConfig(wan, { "default_request.sampling.num_frames": 81 });
  assert.equal(edited.values.default_request.sampling.num_frames, 81);
  assert.deepEqual(resolveConfig(other), initialOther);
  assert.equal(resolveConfig(wan).values.default_request.sampling.num_frames, 121);
});

test("JSON pointers preserve literal dots in public legacy arguments and nullable values", () => {
  const data = catalogs.get(wanId);
  const property = "vae_config.load_encoder";
  const pointer = `/generator/pipeline/experimental/${property}`;
  assert.ok(getEditableFields(data.schema).some((item) => item.pointer === pointer));
  assert.equal(getSchemaField(data.schema, pointer).type, "boolean");
  const result = resolveConfig(data, {
    [pointer]: true,
    "generator.pipeline.vae_tiling": null,
    "generator.pipeline.experimental.text_encoder_precisions": ["fp32"],
  });
  assert.equal(result.config.generator.pipeline.experimental[property], true);
  assert.equal(result.config.generator.pipeline.experimental.vae_config, undefined);
  assert.equal(result.config.generator.pipeline.vae_tiling, null);
  assert.match(result.yaml, /"vae_config.load_encoder": true/);
  assert.throws(() => resolveConfig(data, { "generator.pipeline.experimental.text_encoder_precisions": ["int4"] }));
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function loaderFixture() {
  const requests = [], updates = [];
  const models = ["A", "B"].map((id) => ({ id, label: id, workload_types: ["t2v"], catalog_url: `models/${id}.json` }));
  const load = createCatalogLoader({ models }, "https://example.test/project/assets/cookbook-config/index.json",
    (update) => updates.push(update), (url, options) => {
      const request = { ...deferred(), url, signal: options.signal };
      requests.push(request);
      return request.promise;
    });
  return { load, requests, updates };
}

const modelResponse = (id) => ({ ok: true, json: async () => ({ model: { id } }) });

test("loader fetches only the selected catalog relative to the captured index URL and does not cache", async () => {
  const { load, requests, updates } = loaderFixture();
  assert.equal(requests.length, 0);
  const first = load("B");
  assert.deepEqual(updates.map((update) => update.state), ["loading"]);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "https://example.test/project/assets/cookbook-config/models/B.json");
  requests[0].resolve(modelResponse("B"));
  await first;
  assert.equal(updates.at(-1).catalog.model.id, "B");
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
  assert.equal(updates.at(-1).catalog.model.id, "B");
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
    else body.resolve({ model: { id: "A" } });
    await first;
    assert.deepEqual(updates.map((update) => update.state), ["loading", "loading", "ready"]);
    assert.equal(updates.at(-1).catalog.model.id, "B");
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
    assert.equal(updates.at(-1).catalog.model.id, "B");
  }
});
