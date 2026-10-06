import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import test from "node:test";

const require = createRequire(import.meta.url);
const { resolveConfig, getDefaultConfig, getSchemaField, getEditableFields, toYaml, shellQuote } = require("../../docs/assets/cookbook-config.js");
const object = (properties) => ({ type: "object", properties, required: Object.keys(properties), additionalProperties: false });

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
        }),
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
      }),
    },
    sources: {},
  };
}

const field = (data, path) => getSchemaField(data.schema, path);

test("standard defaults and const values build config without hardcoded frame counts", () => {
  const data = catalog();
  const before = JSON.stringify(data);
  const result = resolveConfig(data);
  assert.equal(result.config.default_request.sampling.num_frames, 121);
  assert.equal(result.config.default_request.sampling.seed, null);
  assert.equal(result.config.default_request.output.return_frames, false);
  assert.equal(result.config.generator.model_path, "Example/Video");
  assert.equal(JSON.stringify(data), before);
  const another = catalog();
  field(another, "default_request.sampling.num_frames").default = 125;
  field(another, "default_request.sampling.guidance_scale").default = 0;
  assert.equal(getDefaultConfig(another.schema).default_request.sampling.guidance_scale, 0);
  assert.equal(resolveConfig(another).config.default_request.sampling.num_frames, 125);
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
  assert.deepEqual(Object.keys(result.clientRequest.body).sort(), ["model", "prompt"]);
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
  assert.equal(resolveConfig(data).config.generator.engine.num_gpus, 2);
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

test("required fields and model/output constants remain schema constraints", () => {
  const missing = catalog();
  delete field(missing, "server.port").default;
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

test("host and port update URLs; host syntax is validated from the schema", () => {
  const result = resolveConfig(catalog(), { "server.host": "0.0.0.0", "server.port": 9000 });
  assert.equal(result.config.server.host, "0.0.0.0");
  assert.equal(result.healthCommand, "curl --fail-with-body http://127.0.0.1:9000/health");
  assert.equal(result.clientRequest.endpoint, "http://127.0.0.1:9000/v1/videos/sync");
  assert.equal(resolveConfig(catalog(), { "server.host": "::1" }).clientRequest.endpoint,
    "http://[::1]:8000/v1/videos/sync");
  assert.equal(resolveConfig(catalog(), { "server.host": "::" }).clientRequest.endpoint,
    "http://127.0.0.1:8000/v1/videos/sync");
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

test("real Wan schema uses one effective default and accepts requested lengths that runtime may align", () => {
  const data = JSON.parse(readFileSync(new URL("../../docs/assets/cookbook-config.json", import.meta.url), "utf8"));
  const result = resolveConfig(data);
  assert.equal(Object.hasOwn(data, "fields"), false);
  assert.equal(Object.hasOwn(data, "base_config"), false);
  assert.equal(result.config.generator.model_path, data.model.id);
  assert.equal(result.config.default_request.sampling.num_frames, 121);
  assert.equal(field(data, "default_request.sampling.num_frames").default, 121);
  assert.equal(field(data, "default_request.sampling.num_frames").minimum, 1);
  assert.equal(field(data, "default_request.sampling.num_frames").multipleOf, undefined);
  assert.equal(result.config.default_request.output.return_frames, false);
  const updated = resolveConfig(data, { "default_request.sampling.num_frames": 82, "generator.engine.offload.vae": false });
  assert.match(updated.yaml, /num_frames: 82/);
  assert.match(updated.yaml, /vae: false/);
  assert.equal(updated.command, "fastvideo serve --config config.yaml");
  assert.throws(() => resolveConfig(data, { "default_request.sampling.num_frames": 0 }));
  assert.deepEqual(Object.keys(updated.clientRequest.body).sort(), ["model", "prompt"]);
});
