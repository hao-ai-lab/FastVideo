import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { buildCatalogs, deepMerge, exportCatalogs, mergeOptions } from "../../docs/build-cookbook-config.mjs";

const clone = (value) => structuredClone(value);
function write(path, data) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(data, null, 2)}\n`);
}
function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), "authored-cookbook-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const catalogPath = join(root, "docs/cookbook/options.yaml"), recipesDir = join(root, "docs/cookbook/recipes");
  const configPath = join(root, "examples/serving/example.yaml"), manifestPath = join(recipesDir, "example.yaml");
  const declarations = {
    options: {
      "server.host": { type: "string", default: "0.0.0.0", title: "Host" },
      "server.port": { type: "integer", minimum: 1, maximum: 65535, default: 8000, title: "Port" },
      "server.served_model_name": { title: "Served model" },
    },
    runtimes: {
      "cuda-rest": {
        metadata: { id: "cuda-rest", label: "CUDA REST", backend: "cuda", interface: "rest",
          launch_argv: ["fastvideo", "serve", "--config", "config.yaml"],
          server_defaults: { host: "0.0.0.0", port: 8000, served_model_name: null } },
        workloads: ["t2v", "i2v"],
        options: {
          "server.served_model_name": { anyOf: [{ type: "string" }, { type: "null" }], default: null },
          "generator.engine.compile.enabled": { type: "boolean", default: false },
          "default_request.sampling.num_frames": { type: "integer", minimum: 1, default: 81 },
          "default_request.sampling.seed": { anyOf: [{ type: "integer" }, { type: "null" }], default: 0 },
        },
      },
    },
  };
  const baseline = {
    generator: { model_path: "Example/Video", engine: { compile: { enabled: true } },
      pipeline: { workload_type: "t2v", experimental: { hidden: [1000, 757, 522] } } },
    server: { host: "0.0.0.0", port: 8000, served_model_name: "video" },
    default_request: { sampling: { num_frames: 81, seed: 0 }, output: { return_frames: false } },
  };
  const deployment = { runtime: "cuda-rest", workload: "t2v", config: "examples/serving/example.yaml",
    controls: ["server", "generator.engine.compile.enabled", "default_request.sampling.num_frames"] };
  const manifest = { title: "Example", deployments: { rest: deployment } };
  const save = () => { write(catalogPath, declarations); write(configPath, baseline); write(manifestPath, manifest); };
  save();
  return { root, catalogPath, recipesDir, configPath, manifestPath, declarations, baseline, deployment, manifest, save,
    options: { root, catalogPath, recipesDir } };
}

test("deep merge preserves sibling metadata, replaces arrays and exact false/zero/null, without mutation", () => {
  const source = { option: { properties: { a: { type: "number" }, b: { type: "string" } }, enum: [1, 2],
    default: true, minimum: 4, description: "keep" }, keep: ["unchanged"] };
  const patch = { option: { properties: { a: { minimum: 0 } }, enum: [0], default: false, minimum: 0,
    examples: null } };
  const before = clone(source), patchBefore = clone(patch);
  const merged = deepMerge(source, patch);
  assert.deepEqual(merged.option.properties, { a: { type: "number", minimum: 0 }, b: { type: "string" } });
  assert.deepEqual(merged.option.enum, [0]);
  assert.equal(merged.option.default, false);
  assert.equal(merged.option.minimum, 0);
  assert.equal(merged.option.examples, null);
  assert.equal(merged.option.description, "keep");
  merged.keep.push("local"); merged.option.enum.push(3);
  assert.deepEqual(source, before); assert.deepEqual(patch, patchBefore);
});

test("common -> runtime -> deployment metadata composes, while controls remain independent", (t) => {
  const f = fixture(t);
  f.declarations.runtimes["cuda-rest"].options["server.port"] = { maximum: 9000, description: "Runtime ports" };
  f.deployment.overrides = {
    "server.port": { default: 0, minimum: 0, description: "Recipe port" },
    "default_request.sampling.seed": { default: null },
    "generator.engine.compile.enabled": { default: false },
    "generator.pipeline.experimental.flow_shift": { type: "number", minimum: 0, default: 3 },
  };
  f.save();
  const data = buildCatalogs(f.options)[0];
  const field = data.controls.find((item) => item.path === "server.port").schema;
  assert.deepEqual(field, { type: "integer", minimum: 0, maximum: 9000, default: 0, title: "Port", description: "Recipe port" });
  assert.equal(data.controls.some((item) => item.path.endsWith("flow_shift")), false);
  assert.deepEqual(data.base_config, f.baseline);
  f.deployment.controls.push("generator.pipeline.experimental.flow_shift", "default_request.sampling.seed");
  f.save();
  const extended = buildCatalogs(f.options)[0];
  assert.equal(extended.controls.at(-2).schema.default, 3);
  assert.equal(extended.controls.at(-1).schema.default, null);
  assert.equal(extended.base_config.generator.pipeline.experimental.flow_shift, undefined);
  assert.equal(extended.base_config.default_request.sampling.seed, 0);
});

test("namespace expansion follows authored option order and preserves the entire baseline", (t) => {
  const f = fixture(t), data = buildCatalogs(f.options)[0];
  assert.deepEqual(data.controls.map((item) => item.path), ["server.host", "server.port", "server.served_model_name",
    "generator.engine.compile.enabled", "default_request.sampling.num_frames"]);
  assert.deepEqual(data.base_config, f.baseline);
  assert.deepEqual(data.model, { id: "Example/Video", key: "example", title: "Example" });
  assert.equal(data.guide, null);
  assert.equal(data.runtime.id, "cuda-rest");
});

for (const [override, message] of [
  [{ type: "string" }, /cannot change.*type/],
  [{ minimum: 9001, maximum: 9000 }, /minimum exceeds maximum/],
  [{ minimum: 8000, exclusiveMaximum: 8000 }, /minimum exceeds maximum/],
  [{ default: "8000" }, /default does not satisfy/],
  [{ enum: [8000, "9000"] }, /enum value does not satisfy/],
  [{ default: 8000, enum: [8080] }, /default does not satisfy/],
  [{ nonsense: true }, /unknown keyword/],
]) {
  test(`invalid merged option is rejected: ${JSON.stringify(override)}`, (t) => {
    const f = fixture(t); f.deployment.overrides = { "server.port": override }; f.save();
    assert.throws(() => buildCatalogs(f.options), message);
  });
}

for (const [path, schema, message] of [
  ["generator.pipeline.experimental.extra", { default: 3 }, /complete schema needs an explicit type/],
  ["generator.pipeline.experimental.extra", { type: "integer", $ref: "#/$defs/N" }, /\$ref is unsupported/],
  ["generator.__proto__.polluted", { type: "boolean" }, /Invalid configuration path/],
  ["generator.pipeline.experimental.extra", { type: "string", minLength: 5, maxLength: 2 }, /minLength exceeds maxLength/],
  ["generator.pipeline.experimental.extra", { type: "array", items: { type: "number", default: "bad" } }, /default does not satisfy/],
]) {
  test(`new option needs a safe complete schema: ${path} ${JSON.stringify(schema)}`, (t) => {
    const f = fixture(t); f.deployment.overrides = { [path]: schema }; f.save();
    assert.throws(() => buildCatalogs(f.options), message);
  });
}

test("literal schema annotations do not become reference instructions", (t) => {
  const f = fixture(t);
  f.deployment.overrides = { "generator.pipeline.experimental.extra": {
    type: "object", additionalProperties: true, default: { $ref: "literal", minimum: "text" },
    examples: [{ $ref: "also literal" }],
  } };
  f.deployment.controls.push("generator.pipeline.experimental.extra"); f.save();
  const data = buildCatalogs(f.options)[0];
  assert.deepEqual(data.controls.at(-1).schema.default, { $ref: "literal", minimum: "text" });
});

for (const selectors of [["server", "server.port"], ["server.unknown"], ["generator.__proto__.polluted"]]) {
  test(`invalid selectors fail: ${JSON.stringify(selectors)}`, (t) => {
    const f = fixture(t); f.deployment.controls = selectors; f.save();
    assert.throws(() => buildCatalogs(f.options), /Overlapping|Unknown control|Invalid configuration path/);
  });
}

for (const path of ["generator.model_path", "generator.pipeline.workload_type", "default_request.output",
  "default_request.output.return_frames"]) {
  test(`custom schemas cannot enable protected fields: ${path}`, (t) => {
    const f = fixture(t); f.deployment.overrides = { [path]: { type: "string" } };
    f.deployment.controls = [path]; f.save();
    assert.throws(() => buildCatalogs(f.options), /Protected configuration control/);
  });
}

test("selected baseline values validate, but hidden values and omitted defaults are not rewritten", (t) => {
  const f = fixture(t);
  delete f.baseline.server.host;
  f.baseline.generator.pipeline.experimental.opaque = { a: false, b: null, c: 0 };
  f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].base_config, f.baseline);
  f.baseline.server.port = 70000; f.save();
  assert.throws(() => buildCatalogs(f.options), /server.port/);
});

test("runtime options may complete common annotations without changing a declared type", (t) => {
  const f = fixture(t);
  const mlx = clone(f.declarations.runtimes["cuda-rest"]);
  mlx.metadata = { ...mlx.metadata, id: "mlx-rest", label: "MLX REST", backend: "mlx" };
  mlx.options["server.served_model_name"] = { type: "string", minLength: 1, default: "mlx" };
  f.declarations.runtimes["mlx-rest"] = mlx;
  f.manifest.deployments.mlx = { ...clone(f.deployment), runtime: "mlx-rest", label: "Apple Silicon" };
  f.manifest.summary = "Shared summary";
  f.deployment.summary = "CUDA-specific";
  f.deployment.env = { BACKEND: "special attention" };
  f.deployment.overrides = { "server.port": { maximum: 9000 } };
  f.save();
  const [cuda, apple] = buildCatalogs(f.options);
  assert.equal(cuda.controls.find((item) => item.path === "server.port").schema.maximum, 9000);
  assert.equal(apple.controls.find((item) => item.path === "server.port").schema.maximum, 65535);
  assert.equal(apple.controls.find((item) => item.path === "server.served_model_name").schema.type, "string");
  assert.equal(apple.summary, "Shared summary"); assert.deepEqual(apple.env, {});
  assert.equal(cuda.summary, "CUDA-specific"); assert.deepEqual(cuda.env, { BACKEND: "special attention" });
});

test("an explicit streaming block must match the authored runtime interface", (t) => {
  const f = fixture(t);
  f.baseline.streaming = {}; f.save();
  assert.throws(() => buildCatalogs(f.options), /streaming block disagrees/);
  f.declarations.runtimes["cuda-rest"].metadata.interface = "websocket";
  f.deployment.controls = ["server.host", "server.port"]; f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].base_config.streaming, {});
  f.deployment.controls.push("server.served_model_name"); f.save();
  assert.throws(() => buildCatalogs(f.options), /Protected configuration control/);
  delete f.baseline.streaming; f.save();
  assert.throws(() => buildCatalogs(f.options), /streaming block disagrees/);
});

test("manifest identity, duplicate keys, environment and file boundaries are validated", (t) => {
  const f = fixture(t);
  f.deployment.env = { "BAD-NAME": "x" }; f.save();
  assert.throws(() => buildCatalogs(f.options), /environment mapping/);
  delete f.deployment.env; f.save();
  writeFileSync(f.manifestPath, "title: One\ntitle: Two\ndeployments: {}\n");
  assert.throws(() => buildCatalogs(f.options), /unique|duplicate/i);
  f.save();
  f.deployment.config = "docs/cookbook/options.yaml"; f.save();
  assert.throws(() => buildCatalogs(f.options), /under examples\/serving/);
  f.deployment.config = "examples/serving/example.yaml";
  f.manifest.deployments.other = { ...clone(f.deployment), config: "examples/serving/other.yaml" };
  write(join(f.root, "examples/serving/other.yaml"), { ...f.baseline, generator: { model_path: "Other/Video" } }); f.save();
  assert.throws(() => buildCatalogs(f.options), /same model_path/);
});

test("unsafe hidden integers and cyclic YAML are rejected before JSON serialization", (t) => {
  const f = fixture(t);
  writeFileSync(f.configPath, "generator:\n  model_path: Example/Video\nhidden: 9007199254740993\n");
  assert.throws(() => buildCatalogs(f.options), /browser-safe range/);
  writeFileSync(f.configPath, "generator:\n  model_path: Example/Video\nhidden: &loop [*loop]\n");
  assert.throws(() => buildCatalogs(f.options), /cyclic YAML|alias/i);
});

test("ordinary YAML anchors reuse authored runtime options", (t) => {
  const f = fixture(t);
  writeFileSync(f.catalogPath, `options: &fields\n  server.port: {type: integer, minimum: 1, default: 8000}\nruntimes:\n  cuda-rest:\n    metadata: ${JSON.stringify(f.declarations.runtimes["cuda-rest"].metadata)}\n    workloads: [t2v]\n    options: *fields\n`);
  f.deployment.controls = ["server.port"]; write(f.manifestPath, f.manifest);
  assert.equal(buildCatalogs(f.options)[0].controls[0].schema.minimum, 1);
});

test("guide output references MkDocs pages and rejects paths outside docs", (t) => {
  const f = fixture(t), path = join(f.root, "docs/guides/My model.md");
  mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, "# Guide\n");
  f.manifest.guide = "docs/guides/My model.md"; f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].guide, { url: "../../guides/My%20model/" });
  f.manifest.guide = "examples/serving/example.yaml"; f.save();
  assert.throws(() => buildCatalogs(f.options), /under docs/);
});

test("export writes matching index/catalogs, preserves mtimes and prunes only current generated files", (t) => {
  const f = fixture(t), outputDir = join(f.root, "output");
  const index = exportCatalogs({ ...f.options, outputDir });
  assert.deepEqual(index.models.map((model) => model.id), ["example"]);
  const url = index.models[0].deployments[0].catalog_url, catalogFile = join(outputDir, url);
  assert.deepEqual(JSON.parse(readFileSync(catalogFile)), buildCatalogs(f.options)[0]);
  const modified = statSync(catalogFile).mtimeMs;
  const stale = join(outputDir, "recipes/example/removed.json"), marker = join(outputDir, "recipes/example/notes.txt");
  write(stale, {}); writeFileSync(marker, "keep");
  const external = join(f.root, "external"); write(join(external, "rest.json"), {});
  symlinkSync(external, join(outputDir, "recipes/external"));
  exportCatalogs({ ...f.options, outputDir });
  assert.equal(statSync(catalogFile).mtimeMs, modified);
  assert.equal(existsSync(stale), false); assert.ok(existsSync(marker));
  assert.ok(existsSync(join(external, "rest.json")));
  f.deployment.controls = ["unknown.path"]; f.save();
  assert.throws(() => exportCatalogs({ ...f.options, outputDir }));
  assert.equal(statSync(catalogFile).mtimeMs, modified);
  unlinkSync(join(outputDir, "recipes/external"));
});

test("mergeOptions refuses unsafe names and does not mutate its inputs", () => {
  const base = { "server.port": { type: "integer", title: "Port", minimum: 1 } };
  const patch = { "server.port": { maximum: 9000 } }, before = clone(base);
  const result = mergeOptions(base, patch);
  result["server.port"].title = "Changed";
  assert.deepEqual(base, before); assert.deepEqual(patch, { "server.port": { maximum: 9000 } });
  assert.throws(() => mergeOptions(base, { "server.port": { type: "string" } }), /declared type/);
});
