import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { ROOT, buildCatalogs, deepMerge, exportCatalogs, mergeOptions, previewCatalog } from "../../docs/build-cookbook-config.mjs";

const { parse } = createRequire(new URL("../../docs/package.json", import.meta.url))("yaml");
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
  const deployment = { runtime: "cuda-rest", workload: "t2v", defaults: "examples/serving/example.yaml",
    options: ["server.*", "generator.engine.compile.enabled", "default_request.sampling.num_frames"] };
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

test("common -> runtime -> deployment metadata composes, while selected options remain independent", (t) => {
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
  f.deployment.options.push("generator.pipeline.experimental.flow_shift", "default_request.sampling.seed");
  f.save();
  const extended = buildCatalogs(f.options)[0];
  assert.equal(extended.controls.at(-2).schema.default, 3);
  assert.equal(extended.controls.at(-1).schema.default, null);
  assert.equal(extended.base_config.generator.pipeline.experimental.flow_shift, undefined);
  assert.equal(extended.base_config.default_request.sampling.seed, 0);
});

test("wildcard expansion follows authored option order and preserves the entire baseline", (t) => {
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
  f.deployment.options.push("generator.pipeline.experimental.extra"); f.save();
  const data = buildCatalogs(f.options)[0];
  assert.deepEqual(data.controls.at(-1).schema.default, { $ref: "literal", minimum: "text" });
});

test("selection patterns deduplicate, exclude and reinclude options in order", (t) => {
  const f = fixture(t);
  f.deployment.options = ["server.*", "server.port", "!server.port", "default_request.sampling.*", "server.port"];
  f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), [
    "server.host", "server.served_model_name", "default_request.sampling.num_frames",
    "default_request.sampling.seed", "server.port",
  ]);
});

test("star spans path segments, matches zero characters and keeps dots literal", (t) => {
  const f = fixture(t);
  f.deployment.overrides = {
    "server.port_extra": { type: "integer", default: 42 },
    "generator.pipeline.port": { type: "integer", default: 43 },
    "generator.pipeline.serverport": { type: "integer", default: 44 },
  };
  f.deployment.options = ["*.port"];
  f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), [
    "server.port", "generator.pipeline.port",
  ]);
  f.deployment.options = ["server.port*"];
  f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), [
    "server.port", "server.port_extra",
  ]);
  f.deployment.options = ["generator.*enabled", "*sampling.*"];
  f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), [
    "generator.engine.compile.enabled", "default_request.sampling.num_frames", "default_request.sampling.seed",
  ]);
});

test("bare namespaces are not expanded and wildcard patterns match whole paths", (t) => {
  const f = fixture(t);
  for (const pattern of ["server", "port", "sampling.*", "*.por", "serve.*"]) {
    f.deployment.options = [pattern]; f.save();
    assert.throws(() => buildCatalogs(f.options), /Unknown option pattern|Invalid option pattern|Invalid configuration path/);
  }
});

for (const selectors of [[], ["!server.port"], ["server.*", "!server.*"]]) {
  test(`empty selection preserves the baseline: ${JSON.stringify(selectors)}`, (t) => {
    const f = fixture(t); f.deployment.options = selectors; f.save();
    const data = buildCatalogs(f.options)[0];
    assert.deepEqual(data.controls, []);
    assert.deepEqual(data.base_config, f.baseline);
  });
}

test("a root wildcard selects all declared options in their authored order", (t) => {
  const f = fixture(t); f.deployment.options = ["*"]; f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), [
    "server.host", "server.port", "server.served_model_name", "generator.engine.compile.enabled",
    "default_request.sampling.num_frames", "default_request.sampling.seed",
  ]);
});

test("local option overrides participate in wildcard selection and exclusion", (t) => {
  const f = fixture(t);
  f.deployment.overrides = {
    "server.port": { maximum: 9000 },
    "generator.pipeline.experimental.flow_shift": { type: "number", minimum: 0, default: 3 },
  };
  f.deployment.options = ["server.*", "generator.pipeline.experimental.*", "!server.port"];
  f.baseline.server.port = 70000;
  f.baseline.generator.pipeline.experimental.flow_shift = 4;
  f.save();
  const data = buildCatalogs(f.options)[0];
  assert.deepEqual(data.controls.map((item) => item.path), [
    "server.host", "server.served_model_name", "generator.pipeline.experimental.flow_shift",
  ]);
  assert.equal(data.controls.at(-1).schema.default, 3);
  assert.deepEqual(data.base_config, f.baseline);
});

test("parent and child selected options overlap unless one is excluded", (t) => {
  const f = fixture(t);
  f.deployment.overrides = { "generator.engine.compile": { type: "object", additionalProperties: true } };
  f.deployment.options = ["generator.engine.compile", "generator.engine.compile.*"];
  f.save();
  assert.throws(() => buildCatalogs(f.options), /Overlapping options/);
  f.deployment.options.push("!generator.engine.compile"); f.save();
  assert.deepEqual(buildCatalogs(f.options)[0].controls.map((item) => item.path), ["generator.engine.compile.enabled"]);
});

test("the old controls manifest key is rejected instead of silently selecting defaults", (t) => {
  const f = fixture(t);
  f.deployment.controls = f.deployment.options;
  delete f.deployment.options;
  f.save();
  assert.throws(() => buildCatalogs(f.options), /expected.*options/);
});

for (const selectors of [
  ["server.unknown"], ["server.*", "!server.unknown"], ["!unknown.*"],
  ["server.**"], ["server.?ort"], ["server.[hp]*"], ["server/port"], [" server.*"], ["server.* "],
  ["!"], ["!!server.port"], ["generator.__proto__.*"], ["*.constructor.*"], ["server.prototype*"],
  ["generator..*"], [false],
]) {
  test(`invalid selectors fail: ${JSON.stringify(selectors)}`, (t) => {
    const f = fixture(t); f.deployment.options = selectors; f.save();
    assert.throws(() => buildCatalogs(f.options), /Unknown option pattern|Invalid option pattern|Invalid configuration path/);
  });
}

test("wildcards may include protected fields if later exclusions remove them", (t) => {
  const f = fixture(t);
  f.deployment.overrides = {
    "generator.model_path": { type: "string" },
    "default_request.output.return_frames": { type: "boolean" },
  };
  f.deployment.options = ["*", "!generator.model_path", "!default_request.output.*"];
  f.save();
  assert.equal(buildCatalogs(f.options)[0].controls.length, 6);
  f.deployment.options.push("generator.model_path"); f.save();
  assert.throws(() => buildCatalogs(f.options), /Protected configuration option/);
});

for (const path of ["generator.model_path", "generator.pipeline.workload_type", "default_request.output",
  "default_request.output.return_frames"]) {
  test(`custom schemas cannot enable protected fields: ${path}`, (t) => {
    const f = fixture(t); f.deployment.overrides = { [path]: { type: "string" } };
    f.deployment.options = [path]; f.save();
    assert.throws(() => buildCatalogs(f.options), /Protected configuration option/);
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
  f.declarations.options["server.output_dir"] = { type: "string", default: "outputs" };
  f.baseline.server.output_dir = "streaming-output";
  f.deployment.options = ["server.*", "!server.served_model_name", "!server.output_dir"]; f.save();
  const data = buildCatalogs(f.options)[0];
  assert.deepEqual(data.controls.map((item) => item.path), ["server.host", "server.port"]);
  assert.deepEqual(data.base_config, f.baseline);
  f.deployment.options.push("server.served_model_name"); f.save();
  assert.throws(() => buildCatalogs(f.options), /Protected configuration option/);
  f.deployment.options.pop();
  f.deployment.options.push("server.output_dir"); f.save();
  assert.throws(() => buildCatalogs(f.options), /Protected configuration option/);
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
  f.deployment.defaults = "docs/cookbook/options.yaml"; f.save();
  assert.throws(() => buildCatalogs(f.options), /under examples\/serving/);
  f.deployment.defaults = "examples/serving/example.yaml";
  f.manifest.deployments.other = { ...clone(f.deployment), defaults: "examples/serving/other.yaml" };
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
  f.deployment.options = ["server.port"]; write(f.manifestPath, f.manifest);
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
  f.deployment.options = ["unknown.path"]; f.save();
  assert.throws(() => exportCatalogs({ ...f.options, outputDir }));
  assert.equal(statSync(catalogFile).mtimeMs, modified);
  unlinkSync(join(outputDir, "recipes/external"));
});

test("preview returns the selected full catalog with merged schemas and exclusions", (t) => {
  const f = fixture(t);
  f.deployment.overrides = {
    "server.port": { maximum: 9000, description: "Recipe-specific port" },
    "generator.pipeline.experimental.flow_shift": { type: "number", default: 3 },
  };
  f.deployment.options = ["server.*", "!server.host", "generator.pipeline.experimental.*"];
  f.deployment.env = { BACKEND: "special attention" };
  f.deployment.requirements = ["Prepare the checkpoint"];
  f.save();
  const expected = buildCatalogs(f.options)[0], data = parse(previewCatalog("example/rest", f.options));
  assert.deepEqual(data, expected);
  assert.deepEqual(data.controls.map((item) => item.path), [
    "server.port", "server.served_model_name", "generator.pipeline.experimental.flow_shift",
  ]);
  assert.equal(data.controls[0].schema.maximum, 9000);
  assert.deepEqual(data.base_config, f.baseline);
  assert.deepEqual(data.env, { BACKEND: "special attention" });
});

test("preview selects an exact deployment and lists available IDs for unknown or bare model IDs", (t) => {
  const f = fixture(t);
  f.manifest.deployments.other = { ...clone(f.deployment), label: "Another deployment", options: ["server.port"] };
  f.save();
  const expected = buildCatalogs(f.options);
  for (const catalog of expected) assert.deepEqual(parse(previewCatalog(catalog.id, f.options)), catalog);
  for (const id of ["example", "missing/rest"]) {
    assert.throws(() => previewCatalog(id, f.options), (error) => {
      assert.ok(error.message.includes(id));
      assert.match(error.message, /example\/rest/);
      assert.match(error.message, /example\/other/);
      return true;
    });
  }
});

test("preview rebuilds current authored sources without reading or writing generated JSON", (t) => {
  const f = fixture(t), outputDir = join(f.root, "docs/assets/cookbook-config");
  exportCatalogs({ ...f.options, outputDir });
  const stalePath = join(outputDir, "recipes/example/rest.json"), staleJson = readFileSync(stalePath, "utf8");
  const staleTime = statSync(stalePath).mtimeMs, indexText = readFileSync(join(outputDir, "index.json"), "utf8");
  f.baseline.server.port = 8080;
  f.deployment.options = ["server.*", "!server.host"];
  f.save();
  assert.deepEqual(parse(previewCatalog("example/rest", f.options)), buildCatalogs(f.options)[0]);
  assert.equal(parse(previewCatalog("example/rest", f.options)).base_config.server.port, 8080);
  assert.equal(readFileSync(stalePath, "utf8"), staleJson);
  assert.equal(statSync(stalePath).mtimeMs, staleTime);
  assert.equal(readFileSync(join(outputDir, "index.json"), "utf8"), indexText);
  rmSync(outputDir, { recursive: true });
  f.baseline.server.port = 8081; f.save();
  assert.equal(parse(previewCatalog("example/rest", f.options)).base_config.server.port, 8081);
  assert.equal(existsSync(outputDir), false);
  f.baseline.server.port = 70000; f.save();
  assert.throws(() => previewCatalog("example/rest", f.options), /server.port/);
  assert.equal(existsSync(outputDir), false);
});

test("preview CLI prints only YAML and reports errors without exporting files", (t) => {
  const f = fixture(t), outputDir = join(f.root, "cli-output");
  f.deployment.defaults = "examples/serving/openai_fastwan21_1_3b.yaml";
  f.deployment.options = ["server.*", "!server.host"];
  f.deployment.overrides = { "server.port": { maximum: 9000 } };
  f.save();
  const script = join(ROOT, "docs/build-cookbook-config.mjs");
  const sourceFlags = ["--recipes-dir", f.recipesDir, "--catalog", f.catalogPath];
  const run = (...flags) => spawnSync(process.execPath, [script, ...flags, ...sourceFlags], { cwd: f.root, encoding: "utf8" });
  const generatedPaths = [join(ROOT, "docs/assets/cookbook-config/index.json"),
    join(ROOT, "docs/assets/cookbook-config/recipes/example/rest.json")];
  const generatedState = () => generatedPaths.map((path) => existsSync(path) ?
    { contents: readFileSync(path, "utf8"), modified: statSync(path).mtimeMs } : null);
  const before = generatedState();
  const success = run("--preview", "example/rest");
  assert.equal(success.status, 0, success.error?.message || success.stderr || success.stdout);
  assert.equal(success.stderr, "");
  assert.deepEqual(parse(success.stdout), buildCatalogs({ ...f.options, root: ROOT })[0]);
  assert.equal(existsSync(outputDir), false);
  const unknown = run("--preview", "example");
  assert.notEqual(unknown.status, 0);
  assert.equal(unknown.stdout, "");
  assert.match(unknown.stderr, /example\/rest/);
  const conflict = run("--preview", "example/rest", "--output-dir", outputDir);
  assert.notEqual(conflict.status, 0);
  assert.equal(conflict.stdout, "");
  assert.match(conflict.stderr, /--preview/);
  assert.match(conflict.stderr, /--output-dir/);
  assert.equal(existsSync(outputDir), false);
  assert.deepEqual(generatedState(), before);
});

test("mergeOptions refuses unsafe names and does not mutate its inputs", () => {
  const base = { "server.port": { type: "integer", title: "Port", minimum: 1 } };
  const patch = { "server.port": { maximum: 9000 } }, before = clone(base);
  const result = mergeOptions(base, patch);
  result["server.port"].title = "Changed";
  assert.deepEqual(base, before); assert.deepEqual(patch, { "server.port": { maximum: 9000 } });
  assert.throws(() => mergeOptions(base, { "server.port": { type: "string" } }), /declared type/);
});
