/** Authored production recipes and synthetic deployment capabilities share the same public API. */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { parse } from "yaml";
import { ROOT, buildCatalogs, exportCatalogs } from "../js/build-cookbook-config.mjs";
import { getOptions, loadDeployment, loadIndex, resolveConfig } from "../js/cookbook-config.mjs";

function at(value, path) {
  return path.split(".").reduce((current, key) => current?.[key], value);
}

function fixture({ interfaceName = "rest", backend = "cuda", workload = "t2v" } = {}) {
  return {
    id: "example/deployment",
    model: { key: "example", id: "Example/Video", title: "Example" },
    deployment: { id: "deployment", label: "Example deployment" },
    workload,
    env: { BACKEND: "example attention" },
    runtime: {
      id: `example-${backend}-${interfaceName}`,
      backend,
      interface: interfaceName,
      launch_argv:
        backend === "mlx"
          ? ["python", "-m", "example.server", "--config", "config.yaml"]
          : ["fastvideo", "serve", "--config", "config.yaml"],
      server_defaults: { host: "0.0.0.0", port: 8000 },
    },
    base_config: {
      generator: {
        model_path: "Example/Video",
        engine: { num_gpus: 3, parallelism: { tp_size: 1, sp_size: 3 } },
        pipeline: { experimental: { hidden: [1000, 700] } },
        vae_dtype: "fp32",
      },
      server: { host: "0.0.0.0", port: 8040, served_model_name: "example" },
      default_request: { sampling: { num_frames: 21, height: 96, seed: 0 } },
      ...(interfaceName === "websocket" ? { streaming: { session: { timeout: 300 } } } : {}),
    },
    controls: [
      { path: "server.host", schema: { type: "string" } },
      { path: "server.port", schema: { type: "integer", minimum: 1, maximum: 65535 } },
      { path: "default_request.sampling.num_frames", schema: { type: "integer", minimum: 1, maximum: 60, default: 5 } },
      {
        path: "default_request.sampling.height",
        schema: { type: "integer", minimum: 16, maximum: 512, multipleOf: 16 },
      },
      { path: "generator.vae_dtype", schema: { type: "string", enum: ["fp32", "bf16"] } },
    ],
  };
}

test("every authored deployment exports and preserves its native baseline through the public API", async (t) => {
  const outputDir = mkdtempSync(join(tmpdir(), "cookbook-integration-"));
  t.after(() => rmSync(outputDir, { recursive: true, force: true }));
  const expected = buildCatalogs();
  const recipesDir = join(ROOT, "docs/cookbook/recipes");
  const authoredIds = readdirSync(recipesDir)
    .filter((name) => /\.ya?ml$/.test(name))
    .sort()
    .flatMap((name) => {
      const manifest = parse(readFileSync(join(recipesDir, name), "utf8"));
      return Object.keys(manifest.deployments).map((deployment) => `${name.replace(/\.ya?ml$/, "")}/${deployment}`);
    });
  const writtenIndex = exportCatalogs({ outputDir });
  const baseUrl = "https://example.test/assets/cookbook-config/";
  const fetcher = async (url) => ({
    ok: true,
    url,
    json: async () =>
      JSON.parse(readFileSync(join(outputDir, new URL(url).pathname.replace("/assets/cookbook-config/", "")), "utf8")),
  });
  const index = await loadIndex(`${baseUrl}index.json`, { fetcher });
  assert.deepEqual(index.models, writtenIndex.models);
  const deployments = index.models.flatMap((model) => model.deployments);
  assert.deepEqual(
    deployments.map((deployment) => deployment.id),
    authoredIds,
  );
  assert.equal(new Set(authoredIds).size, authoredIds.length);
  assert.deepEqual(
    expected.map((catalog) => catalog.id),
    authoredIds,
  );
  for (const [position, deployment] of deployments.entries()) {
    await t.test(deployment.id, async () => {
      const catalog = await loadDeployment(index, deployment.id, { fetcher });
      assert.deepEqual(catalog, expected[position]);
      const nativeBaseline = parse(readFileSync(join(ROOT, catalog.source_config), "utf8"));
      assert.deepEqual(catalog.base_config, nativeBaseline);
      const before = structuredClone(catalog);
      const initial = resolveConfig(catalog);
      assert.deepEqual(initial.config, nativeBaseline);
      assert.deepEqual(parse(initial.yaml, { version: "1.1" }), nativeBaseline);
      const options = getOptions(catalog);
      assert.deepEqual(
        options.map((option) => option.path),
        catalog.controls.map((control) => control.path),
      );
      for (const option of options) {
        const value = at(nativeBaseline, option.path);
        assert.deepEqual(option.value, value);
        if (value !== undefined) {
          const result = resolveConfig(catalog, { [option.path]: structuredClone(value) });
          assert.deepEqual(result.config, nativeBaseline, option.path);
          assert.deepEqual(parse(result.yaml, { version: "1.1" }), nativeBaseline, option.path);
        }
      }
      assert.deepEqual(resolveConfig(catalog), initial);
      assert.deepEqual(catalog, before);
    });
  }
});

test("a deployment with hidden parallelism and its own bounds needs no integration special case", () => {
  const catalog = fixture();
  const result = resolveConfig(catalog, {
    "default_request.sampling.num_frames": 60,
    "default_request.sampling.height": 128,
    "generator.vae_dtype": "bf16",
  });
  assert.equal(result.config.default_request.sampling.num_frames, 60);
  assert.deepEqual(result.config.generator.engine, catalog.base_config.generator.engine);
  assert.deepEqual(result.config.generator.pipeline, catalog.base_config.generator.pipeline);
  for (const frames of [0, 61, 2.5]) {
    assert.throws(() => resolveConfig(catalog, { "default_request.sampling.num_frames": frames }), /num_frames/);
  }
  for (const height of [0, 17, 528]) {
    assert.throws(() => resolveConfig(catalog, { "default_request.sampling.height": height }), /height/);
  }
  assert.throws(() => resolveConfig(catalog, { "generator.vae_dtype": "int8" }), /vae_dtype/);
  assert.deepEqual(resolveConfig(catalog).config, catalog.base_config);
});

test("opaque request defaults survive a deployment with a different launcher", () => {
  const catalog = fixture({ backend: "mlx" });
  catalog.controls = catalog.controls.filter((control) => !control.path.startsWith("default_request."));
  const result = resolveConfig(catalog, { "server.port": 9002, "generator.vae_dtype": "bf16" });
  assert.match(result.command, /python -m example.server --config config.yaml/);
  assert.deepEqual(result.config.default_request, catalog.base_config.default_request);
  assert.match(result.clientCommand, /127\.0\.0\.1:9002/);
  assert.throws(() => resolveConfig(catalog, { "default_request.sampling.num_frames": 40 }), /not an editable/);
});

test("I2V client examples include a source image while preserving hidden settings", () => {
  const catalog = fixture({ workload: "i2v" });
  const result = resolveConfig(catalog);
  assert.equal(result.clientRequest.input_reference, "/absolute/path/to/first-frame.png");
  assert.match(result.clientCommand, /\/v1\/videos\/sync/);
  assert.equal(result.websocketUrl, undefined);
  assert.deepEqual(result.config, catalog.base_config);
});

test("WebSocket examples follow edited client addresses without REST generation requests", () => {
  const catalog = fixture({ interfaceName: "websocket" });
  for (const [host, clientHost] of [
    ["0.0.0.0", "127.0.0.1"],
    ["::", "[::1]"],
    ["[::]", "[::1]"],
    ["2001:db8::5", "[2001:db8::5]"],
    ["video.example.test", "video.example.test"],
  ]) {
    const result = resolveConfig(catalog, { "server.host": host, "server.port": 9012 });
    assert.ok(result.clientCommand.includes(`http://${clientHost}:9012/health`));
    assert.match(result.clientCommand, /^curl --fail-with-body /);
    assert.equal(result.websocketUrl, `ws://${clientHost}:9012/v1/stream`);
    assert.equal(result.clientRequest, null);
    assert.doesNotMatch(result.clientCommand, /--data|--output|\/v1\/videos/);
    assert.deepEqual(result.config.streaming, catalog.base_config.streaming);
  }
});
