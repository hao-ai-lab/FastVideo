/** Integration contract: authored YAML catalog -> reusable browser resolver. Node only. */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const require = createRequire(import.meta.url);
const { resolveConfig } = require("../../docs/assets/cookbook-config.js");

function exportedCatalogs() {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const output = mkdtempSync(join(tmpdir(), "fastvideo-cookbook-recipes-"));
  try {
    const result = spawnSync(process.execPath, ["docs/build-cookbook-config.mjs", "--output-dir", output], { cwd: root, encoding: "utf8" });
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
    "server.host", "server.port", "generator.engine.num_gpus",
    "generator.engine.parallelism.tp_size", "generator.engine.parallelism.sp_size",
    "generator.engine.parallelism.hsdp_replicate_dim", "generator.engine.parallelism.hsdp_shard_dim",
    "generator.engine.parallelism.dist_timeout", "default_request.sampling.num_frames",
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
    "generator.engine.num_gpus", "generator.engine.parallelism.tp_size", "generator.engine.parallelism.sp_size",
    "generator.engine.parallelism.hsdp_replicate_dim", "generator.engine.parallelism.hsdp_shard_dim",
    "generator.engine.parallelism.dist_timeout",
    "generator.engine.offload.dit_layerwise", "generator.engine.offload.text_encoder", "generator.engine.offload.vae",
    "generator.engine.compile.enabled", "default_request.sampling.num_frames", "default_request.sampling.height",
    "default_request.sampling.width", "default_request.sampling.fps", "default_request.sampling.seed",
  ]);
  assert.match(result.command, /FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN/);

});

test("GPU and pattern-selected parallelism controls validate complete edits", () => {
  const data = catalogs.get("fasth3-8step/cuda-rest");
  const paths = new Set(data.controls.map((control) => control.path));
  for (const path of ["generator.engine.num_gpus", "generator.engine.parallelism.tp_size",
    "generator.engine.parallelism.sp_size", "generator.engine.parallelism.hsdp_replicate_dim",
    "generator.engine.parallelism.hsdp_shard_dim", "generator.engine.parallelism.dist_timeout"]) {
    assert.ok(paths.has(path), path);
  }
  assert.throws(() => resolveConfig(data, { "generator.engine.num_gpus": 2 }), /sp_size.*num_gpus/);
  const result = resolveConfig(data, {
    "generator.engine.num_gpus": 2, "generator.engine.parallelism.sp_size": 2,
  });
  assert.equal(result.config.generator.engine.num_gpus, 2);
  assert.equal(result.config.generator.engine.parallelism.sp_size, 2);
  assert.equal(result.config.generator.engine.parallelism.tp_size, 1);
  assert.equal(result.config.generator.engine.use_fsdp_inference, false);
  assert.deepEqual(result.config.generator.pipeline, data.base_config.generator.pipeline);
  assert.deepEqual(resolveConfig(data).config, data.base_config);
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
