import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { parse, stringify } from "yaml";

import { ROOT, buildCatalogs } from "../js/build-cookbook-config.mjs";
import { buildCapacityCatalog, exportCapacityCatalog } from "../js/build-cookbook-capacity.mjs";
import { loadCapacityHardware, loadCapacityIndex, loadCapacityModel } from "../js/cookbook-capacity.mjs";

const SOURCE = "https://example.test/specification";
const SCRIPT = fileURLToPath(new URL("../js/build-cookbook-capacity.mjs", import.meta.url));
const clone = (value) => structuredClone(value);

function write(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, stringify(value));
}

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), "cookbook-capacity-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const catalogDir = join(root, "docs/cookbook/capacity");
  const outputDir = join(root, "docs/assets/cookbook-config/capacity");
  const paths = {
    hardware: join(catalogDir, "hardware.yaml"),
    transformers: join(catalogDir, "components/transformers.yaml"),
    encoders: join(catalogDir, "components/encoders.yaml"),
    vaes: join(catalogDir, "components/vaes.yaml"),
    model: join(root, "docs/cookbook/recipes/example-model.yaml"),
    options: join(root, "docs/cookbook/options.yaml"),
    config: join(root, "examples/serving/example.yaml"),
  };
  const hardware = {
    hardware: {
      "example-gpu": {
        name: "Fixture GPU",
        memory_type: "dedicated",
        memory_gb: 24,
        backend: "cuda",
        sources: [SOURCE],
      },
    },
  };
  const transformers = {
    transformers: {
      "example-transformer": {
        name: "Fixture transformer",
        parameter_count: "1_418_996_800",
        block_count: 30,
        block_parameter_count: "1_393_221_120",
        geometry: { hidden_size: 1536, ffn_size: 8960, patch_size: [1, 2, 2], gated_ffn: false },
        sources: [SOURCE],
      },
    },
  };
  const encoders = {
    encoders: {
      "example-encoder": {
        name: "Fixture encoder",
        parameter_count: 1000,
        geometry: { hidden_size: 32, ffn_size: 128, num_attention_heads: 4, attention_head_dim: 8, gated_ffn: true },
        sources: [SOURCE],
      },
    },
  };
  const vaes = {
    vaes: {
      "example-vae": {
        name: "Fixture VAE decoder",
        parameter_count: 2000,
        loaded_subset: "Decoder only",
        geometry: {
          temporal_compression: 4,
          spatial_compression: 8,
          latent_channels: 16,
          output_channels: 3,
          vae_decoder_channels: 96,
        },
        decode_feature_multiplier: 48,
        sources: [SOURCE],
      },
    },
  };
  const config = {
    generator: {
      model_path: "Example/Video",
      engine: { num_gpus: 1, offload: { text_encoder: true } },
      pipeline: { experimental: { text_encoder_precisions: ["bf16"] } },
    },
    default_request: { sampling: { height: 480, width: 832, num_frames: 81 } },
  };
  const model = {
    memory_rules: "wan",
    components: {
      transformer: { ref: "example-transformer" },
      text_encoder: { ref: "example-encoder" },
      vae: { ref: "example-vae" },
    },
    reference_runs: [],
  };
  const manifest = {
    title: "Fixture model",
    deployments: {
      "cuda-rest": {
        runtime: "fixture-rest",
        workload: "t2v",
        defaults: "examples/serving/example.yaml",
        env: { TEST_BACKEND: "fixture" },
        options: [],
      },
    },
    capacity: model,
  };
  const options = {
    options: {},
    runtimes: {
      "fixture-rest": {
        metadata: {
          id: "fixture-rest",
          label: "Fixture REST",
          backend: "cuda",
          interface: "rest",
          launch_argv: ["fastvideo", "serve", "--config", "config.yaml"],
          server_defaults: { host: "0.0.0.0", port: 8000 },
        },
        workloads: ["t2v"],
        options: {},
      },
    },
  };
  const save = () => {
    for (const [name, data] of Object.entries({
      hardware,
      transformers,
      encoders,
      vaes,
      model: manifest,
      config,
      options,
    })) {
      write(paths[name], data);
    }
  };
  save();
  return { root, catalogDir, outputDir, paths, hardware, transformers, encoders, vaes, model, manifest, config, save };
}

function transformer(f) {
  return f.transformers.transformers["example-transformer"];
}

// Synthetic measurement records exercise the schema; they are never authored evidence.
function referenceRun(f) {
  const config = clone(f.config);
  config.generator.engine.num_gpus = 2;
  return {
    id: "fixture-run",
    date: "2026-10-18",
    hardware: "example-gpu",
    gpu_count: 2,
    outcome: "completed",
    execution: "warm_request",
    evidence_url: "https://example.test/fixture-run",
    context: {
      revisions: { fastvideo: "a".repeat(40), model: "b".repeat(40) },
      config,
      request: { sampling: { height: 480, width: 832, num_frames: 81 } },
      env: { TEST_BACKEND: "fixture" },
    },
    memory: {
      metric: "torch_peak_reserved",
      unit: "GiB",
      per_gpu: [
        { rank: 0, request_peak: 8, startup_peak: 10, stages: { denoising_stage: 7, decoding_stage: 8 } },
        { rank: 1, request_peak: 7, stages: { denoising_stage: 6, decoding_stage: 7 } },
      ],
    },
  };
}

test("capacity export resolves shared metadata and preserves native example configuration", (t) => {
  const f = fixture(t);
  const data = buildCapacityCatalog({ root: f.root });
  assert.deepEqual(data.hardware["example-gpu"], f.hardware.hardware["example-gpu"]);
  const model = data.models[0];
  assert.equal(model.id, "example-model");
  assert.equal(model.name, f.manifest.title);
  assert.equal(model.model_id, f.config.generator.model_path);
  assert.equal(model.examples[0].deployment_id, "example-model/cuda-rest");
  assert.equal(model.examples[0].config, f.manifest.deployments["cuda-rest"].defaults);
  assert.deepEqual(model.examples[0].base_config, f.config);
  assert.deepEqual(model.examples[0].env, { TEST_BACKEND: "fixture" });
  assert.deepEqual(model.reference_runs, []);
  assert.equal(existsSync(f.outputDir), false);
  assert.equal(model.components.transformer.parameter_count, 1_418_996_800);
  assert.equal(model.components.transformer.block_parameter_count, 1_393_221_120);
  assert.equal(model.components.transformer.kind, "transformer");
  const serialized = JSON.stringify(model);
  assert.equal(serialized.includes("1_418_996_800"), false);
  assert.equal(serialized.includes('"dtype"'), false);
});

test("component references must exist and match their role", (t) => {
  const f = fixture(t);
  f.model.components.transformer.ref = "missing";
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /missing|reference|ref/i);
  f.model.components.transformer.ref = "example-encoder";
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /transformer|kind|role/i);
});

test("unused components fail, while declared alternatives count as used references", (t) => {
  const f = fixture(t);
  f.vaes.vaes["alternative-vae"] = { ...clone(f.vaes.vaes["example-vae"]), name: "Alternative VAE" };
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /unused|unreferenced|alternative-vae/i);
  f.model.components.vae.alternatives = ["alternative-vae"];
  f.save();
  const model = buildCapacityCatalog({ root: f.root }).models[0];
  assert.equal(model.components.vae.alternatives[0].ref, "alternative-vae");
  assert.equal(model.components.vae.alternatives[0].parameter_count, 2000);
  f.model.components.vae.alternatives = ["example-encoder"];
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /vae|kind|role/i);
});

test("component IDs are globally unique and model IDs cannot collide", (t) => {
  const f = fixture(t);
  f.encoders.encoders["example-transformer"] = clone(f.encoders.encoders["example-encoder"]);
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /duplicate|unique|example-transformer/i);
  delete f.encoders.encoders["example-transformer"];
  f.save();
  const recipesDir = dirname(f.paths.model);
  const duplicatePath = join(recipesDir, "example-model.yml");
  write(duplicatePath, f.manifest);
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /duplicate|unique/i);
  rmSync(duplicatePath);
  write(join(recipesDir, "other-model.yaml"), f.manifest);
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /duplicate|unique/i);
});

test("duplicate authored YAML mapping keys are rejected before export", (t) => {
  const f = fixture(t);
  writeFileSync(f.paths.hardware, `${readFileSync(f.paths.hardware, "utf8")}hardware: {}\n`);
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /unique|duplicate|map keys/i);
  assert.equal(existsSync(f.outputDir), false);
});

test("parameter counts reject malformed grouping, non-finite, negative and unsafe values", (t) => {
  const f = fixture(t);
  for (const value of ["1_41_800", "1__418", "_1418", "1418_", "1,418", "1.5", NaN, -1, 1.5, 9007199254740992]) {
    transformer(f).parameter_count = value;
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }), undefined, String(value));
  }
});

test("streaming block metadata is paired and bounded by the component total", (t) => {
  const f = fixture(t);
  const original = clone(transformer(f));
  for (const change of [
    (component) => delete component.block_count,
    (component) => delete component.block_parameter_count,
    (component) => (component.block_count = -1),
    (component) => (component.block_count = 1.5),
    (component) => (component.block_parameter_count = 2_000_000_000),
  ]) {
    f.transformers.transformers["example-transformer"] = clone(original);
    change(transformer(f));
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }));
  }
});

test("geometry requires known fields and positive dimensions", (t) => {
  const f = fixture(t);
  const original = clone(transformer(f).geometry);
  for (const geometry of [
    { ...original, hidden_szie: 1536 },
    { ...original, hidden_size: 0 },
    { ...original, ffn_size: -1 },
    { ...original, patch_size: [1, 0, 2] },
  ]) {
    transformer(f).geometry = geometry;
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }));
  }
});

test("Wan model geometry is complete, while other families can catalog component counts alone", (t) => {
  const f = fixture(t);
  delete transformer(f).geometry.ffn_size;
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /geometry|ffn_size/i);
  transformer(f).geometry.ffn_size = 8960;
  delete f.vaes.vaes["example-vae"].geometry.temporal_compression;
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /geometry|temporal_compression/i);
  f.model.memory_rules = "future-family";
  delete transformer(f).geometry;
  delete f.vaes.vaes["example-vae"].geometry;
  f.save();
  assert.equal(buildCapacityCatalog({ root: f.root }).models[0].memory_rules, "future-family");
});

test("VAE geometry validates compression, latent/output channels and decoder width", (t) => {
  const f = fixture(t);
  const vae = f.vaes.vaes["example-vae"];
  const original = clone(vae.geometry);
  for (const [key, value, expected = new RegExp(key)] of [
    ["temporal_compression", 0],
    ["temporal_compression", "4"],
    ["spatial_compression", -1],
    ["spatial_compression", 1.5],
    ["latent_channels", 0],
    ["latent_channels", "16"],
    ["latent_channels", 1.5],
    ["output_channels", 0],
    ["output_channels", "3"],
    ["output_channels", true],
    ["vae_decoder_channels", 0],
    ["vae_decoder_channels", true],
    ["vae_decoder_channels", 9007199254740992, /browser-safe range/],
  ]) {
    vae.geometry = { ...clone(original), [key]: value };
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }), expected, `${key}: ${JSON.stringify(value)}`);
  }
});

test("gated FFN metadata requires booleans for both encoders and transformers", (t) => {
  const f = fixture(t);
  for (const component of [transformer(f), f.encoders.encoders["example-encoder"]]) {
    for (const value of ["true", 1, null, {}]) {
      component.geometry.gated_ffn = value;
      f.save();
      assert.throws(() => buildCapacityCatalog({ root: f.root }), /gated_ffn.*boolean/);
    }
    component.geometry.gated_ffn = false;
  }
  f.save();
  const { components } = buildCapacityCatalog({ root: f.root }).models[0];
  assert.equal(components.transformer.geometry.gated_ffn, false);
  assert.equal(components.text_encoder.geometry.gated_ffn, false);
});

test("VAE decode feature multipliers require positive finite numbers and a VAE component", (t) => {
  const f = fixture(t);
  const vae = f.vaes.vaes["example-vae"];
  for (const value of [0, -1, NaN, Infinity, "48", null, true, {}, []]) {
    vae.decode_feature_multiplier = value;
    f.save();
    assert.throws(
      () => buildCapacityCatalog({ root: f.root }),
      /decode_feature_multiplier.*positive finite|non-finite number/,
    );
  }
  vae.decode_feature_multiplier = 48;
  for (const component of [transformer(f), f.encoders.encoders["example-encoder"]]) {
    component.decode_feature_multiplier = 48;
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }), /decode_feature_multiplier is only supported for VAE/);
    delete component.decode_feature_multiplier;
  }
});

test("new activation fields are optional for existing catalog entries", (t) => {
  const f = fixture(t);
  f.vaes.vaes["example-vae"].geometry = {
    temporal_compression: 4,
    spatial_compression: 8,
    vae_decoder_channels: 96,
  };
  delete transformer(f).geometry.gated_ffn;
  delete f.encoders.encoders["example-encoder"].geometry;
  delete f.vaes.vaes["example-vae"].decode_feature_multiplier;
  f.save();
  const { models } = buildCapacityCatalog({ root: f.root });
  assert.equal(models.length, 1);
  assert.equal(Object.hasOwn(models[0].components.vae, "decode_feature_multiplier"), false);
});

test("VAE geometry excludes execution settings and unused architecture details", (t) => {
  const f = fixture(t);
  const vae = f.vaes.vaes["example-vae"];
  f.config.generator.pipeline.vae_tiling = true;
  Object.assign(f.config.generator.pipeline.experimental, {
    "vae_config.use_feature_cache": false,
    "vae_config.tile_sample_min_height": 512,
    "vae_config.tile_sample_min_width": 512,
    "vae_config.tile_sample_stride_height": 384,
    "vae_config.tile_sample_stride_width": 384,
  });
  f.save();
  assert.deepEqual(buildCapacityCatalog({ root: f.root }).models[0].examples[0].base_config, f.config);
  for (const [key, value] of [
    ["tiling", { enabled: true }],
    ["tile_size", [32, 32]],
    ["use_feature_cache", true],
    ["dtype", "bf16"],
    ["feature_multiplier", 48],
    ["decoder_channel_multipliers", [1, 2, 4, 4]],
    ["unrecognized_dimension", 4],
    ["gated_ffn", true],
  ]) {
    vae.geometry[key] = value;
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }), new RegExp(`unknown key ${key}`));
    delete vae.geometry[key];
  }
});

test("hardware and components require source URLs and do not fix a runtime dtype", (t) => {
  const f = fixture(t);
  transformer(f).sources = [];
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /source/i);
  transformer(f).sources = [[SOURCE]];
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /source|URL/i);
  transformer(f).sources = [SOURCE];
  f.hardware.hardware["example-gpu"].sources = ["not-a-url"];
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /source|URL|https/i);
  f.hardware.hardware["example-gpu"].sources = [SOURCE];
  transformer(f).dtype = "bf16";
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /dtype|unknown|unsupported/i);
});

test("capacity identity and examples follow the existing manifest deployments and baseline", (t) => {
  const f = fixture(t);
  f.manifest.title = "Updated title";
  f.config.generator.model_path = "Example/Updated";
  f.manifest.deployments["cuda-rest"].env.TEST_BACKEND = "updated";
  f.save();
  const model = buildCapacityCatalog({ root: f.root }).models[0];
  assert.equal(model.name, "Updated title");
  assert.equal(model.model_id, "Example/Updated");
  assert.equal(model.examples[0].env.TEST_BACKEND, "updated");
  assert.deepEqual(model.examples[0].base_config, f.config);
  f.manifest.deployments["cuda-rest"].defaults = "examples/serving/missing.yaml";
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /missing|exist/i);
  f.manifest.deployments = {};
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /deployment/i);
});

test("capacity fields are optional, strictly validated, and do not become serving controls", (t) => {
  const f = fixture(t);
  const before = buildCatalogs({ root: f.root });
  delete f.manifest.capacity;
  f.save();
  assert.deepEqual(buildCatalogs({ root: f.root }), before);
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /unused/i);
  f.manifest.capacity = f.model;
  for (const [key, value] of [
    ["model_id", "Example/Video"],
    ["examples", []],
    ["unexpected", true],
  ]) {
    f.model[key] = value;
    f.save();
    assert.throws(() => buildCatalogs({ root: f.root }), /unknown key/i);
    delete f.model[key];
  }
  f.manifest.capacity = null;
  f.save();
  assert.throws(() => buildCatalogs({ root: f.root }), /mapping/i);
});

test("capacity export reuses every deployment while leaving models without capacity valid", (t) => {
  const f = fixture(t);
  f.manifest.deployments["second-rest"] = { ...clone(f.manifest.deployments["cuda-rest"]), env: { SECOND: "1" } };
  f.save();
  const other = clone(f.manifest);
  delete other.capacity;
  other.deployments = { "cuda-rest": { ...other.deployments["cuda-rest"], defaults: "examples/serving/other.yaml" } };
  write(join(dirname(f.paths.model), "other.yaml"), other);
  write(join(f.root, "examples/serving/other.yaml"), { ...f.config, generator: { model_path: "Example/Other" } });
  const data = buildCapacityCatalog({ root: f.root });
  assert.equal(data.models.length, 1);
  assert.deepEqual(
    data.models[0].examples.map((example) => example.id),
    ["cuda-rest", "second-rest"],
  );
  assert.deepEqual(data.models[0].examples[1].env, { SECOND: "1" });
});

test("shipped FastWan capacity preserves the existing VSA and DMD serving recipe", () => {
  const model = buildCapacityCatalog().models.find((model) => model.id === "fastwan21");
  const existing = parse(readFileSync(join(ROOT, "examples/serving/openai_fastwan21_1_3b.yaml"), "utf8"));
  assert.equal(model.model_id, "FastVideo/FastWan2.1-T2V-1.3B-Diffusers");
  assert.deepEqual(model.examples[0].base_config, existing);
  assert.deepEqual(model.examples[0].env, { FASTVIDEO_ATTENTION_BACKEND: "VIDEO_SPARSE_ATTN" });
  assert.equal(model.components.transformer.parameter_count, 1_489_821_760);
  assert.equal(model.components.transformer.block_parameter_count, 1_464_046_080);
  assert.equal(model.components.text_encoder.parameter_count, 5_680_910_336);
  assert.equal(model.components.vae.parameter_count, 73_296_659);
  assert.deepEqual(model.components.vae.geometry, {
    temporal_compression: 4,
    spatial_compression: 8,
    latent_channels: 16,
    output_channels: 3,
    vae_decoder_channels: 96,
  });
  assert.equal(model.components.vae.decode_feature_multiplier, 48);
  assert.deepEqual(model.reference_runs, []);
});

test("reference runs are optional and measured records preserve request and per-rank peaks", (t) => {
  const f = fixture(t);
  delete f.model.reference_runs;
  f.save();
  assert.equal(buildCapacityCatalog({ root: f.root }).models.length, 1);
  f.model.reference_runs = null;
  f.save();
  assert.throws(() => buildCapacityCatalog({ root: f.root }), /reference_runs must be a list/);
  const run = referenceRun(f);
  f.model.reference_runs = [run];
  f.save();
  assert.deepEqual(buildCapacityCatalog({ root: f.root }).models[0].reference_runs, [run]);
  delete run.memory;
  f.save();
  assert.deepEqual(buildCapacityCatalog({ root: f.root }).models[0].reference_runs, [run]);
});

test("reference runs reject inconsistent hardware, configuration and measurement coverage", (t) => {
  const f = fixture(t);
  f.model.reference_runs = [referenceRun(f)];
  f.save();
  assert.equal(buildCapacityCatalog({ root: f.root }).models[0].reference_runs.length, 1);
  const cases = [
    ["missing hardware", (run) => (run.hardware = "missing-gpu")],
    ["wrong model", (run) => (run.context.config.generator.model_path = "Example/Wrong")],
    ["wrong GPU count", (run) => (run.context.config.generator.engine.num_gpus = 1)],
    ["missing rank", (run) => run.memory.per_gpu.pop()],
    ["duplicate rank", (run) => (run.memory.per_gpu[1].rank = 0)],
    ["rank outside layout", (run) => (run.memory.per_gpu[1].rank = 2)],
    ["nonpositive peak", (run) => (run.memory.per_gpu[0].request_peak = 0)],
    ["stage exceeds request", (run) => (run.memory.per_gpu[0].stages.decoding_stage = 9)],
    ["unsupported metric", (run) => (run.memory.metric = "unknown")],
    ["failed execution", (run) => (run.outcome = "oom")],
    ["missing evidence", (run) => delete run.evidence_url],
    ["evidence URL must be text", (run) => (run.evidence_url = [run.evidence_url])],
  ];
  for (const [name, change] of cases) {
    const run = referenceRun(f);
    change(run);
    f.model.reference_runs = [run];
    f.save();
    assert.throws(() => buildCapacityCatalog({ root: f.root }), undefined, name);
  }
});

test("exports use relative URLs and browser loaders resolve them under a deployment prefix", async (t) => {
  const f = fixture(t);
  const index = exportCapacityCatalog({ root: f.root });
  assert.equal(index.hardware_url, "hardware.json");
  assert.deepEqual(
    index.models.map(({ id, catalog_url }) => ({ id, catalog_url })),
    [{ id: "example-model", catalog_url: "models/example-model.json" }],
  );
  const indexUrl = "https://example.test/FastVideo/assets/cookbook-config/capacity/index.json";
  const requested = [];
  const signal = new AbortController().signal;
  const fetcher = async (url, options) => {
    requested.push(url);
    assert.equal(options.signal, signal);
    const relative = new URL(url).pathname.split("/capacity/")[1];
    const data = JSON.parse(readFileSync(join(f.outputDir, relative), "utf8"));
    return { ok: true, url, json: async () => data };
  };
  const loadedIndex = await loadCapacityIndex(indexUrl, { fetcher, signal });
  assert.equal(loadedIndex.url, indexUrl);
  const model = await loadCapacityModel(loadedIndex, "example-model", { fetcher, signal });
  assert.equal(model.model_id, "Example/Video");
  assert.deepEqual(model.components.transformer.geometry, transformer(f).geometry);
  assert.deepEqual(model.components.text_encoder.geometry, f.encoders.encoders["example-encoder"].geometry);
  assert.deepEqual(model.components.vae.geometry, f.vaes.vaes["example-vae"].geometry);
  assert.equal(model.components.vae.decode_feature_multiplier, f.vaes.vaes["example-vae"].decode_feature_multiplier);
  assert.deepEqual(model.examples[0].base_config, f.config);
  const hardware = await loadCapacityHardware(loadedIndex, { fetcher, signal });
  assert.deepEqual(hardware["example-gpu"], f.hardware.hardware["example-gpu"]);
  assert.deepEqual(requested, [
    indexUrl,
    new URL(index.models[0].catalog_url, indexUrl).href,
    new URL(index.hardware_url, indexUrl).href,
  ]);
  await assert.rejects(() => loadCapacityModel(loadedIndex, "missing", { fetcher, signal }), /missing|model/i);
  assert.equal(requested.length, 3);
});

test("browser loaders reject malformed or mismatched catalog responses", async () => {
  const url = "https://example.test/capacity/index.json";
  const index = {
    url,
    hardware_url: "hardware.json",
    models: [
      {
        id: "example-model",
        name: "Fixture model",
        model_id: "Example/Video",
        catalog_url: "models/example-model.json",
      },
    ],
  };
  const fetcher = (data) => async () => ({ ok: true, json: async () => clone(data) });
  await assert.rejects(() => loadCapacityIndex(url, { fetcher: fetcher({ models: [] }) }), /index/i);
  await assert.rejects(
    () => loadCapacityIndex(url, { fetcher: fetcher({ ...index, models: [...index.models, ...index.models] }) }),
    /duplicate/i,
  );
  await assert.rejects(
    () => loadCapacityModel(index, "example-model", { fetcher: fetcher({ id: "other-model" }) }),
    /invalid|model/i,
  );
  await assert.rejects(
    () =>
      loadCapacityHardware(index, {
        fetcher: fetcher({ gpu: { name: "Invalid capacity", memory_type: "dedicated", memory_gb: -1 } }),
      }),
    /hardware/i,
  );
  for (const load of [
    () => loadCapacityModel(index, "example-model", { fetcher: async () => ({ ok: false, status: 404 }) }),
    () => loadCapacityHardware(index, { fetcher: async () => ({ ok: false, status: 404 }) }),
  ]) {
    await assert.rejects(load, /404/);
  }
});

test("browser loader failures preserve HTTP errors and cancellation", async () => {
  const url = "https://example.test/capacity/index.json";
  await assert.rejects(() => loadCapacityIndex(url, { fetcher: async () => ({ ok: false, status: 503 }) }), /503/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () =>
      loadCapacityIndex(url, {
        signal: controller.signal,
        fetcher: async () => ({ ok: true, json: async () => ({}) }),
      }),
    { name: "AbortError" },
  );
});

test("check and preview validate fixture catalogs without writing generated assets", (t) => {
  const f = fixture(t);
  for (const args of [["--check"], ["--preview", "example-model"]]) {
    const result = spawnSync(process.execPath, [SCRIPT, "--root", f.root, ...args], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    if (args[0] === "--preview") {
      const preview = parse(result.stdout);
      assert.equal(preview.id, "example-model");
      assert.deepEqual(preview.examples[0].base_config, f.config);
    }
    assert.equal(existsSync(f.outputDir), false);
  }
});

test(
  "export rejects a dangling output-file symlink without creating its target",
  { skip: process.platform === "win32" },
  (t) => {
    const f = fixture(t);
    const outside = mkdtempSync(join(tmpdir(), "cookbook-capacity-external-"));
    t.after(() => rmSync(outside, { recursive: true, force: true }));
    const target = join(outside, "must-not-be-created.json");
    mkdirSync(f.outputDir, { recursive: true });
    symlinkSync(target, join(f.outputDir, "index.json"));
    assert.equal(existsSync(target), false);
    assert.throws(() => exportCapacityCatalog({ root: f.root }), /symlink/i);
    assert.equal(existsSync(target), false);
    assert.deepEqual(readdirSync(outside), []);
  },
);

test(
  "export rejects a symlinked output ancestor without writing outside the fixture",
  { skip: process.platform === "win32" },
  (t) => {
    const f = fixture(t);
    const outside = mkdtempSync(join(tmpdir(), "cookbook-capacity-external-"));
    t.after(() => rmSync(outside, { recursive: true, force: true }));
    const sentinel = join(outside, "keep.txt");
    writeFileSync(sentinel, "unchanged\n");
    const ancestor = join(f.root, "docs/assets/cookbook-config");
    mkdirSync(dirname(ancestor), { recursive: true });
    symlinkSync(outside, ancestor, "dir");
    assert.throws(() => exportCapacityCatalog({ root: f.root }), /symlink/i);
    assert.deepEqual(readdirSync(outside), ["keep.txt"]);
    assert.equal(readFileSync(sentinel, "utf8"), "unchanged\n");
  },
);
