import assert from "node:assert/strict";
import test from "node:test";

import { buildCapacityCatalog } from "../js/build-cookbook-capacity.mjs";
import {
  BYTES_PER_WEIGHT,
  DEFAULT_HEADROOM,
  assessCapacity,
  estimateMemory,
  matchEvidence,
  recommendCapacity,
  resolveExecution,
  resolveTopology,
} from "../js/cookbook-memory.mjs";

const GIB = 1024 ** 3;
const NO_OFFLOAD = { dit: false, dit_layerwise: false, text_encoder: false, vae: false };
const FASTWAN_OFFLOAD = { dit: false, dit_layerwise: true, text_encoder: true, vae: false };

const DIT = 1_418_996_800;
const DIT_BLOCKS = 1_393_221_120;
const TEXT = 5_680_910_336;
const TEXT_BLOCKS = 4_630_757_376;
const VAE = 73_296_659;

function wanModel(change = () => {}) {
  const model = {
    id: "wan21",
    name: "Wan2.1 T2V 1.3B",
    model_id: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    memory_rules: "wan",
    components: {
      transformer: {
        ref: "wan2.1-t2v-1.3b-dit",
        kind: "transformer",
        parameter_count: DIT,
        block_count: 30,
        block_parameter_count: DIT_BLOCKS,
        geometry: { hidden_size: 1536, ffn_size: 8960, patch_size: [1, 2, 2], gated_ffn: false },
      },
      text_encoder: {
        ref: "umt5-xxl-encoder",
        kind: "encoder",
        parameter_count: TEXT,
        block_count: 24,
        block_parameter_count: TEXT_BLOCKS,
      },
      vae: {
        ref: "wan2.1-vae-decoder",
        kind: "vae",
        parameter_count: VAE,
        geometry: {
          temporal_compression: 4,
          spatial_compression: 8,
          latent_channels: 16,
          output_channels: 3,
          vae_decoder_channels: 96,
        },
        decode_feature_multiplier: 48,
      },
    },
    examples: [],
    reference_runs: [],
  };
  change(model);
  return model;
}

function hardware({
  id = "test-80gb",
  memory_gb = 80,
  memory_type = "dedicated",
  backend = "cuda",
  dtypes = ["fp32", "fp16", "bf16", "fp8", "nvfp4"],
  host_memory_gb,
  partitions,
} = {}) {
  return {
    id,
    name: `Test GPU ${memory_gb} GB`,
    memory_type,
    memory_gb,
    backend,
    supported_dtypes: dtypes,
    ...(host_memory_gb === undefined ? {} : { host_memory_gb }),
    ...(partitions === undefined ? {} : { partitions }),
  };
}

function config({ offload = FASTWAN_OFFLOAD, engine = {}, pipeline = {}, sampling, server = {} } = {}) {
  return {
    generator: {
      model_path: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      engine: { num_gpus: 1, offload, ...engine },
      pipeline,
    },
    server: { host: "0.0.0.0", port: 8000, ...server },
    default_request: { sampling: sampling ?? { num_frames: 81, height: 480, width: 832, seed: 1024, fps: 16 } },
  };
}

function example(base = config(), env = {}) {
  return {
    id: "cuda-rest",
    deployment_id: "wan21/cuda-rest",
    workload: "t2v",
    runtime: "fastvideo-cuda-rest",
    config: "examples/serving/openai_wan21.yaml",
    env,
    base_config: base,
  };
}

function estimate(options = {}) {
  return estimateMemory({
    config: config(),
    workload: "t2v",
    model: wanModel(),
    hardware: hardware({ memory_gb: 80 }),
    ...options,
  });
}

const gib = (parameters, dtype) => (parameters * BYTES_PER_WEIGHT[dtype]) / GIB;
const streamed = (parameters, blockParameters, blocks, dtype) =>
  gib(parameters - blockParameters, dtype) + (gib(blockParameters, dtype) * 2) / blocks;

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} differs from ${expected}`);
}

function assertWeights(result, expected) {
  for (const [phase, weightsGiB] of Object.entries(expected)) close(result.phases[phase].weightsGiB, weightsGiB);
}

test("each storage format has its byte size per weight", () => {
  assert.deepEqual(BYTES_PER_WEIGHT, { fp32: 4, fp16: 2, bf16: 2, fp8: 1, nvfp4: 0.5625 });
});

test("automatic topology follows the native -1 sentinels", () => {
  assert.deepEqual(resolveTopology(), { gpus: 1, tpSize: 1, spSize: 1 });
  assert.deepEqual(resolveTopology({ num_gpus: 4 }), { gpus: 4, tpSize: 1, spSize: 4 });
  assert.deepEqual(resolveTopology({ num_gpus: 4, parallelism: { tp_size: 2, sp_size: -1 } }), {
    gpus: 4,
    tpSize: 2,
    spSize: 4,
  });
});

test("native Wan precision defaults resolve without metadata dtypes", () => {
  const execution = resolveExecution({ model: wanModel(), config: config({ offload: {} }) });
  assert.deepEqual(execution.precision, {
    transformer: "bf16",
    text_encoder: ["fp32"],
    image_encoder: ["fp32"],
    vae: "fp32",
    vae_decode: "bf16",
  });
  assert.equal(execution.offload.dit_layerwise, true);
  assert.equal(execution.vae_tiling, false);
});

test("explicit serving precision and offload settings override family defaults", () => {
  const execution = resolveExecution({
    model: wanModel(),
    config: config({
      offload: { dit_layerwise: false, text_encoder: false, vae: false },
      pipeline: {
        experimental: {
          dit_precision: "fp16",
          text_encoder_precisions: ["bf16"],
          vae_precision: "bf16",
          vae_decode_precision: "fp32",
        },
      },
    }),
  });
  assert.equal(execution.precision.transformer, "fp16");
  assert.deepEqual(execution.precision.text_encoder, ["bf16"]);
  assert.equal(execution.precision.vae, "bf16");
  assert.equal(execution.precision.vae_decode, "fp32");
  assert.equal(execution.offload.dit_layerwise, false);
});

test("startup, encoding, denoising and decoding keep only resident weights", () => {
  const result = estimate();
  const ditOutside = gib(DIT - DIT_BLOCKS, "bf16");
  assert.equal(result.status, "fits");
  assert.equal(result.basis, "analytical");
  assertWeights(result, {
    startup: gib(DIT, "bf16") + gib(VAE, "fp32"),
    encoding: gib(TEXT, "fp32") + ditOutside + gib(VAE, "fp32"),
    denoising: streamed(DIT, DIT_BLOCKS, 30, "bf16") + gib(VAE, "fp32"),
    decoding: ditOutside + gib(VAE, "fp32"),
  });
  assert.equal(result.peakPhase, "encoding");
  close(result.requiredGiB, result.peakGiB * (1 + DEFAULT_HEADROOM.fraction) + DEFAULT_HEADROOM.reserveGiB);
});

test("layerwise resident blocks stay on the device outside denoising", () => {
  const result = estimate({ env: { FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS: "4" } });
  const outside = gib(DIT - DIT_BLOCKS, "bf16");
  const blocks = gib(DIT_BLOCKS, "bf16");
  assertWeights(result, {
    encoding: gib(TEXT, "fp32") + outside + (blocks * 4) / 30 + gib(VAE, "fp32"),
    denoising: outside + (blocks * 6) / 30 + gib(VAE, "fp32"),
    decoding: outside + (blocks * 4) / 30 + gib(VAE, "fp32"),
  });
  assert.equal(result.execution.layerwise_resident_blocks, 4);
});

test("unified memory disables native CPU offload before counting weights", () => {
  const unified = {
    id: "unified",
    name: "Fixture unified device",
    memory: { type: "unified", nominal: { value: 128, unit: "GB" } },
    backend: "cuda",
    supported_dtypes: ["fp32", "fp16", "bf16"],
  };
  const result = estimate({
    hardware: unified,
    config: config({
      offload: {
        dit: true,
        dit_layerwise: true,
        text_encoder: true,
        image_encoder: true,
        vae: true,
        lazy_module_load: false,
      },
    }),
  });
  assert.equal(result.status, "fits");
  assert.deepEqual(result.execution.offload, {
    dit: false,
    dit_layerwise: false,
    text_encoder: false,
    image_encoder: false,
    vae: false,
    lazy_module_load: false,
  });
  assertWeights(result, {
    startup: gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32"),
    encoding: gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32"),
    denoising: gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32"),
    decoding: gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32"),
  });
  assert.ok(result.notes.some((note) => /offload is disabled/i.test(note)));
});

test("a disabled VAE feature cache returns unknown", () => {
  const result = estimate({
    config: config({ pipeline: { experimental: { "vae_config.use_feature_cache": false } } }),
  });
  assert.equal(result.status, "unknown");
  assert.match(result.reason, /feature caching/i);
});

test("trajectory capture returns unknown instead of a dense estimate", () => {
  assert.notEqual(estimate().status, "unknown");
  for (const flag of ["return_trajectory_latents", "return_trajectory_decoded"]) {
    const result = estimate({ request: { sampling: config().default_request.sampling, runtime: { [flag]: true } } });
    assert.equal(result.status, "unknown");
    assert.match(result.reason, /trajectory/i);
  }
});

test("layerwise offload requires block counts", () => {
  const model = wanModel((item) => {
    delete item.components.transformer.block_count;
    delete item.components.transformer.block_parameter_count;
  });
  const result = estimate({ model });
  assert.equal(result.status, "unknown");
  assert.match(result.reason, /block counts/i);
});

test("a 24 GB device does not fit the fp32 UMT5 encoding phase", () => {
  const result = estimate({ hardware: hardware({ memory_gb: 24 }) });
  assert.equal(result.status, "exceeds");
  assert.equal(result.peakPhase, "encoding");
  close(result.capacityGiB, 24e9 / GIB);
});

test("whole-model DiT offload keeps the transformer on the device in every phase", () => {
  const result = estimate({
    config: config({ offload: { dit: true, dit_layerwise: false, text_encoder: true, vae: false } }),
  });
  assertWeights(result, {
    encoding: gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32"),
    denoising: gib(DIT, "bf16") + gib(VAE, "fp32"),
    decoding: gib(DIT, "bf16") + gib(VAE, "fp32"),
  });
});

test("disabling every offload keeps complete components resident and counts load-time startup", () => {
  const result = estimate({ config: config({ offload: NO_OFFLOAD }) });
  const complete = gib(DIT, "bf16") + gib(TEXT, "fp32") + gib(VAE, "fp32");
  assertWeights(result, { startup: complete, encoding: complete, denoising: complete, decoding: complete });
  assert.equal(result.peakPhase, "decoding");
});

test("explicit precisions change storage and activation bytes", () => {
  const result = estimate({
    config: config({
      pipeline: {
        experimental: {
          dit_precision: "fp16",
          text_encoder_precisions: ["bf16"],
          vae_precision: "bf16",
          vae_decode_precision: "fp32",
        },
      },
    }),
  });
  assertWeights(result, {
    encoding: gib(TEXT, "bf16") + gib(DIT - DIT_BLOCKS, "fp16") + gib(VAE, "bf16"),
    denoising: streamed(DIT, DIT_BLOCKS, 30, "fp16") + gib(VAE, "bf16"),
    decoding: gib(DIT - DIT_BLOCKS, "fp16") + gib(VAE, "bf16"),
  });
  const pixels = 480 * 832;
  close(
    result.phases.decoding.workingGiB,
    (48 * 96 * pixels * BYTES_PER_WEIGHT.fp32 + 81 * 3 * pixels * (2 * BYTES_PER_WEIGHT.fp32 + 4)) / GIB,
  );
});

test("the VAE decode multiplier comes from component metadata", () => {
  const model = wanModel((item) => (item.components.vae.decode_feature_multiplier = 24));
  const result = estimate({ model });
  const pixels = 480 * 832;
  const output = 81 * 3 * pixels * (2 * BYTES_PER_WEIGHT.bf16 + 4);
  close(result.phases.decoding.workingGiB, (24 * 96 * pixels * BYTES_PER_WEIGHT.bf16 + output) / GIB);
});

test("VAE tiling is disclosed but never treated as a measured discount", () => {
  const result = estimate({ config: config({ pipeline: { vae_tiling: true } }) });
  assert.ok(result.notes.some((note) => /tiling/.test(note)));
  assert.equal(result.status, "fits");
});

test("multi-GPU, FSDP and lazy loading are unknown rather than pooled", () => {
  const cases = [
    ["multiple GPUs", { config: config({ engine: { num_gpus: 2 } }) }],
    ["tensor parallel", { config: config({ engine: { num_gpus: 2, parallelism: { tp_size: 2 } } }) }],
    ["FSDP", { config: config({ engine: { use_fsdp_inference: true } }) }],
    ["lazy loading", { config: config({ engine: { offload: { lazy_module_load: true } } }) }],
  ];
  for (const [name, options] of cases) {
    const result = estimate(options);
    assert.equal(result.status, "unknown", name);
    assert.ok(result.reason.length > 0, name);
  }
});

test("VSA, compilation and quantization stop the analytical rule", () => {
  const cases = [
    ["VSA", { env: { FASTVIDEO_ATTENTION_BACKEND: "VIDEO_SPARSE_ATTN" } }],
    ["VSA experiment", { config: config({ pipeline: { experimental: { VSA_sparsity: 0.8 } } }) }],
    ["compile", { config: config({ engine: { compile: { enabled: true } } }) }],
    ["transformer quantization", { config: config({ engine: { quantization: { transformer_quant: "NVFP4" } } }) }],
    [
      "retained weights",
      { config: config({ engine: { quantization: { transformer_retain_original_weights: true } } }) },
    ],
    ["text quantization", { config: config({ engine: { quantization: { text_encoder_quant: "fp8" } } }) }],
    ["unknown backend", { env: { FASTVIDEO_ATTENTION_BACKEND: "BSA_ATTN" } }],
  ];
  for (const [name, options] of cases) {
    const result = estimate(options);
    assert.equal(result.status, "unknown", name);
    assert.ok(result.reason.length > 0, name);
  }
});

test("unsupported weight formats are distinct from unknown coverage", () => {
  const result = estimate({ hardware: hardware({ dtypes: ["fp16", "bf16"] }) });
  assert.equal(result.status, "unsupported");
  assert.match(result.reason, /fp32/);
});

test("missing geometry, counts, request size and modeled components stay unknown", () => {
  const cases = [
    ["unknown rule", wanModel((model) => (model.memory_rules = "ltx2"))],
    ["missing geometry", wanModel((model) => delete model.components.transformer.geometry.ffn_size)],
    ["missing count", wanModel((model) => delete model.components.vae.parameter_count)],
    ["missing VAE", wanModel((model) => delete model.components.vae)],
    ["unmodeled role", wanModel((model) => (model.components.audio_encoder = { kind: "encoder", parameter_count: 1 }))],
    ["wrong kind", wanModel((model) => (model.components.vae.kind = "encoder"))],
  ];
  for (const [name, model] of cases) {
    assert.equal(estimate({ model }).status, "unknown", name);
  }
  assert.equal(estimate({ config: config({ sampling: {} }) }).status, "unknown");
  assert.equal(
    estimate({
      config: config({ sampling: { num_frames: 81, height: 480, width: 832, num_videos_per_prompt: 2 } }),
    }).status,
    "unknown",
  );
  assert.equal(
    estimate({ config: config({ sampling: { num_frames: 81, height: 480, width: 832, batch_cfg: true } }) }).status,
    "unknown",
  );
  assert.equal(estimate({ workload: "i2v" }).status, "unknown");
});

test("the FastWan catalog baseline is unknown because VSA and compilation are not modeled", () => {
  const model = buildCapacityCatalog().models.find((item) => item.id === "fastwan21");
  const result = assessCapacity({ model, example: model.examples[0], hardware: hardware({ memory_gb: 32 }) });
  assert.equal(result.status, "unknown");
  assert.match(result.reason, /VSA/);
  assert.equal(result.compatibility, "unverified");
});

function referenceRun(base, { peak = 30, scope = "device" } = {}) {
  return {
    id: "run-1",
    date: "2026-10-18",
    hardware: "test-80gb",
    gpu_count: 1,
    outcome: "completed",
    execution: "warm_request",
    evidence_url: "https://example.test/run-1",
    context: {
      runtime: "fastvideo-cuda-rest",
      deployment: "wan21/cuda-rest",
      revisions: { fastvideo: "a".repeat(40), model: "b".repeat(40) },
      config: structuredClone(base),
      request: { sampling: structuredClone(base.default_request.sampling) },
      env: { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" },
    },
    memory: {
      metric: scope === "device" ? "device_used" : "torch_peak_reserved",
      scope,
      unit: "GiB",
      per_gpu: [{ rank: 0, request_peak: peak, startup_peak: peak - 2 }],
    },
  };
}

const REVISIONS = { fastvideo: "a".repeat(40), model: "b".repeat(40) };

test("measured evidence on the full card does not apply to a selected partition", () => {
  const partitioned = hardware({
    id: "h100",
    memory_gb: 80,
    partitions: [{ id: "1g.10gb", label: "1g.10gb", memory_gb: 10 }],
  });
  const base = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const model = wanModel();
  const ex = example(base, env);
  const run = referenceRun(base);
  run.hardware = "h100";
  model.reference_runs = [run];
  const fullCard = assessCapacity({
    model,
    example: ex,
    hardware: partitioned,
    env: ex.env,
    revisions: REVISIONS,
    partitionId: "1g.10gb",
  });
  assert.equal(fullCard.evidence.applicable.length, 0);
  assert.equal(fullCard.evidence.rejected[0].differences[0].path, "hardware_partition");
  assert.notEqual(fullCard.basis, "measured");
  run.hardware_partition = "1g.10gb";
  const partitionedRun = assessCapacity({
    model,
    example: ex,
    hardware: partitioned,
    env: ex.env,
    revisions: REVISIONS,
    partitionId: "1g.10gb",
  });
  assert.equal(partitionedRun.evidence.applicable.length, 1);
  assert.equal(partitionedRun.basis, "measured");
  assert.equal(partitionedRun.status, "exceeds");
});

test("matchEvidence uses the example environment when the caller omits env", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const model = wanModel((item) => (item.reference_runs = [referenceRun(base)]));
  const match = matchEvidence({
    model,
    example: example(base, env),
    config: base,
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(match.applicable.length, 1);
});

test("a request-only measurement cannot establish an overall device fit", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const run = referenceRun(base, { peak: 20 });
  delete run.memory.per_gpu[0].startup_peak;
  const model = wanModel((item) => (item.reference_runs = [run]));
  const result = assessCapacity({
    model,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(result.basis, "measured");
  assert.equal(result.status, "unknown");
  assert.match(result.reason, /Startup peak/i);
});

test("a measured device fit does not override a host-memory failure", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const model = wanModel((item) => (item.reference_runs = [referenceRun(base, { peak: 20 })]));
  const result = assessCapacity({
    model,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
    hostMemoryGb: 8,
  });
  assert.equal(result.basis, "measured");
  assert.equal(result.deviceStatus, "fits");
  assert.equal(result.status, "exceeds");
  assert.equal(result.host.status, "exceeds");
});

test("measured fits use every applicable run rather than catalog order", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const first = referenceRun(base, { peak: 20 });
  const second = referenceRun(base, { peak: 30 });
  second.id = "run-2";
  const model = wanModel((item) => (item.reference_runs = [first, second]));
  const result = assessCapacity({
    model,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb", memory_gb: 25 }),
    revisions: REVISIONS,
  });
  assert.equal(result.basis, "measured");
  assert.equal(result.status, "exceeds");
  close(result.peakGiB, 30);
  assert.equal(result.run, "run-2");
  assert.deepEqual(result.runs, ["run-1", "run-2"]);
});

test("unified memory uses an explicit application budget and never nominal capacity", () => {
  const spark = {
    id: "spark-128",
    name: "Fixture unified system",
    memory: { type: "unified", nominal: { value: 128, unit: "GB" } },
    backend: "cuda",
    supported_dtypes: ["fp32", "fp16", "bf16", "fp8"],
  };
  const withoutBudget = estimate({ hardware: spark });
  assert.equal(withoutBudget.status, "fits");
  assert.equal(withoutBudget.memory.budgetSource, "assumed_fraction");
  assert.ok(withoutBudget.capacityGiB < 128);
  assert.match(withoutBudget.notes.join(" "), /75% of unified memory/);
  const withBudget = estimate({ hardware: spark, memoryBudgetGiB: 64 });
  assert.equal(withBudget.status, "fits");
  assert.equal(withBudget.capacityGiB, 64);
  assert.equal(withBudget.memory.budgetGiB, 64);
  assert.equal(withBudget.memory.type, "unified");
  assert.equal(withBudget.hardware.capacity_gib, 64);
  assert.equal(withBudget.host.status, "shared");
  assert.match(withBudget.notes.join(" "), /shared with the CPU and OS/);
  assert.equal(estimate({ hardware: spark, memoryBudgetGiB: 129 }).status, "unknown");
});

test("a selected partition never inherits the full device capacity", () => {
  const partitioned = hardware({
    id: "h100",
    memory_gb: 80,
    partitions: [{ id: "1g.10gb", label: "1g.10gb", memory_gb: 10 }],
  });
  const partition = estimate({ hardware: partitioned, partitionId: "1g.10gb" });
  assert.ok(partition.capacityGiB < 10);
  assert.equal(partition.status, "exceeds");
  assert.equal(partition.memory.partition, "1g.10gb");
  assert.equal(partition.hardware.capacity_gib, partition.capacityGiB);
  assert.match(partition.notes.join(" "), /full 74\.5 GiB is not available/);
  assert.equal(estimate({ hardware: partitioned, partitionId: "missing" }).status, "unknown");
});

test("a selected partition is capped by the smaller application budget", () => {
  const partitioned = hardware({
    id: "h100",
    memory_gb: 80,
    partitions: [{ id: "3g.40gb", label: "3g.40gb", memory: { value: 40, unit: "GB" } }],
  });
  const result = estimate({ hardware: partitioned, partitionId: "3g.40gb", memoryBudgetGiB: 16 });
  assert.equal(result.capacityGiB, 16);
  assert.equal(result.memory.budgetGiB, 16);
  assert.match(result.notes.join(" "), /smaller than/i);
  const allocation = estimate({ hardware: hardware({ memory_gb: 80 }), allocationMemoryGb: 40, memoryBudgetGiB: 16 });
  assert.equal(allocation.capacityGiB, 16);
  assert.equal(allocation.memory.partition, null);
});

test("custom hardware normalizes authored partition, nominal and host quantities", () => {
  const custom = {
    id: "custom",
    name: "Custom partitioned device",
    custom: true,
    memory: { type: "dedicated", nominal: { value: 80, unit: "GB" } },
    partitions: [{ id: "slice", label: "slice", memory: { value: 20, unit: "GB" } }],
    host_memory: { value: 64, unit: "GB" },
    backend: "cuda",
    supported_dtypes: ["fp32", "fp16", "bf16"],
  };
  const result = estimate({ hardware: custom, partitionId: "slice" });
  close(result.capacityGiB, (20 * 1e9) / GIB);
  close(result.host.capacityGiB, (64 * 1e9) / GIB);
  assert.notEqual(result.status, "unknown");
  assert.equal(
    estimate({
      hardware: { ...custom, partitions: [{ id: "bad", label: "bad" }] },
      partitionId: "bad",
    }).status,
    "unknown",
  );
});

test("custom hardware uses an explicit memory type and capacity", () => {
  const custom = {
    id: "custom",
    name: "Custom accelerator",
    custom: true,
    memory: { type: "dedicated", nominal: { value: 48, unit: "GB" } },
    backend: "cuda",
    supported_dtypes: ["fp32", "fp16", "bf16", "fp8"],
  };
  const result = estimate({ hardware: custom });
  assert.equal(result.status, "fits");
  assert.equal(result.hardware.custom, true);
  assert.match(result.notes.join(" "), /not catalog-verified/);
  assert.equal(estimate({ hardware: { ...custom, backend: "rocm" } }).status, "unknown");
  assert.equal(estimate({ hardware: { ...custom, memory: undefined, memory_type: undefined } }).status, "unknown");
  const unspecifiedFormats = { ...custom, supported_dtypes: undefined };
  const defaultFormats = estimate({ hardware: unspecifiedFormats });
  assert.equal(defaultFormats.status, "fits");
  assert.match(defaultFormats.notes.join(" "), /assumes fp32, fp16, or bf16/);
  assert.equal(
    estimate({ hardware: { ...unspecifiedFormats, supported_dtypes: ["fp32", "fp16", "bf16", "fp8"] } }).status,
    "fits",
  );
});

test("host memory for offload is separate from device memory", () => {
  const constrained = estimate({ hardware: hardware({ memory_gb: 80, host_memory_gb: 8 }) });
  assert.equal(constrained.status, "exceeds");
  assert.equal(constrained.host.status, "exceeds");
  close(constrained.host.capacityGiB, (8 * 1e9) / GIB);
  assert.ok(constrained.host.requiredGiB > constrained.host.capacityGiB);
  const sufficient = estimate({ hardware: hardware({ memory_gb: 80, host_memory_gb: 64 }) });
  assert.equal(sufficient.status, "fits");
  assert.equal(sufficient.host.status, "fits");
  const override = estimate({ hardware: hardware({ memory_gb: 80, host_memory_gb: 8 }), hostMemoryGb: 24 });
  const authored = estimate({ hardware: hardware({ memory_gb: 80, host_memory_gb: 24 }) });
  close(override.host.capacityGiB, (24 * 1e9) / GIB);
  close(override.host.capacityGiB, authored.host.capacityGiB);
  assert.equal(override.host.status, "exceeds");
  const unknown = estimate({ hardware: hardware({ memory_gb: 80 }) });
  assert.equal(unknown.status, "fits");
  assert.equal(unknown.host.status, "unknown");
  assert.match(unknown.notes.join(" "), /Host memory for offload is not specified/);
  const noOffload = estimate({
    hardware: hardware({ memory_gb: 80, host_memory_gb: 8 }),
    config: config({ offload: NO_OFFLOAD }),
  });
  assert.equal(noOffload.host.status, "not_required");
});

test("evidence applies only when memory-relevant settings and revisions match", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const model = wanModel((item) => (item.reference_runs = [referenceRun(base)]));
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const match = matchEvidence({
    model,
    example: example(base, env),
    config: base,
    env,
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(match.applicable.length, 1);
  assert.equal(match.unverified.length, 0);
  assert.equal(match.rejected.length, 0);

  const harmless = structuredClone(base);
  harmless.server.port = 9000;
  harmless.default_request.sampling.seed = 7;
  harmless.default_request.sampling.fps = 24;
  const still = matchEvidence({
    model,
    example: example(harmless, env),
    config: harmless,
    env,
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(still.applicable.length, 1);

  const changed = structuredClone(base);
  changed.default_request.sampling.height = 720;
  const rejected = matchEvidence({
    model,
    example: example(changed, env),
    config: changed,
    env,
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(rejected.applicable.length, 0);
  assert.equal(rejected.rejected.length, 1);
  assert.ok(rejected.rejected[0].differences.some((item) => item.path === "default_request.sampling.height"));
});

test("evidence rejects another hardware layout or backend and reports unverified revisions", () => {
  const base = config({ engine: { compile: { enabled: false } } });
  const model = wanModel((item) => (item.reference_runs = [referenceRun(base)]));
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const otherHardware = matchEvidence({
    model,
    example: example(base, env),
    config: base,
    env,
    hardware: hardware({ id: "other", memory_gb: 80 }),
    revisions: REVISIONS,
  });
  assert.equal(otherHardware.rejected.length, 1);

  const otherEnv = matchEvidence({
    model,
    example: example(base, { FASTVIDEO_ATTENTION_BACKEND: "TORCH_SDPA" }),
    config: base,
    env: { FASTVIDEO_ATTENTION_BACKEND: "TORCH_SDPA" },
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(otherEnv.rejected.length, 1);

  const unverified = matchEvidence({
    model,
    example: example(base, env),
    config: base,
    env,
    hardware: hardware({ id: "test-80gb" }),
  });
  assert.equal(unverified.applicable.length, 0);
  assert.equal(unverified.unverified.length, 1);
});

test("a device-wide measurement can cover execution the analytical rule does not", () => {
  const base = config({ engine: { compile: { enabled: true } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const model = wanModel();
  const fitted = assessCapacity({
    model,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(fitted.status, "unknown");

  model.reference_runs = [referenceRun(base, { peak: 20 })];
  const measured = assessCapacity({
    model,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(measured.status, "fits");
  assert.equal(measured.basis, "measured");
  assert.equal(measured.compatibility, "completed");
  close(measured.peakGiB, 20);

  const allocatorOnly = structuredClone(model);
  allocatorOnly.reference_runs[0].memory = referenceRun(base, { peak: 20, scope: "allocator" }).memory;
  const allocatorResult = assessCapacity({
    model: allocatorOnly,
    example: example(base, env),
    hardware: hardware({ id: "test-80gb" }),
    revisions: REVISIONS,
  });
  assert.equal(allocatorResult.status, "unknown");
  assert.equal(allocatorResult.basis, undefined);
});

test("recommendations prefer measured fits, keep unknown explicit and never rank speed or quality", () => {
  const fittedConfig = config({ engine: { compile: { enabled: false } } });
  const env = { FASTVIDEO_ATTENTION_BACKEND: "FLASH_ATTN" };
  const fitted = wanModel((model) => {
    model.id = "fitted";
    model.examples = [example(fittedConfig, env)];
    model.reference_runs = [referenceRun(fittedConfig, { peak: 20 })];
  });
  const unknown = wanModel((model) => {
    model.id = "unknown";
    model.examples = [example(config({ engine: { compile: { enabled: true } } }), {})];
  });
  const recommendations = recommendCapacity({
    models: [unknown, fitted],
    hardware: hardware({ id: "test-80gb" }),
    task: "t2v",
    revisions: REVISIONS,
  });
  assert.equal(recommendations[0].model.id, "fitted");
  assert.equal(recommendations[0].assessment.basis, "measured");
  assert.equal(recommendations[1].model.id, "unknown");
  assert.match(recommendations[1].reasons[0], /unknown/i);
  assert.doesNotMatch(JSON.stringify(recommendations), /speed|quality|benchmark/i);
});

test("a lower-memory alternative reports concrete native edits and their new status", () => {
  const base = config({ offload: { dit_layerwise: true, text_encoder: true, vae: false } });
  const model = wanModel((item) => {
    item.examples = [example(base)];
  });
  const [candidate] = recommendCapacity({
    models: [model],
    hardware: hardware({ id: "test-24gb", memory_gb: 24 }),
    task: "t2v",
  });
  assert.equal(candidate.assessment.status, "exceeds");
  assert.ok(candidate.lower_memory.length > 0);
  for (const alternative of candidate.lower_memory) {
    assert.ok(
      alternative.changes.every(
        (change) => change.path.startsWith("default_request.sampling.") || change.path === "generator.engine.offload",
      ),
    );
    assert.equal(typeof alternative.required_gib, "number");
  }

  const request = { sampling: { num_frames: 121, height: 720, width: 1280 } };
  const [requestCandidate] = recommendCapacity({
    models: [model],
    hardware: hardware({ id: "test-24gb", memory_gb: 24 }),
    task: "t2v",
    request,
  });
  assert.equal(requestCandidate.assessment.status, "exceeds");
  assert.ok(requestCandidate.lower_memory.length > 0);
  for (const alternative of requestCandidate.lower_memory) {
    assert.ok(
      alternative.changes.every(
        (change) => change.path.startsWith("request.sampling.") || change.path === "generator.engine.offload",
      ),
    );
    assert.equal(typeof alternative.required_gib, "number");
  }
});
