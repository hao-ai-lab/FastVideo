import assert from "node:assert/strict";
import test from "node:test";

import { buildCatalogs } from "../js/build-cookbook-config.mjs";
import { resolveConfig } from "../js/cookbook-config.mjs";
import {
  BYTES_PER_WEIGHT,
  DEFAULT_HEADROOM,
  WAN_RULE,
  estimateMemory,
  resolveTopology,
} from "../js/cookbook-memory.mjs";

const GIB = 1024 ** 3;
const NO_OFFLOAD = { dit_layerwise: false, text_encoder: false, vae: false };

const DIT = 1_418_996_800;
const DIT_BLOCKS = 1_393_221_120;
const TEXT = 5_680_910_336;
const TEXT_BLOCKS = 4_630_757_376;
const VAE = 73_296_659;

/** Wan2.1 T2V 1.3B; block totals follow WanTransformerBlock and the UMT5-XXL encoder layer shapes. */
function wanModel() {
  return {
    memory_rules: "wan",
    geometry: {
      temporal_compression: 4,
      spatial_compression: 8,
      patch_size: [1, 2, 2],
      hidden_size: 1536,
      ffn_size: 8960,
      vae_decoder_channels: 96,
    },
    components: [
      {
        role: "text_encoder",
        parameter_count: TEXT,
        dtype: "fp32",
        block_count: 24,
        block_parameter_count: TEXT_BLOCKS,
        streams_when_offloaded: true,
      },
      { role: "transformer", parameter_count: DIT, dtype: "bf16", block_count: 30, block_parameter_count: DIT_BLOCKS },
      { role: "vae", parameter_count: VAE, dtype: "fp32", activation_dtype: "bf16" },
    ],
  };
}

function gpu(value, { unit = "GiB", dtypes = ["fp32", "fp16", "bf16"] } = {}) {
  return { name: "Test GPU", type: "discrete_gpu", memory_per_device: { value, unit }, supported_dtypes: dtypes };
}

function config({ offload = NO_OFFLOAD, engine = {}, pipeline = {}, sampling } = {}) {
  return {
    generator: { engine: { num_gpus: 1, offload, ...engine }, pipeline },
    default_request: { sampling: sampling ?? { num_frames: 81, height: 480, width: 832 } },
  };
}

function estimate(options = {}) {
  return estimateMemory({ config: config(), workload: "t2v", model: wanModel(), hardware: gpu(80), ...options });
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

test("automatic parallel degrees resolve like FastVideoArgs", () => {
  assert.deepEqual(resolveTopology(), { gpus: 1, tpSize: 1, spSize: 1 });
  assert.deepEqual(resolveTopology({ num_gpus: 4, parallelism: { tp_size: -1, sp_size: -1 } }), {
    gpus: 4,
    tpSize: 1,
    spSize: 4,
  });
  assert.equal(resolveTopology({ num_gpus: 1, parallelism: { tp_size: 2, sp_size: 1 } }).gpus, 2);
});

test("component weights match the Wan 1.3B sizes in the VRAM design proposal", () => {
  assert.equal(gib(DIT, "bf16").toFixed(2), "2.64");
  assert.equal(gib(TEXT, "bf16").toFixed(2), "10.58");
  assert.equal(gib(VAE, "fp32").toFixed(2), "0.27");
});

test("without offload every component stays resident in every phase", () => {
  const total = gib(TEXT, "fp32") + gib(DIT, "bf16") + gib(VAE, "fp32");
  assertWeights(estimate(), { encoding: total, denoising: total, decoding: total });
});

test("an offloaded component occupies the GPU only during its own phase", () => {
  const model = wanModel();
  delete model.components[0].streams_when_offloaded;
  const result = estimate({
    model,
    config: config({ offload: { dit_layerwise: false, text_encoder: true, vae: true } }),
  });
  assertWeights(result, {
    encoding: gib(TEXT, "fp32") + gib(DIT, "bf16"),
    denoising: gib(DIT, "bf16"),
    decoding: gib(DIT, "bf16") + gib(VAE, "fp32"),
  });
});

test("streaming offload keeps the weights outside the blocks plus two blocks", () => {
  const result = estimate({ config: config({ offload: { dit_layerwise: true, text_encoder: true, vae: true } }) });
  const ditOutsideBlocks = gib(DIT - DIT_BLOCKS, "bf16");
  assertWeights(result, {
    encoding: streamed(TEXT, TEXT_BLOCKS, 24, "fp32") + ditOutsideBlocks,
    denoising: streamed(DIT, DIT_BLOCKS, 30, "bf16"),
    decoding: ditOutsideBlocks + gib(VAE, "fp32"),
  });
});

test("unset offload switches default on, so layerwise DiT offload replaces whole-model DiT offload", () => {
  const result = estimate({ config: config({ offload: { dit: false, vae: false } }) });
  close(result.phases.denoising.weightsGiB, streamed(DIT, DIT_BLOCKS, 30, "bf16") + gib(VAE, "fp32"));
});

test("whole-model DiT offload without layerwise streaming keeps the transformer resident", () => {
  const result = estimate({ config: config({ offload: { ...NO_OFFLOAD, dit: true, text_encoder: true, vae: true } }) });
  close(result.phases.decoding.weightsGiB, gib(DIT, "bf16") + gib(VAE, "fp32"));
});

test("lazy loading frees every component outside its own phase", () => {
  const result = estimate({ config: config({ offload: { ...NO_OFFLOAD, lazy_module_load: true } }) });
  assertWeights(result, { encoding: gib(TEXT, "fp32"), denoising: gib(DIT, "bf16"), decoding: gib(VAE, "fp32") });
});

test("Wan working memory scales with tokens, hidden size, FFN size and decoded pixels", () => {
  const result = estimate();
  const tokens = 21 * 30 * 52;
  const hidden = tokens * 1536 * 2;
  const ffn = tokens * 8960 * 2;
  const pixels = 480 * 832;
  assert.equal((hidden / GIB).toFixed(3), "0.094");
  close(result.phases.encoding.workingGiB, WAN_RULE.encodingWorkspaceGiB);
  close(
    result.phases.denoising.workingGiB,
    Math.max(WAN_RULE.attentionHiddenTensors * hidden, WAN_RULE.ffnHiddenTensors * hidden + WAN_RULE.ffnTensors * ffn) /
      GIB,
  );
  close(
    result.phases.decoding.workingGiB,
    (WAN_RULE.decodeFrameFeatures * 96 * pixels * 2 + 81 * 3 * pixels * (2 * 2 + 4)) / GIB,
  );
  const longer = estimate({ config: config({ sampling: { num_frames: 161, height: 480, width: 832 } }) });
  assert.ok(longer.phases.denoising.workingGiB > result.phases.denoising.workingGiB);
});

test("the peak phase plus headroom decides fit against device memory", () => {
  const result = estimate({ config: config({ offload: { dit_layerwise: true, text_encoder: true, vae: true } }) });
  assert.equal(result.status, "fits");
  assert.equal(result.basis, "analytical");
  assert.equal(result.peakPhase, "encoding");
  close(result.peakGiB, result.phases.encoding.totalGiB);
  close(result.requiredGiB, result.peakGiB * (1 + DEFAULT_HEADROOM.fraction) + DEFAULT_HEADROOM.reserveGiB);
  const offload = { dit_layerwise: true, text_encoder: true, vae: true };
  assert.equal(estimate({ config: config({ offload }), hardware: gpu(result.requiredGiB) }).status, "fits");
  assert.equal(estimate({ config: config({ offload }), hardware: gpu(result.requiredGiB - 0.01) }).status, "exceeds");
  close(
    estimate({ config: config({ offload }), headroom: { fraction: 0, reserveGiB: 0 } }).requiredGiB,
    result.peakGiB,
  );
});

test("device memory honors decimal and binary units", () => {
  close(estimate({ hardware: gpu(180, { unit: "GB" }) }).capacityGiB, 180e9 / GIB);
  close(estimate({ hardware: gpu(80) }).capacityGiB, 80);
});

test("VAE tiling adds a note because the decode estimate assumes untiled decoding", () => {
  assert.deepEqual(estimate().notes, []);
  assert.equal(estimate({ config: config({ pipeline: { vae_tiling: true } }) }).notes.length, 1);
});

test("requests outside the modeled coverage are unknown with a reason", () => {
  const withModel = (change) => {
    const model = wanModel();
    change(model);
    return { model };
  };
  const cases = {
    "multiple GPUs": { config: config({ engine: { num_gpus: 2 } }) },
    "FSDP inference": { config: config({ engine: { use_fsdp_inference: true } }) },
    "unified memory": { hardware: { ...gpu(128), type: "unified_memory" } },
    "unknown memory unit": { hardware: gpu(80, { unit: "TB" }) },
    "no supported weight types": { hardware: { ...gpu(80), supported_dtypes: undefined } },
    "image to video": { workload: "i2v" },
    "missing request size": { config: config({ sampling: {} }) },
    "unknown working-memory rule": withModel((model) => (model.memory_rules = "ltx2")),
    "missing geometry": withModel((model) => delete model.geometry.ffn_size),
    "unmodeled component": withModel((model) =>
      model.components.push({ role: "image_encoder", parameter_count: 1, dtype: "fp16" }),
    ),
    "missing VAE": withModel((model) => model.components.pop()),
    "unknown precision": withModel((model) => (model.components[1].dtype = "int4")),
    "text-encoder quantization": { config: config({ engine: { quantization: { text_encoder_quant: "fp8" } } }) },
  };
  for (const [name, options] of Object.entries(cases)) {
    const result = estimate(options);
    assert.equal(result.status, "unknown", name);
    assert.ok(result.reason.length > 0, name);
  }
});

test("NVFP4 weights need device support and a quantized parameter count", () => {
  const quantized = 1_300_000_000;
  const nvfp4 = config({
    offload: { dit_layerwise: true, text_encoder: true, vae: true },
    engine: { quantization: { transformer_quant: "NVFP4" } },
  });
  const blackwell = gpu(180, { unit: "GB", dtypes: ["fp32", "fp16", "bf16", "fp8", "nvfp4"] });

  const unsupported = estimate({ config: nvfp4 });
  assert.equal(unsupported.status, "unsupported");
  assert.match(unsupported.reason, /nvfp4/);
  assert.equal(estimate({ config: nvfp4, hardware: blackwell }).status, "unknown");

  const model = wanModel();
  model.components[1].quantized_parameter_count = quantized;
  const result = estimate({ config: nvfp4, hardware: blackwell, model });
  const packed = (quantized * 0.5625) / GIB;
  const outsideBlocks = gib(DIT - DIT_BLOCKS, "bf16");
  const denseBlocks = gib(DIT_BLOCKS - quantized, "bf16");
  close(result.phases.denoising.weightsGiB, packed + outsideBlocks + (denseBlocks * 2) / 30);
  close(result.phases.decoding.weightsGiB, packed + outsideBlocks + gib(VAE, "fp32"));
});

test("the FastWan cookbook baseline produces a single-GPU estimate without changing its configuration", () => {
  const catalog = buildCatalogs().find((item) => item.id === "fastwan21/cuda-rest");
  const { config: resolved } = resolveConfig(catalog);
  const before = JSON.stringify(resolved);
  const result = estimateMemory({ config: resolved, workload: catalog.workload, model: wanModel(), hardware: gpu(24) });
  assert.equal(JSON.stringify(resolved), before);
  assert.equal(result.status, "fits");
  close(result.phases.denoising.weightsGiB, streamed(DIT, DIT_BLOCKS, 30, "bf16") + gib(VAE, "fp32"));
});
