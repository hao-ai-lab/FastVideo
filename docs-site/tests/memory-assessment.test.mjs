import assert from "node:assert/strict";
import test from "node:test";
import { assessRecipe, assessWeightsRecipe, capacityHardware, weightsFor } from "../lib/memory-assessment.ts";
import { buildCapacityCatalog } from "../../docs/js/build-cookbook-capacity.mjs";
import { assessCapacity } from "../../docs/js/cookbook-memory.mjs";

const hardware = { backend: "cuda", memory_gb: 80, memory_unit: "GiB", memory_type: "dedicated" };
const weights = weightsFor("fasth3-8step");
const config = {
  generator: {
    engine: { num_gpus: 4, offload: { text_encoder: true, vae: true } },
    pipeline: { experimental: {} },
  },
};

test("checkpoint data never certifies runtime fit or subtracts CPU-offloaded weights", () => {
  for (const unified of [false, true]) {
    const target = { ...hardware, memory_type: unified ? "unified" : "dedicated", usable_budget_gb: 60 };
    const baseline = assessWeightsRecipe({ weights, config, hardware: target });
    const edited = structuredClone(config);
    edited.generator.engine.offload.text_encoder = false;
    edited.generator.engine.num_gpus = 1;
    const result = assessWeightsRecipe({ weights, config: edited, hardware: target });
    assert.equal(result.status, "unknown");
    assert.ok(Math.abs(result.capacityGiB - 60) < 1e-9);
    assert.deepEqual(result.parts, weights.parts);
    assert.equal(result.checkpointGiB, baseline.checkpointGiB);
    assert.equal(result.requiredGiB, undefined);
  }
});

test("a component swap preserves other known sizes and identifies missing coverage", () => {
  for (const swap of ["taeh3", "nvfp4-decoder", "nvfp4-transformer", "encoder"]) {
    const edited = structuredClone(config);
    if (swap === "taeh3") edited.generator.pipeline.experimental.video_decode_backend = "taeh3";
    if (swap === "nvfp4-decoder") edited.generator.pipeline.experimental.h3_vae_nvfp4_decoder = "decoder.safetensors";
    if (swap === "nvfp4-transformer") edited.generator.engine.quantization = { transformer_quant: "NVFP4" };
    if (swap === "encoder") edited.generator.pipeline.components = { text_encoder_weights: "replacement" };
    const result = assessWeightsRecipe({ weights, config: edited, hardware });
    const missingRole =
      swap.includes("decoder") || swap === "taeh3" ? "vae" : swap === "encoder" ? "text_encoder" : "transformer";
    assert.equal(result.status, "unknown");
    assert.match(result.reason, /Partial checkpoint data/);
    assert.equal(result.parts[missingRole], undefined);
    assert.equal(result.parts.audio_vae, weights.parts.audio_vae);
    assert.equal(
      result.checkpointGiB,
      Object.values(result.parts).reduce((sum, size) => sum + size, 0),
    );
  }
});

test("host RAM is rejected and unified memory gets one budget", () => {
  assert.equal(
    assessWeightsRecipe({ weights, config, hardware: { ...hardware, memory_type: "system" } }).parts,
    undefined,
  );
  const result = assessWeightsRecipe({ weights, config, hardware: { ...hardware, memory_type: "unified" } });
  assert.equal(result.capacityGiB, 60);
  assert.ok(result.notes.some((note) => note.includes("75%")));
});

test("the UI adapter preserves GB, GiB, usable budgets and host memory through the analytical engine", () => {
  const model = buildCapacityCatalog().models.find((model) => model.id === "fastwan21");
  const example = { workload: "t2v", env: { FASTVIDEO_ATTENTION_BACKEND: "TORCH_SDPA" } };
  const dense = structuredClone(model.examples[0].base_config);
  dense.generator.engine.compile.enabled = false;
  dense.generator.pipeline.experimental = {};
  for (const unit of ["GB", "GiB"]) {
    const target = { ...hardware, memory_unit: unit, usable_budget_gb: 70, host_ram_gb: 128 };
    const result = assessRecipe({ model, example, config: dense, hardware: target });
    const raw = assessCapacity({ model, example, config: dense, hardware: capacityHardware(target) });
    assert.equal(result.status, raw.status);
    assert.notEqual(result.status, "unknown");
    assert.equal(result.requiredGiB, raw.requiredGiB);
    assert.equal(result.capacityGiB, unit === "GiB" ? 70 : 70e9 / 1024 ** 3);
  }
});
