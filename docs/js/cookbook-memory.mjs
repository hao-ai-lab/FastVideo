/**
 * Analytical single-GPU memory estimates for cookbook deployments.
 * Estimates are not measurements and never mark a recipe as verified.
 */

const GIB = 1024 ** 3;
const UNIT_BYTES = { GiB: GIB, GB: 1e9 };

/** NVFP4 stores two 4-bit values per byte plus a 1-byte scale for every 16 values. */
export const BYTES_PER_WEIGHT = { fp32: 4, fp16: 2, bf16: 2, fp8: 1, nvfp4: 0.5 + 1 / 16 };

export const DEFAULT_HEADROOM = { fraction: 0.1, reserveGiB: 1 };

/**
 * Wan working-memory terms. Tensor counts follow WanTransformerBlock.forward and the chunked
 * AutoencoderKLWan decode with its causal feature cache. They are uncalibrated until measured.
 */
export const WAN_RULE = {
  encodingWorkspaceGiB: 0.5,
  attentionHiddenTensors: 10,
  ffnHiddenTensors: 2,
  ffnTensors: 2,
  decodeFrameFeatures: 48,
};

const PHASES = ["encoding", "denoising", "decoding"];
const COMPONENT_PHASE = { text_encoder: "encoding", transformer: "denoising", vae: "decoding" };
const SAMPLING_FIELDS = ["num_frames", "height", "width"];

/** Unset switches default on, matching OffloadConfig in fastvideo/api/schema.py. */
const OFFLOAD_DEFAULTS = { dit_layerwise: true, text_encoder: true, vae: true, lazy_module_load: false };

/** Streaming offload holds the running block and one prefetched block. */
const STREAMED_BLOCKS = 2;

const WORKING_MEMORY = {
  wan: {
    geometry: [
      "temporal_compression",
      "spatial_compression",
      "patch_size",
      "hidden_size",
      "ffn_size",
      "vae_decoder_channels",
    ],
    estimate: wanWorkingMemory,
  },
};

/** Resolve automatic degrees as FastVideoArgs does: tensor parallelism 1, sequence parallelism across every GPU. */
export function resolveTopology(engine = {}) {
  const gpus = engine.num_gpus ?? 1;
  const { tp_size: tp = -1, sp_size: sp = -1 } = engine.parallelism ?? {};
  const tpSize = tp === -1 ? 1 : tp;
  const spSize = sp === -1 ? gpus : sp;
  return { gpus: Math.max(gpus, tpSize, spSize), tpSize, spSize };
}

function componentBytes(component, quantization) {
  const quantized = quantization ? component.quantized_parameter_count : 0;
  const perWeight = BYTES_PER_WEIGHT[component.dtype];
  return {
    quantized: quantization ? quantized * BYTES_PER_WEIGHT[quantization] : 0,
    dense: (component.parameter_count - quantized) * perWeight,
    denseBlocks: component.block_count ? (component.block_parameter_count - quantized) * perWeight : 0,
  };
}

/**
 * Weights a component holds on the GPU during one phase. A component is complete during its own phase
 * unless it streams blocks; offload only frees it during other phases. Without FSDP, whole-model DiT
 * offload leaves the transformer on the GPU, and packed quantized weights are buffers that stay resident.
 */
function residentBytes(component, phase, offload, bytes) {
  const full = bytes.quantized + bytes.dense;
  const withoutBlocks = full - bytes.denseBlocks;
  const share = component.block_count ? Math.min(1, STREAMED_BLOCKS / component.block_count) : 1;
  const streamed = withoutBlocks + bytes.denseBlocks * share;
  const active = COMPONENT_PHASE[component.role] === phase;
  if (component.role === "transformer") {
    if (offload.lazy_module_load && !active) return 0;
    if (offload.dit_layerwise) return active ? streamed : withoutBlocks;
    return full;
  }
  if (!active) return offload[component.role] || offload.lazy_module_load ? 0 : full;
  const streams = component.role === "text_encoder" && offload.text_encoder && component.streams_when_offloaded;
  return streams ? streamed : full;
}

function activationBytes(components, role) {
  const component = components.find((item) => item.role === role);
  return BYTES_PER_WEIGHT[component.activation_dtype ?? component.dtype];
}

function wanWorkingMemory(geometry, sampling, components) {
  const { temporal_compression, spatial_compression, patch_size, hidden_size, ffn_size, vae_decoder_channels } =
    geometry;
  const { num_frames, height, width } = sampling;
  const latentFrames = Math.floor((num_frames - 1) / temporal_compression) + 1;
  const tokens =
    Math.ceil(latentFrames / patch_size[0]) *
    Math.ceil(height / (spatial_compression * patch_size[1])) *
    Math.ceil(width / (spatial_compression * patch_size[2]));
  const dit = activationBytes(components, "transformer");
  const vae = activationBytes(components, "vae");
  const hidden = tokens * hidden_size * dit;
  const ffn = tokens * ffn_size * dit;
  const pixels = height * width;
  const outputFrames = (latentFrames - 1) * temporal_compression + 1;
  return {
    encoding: WAN_RULE.encodingWorkspaceGiB * GIB,
    denoising: Math.max(
      WAN_RULE.attentionHiddenTensors * hidden,
      WAN_RULE.ffnHiddenTensors * hidden + WAN_RULE.ffnTensors * ffn,
    ),
    // Decoded RGB frames are held twice while chunks are concatenated, then once more as float32.
    decoding:
      WAN_RULE.decodeFrameFeatures * vae_decoder_channels * pixels * vae +
      outputFrames * 3 * pixels * (2 * vae + BYTES_PER_WEIGHT.fp32),
  };
}

/** Return an unknown or unsupported result when the request falls outside what the estimator models. */
function checkCoverage({ engine, sampling, workload, model, hardware, quantization }) {
  const unknown = (reason) => ({ status: "unknown", reason });
  const components = model.components ?? [];
  if (hardware.type !== "discrete_gpu") return unknown("Only discrete GPUs are modeled.");
  if (!UNIT_BYTES[hardware.memory_per_device?.unit]) return unknown(`${hardware.name} has no known memory size.`);
  if (!Array.isArray(hardware.supported_dtypes)) return unknown(`${hardware.name} lists no supported weight types.`);
  if (workload !== "t2v") return unknown(`${workload} workloads are not modeled yet.`);
  if (resolveTopology(engine).gpus > 1 || engine.use_fsdp_inference)
    return unknown("Multi-GPU and FSDP layouts are not modeled yet.");
  const rule = WORKING_MEMORY[model.memory_rules];
  if (!rule) return unknown(`No working-memory rule for ${model.memory_rules}.`);
  const missing = rule.geometry.filter((field) => model.geometry?.[field] === undefined);
  if (missing.length) return unknown(`Model geometry lacks ${missing.join(", ")}.`);
  if (!SAMPLING_FIELDS.every((field) => Number.isInteger(sampling[field]) && sampling[field] > 0))
    return unknown("Frames, height and width must be set.");
  const unmodeled = components.find((component) => !COMPONENT_PHASE[component.role]);
  if (unmodeled) return unknown(`The ${unmodeled.role} component is not modeled yet.`);
  if (!["transformer", "vae"].every((role) => components.some((component) => component.role === role)))
    return unknown("The model needs a transformer and a VAE.");
  const unsized = components.find(
    (component) =>
      !BYTES_PER_WEIGHT[component.dtype] || !BYTES_PER_WEIGHT[component.activation_dtype ?? component.dtype],
  );
  if (unsized) return unknown(`The ${unsized.role} precision is not modeled.`);
  if (engine.quantization?.text_encoder_quant) return unknown("Text-encoder quantization is not modeled yet.");
  if (quantization && !BYTES_PER_WEIGHT[quantization]) return unknown(`${quantization} weights are not modeled.`);
  const dtypes = new Set([...components.map((component) => component.dtype), quantization].filter(Boolean));
  const unsupported = [...dtypes].find((dtype) => !hardware.supported_dtypes.includes(dtype));
  if (unsupported)
    return { status: "unsupported", reason: `${hardware.name} does not support ${unsupported} weights.` };
  if (!quantization) return null;
  if (engine.quantization.transformer_retain_original_weights)
    return unknown("Retained original weights are not modeled.");
  const transformer = components.find((component) => component.role === "transformer");
  if (!Number.isInteger(transformer.quantized_parameter_count))
    return unknown(`The transformer has no ${quantization} parameter count.`);
  return null;
}

/**
 * Estimate one request's peak memory on one device from a resolved serving configuration.
 * Each phase adds the weights resident on the GPU to that phase's working memory; the peak phase plus
 * headroom is compared with the device's memory. Uncovered requests return `unknown` with a reason.
 */
export function estimateMemory({ config, workload, model, hardware, headroom = DEFAULT_HEADROOM }) {
  const engine = config.generator?.engine ?? {};
  const sampling = config.default_request?.sampling ?? {};
  const quantization = engine.quantization?.transformer_quant?.toLowerCase() ?? null;
  const gap = checkCoverage({ engine, sampling, workload, model, hardware, quantization });
  if (gap) return gap;

  const offload = { ...OFFLOAD_DEFAULTS, ...engine.offload };
  const working = WORKING_MEMORY[model.memory_rules].estimate(model.geometry, sampling, model.components);
  const bytes = model.components.map((component) =>
    componentBytes(component, component.role === "transformer" ? quantization : null),
  );
  const phases = Object.fromEntries(
    PHASES.map((phase) => {
      const weights = model.components.reduce(
        (sum, component, index) => sum + residentBytes(component, phase, offload, bytes[index]),
        0,
      );
      return [
        phase,
        { weightsGiB: weights / GIB, workingGiB: working[phase] / GIB, totalGiB: (weights + working[phase]) / GIB },
      ];
    }),
  );
  const peakPhase = PHASES.reduce((peak, phase) => (phases[phase].totalGiB > phases[peak].totalGiB ? phase : peak));
  const peakGiB = phases[peakPhase].totalGiB;
  const requiredGiB = peakGiB * (1 + headroom.fraction) + headroom.reserveGiB;
  const { value, unit } = hardware.memory_per_device;
  const capacityGiB = (value * UNIT_BYTES[unit]) / GIB;
  return {
    status: requiredGiB <= capacityGiB ? "fits" : "exceeds",
    basis: "analytical",
    phases,
    peakPhase,
    peakGiB,
    requiredGiB,
    capacityGiB,
    notes: config.generator?.pipeline?.vae_tiling
      ? ["Decode memory assumes untiled VAE decoding; tiling lowers it."]
      : [],
  };
}
