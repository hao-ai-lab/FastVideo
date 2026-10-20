/**
 * Analytical memory estimates, evidence applicability and candidate recommendations for cookbook
 * deployments. Analytical results are not measurements and never mark a recipe as verified.
 *
 * Inputs use the static capacity catalog exported by build-cookbook-capacity.mjs:
 * model.components is a mapping by role, hardware uses memory.nominal/memory.type (the
 * normalizer also accepts legacy memory_gb/memory_type for custom callers), and examples
 * preserve the native baseline. Execution settings are resolved from the serving configuration.
 */

const GIB = 1024 ** 3;
const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const isCount = (value) => Number.isInteger(value) && value > 0;

/** NVFP4 stores two 4-bit values per byte plus a 1-byte scale for every 16 values. */
export const BYTES_PER_WEIGHT = { fp32: 4, fp16: 2, bf16: 2, fp8: 1, nvfp4: 0.5 + 1 / 16 };

export const DEFAULT_HEADROOM = { fraction: 0.1, reserveGiB: 1 };
/** Conservative application share for unified memory when the caller gives no budget. */
export const DEFAULT_UNIFIED_BUDGET_FRACTION = 0.75;

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

const PHASES = ["startup", "encoding", "denoising", "decoding"];
const COMPONENT_PHASE = {
  text_encoder: "encoding",
  text_encoder_2: "encoding",
  image_encoder: "encoding",
  image_encoder_2: "encoding",
  transformer: "denoising",
  transformer_2: "denoising",
  vae: "decoding",
  audio_vae: "decoding",
};
const COMPONENT_KIND = {
  transformer: "transformer",
  transformer_2: "transformer",
  text_encoder: "encoder",
  text_encoder_2: "encoder",
  image_encoder: "encoder",
  image_encoder_2: "encoder",
  vae: "vae",
  audio_vae: "vae",
};
const SAMPLING_FIELDS = ["num_frames", "height", "width"];

/** Unset switches default on, matching OffloadConfig in fastvideo/api/schema.py. */
const OFFLOAD_DEFAULTS = {
  dit: true,
  dit_layerwise: true,
  text_encoder: true,
  image_encoder: true,
  vae: true,
};

/** Streaming offload holds the running block and one prefetched block. */
const STREAMED_BLOCKS = 2;

/** Native pipeline precision defaults for a modeled family; explicit config values override them. */
export const PRECISION_DEFAULTS = {
  wan: {
    transformer: "bf16",
    text_encoder: ["fp32"],
    image_encoder: ["fp32"],
    vae: "fp32",
    vae_decode: "bf16",
    vae_tiling: false,
  },
};

const DENSE_ATTENTION_BACKENDS = new Set(["FLASH_ATTN", "TORCH_SDPA", "SAGE_ATTN", "SAGE_ATTN_THREE"]);

/** Settings that do not change a request's peak memory; other experimental keys do. */
const HARMLESS_EXPERIMENTAL = new Set([
  "flow_shift",
  "flow_shift_sr",
  "dmd_denoising_steps",
  "num_inference_steps",
  "seed",
  "fps",
]);

const MEMORY_ENV_KEYS = new Set([
  "FASTVIDEO_ATTENTION_BACKEND",
  "FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS",
  "PYTORCH_CUDA_ALLOC_CONF",
]);
const MEMORY_ENV_PREFIXES = [
  "FASTVIDEO_VSA_",
  "FASTVIDEO_H3_",
  "FASTVIDEO_LAYERWISE_",
  "FASTVIDEO_NVFP4",
  "FASTVIDEO_FP8",
  "FASTVIDEO_COMPILE",
  "FASTVIDEO_OFFLOAD",
  "FASTVIDEO_TORCH",
];

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

function normalizeComponents(model) {
  if (!isObject(model.components)) return null;
  const components = [];
  for (const [role, component] of Object.entries(model.components)) {
    if (!isObject(component) || !COMPONENT_PHASE[role]) return null;
    if (component.kind !== COMPONENT_KIND[role]) return null;
    components.push({ role, ...component });
  }
  return components.length ? components : null;
}

function geometryFor(model, role) {
  const geometry = model.components?.[role]?.geometry;
  return isObject(geometry) ? geometry : null;
}

function asPrecisionList(value, fallback) {
  if (typeof value === "string" && value) return [value];
  if (Array.isArray(value) && value.length && value.every((item) => typeof item === "string")) return value;
  return fallback;
}

function experimentalPrecision(experimental, name, fallback) {
  const value = experimental[name];
  return typeof value === "string" && value ? value : fallback;
}

/**
 * Resolve the native execution settings that change memory. Values come from the serving config,
 * then from family defaults keyed by memory_rules. The component catalog never fixes precision.
 */
export function resolveExecution({ model, config, env = {} }) {
  const family = model.memory_rules;
  const defaults = PRECISION_DEFAULTS[family] ?? {};
  const engine = config.generator?.engine ?? {};
  const pipeline = config.generator?.pipeline ?? {};
  const experimental = isObject(pipeline.experimental) ? pipeline.experimental : {};
  const textEncoderPrecisions = asPrecisionList(experimental.text_encoder_precisions, defaults.text_encoder ?? []);
  const imageEncoderPrecisions = asPrecisionList(
    experimental.image_encoder_precisions,
    experimental.image_encoder_precision ? [experimental.image_encoder_precision] : (defaults.image_encoder ?? []),
  );
  const vae = experimentalPrecision(experimental, "vae_precision", defaults.vae);
  const vaeDecode = experimentalPrecision(experimental, "vae_decode_precision", defaults.vae_decode ?? vae);
  const transformer = experimentalPrecision(experimental, "dit_precision", defaults.transformer);
  const offload = { ...OFFLOAD_DEFAULTS, ...engine.offload };
  const quantization = isObject(engine.quantization) ? engine.quantization : {};
  const attentionBackend = typeof env.FASTVIDEO_ATTENTION_BACKEND === "string" ? env.FASTVIDEO_ATTENTION_BACKEND : null;
  const residentBlocks = Number.parseInt(env.FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS ?? "0", 10);
  const layerwiseResidentBlocks = Number.isInteger(residentBlocks) && residentBlocks >= 0 ? residentBlocks : 0;
  const vsa =
    (attentionBackend && attentionBackend.includes("SPARSE")) ||
    Object.keys(experimental).some((key) => key.startsWith("VSA_")) ||
    Object.keys(env).some((key) => key.startsWith("FASTVIDEO_VSA_"));
  return {
    family,
    workload: pipeline.workload_type,
    num_gpus: engine.num_gpus ?? 1,
    topology: resolveTopology(engine),
    use_fsdp: engine.use_fsdp_inference === true,
    offload,
    lazy_module_load: offload.lazy_module_load === true,
    compile:
      engine.compile?.enabled === true ||
      engine.compile?.text_encoder_enabled === true ||
      engine.compile?.vae_enabled === true,
    quantization,
    attentionBackend,
    layerwise_resident_blocks: layerwiseResidentBlocks,
    vae_feature_cache: experimental["vae_config.use_feature_cache"] !== false,
    vsa,
    vae_tiling: typeof pipeline.vae_tiling === "boolean" ? pipeline.vae_tiling : (defaults.vae_tiling ?? false),
    vae_sp: pipeline.vae_sp === true,
    precision: {
      transformer,
      text_encoder: textEncoderPrecisions,
      image_encoder: imageEncoderPrecisions,
      vae,
      vae_decode: vaeDecode,
    },
    experimental,
  };
}

function componentPrecision(component, execution) {
  const { role } = component;
  if (role === "transformer" || role === "transformer_2") return execution.precision.transformer;
  if (role === "vae" || role === "audio_vae") return execution.precision.vae;
  if (role === "image_encoder" || role === "image_encoder_2") {
    const index = role === "image_encoder_2" ? 1 : 0;
    return execution.precision.image_encoder[index];
  }
  const index = role === "text_encoder_2" ? 1 : 0;
  return execution.precision.text_encoder[index];
}

function componentActivationPrecision(component, execution) {
  if (component.role === "vae" || component.role === "audio_vae") return execution.precision.vae_decode;
  return componentPrecision(component, execution);
}

function componentBytes(component, execution) {
  const storage = componentPrecision(component, execution);
  const perWeight = BYTES_PER_WEIGHT[storage];
  return { storage, perWeight, full: component.parameter_count * perWeight };
}

/** Weights resident during a request phase after offload and layerwise streaming. */
function offloadFlag(role) {
  if (role.startsWith("text_encoder")) return "text_encoder";
  if (role.startsWith("image_encoder")) return "image_encoder";
  return role;
}

function residentBytes(component, phase, execution, bytes) {
  const full = bytes.full;
  if (component.role === "transformer" || component.role === "transformer_2") {
    const blocks = component.block_parameter_count * bytes.perWeight;
    const outside = full - blocks;
    if (execution.offload.dit_layerwise) {
      const resident = execution.layerwise_resident_blocks ?? 0;
      const share = component.block_count
        ? Math.min(1, (phase === "denoising" ? resident + STREAMED_BLOCKS : resident) / component.block_count)
        : 1;
      return outside + blocks * share;
    }
    return full;
  }
  const offloaded = execution.offload[offloadFlag(component.role)];
  if (COMPONENT_PHASE[component.role] !== phase) return offloaded ? 0 : full;
  return full;
}

/** Load-time residency: text/image encoders and the VAE rest on CPU when offloaded, the DiT loads on GPU. */
function startupBytes(component, execution, bytes) {
  if (component.role === "transformer" || component.role === "transformer_2") return bytes.full;
  return execution.offload[offloadFlag(component.role)] ? 0 : bytes.full;
}

function activationBytes(component, execution) {
  return BYTES_PER_WEIGHT[componentActivationPrecision(component, execution)];
}

function wanWorkingMemory({ transformerGeometry, vaeGeometry, vaeComponent, sampling, components, execution }) {
  const { temporal_compression, spatial_compression, vae_decoder_channels, output_channels } = vaeGeometry;
  const { hidden_size, ffn_size, patch_size } = transformerGeometry;
  const { num_frames, height, width } = sampling;
  const latentFrames = Math.floor((num_frames - 1) / temporal_compression) + 1;
  const tokens =
    Math.ceil(latentFrames / patch_size[0]) *
    Math.ceil(height / (spatial_compression * patch_size[1])) *
    Math.ceil(width / (spatial_compression * patch_size[2]));
  const transformer = components.find((component) => component.role === "transformer");
  const vae = components.find((component) => component.role === "vae");
  const dit = activationBytes(transformer, execution);
  const vaeActivation = activationBytes(vae, execution);
  const hidden = tokens * hidden_size * dit;
  const ffn = tokens * ffn_size * dit;
  const pixels = height * width;
  const outputFrames = (latentFrames - 1) * temporal_compression + 1;
  const decodeMultiplier = Number.isFinite(vaeComponent?.decode_feature_multiplier)
    ? vaeComponent.decode_feature_multiplier
    : WAN_RULE.decodeFrameFeatures;
  return {
    startup: 0,
    encoding: WAN_RULE.encodingWorkspaceGiB * GIB,
    denoising: Math.max(
      WAN_RULE.attentionHiddenTensors * hidden,
      WAN_RULE.ffnHiddenTensors * hidden + WAN_RULE.ffnTensors * ffn,
    ),
    decoding:
      decodeMultiplier * vae_decoder_channels * pixels * vaeActivation +
      outputFrames * (output_channels ?? 3) * pixels * (2 * vaeActivation + BYTES_PER_WEIGHT.fp32),
  };
}

function quantityGiB(quantity) {
  if (!isObject(quantity) || !Number.isFinite(quantity.value) || !["GB", "GiB"].includes(quantity.unit)) return null;
  return (quantity.value * (quantity.unit === "GiB" ? 1.073741824 : 1) * 1e9) / GIB;
}

/**
 * Normalize catalog or custom hardware. Catalog capacities are nominal; a dedicated
 * device uses its nominal (or selected partition) capacity, while a unified device
 * uses an explicit application budget or a labeled 75% default and never exposes
 * system memory as dedicated VRAM.
 */
export function normalizeHardware(hardware, { memoryBudgetGiB, allocationMemoryGb, partitionId } = {}) {
  if (!isObject(hardware)) return null;
  const memory = isObject(hardware.memory) ? hardware.memory : null;
  const memoryType = memory?.type ?? hardware.memory_type;
  if (!["dedicated", "unified"].includes(memoryType)) return null;
  const nominalGiB = memory?.nominal
    ? quantityGiB(memory.nominal)
    : Number.isFinite(hardware.memory_gb)
      ? (hardware.memory_gb * 1e9) / GIB
      : null;
  if (!(nominalGiB > 0)) return null;
  let allocationGiB = Number.isFinite(allocationMemoryGb) ? (allocationMemoryGb * 1e9) / GIB : null;
  let partition = null;
  if (partitionId !== undefined) {
    partition = (hardware.partitions ?? []).find((item) => item.id === partitionId);
    if (!partition) return null;
    allocationGiB =
      quantityGiB(partition.memory) ??
      (Number.isFinite(partition.memory_gb) ? (partition.memory_gb * 1e9) / GIB : null);
    if (!(allocationGiB > 0)) return null;
  }
  const catalogUsableGiB = memory?.usable
    ? quantityGiB(memory.usable)
    : Number.isFinite(hardware.usable_memory_gb)
      ? (hardware.usable_memory_gb * 1e9) / GIB
      : null;
  const requestedBudget = Number.isFinite(memoryBudgetGiB) ? memoryBudgetGiB : null;
  const catalogBudget = requestedBudget ?? catalogUsableGiB;
  const assumedBudget =
    memoryType === "unified" && catalogBudget === null ? nominalGiB * DEFAULT_UNIFIED_BUDGET_FRACTION : null;
  const budgetGiB = catalogBudget ?? assumedBudget;
  if (budgetGiB !== null && !(budgetGiB > 0)) return null;
  if (budgetGiB !== null && budgetGiB > nominalGiB) return null;
  if (allocationGiB !== null && allocationGiB > nominalGiB) return null;
  const hostGiB =
    quantityGiB(hardware.host_memory) ??
    (Number.isFinite(hardware.host_memory_gb) ? (hardware.host_memory_gb * 1e9) / GIB : null);
  const capacity = Math.min(allocationGiB ?? nominalGiB, budgetGiB ?? nominalGiB);
  return {
    ...hardware,
    id: hardware.id ?? (hardware.custom ? "custom" : "unknown"),
    name: hardware.name ?? "Custom device",
    memoryType,
    nominalGiB,
    catalogUsableGiB,
    budgetGiB,
    budgetSource:
      requestedBudget !== null
        ? "user"
        : catalogUsableGiB !== null
          ? "catalog_policy"
          : assumedBudget !== null
            ? "assumed_fraction"
            : "nominal",
    budgetAssumed: assumedBudget !== null,
    allocationGiB,
    partition,
    capacityGiB: capacity,
    hostGiB,
    backend: hardware.backend ?? "unknown",
    supported_dtypes: hardware.supported_dtypes,
    custom: hardware.custom === true,
  };
}

function applyUnifiedMemoryPolicy(execution, device) {
  if (device?.memoryType !== "unified") return execution;
  return {
    ...execution,
    offload: {
      ...execution.offload,
      dit: false,
      dit_layerwise: false,
      text_encoder: false,
      image_encoder: false,
      vae: false,
    },
  };
}

function checkCoverage({ workload, model, device, execution, sampling, components }) {
  const unknown = (reason) => ({ status: "unknown", reason });
  if (!device) return unknown("Select a catalogued hardware variant or enter a custom device with capacity.");
  if (device.backend !== "cuda") return unknown(`${device.name} is not a CUDA device.`);
  if (device.gpu_count > 1) {
    return unknown("Multi-GPU hardware records are not modeled as one memory pool.");
  }
  if (!(device.capacityGiB > 0)) return unknown(`${device.name} has no known memory size.`);
  if (workload !== "t2v") return unknown(`${workload} workloads are not modeled yet.`);
  if (execution.topology.gpus > 1 || execution.topology.tpSize > 1 || execution.topology.spSize > 1) {
    return unknown("Multi-GPU layouts are not modeled as one memory pool yet.");
  }
  if (execution.use_fsdp) return unknown("FSDP inference sharding is not modeled yet.");
  if (execution.lazy_module_load) return unknown("Lazy component loading is not modeled yet.");
  if (execution.vsa) return unknown("VSA workspace is not modeled; a successful run alone is not a fit.");
  if (execution.compile) return unknown("torch.compile buffers are not modeled yet.");
  if (execution.vae_sp) return unknown("Parallel VAE decode is not modeled yet.");
  if (execution.vae_feature_cache === false) {
    return unknown("VAE feature caching is disabled; the full-clip decode path is not modeled.");
  }
  if (execution.quantization.transformer_quant) {
    return unknown("Transformer quantization needs an explicit quantized parameter count.");
  }
  if (execution.quantization.transformer_retain_original_weights) {
    return unknown("Retained original transformer weights are not modeled.");
  }
  if (execution.quantization.text_encoder_quant) return unknown("Text-encoder quantization is not modeled yet.");
  if (execution.quantization.layer_profile) return unknown("Per-layer quantization profiles are not modeled.");
  if (execution.attentionBackend && !DENSE_ATTENTION_BACKENDS.has(execution.attentionBackend)) {
    return unknown(`Attention backend ${execution.attentionBackend} has no memory rule.`);
  }
  const rule = WORKING_MEMORY[model.memory_rules];
  if (!rule) return unknown(`No working-memory rule for ${model.memory_rules}.`);
  const transformerGeometry = geometryFor(model, "transformer");
  const vaeGeometry = geometryFor(model, "vae");
  const missing = rule.geometry.filter(
    (field) => transformerGeometry?.[field] === undefined && vaeGeometry?.[field] === undefined,
  );
  if (missing.length) return unknown(`Model geometry lacks ${missing.join(", ")}.`);
  if (!SAMPLING_FIELDS.every((field) => isCount(sampling[field]))) {
    return unknown("Frames, height and width must be positive integers.");
  }
  if ((sampling.num_videos_per_prompt ?? 1) !== 1) return unknown("Only one video per prompt is modeled.");
  if (sampling.batch_cfg === true) return unknown("Batch CFG is not modeled.");
  if (!components) return unknown("Components must be a mapping by role.");
  const unmodeled = components.find((component) => !COMPONENT_PHASE[component.role]);
  if (unmodeled) return unknown(`The ${unmodeled.role} component is not modeled yet.`);
  for (const role of ["transformer", "vae", "text_encoder"]) {
    if (!components.some((component) => component.role === role)) {
      return unknown(`The model needs a ${role} component for this rule.`);
    }
  }
  const uncounted = components.find(
    (component) =>
      !isCount(component.parameter_count) ||
      (component.block_count !== undefined &&
        !(isCount(component.block_count) && isCount(component.block_parameter_count))),
  );
  if (uncounted) return unknown(`The ${uncounted.role} parameter counts are incomplete.`);
  if (execution.offload.dit_layerwise) {
    const missingBlocks = components.find(
      (component) =>
        component.role.startsWith("transformer") &&
        (!isCount(component.block_count) || !isCount(component.block_parameter_count)),
    );
    if (missingBlocks) {
      return unknown(`Layerwise offload needs block counts for ${missingBlocks.role}.`);
    }
  }
  const unsized = components.find((component) => {
    const storage = componentPrecision(component, execution);
    const activation = componentActivationPrecision(component, execution);
    return !BYTES_PER_WEIGHT[storage] || !BYTES_PER_WEIGHT[activation];
  });
  if (unsized) return unknown(`The ${unsized.role} precision is not modeled.`);
  const dtypes = new Set(components.map((component) => componentPrecision(component, execution)));
  if (!Array.isArray(device.supported_dtypes) || !device.supported_dtypes.length) {
    const uncatalogued = [...dtypes].filter((dtype) => !["fp32", "fp16", "bf16"].includes(dtype));
    if (uncatalogued.length) {
      return unknown(`${device.name} does not catalog ${uncatalogued.join(", ")} weight support.`);
    }
  } else {
    const unsupported = [...dtypes].find((dtype) => !device.supported_dtypes.includes(dtype));
    if (unsupported) {
      return { status: "unsupported", reason: `${device.name} does not support ${unsupported} weights.` };
    }
  }
  return null;
}

const HOST_STAGING_GIB = 2;

/** Host bytes needed for CPU offload. Unified systems share one pool and are not double-counted. */
function hostRequirement(components, bytes, execution) {
  if (execution.topology.gpus > 1) return 0;
  let used = 0;
  for (const [index, component] of components.entries()) {
    const flag = offloadFlag(component.role);
    if (component.role === "transformer" || component.role === "transformer_2") {
      if (execution.offload.dit_layerwise) used += bytes[index].full;
    } else if (execution.offload[flag]) used += bytes[index].full;
  }
  return used ? used / GIB + HOST_STAGING_GIB : 0;
}

/**
 * Estimate one request's peak memory on one device from a resolved serving configuration.
 * Startup (load) and request phases are compared separately. Uncovered execution returns unknown.
 */
export function estimateMemory({
  config,
  workload,
  model,
  hardware,
  request,
  env = {},
  headroom = DEFAULT_HEADROOM,
  memoryBudgetGiB,
  allocationMemoryGb,
  partitionId,
  hostMemoryGb,
}) {
  const resolvedExecution = resolveExecution({ model, config, env });
  const sampling = request?.sampling ?? config.default_request?.sampling ?? {};
  const components = normalizeComponents(model);
  const device = normalizeHardware(hardware, { memoryBudgetGiB, allocationMemoryGb, partitionId });
  const execution = applyUnifiedMemoryPolicy(resolvedExecution, device);
  const gap = checkCoverage({
    workload: workload ?? execution.workload,
    model,
    device,
    execution,
    sampling,
    components,
  });
  if (gap) return gap;
  const bytes = components.map((component) => componentBytes(component, execution));
  const transformerGeometry = geometryFor(model, "transformer");
  const vaeGeometry = geometryFor(model, "vae");
  const working = WORKING_MEMORY[model.memory_rules].estimate({
    transformerGeometry,
    vaeGeometry,
    vaeComponent: model.components.vae,
    sampling,
    components,
    execution,
  });
  const phases = Object.fromEntries(
    PHASES.map((phase) => {
      const weights = components.reduce((sum, component, index) => {
        const resident =
          phase === "startup"
            ? startupBytes(component, execution, bytes[index])
            : residentBytes(component, phase, execution, bytes[index]);
        return sum + resident;
      }, 0);
      const workingBytes = working[phase] ?? 0;
      return [
        phase,
        {
          weightsGiB: weights / GIB,
          workingGiB: workingBytes / GIB,
          totalGiB: (weights + workingBytes) / GIB,
        },
      ];
    }),
  );
  const peak = PHASES.reduce((best, phase) => (phases[phase].totalGiB > phases[best].totalGiB ? phase : best));
  const peakGiB = phases[peak].totalGiB;
  const requiredGiB = peakGiB * (1 + headroom.fraction) + headroom.reserveGiB;
  const capacity = device.capacityGiB;
  const hostRequiredGiB = device.memoryType === "unified" ? 0 : hostRequirement(components, bytes, execution);
  const hostCapacityGiB =
    device.memoryType === "unified"
      ? null
      : Number.isFinite(hostMemoryGb)
        ? (hostMemoryGb * 1e9) / GIB
        : device.hostGiB;
  const host =
    device.memoryType === "unified"
      ? { requiredGiB: 0, capacityGiB: null, status: "shared" }
      : {
          requiredGiB: hostRequiredGiB,
          capacityGiB: hostCapacityGiB,
          status:
            hostRequiredGiB === 0
              ? "not_required"
              : hostCapacityGiB === null
                ? "unknown"
                : hostRequiredGiB <= hostCapacityGiB
                  ? "fits"
                  : "exceeds",
        };
  let status = requiredGiB <= capacity ? "fits" : "exceeds";
  if (host.status === "exceeds") status = "exceeds";
  const notes = [];
  if (device.memoryType === "unified") {
    notes.push(
      `Unified memory: ${capacity.toFixed(1)} GiB application budget, shared with the CPU and OS; not dedicated VRAM.`,
    );
    if (device.budgetAssumed) {
      notes.push(
        `Assumed ${Math.round(DEFAULT_UNIFIED_BUDGET_FRACTION * 100)}% of unified memory is available to the application; set a budget to override.`,
      );
    }
  }
  if (device.allocationGiB !== null && device.budgetGiB !== null && device.budgetGiB < device.allocationGiB) {
    notes.push(
      `The ${device.budgetGiB.toFixed(1)} GiB application budget is smaller than the ${device.allocationGiB.toFixed(1)} GiB allocation.`,
    );
  }
  if (device.partition) {
    notes.push(
      device.allocationGiB < device.nominalGiB
        ? `Using the ${device.partition.label} partition; the full ${device.nominalGiB.toFixed(1)} GiB is not available.`
        : `Using the ${device.partition.label} partition, which spans the full ${device.nominalGiB.toFixed(1)} GiB.`,
    );
  } else if (device.allocationGiB !== null) {
    notes.push(
      `Using a ${device.allocationGiB.toFixed(1)} GiB provisioned allocation; the full ${device.nominalGiB.toFixed(1)} GiB is not available.`,
    );
  }
  if (device.custom) notes.push("Custom hardware specification; not catalog-verified.");
  if (!Array.isArray(device.supported_dtypes) || !device.supported_dtypes.length) {
    notes.push("Weight-format support is not cataloged; the estimate assumes fp32, fp16, or bf16 storage.");
  }
  if (execution.vae_tiling) notes.push("Decode memory assumes untiled VAE decoding; tiling may lower it.");
  if (
    device.memoryType === "unified" &&
    ["dit", "dit_layerwise", "text_encoder", "image_encoder", "vae"].some((key) => resolvedExecution.offload[key])
  ) {
    notes.push("CPU offload is disabled on unified memory; selected components share one memory pool.");
  }
  if (host.status === "unknown") {
    notes.push(`Host memory for offload is not specified; ${hostRequiredGiB.toFixed(1)} GiB is needed.`);
  } else if (host.status === "exceeds") {
    notes.push(`Host memory for offload exceeds the ${hostCapacityGiB.toFixed(1)} GiB available.`);
  }
  notes.push("Analytical estimate only; no memory was measured.");
  return {
    status,
    basis: "analytical",
    phases,
    peakPhase: peak,
    peakGiB,
    requiredGiB,
    capacityGiB: capacity,
    headroom: { fraction: headroom.fraction, reserveGiB: headroom.reserveGiB },
    memory: {
      type: device.memoryType,
      nominalGiB: device.nominalGiB,
      budgetGiB: device.budgetGiB ?? null,
      budgetSource: device.budgetSource,
      partition: device.partition?.id ?? null,
    },
    host,
    hardware: {
      id: device.id,
      name: device.name,
      memory_type: device.memoryType,
      capacity_gib: capacity,
      custom: device.custom,
    },
    execution: {
      topology: execution.topology,
      offload: execution.offload,
      precision: execution.precision,
      attentionBackend: execution.attentionBackend,
      vae_tiling: execution.vae_tiling,
      layerwise_resident_blocks: execution.layerwise_resident_blocks,
    },
    notes,
  };
}

function memoryEnv(env = {}) {
  return Object.fromEntries(
    Object.entries(env)
      .filter(([key]) => MEMORY_ENV_KEYS.has(key) || MEMORY_ENV_PREFIXES.some((prefix) => key.startsWith(prefix)))
      .sort(([left], [right]) => left.localeCompare(right)),
  );
}

function normalizedExperimental(experimental = {}) {
  return Object.fromEntries(
    Object.entries(experimental)
      .filter(([key]) => !HARMLESS_EXPERIMENTAL.has(key))
      .sort(([left], [right]) => left.localeCompare(right)),
  );
}

function difference(path, recorded, current) {
  if (JSON.stringify(recorded) === JSON.stringify(current)) return null;
  return { path, recorded, current };
}

const EXPERIMENTAL_PRECISION_KEYS = new Set([
  "dit_precision",
  "text_encoder_precisions",
  "image_encoder_precision",
  "image_encoder_precisions",
  "vae_precision",
  "vae_decode_precision",
]);

function executionFingerprint({ model, config, env = {} }) {
  const execution = resolveExecution({ model, config, env });
  const sampling = config.default_request?.sampling ?? {};
  const components = config.generator?.pipeline?.components ?? {};
  const quantization = Object.fromEntries(
    Object.entries(execution.quantization).sort(([left], [right]) => left.localeCompare(right)),
  );
  const experimental = Object.fromEntries(
    Object.entries(normalizedExperimental(execution.experimental))
      .filter(([key]) => !EXPERIMENTAL_PRECISION_KEYS.has(key))
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  return {
    "generator.model_path": config.generator?.model_path ?? null,
    "generator.revision": config.generator?.revision ?? null,
    "generator.engine.num_gpus": execution.num_gpus,
    "generator.engine.use_fsdp_inference": execution.use_fsdp,
    "generator.engine.parallelism.tp_size": execution.topology.tpSize,
    "generator.engine.parallelism.sp_size": execution.topology.spSize,
    "generator.engine.offload.dit": execution.offload.dit,
    "generator.engine.offload.dit_layerwise": execution.offload.dit_layerwise,
    "generator.engine.offload.text_encoder": execution.offload.text_encoder,
    "generator.engine.offload.image_encoder": execution.offload.image_encoder,
    "generator.engine.offload.vae": execution.offload.vae,
    "generator.engine.offload.lazy_module_load": execution.lazy_module_load,
    "generator.engine.compile": {
      enabled: execution.compile,
      text_encoder_enabled: config.generator?.engine?.compile?.text_encoder_enabled === true,
      vae_enabled: config.generator?.engine?.compile?.vae_enabled === true,
    },
    "generator.engine.quantization": quantization,
    "generator.pipeline.workload_type": execution.workload ?? null,
    "generator.pipeline.vae_tiling": execution.vae_tiling,
    "generator.pipeline.vae_sp": execution.vae_sp,
    "generator.pipeline.precision": execution.precision,
    "generator.pipeline.experimental": experimental,
    "generator.pipeline.components.transformer_weights": components.transformer_weights ?? null,
    "generator.pipeline.components.vae_weights": components.vae_weights ?? null,
    "generator.pipeline.components.text_encoder_weights": components.text_encoder_weights ?? null,
    "default_request.sampling.height": sampling.height ?? null,
    "default_request.sampling.width": sampling.width ?? null,
    "default_request.sampling.num_frames": sampling.num_frames ?? null,
    "default_request.sampling.num_videos_per_prompt": sampling.num_videos_per_prompt ?? null,
    "default_request.sampling.batch_cfg": sampling.batch_cfg ?? false,
  };
}

function configDifferences(model, recorded = {}, current = {}) {
  const recordedFingerprint = executionFingerprint({ model, config: recorded });
  const currentFingerprint = executionFingerprint({ model, config: current });
  const differences = [];
  for (const [path, recordedValue] of Object.entries(recordedFingerprint)) {
    const item = difference(path, recordedValue, currentFingerprint[path]);
    if (item) differences.push(item);
  }
  return differences;
}

function revisionDifferences(recorded = {}, current) {
  if (!isObject(current)) return { status: "unknown", differences: [] };
  const differences = [];
  for (const key of ["fastvideo", "model"]) {
    const value = current[key];
    if (value === undefined) return { status: "unknown", differences };
    if (value !== recorded[key])
      differences.push({ path: `revisions.${key}`, recorded: recorded[key], current: value });
  }
  const recordedComponents = recorded.components ?? {};
  const currentComponents = current.components;
  if (Object.keys(recordedComponents).length && currentComponents === undefined) {
    return { status: "unknown", differences };
  }
  if (currentComponents !== undefined) {
    for (const role of new Set([...Object.keys(recordedComponents), ...Object.keys(currentComponents)])) {
      if (recordedComponents[role] !== currentComponents[role]) {
        differences.push({
          path: `revisions.components.${role}`,
          recorded: recordedComponents[role],
          current: currentComponents[role],
        });
      }
    }
  }
  return { status: differences.length ? "mismatched" : "matched", differences };
}

/**
 * Match completed-run evidence to the current deployment, configuration, request, environment and
 * hardware. Non-memory settings (server port, prompt, seed, fps) do not invalidate evidence.
 */
export function matchEvidence({ model, example, config, request, env = example?.env ?? {}, hardware, revisions } = {}) {
  const applicable = [];
  const unverified = [];
  const rejected = [];
  const currentConfig = config ?? example?.base_config;
  const currentRequest = request ?? { sampling: currentConfig?.default_request?.sampling ?? {} };
  const currentEnv = env ?? example?.env ?? {};
  for (const run of model.reference_runs ?? []) {
    const differences = [];
    if (run.hardware !== hardware?.id) {
      differences.push({ path: "hardware", recorded: run.hardware, current: hardware?.id });
    }
    const partition = hardware?.partition?.id ?? hardware?.partition_id ?? null;
    const recordedPartition = run.hardware_partition ?? null;
    if (partition !== recordedPartition) {
      differences.push({ path: "hardware_partition", recorded: recordedPartition, current: partition });
    }
    const allocationGiB = Number.isFinite(hardware?.allocationGiB)
      ? hardware.allocationGiB
      : Number.isFinite(hardware?.allocationMemoryGb)
        ? (hardware.allocationMemoryGb * 1e9) / GIB
        : null;
    const recordedAllocationGb = run.hardware_allocation_gb ?? null;
    if (partition === null && (allocationGiB !== null || recordedAllocationGb !== null)) {
      const allocationGb = allocationGiB === null ? null : (allocationGiB * GIB) / 1e9;
      if (
        allocationGb === null ||
        recordedAllocationGb === null ||
        Math.abs(allocationGb - recordedAllocationGb) > 1e-6
      ) {
        differences.push({
          path: "hardware_allocation_gb",
          recorded: recordedAllocationGb,
          current: allocationGb,
        });
      }
    } else if (partition !== null && recordedAllocationGb !== null && allocationGiB !== null) {
      const allocationGb = (allocationGiB * GIB) / 1e9;
      if (Math.abs(allocationGb - recordedAllocationGb) > 1e-6) {
        differences.push({
          path: "hardware_allocation_gb",
          recorded: recordedAllocationGb,
          current: allocationGb,
        });
      }
    }
    if (run.gpu_count !== (currentConfig?.generator?.engine?.num_gpus ?? 1)) {
      differences.push({
        path: "gpu_count",
        recorded: run.gpu_count,
        current: currentConfig?.generator?.engine?.num_gpus ?? 1,
      });
    }
    if (example && run.context.deployment !== example.deployment_id) {
      differences.push({
        path: "context.deployment",
        recorded: run.context.deployment,
        current: example.deployment_id,
      });
    }
    if (example && run.context.runtime !== example.runtime) {
      differences.push({ path: "context.runtime", recorded: run.context.runtime, current: example.runtime });
    }
    const recordedConfig = { ...run.context.config, default_request: { sampling: run.context.request.sampling } };
    const currentSamplingConfig = {
      ...currentConfig,
      default_request: { sampling: currentRequest.sampling ?? currentConfig?.default_request?.sampling },
    };
    differences.push(...configDifferences(model, recordedConfig, currentSamplingConfig));
    const envDifference = difference("context.env", memoryEnv(run.context.env), memoryEnv(currentEnv));
    if (envDifference) differences.push(envDifference);
    const revisionsMatch = revisionDifferences(run.context.revisions, revisions);
    const entry = {
      id: run.id,
      date: run.date,
      execution: run.execution,
      evidence_url: run.evidence_url,
      memory: run.memory ?? null,
      revision_status: revisionsMatch.status,
      revision_differences: revisionsMatch.differences,
      applicable: differences.length === 0 && revisionsMatch.status === "matched",
      differences,
    };
    if (entry.applicable) applicable.push(entry);
    else if (differences.length === 0) unverified.push(entry);
    else rejected.push(entry);
  }
  return { applicable, unverified, rejected };
}

function measuredDevicePeak(evidence, device) {
  const capacity = device?.capacityGiB;
  if (!(capacity > 0)) return null;
  const measurements = [];
  for (const run of evidence.applicable) {
    if (run.memory?.scope !== "device" || !Array.isArray(run.memory.per_gpu) || !run.memory.per_gpu.length) continue;
    const startupMeasured = run.memory.per_gpu.every((gpu) => Number.isFinite(gpu.startup_peak));
    const requestPeak = Math.max(...run.memory.per_gpu.map((gpu) => gpu.request_peak));
    const peak = Math.max(
      requestPeak,
      ...run.memory.per_gpu.map((gpu) => (Number.isFinite(gpu.startup_peak) ? gpu.startup_peak : 0)),
    );
    measurements.push({
      run: run.id,
      metric: run.memory.metric,
      scope: run.memory.scope,
      execution: run.execution,
      requestPeak,
      peak,
      startupMeasured,
    });
  }
  if (!measurements.length) return null;
  const highest = measurements.reduce((best, item) => (item.peak > best.peak ? item : best));
  const base = {
    basis: "measured",
    peakGiB: highest.peak,
    capacityGiB: capacity,
    run: highest.run,
    runs: measurements.map((item) => item.run),
    metric: highest.metric,
    scope: highest.scope,
    execution: highest.execution,
  };
  const unmeasured = measurements.filter((item) => !item.startupMeasured);
  if (unmeasured.length) {
    return {
      ...base,
      status: "unknown",
      reason: `Startup peak was not measured for ${unmeasured.map((item) => item.run).join(", ")}; a request-only peak cannot establish device fit.`,
    };
  }
  return {
    ...base,
    status: highest.peak <= capacity ? "fits" : "exceeds",
    spreadGiB: Math.max(...measurements.map((item) => item.peak)) - Math.min(...measurements.map((item) => item.peak)),
  };
}

/**
 * Combine the analytical estimate with applicable completed-run evidence. A successful generation
 * without a device-wide measurement establishes compatibility, not a memory minimum or fit.
 */
export function assessCapacity({
  model,
  example,
  hardware,
  config = example?.base_config,
  request,
  env = example?.env ?? {},
  revisions,
  headroom,
  memoryBudgetGiB,
  allocationMemoryGb,
  partitionId,
  hostMemoryGb,
} = {}) {
  const sampling = request?.sampling ?? config?.default_request?.sampling ?? {};
  const workload = example?.workload ?? config?.generator?.pipeline?.workload_type;
  const device = normalizeHardware(hardware, { memoryBudgetGiB, allocationMemoryGb, partitionId });
  const analytical = estimateMemory({
    config,
    workload,
    model,
    hardware: device ?? hardware,
    request,
    env,
    headroom,
    memoryBudgetGiB,
    allocationMemoryGb,
    partitionId,
    hostMemoryGb,
  });
  const evidence = matchEvidence({
    model,
    example,
    config,
    request: { sampling },
    env,
    hardware: device ?? hardware,
    revisions,
  });
  const measured = analytical.status === "unsupported" ? null : measuredDevicePeak(evidence, device);
  const compatibility = evidence.applicable.length
    ? "completed"
    : evidence.unverified.length
      ? "completed_unverified_revision"
      : "unverified";
  if (measured) {
    const hostFailure = analytical.host?.status === "exceeds";
    return {
      ...measured,
      status: hostFailure ? "exceeds" : measured.status,
      deviceStatus: measured.status,
      reason: hostFailure
        ? `Host memory for offload exceeds the ${analytical.host.capacityGiB.toFixed(1)} GiB available.`
        : measured.reason,
      analytical,
      evidence,
      compatibility,
      hardware: analytical.hardware ?? device,
      memory: analytical.memory ?? null,
      host: analytical.host ?? null,
      notes: [
        `Measured ${measured.metric} (${measured.scope}) on ${analytical.hardware?.name ?? hardware?.name ?? "the selected device"} for run ${measured.run}.`,
        hostFailure
          ? `Host memory for offload exceeds the ${analytical.host.capacityGiB.toFixed(1)} GiB available; a device measurement does not establish host fit.`
          : null,
        analytical.status === "unknown"
          ? `Analytical coverage remains unknown: ${analytical.reason}`
          : "The analytical estimate is retained for comparison.",
      ].filter(Boolean),
    };
  }
  return {
    ...analytical,
    analytical,
    evidence,
    compatibility,
    notes: [
      ...(analytical.notes ?? []),
      ...(evidence.applicable.length
        ? ["Applicable evidence records a completed generation but no device-wide memory peak."]
        : []),
      ...(evidence.unverified.length
        ? ["A completed run matches the memory-relevant settings, but its revisions are not verified."]
        : []),
      ...(evidence.rejected.length ? ["No evidence matches every memory-relevant setting."] : []),
    ],
  };
}

function lowerMemoryConfigs({ model, example, hardware, config, request, env, revisions, headroom }) {
  const alternatives = [];
  const sampling = request?.sampling ?? config.default_request?.sampling ?? {};
  const assess = (mutate, edit) => {
    const changed = structuredClone(config);
    const changedRequest = request ? structuredClone(request) : undefined;
    mutate(changed, changedRequest);
    const result = assessCapacity({
      model,
      example,
      hardware,
      config: changed,
      request: changedRequest,
      env,
      revisions,
      headroom,
    });
    alternatives.push({
      changes: [edit],
      status: result.status,
      basis: result.basis ?? null,
      required_gib: result.requiredGiB ?? null,
      reason: result.reason ?? null,
    });
  };
  const samplingChange = (key, value) =>
    assess(
      (changed, changedRequest) => {
        if (changedRequest) {
          changedRequest.sampling = { ...sampling, [key]: value };
        } else {
          changed.default_request = {
            ...(changed.default_request ?? {}),
            sampling: { ...sampling, [key]: value },
          };
        }
      },
      { path: request ? `request.sampling.${key}` : `default_request.sampling.${key}`, value },
    );
  if (isCount(sampling.num_frames) && sampling.num_frames > 1) {
    const frames = Math.max(1, Math.floor((sampling.num_frames - 1) / 2) + 1);
    if (frames < sampling.num_frames) samplingChange("num_frames", frames);
  }
  for (const key of ["height", "width"]) {
    if (isCount(sampling[key]) && sampling[key] > 16) {
      const value = Math.max(16, Math.floor(sampling[key] / 2 / 16) * 16);
      if (value < sampling[key]) samplingChange(key, value);
    }
  }
  const offload = config.generator?.engine?.offload ?? {};
  const changes = {};
  for (const key of ["dit_layerwise", "text_encoder", "vae"]) {
    if (offload[key] === false) changes[`generator.engine.offload.${key}`] = true;
  }
  if (Object.keys(changes).length) {
    assess(
      (changed) => {
        for (const [key, value] of Object.entries(changes)) {
          const parts = key.split(".");
          let cursor = changed;
          for (const part of parts.slice(0, -1)) cursor = cursor[part] ??= {};
          cursor[parts.at(-1)] = value;
        }
      },
      { path: "generator.engine.offload", value: changes },
    );
  }
  return alternatives;
}
/**
 * Rank maintained examples for one hardware/task choice. Ranking uses measured evidence first, then
 * analytical memory fit. No speed or quality ordering is inferred from memory estimates.
 */
export function recommendCapacity({ models, hardware, task, request, env, revisions, headroom, limit = 3 } = {}) {
  const candidates = [];
  for (const model of models ?? []) {
    for (const example of model.examples ?? []) {
      if (task && example.workload !== task) continue;
      const config = example.base_config;
      const assessment = assessCapacity({
        model,
        example,
        hardware,
        config,
        request,
        env: env ?? example.env,
        revisions,
        headroom,
      });
      candidates.push({ model, example, assessment });
    }
  }
  const rank = (candidate) => {
    const { status, basis } = candidate.assessment;
    if (status === "fits" && basis === "measured") return 0;
    if (status === "fits") return 1;
    if (status === "unknown") return 2;
    if (status === "exceeds") return 3;
    return 4;
  };
  candidates.sort((left, right) => {
    const delta = rank(left) - rank(right);
    if (delta) return delta;
    const leftMemory = left.assessment.requiredGiB ?? left.assessment.peakGiB ?? Infinity;
    const rightMemory = right.assessment.requiredGiB ?? right.assessment.peakGiB ?? Infinity;
    return leftMemory - rightMemory;
  });
  return candidates.slice(0, limit).map(({ model, example, assessment }) => ({
    model: { id: model.id, name: model.name, model_id: model.model_id },
    example: {
      id: example.id,
      deployment_id: example.deployment_id,
      workload: example.workload,
      runtime: example.runtime,
    },
    assessment,
    reasons: [
      assessment.status === "fits"
        ? `${assessment.basis === "measured" ? "Measured" : "Estimated"} to fit ${hardware?.name ?? "the selected device"}.`
        : assessment.status === "exceeds"
          ? `Estimated to exceed ${hardware?.name ?? "the selected device"}.`
          : `Fit is unknown: ${assessment.reason ?? "coverage is incomplete"}.`,
      assessment.compatibility === "completed"
        ? "Applicable evidence records a completed generation."
        : "No applicable completed-run evidence.",
    ],
    lower_memory:
      assessment.status === "fits"
        ? []
        : lowerMemoryConfigs({
            model,
            example,
            hardware,
            config: example.base_config,
            request,
            env: env ?? example.env,
            revisions,
            headroom,
          }),
  }));
}
