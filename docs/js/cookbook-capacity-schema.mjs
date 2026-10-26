/** Shared validation for optional capacity fields in existing cookbook model manifests. */
const ID = /^[a-z0-9]+(?:[.-][a-z0-9]+)*$/;
export const WEIGHT_FORMATS = new Set(["fp32", "fp16", "bf16", "fp8", "nvfp4"]);
export const MEASUREMENT_SCOPES = {
  torch_peak_allocated: "allocator",
  torch_peak_reserved: "allocator",
  device_used: "device",
};
export const CAPACITY_ROLES = {
  transformer: "transformer",
  transformer_2: "transformer",
  text_encoder: "encoder",
  text_encoder_2: "encoder",
  image_encoder: "encoder",
  vae: "vae",
  audio_vae: "vae",
};
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value) => typeof value === "string" && value.trim().length > 0;
const positive = (value) => typeof value === "number" && Number.isFinite(value) && value > 0;
const integer = (value) => Number.isSafeInteger(value) && value > 0;
const requireValue = (condition, message) => {
  if (!condition) throw new Error(message);
};

function keys(value, required, optional, context) {
  requireValue(object(value), `${context}: expected a mapping`);
  for (const key of required) requireValue(Object.hasOwn(value, key), `${context}: missing ${key}`);
  for (const key of Object.keys(value))
    requireValue([...required, ...optional].includes(key), `${context}: unknown key ${key}`);
}

function identifier(value, context) {
  requireValue(text(value) && ID.test(value), `${context}: expected a lowercase ID with dots or hyphens`);
}

function url(value, context) {
  requireValue(text(value), `${context}: expected an HTTP(S) source URL`);
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${context}: expected an HTTP(S) source URL`);
  }
  requireValue(["https:", "http:"].includes(parsed.protocol), `${context}: expected an HTTP(S) source URL`);
}

function sources(values, context) {
  requireValue(Array.isArray(values) && values.length > 0, `${context}: provide source URLs`);
  values.forEach((value) => url(value, context));
}

function environment(value, context) {
  requireValue(
    object(value) &&
      Object.entries(value).every(
        ([key, item]) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && typeof item === "string" && !item.includes("\0"),
      ),
    `${context}: expected environment names mapped to strings`,
  );
}

function measurements(memory, gpuCount, context) {
  keys(memory, ["metric", "unit", "scope", "per_gpu"], [], context);
  requireValue(Object.hasOwn(MEASUREMENT_SCOPES, memory.metric), `${context}: unknown memory metric`);
  requireValue(
    MEASUREMENT_SCOPES[memory.metric] === memory.scope,
    `${context}: scope must be ${MEASUREMENT_SCOPES[memory.metric]} for ${memory.metric}`,
  );
  requireValue(memory.unit === "GiB", `${context}: memory unit must be GiB`);
  requireValue(
    Array.isArray(memory.per_gpu) && memory.per_gpu.length === gpuCount,
    `${context}: record every GPU rank`,
  );
  const ranks = new Set();
  for (const gpu of memory.per_gpu) {
    keys(gpu, ["rank", "request_peak"], ["startup_peak", "stages"], context);
    requireValue(
      Number.isSafeInteger(gpu.rank) && gpu.rank >= 0 && gpu.rank < gpuCount && !ranks.has(gpu.rank),
      `${context}: invalid or duplicate GPU rank`,
    );
    ranks.add(gpu.rank);
    requireValue(positive(gpu.request_peak), `${context}: request_peak must be positive and finite`);
    if (gpu.startup_peak !== undefined) requireValue(positive(gpu.startup_peak), `${context}: invalid startup_peak`);
    if (gpu.stages !== undefined) {
      requireValue(object(gpu.stages), `${context}: stages must be a mapping`);
      for (const [phase, peak] of Object.entries(gpu.stages)) {
        requireValue(/^[a-z][a-z0-9_-]*$/.test(phase), `${context}: invalid stage name ${phase}`);
        requireValue(
          positive(peak) && peak <= gpu.request_peak,
          `${context}: stage peak must be positive and no greater than request_peak`,
        );
      }
    }
  }
}

function referenceRuns(model, hardware, examples) {
  requireValue(Array.isArray(model.reference_runs), `${model.id}: reference_runs must be a list (empty is allowed)`);
  const ids = new Set();
  for (const run of model.reference_runs) {
    const at = `${model.id}: reference_runs.${run?.id}`;
    keys(
      run,
      ["id", "date", "hardware", "gpu_count", "outcome", "execution", "evidence_url", "context"],
      ["memory", "hardware_partition", "hardware_allocation_gb"],
      at,
    );
    identifier(run.id, at);
    requireValue(!ids.has(run.id), `${at}: duplicate run ID`);
    ids.add(run.id);
    const date =
      typeof run.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(run.date) && Date.parse(`${run.date}T00:00:00Z`);
    requireValue(
      date !== false && Number.isFinite(date) && new Date(date).toISOString().slice(0, 10) === run.date,
      `${at}: date must be YYYY-MM-DD`,
    );
    identifier(run.hardware, `${at}.hardware`);
    requireValue(
      hardware === undefined || Object.hasOwn(hardware, run.hardware),
      `${at}: unknown hardware ${run.hardware}`,
    );
    if (run.hardware_partition !== undefined) identifier(run.hardware_partition, `${at}.hardware_partition`);
    if (run.hardware_allocation_gb !== undefined)
      requireValue(positive(run.hardware_allocation_gb), `${at}.hardware_allocation_gb must be positive`);
    requireValue(integer(run.gpu_count), `${at}: gpu_count must be a positive safe integer`);
    requireValue(run.outcome === "completed", `${at}: outcome must describe a completed generation`);
    requireValue(["first_request", "warm_request"].includes(run.execution), `${at}: unknown execution condition`);
    url(run.evidence_url, `${at}.evidence_url`);
    keys(run.context, ["runtime", "deployment", "revisions", "config", "request", "env"], [], `${at}.context`);
    requireValue(text(run.context.runtime), `${at}: record the serving runtime`);
    requireValue(text(run.context.deployment), `${at}: record the cookbook deployment`);
    if (examples !== undefined) {
      const example = examples.find((item) => item.deployment_id === run.context.deployment);
      requireValue(example, `${at}: unknown deployment ${run.context.deployment}`);
      requireValue(
        example.runtime === run.context.runtime,
        `${at}: runtime ${run.context.runtime} does not match deployment ${example.deployment_id}`,
      );
    }
    environment(run.context.env, `${at}.env`);
    keys(run.context.revisions, ["fastvideo", "model"], ["components"], `${at}.revisions`);
    requireValue(
      text(run.context.revisions.fastvideo) && text(run.context.revisions.model),
      `${at}: pin code and model revisions`,
    );
    if (run.context.revisions.components !== undefined) {
      requireValue(
        object(run.context.revisions.components) && Object.values(run.context.revisions.components).every(text),
        `${at}: component revisions must map roles to revisions`,
      );
      for (const role of Object.keys(run.context.revisions.components))
        requireValue(Object.hasOwn(model.components, role), `${at}: unknown component role ${role}`);
    }
    requireValue(
      object(run.context.config) && run.context.config.generator?.model_path === model.model_id,
      `${at}: recorded config model_path must match model_id`,
    );
    requireValue(
      run.context.config.generator.engine?.num_gpus === run.gpu_count,
      `${at}: recorded num_gpus must match gpu_count`,
    );
    requireValue(
      object(run.context.request) && object(run.context.request.sampling),
      `${at}: record the actual request with sampling`,
    );
    for (const key of ["height", "width", "num_frames"])
      requireValue(
        integer(run.context.request.sampling[key]),
        `${at}: request sampling.${key} must be a positive integer`,
      );
    if (run.memory !== undefined) measurements(run.memory, run.gpu_count, `${at}.memory`);
  }
}

/** Validate authored structure without resolving shared hardware/component catalogs. */
export function validateCapacityMetadata(capacity, { modelId, hardware, examples, context = "capacity" } = {}) {
  keys(capacity, ["memory_rules", "components"], ["sources", "reference_runs"], context);
  identifier(capacity.memory_rules, `${context}.memory_rules`);
  requireValue(
    object(capacity.components) && Object.keys(capacity.components).length > 0,
    `${context}: components must be nonempty`,
  );
  for (const [role, selection] of Object.entries(capacity.components)) {
    requireValue(Object.hasOwn(CAPACITY_ROLES, role), `${context}: unknown component role ${role}`);
    keys(selection, ["ref"], ["alternatives"], `${context}: ${role}`);
    requireValue(
      selection.alternatives === undefined || Array.isArray(selection.alternatives),
      `${context}: alternatives must be a list`,
    );
    const refs = [selection.ref, ...(selection.alternatives ?? [])];
    refs.forEach((ref) => identifier(ref, `${context}: ${role}.ref`));
    requireValue(new Set(refs).size === refs.length, `${context}: duplicate component selection`);
  }
  if (capacity.sources !== undefined) sources(capacity.sources, `${context}.sources`);
  referenceRuns(
    {
      ...capacity,
      id: context,
      model_id: modelId,
      reference_runs: Object.hasOwn(capacity, "reference_runs") ? capacity.reference_runs : [],
    },
    hardware,
    examples,
  );
}
