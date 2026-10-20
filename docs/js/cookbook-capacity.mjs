/** Static capacity metadata for future inference calculators. No estimates, DOM or runtime imports. */
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value) => typeof value === "string" && value.trim().length > 0;
function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function validateIndex(index) {
  requireValue(object(index) && text(index.hardware_url) && Array.isArray(index.models), "Invalid capacity index");
  const ids = new Set();
  for (const model of index.models) {
    requireValue(
      object(model) && [model.id, model.name, model.model_id, model.catalog_url].every(text),
      "Invalid capacity model index entry",
    );
    requireValue(!ids.has(model.id), `Duplicate capacity model ${model.id}`);
    ids.add(model.id);
  }
}

async function fetchJson(url, { fetcher = (...args) => globalThis.fetch(...args), signal } = {}) {
  const response = await fetcher(url, { signal });
  signal?.throwIfAborted();
  requireValue(response.ok, `Could not load capacity metadata: HTTP ${response.status}`);
  const data = await response.json();
  signal?.throwIfAborted();
  return { data, url: response.url || new URL(url, globalThis.location?.href).href };
}

/** Start with /assets/cookbook-config/capacity/index.json on the current documentation site. */
export async function loadCapacityIndex(url, options = {}) {
  const result = await fetchJson(url, options);
  validateIndex(result.data);
  return { ...result.data, url: result.url };
}

/** Load one self-contained model. Component references and authored baselines are already resolved. */
export async function loadCapacityModel(index, id, options = {}) {
  validateIndex(index);
  requireValue(text(index.url), "Capacity index requires its source URL");
  const entry = index.models.find((model) => model.id === id);
  requireValue(entry, `Unknown capacity model ${id}`);
  const { data } = await fetchJson(new URL(entry.catalog_url, index.url).href, options);
  requireValue(
    object(data) &&
      data.id === id &&
      data.model_id === entry.model_id &&
      object(data.components) &&
      Array.isArray(data.examples) &&
      Array.isArray(data.reference_runs),
    `Invalid capacity model ${id}`,
  );
  return data;
}

function validateHardwareDevice(hardware) {
  requireValue(
    object(hardware) &&
      text(hardware.id) &&
      text(hardware.name) &&
      object(hardware.memory) &&
      ["dedicated", "unified"].includes(hardware.memory.type) &&
      object(hardware.memory.nominal) &&
      Number.isFinite(hardware.memory.nominal.value) &&
      ["GB", "GiB"].includes(hardware.memory.nominal.unit) &&
      Number.isFinite(hardware.memory_gb) &&
      hardware.memory_gb > 0 &&
      Number.isSafeInteger(hardware.gpu_count) &&
      hardware.gpu_count > 0 &&
      ["cuda", "rocm", "metal", "sycl", "unknown"].includes(hardware.backend) &&
      Array.isArray(hardware.aliases) &&
      hardware.aliases.every(text),
    `Invalid capacity hardware ${hardware?.id ?? "(missing id)"}`,
  );
  if (hardware.usable_memory_gb !== undefined)
    requireValue(
      Number.isFinite(hardware.usable_memory_gb) &&
        hardware.usable_memory_gb > 0 &&
        hardware.usable_memory_gb <= hardware.memory_gb,
      `Invalid usable memory for ${hardware.id}`,
    );
  if (hardware.host_memory_gb !== undefined)
    requireValue(
      Number.isFinite(hardware.host_memory_gb) && hardware.host_memory_gb > 0,
      `Invalid host memory for ${hardware.id}`,
    );
  if (hardware.partitions !== undefined) {
    requireValue(
      Array.isArray(hardware.partitions) &&
        hardware.partitions.length > 0 &&
        hardware.partitions.every(
          (partition) =>
            object(partition) &&
            text(partition.id) &&
            text(partition.label) &&
            Number.isFinite(partition.memory_gb) &&
            partition.memory_gb > 0 &&
            partition.memory_gb < hardware.memory_gb,
        ),
      `Invalid partitions for ${hardware.id}`,
    );
  }
}

/** Hardware specifications are separate from model-specific successful run evidence. */
export async function loadCapacityHardware(index, options = {}) {
  validateIndex(index);
  requireValue(text(index.url), "Capacity index requires its source URL");
  const { data, url } = await fetchJson(new URL(index.hardware_url, index.url).href, options);
  requireValue(object(data) && data.version === 1 && Array.isArray(data.devices), "Invalid capacity hardware catalog");
  const ids = new Set();
  for (const hardware of data.devices) {
    validateHardwareDevice(hardware);
    requireValue(!ids.has(hardware.id), `Duplicate capacity hardware ${hardware.id}`);
    ids.add(hardware.id);
  }
  requireValue(data.devices.length > 0, "Invalid capacity hardware catalog");
  return { version: data.version, devices: data.devices, url };
}
