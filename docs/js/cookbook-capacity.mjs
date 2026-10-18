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

/** Hardware specifications are separate from model-specific successful run evidence. */
export async function loadCapacityHardware(index, options = {}) {
  validateIndex(index);
  requireValue(text(index.url), "Capacity index requires its source URL");
  const { data } = await fetchJson(new URL(index.hardware_url, index.url).href, options);
  requireValue(object(data) && Object.keys(data).length > 0, "Invalid capacity hardware catalog");
  for (const [id, hardware] of Object.entries(data))
    requireValue(
      object(hardware) &&
        text(hardware.name) &&
        ["dedicated", "unified"].includes(hardware.memory_type) &&
        Number.isFinite(hardware.memory_gb) &&
        hardware.memory_gb > 0,
      `Invalid capacity hardware ${id}`,
    );
  return data;
}
