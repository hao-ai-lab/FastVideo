/** Validate and export inference capacity facts. No memory estimates or runtime imports. */
import { lstatSync, mkdirSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { stringify } from "yaml";
import { ROOT, buildCatalogs, readYaml, within } from "./build-cookbook-config.mjs";
import { CAPACITY_ROLES, WEIGHT_FORMATS, validateCapacityMetadata } from "./cookbook-capacity-schema.mjs";

const ID = /^[a-z0-9]+(?:[.-][a-z0-9]+)*$/;
const GROUPS = { transformers: "transformer", encoders: "encoder", vaes: "vae" };
const GEOMETRY = {
  transformer: ["hidden_size", "ffn_size", "patch_size", "num_attention_heads", "attention_head_dim", "gated_ffn"],
  encoder: ["hidden_size", "ffn_size", "num_attention_heads", "attention_head_dim", "gated_ffn"],
  vae: ["temporal_compression", "spatial_compression", "latent_channels", "output_channels", "vae_decoder_channels"],
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

function count(value, context) {
  if (typeof value === "string") {
    requireValue(/^[1-9]\d{0,2}(?:_\d{3})+$/.test(value), `${context}: invalid underscore-separated count`);
    value = Number(value.replaceAll("_", ""));
  }
  requireValue(integer(value), `${context}: expected a positive safe integer`);
  return value;
}

const HARDWARE_VENDORS = new Set(["nvidia", "amd", "intel", "apple"]);
const HARDWARE_KINDS = new Set(["gpu", "accelerator", "soc", "system"]);
const HARDWARE_BACKENDS = new Set(["cuda", "rocm", "metal", "sycl", "unknown"]);
const FORM_FACTORS = new Set(["pcie", "sxm", "oam", "desktop", "laptop", "soc", "system", "unknown"]);
const MEMORY_TYPES = new Set(["dedicated", "unified"]);
const MEMORY_UNITS = { GB: 1, GiB: 1.073741824 };

function memoryQuantity(value, context) {
  keys(value, ["value", "unit"], [], context);
  requireValue(positive(value.value), `${context}: value must be positive and finite`);
  requireValue(Object.hasOwn(MEMORY_UNITS, value.unit), `${context}: unit must be GB or GiB`);
  return value;
}

function memorySizeGb(value, context) {
  memoryQuantity(value, context);
  return value.value * MEMORY_UNITS[value.unit];
}

function checkedSources(values, context) {
  requireValue(Array.isArray(values) && values.length > 0, `${context}: provide source URLs`);
  const seen = new Set();
  for (const source of values) {
    keys(source, ["url", "checked"], [], context);
    url(source.url, `${context}.url`);
    requireValue(!seen.has(source.url), `${context}: duplicate source ${source.url}`);
    seen.add(source.url);
    const date =
      typeof source.checked === "string" &&
      /^\d{4}-\d{2}-\d{2}$/.test(source.checked) &&
      Date.parse(`${source.checked}T00:00:00Z`);
    requireValue(
      date !== false && Number.isFinite(date) && new Date(date).toISOString().slice(0, 10) === source.checked,
      `${context}.checked: expected a YYYY-MM-DD date`,
    );
  }
  return values;
}

function hardwareRecords(document, context) {
  keys(document, ["hardware"], [], context);
  requireValue(object(document.hardware) && Object.keys(document.hardware).length > 0, `${context}: no hardware`);
  const names = new Set();
  const identities = new Map();
  const families = new Map();
  const records = {};
  for (const [id, hardware] of Object.entries(document.hardware)) {
    const at = `${context}: hardware.${id}`;
    identifier(id, at);
    keys(
      hardware,
      ["name", "vendor", "kind", "memory", "backend", "sources"],
      [
        "aliases",
        "family",
        "variant",
        "form_factor",
        "gpu_count",
        "host_memory",
        "partitionable",
        "partitions",
        "supported_dtypes",
        "notes",
        "ui_id",
      ],
      at,
    );
    requireValue(text(hardware.name), `${at}: name must be text`);
    const foldedName = hardware.name.toLocaleLowerCase();
    requireValue(!names.has(foldedName), `${at}: duplicate hardware name ${hardware.name}`);
    names.add(foldedName);
    requireValue(HARDWARE_VENDORS.has(hardware.vendor), `${at}: unknown vendor`);
    requireValue(HARDWARE_KINDS.has(hardware.kind), `${at}: unknown kind`);
    requireValue(HARDWARE_BACKENDS.has(hardware.backend), `${at}: unknown backend`);
    if (hardware.form_factor !== undefined)
      requireValue(FORM_FACTORS.has(hardware.form_factor), `${at}: unknown form_factor`);
    keys(hardware.memory, ["type", "nominal"], ["usable", "policy"], `${at}.memory`);
    requireValue(MEMORY_TYPES.has(hardware.memory.type), `${at}.memory: invalid type`);
    const nominalGb = memorySizeGb(hardware.memory.nominal, `${at}.memory.nominal`);
    if (hardware.memory.usable !== undefined) {
      const usableGb = memorySizeGb(hardware.memory.usable, `${at}.memory.usable`);
      requireValue(usableGb <= nominalGb, `${at}.memory.usable: exceeds nominal capacity`);
      requireValue(text(hardware.memory.policy), `${at}.memory.policy: explain how the usable budget is derived`);
    } else requireValue(hardware.memory.policy === undefined, `${at}.memory.policy: needs a usable budget`);
    if (hardware.host_memory !== undefined) memorySizeGb(hardware.host_memory, `${at}.host_memory`);
    if (hardware.gpu_count !== undefined) requireValue(integer(hardware.gpu_count), `${at}.gpu_count: invalid`);
    if (hardware.partitionable !== undefined)
      requireValue(typeof hardware.partitionable === "boolean", `${at}.partitionable: expected a boolean`);
    const partitions = [];
    if (hardware.partitions !== undefined) {
      requireValue(
        Array.isArray(hardware.partitions) && hardware.partitions.length > 0,
        `${at}.partitions: expected a list`,
      );
      requireValue(hardware.memory.type === "dedicated", `${at}.partitions: only dedicated devices can be partitioned`);
      requireValue(hardware.partitionable !== false, `${at}.partitions: conflicts with partitionable: false`);
      const ids = new Set();
      for (const partition of hardware.partitions) {
        keys(partition, ["id", "label", "memory"], [], `${at}.partitions`);
        identifier(partition.id, `${at}.partitions.id`);
        requireValue(!ids.has(partition.id), `${at}.partitions: duplicate partition ${partition.id}`);
        ids.add(partition.id);
        requireValue(text(partition.label), `${at}.partitions.${partition.id}.label: expected text`);
        const partitionGb = memorySizeGb(partition.memory, `${at}.partitions.${partition.id}.memory`);
        requireValue(partitionGb < nominalGb, `${at}.partitions.${partition.id}: must be smaller than the full device`);
        partitions.push({ ...structuredClone(partition), memory_gb: partitionGb });
      }
    }
    if (hardware.notes !== undefined) requireValue(text(hardware.notes), `${at}: notes must be text`);
    if (hardware.supported_dtypes !== undefined)
      requireValue(
        Array.isArray(hardware.supported_dtypes) &&
          hardware.supported_dtypes.length > 0 &&
          hardware.supported_dtypes.every((value) => WEIGHT_FORMATS.has(value)) &&
          new Set(hardware.supported_dtypes).size === hardware.supported_dtypes.length,
        `${at}: supported_dtypes must be a nonempty unique list of ${[...WEIGHT_FORMATS].join(", ")}`,
      );
    if (hardware.family !== undefined) {
      identifier(hardware.family, `${at}.family`);
      requireValue(text(hardware.variant), `${at}.variant: required with family`);
      const variants = families.get(hardware.family) ?? new Set();
      const foldedVariant = hardware.variant.toLocaleLowerCase();
      requireValue(!variants.has(foldedVariant), `${at}: duplicate variant in family ${hardware.family}`);
      variants.add(foldedVariant);
      families.set(hardware.family, variants);
    } else requireValue(hardware.variant === undefined, `${at}.variant: needs a family`);
    const aliases = [...(hardware.aliases ?? [])];
    if (hardware.ui_id !== undefined) {
      identifier(hardware.ui_id, `${at}.ui_id`);
      aliases.push(hardware.ui_id);
    }
    requireValue(
      aliases.every((alias) => text(alias)),
      `${at}.aliases: expected nonempty text`,
    );
    const record = {
      id,
      ...structuredClone(hardware),
      aliases,
      gpu_count: hardware.gpu_count ?? 1,
      memory_gb: nominalGb,
    };
    delete record.ui_id;
    if (partitions.length) record.partitions = partitions;
    if (hardware.memory.usable !== undefined)
      record.usable_memory_gb = memorySizeGb(hardware.memory.usable, `${at}.memory.usable`);
    if (hardware.host_memory !== undefined)
      record.host_memory_gb = memorySizeGb(hardware.host_memory, `${at}.host_memory`);
    checkedSources(hardware.sources, `${at}.sources`);
    records[id] = record;
  }
  for (const [id, hardware] of Object.entries(records)) {
    const local = new Set([id.toLocaleLowerCase()]);
    for (const identity of [id, ...hardware.aliases]) {
      const folded = identity.toLocaleLowerCase();
      const owner = identities.get(folded);
      requireValue(
        owner === undefined || owner === id,
        `${context}: hardware ${id} reuses identity ${identity} from ${owner}`,
      );
      requireValue(
        !(identity !== id && local.has(folded) && folded !== id),
        `${context}: hardware ${id} repeats its own identity ${identity}`,
      );
      identities.set(folded, id);
    }
  }
  return records;
}

function componentRecords(root, catalogDir) {
  const components = new Map();
  for (const [group, kind] of Object.entries(GROUPS)) {
    const path = within(root, relative(root, join(catalogDir, "components", `${group}.yaml`)), "docs", [".yaml"]);
    const document = readYaml(path);
    keys(document, [group], [], path);
    requireValue(object(document[group]), `${path}: ${group} must be a mapping`);
    for (const [id, record] of Object.entries(document[group])) {
      const at = `${path}: ${id}`;
      identifier(id, at);
      requireValue(!components.has(id), `${at}: duplicate component ID`);
      keys(
        record,
        ["name", "parameter_count", "sources"],
        ["block_count", "block_parameter_count", "loaded_subset", "geometry", "decode_feature_multiplier"],
        at,
      );
      requireValue(text(record.name), `${at}: name must be text`);
      record.parameter_count = count(record.parameter_count, `${at}.parameter_count`);
      const blocks = Object.hasOwn(record, "block_count"),
        blockParameters = Object.hasOwn(record, "block_parameter_count");
      requireValue(
        blocks === blockParameters,
        `${at}: block_count and block_parameter_count must be supplied together`,
      );
      if (blocks) {
        record.block_count = count(record.block_count, `${at}.block_count`);
        record.block_parameter_count = count(record.block_parameter_count, `${at}.block_parameter_count`);
        requireValue(
          record.block_parameter_count <= record.parameter_count,
          `${at}: block parameters exceed total parameters`,
        );
        requireValue(record.block_count <= record.block_parameter_count, `${at}: more blocks than block parameters`);
      }
      if (record.loaded_subset !== undefined)
        requireValue(text(record.loaded_subset), `${at}: loaded_subset must be text`);
      if (record.decode_feature_multiplier !== undefined) {
        requireValue(kind === "vae", `${at}: decode_feature_multiplier is only supported for VAE components`);
        requireValue(
          positive(record.decode_feature_multiplier),
          `${at}.decode_feature_multiplier: expected a positive finite number`,
        );
      }
      if (record.geometry !== undefined) {
        keys(record.geometry, [], GEOMETRY[kind], `${at}.geometry`);
        for (const [key, value] of Object.entries(record.geometry)) {
          if (key === "patch_size")
            requireValue(
              Array.isArray(value) && value.length === 3 && value.every(integer),
              `${at}: patch_size needs three positive integers`,
            );
          else if (key === "gated_ffn")
            requireValue(typeof value === "boolean", `${at}.geometry.${key}: expected a boolean`);
          else requireValue(integer(value), `${at}.geometry.${key}: expected a positive safe integer`);
        }
      }
      sources(record.sources, `${at}.sources`);
      components.set(id, { ref: id, kind, ...record });
    }
  }
  return components;
}

/** Resolve references once at build time. The browser receives self-contained model facts. */
export function buildCapacityCatalog({
  root = ROOT,
  catalogDir = join(root, "docs/cookbook/capacity"),
  recipesDir = join(root, "docs/cookbook/recipes"),
  catalogPath = join(root, "docs/cookbook/options.yaml"),
} = {}) {
  const hardwarePath = within(root, relative(root, join(catalogDir, "hardware.yaml")), "docs", [".yaml"]);
  const hardware = hardwareRecords(readYaml(hardwarePath), hardwarePath);
  const components = componentRecords(root, catalogDir),
    used = new Set();
  const deployments = buildCatalogs({ root, recipesDir, catalogPath });
  const models = [];
  for (const name of readdirSync(recipesDir)
    .filter((name) => /\.ya?ml$/.test(name))
    .sort()) {
    const path = within(root, relative(root, join(recipesDir, name)), "docs", [".yaml", ".yml"]);
    const manifest = readYaml(path);
    if (manifest.capacity === undefined) continue;
    const id = name.replace(/\.ya?ml$/, "");
    const examples = deployments.filter((deployment) => deployment.model.key === id);
    const model = {
      id,
      name: manifest.title,
      model_id: examples[0].model.id,
      ...structuredClone(manifest.capacity),
      examples: examples.map((deployment) => ({
        id: deployment.deployment.id,
        deployment_id: deployment.id,
        workload: deployment.workload,
        config: deployment.source_config,
        runtime: deployment.runtime.id,
        env: structuredClone(deployment.env),
        base_config: structuredClone(deployment.base_config),
      })),
      reference_runs: manifest.capacity.reference_runs ?? [],
    };
    validateCapacityMetadata(manifest.capacity, {
      modelId: model.model_id,
      hardware,
      examples: model.examples,
      context: `${path}: capacity`,
    });
    for (const [role, selection] of Object.entries(model.components)) {
      const refs = [selection.ref, ...(selection.alternatives ?? [])];
      for (const ref of refs) {
        requireValue(components.has(ref), `${path}: unknown component ${ref}`);
        requireValue(
          components.get(ref).kind === CAPACITY_ROLES[role],
          `${path}: wrong component type for ${role}: ${ref}`,
        );
        used.add(ref);
      }
      model.components[role] = structuredClone(components.get(selection.ref));
      if (selection.alternatives !== undefined)
        model.components[role].alternatives = selection.alternatives.map((ref) => structuredClone(components.get(ref)));
    }
    if (model.memory_rules === "wan") {
      const required = {
        transformer: ["hidden_size", "ffn_size", "patch_size"],
        vae: ["temporal_compression", "spatial_compression", "vae_decoder_channels"],
      };
      for (const [role, fields] of Object.entries(required))
        for (const component of [model.components[role], ...(model.components[role]?.alternatives ?? [])])
          requireValue(
            fields.every((field) => component?.geometry?.[field] !== undefined),
            `${path}: wan ${role} geometry requires ${fields.join(", ")}`,
          );
    }
    models.push(model);
  }
  const unused = [...components.keys()].filter((id) => !used.has(id));
  requireValue(unused.length === 0, `Unused capacity components: ${unused.join(", ")}`);
  return { hardware, models };
}

/** Generated assets are ignored by Git and copied with the existing cookbook assets. */
export function exportCapacityCatalog({
  root = ROOT,
  outputDir = join(root, "docs/assets/cookbook-config/capacity"),
  ...options
} = {}) {
  const { hardware, models } = buildCapacityCatalog({ root, ...options });
  function checkOutput(path) {
    let current = resolve(path);
    while (true) {
      requireValue(
        !lstatSync(current, { throwIfNoEntry: false })?.isSymbolicLink(),
        `Generated capacity paths must not contain symlinks: ${current}`,
      );
      if (current === resolve(root) || current === dirname(current)) break;
      current = dirname(current);
    }
  }
  for (const path of [outputDir, join(outputDir, "models")]) {
    checkOutput(path);
    mkdirSync(path, { recursive: true });
  }
  const write = (path, value) => {
    checkOutput(path);
    writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
  };
  const index = { hardware_url: "hardware.json", models: [] },
    expected = new Set();
  for (const model of models) {
    const filename = `${model.id}.json`;
    expected.add(filename);
    write(join(outputDir, "models", filename), model);
    index.models.push({ id: model.id, name: model.name, model_id: model.model_id, catalog_url: `models/${filename}` });
  }
  write(join(outputDir, "hardware.json"), {
    version: 1,
    devices: Object.values(hardware).sort((left, right) => left.id.localeCompare(right.id)),
  });
  write(join(outputDir, "index.json"), index);
  for (const name of readdirSync(join(outputDir, "models"))) {
    if (name.endsWith(".json") && ID.test(name.slice(0, -5)) && !expected.has(name))
      unlinkSync(join(outputDir, "models", name));
  }
  return index;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = {};
    let check = false,
      preview;
    const flags = {
      "--root": "root",
      "--catalog-dir": "catalogDir",
      "--recipes-dir": "recipesDir",
      "--catalog": "catalogPath",
      "--output-dir": "outputDir",
    };
    for (let offset = 2; offset < process.argv.length; offset++) {
      const flag = process.argv[offset];
      if (flag === "--check") check = true;
      else {
        const value = process.argv[++offset];
        requireValue(value && !value.startsWith("--"), `Missing value for ${flag}`);
        if (flag === "--preview") preview = value;
        else {
          requireValue(Object.hasOwn(flags, flag), `Unknown argument ${flag}`);
          options[flags[flag]] = resolve(value);
        }
      }
    }
    requireValue(
      !(check && preview) && !((check || preview) && options.outputDir),
      "Use --check, --preview MODEL, or export --output-dir separately",
    );
    if (check || preview) {
      const catalog = buildCapacityCatalog(options);
      if (preview) {
        const model = catalog.models.find((item) => item.id === preview);
        requireValue(model, `Unknown capacity model ${preview}`);
        process.stdout.write(stringify(model, { aliasDuplicateObjects: false }));
      } else
        console.log(
          `Validated ${catalog.models.length} capacity model(s) and ${Object.keys(catalog.hardware).length} hardware variant(s).`,
        );
    } else {
      exportCapacityCatalog(options);
      console.log(
        join(options.outputDir ?? join(options.root ?? ROOT, "docs/assets/cookbook-config/capacity"), "index.json"),
      );
    }
  } catch (failure) {
    console.error(failure.message);
    process.exitCode = 1;
  }
}
