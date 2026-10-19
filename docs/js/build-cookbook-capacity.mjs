/** Validate and export inference capacity facts. No memory estimates or runtime imports. */
import { lstatSync, mkdirSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { stringify } from "yaml";
import { ROOT, buildCatalogs, readYaml, within } from "./build-cookbook-config.mjs";
import { CAPACITY_ROLES, validateCapacityMetadata } from "./cookbook-capacity-schema.mjs";

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

/** Separators are accepted only in count fields, never by partially parsing a scalar. */
function count(value, context) {
  if (typeof value === "string") {
    requireValue(/^[1-9]\d{0,2}(?:_\d{3})+$/.test(value), `${context}: invalid underscore-separated count`);
    value = Number(value.replaceAll("_", ""));
  }
  requireValue(integer(value), `${context}: expected a positive safe integer`);
  return value;
}

function hardwareRecords(document, context) {
  keys(document, ["hardware"], [], context);
  requireValue(object(document.hardware) && Object.keys(document.hardware).length > 0, `${context}: no hardware`);
  for (const [id, hardware] of Object.entries(document.hardware)) {
    const at = `${context}: hardware.${id}`;
    identifier(id, at);
    keys(
      hardware,
      ["name", "memory_type", "memory_gb", "backend", "sources"],
      ["ui_id", "supported_dtypes", "notes"],
      at,
    );
    requireValue(text(hardware.name) && text(hardware.backend), `${at}: name and backend must be text`);
    requireValue(["dedicated", "unified"].includes(hardware.memory_type), `${at}: invalid memory_type`);
    requireValue(positive(hardware.memory_gb), `${at}: memory_gb must be positive and finite`);
    if (hardware.ui_id !== undefined) identifier(hardware.ui_id, `${at}.ui_id`);
    if (hardware.notes !== undefined) requireValue(text(hardware.notes), `${at}: notes must be text`);
    if (hardware.supported_dtypes !== undefined)
      requireValue(
        Array.isArray(hardware.supported_dtypes) &&
          hardware.supported_dtypes.length > 0 &&
          hardware.supported_dtypes.every(text) &&
          new Set(hardware.supported_dtypes).size === hardware.supported_dtypes.length,
        `${at}: supported_dtypes must be a nonempty unique text list`,
      );
    sources(hardware.sources, `${at}.sources`);
  }
  return document.hardware;
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
  // Deployment validation owns identity, baseline paths, options and launch environment.
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
        config: deployment.source_config,
        runtime: deployment.runtime.id,
        env: structuredClone(deployment.env),
        base_config: structuredClone(deployment.base_config),
      })),
      reference_runs: manifest.capacity.reference_runs ?? [],
    };
    validateCapacityMetadata(manifest.capacity, { modelId: model.model_id, hardware, context: `${path}: capacity` });
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
  write(join(outputDir, "hardware.json"), hardware);
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
